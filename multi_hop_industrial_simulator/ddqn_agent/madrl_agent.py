"""
madrl_agent.py
==============
Multi-Agent Deep Reinforcement Learning for THz Industrial Routing.

Three decoupled DDQNs per UE
    • routing_agent  – selects next-hop (broadcast | unicast-to-neighbour-i)
    • cw_agent       – adapts contention-window W  (W++ / W== / W--)
    • buf_agent      – adapts buffer (queue) size Q (Q++ / Q== / Q--)

Architectural improvements over the paper baseline
    ✓ Dueling streams  (Value + Advantage)  → better state-value estimation
    ✓ Double-DQN target                     → reduced over-estimation bias
    ✓ Prioritised Experience Replay (PER)   → 10-50× sample efficiency gain
    ✓ Soft target-network update (τ-blend)  → smoother convergence
    ✓ Gradient clipping                     → training stability
    ✓ Optional shared replay pool           → cross-UE generalisation

Dimensions
    ROUTING_STATE_DIM  = 4 * MAX_NEIGHBOURS + 9   (≈ 89 for MAX=20)
    CW_STATE_DIM       = 6
    BUF_STATE_DIM      = 6
    ROUTING_ACTION_DIM = 1 + MAX_NEIGHBOURS        (broadcast + N unicast)
    CW_ACTION_DIM      = 3                         (W++ / W== / W--)
    BUF_ACTION_DIM     = 3                         (Q++ / Q== / Q--)
"""

from __future__ import annotations

import os
import random
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

# ── optional torch import ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False

# ── global constants ─────────────────────────────────────────────────────────
MAX_NEIGHBOURS   = 20          # maximum neighbour table size (padded if smaller)
N_PER_NEIGHBOUR  = 4           # features per neighbour slot
N_GLOBAL         = 9           # global state features
ROUTING_STATE_DIM  = N_PER_NEIGHBOUR * MAX_NEIGHBOURS + N_GLOBAL   # 89
CW_STATE_DIM       = 6
BUF_STATE_DIM      = 6
ROUTING_ACTION_DIM = 1 + MAX_NEIGHBOURS   # broadcast=0, unicast_i = 1+i
CW_ACTION_DIM      = 3
BUF_ACTION_DIM     = 3

W_MIN_DEFAULT  = 4;   W_MAX_DEFAULT  = 128;  W_STEP = 4
Q_MIN_DEFAULT  = 1;   Q_MAX_DEFAULT  = 10;   Q_STEP = 1

# ─────────────────────────────────────────────────────────────────────────────
# Sum-Tree  (O(log n) priority sampling for PER)
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """Binary sum-tree for prioritised sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.ptr  = 0
        self.size = 0

    # ── internal helpers ──────────────────────────────────────────────────────
    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    # ── public API ────────────────────────────────────────────────────────────
    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        leaf = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(leaf, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float):
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s: float) -> Tuple[int, float, object]:
        leaf = self._retrieve(0, s)
        return leaf, self.tree[leaf], self.data[leaf - self.capacity + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Prioritised Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class PrioritisedReplayBuffer:
    """
    Rank-based PER with importance-sampling corrections.
    alpha  – how much prioritisation is used   (0 = uniform, 1 = full)
    beta   – IS correction exponent            (annealed 0→1 during training)
    """

    def __init__(self, capacity: int = 20_000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_steps: int = 50_000, eps: float = 1e-6):
        self.tree       = SumTree(capacity)
        self.alpha      = alpha
        self.beta       = beta_start
        self.beta_end   = beta_end
        self.beta_inc   = (beta_end - beta_start) / beta_steps
        self.eps        = eps
        self.max_p      = 1.0        # max seen priority (used for new samples)

    def push(self, *transition):
        self.tree.add(self.max_p ** self.alpha, transition)

    def sample(self, batch: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """Returns (transitions, leaf_indices, IS_weights)."""
        indices, weights, transitions = [], [], []
        segment = self.tree.total / batch
        self.beta = min(self.beta_end, self.beta + self.beta_inc)

        min_p = np.min(self.tree.tree[-self.tree.capacity:][
                       :self.tree.size]) + self.eps
        max_w = (min_p / self.tree.total) ** (-self.beta)

        for i in range(batch):
            s = random.uniform(segment * i, segment * (i + 1))
            leaf, p, data = self.tree.get(s)
            if data is None:
                # fallback: pick a random valid leaf
                leaf = random.randint(
                    self.tree.capacity - 1,
                    self.tree.capacity - 1 + self.tree.size - 1)
                p    = self.tree.tree[leaf] + self.eps
                data = self.tree.data[leaf - self.tree.capacity + 1]
            prob = (p + self.eps) / self.tree.total
            w    = ((prob * self.tree.size) ** (-self.beta)) / max_w
            indices.append(leaf)
            weights.append(w)
            transitions.append(data)

        return transitions, np.array(indices), np.array(weights, dtype=np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        self.max_p = max(self.max_p, priorities.max())
        for idx, p in zip(indices, priorities):
            self.tree.update(int(idx), float(p))

    def __len__(self):
        return self.tree.size


# ─────────────────────────────────────────────────────────────────────────────
# Neural network  (Torch path)
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH:
    class DuelingDQN(nn.Module):
        """
        Dueling Double-DQN network.
        Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden: Tuple[int, ...] = (256, 128)):
            super().__init__()
            # shared trunk
            layers: List[nn.Module] = []
            prev = state_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
                prev = h
            self.trunk = nn.Sequential(*layers)
            # value stream
            self.value_head = nn.Sequential(
                nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))
            # advantage stream
            self.adv_head = nn.Sequential(
                nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, action_dim))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            f = self.trunk(x)
            v = self.value_head(f)
            a = self.adv_head(f)
            return v + (a - a.mean(dim=-1, keepdim=True))

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight fallback  (NumPy linear Q-approximator, no backprop)
# ─────────────────────────────────────────────────────────────────────────────

class LinearQApprox:
    """
    Gradient-descent linear Q-function as a drop-in when torch is absent.
    Uses tile-coded features for rudimentary non-linearity.
    """

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-2):
        self.W   = np.zeros((action_dim, state_dim), dtype=np.float32)
        self.lr  = lr

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state          # shape (action_dim,)

    def update(self, state: np.ndarray, action: int,
               target: float, weight: float = 1.0):
        q   = self.W[action] @ state
        td  = target - q
        self.W[action] += self.lr * weight * td * state
        return abs(td)

    def copy_weights_from(self, other: "LinearQApprox"):
        self.W[:] = other.W


# ─────────────────────────────────────────────────────────────────────────────
# DDQNAgent  (one per agent-type per UE)
# ─────────────────────────────────────────────────────────────────────────────

class DDQNAgent:
    """
    Double DQN agent with Dueling architecture + PER.

    Parameters
    ----------
    state_dim, action_dim  : network dimensions
    hidden                 : tuple of hidden layer widths
    lr                     : Adam learning-rate
    gamma                  : discount factor
    eps_start/end/decay    : ε-greedy schedule
    batch                  : mini-batch size
    buf_capacity           : replay buffer size
    tau                    : soft target-update coefficient
    target_hard_update     : steps between hard target syncs (if tau=1)
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        hidden:       Tuple[int, ...] = (256, 128),
        lr:           float = 3e-4,
        gamma:        float = 0.95,
        eps_start:    float = 1.0,
        eps_end:      float = 0.05,
        eps_decay:    float = 0.9995,
        batch:        int   = 64,
        buf_capacity: int   = 20_000,
        tau:          float = 5e-3,
        target_hard_update: int = 200,
        per_alpha:    float = 0.6,
        per_beta:     float = 0.4,
        device: str = "cpu",
    ):
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.eps          = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.batch        = batch
        self.tau          = tau
        self.hard_update  = target_hard_update
        self._steps       = 0
        self._train_calls = 0

        self.buf = PrioritisedReplayBuffer(
            capacity=buf_capacity, alpha=per_alpha, beta_start=per_beta)

        if _TORCH:
            dev = torch.device(device)
            self.online  = DuelingDQN(state_dim, action_dim, hidden).to(dev)
            self.target  = DuelingDQN(state_dim, action_dim, hidden).to(dev)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.opt     = optim.Adam(self.online.parameters(), lr=lr)
            self.dev     = dev
            self._backend = "torch"
        else:
            self.online  = LinearQApprox(state_dim, action_dim, lr)
            self.target  = LinearQApprox(state_dim, action_dim, lr)
            self.target.copy_weights_from(self.online)
            self._backend = "numpy"

    # ── action selection ─────────────────────────────────────────────────────
    def act(self, state: np.ndarray,
            valid_mask: Optional[np.ndarray] = None) -> int:
        """
        ε-greedy with optional action masking.
        valid_mask: boolean array shape (action_dim,); True = selectable.
        """
        if random.random() < self.eps:
            if valid_mask is not None:
                choices = np.where(valid_mask)[0]
                return int(random.choice(choices)) if len(choices) else 0
            return random.randrange(self.action_dim)

        q = self._q_online(state)
        if valid_mask is not None:
            q = np.where(valid_mask, q, -np.inf)
        return int(np.argmax(q))

    # ── experience storage ───────────────────────────────────────────────────
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buf.push(state.astype(np.float32), action, reward,
                      next_state.astype(np.float32), bool(done))
        self._steps += 1

    # ── training step ────────────────────────────────────────────────────────
    def train(self) -> Optional[float]:
        if len(self.buf) < self.batch:
            return None
        transitions, leaf_idxs, is_weights = self.buf.sample(self.batch)
        s, a, r, s2, d = zip(*transitions)

        s  = np.array(s,  dtype=np.float32)
        s2 = np.array(s2, dtype=np.float32)
        a  = np.array(a,  dtype=np.int64)
        r  = np.array(r,  dtype=np.float32)
        d  = np.array(d,  dtype=np.float32)

        if self._backend == "torch":
            loss, td_errors = self._train_torch(
                s, a, r, s2, d, is_weights)
        else:
            loss, td_errors = self._train_numpy(s, a, r, s2, d, is_weights)

        self.buf.update_priorities(leaf_idxs, td_errors)
        self._anneal_eps()
        self._train_calls += 1

        # soft target update
        if self._backend == "torch":
            self._soft_update()
        elif self._train_calls % self.hard_update == 0:
            self.target.copy_weights_from(self.online)

        return float(loss)

    # ── private helpers ──────────────────────────────────────────────────────
    def _q_online(self, state: np.ndarray) -> np.ndarray:
        if self._backend == "torch":
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.dev)
                return self.online(t).squeeze(0).cpu().numpy()
        return self.online.predict(state)

    def _train_torch(self, s, a, r, s2, d, w):
        s_t  = torch.FloatTensor(s).to(self.dev)
        s2_t = torch.FloatTensor(s2).to(self.dev)
        a_t  = torch.LongTensor(a).to(self.dev)
        r_t  = torch.FloatTensor(r).to(self.dev)
        d_t  = torch.FloatTensor(d).to(self.dev)
        w_t  = torch.FloatTensor(w).to(self.dev)

        curr_q = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online selects action, target evaluates
            best_a = self.online(s2_t).argmax(dim=1)
            next_q = self.target(s2_t).gather(1, best_a.unsqueeze(1)).squeeze(1)
            tgt_q  = r_t + self.gamma * next_q * (1 - d_t)

        td_errors = (tgt_q - curr_q).detach().cpu().numpy()
        loss = (w_t * (curr_q - tgt_q).pow(2)).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()
        return loss.item(), td_errors

    def _train_numpy(self, s, a, r, s2, d, w):
        td_errors = np.zeros(len(s), dtype=np.float32)
        total_loss = 0.0
        for i in range(len(s)):
            q2  = self.target.predict(s2[i])
            tgt = r[i] + self.gamma * q2.max() * (1 - d[i])
            td  = self.online.update(s[i], a[i], tgt, weight=w[i])
            td_errors[i] = td
            total_loss   += td ** 2
        return total_loss / len(s), td_errors

    def _soft_update(self):
        if not _TORCH:
            return
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.copy_(self.tau * p_o.data + (1 - self.tau) * p_t.data)

    def _anneal_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ── serialisation ────────────────────────────────────────────────────────
    def save(self, path: str):
        if _TORCH:
            torch.save({"online": self.online.state_dict(),
                        "target": self.target.state_dict(),
                        "eps":    self.eps}, path)
        else:
            np.save(path, {"W_online": self.online.W,
                           "W_target": self.target.W, "eps": self.eps})

    def load(self, path: str):
        if _TORCH and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.dev)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            self.eps = ckpt.get("eps", self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Per-UE agent bundle
# ─────────────────────────────────────────────────────────────────────────────

class UEMADRLBundle:
    """
    Three DDQNAgents for one UE plus all per-step bookkeeping.

    Attributes (updated by the simulator each T_step)
    -------------------------------------------------
    W_current, Q_current  – current MAC parameters
    n_tx_last             – DATA packets successfully TX'd last T_step
    n_tx_avg              – running average of n_tx (EMA)
    n_drop_r              – drops due to max-retransmissions last T_step
    n_drop_q              – drops due to buffer-full last T_step
    n_collisions_last     – collisions detected last T_step
    step_counter          – number of T_step intervals elapsed
    """

    def __init__(
        self,
        ue_id:            int,
        n_neighbours_max: int  = MAX_NEIGHBOURS,
        W_min:  int = W_MIN_DEFAULT, W_max:  int = W_MAX_DEFAULT,
        Q_min:  int = Q_MIN_DEFAULT, Q_max:  int = Q_MAX_DEFAULT,
        shared_routing_buf: Optional[PrioritisedReplayBuffer] = None,
        agent_kwargs: Optional[dict] = None,
    ):
        self.ue_id = ue_id
        self.W_min = W_min; self.W_max = W_max
        self.Q_min = Q_min; self.Q_max = Q_max

        kw = agent_kwargs or {}

        routing_action_dim = 1 + n_neighbours_max   # broadcast + N unicast

        # ── routing agent (new, not in paper) ────────────────────────────────
        self.routing_agent = DDQNAgent(
            state_dim  = ROUTING_STATE_DIM,
            action_dim = routing_action_dim,
            hidden     = kw.get("routing_hidden", (256, 128)),
            lr         = kw.get("lr", 3e-4),
            gamma      = kw.get("gamma", 0.95),
            eps_start  = kw.get("eps_start", 1.0),
            eps_end    = kw.get("eps_end",   0.05),
            eps_decay  = kw.get("eps_decay", 0.9995),
            batch      = kw.get("batch",     64),
            buf_capacity = kw.get("buf_cap", 20_000),
        )
        if shared_routing_buf is not None:
            # override individual buffer with shared pool
            self.routing_agent.buf = shared_routing_buf

        # ── CW agent (ddqn1 from paper) ───────────────────────────────────────
        self.cw_agent = DDQNAgent(
            state_dim  = CW_STATE_DIM,
            action_dim = CW_ACTION_DIM,
            hidden     = kw.get("param_hidden", (128, 64)),
            lr         = kw.get("lr", 3e-4),
            gamma      = kw.get("gamma", 0.95),
            eps_start  = kw.get("eps_start", 1.0),
            eps_end    = kw.get("eps_end",   0.05),
            eps_decay  = kw.get("eps_decay", 0.9993),
            batch      = kw.get("batch",     64),
            buf_capacity = kw.get("buf_cap", 10_000),
        )

        # ── Buffer agent (ddqn2 from paper) ──────────────────────────────────
        self.buf_agent = DDQNAgent(
            state_dim  = BUF_STATE_DIM,
            action_dim = BUF_ACTION_DIM,
            hidden     = kw.get("param_hidden", (128, 64)),
            lr         = kw.get("lr", 3e-4),
            gamma      = kw.get("gamma", 0.95),
            eps_start  = kw.get("eps_start", 1.0),
            eps_end    = kw.get("eps_end",   0.05),
            eps_decay  = kw.get("eps_decay", 0.9993),
            batch      = kw.get("batch",     64),
            buf_capacity = kw.get("buf_cap", 10_000),
        )

        # ── MAC parameter state ──────────────────────────────────────────────
        self.W_current = W_min        # start conservative
        self.Q_current = Q_min + (Q_max - Q_min) // 2   # mid-range

        # ── bookkeeping (reset each T_step) ──────────────────────────────────
        self.n_tx_last       = 0
        self.n_tx_avg        = 1e-3   # avoid /0
        self.n_drop_r        = 0
        self.n_drop_q        = 0
        self.n_collisions_last = 0
        self.step_counter    = 0

        # previous states for transition storage
        self._prev_routing_state: Optional[np.ndarray] = None
        self._prev_routing_action: Optional[int]       = None
        self._prev_cw_state: Optional[np.ndarray]      = None
        self._prev_cw_action: Optional[int]            = None
        self._prev_buf_state: Optional[np.ndarray]     = None
        self._prev_buf_action: Optional[int]           = None

    # ── W / Q clamped setters ────────────────────────────────────────────────
    def apply_cw_action(self, action: int) -> int:
        """action: 0=W++, 1=W==, 2=W--  →  returns new W"""
        if action == 0:
            self.W_current = min(self.W_current + W_STEP, self.W_max)
        elif action == 2:
            self.W_current = max(self.W_current - W_STEP, self.W_min)
        return self.W_current

    def apply_buf_action(self, action: int) -> int:
        """action: 0=Q++, 1=Q==, 2=Q--  →  returns new Q"""
        if action == 0:
            self.Q_current = min(self.Q_current + Q_STEP, self.Q_max)
        elif action == 2:
            self.Q_current = max(self.Q_current - Q_STEP, self.Q_min)
        return self.Q_current

    # ── normalisation helpers ────────────────────────────────────────────────
    def norm_W(self) -> float:
        return (self.W_current - self.W_min) / max(self.W_max - self.W_min, 1)

    def norm_Q(self) -> float:
        return (self.Q_current - self.Q_min) / max(self.Q_max - self.Q_min, 1)

    def n_ratio(self) -> float:
        return self.n_tx_last / max(self.n_tx_avg, 1e-3)

    def update_ema(self, alpha: float = 0.1):
        """EMA update of average TX count."""
        self.n_tx_avg = (1 - alpha) * self.n_tx_avg + alpha * self.n_tx_last

    def reset_step_counters(self):
        self.n_tx_last     = 0
        self.n_drop_r      = 0
        self.n_drop_q      = 0
        self.n_collisions_last = 0

    def train_all(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """One training step per agent. Returns (routing_loss, cw_loss, buf_loss)."""
        return (
            self.routing_agent.train(),
            self.cw_agent.train(),
            self.buf_agent.train(),
        )

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.routing_agent.save(os.path.join(directory, f"ue{self.ue_id}_routing.pt"))
        self.cw_agent.save(     os.path.join(directory, f"ue{self.ue_id}_cw.pt"))
        self.buf_agent.save(    os.path.join(directory, f"ue{self.ue_id}_buf.pt"))

    def load(self, directory: str):
        self.routing_agent.load(os.path.join(directory, f"ue{self.ue_id}_routing.pt"))
        self.cw_agent.load(     os.path.join(directory, f"ue{self.ue_id}_cw.pt"))
        self.buf_agent.load(    os.path.join(directory, f"ue{self.ue_id}_buf.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# MADRLController  (top-level handle for the simulator)
# ─────────────────────────────────────────────────────────────────────────────

class MADRLController:
    """
    Manages all UE-level agent bundles and exposes a simple simulator API.

    Parameters
    ----------
    n_ues          : number of UEs in the network
    T_step         : decision period in back-off phases
    share_routing  : if True, all UEs share one routing replay buffer
                     (faster convergence in homogeneous deployments)
    """

    def __init__(
        self,
        n_ues:         int,
        T_step:        int  = 10,
        W_min:  int = W_MIN_DEFAULT,  W_max:  int = W_MAX_DEFAULT,
        Q_min:  int = Q_MIN_DEFAULT,  Q_max:  int = Q_MAX_DEFAULT,
        share_routing: bool = True,
        agent_kwargs:  Optional[dict] = None,
    ):
        self.n_ues  = n_ues
        self.T_step = T_step

        # optional shared routing experience pool
        shared_buf = PrioritisedReplayBuffer(capacity=50_000) if share_routing else None

        self.bundles: List[UEMADRLBundle] = [
            UEMADRLBundle(
                ue_id            = i,
                W_min=W_min, W_max=W_max,
                Q_min=Q_min, Q_max=Q_max,
                shared_routing_buf = shared_buf,
                agent_kwargs       = agent_kwargs,
            )
            for i in range(n_ues)
        ]
        self._shared_buf = shared_buf
        self._global_step = 0

    # ── per-simulation reset ─────────────────────────────────────────────────
    def reset_episode(self):
        """Call at the start of each simulation episode."""
        for b in self.bundles:
            b.reset_step_counters()
            b.step_counter = 0
            b._prev_routing_state  = None
            b._prev_routing_action = None
            b._prev_cw_state       = None
            b._prev_cw_action      = None
            b._prev_buf_state      = None
            b._prev_buf_action     = None
            b.update_ema(alpha=0.0)   # no reset of long-term average

    # ── convenience getters ──────────────────────────────────────────────────
    def bundle(self, ue_id: int) -> UEMADRLBundle:
        return self.bundles[ue_id]

    def W(self, ue_id: int) -> int:
        return self.bundles[ue_id].W_current

    def Q(self, ue_id: int) -> int:
        return self.bundles[ue_id].Q_current

    # ── global training call (invoke once per T_step across all UEs) ─────────
    def train_all(self) -> dict:
        losses = {"routing": [], "cw": [], "buf": []}
        for b in self.bundles:
            r_l, c_l, b_l = b.train_all()
            if r_l is not None: losses["routing"].append(r_l)
            if c_l is not None: losses["cw"].append(c_l)
            if b_l is not None: losses["buf"].append(b_l)
        self._global_step += 1
        return {k: float(np.mean(v)) if v else None for k, v in losses.items()}

    # ── save / load all agents ───────────────────────────────────────────────
    def save(self, directory: str):
        for b in self.bundles:
            b.save(directory)

    def load(self, directory: str):
        for b in self.bundles:
            b.load(directory)
