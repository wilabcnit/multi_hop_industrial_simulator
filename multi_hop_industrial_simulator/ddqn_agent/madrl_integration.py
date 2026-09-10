"""
madrl_integration.py
====================
State builders, reward functions, and the drop-in routing function
``choose_next_action_madrl`` that replaces ``choose_next_action_tb_no_RL``.

State definitions
-----------------
Routing state  (ROUTING_STATE_DIM = 89)
    Per-neighbour block (MAX_NEIGHBOURS × 4):
        [0] ack_rate_i      – ACK success rate toward neighbour i  ∈ [0,1]
        [1] power_i         – normalised RX power from neighbour i ∈ [0,1]
        [2] bs_seen_i       – 1 if neighbour i can reach BS        ∈ {0,1}
        [3] active_i        – 1 if slot is occupied by a real UE   ∈ {0,1}
    Global block (9):
        [80] buffer_util    – len(buffer) / Q_current              ∈ [0,1]
        [81] n_ratio        – N_t / N_t_avg                        ∈ [0,∞)  clipped 2
        [82] norm_W         – (W-W_min)/(W_max-W_min)              ∈ [0,1]
        [83] norm_Q         – (Q-Q_min)/(Q_max-Q_min)              ∈ [0,1]
        [84] hop_norm       – hop_count / hop_limit                ∈ [0,1]
        [85] n_active_norm  – #active neighbours / MAX_NEIGHBOURS  ∈ [0,1]
        [86] direct_bs      – 1 if this UE can reach BS directly   ∈ {0,1}
        [87] collision_rate – recent collision fraction            ∈ [0,1]
        [88] drop_rate      – (D_R==0 ? 1 : 0)  last step         ∈ {0,1}

CW state  (CW_STATE_DIM = 6)
    [n_ratio, norm_W, collision_rate, buffer_util, direct_bs, drop_r_flag]

Buffer state  (BUF_STATE_DIM = 6)
    [n_ratio, norm_Q, buffer_util, drop_q_flag, n_active_norm, direct_bs]

Reward (shared base + agent-specific bonus)
-------------------------------------------
    base   = σ·N_t  +  β·D_R  +  δ·D_Q
    routing bonus:  +hop_efficiency  (shorter path preferred)
                    -collision_penalty
    cw bonus:       -collision_fraction * 0.5
    buf bonus:      -drop_q_flag * 0.5
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from madrl_agent import UEMADRLBundle, MADRLController

# avoid circular import at runtime
from madrl_agent import (
    MAX_NEIGHBOURS, ROUTING_STATE_DIM, CW_STATE_DIM, BUF_STATE_DIM,
)

# ── reward hyper-parameters ───────────────────────────────────────────────────
SIGMA = 1.0    # weight for successful TX count
BETA  = 2.0    # reward for zero retx-drops
DELTA = 2.0    # reward for zero buffer-drops
HOP_EFF_SCALE  = 0.5   # hop-efficiency bonus scale
COLL_PENALTY   = 0.3   # collision penalty scale

# ── power normalisation constants (adjust to your THz scenario) ───────────────
P_RX_MIN_DBM = -100.0   # minimum expected received power [dBm]
P_RX_MAX_DBM = -30.0    # maximum expected received power [dBm]


# ═════════════════════════════════════════════════════════════════════════════
# State builders
# ═════════════════════════════════════════════════════════════════════════════

def _norm_power(p_dbm: float) -> float:
    """Map received power in dBm to [0, 1]."""
    return float(np.clip(
        (p_dbm - P_RX_MIN_DBM) / (P_RX_MAX_DBM - P_RX_MIN_DBM), 0.0, 1.0))


def build_routing_state(
    ue,                         # simulator UE object
    bundle:  "UEMADRLBundle",
    hop_limit: int,
) -> np.ndarray:
    """
    Construct the routing-agent state vector (length ROUTING_STATE_DIM).

    ue.obs layout expected from simulator:
        ue.obs[0]  – list of bs_seen flags per neighbour (+ [-1] = own BS flag)
        ue.obs[1]  – list of ACK count per neighbour
        ue.obs[2]  – list of last RX power per neighbour
        ue.neighbour_table – list of neighbour ID strings

    The UE's ul_buffer and W/Q come from the bundle.
    """
    state = np.zeros(ROUTING_STATE_DIM, dtype=np.float32)

    n_neighbours = len(ue.neighbour_table)
    n_active = 0

    for i in range(min(n_neighbours, MAX_NEIGHBOURS)):
        base = i * 4

        # ack_rate: ACK count normalised by max observed
        ack_cnt = float(ue.obs[1][i]) if len(ue.obs[1]) > i else 0.0
        max_ack = max(float(np.max(ue.obs[1])) if len(ue.obs[1]) else 1.0, 1.0)
        state[base]   = np.clip(ack_cnt / max_ack, 0.0, 1.0)

        # rx power
        pwr_val = float(ue.obs[2][i]) if len(ue.obs[2]) > i else P_RX_MIN_DBM
        state[base+1] = _norm_power(pwr_val)

        # bs_seen flag — whether neighbour i has visibility to the BS (obs row 4)
        obs4 = ue.obs[4] if len(ue.obs) > 4 else []
        state[base+2] = float(obs4[i]) if len(obs4) > i else 0.0

        # active/discovered — whether this neighbour has been seen at all (obs row 0)
        state[base+3] = float(ue.obs[0][i]) if len(ue.obs[0]) > i else 0.0
        n_active += 1

    # ── global features ───────────────────────────────────────────────────────
    g = MAX_NEIGHBOURS * 4   # = 80

    buf_len = len(ue.ul_buffer.buffer_packet_list)
    state[g]   = float(buf_len) / max(bundle.Q_current, 1)           # buffer_util
    state[g+1] = float(np.clip(bundle.n_ratio(), 0.0, 2.0))          # n_ratio
    state[g+2] = bundle.norm_W()                                       # norm_W
    state[g+3] = bundle.norm_Q()                                       # norm_Q

    # hop count of first packet in buffer
    if buf_len > 0:
        hc = getattr(ue.ul_buffer.get_first_packet(), 'hop_count', 0)
        state[g+4] = float(np.clip(hc / max(hop_limit, 1), 0.0, 1.0))
    state[g+5] = float(n_active) / MAX_NEIGHBOURS                     # n_active_norm
    state[g+6] = float(ue.obs[0][-1]) if len(ue.obs[0]) > 0 else 0.0  # direct_bs
    # collision_rate from bundle counters
    total_tx   = max(bundle.n_tx_last + bundle.n_collisions_last, 1)
    state[g+7] = float(bundle.n_collisions_last) / total_tx
    state[g+8] = float(bundle.n_drop_r > 0)                           # drop_r flag

    return state


def build_cw_state(
    bundle: "UEMADRLBundle",
    direct_bs: float,
) -> np.ndarray:
    """Construct the CW-agent state vector (length CW_STATE_DIM = 6)."""
    total_tx = max(bundle.n_tx_last + bundle.n_collisions_last, 1)
    coll_rate = float(bundle.n_collisions_last) / total_tx
    buf_util  = float(len(bundle.__dict__.get("_buf_len_cache", [0]))) \
                if hasattr(bundle, "_buf_len_cache") else 0.0

    return np.array([
        float(np.clip(bundle.n_ratio(), 0.0, 2.0)),   # n_ratio
        bundle.norm_W(),                                # norm_W
        coll_rate,                                      # collision_rate
        float(bundle.n_drop_r > 0),                    # drop_r_flag
        direct_bs,                                      # direct_bs
        float(bundle.n_drop_q > 0),                    # drop_q_flag
    ], dtype=np.float32)


def build_buf_state(
    ue,
    bundle: "UEMADRLBundle",
    n_active_norm: float,
    direct_bs: float,
) -> np.ndarray:
    """Construct the buffer-agent state vector (length BUF_STATE_DIM = 6)."""
    buf_util = float(len(ue.ul_buffer.buffer_packet_list)) / max(bundle.Q_current, 1)

    return np.array([
        float(np.clip(bundle.n_ratio(), 0.0, 2.0)),   # n_ratio
        bundle.norm_Q(),                                # norm_Q
        float(np.clip(buf_util, 0.0, 1.0)),            # buffer_util
        float(bundle.n_drop_q > 0),                    # drop_q_flag
        n_active_norm,                                  # n_active_norm
        direct_bs,                                      # direct_bs
    ], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Reward functions
# ═════════════════════════════════════════════════════════════════════════════

def compute_base_reward(bundle: "UEMADRLBundle") -> float:
    """
    Shared base reward (identical formula for all three agents,
    as in the paper):  r = σ·N_t + β·D_R + δ·D_Q
    """
    D_R = 0.0 if bundle.n_drop_r  > 0 else 1.0
    D_Q = 0.0 if bundle.n_drop_q  > 0 else 1.0
    r   = SIGMA * bundle.n_tx_last + BETA * D_R + DELTA * D_Q
    return float(r)


def compute_routing_reward(
    bundle:         "UEMADRLBundle",
    hop_count:      int,
    hop_limit:      int,
    ack_received:   bool,
    packet_at_bs:   bool,
) -> float:
    """
    Extended reward for the routing agent:
        base_reward
      + hop_efficiency_bonus   (fewer hops = better)
      + delivery_bonus         (packet confirmed at BS)
      - collision_penalty
    """
    r = compute_base_reward(bundle)

    # hop efficiency: reward if path is short
    hop_frac = float(np.clip(hop_count / max(hop_limit, 1), 0.0, 1.0))
    r += HOP_EFF_SCALE * (1.0 - hop_frac)

    # confirmed delivery at BS
    if packet_at_bs:
        r += 3.0

    # ACK received (intermediate feedback)
    if ack_received:
        r += 0.5

    # collision penalty
    total_tx = max(bundle.n_tx_last + bundle.n_collisions_last, 1)
    coll_frac = float(bundle.n_collisions_last) / total_tx
    r -= COLL_PENALTY * coll_frac

    return float(r)


def compute_cw_reward(bundle: "UEMADRLBundle") -> float:
    """CW-agent reward: base + collision shaping."""
    r = compute_base_reward(bundle)
    total_tx  = max(bundle.n_tx_last + bundle.n_collisions_last, 1)
    coll_frac = float(bundle.n_collisions_last) / total_tx
    r -= 0.5 * coll_frac
    return float(r)


def compute_buf_reward(bundle: "UEMADRLBundle") -> float:
    """Buffer-agent reward: base + overflow shaping."""
    r  = compute_base_reward(bundle)
    r -= 0.5 * float(bundle.n_drop_q > 0)
    return float(r)


# ═════════════════════════════════════════════════════════════════════════════
# Routing action → simulator action mapping
# ═════════════════════════════════════════════════════════════════════════════
# Simulator TB actions:
#   0 = unicast to best neighbour / BS
#   1 = (reserved / second unicast)
#   2 = broadcast
#   3 = forced broadcast
#
# MADRL routing actions:
#   0            = broadcast
#   1 .. N_nbrs  = unicast to neighbour index (i-1) in neighbour_table
#
# We translate MADRL actions to simulator actions below.


def _build_valid_mask(ue, n_neighbours: int) -> np.ndarray:
    """
    Boolean mask over routing action space.
    Action 0 (broadcast) is always valid.
    Action 1+i is valid only if neighbour i has exchanged at least one ACK
    or has a positive RX power entry.
    """
    from madrl_agent import ROUTING_ACTION_DIM
    mask = np.zeros(ROUTING_ACTION_DIM, dtype=bool)
    mask[0] = True  # broadcast always valid

    for i in range(min(n_neighbours, MAX_NEIGHBOURS)):
        ack_ok = (len(ue.obs[1]) > i and ue.obs[1][i] > 0)
        pwr_ok = (len(ue.obs[2]) > i and ue.obs[2][i] > P_RX_MIN_DBM + 5)
        if ack_ok or pwr_ok:
            mask[1 + i] = True

    # if no unicast candidate known yet → force broadcast only
    if not mask[1:].any():
        mask[0] = True

    return mask


def choose_next_action_madrl(
    ue,
    bundle:       "UEMADRLBundle",
    hop_limit:    int,
    enable_print: bool = False,
) -> None:
    """
    Drop-in replacement for ``choose_next_action_tb_no_RL``.

    Selects the routing action via the routing DDQN and applies it
    to the UE's routing state (broadcast_bool, unicast_rx_address, etc.).

    Must be called at the same code-path as the original function.
    """
    n_nbrs = len(ue.neighbour_table)
    state  = build_routing_state(ue, bundle, hop_limit)
    mask   = _build_valid_mask(ue, n_nbrs)

    action = bundle.routing_agent.act(state, valid_mask=mask)

    # ── store previous transition if we have one ──────────────────────────────
    if bundle._prev_routing_state is not None:
        # reward is not yet final here; use base reward as proxy
        # (the per-BO-phase reward update happens in _madrl_step_update)
        dummy_reward = compute_base_reward(bundle)
        bundle.routing_agent.push(
            bundle._prev_routing_state,
            bundle._prev_routing_action,
            dummy_reward,
            state,
            False,   # not terminal mid-simulation
        )

    bundle._prev_routing_state  = state.copy()
    bundle._prev_routing_action = action

    # ── translate MADRL action → simulator action ─────────────────────────────
    if action == 0:
        # broadcast
        ue.set_broadcast_bool(input_broadcast_bool=True)
        ue.set_last_action(input_last_action=2)
        if enable_print:
            print(f"[MADRL] UE {ue.get_ue_id()} → BROADCAST")

    else:
        nbr_idx = action - 1   # 0-based index into neighbour_table
        if nbr_idx < n_nbrs:
            nbr_id  = ue.neighbour_table[nbr_idx]
            # check if neighbour is the BS
            if nbr_id == "BS" or str(nbr_id).upper() == "BS":
                ue.set_unicast_rx_address(input_unicast_rx_address="BS")
                ue.set_unicast_rx_index(input_unicast_rx_index=nbr_idx)
                ue.set_broadcast_bool(input_broadcast_bool=False)
                ue.set_last_action(input_last_action=0)
                if enable_print:
                    print(f"[MADRL] UE {ue.get_ue_id()} → UNICAST to BS")
            else:
                ue.set_unicast_rx_address(input_unicast_rx_address=str(nbr_id))
                ue.set_unicast_rx_index(input_unicast_rx_index=nbr_idx)
                ue.set_broadcast_bool(input_broadcast_bool=False)
                ue.set_last_action(input_last_action=0)
                if enable_print:
                    print(f"[MADRL] UE {ue.get_ue_id()} → UNICAST to UE {nbr_id}")
        else:
            # fallback: broadcast if index out of range
            ue.set_broadcast_bool(input_broadcast_bool=True)
            ue.set_last_action(input_last_action=2)

    # ── assign address to all buffered packets (matches choose_next_action_tb_no_RL) ──
    for packet in ue.ul_buffer.buffer_packet_list:
        if ue.get_broadcast_bool() is False:
            packet.address = str(ue.get_unicast_rx_address())
        else:
            packet.address = "-1"


# ═════════════════════════════════════════════════════════════════════════════
# T_step update hook  (call every T_step back-off phases per UE)
# ═════════════════════════════════════════════════════════════════════════════

def madrl_step_update(
    ue,
    bundle:       "UEMADRLBundle",
    hop_limit:    int,
    ack_received: bool      = False,
    packet_at_bs: bool      = False,
    done:         bool      = False,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Called every T_step back-off phases.  Performs:
        1. EMA update of average TX count
        2. Build next states for all three agents
        3. Compute rewards and push final transitions to replay buffers
        4. Apply CW and buffer actions
        5. Trigger one training step per agent
        6. Reset step counters

    Returns (routing_loss, cw_loss, buf_loss) – may be None if buffer not warm.
    """
    # ── 1. update long-run average ────────────────────────────────────────────
    bundle.update_ema()

    # ── 2. direct-BS flag ─────────────────────────────────────────────────────
    direct_bs    = float(ue.obs[0][-1]) if len(ue.obs[0]) > 0 else 0.0
    n_nbrs       = len(ue.neighbour_table)
    n_active_norm = float(min(n_nbrs, MAX_NEIGHBOURS)) / MAX_NEIGHBOURS

    # ── 3. build current (next) states ───────────────────────────────────────
    next_routing_state = build_routing_state(ue, bundle, hop_limit)
    next_cw_state      = build_cw_state(bundle, direct_bs)
    next_buf_state     = build_buf_state(ue, bundle, n_active_norm, direct_bs)

    hop_count = 0
    if len(ue.ul_buffer.buffer_packet_list) > 0:
        hop_count = getattr(ue.ul_buffer.get_first_packet(), 'hop_count', 0)

    # ── 4. compute rewards ────────────────────────────────────────────────────
    r_routing = compute_routing_reward(bundle, hop_count, hop_limit,
                                       ack_received, packet_at_bs)
    r_cw      = compute_cw_reward(bundle)
    r_buf     = compute_buf_reward(bundle)

    # ── 5. push transitions ───────────────────────────────────────────────────
    if bundle._prev_routing_state is not None:
        bundle.routing_agent.push(
            bundle._prev_routing_state, bundle._prev_routing_action,
            r_routing, next_routing_state, done)

    if bundle._prev_cw_state is not None:
        bundle.cw_agent.push(
            bundle._prev_cw_state, bundle._prev_cw_action,
            r_cw, next_cw_state, done)

    if bundle._prev_buf_state is not None:
        bundle.buf_agent.push(
            bundle._prev_buf_state, bundle._prev_buf_action,
            r_buf, next_buf_state, done)

    # ── 6. select new CW and buffer actions ──────────────────────────────────
    cw_action  = bundle.cw_agent.act(next_cw_state)
    buf_action = bundle.buf_agent.act(next_buf_state)

    # ── 7. apply parameter actions ────────────────────────────────────────────
    bundle.apply_cw_action(cw_action)
    bundle.apply_buf_action(buf_action)

    # ── 8. save states for next transition ────────────────────────────────────
    bundle._prev_routing_state  = next_routing_state.copy()
    # routing action already saved in choose_next_action_madrl; keep as-is

    bundle._prev_cw_state  = next_cw_state.copy()
    bundle._prev_cw_action = cw_action

    bundle._prev_buf_state  = next_buf_state.copy()
    bundle._prev_buf_action = buf_action

    # ── 9. train ──────────────────────────────────────────────────────────────
    losses = bundle.train_all()
    bundle.step_counter += 1
    bundle.reset_step_counters()

    return losses


# ═════════════════════════════════════════════════════════════════════════════
# Episode termination hook
# ═════════════════════════════════════════════════════════════════════════════

def madrl_episode_end(
    ue,
    bundle:    "UEMADRLBundle",
    hop_limit: int,
) -> None:
    """
    Push terminal transitions for all pending previous states.
    Call at simulation end for every UE.
    """
    direct_bs     = float(ue.obs[0][-1]) if len(ue.obs[0]) > 0 else 0.0
    n_active_norm = float(min(len(ue.neighbour_table), MAX_NEIGHBOURS)) / MAX_NEIGHBOURS

    terminal_routing = build_routing_state(ue, bundle, hop_limit)
    terminal_cw      = build_cw_state(bundle, direct_bs)
    terminal_buf     = build_buf_state(ue, bundle, n_active_norm, direct_bs)

    r_routing = compute_routing_reward(bundle, 0, hop_limit, False, False)
    r_cw      = compute_cw_reward(bundle)
    r_buf     = compute_buf_reward(bundle)

    if bundle._prev_routing_state is not None:
        bundle.routing_agent.push(
            bundle._prev_routing_state, bundle._prev_routing_action or 0,
            r_routing, terminal_routing, True)

    if bundle._prev_cw_state is not None:
        bundle.cw_agent.push(
            bundle._prev_cw_state, bundle._prev_cw_action or 1,
            r_cw, terminal_cw, True)

    if bundle._prev_buf_state is not None:
        bundle.buf_agent.push(
            bundle._prev_buf_state, bundle._prev_buf_action or 1,
            r_buf, terminal_buf, True)