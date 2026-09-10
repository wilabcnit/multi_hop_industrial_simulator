"""
train_madrl.py
==============
Training orchestrator for the MADRL-TB routing system.

Features
--------
Curriculum scheduling
    Phase 0  (warm-up)      : high ε, no parameter updates → fill replay buffers
    Phase 1  (exploration)  : ε anneals, agents learn basic routing
    Phase 2  (exploitation) : ε low, W/Q fine-tuning active
    Phase 3  (convergence)  : frozen routing agent, only W/Q adapt

Per-episode logging
    JSON-lines file with: rewards, losses, W/Q values, KPIs, epsilon

Early stopping
    Stop if mean Jain index > JAIN_THRESHOLD for PATIENCE consecutive episodes

Evaluation harness
    Every EVAL_EVERY episodes run a deterministic greedy rollout (ε=0)
    and log the three KPIs: J, S [Mbit/s], L̄ [ms]
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from madrl_agent import (
    MADRLController,
    W_MIN_DEFAULT, W_MAX_DEFAULT,
    Q_MIN_DEFAULT, Q_MAX_DEFAULT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum schedule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurriculumPhase:
    name:            str
    min_episode:     int        # first episode of this phase
    eps_override:    Optional[float] = None   # None = use agent's own schedule
    freeze_routing:  bool = False
    freeze_cw:       bool = False
    freeze_buf:      bool = False
    description:     str = ""


CURRICULUM: List[CurriculumPhase] = [
    CurriculumPhase(
        name="warm_up", min_episode=0, eps_override=1.0,
        freeze_routing=True, freeze_cw=True, freeze_buf=True,
        description="Fill replay buffers; all actions random"),
    CurriculumPhase(
        name="explore_routing", min_episode=10, eps_override=None,
        freeze_routing=False, freeze_cw=True, freeze_buf=True,
        description="Train routing only; W/Q fixed"),
    CurriculumPhase(
        name="joint_learning", min_episode=30, eps_override=None,
        freeze_routing=False, freeze_cw=False, freeze_buf=False,
        description="All agents learn jointly"),
    CurriculumPhase(
        name="fine_tune_params", min_episode=80, eps_override=None,
        freeze_routing=True, freeze_cw=False, freeze_buf=False,
        description="Routing frozen; fine-tune W and Q only"),
]


def get_current_phase(episode: int) -> CurriculumPhase:
    phase = CURRICULUM[0]
    for p in CURRICULUM:
        if episode >= p.min_episode:
            phase = p
    return phase


# ─────────────────────────────────────────────────────────────────────────────
# Episode record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    episode:         int
    phase:           str
    n_ue:            int
    seed:            int
    wall_time_s:     float = 0.0
    # KPIs (averaged over UEs)
    jain_index:      float = 0.0
    throughput_mbps: float = 0.0
    latency_ms:      float = 0.0
    p_mac:           float = 0.0
    # Agent statistics (averaged over UEs)
    mean_routing_loss: Optional[float] = None
    mean_cw_loss:      Optional[float] = None
    mean_buf_loss:     Optional[float] = None
    mean_epsilon:      float = 1.0
    # MAC parameters (averaged over UEs)
    mean_W:          float = 0.0
    mean_Q:          float = 0.0
    std_W:           float = 0.0
    std_Q:           float = 0.0
    # Per-UE reward proxy
    mean_reward:     float = 0.0
    # Raw losses per agent type
    routing_losses:  List[float] = field(default_factory=list)
    cw_losses:       List[float] = field(default_factory=list)
    buf_losses:      List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # convert numpy types to python
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            elif isinstance(v, list):
                d[k] = [float(x) for x in v]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """Writes JSON-lines to a log file and prints a human-readable summary."""

    def __init__(self, log_path: str, print_every: int = 1):
        self.log_path    = log_path
        self.print_every = print_every
        self._fh         = open(log_path, 'a')
        self._episode    = 0

    def log(self, rec: EpisodeRecord):
        self._fh.write(json.dumps(rec.to_dict()) + "\n")
        self._fh.flush()
        self._episode += 1
        if self._episode % self.print_every == 0:
            self._print(rec)

    def _print(self, r: EpisodeRecord):
        rl  = f"{r.mean_routing_loss:.4f}" if r.mean_routing_loss else "  —   "
        cwl = f"{r.mean_cw_loss:.4f}"      if r.mean_cw_loss      else "  —   "
        bl  = f"{r.mean_buf_loss:.4f}"     if r.mean_buf_loss     else "  —   "
        print(
            f"[ep {r.episode:4d} | {r.phase:<20s}]  "
            f"J={r.jain_index:.3f}  "
            f"S={r.throughput_mbps:.2f}Mbit/s  "
            f"L={r.latency_ms:.2f}ms  "
            f"ε={r.mean_epsilon:.3f}  "
            f"W={r.mean_W:.1f}±{r.std_W:.1f}  "
            f"Q={r.mean_Q:.1f}±{r.std_Q:.1f}  "
            f"r_loss={rl}  cw_loss={cwl}  buf_loss={bl}  "
            f"t={r.wall_time_s:.1f}s"
        )

    def close(self):
        self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopper:
    """
    Triggers when mean Jain index exceeds threshold for `patience` consecutive
    evaluation windows, or when throughput improvement stalls.
    """

    def __init__(self,
                 jain_threshold:   float = 0.95,
                 patience:         int   = 5,
                 min_improvement:  float = 1e-4):
        self.threshold      = jain_threshold
        self.patience       = patience
        self.min_improve    = min_improvement
        self._consec        = 0
        self._best_j        = 0.0
        self._best_s        = 0.0
        self._no_improve    = 0

    def step(self, jain: float, throughput: float) -> bool:
        """Returns True if training should stop."""
        improved = False
        if jain > self._best_j + self.min_improve:
            self._best_j   = jain
            improved       = True
        if throughput > self._best_s + self.min_improve:
            self._best_s   = throughput
            improved       = True

        if improved:
            self._no_improve = 0
        else:
            self._no_improve += 1

        if jain >= self.threshold:
            self._consec += 1
        else:
            self._consec = 0

        return (self._consec >= self.patience
                or self._no_improve >= self.patience * 5)


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum controller
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumController:
    """
    Applies per-phase overrides to a MADRLController and its agents.

    Call `apply(episode)` before each episode to enforce the correct
    freeze / epsilon settings.
    """

    def __init__(self, ctrl: MADRLController):
        self.ctrl         = ctrl
        self._last_phase  = None

    def apply(self, episode: int):
        phase = get_current_phase(episode)
        if phase.name == self._last_phase:
            return   # nothing changed

        print(f"\n[Curriculum] ── Phase: {phase.name}  ──  {phase.description}")
        self._last_phase = phase.name

        for bundle in self.ctrl.bundles:
            # epsilon override
            if phase.eps_override is not None:
                bundle.routing_agent.eps = phase.eps_override
                bundle.cw_agent.eps      = phase.eps_override
                bundle.buf_agent.eps     = phase.eps_override

            # freeze flags are enforced in train_step() wrapper below

        self._freeze_routing = phase.freeze_routing
        self._freeze_cw      = phase.freeze_cw
        self._freeze_buf     = phase.freeze_buf

    def train_all_gated(self) -> dict:
        """
        Like MADRLController.train_all() but respects curriculum freeze flags.
        """
        losses = {"routing": [], "cw": [], "buf": []}
        for b in self.ctrl.bundles:
            if not self._freeze_routing:
                l = b.routing_agent.train()
                if l is not None: losses["routing"].append(l)
            if not self._freeze_cw:
                l = b.cw_agent.train()
                if l is not None: losses["cw"].append(l)
            if not self._freeze_buf:
                l = b.buf_agent.train()
                if l is not None: losses["buf"].append(l)
        self.ctrl._global_step += 1
        return {k: float(np.mean(v)) if v else None for k, v in losses.items()}

    @property
    def freeze_routing(self): return self._freeze_routing
    @property
    def freeze_cw(self):      return self._freeze_cw
    @property
    def freeze_buf(self):     return self._freeze_buf


# ─────────────────────────────────────────────────────────────────────────────
# KPI aggregator
# ─────────────────────────────────────────────────────────────────────────────

class KPIAggregator:
    """
    Accumulates per-simulation output_dict entries and computes
    Jain index, throughput, and latency.

    Compatible with the ``output_dict`` structure written by
    compute_simulator_outputs().
    """

    def __init__(self, window: int = 10):
        self._window  = window
        self._j_hist  = deque(maxlen=window)
        self._s_hist  = deque(maxlen=window)
        self._l_hist  = deque(maxlen=window)
        self._pm_hist = deque(maxlen=window)

    def update(self, output_dict: dict, n_ue: int, n_sim: int):
        tag = f"N={n_ue}"
        sim = f"Sim={n_sim}"

        j_arr  = output_dict["j_index"][tag][sim]
        s_arr  = output_dict["s"][tag][sim]
        l_arr  = output_dict["l"][tag][sim]
        pm_arr = output_dict["p_mac"][tag][sim]

        # Jain index is a scalar computed over UEs; take the mean value stored
        self._j_hist.append(float(np.mean(j_arr[j_arr > 0])) if np.any(j_arr > 0) else 0.0)
        self._s_hist.append(float(np.mean(s_arr)) * 1e-6)    # → Mbit/s
        nz = l_arr[l_arr > 0]
        self._l_hist.append(float(np.mean(nz)) * 1e3 if len(nz) else 0.0)  # → ms
        self._pm_hist.append(float(np.mean(pm_arr)))

    def latest(self) -> Tuple[float, float, float, float]:
        """Returns (jain, throughput_mbps, latency_ms, p_mac)."""
        def safe_mean(q):
            return float(np.mean(list(q))) if q else 0.0
        return (safe_mean(self._j_hist), safe_mean(self._s_hist),
                safe_mean(self._l_hist), safe_mean(self._pm_hist))

    def smoothed_jain(self) -> float:
        return float(np.mean(list(self._j_hist))) if self._j_hist else 0.0

    def smoothed_throughput(self) -> float:
        return float(np.mean(list(self._s_hist))) if self._s_hist else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # topology
    n_ue:               int   = 10
    n_simulations:      int   = 100       # total training episodes per (seed, n_ue)
    initial_seed:       int   = 0
    final_seed:         int   = 0

    # MADRL hyper-parameters
    T_step:             int   = 10
    W_min:              int   = W_MIN_DEFAULT
    W_max:              int   = W_MAX_DEFAULT
    Q_min:              int   = Q_MIN_DEFAULT
    Q_max:              int   = Q_MAX_DEFAULT
    share_routing:      bool  = True
    agent_lr:           float = 3e-4
    agent_gamma:        float = 0.95
    agent_eps_start:    float = 1.0
    agent_eps_end:      float = 0.05
    agent_eps_decay:    float = 0.9995
    agent_batch:        int   = 64
    agent_buf_cap:      int   = 20_000

    # training schedule
    eval_every:         int   = 5
    save_every:         int   = 10
    log_every:          int   = 1
    patience:           int   = 8
    jain_threshold:     float = 0.95

    # paths
    ckpt_dir:           str   = "madrl_checkpoints"
    log_dir:            str   = "madrl_logs"
    results_dir:        str   = "madrl_results"

    def agent_kwargs(self) -> dict:
        return dict(
            routing_hidden = (256, 128),
            param_hidden   = (128, 64),
            lr             = self.agent_lr,
            gamma          = self.agent_gamma,
            eps_start      = self.agent_eps_start,
            eps_end        = self.agent_eps_end,
            eps_decay      = self.agent_eps_decay,
            batch          = self.agent_batch,
            buf_cap        = self.agent_buf_cap,
        )

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            return cls(**json.load(f))


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

class GreedyEvaluator:
    """
    Temporarily sets ε=0 on all agents, runs one deterministic episode,
    then restores the original ε values.
    """

    def __init__(self, ctrl: MADRLController):
        self.ctrl = ctrl

    def __enter__(self):
        self._saved_eps = []
        for b in self.ctrl.bundles:
            self._saved_eps.append((
                b.routing_agent.eps,
                b.cw_agent.eps,
                b.buf_agent.eps,
            ))
            b.routing_agent.eps = 0.0
            b.cw_agent.eps      = 0.0
            b.buf_agent.eps     = 0.0
        return self

    def __exit__(self, *_):
        for b, (re, ce, be) in zip(self.ctrl.bundles, self._saved_eps):
            b.routing_agent.eps = re
            b.cw_agent.eps      = ce
            b.buf_agent.eps     = be


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop (simulator-agnostic wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class MADRLTrainer:
    """
    Wraps the simulation loop and orchestrates:
        - curriculum phase transitions
        - gated training (freeze / unfreeze)
        - KPI aggregation
        - early stopping
        - logging and checkpointing

    Usage
    -----
    ::

        from train_madrl import MADRLTrainer, TrainingConfig

        cfg     = TrainingConfig(n_ue=10, n_simulations=150)
        trainer = MADRLTrainer(cfg)
        trainer.run(sim_run_fn)   # sim_run_fn(ctrl) → output_dict

    where ``sim_run_fn`` accepts a MADRLController and runs one full
    simulation episode, returning the output_dict.
    """

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        os.makedirs(cfg.ckpt_dir,   exist_ok=True)
        os.makedirs(cfg.log_dir,    exist_ok=True)
        os.makedirs(cfg.results_dir, exist_ok=True)

        self.ctrl = MADRLController(
            n_ues         = cfg.n_ue,
            T_step        = cfg.T_step,
            W_min         = cfg.W_min,   W_max = cfg.W_max,
            Q_min         = cfg.Q_min,   Q_max = cfg.Q_max,
            share_routing = cfg.share_routing,
            agent_kwargs  = cfg.agent_kwargs(),
        )

        self.curriculum = CurriculumController(self.ctrl)
        self.kpi        = KPIAggregator(window=cfg.eval_every)
        self.stopper    = EarlyStopper(cfg.jain_threshold, cfg.patience)
        self.logger     = TrainingLogger(
            os.path.join(cfg.log_dir, "training_log.jsonl"),
            print_every=cfg.log_every)

        # load existing checkpoint if available
        ckpt_tag = os.path.join(cfg.ckpt_dir, "ue0_routing.pt")
        if os.path.exists(ckpt_tag):
            self.ctrl.load(cfg.ckpt_dir)
            print(f"[Trainer] Resumed from checkpoint: {cfg.ckpt_dir}")

        cfg.save(os.path.join(cfg.log_dir, "training_config.json"))
        self._global_episode = 0

    def run(self, sim_run_fn, output_dict_ref: dict):
        """
        Main training loop.

        Parameters
        ----------
        sim_run_fn      : callable(ctrl, greedy=False) → None
            Runs one full simulation episode using the given controller.
            Must populate output_dict_ref in-place.
        output_dict_ref : dict
            The shared output_dict used by the simulator.
        """
        cfg = self.cfg
        for seed in range(cfg.initial_seed, cfg.final_seed + 1):
            for episode in range(cfg.n_simulations):
                t0 = time.time()
                ep = self._global_episode

                # ── apply curriculum ──────────────────────────────────────────
                self.curriculum.apply(ep)
                self.ctrl.reset_episode()

                # ── evaluation episode (greedy) ───────────────────────────────
                is_eval = (ep % cfg.eval_every == 0 and ep > 0)
                if is_eval:
                    with GreedyEvaluator(self.ctrl):
                        sim_run_fn(self.ctrl, greedy=True)
                else:
                    sim_run_fn(self.ctrl, greedy=False)

                # ── gated training step ───────────────────────────────────────
                losses = self.curriculum.train_all_gated()

                # ── KPI update ────────────────────────────────────────────────
                self.kpi.update(output_dict_ref, cfg.n_ue, episode % 10)
                j, s, l, pm = self.kpi.latest()

                # ── build episode record ──────────────────────────────────────
                Ws = [b.W_current for b in self.ctrl.bundles]
                Qs = [b.Q_current for b in self.ctrl.bundles]
                eps_vals = [b.routing_agent.eps for b in self.ctrl.bundles]
                total_tx  = sum(b.n_tx_last for b in self.ctrl.bundles)
                total_dr  = sum(b.n_drop_r  for b in self.ctrl.bundles)
                total_dq  = sum(b.n_drop_q  for b in self.ctrl.bundles)
                base_r = (total_tx
                          + 2.0 * float(total_dr == 0)
                          + 2.0 * float(total_dq == 0))

                rec = EpisodeRecord(
                    episode           = ep,
                    phase             = get_current_phase(ep).name,
                    n_ue              = cfg.n_ue,
                    seed              = seed,
                    wall_time_s       = time.time() - t0,
                    jain_index        = j,
                    throughput_mbps   = s,
                    latency_ms        = l,
                    p_mac             = pm,
                    mean_routing_loss = losses.get("routing"),
                    mean_cw_loss      = losses.get("cw"),
                    mean_buf_loss     = losses.get("buf"),
                    mean_epsilon      = float(np.mean(eps_vals)),
                    mean_W            = float(np.mean(Ws)),
                    mean_Q            = float(np.mean(Qs)),
                    std_W             = float(np.std(Ws)),
                    std_Q             = float(np.std(Qs)),
                    mean_reward       = base_r / max(cfg.n_ue, 1),
                    routing_losses    = [],
                    cw_losses         = [],
                    buf_losses        = [],
                )
                self.logger.log(rec)

                # ── checkpoint ───────────────────────────────────────────────
                if ep % cfg.save_every == 0 and ep > 0:
                    self.ctrl.save(cfg.ckpt_dir)
                    print(f"  [Trainer] Checkpoint saved (ep {ep})")

                # ── early stopping ────────────────────────────────────────────
                if self.stopper.step(j, s):
                    print(f"\n[Trainer] Early stopping at episode {ep}  "
                          f"(J={j:.4f}  S={s:.3f} Mbit/s)")
                    self.ctrl.save(cfg.ckpt_dir)
                    self.logger.close()
                    self._save_final_results(output_dict_ref)
                    return

                self._global_episode += 1

        # final save
        self.ctrl.save(cfg.ckpt_dir)
        self.logger.close()
        self._save_final_results(output_dict_ref)
        print("[Trainer] Training complete.")

    def _save_final_results(self, output_dict: dict):
        cfg = self.cfg
        path = os.path.join(cfg.results_dir, "final_averaged_data.json")
        averaged = {}
        for metric, users_data in output_dict.items():
            averaged[metric] = []
            for _, simulations in users_data.items():
                avgs = []
                for _, arr in simulations.items():
                    if metric == 'l':
                        nz = arr[arr != 0]
                        avgs.append(float(np.mean(nz)) if len(nz) else 0.0)
                    else:
                        avgs.append(float(np.mean(arr)))
                averaged[metric].append(float(np.mean(avgs)))
        with open(path, 'w') as f:
            json.dump(averaged, f, indent=2)
        print(f"[Trainer] Final results → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Learning-curve analysis utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_training_log(log_path: str) -> List[dict]:
    """Load a JSON-lines log file into a list of episode dicts."""
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values: List[float], window: int = 5) -> List[float]:
    """Exponential moving average smoothing."""
    out, ema = [], None
    alpha = 2.0 / (window + 1)
    for v in values:
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out


def compute_convergence_episode(records: List[dict],
                                 jain_threshold: float = 0.90,
                                 window: int = 5) -> int:
    """
    Return the first episode after which the smoothed Jain index stays
    above `jain_threshold` for `window` consecutive episodes.
    Returns -1 if never converged.
    """
    jains = [r["jain_index"] for r in records]
    sm    = smooth(jains, window)
    count = 0
    for i, v in enumerate(sm):
        if v >= jain_threshold:
            count += 1
            if count >= window:
                return i - window + 1
        else:
            count = 0
    return -1


def summarise_training(log_path: str, print_output: bool = True) -> dict:
    """Print and return a summary of a completed training run."""
    records = load_training_log(log_path)
    if not records:
        return {}

    jains = [r["jain_index"]      for r in records]
    S_vals = [r["throughput_mbps"] for r in records]
    L_vals = [r["latency_ms"]      for r in records]
    eps    = [r["mean_epsilon"]    for r in records]
    Ws     = [r["mean_W"]          for r in records]
    Qs     = [r["mean_Q"]          for r in records]

    conv_ep = compute_convergence_episode(records, jain_threshold=0.90)
    summary = {
        "n_episodes":          len(records),
        "best_jain":           max(jains),
        "final_jain":          jains[-1],
        "best_throughput":     max(S_vals),
        "final_throughput":    S_vals[-1],
        "min_latency":         min(L for L in L_vals if L > 0) if any(L > 0 for L in L_vals) else 0.0,
        "final_latency":       L_vals[-1],
        "final_epsilon":       eps[-1],
        "final_mean_W":        Ws[-1],
        "final_mean_Q":        Qs[-1],
        "convergence_episode": conv_ep,
        "phases_seen":         list(dict.fromkeys(r["phase"] for r in records)),
    }

    if print_output:
        print("\n" + "═" * 60)
        print("  MADRL Training Summary")
        print("═" * 60)
        print(f"  Episodes         : {summary['n_episodes']}")
        print(f"  Best Jain index  : {summary['best_jain']:.4f}")
        print(f"  Final Jain index : {summary['final_jain']:.4f}")
        print(f"  Best S [Mbit/s]  : {summary['best_throughput']:.3f}")
        print(f"  Final S [Mbit/s] : {summary['final_throughput']:.3f}")
        print(f"  Min latency [ms] : {summary['min_latency']:.3f}")
        print(f"  Final latency[ms]: {summary['final_latency']:.3f}")
        print(f"  Final ε          : {summary['final_epsilon']:.4f}")
        print(f"  Final W / Q      : {summary['final_mean_W']:.1f} / {summary['final_mean_Q']:.1f}")
        print(f"  Convergence ep   : {summary['convergence_episode']}")
        print(f"  Curriculum phases: {' → '.join(summary['phases_seen'])}")
        print("═" * 60 + "\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry-point (standalone smoke-test, no simulator needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MADRL-TB training utilities")
    parser.add_argument("--summarise", type=str, default=None,
                        help="Path to a training_log.jsonl to summarise")
    parser.add_argument("--smoke",    action="store_true",
                        help="Run a quick smoke-test of CurriculumController")
    args = parser.parse_args()

    if args.summarise:
        summarise_training(args.summarise)

    elif args.smoke:
        print("[Smoke test] Instantiating MADRLController with 4 UEs …")
        ctrl = MADRLController(n_ues=4, T_step=5)
        cc   = CurriculumController(ctrl)
        for ep in [0, 10, 30, 80]:
            cc.apply(ep)
            phase = get_current_phase(ep)
            print(f"  ep={ep:3d}  phase={phase.name:<25s}  "
                  f"freeze_r={cc.freeze_routing}  "
                  f"freeze_cw={cc.freeze_cw}  "
                  f"freeze_buf={cc.freeze_buf}")
        print("[Smoke test] Training a random batch …")
        import numpy as np
        for b in ctrl.bundles:
            for _ in range(70):
                s  = np.random.randn(89).astype(np.float32)
                s2 = np.random.randn(89).astype(np.float32)
                b.routing_agent.push(s, 0, 1.0, s2, False)
        cc.apply(30)   # joint phase – routing unfrozen
        losses = cc.train_all_gated()
        print(f"  routing_loss = {losses['routing']}")
        print("[Smoke test] PASSED.")
