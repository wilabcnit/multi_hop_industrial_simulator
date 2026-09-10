"""
evaluate_madrl.py
=================
Evaluation and comparison harness for MADRL-TB vs. baseline TB routing.

Computes for each method across N seeds × N_ue values:
    • Jain fairness index   J       ∈ [1/N, 1]
    • Network throughput    S       [Mbit/s]
    • Average latency       L̄      [ms]
    • MAC success prob.     p_mac   ∈ [0, 1]

Statistical testing
    Bootstrap 95 % confidence intervals (2 000 resamples)
    Welch t-test for pairwise significance

Output
    results/comparison_<tag>.json   – raw + aggregated numbers
    results/comparison_<tag>.txt    – human-readable table
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    """Per-method, per-n_ue aggregated results."""
    method:    str
    n_ue:      int
    # raw arrays: shape (n_seeds * n_sims, n_ue)
    jain:      np.ndarray = field(default_factory=lambda: np.array([]))
    s_mbps:    np.ndarray = field(default_factory=lambda: np.array([]))
    l_ms:      np.ndarray = field(default_factory=lambda: np.array([]))
    p_mac:     np.ndarray = field(default_factory=lambda: np.array([]))

    # aggregated scalars (filled by aggregate())
    jain_mean:  float = 0.0;  jain_ci:  Tuple[float, float] = (0.0, 0.0)
    s_mean:     float = 0.0;  s_ci:     Tuple[float, float] = (0.0, 0.0)
    l_mean:     float = 0.0;  l_ci:     Tuple[float, float] = (0.0, 0.0)
    pm_mean:    float = 0.0;  pm_ci:    Tuple[float, float] = (0.0, 0.0)

    def aggregate(self, n_bootstrap: int = 2_000):
        self.jain_mean, self.jain_ci = _bootstrap_ci(self.jain.mean(axis=-1), n_bootstrap)
        self.s_mean,    self.s_ci    = _bootstrap_ci(self.s_mbps.mean(axis=-1), n_bootstrap)
        non_zero = self.l_ms[self.l_ms > 0]
        self.l_mean, self.l_ci = _bootstrap_ci(non_zero, n_bootstrap)
        self.pm_mean, self.pm_ci = _bootstrap_ci(self.p_mac.mean(axis=-1), n_bootstrap)


def _bootstrap_ci(data: np.ndarray, n: int = 2_000,
                   alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """Return (mean, (ci_low, ci_high))."""
    if len(data) == 0:
        return (0.0, (0.0, 0.0))
    mean = float(np.mean(data))
    if len(data) == 1:
        return (mean, (mean, mean))
    rng  = np.random.default_rng(seed=42)
    boot = rng.choice(data, size=(n, len(data)), replace=True).mean(axis=1)
    lo   = float(np.percentile(boot, 100 * alpha / 2))
    hi   = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (mean, (lo, hi))


def _welch_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Welch t-test, returns (t_statistic, p_value)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        return float(t_stat), float(p_val)
    except ImportError:
        pass
    # fallback: normal approximation
    from math import sqrt
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    p = float(2.0 * (1.0 - min(abs(t) / 3.0, 1.0)))   # crude approximation
    return float(t), p


# ─────────────────────────────────────────────────────────────────────────────
# Output-dict → MethodResult extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_results(output_dict: dict, method: str,
                    n_ue_list: List[int]) -> Dict[int, MethodResult]:
    """
    Convert a simulator output_dict into MethodResult objects.

    output_dict structure:
        output_dict[metric][f"N={n_ue}"][f"Sim={n_sim}"] = np.ndarray(shape=n_ue)
    """
    results: Dict[int, MethodResult] = {}

    for n_ue in n_ue_list:
        tag = f"N={n_ue}"
        if tag not in output_dict.get("j_index", {}):
            continue

        jains, smps, lmss, pmacs = [], [], [], []
        for sim_key, arr in output_dict["j_index"][tag].items():
            jains.append(arr.copy())
        for sim_key, arr in output_dict["s"][tag].items():
            smps.append(arr.copy() * 1e-6)   # bits/s → Mbit/s
        for sim_key, arr in output_dict["l"][tag].items():
            lmss.append(arr.copy() * 1e3)    # s → ms
        for sim_key, arr in output_dict["p_mac"][tag].items():
            pmacs.append(arr.copy())

        mr = MethodResult(
            method = method,
            n_ue   = n_ue,
            jain   = np.array(jains)   if jains  else np.zeros((1, n_ue)),
            s_mbps = np.array(smps)    if smps   else np.zeros((1, n_ue)),
            l_ms   = np.array(lmss)    if lmss   else np.zeros((1, n_ue)),
            p_mac  = np.array(pmacs)   if pmacs  else np.zeros((1, n_ue)),
        )
        mr.aggregate()
        results[n_ue] = mr

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison engine
# ─────────────────────────────────────────────────────────────────────────────

class Comparator:
    """
    Accepts two MethodResult dicts (MADRL and TB) and computes:
        - relative improvement on each KPI
        - Welch t-test p-values
        - formatted comparison table
    """

    METRICS = [
        ("Jain index J",    "jain_mean",  "jain",  "higher"),
        ("Throughput Mbit/s", "s_mean",   "s_mbps", "higher"),
        ("Latency ms",       "l_mean",    "l_ms",   "lower"),
        ("p_mac",            "pm_mean",   "p_mac",  "higher"),
    ]

    def __init__(self,
                 madrl_results: Dict[int, MethodResult],
                 tb_results:    Dict[int, MethodResult]):
        self.madrl = madrl_results
        self.tb    = tb_results

    def compare(self, n_ue: int) -> dict:
        """Return a comparison dict for a specific n_ue value."""
        if n_ue not in self.madrl or n_ue not in self.tb:
            return {}

        m = self.madrl[n_ue]
        t = self.tb[n_ue]
        out = {"n_ue": n_ue, "metrics": {}}

        for label, scalar_key, array_key, direction in self.METRICS:
            mv  = getattr(m, scalar_key)
            tv  = getattr(t, scalar_key)
            rel = (mv - tv) / max(abs(tv), 1e-12) * 100.0   # % change
            if direction == "lower":
                rel = -rel   # negative latency change is good

            ma  = getattr(m, array_key).mean(axis=-1) if hasattr(getattr(m, array_key), 'mean') else np.array([mv])
            ta  = getattr(t, array_key).mean(axis=-1) if hasattr(getattr(t, array_key), 'mean') else np.array([tv])
            stat, pval = _welch_t(ma, ta)

            out["metrics"][label] = {
                "madrl":        mv,
                "madrl_ci":     getattr(m, scalar_key.replace("_mean", "_ci"), (mv, mv)),
                "tb":           tv,
                "tb_ci":        getattr(t, scalar_key.replace("_mean", "_ci"), (tv, tv)),
                "rel_improve_%": rel,
                "t_stat":       stat,
                "p_value":      pval,
                "significant":  pval < 0.05,
            }
        return out

    def compare_all(self) -> Dict[int, dict]:
        return {n: self.compare(n) for n in self.madrl if n in self.tb}

    def format_table(self) -> str:
        lines = []
        SEP   = "-" * 96

        lines.append("|" + "=" * 94 + "|")
        lines.append("|  MADRL-TB vs. TB Routing - KPI Comparison" + " " * 51 + "|")
        lines.append("|" + "=" * 94 + "|")

        hdr = (f"{'N UEs':>6}  {'Metric':<24}  "
               f"{'MADRL':>12}  {'TB':>12}  {'delta%':>8}  "
               f"{'95% CI (M)':>18}  {'p-val':>8}  {'Sig':>4}")
        lines.append("|  " + hdr + "  |")
        lines.append("|  " + SEP + "  |")

        for n_ue, cmp in sorted(self.compare_all().items()):
            for label, data in cmp["metrics"].items():
                mv   = data["madrl"]
                tv   = data["tb"]
                rel  = data["rel_improve_%"]
                pv   = data["p_value"]
                ci   = data["madrl_ci"]
                sig  = "ok" if data["significant"] else " "
                row  = (f"{n_ue:>6}  {label:<24}  "
                        f"{mv:>12.4f}  {tv:>12.4f}  {rel:>+8.2f}  "
                        f"[{ci[0]:.4f},{ci[1]:.4f}]  {pv:>8.4f}  {sig:>4}")
                lines.append("|  " + row + "  |")
            lines.append("|  " + SEP + "  |")

        lines.append("|" + "=" * 94 + "|")
        lines.append("  delta% = (MADRL - TB) / |TB| x 100  "
                      "(positive = MADRL better for all metrics except latency)")
        lines.append("  Sig: Welch t-test, alpha = 0.05")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Jain index computation (independent of simulator)
# ─────────────────────────────────────────────────────────────────────────────

def compute_jain_index(throughputs: np.ndarray) -> float:
    """
    Compute Jain's fairness index from an array of per-UE throughputs.
    J = (Σ S_j)² / (N · Σ S_j²)
    """
    n = len(throughputs)
    if n == 0 or np.all(throughputs == 0):
        return 0.0
    s   = np.sum(throughputs)
    s2  = np.sum(throughputs ** 2)
    return float(s ** 2 / (n * s2)) if s2 > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationPipeline:
    """
    Runs N_EVAL_EPISODES greedy evaluation episodes for each method,
    accumulates output_dicts, and produces a Comparator.

    Usage
    -----
    ::

        pipeline = EvaluationPipeline(n_ue_list=[5, 10, 15], n_eval=20)

        # run MADRL (greedy)
        pipeline.run_method("MADRL", madrl_run_fn, madrl_output_dict)

        # run baseline TB
        pipeline.run_method("TB",    tb_run_fn,    tb_output_dict)

        cmp = pipeline.compare()
        print(cmp.format_table())
        pipeline.save("results/comparison_run1")
    """

    def __init__(self, n_ue_list: List[int], n_eval: int = 20):
        self.n_ue_list = n_ue_list
        self.n_eval    = n_eval
        self._results: Dict[str, Dict[int, MethodResult]] = {}

    def run_method(self, method: str,
                   run_fn,           # callable(n_ue, seed) → output_dict
                   ):
        """
        Calls run_fn(n_ue, seed) for each (n_ue, seed) combination,
        accumulates results.
        """
        per_nue: Dict[int, MethodResult] = {}

        for n_ue in self.n_ue_list:
            jains_acc, s_acc, l_acc, pm_acc = [], [], [], []

            for seed in range(self.n_eval):
                od = run_fn(n_ue=n_ue, seed=seed)
                tag = f"N={n_ue}"
                sim = f"Sim=0"

                j_arr  = od.get("j_index", {}).get(tag, {}).get(sim, np.zeros(n_ue))
                s_arr  = od.get("s",       {}).get(tag, {}).get(sim, np.zeros(n_ue))
                l_arr  = od.get("l",       {}).get(tag, {}).get(sim, np.zeros(n_ue))
                pm_arr = od.get("p_mac",   {}).get(tag, {}).get(sim, np.zeros(n_ue))

                jains_acc.append(j_arr)
                s_acc.append(s_arr * 1e-6)
                l_acc.append(l_arr * 1e3)
                pm_acc.append(pm_arr)

            mr = MethodResult(
                method = method,
                n_ue   = n_ue,
                jain   = np.stack(jains_acc),
                s_mbps = np.stack(s_acc),
                l_ms   = np.stack(l_acc),
                p_mac  = np.stack(pm_acc),
            )
            mr.aggregate()
            per_nue[n_ue] = mr
            print(f"  [{method}] N={n_ue:3d}  "
                  f"J={mr.jain_mean:.4f}  "
                  f"S={mr.s_mean:.3f}Mbit/s  "
                  f"L={mr.l_mean:.2f}ms")

        self._results[method] = per_nue
        return per_nue

    def compare(self) -> Comparator:
        methods = list(self._results.keys())
        if len(methods) < 2:
            raise ValueError("Need at least 2 methods to compare.")
        return Comparator(self._results[methods[0]], self._results[methods[1]])

    def save(self, prefix: str):
        """Save comparison table and raw JSON to files."""
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        cmp = self.compare()

        # text table
        table = cmp.format_table()
        txt_path = prefix + ".txt"
        with open(txt_path, 'w') as f:
            f.write(table)
        print(f"\n[Eval] Table → {txt_path}")
        print(table)

        # JSON
        raw: dict = {}
        for method, per_nue in self._results.items():
            raw[method] = {}
            for n_ue, mr in per_nue.items():
                raw[method][str(n_ue)] = {
                    "jain_mean": mr.jain_mean,
                    "jain_ci":   list(mr.jain_ci),
                    "s_mean":    mr.s_mean,
                    "s_ci":      list(mr.s_ci),
                    "l_mean":    mr.l_mean,
                    "l_ci":      list(mr.l_ci),
                    "pm_mean":   mr.pm_mean,
                    "pm_ci":     list(mr.pm_ci),
                }
        raw["comparison"] = {
            str(n): cmp.compare(n) for n in self._results[list(self._results)[0]]
        }
        json_path = prefix + ".json"
        with open(json_path, 'w') as f:
            json.dump(raw, f, indent=2, default=str)
        print(f"[Eval] JSON  → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# KPI improvement summary (for paper tables)
# ─────────────────────────────────────────────────────────────────────────────

def print_paper_table(comparison_json_path: str):
    """
    Format results from a saved comparison JSON as a LaTeX-ready table row.
    """
    with open(comparison_json_path) as f:
        data = json.load(f)

    methods = [k for k in data if k != "comparison"]
    m1, m2  = methods[0], methods[1]

    print("\n% Auto-generated LaTeX table rows")
    print("% Format: N_UE & J_M1 & J_M2 & ΔJ% & S_M1 & S_M2 & ΔS% & L_M1 & L_M2 & ΔL%")

    for n_ue_str, cmp_data in sorted(data["comparison"].items(),
                                      key=lambda x: int(x[0])):
        metrics = cmp_data.get("metrics", {})
        j  = metrics.get("Jain index J",       {})
        s  = metrics.get("Throughput Mbit/s",  {})
        l  = metrics.get("Latency ms",         {})

        def fmt(d, key="madrl"):
            return f"{d.get(key, 0.0):.4f}" if d else "—"
        def fmtp(d):
            return f"{d.get('rel_improve_%', 0.0):+.2f}"

        print(
            f"  {n_ue_str:>3} & "
            f"{fmt(j,'madrl')} & {fmt(j,'tb')} & {fmtp(j)} & "
            f"{fmt(s,'madrl')} & {fmt(s,'tb')} & {fmtp(s)} & "
            f"{fmt(l,'madrl')} & {fmt(l,'tb')} & {fmtp(l)} \\\\"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MADRL-TB evaluation utilities")
    parser.add_argument("--paper-table", type=str, default=None,
                        help="Path to comparison .json to format as LaTeX rows")
    parser.add_argument("--smoke",    action="store_true",
                        help="Run bootstrap CI smoke test")
    args = parser.parse_args()

    if args.paper_table:
        print_paper_table(args.paper_table)

    elif args.smoke:
        print("[Smoke] Bootstrap CI test …")
        rng  = np.random.default_rng(0)
        a    = rng.normal(0.85, 0.05, 50)
        b    = rng.normal(0.75, 0.08, 50)
        mean_a, ci_a = _bootstrap_ci(a)
        mean_b, ci_b = _bootstrap_ci(b)
        t, p = _welch_t(a, b)
        print(f"  A: mean={mean_a:.4f}  CI={ci_a}")
        print(f"  B: mean={mean_b:.4f}  CI={ci_b}")
        print(f"  Welch t={t:.3f}  p={p:.5f}  sig={'YES' if p<0.05 else 'NO'}")
        j = compute_jain_index(np.array([10.0, 10.0, 10.0, 10.0]))
        assert abs(j - 1.0) < 1e-9, "Jain index failed for equal throughputs"
        j2 = compute_jain_index(np.array([10.0, 0.0, 0.0, 0.0]))
        assert abs(j2 - 0.25) < 1e-9, "Jain index failed for single-UE case"
        print("[Smoke] PASSED.")
