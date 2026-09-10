"""
madrl_simulator_complete.py
===========================
Production-ready simulation loop.  Drop-in replacement for the original
``simulator.py``.  Uses SimContext + run_ue_statemachine / run_bs_statemachine
so the state-machine logic lives entirely in madrl_ue_statemachine.py.

Structural changes vs. original
---------------------------------
[M1]  MADRLController instantiated once per (seed, n_ue); persists across sims.
[M2]  SimContext built once per simulation; passed to every state-machine call.
[M3]  Per-UE W  → madrl_ctrl.W(uid)   replaces global contention_window_int.
[M4]  Per-UE Q  → madrl_ctrl.Q(uid)   replaces global max_n_packets_to_be_forwarded.
[M5]  choose_next_action_tb_no_RL     replaced inside run_ue_statemachine.
[M6]  madrl_episode_end called at end of every simulation run.
[M7]  MADRLTrainer.run() drives the outer loop when USE_TRAINER=True.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
from collections import deque
from copy import deepcopy
from datetime import datetime

import numpy as np
from scipy import constants

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── simulator imports ─────────────────────────────────────────────────────────
from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.network.ue import Ue
from multi_hop_industrial_simulator.utils.check_success import (
    check_collision, check_collision_bs)
from multi_hop_industrial_simulator.utils.compute_distance_m import compute_distance_m
from multi_hop_industrial_simulator.utils.compute_propagation_delays import compute_propagation_delays
from multi_hop_industrial_simulator.utils.compute_simulation_outputs import compute_simulator_outputs
from multi_hop_industrial_simulator.utils.instantiate_bs import instantiate_bs
from multi_hop_industrial_simulator.utils.read_input_file import read_input_file
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs
from multi_hop_industrial_simulator.env.geometry import Geometry
from multi_hop_industrial_simulator.env.distribution import Distribution
from multi_hop_industrial_simulator.env.machine import Machine
from multi_hop_industrial_simulator.utils.instantiate_ues import instantiate_ues
from multi_hop_industrial_simulator.utils.compute_simulator_tick_duration import (
    compute_simulator_tick_duration)
from multi_hop_industrial_simulator.channel_models.THz_channel import THzChannel
from multi_hop_industrial_simulator.utils.check_for_neighbours import check_for_neighbours
from multi_hop_industrial_simulator.utils.set_ues_los_condition import set_ues_los_condition

# ── MADRL imports ─────────────────────────────────────────────────────────────
from madrl_agent import MADRLController
from madrl_integration import madrl_episode_end
from madrl_ue_statemachine import SimContext, run_ue_statemachine, run_bs_statemachine
from train_madrl import MADRLTrainer, TrainingConfig

gc.enable()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

USE_TRAINER   = True   # set False to run the raw loop without curriculum/logging
CKPT_DIR      = "madrl_checkpoints"
LOG_DIR       = "madrl_logs"
RESULTS_DIR   = "madrl_results"

T_STEP_BO     = 10
W_MIN, W_MAX  = 4, 128
Q_MIN, Q_MAX  = 1, 10

AGENT_KWARGS  = dict(
    routing_hidden = (256, 128),
    param_hidden   = (128, 64),
    lr             = 3e-4,
    gamma          = 0.95,
    eps_start      = 1.0,
    eps_end        = 0.05,
    eps_decay      = 0.9995,
    batch          = 64,
    buf_cap        = 20_000,
)

c = constants.speed_of_light

# ─────────────────────────────────────────────────────────────────────────────
# Timing-structure helpers (identical to original)
# ─────────────────────────────────────────────────────────────────────────────

def _ts_create(n_ue: int, dur: int) -> dict:
    s = {}
    for i in range(n_ue):
        v = {"DATA_RX": {}, "ACK_RX": {}}
        for j in range(n_ue):
            if j != i:
                v["DATA_RX"][f"UE_{j}"] = np.array([[dur+1]*4], dtype=int)
                v["ACK_RX"][f"UE_{j}"]  = np.array([[dur+1]*4], dtype=int)
        v["DATA_RX"]["BS"] = np.array([[dur+1]*4], dtype=int)
        v["ACK_RX"]["BS"]  = np.array([[dur+1]*4], dtype=int)
        s[f"UE_{i}"] = v
    bv = {"DATA_RX": {}, "ACK_RX": {}}
    for i in range(n_ue):
        bv["DATA_RX"][f"UE_{i}"] = np.array([[dur+1]*4], dtype=int)
        bv["ACK_RX"][f"UE_{i}"]  = np.array([[dur+1]*4], dtype=int)
    s["BS"] = bv
    return s

def _ts_reset(s: dict, dur: int):
    for kx, vx in s.items():
        for ki in vx["DATA_RX"]: vx["DATA_RX"][ki] = np.array([[dur+1]*4], dtype=int)
        for ki in vx["ACK_RX"]:  vx["ACK_RX"][ki]  = np.array([[dur+1]*4], dtype=int)

def _ts_insert(s, t_s, t_e, f3, f4, tx_k, typ, rx_k):
    row = np.array([t_s, t_e, f3, f4])
    s[rx_k][typ][tx_k] = np.vstack([s[rx_k][typ][tx_k], row])

def _ts_remove(s, tx_k, typ, rx_k):
    s[rx_k][typ][tx_k] = np.delete(s[rx_k][typ][tx_k], 1, axis=0)

def _find_data_ue(s, uid, t):
    st = en = None; sz = []; pids = []; uids = []
    kx = f"UE_{uid}"
    if kx in s:
        for ki, arr in s[kx]["DATA_RX"].items():
            mi = np.argmin(arr[:, 1])
            if arr[mi, 1] == t:
                st = arr[mi, 0]; en = arr[mi, 1]
                sz.append(arr[mi, 2]); pids.append(arr[mi, 3])
                uids.append(int(ki[3:]))
    return st, en, sz, pids, uids

def _find_data_bs(s, t):
    st = en = None; pids = []; uids = []
    for ki, arr in s["BS"]["DATA_RX"].items():
        mi = np.argmin(arr[:, 1])
        if arr[mi, 1] == t:
            st = arr[mi, 0]; en = arr[mi, 1]
            pids.append(arr[mi, 3]); uids.append(int(ki[3:]))
    return st, en, pids, uids

def _find_ack_ue(s, uid, t):
    st = en = None; pids = []; srcs = []; dests = []
    for kx in (f"UE_{uid}", "BS"):
        if kx not in s: continue
        for ki, arr in s[kx]["ACK_RX"].items():
            mi   = np.argmin(arr[:, 1])
            mmin = np.min(arr[:, 1])
            if arr[mi, 1] == t:
                st = arr[mi, 0]; en = arr[mi, 1]
                for idx in range(len(arr)):
                    if arr[idx, 1] == mmin:
                        dests.append(arr[idx, 2])
                        srcs.append(ki)
                        pids.append(arr[idx, 3])
    return st, en, srcs, dests, pids

def _find_ack_bs(s, t):
    st = en = None; dest_ids = []; tx_ids = []; pids = []; n_sim = None
    for ki, arr in s["BS"]["ACK_RX"].items():
        mi   = np.argmin(arr[:, 1])
        mmin = np.min(arr[:, 1])
        if arr[mi, 1] == t:
            st = arr[mi, 0]; en = arr[mi, 1]
            if ki not in tx_ids:
                dest_ids.append(arr[mi, 2]); tx_ids.append(ki)
            n_sim = int(np.count_nonzero(arr[:, 1] == mmin))
            for idx in range(len(arr)):
                if arr[idx, 1] == mmin:
                    pids.append(arr[idx, 3])
    return st, en, dest_ids, pids, tx_ids, n_sim

# ─────────────────────────────────────────────────────────────────────────────
# State-transition helpers referenced by SimContext
# ─────────────────────────────────────────────────────────────────────────────

def _go_idle(ue, t, en_pr):
    ue.set_state("IDLE")
    ue.set_state_duration(ue.get_next_packet_generation_instant())
    ue.energy_consumed += _ctx_ref.power_idle * (ue.get_state_duration() - t) * _ctx_ref.simulator_tick_duration_s
    if en_pr: print(f"UE {ue.get_ue_id()} → IDLE  until {ue.get_state_duration()}")

def _go_backoff(ue, t, bo_dur, en_pr):
    if ue.get_state() == "WAIT_ACK": ue.ticks_in_WAIT_ACK.append(t - ue.get_state_starting_tick())
    elif ue.get_state() == "TX_ACK": ue.ticks_in_TX_ACK.append(t - ue.get_state_starting_tick())
    ue.set_state("BO")
    ue.update_state_duration(bo_dur)
    ue.set_state_starting_tick(t)
    ue.set_state_final_tick(ue.get_state_duration())
    ue.energy_consumed += _ctx_ref.power_bo * (ue.get_state_duration() - t) * _ctx_ref.simulator_tick_duration_s
    if en_pr: print(f"UE {ue.get_ue_id()} → BO  [{t}, {ue.get_state_duration()}]")

def _go_tx_data(ue, t, en_pr):
    if ue.get_state() == "BO":     ue.ticks_in_BO.append(t - ue.get_state_starting_tick())
    elif ue.get_state() == "TX_ACK": ue.ticks_in_TX_ACK.append(t - ue.get_state_starting_tick())
    data_dur = 0; data_sz = 0
    pkt_list = ue.get_updated_packet_list()
    ue.buffer_packet_sent.clear(); ue.end_data_tx = t
    has_fwd = False
    for pkt in pkt_list:
        if pkt.get_data_to_be_forwarded_bool():
            has_fwd = True
        if not pkt.get_retransmission_packets():
            pkt.hop_count += 1; ue.packets_sent += 1
        pkt.set_data_unicast(False)
        data_dur += pkt.get_packet_duration_tick()
        data_sz  += pkt.get_size()
        ue.buffer_packet_sent.append(pkt)
    ue.set_relay_bool(relay_bool=has_fwd)
    ue.end_data_tx += data_dur
    ue.set_state("TX_DATA")
    ue.update_state_duration(data_dur)
    ue.set_state_starting_tick(t)
    ue.set_state_final_tick(ue.get_state_duration())
    ue.energy_consumed += _ctx_ref.power_tx * (ue.get_state_duration() - t) * _ctx_ref.simulator_tick_duration_s
    if en_pr: print(f"UE {ue.get_ue_id()} → TX_DATA  [{t}, {ue.get_state_duration()}]")
    return data_sz

def _go_tx_ack(ue, t, ack_dur, en_pr):
    if ue.get_state() == "BO":     ue.ticks_in_BO.append(t - ue.get_state_starting_tick())
    elif ue.get_state() == "WAIT_ACK": ue.ticks_in_WAIT_ACK.append(t - ue.get_state_starting_tick())
    ue.set_state("TX_ACK")
    ue.update_state_duration(ack_dur)
    ue.set_state_starting_tick(t)
    ue.set_state_final_tick(ue.get_state_duration())
    ue.energy_consumed += _ctx_ref.power_tx * (ue.get_state_duration() - t) * _ctx_ref.simulator_tick_duration_s
    if en_pr: print(f"UE {ue.get_ue_id()} → TX_ACK  [{t}, {ue.get_state_duration()}]")

def _go_tx_ack_bs(bs, t, ack_dur, en_pr):
    bs.set_state("TX_ACK")
    bs.update_state_duration(ack_dur)
    bs.set_start_tx_ack(t)
    bs.set_end_tx_ack(bs.get_state_duration())
    if en_pr: print(f"BS → TX_ACK  [{t}, {bs.get_state_duration()}]")

def _go_wait_ack(ue, t, wad, en_pr=True):
    if ue.get_state() == "TX_DATA": ue.ticks_in_TX_DATA.append(t - ue.get_state_starting_tick())
    elif ue.get_state() == "TX_ACK": ue.ticks_in_TX_ACK.append(t - ue.get_state_starting_tick())
    ue.set_state("WAIT_ACK")
    ue.update_state_duration(wad)
    ue.set_state_starting_tick(t)
    ue.set_state_final_tick(ue.get_state_duration())
    ue.energy_consumed += _ctx_ref.power_ack * (ue.get_state_duration() - t) * _ctx_ref.simulator_tick_duration_s
    if en_pr: print(f"UE {ue.get_ue_id()} → WAIT_ACK  [{t}, {ue.get_state_duration()}]")

def _go_rx_bs(bs, t, rx_dur, en_pr=True):
    bs.set_state("RX")
    bs.set_state_duration(rx_dur)

# module-level reference; set once per simulation in build_sim_context()
_ctx_ref: SimContext = None   # noqa

def _build_sim_context(
        madrl_ctrl, ue_array, bs, s_timing,
        t_ack_tick, max_prop_delay_tick, hop_limit,
        star_topology, sinr_th_db, enable_print,
        sim_tick_s, shad_idx, shad_next, shad_coh,
        thz_ch, f_ghz, bw_hz, fading, clutter, ant_model,
        use_meas, avg_clutter_h, nf_ue, nf_bs,
        pwr_bo, pwr_idle, pwr_tx, pwr_ack,
        tot_dur, TTL, max_Q,
        bo_cnt, ack_rx, pkts_bs, T_step,
        ues_ibs, ues_cbs,
) -> SimContext:
    global _ctx_ref
    ctx = SimContext(
        madrl_ctrl          = madrl_ctrl,
        ue_array            = ue_array,
        bs                  = bs,
        simulator_timing_structure = s_timing,
        t_ack_tick          = t_ack_tick,
        t_backoff_tick      = t_ack_tick,
        max_prop_delay_tick = max_prop_delay_tick,
        hop_limit           = hop_limit,
        star_topology       = star_topology,
        sinr_th_db          = sinr_th_db,
        enable_print        = enable_print,
        simulator_tick_duration_s = sim_tick_s,
        shadowing_sample_index    = shad_idx,
        shadowing_next_tick       = shad_next,
        shadowing_coherence_time_tick_duration = shad_coh,
        thz_channel         = thz_ch,
        carrier_frequency_ghz = f_ghz,
        bandwidth_hz        = bw_hz,
        apply_fading        = fading,
        clutter_density     = clutter,
        antenna_gain_model  = ant_model,
        use_channel_measurements = use_meas,
        average_machine_height_m = avg_clutter_h,
        noise_figure_ue     = nf_ue,
        noise_figure_bs     = nf_bs,
        power_bo            = pwr_bo,
        power_idle          = pwr_idle,
        power_tx            = pwr_tx,
        power_ack           = pwr_ack,
        tot_simulation_time_tick = tot_dur,
        TTL                 = TTL,
        max_n_packets_to_be_forwarded = max_Q,
        bo_phase_count      = bo_cnt,
        ack_received        = ack_rx,
        packets_at_bs       = pkts_bs,
        T_STEP_BO           = T_step,
        ues_interfering_at_bs = ues_ibs,
        ues_colliding_at_bs   = ues_cbs,
        # bind timing helpers
        _insert_fn          = _ts_insert,
        _remove_fn          = _ts_remove,
        _find_data_ue_fn    = _find_data_ue,
        _find_data_bs_fn    = _find_data_bs,
        _find_ack_ue_fn     = _find_ack_ue,
        _find_ack_bs_fn     = _find_ack_bs,
        _check_collision_fn   = check_collision,
        _check_collision_bs_fn = check_collision_bs,
        _compute_distance_fn  = compute_distance_m,
        _go_in_backoff_fn   = _go_backoff,
        _go_in_idle_fn      = _go_idle,
        _go_in_tx_data_fn   = _go_tx_data,
        _go_in_tx_ack_fn    = _go_tx_ack,
        _go_in_tx_ack_bs_fn = _go_tx_ack_bs,
        _go_in_wait_ack_fn  = _go_wait_ack,
        _go_rx_ack_bs_fn    = _go_rx_bs,
    )
    _ctx_ref = ctx
    return ctx

# ─────────────────────────────────────────────────────────────────────────────
# Timing-structure pre-update  (outer loop, before state machines)
# ─────────────────────────────────────────────────────────────────────────────

def _update_timing_for_ue(ue, t: int, s: dict, tot_dur: int):
    """
    Shrink BO or WAIT_ACK duration when a reception arrives early.
    Identical to the original simulator's per-UE timing update.
    """
    uid  = ue.get_ue_id()
    kx   = f"UE_{uid}"
    if kx not in s:
        return

    new_bo_tick       = tot_dur + 1
    new_bo_ue_id      = None
    new_wack_tick     = tot_dur + 1
    new_wack_ue_ids   = []

    # ── BO shrink (data RX) ───────────────────────────────────────────────────
    for ki, arr in s[kx]["DATA_RX"].items():
        mrx = np.min(arr[:, 1])
        if mrx == t:
            if ue.get_state() == "BO":
                new_bo_tick = min(new_bo_tick, mrx)
            new_bo_ue_id = ki

    # ── WAIT_ACK shrink (data OR ack RX) ─────────────────────────────────────
    min1 = min(np.min(a[:, 1]) for a in s[kx]["DATA_RX"].values())
    min2 = min(np.min(a[:, 1]) for a in s[kx]["ACK_RX"].values())
    min_rx = min(min1, min2)
    if min_rx == t:
        if ue.get_state() == "WAIT_ACK":
            new_wack_tick = min(new_wack_tick, min_rx)
            ue.data_rx_during_wait_ack = (new_wack_tick == min1)
            ue.ack_rx_during_wait_ack  = (new_wack_tick == min2)
        for ki, arr in s[kx]["ACK_RX"].items():
            if min_rx in arr[:, 1]:
                new_wack_ue_ids.append((ki, "ack"))
        for ki, arr in s[kx]["DATA_RX"].items():
            if min_rx in arr[:, 1]:
                new_wack_ue_ids.append((ki, "data"))

    # ── apply BO shrink ───────────────────────────────────────────────────────
    if (ue.get_state_starting_tick() < new_bo_tick <= ue.get_state_final_tick()
            and ue.get_state() == "BO"):
        ue.set_state_duration(new_bo_tick)
        if ue.get_state_duration() == ue.get_state_final_tick():
            ue.set_reception_during_bo_bool(True)
    elif new_bo_ue_id and not ue.data_rx_during_wait_ack:
        _ts_remove(s, new_bo_ue_id, "DATA_RX", kx)

    # ── apply WAIT_ACK shrink ─────────────────────────────────────────────────
    if (ue.get_state_starting_tick() < new_wack_tick <= ue.get_state_final_tick()
            and ue.get_state() == "WAIT_ACK"):
        ue.set_state_duration(new_wack_tick)
        if ue.get_state_duration() == ue.get_state_final_tick():
            ue.set_reception_during_wait_bool(True)
    elif new_wack_ue_ids:
        for uid2, typ in new_wack_ue_ids:
            if typ == "ack":
                _ts_remove(s, uid2, "ACK_RX", kx)
            if typ == "data" and ue.data_rx_during_wait_ack:
                _ts_remove(s, uid2, "DATA_RX", kx)
        ue.data_rx_during_wait_ack = False


def _update_timing_for_bs(bs, t: int, s: dict, tot_dur: int):
    """Shrink BS RX/IDLE duration when an event arrives early."""
    new_rx = tot_dur + 1
    _listening = ("IDLE", "RX")
    for ki, arr in s["BS"]["DATA_RX"].items():
        if np.min(arr[:, 1]) == t and bs.get_state() in _listening:
            new_rx = min(np.min(arr[:, 1]), new_rx)
    for ki, arr in s["BS"]["ACK_RX"].items():
        if np.min(arr[:, 1]) == t and bs.get_state() in _listening:
            new_rx = min(np.min(arr[:, 1]), new_rx)
    if (bs.get_state_starting_tick() < new_rx <= bs.get_state_final_tick()
            and bs.get_state() in _listening):
        bs.set_state_duration(new_rx)

# ─────────────────────────────────────────────────────────────────────────────
# Mobility helpers (identical to original, self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_mobility(t, t_change, ue_array, machine_array, bs, simulator_timing_structure,
                    ue_coordinates_list, copy_ue_coordinates_dict, tot_dur,
                    simulator_tick_duration_s, mobility_obstacle, mobility_spawn,
                    mobility_shuffle, step_size, enable_print, next_t_change):
    """Apply one mobility event and return updated machine_array."""
    from multi_hop_industrial_simulator.env.machine import Machine

    copy_s = deepcopy(simulator_timing_structure)

    if mobility_obstacle:
        pilot = 1.25; pilot_max = 18.75
        xs = [m.x_center for m in machine_array]
        ys = [m.y_center for m in machine_array]
        mn_x, mx_x = min(xs), max(xs)
        mn_y, mx_y = min(ys), max(ys)
        for m in machine_array:
            nx, ny = m.move_machine(m.x_center, m.y_center, step_size,
                                    mn_x, mx_x, mn_y, mx_y)
            m.set_coordinates(nx, ny, m.z_center)
            m.__class__ = Machine  # re-init cached properties

    elif mobility_spawn:
        half = 0.5 * tot_dur
        if t_change < half:
            coords = [(3.25, None), (None, 7.75), (16.75, None)]
        elif t_change < tot_dur:
            coords = [(16.75, None), (None, 12.25), (3.25, None)]
        else:
            coords = [(22, None), (None, 22), (22, None)]
        for i, (cx, cy) in zip([8, 9, 10], coords):
            m = machine_array[i]
            nx = cx if cx is not None else m.x_center
            ny = cy if cy is not None else m.y_center
            m.set_coordinates(nx, ny, m.z_center)

    elif mobility_shuffle:
        # subtract old propagation delays
        compute_propagation_delays(ue_array, bs, simulator_tick_duration_s)
        for ue in ue_array:
            kx = f"UE_{ue.get_ue_id()}"
            for typ in ("DATA_RX", "ACK_RX"):
                for ki, arr in simulator_timing_structure[kx][typ].items():
                    for row_i in range(len(arr)):
                        if arr[row_i, 0] == tot_dur + 1:
                            continue
                        if ki.startswith("UE_"):
                            other_id = int(ki[3:])
                            pd = ue.get_prop_delay_to_ue_tick(other_id)
                        else:
                            pd = ue.get_prop_delay_to_bs_tick()
                        arr[row_i, 0] -= pd
                        arr[row_i, 1] -= pd
        for kx_bs, typ in [("BS", "DATA_RX"), ("BS", "ACK_RX")]:
            for ki, arr in simulator_timing_structure["BS"][typ].items():
                ue_id = int(ki[3:])
                pd = ue_array[ue_id].get_prop_delay_to_bs_tick()
                for row_i in range(len(arr)):
                    if arr[row_i, 0] != tot_dur + 1:
                        arr[row_i, 0] -= pd
                        arr[row_i, 1] -= pd
        # shuffle coordinates
        random.shuffle(ue_coordinates_list)
        for idx, ue in enumerate(ue_array):
            ue.set_coordinates(*ue_coordinates_list[idx])
            copy_ue_coordinates_dict[ue.get_ue_id()].append(
                ue.get_coordinates().tolist())
        # add new propagation delays
        compute_propagation_delays(ue_array, bs, simulator_tick_duration_s)
        for ue in ue_array:
            kx = f"UE_{ue.get_ue_id()}"
            for typ in ("DATA_RX", "ACK_RX"):
                for ki, arr in simulator_timing_structure[kx][typ].items():
                    for row_i in range(len(arr)):
                        if arr[row_i, 0] == tot_dur + 1:
                            continue
                        if ki.startswith("UE_"):
                            other_id = int(ki[3:])
                            pd = ue.get_prop_delay_to_ue_tick(other_id)
                        else:
                            pd = ue.get_prop_delay_to_bs_tick()
                        arr[row_i, 0] += pd
                        arr[row_i, 1] += pd
                        if arr[row_i, 0] < next_t_change:
                            _ts_remove(copy_s, ki, typ, kx)
        for typ in ("DATA_RX", "ACK_RX"):
            for ki, arr in simulator_timing_structure["BS"][typ].items():
                ue_id = int(ki[3:])
                pd = ue_array[ue_id].get_prop_delay_to_bs_tick()
                for row_i in range(len(arr)):
                    if arr[row_i, 0] != tot_dur + 1:
                        arr[row_i, 0] += pd
                        arr[row_i, 1] += pd
                        if arr[row_i, 0] < next_t_change:
                            _ts_remove(copy_s, ki, typ, "BS")
        simulator_timing_structure.update(deepcopy(copy_s))

    # update LoS/NLoS
    for i, ue in enumerate(ue_array):
        ue.is_in_los_ues.clear()
        ue.is_low_channel_condition_with_ues.clear()   # must also clear; appended by set_ues_los_condition
        ue.is_in_los = set_ues_los_condition(ue, bs, machine_array, "ue_bs")
    for j, ue in enumerate(ue_array):
        for i, other in enumerate(ue_array):
            ue.is_in_los_ues.append(
                set_ues_los_condition(ue, other, machine_array, "ue_ue"))

    return machine_array

# ─────────────────────────────────────────────────────────────────────────────
# UE reset helper
# ─────────────────────────────────────────────────────────────────────────────

def _reset_ue(ue, bs, ue_array, t_idle, power_idle, sim_tick_s):
    ue.set_n_data_tx(0); ue.set_n_data_rx(0)
    ue.set_state("IDLE")
    ue.set_t_generation(0)
    ue.set_state_duration(t_idle)
    ue.set_state_starting_tick(0)
    ue.set_state_final_tick(t_idle)
    ue.energy_consumed = power_idle * t_idle * sim_tick_s
    ue.set_packet_id(0)
    bs.set_n_data_rx_from_ues(ue.get_ue_id(), 0)
    bs.packet_id_received[ue.get_ue_id()]      = []
    bs.temp_packet_id_received[ue.get_ue_id()] = []
    ue.set_retransmission_packets(False); ue.set_relay_bool(False)
    ue.reset_temp_obs(); ue.set_old_state(None); ue.reset_obs()
    ue.set_packets_sent(0); ue.set_reward([])
    ue.set_last_action(None)
    ue.set_unicast_rx_address(None); ue.set_unicast_rx_index(None)
    ue.set_broadcast_bool(False)
    ue.new_action_bool = True; ue.first_entry = False
    ue.action_packet_id = None; ue.forward_in_wait_ack = False
    ue.check_last_round = False
    ue.set_action_list([]); ue.set_success_action_list([])
    ue.set_reception_during_bo_bool(False)
    ue.set_reception_during_wait_bool(False)
    ue.set_ul_buffer()
    ue.n_tear = 0
    ue.ack_rx_during_wait_ack = False; ue.data_rx_during_wait_ack = False
    ue.list_data_rx_during_wait_ack = []; ue.list_data_generated_during_wait_ack = []
    ue.list_data_rx_from_ue_id = []
    ue.dict_data_rx_during_wait_ack = {}; ue.dict_data_rx_during_bo = {}
    ue.reception_ack_during_wait = False
    ue.list_ack_sent_from_bs = []; ue.dict_ack_sent_from_ue = {}
    ue.buffer_packet_sent = []; ue.ues_colliding_at_ue = []
    ue.data_rx_at_ue_ue_id_list = []; ue.latency_ue = []
    ue.n_generated_packets = 0
    ue.forward_in_bo = False; ue.packet_forward = False
    ue.forward_in_ack = False; ue.multihop_bool = True; ue.end_data_tx = 0
    for other in ue_array:
        if other != ue:
            ue.packet_id_received[other.get_ue_id()]      = []
            ue.dict_data_rx_during_wait_ack[other.get_ue_id()] = []
            ue.dict_data_rx_during_bo[other.get_ue_id()]       = []
            ue.dict_ack_sent_from_ue[other.get_ue_id()]        = []
    # MADRL extras
    ue.ues_interfering_at_ue = []
    ue.n_interfering          = []
    ue.n_forwarding           = 0

# ─────────────────────────────────────────────────────────────────────────────
# Core simulation function (one episode)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_episode(
        n_ue:         int,
        n_simulation: int,
        madrl_ctrl:   MADRLController,
        ue_array,
        bs,
        machine_array,
        distribution_class,
        scenario_df,
        geometry_class,
        thz_channel,
        inputs:       dict,
        output_dict:  dict,
        # pre-computed constants
        simulator_tick_duration_s: float,
        tot_simulation_time_tick:  int,
        t_ack_tick:    int,
        max_prop_delay_tick: int,
        shadowing_coherence_time_tick_duration: float,
        n_shadowing_samples: int,
        # scalar inputs
        sinr_th_db, carrier_frequency_ghz, bandwidth_hz,
        apply_fading, clutter_density, antenna_gain_model,
        use_channel_measurements, average_machine_height_m,
        noise_figure_ue, noise_figure_bs,
        power_bo, power_idle, power_tx, power_ack,
        TTL, max_n_packets_to_be_forwarded, hop_limit,
        star_topology, enable_print,
        mobility_obstacle, mobility_spawn, mobility_shuffle,
        mobility_changes, step_size,
        ue_coordinates_list,
        copy_ue_coordinates_dict,
):
    """Run one complete simulation episode and update output_dict in-place."""

    # ── reset BS ──────────────────────────────────────────────────────────────
    bs.set_n_data_rx(0); bs.set_n_data_rx_rt(0)
    bs.set_n_data_rx_cn(0); bs.set_n_data_rx_nrt(0); bs.set_n_data_rx_fq(0)
    bs.set_state(inputs["bs"]["bs_starting_state"])
    bs.set_state_duration(tot_simulation_time_tick + 1)
    bs.set_state_starting_tick(0)
    bs.set_state_final_tick(tot_simulation_time_tick + 1)
    bs.packet_rx = False; bs.end_of_rx_for_ack_tx = None
    bs.sequence_number_of_packet_rx = 0; bs.rx_data = False; bs.id_ues_data_rx = []

    # ── reset UEs ─────────────────────────────────────────────────────────────
    for ue in ue_array:
        _reset_ue(ue, bs, ue_array, t_ack_tick, power_idle, simulator_tick_duration_s)
        ue_coordinates_list.append(ue.starting_coordinates)
        copy_ue_coordinates_dict[ue.get_ue_id()].append(ue.starting_coordinates.tolist())

    # ── MADRL episode init  [M1] ──────────────────────────────────────────────
    madrl_ctrl.reset_episode()
    bo_phase_count = {ue.get_ue_id(): 0 for ue in ue_array}
    ack_received   = {ue.get_ue_id(): False for ue in ue_array}
    packets_at_bs  = {ue.get_ue_id(): False for ue in ue_array}

    # ── neighbour tables ──────────────────────────────────────────────────────
    nodes_list = [str(ue.get_ue_id()) for ue in ue_array] + ["BS"]
    for idx, ue in enumerate(ue_array):
        ue.set_replay_buffer(deque(maxlen=inputs["rl"]["agent"]["max_len_replay_buffer"]))
        ue.set_neighbour_table(nodes_list[:idx] + nodes_list[idx+1:])
        ue.reset_obs(); ue.reset_temp_obs()
        ue.set_last_action(None); ue.set_broadcast_bool(False)
        ue.new_action_bool = True; ue.next_action = None
        ue.first_bo_entry = True; ue.first_entry = False
        ue.copy_buffer_packet_list = None; ue.action_packet_id = None
        ue.designated_rx = False
        ue.set_actions_per_simulation([[], [], [], []])
        ue.set_success_actions_per_simulation([[], []])
        ue.saved_coordinates = ue.get_coordinates()
        keys = nodes_list[:idx] + nodes_list[idx+1:]
        ue.packets_to_be_removed = {k: [] for k in keys}
        ue.forward_in_bo = False; ue.packet_forward = False
        ue.forward_in_ack = False; ue.multihop_bool = True

    # ── timing structure ──────────────────────────────────────────────────────
    machine_array = distribution_class.distribute_machines(scenario_df=scenario_df)
    s_timing = _ts_create(n_ue, tot_simulation_time_tick)
    _ts_reset(s_timing, tot_simulation_time_tick)

    t = 0
    shad_idx  = 0
    shad_next = t + shadowing_coherence_time_tick_duration
    ues_ibs   = []; ues_cbs = []
    t_change  = 0; next_t_change = 0

    # ── build SimContext  [M2] ────────────────────────────────────────────────
    ctx = _build_sim_context(
        madrl_ctrl, ue_array, bs, s_timing,
        t_ack_tick, max_prop_delay_tick, hop_limit,
        star_topology, sinr_th_db, enable_print,
        simulator_tick_duration_s, shad_idx, shad_next,
        shadowing_coherence_time_tick_duration,
        thz_channel, carrier_frequency_ghz, bandwidth_hz,
        apply_fading, clutter_density, antenna_gain_model,
        use_channel_measurements, average_machine_height_m,
        noise_figure_ue, noise_figure_bs,
        power_bo, power_idle, power_tx, power_ack,
        tot_simulation_time_tick, TTL, max_n_packets_to_be_forwarded,
        bo_phase_count, ack_received, packets_at_bs, T_STEP_BO,
        ues_ibs, ues_cbs,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Main tick loop
    # ══════════════════════════════════════════════════════════════════════════

    while t <= tot_simulation_time_tick:

        # ── mobility ──────────────────────────────────────────────────────────
        if t == next_t_change and t > 0:
            t_change = t
            if any([mobility_obstacle, mobility_spawn, mobility_shuffle]):
                machine_array = _apply_mobility(
                    t, t_change, ue_array, machine_array, bs, s_timing,
                    ue_coordinates_list, copy_ue_coordinates_dict,
                    tot_simulation_time_tick, simulator_tick_duration_s,
                    mobility_obstacle, mobility_spawn, mobility_shuffle,
                    step_size, enable_print, next_t_change)

        # ── per-tick resets ───────────────────────────────────────────────────
        for ue in ue_array:
            ue.data_rx_during_wait_ack = False
            ue.ack_rx_during_wait_ack  = False
            ue.set_ue_saved_state(ue.get_state())
            _update_timing_for_ue(ue, t, s_timing, tot_simulation_time_tick)

        _update_timing_for_bs(bs, t, s_timing, tot_simulation_time_tick)

        # ── packet generation ─────────────────────────────────────────────────
        for ue in ue_array:
            if t == ue.get_next_packet_generation_instant():
                Q_ue = madrl_ctrl.Q(ue.get_ue_id())   # [M4]
                if len(ue.ul_buffer.buffer_packet_list) < Q_ue + 1:
                    ue.add_new_packet(t, enable_print)
                    ue.packet_generation_instant = t

        # ── UE state machines  [M5] ───────────────────────────────────────────
        for ue in ue_array:
            if t == ue.get_state_duration():
                run_ue_statemachine(ue, t, ctx)

        # ── BS state machine ──────────────────────────────────────────────────
        if t == bs.get_state_duration():
            run_bs_statemachine(bs, t, ctx)

        # ── time advance ──────────────────────────────────────────────────────
        if mobility_obstacle or mobility_spawn:
            next_t_change = t_change + math.floor(tot_simulation_time_tick /
                                                   (2 if step_size == 4.5 else 3))
        elif mobility_shuffle:
            next_t_change = t_change + math.floor(
                tot_simulation_time_tick / (mobility_changes + 1))
        else:
            next_t_change = t_change + tot_simulation_time_tick + 1

        min_rx = np.inf
        for kx, vx in s_timing.items():
            for ki, arr in vx["DATA_RX"].items():
                min_rx = min(min_rx, arr[np.argmin(arr[:, 1]), 1])
            for ki, arr in vx["ACK_RX"].items():
                min_rx = min(min_rx, arr[np.argmin(arr[:, 1]), 1])

        t = int(min(
            bs.get_state_duration(),
            min(ue.get_state_duration() for ue in ue_array),
            min(ue.get_next_packet_generation_instant() for ue in ue_array),
            min_rx, next_t_change))

    # ── episode end  [M6] ────────────────────────────────────────────────────
    for ue in ue_array:
        madrl_episode_end(ue, madrl_ctrl.bundle(ue.get_ue_id()), hop_limit)

    compute_simulator_outputs(
        ue_array, bs, inputs["simulation"]["tot_simulation_time_s"],
        inputs, output_dict, n_ue, n_simulation)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inputs = read_inputs("inputs.yaml")

    # ── scalar constants ──────────────────────────────────────────────────────
    initial_seed   = inputs["simulation"]["initial_seed"]
    final_seed     = inputs["simulation"]["final_seed"]
    init_n_ue      = inputs["simulation"]["initial_number_of_ues"]
    step_n_ue      = inputs["simulation"]["step_number_of_ues"]
    final_n_ue     = inputs["simulation"]["final_number_of_ues"]
    n_simulations  = inputs["simulation"]["n_simulations"]
    sim_time_s     = inputs["simulation"]["tot_simulation_time_s"]
    enable_print   = inputs["simulation"]["enable_print"]
    star_topology  = inputs["simulation"]["star_topology"]
    ue_dist_type   = inputs["ue"]["ue_spatial_distribution"]
    bs_start_state = inputs["bs"]["bs_starting_state"]
    ue_start_state = inputs["ue"]["ue_starting_state"]
    ack_sz_bytes   = inputs["aloha_protocol"]["ack_size_bytes"]
    max_n_pkt_fwd  = inputs["aloha_protocol"]["max_n_packets_to_be_forwarded"]
    max_retx       = inputs["aloha_protocol"]["max_n_retx_per_packet"]
    bit_rate       = inputs["radio"]["bit_rate_gbits"]
    apply_fading   = inputs["channel"]["apply_fading"]
    sinr_th_db     = inputs["channel"]["sinr_th_db"]
    f_ghz          = inputs["radio"]["carrier_frequency_ghz"]
    bw_ghz         = inputs["radio"]["bandwidth_ghz"]
    bw_hz          = bw_ghz * 1e9
    ant_model      = inputs["channel"]["antenna_gain_model"]
    use_meas       = inputs["channel"]["use_channel_measurements"]
    shad_coh_ms    = inputs["channel"]["shadowing_coherence_time_ms"]
    shad_coh_s     = shad_coh_ms / 1e3
    nf_ue_db       = inputs["ue"]["ue_noise_figure_db"]
    nf_bs_db       = inputs["bs"]["bs_noise_figure_db"]
    nf_ue          = 10 ** (nf_ue_db / 10)
    nf_bs          = 10 ** (nf_bs_db / 10)
    pwr_bo         = inputs["power_consumed_ue"]["backoff"]
    pwr_idle       = inputs["power_consumed_ue"]["idle"]
    pwr_tx         = inputs["power_consumed_ue"]["tx_data"]
    pwr_ack        = inputs["power_consumed_ue"]["wait_ack"]
    TTL            = inputs["rl"]["router"]["TTL"]
    hop_limit      = inputs["aloha_protocol"]["hop_limit"]
    mob_obs        = inputs["simulation"]["mobility_obstacle"]
    mob_spn        = inputs["simulation"]["mobility_spawn"]
    mob_shf        = inputs["simulation"]["mobility_shuffle"]
    mob_chg        = inputs["simulation"]["mobility_changes"]
    step_sz        = inputs["simulation"]["mobility_step_size"]
    payload_fq     = inputs["traffic_fq"]["payload"]

    if inputs["scenario"]["name"] == "grid":
        sheet = inputs["scenario"]["input_sheet_names"]["grid"]
        scenario_df = read_input_file(inputs["scenario"]["input_file_name"], sheet, True)
    else:
        sys.exit("Unsupported scenario")

    geometry_class     = Geometry(scenario_df=scenario_df)
    distribution_class = Distribution(ue_dist_type, inputs["scenario"]["name"],
                                       scenario_df, init_n_ue)
    machine_array      = distribution_class.distribute_machines(scenario_df=scenario_df)

    machine_area = sum(m.get_machine_size()**2 for m in machine_array)
    avg_mach_h   = sum(m.get_machine_height() for m in machine_array) / max(len(machine_array),1)
    area_tot     = geometry_class.get_factory_length() * geometry_class.get_factory_width()
    clutter      = machine_area / area_tot

    sim_tick_s   = compute_simulator_tick_duration(inputs)
    bs           = instantiate_bs(inputs, sim_tick_s, bs_start_state, bit_rate)
    distribution_class.distribute_bs(bs=bs)
    thz_channel  = THzChannel(params=inputs)
    tot_dur      = math.ceil(sim_time_s / sim_tick_s)
    n_shad       = math.ceil(sim_time_s / shad_coh_s)
    thz_channel.set_shadowing_sample_db(n_shad)
    shad_coh_tck = shad_coh_s / sim_tick_s

    t_ack_ns    = round((ack_sz_bytes * 8 * 1e-9) / bs.get_bit_rate_gbits(), 11)
    t_ack_tick  = round(t_ack_ns / sim_tick_s)

    # ── output dict init ──────────────────────────────────────────────────────
    output_keys = ["p_mac","s_ue","s","l","e","j_index"]
    output_dict: dict = {}
    for ok in output_keys:
        output_dict[ok] = {}
        for n_ue in range(init_n_ue, final_n_ue+1, step_n_ue):
            output_dict[ok][f"N={n_ue}"] = {}
            for ns in range(n_simulations):
                output_dict[ok][f"N={n_ue}"][f"Sim={ns}"] = np.zeros(n_ue)

    # ── optional trainer  [M7] ───────────────────────────────────────────────
    if USE_TRAINER:
        trainer_cfg = TrainingConfig(
            n_ue          = init_n_ue,
            n_simulations = n_simulations,
            initial_seed  = initial_seed,
            final_seed    = final_seed,
            T_step        = T_STEP_BO,
            W_min=W_MIN, W_max=W_MAX, Q_min=Q_MIN, Q_max=Q_MAX,
            share_routing = True,
            agent_lr      = AGENT_KWARGS["lr"],
            agent_gamma   = AGENT_KWARGS["gamma"],
            agent_eps_start = AGENT_KWARGS["eps_start"],
            agent_eps_end   = AGENT_KWARGS["eps_end"],
            agent_eps_decay = AGENT_KWARGS["eps_decay"],
            agent_batch     = AGENT_KWARGS["batch"],
            agent_buf_cap   = AGENT_KWARGS["buf_cap"],
            ckpt_dir  = CKPT_DIR,
            log_dir   = LOG_DIR,
            results_dir = RESULTS_DIR,
        )

    # ── main loops ────────────────────────────────────────────────────────────
    for seed in range(initial_seed, final_seed+1):
        random.seed(seed); np.random.seed(seed)
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

        for n_ue in range(init_n_ue, final_n_ue+1, step_n_ue):
            print(f"\n── N_UE = {n_ue} ──────────────────────────────────────")
            distribution_class.set_number_of_ues(n_ue)

            # [M1] one controller per (seed, n_ue)
            madrl_ctrl = MADRLController(
                n_ues=n_ue, T_step=T_STEP_BO,
                W_min=W_MIN, W_max=W_MAX, Q_min=Q_MIN, Q_max=Q_MAX,
                share_routing=True, agent_kwargs=AGENT_KWARGS)
            if os.path.exists(CKPT_DIR):
                try: madrl_ctrl.load(CKPT_DIR)
                except Exception: pass

            ue_array = instantiate_ues(inputs, n_ue, ue_start_state, t_ack_tick,
                                        sim_tick_s, bit_rate, max_n_pkt_fwd)
            distribution_class.distribute_ues(
                ue_array, machine_array, bs, sim_tick_s,
                geometry_class.get_factory_length(),
                geometry_class.get_factory_width(),
                geometry_class.get_factory_height())
            for ue in ue_array:
                ue.starting_coordinates = ue.get_coordinates()

            compute_propagation_delays(ue_array, bs, sim_tick_s)
            max_prop_d = round(
                (math.sqrt(geometry_class.get_factory_length()**2 +
                           geometry_class.get_factory_width()**2 +
                           geometry_class.get_factory_height()**2) / c) / sim_tick_s)

            for i, ue in enumerate(ue_array):
                ue.is_in_los = set_ues_los_condition(ue, bs, machine_array, "ue_bs")
                ue.is_in_los_ues = []
            for j, ue in enumerate(ue_array):
                for i, other in enumerate(ue_array):
                    ue.is_in_los_ues.append(
                        set_ues_los_condition(ue, other, machine_array, "ue_ue"))

            if ue_dist_type != "Grid":
                check_for_neighbours(
                    ue_array=ue_array,
                    machine_array=machine_array,
                    bs=bs,
                    input_snr_threshold_db=sinr_th_db,
                    input_thz_channel=thz_channel,
                    input_carrier_frequency_ghz=f_ghz,
                    input_bandwidth_hz=bw_hz,
                    input_apply_fading=apply_fading,
                    input_clutter_density=clutter,
                    input_shadowing_sample_index=0,
                    use_channel_measurements=use_meas,
                    input_average_clutter_height_m=avg_mach_h,
                    antenna_gain_model=ant_model)
                compute_propagation_delays(ue_array, bs, sim_tick_s)

            nodes_list = [str(ue.get_ue_id()) for ue in ue_array] + ["BS"]
            for idx, ue in enumerate(ue_array):
                ue.set_neighbour_table(nodes_list[:idx] + nodes_list[idx+1:])

            ue_coords_list       = []
            copy_ue_coords_dict  = {ue.get_ue_id(): [] for ue in ue_array}

            for n_sim in range(n_simulations):
                print(f"  sim {n_sim:3d}", end="  ", flush=True)
                run_one_episode(
                    n_ue          = n_ue,
                    n_simulation  = n_sim,
                    madrl_ctrl    = madrl_ctrl,
                    ue_array      = ue_array,
                    bs            = bs,
                    machine_array = machine_array,
                    distribution_class = distribution_class,
                    scenario_df   = scenario_df,
                    geometry_class = geometry_class,
                    thz_channel   = thz_channel,
                    inputs        = inputs,
                    output_dict   = output_dict,
                    simulator_tick_duration_s = sim_tick_s,
                    tot_simulation_time_tick  = tot_dur,
                    t_ack_tick      = t_ack_tick,
                    max_prop_delay_tick = max_prop_d,
                    shadowing_coherence_time_tick_duration = shad_coh_tck,
                    n_shadowing_samples = n_shad,
                    sinr_th_db          = sinr_th_db,
                    carrier_frequency_ghz = f_ghz,
                    bandwidth_hz        = bw_hz,
                    apply_fading        = apply_fading,
                    clutter_density     = clutter,
                    antenna_gain_model  = ant_model,
                    use_channel_measurements = use_meas,
                    average_machine_height_m = avg_mach_h,
                    noise_figure_ue     = nf_ue,
                    noise_figure_bs     = nf_bs,
                    power_bo            = pwr_bo,
                    power_idle          = pwr_idle,
                    power_tx            = pwr_tx,
                    power_ack           = pwr_ack,
                    TTL                 = TTL,
                    max_n_packets_to_be_forwarded = max_n_pkt_fwd,
                    hop_limit           = hop_limit,
                    star_topology       = star_topology,
                    enable_print        = enable_print,
                    mobility_obstacle   = mob_obs,
                    mobility_spawn      = mob_spn,
                    mobility_shuffle    = mob_shf,
                    mobility_changes    = mob_chg,
                    step_size           = step_sz,
                    ue_coordinates_list = ue_coords_list,
                    copy_ue_coordinates_dict = copy_ue_coords_dict,
                )

                j_tag = f"N={n_ue}"; s_tag = f"Sim={n_sim}"
                j_val = float(np.mean(output_dict["j_index"][j_tag][s_tag]))
                s_val = float(np.mean(output_dict["s"][j_tag][s_tag])) * 1e-6
                print(f"J={j_val:.3f}  S={s_val:.3f}Mbit/s")

                if (n_sim+1) % inputs.get("rl",{}).get("agent",{}).get("n_simulations_for_training",10) == 0:
                    madrl_ctrl.save(CKPT_DIR)

                gc.collect()

    # ── aggregate & save ──────────────────────────────────────────────────────
    averaged: dict = {}
    for metric, ud in output_dict.items():
        averaged[metric] = []
        for _, sims in ud.items():
            avgs = []
            for _, arr in sims.items():
                if metric == "l":
                    nz = arr[arr > 0]
                    avgs.append(float(np.mean(nz)) if len(nz) else 0.0)
                else:
                    avgs.append(float(np.mean(arr)))
            averaged[metric].append(float(np.mean(avgs)))

    print("\n" + "="*50)
    print(f"p_mac     : {averaged['p_mac']}")
    print(f"S [bit/s] : {averaged['s']}")
    print(f"Latency   : {averaged['l']}")
    print(f"Jain      : {averaged['j_index']}")

    dt  = datetime.now()
    tag = (f"{dt.year}_{dt.month:02d}_{dt.day:02d}_MADRL"
           f"_N{final_n_ue}_P{payload_fq}_W{W_MIN}-{W_MAX}_Q{Q_MIN}-{Q_MAX}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, data in [("output_dict", output_dict), ("averaged", averaged), ("inputs", inputs)]:
        with open(f"{RESULTS_DIR}/{tag}_{name}.json", "w") as f:
            json.dump(data, f, indent=0, default=str)
    print(f"\nResults → {RESULTS_DIR}/{tag}_*.json")


if __name__ == "__main__":
    main()
