"""
madrl_ue_statemachine.py
========================
Complete UE and BS state-machine logic with every MADRL hook wired in.
This module is imported by madrl_simulator.py and called as:

    run_ue_statemachine(ue, t, context)
    run_bs_statemachine(bs, t, context)

where ``context`` is a SimContext dataclass holding all shared simulation
variables.  This keeps madrl_simulator.py clean and makes unit testing easy.
"""

from __future__ import annotations

import random
from copy import deepcopy
from math import log10
from typing import TYPE_CHECKING

import numpy as np

from madrl_integration import (
    choose_next_action_madrl,
    madrl_step_update,
)

if TYPE_CHECKING:
    from madrl_agent import MADRLController, UEMADRLBundle
    from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Shared context (all loop-scoped variables passed as one object)
# ─────────────────────────────────────────────────────────────────────────────

class SimContext:
    """
    Holds every variable that the state-machine functions need from the
    outer simulation loop.  Assign once per simulation episode.
    """
    __slots__ = [
        "madrl_ctrl", "ue_array", "bs",
        "simulator_timing_structure",
        "t_ack_tick", "t_backoff_tick", "max_prop_delay_tick",
        "hop_limit", "star_topology",
        "sinr_th_db", "enable_print",
        "simulator_tick_duration_s",
        "shadowing_sample_index", "shadowing_next_tick",
        "shadowing_coherence_time_tick_duration",
        "thz_channel", "carrier_frequency_ghz", "bandwidth_hz",
        "apply_fading", "clutter_density", "antenna_gain_model",
        "use_channel_measurements", "average_machine_height_m",
        "noise_figure_ue", "noise_figure_bs",
        "power_bo", "power_idle", "power_tx", "power_ack",
        "tot_simulation_time_tick",
        "TTL", "max_n_packets_to_be_forwarded",
        # MADRL-specific bookkeeping
        "bo_phase_count",    # dict ue_id → int
        "ack_received",      # dict ue_id → bool
        "packets_at_bs",     # dict ue_id → bool
        "T_STEP_BO",
        # helpers
        "ues_interfering_at_bs",
        "ues_colliding_at_bs",
        # timing helpers (imported in context)
        "_insert_fn", "_remove_fn",
        "_find_data_ue_fn", "_find_data_bs_fn",
        "_find_ack_ue_fn",  "_find_ack_bs_fn",
        "_check_collision_fn", "_check_collision_bs_fn",
        "_compute_distance_fn",
        "_go_in_backoff_fn", "_go_in_idle_fn",
        "_go_in_tx_data_fn", "_go_in_tx_ack_fn",
        "_go_in_tx_ack_bs_fn", "_go_in_wait_ack_fn",
        "_go_rx_ack_bs_fn",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# SINR helper (encapsulates the repeated collision/SINR pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sinr(
    tx_node, rx_node, ctx: SimContext,
    t_start_rx: int, t_end_rx: int,
    tx_key: str, ue_id_rx,
    colliding_list: list,
    interfering_list: list,
    los_cond: str,
) -> float:
    """
    Full SINR computation including capture effect and interference.
    Returns sinr_db.
    """
    tx_rx_d = ctx._compute_distance_fn(tx_node, rx_node)
    snr_db  = ctx.thz_channel.get_3gpp_snr_db(
        tx=tx_node, rx=rx_node,
        carrier_frequency_ghz=ctx.carrier_frequency_ghz,
        tx_rx_distance_m=tx_rx_d,
        apply_fading=ctx.apply_fading,
        bandwidth_hz=ctx.bandwidth_hz,
        clutter_density=ctx.clutter_density,
        input_shadowing_sample_index=ctx.shadowing_sample_index,
        use_channel_measurements=ctx.use_channel_measurements,
        input_average_clutter_height_m=ctx.average_machine_height_m,
        los_cond=los_cond,
        antenna_gain_model=ctx.antenna_gain_model)
    useful_prx_db = ctx.thz_channel.get_3gpp_prx_db(
        tx=tx_node, rx=rx_node,
        carrier_frequency_ghz=ctx.carrier_frequency_ghz,
        tx_rx_distance_m=tx_rx_d,
        apply_fading=ctx.apply_fading,
        bandwidth_hz=ctx.bandwidth_hz,
        clutter_density=ctx.clutter_density,
        input_shadowing_sample_index=ctx.shadowing_sample_index,
        use_channel_measurements=ctx.use_channel_measurements,
        input_average_clutter_height_m=ctx.average_machine_height_m,
        los_cond=los_cond,
        antenna_gain_model=ctx.antenna_gain_model)

    # check / update interfering-UE list
    add_it = True
    tx_id_str = (f"UE_{tx_node.get_ue_id()}"
                 if hasattr(tx_node, 'get_ue_id') else "BS")
    for item in interfering_list:
        if item[0] == tx_id_str and t_end_rx == item[2]:   # FIX-B: was item[1] (t_start_rx), must be item[2] (t_end_rx)
            add_it = False
    if add_it:
        interfering_list.append((tx_id_str, t_start_rx, t_end_rx))

    interference_pw = 0.0
    for user in ctx.ue_array:
        for col in colliding_list:
            uid_col = f"UE_{user.get_ue_id()}"
            if uid_col != tx_id_str and uid_col == col[0]:
                if uid_col == f"UE_{ue_id_rx if isinstance(ue_id_rx, int) else -1}":
                    continue
                t_ov = _overlap_fraction(t_start_rx, t_end_rx, col[1], col[2])
                d2   = ctx._compute_distance_fn(user, rx_node)
                interference_pw += t_ov * ctx.thz_channel.get_3gpp_prx_lin(
                    tx=user, rx=rx_node,
                    carrier_frequency_ghz=ctx.carrier_frequency_ghz,
                    tx_rx_distance_m=d2,
                    apply_fading=ctx.apply_fading,
                    bandwidth_hz=ctx.bandwidth_hz,
                    clutter_density=ctx.clutter_density,
                    input_shadowing_sample_index=ctx.shadowing_sample_index,
                    use_channel_measurements=ctx.use_channel_measurements,
                    input_average_clutter_height_m=ctx.average_machine_height_m,
                    los_cond=los_cond,
                    antenna_gain_model=ctx.antenna_gain_model)

    # also check interfering_list for lingering interferers
    copy_ifl = list(interfering_list)
    for user in ctx.ue_array:
        for item in copy_ifl:
            if (f"UE_{user.get_ue_id()}" == item[0]
                    and f"UE_{user.get_ue_id()}" != tx_id_str
                    and user.get_ue_id() != getattr(rx_node, 'get_ue_id', lambda: -1)()):
                if t_start_rx < item[2]:
                    t_ov = _overlap_fraction(t_start_rx, t_end_rx, item[1], item[2])
                    d2   = ctx._compute_distance_fn(user, rx_node)
                    interference_pw += t_ov * ctx.thz_channel.get_3gpp_prx_lin(
                        tx=user, rx=rx_node,
                        carrier_frequency_ghz=ctx.carrier_frequency_ghz,
                        tx_rx_distance_m=d2,
                        apply_fading=ctx.apply_fading,
                        bandwidth_hz=ctx.bandwidth_hz,
                        clutter_density=ctx.clutter_density,
                        input_shadowing_sample_index=ctx.shadowing_sample_index,
                        use_channel_measurements=ctx.use_channel_measurements,
                        input_average_clutter_height_m=ctx.average_machine_height_m,
                        los_cond=los_cond,
                        antenna_gain_model=ctx.antenna_gain_model)
                else:
                    # FIX-A: entry is stale (t_end_rx <= t_start_rx of current rx),
                    # prune it so the list stays bounded and SINR stays O(1) per call
                    try:
                        interfering_list.remove(item)
                    except ValueError:
                        pass

    if interference_pw == 0.0:
        return float(snr_db)

    noise_figure = (ctx.noise_figure_ue
                    if hasattr(rx_node, 'get_ue_id') else ctx.noise_figure_bs)
    noise_pw_dbw = ctx.thz_channel.get_thermal_noise_power_dbw(
        noise_figure, ctx.bandwidth_hz)
    noise_pw = 10 ** (noise_pw_dbw / 10)
    useful_pw = 10 ** (useful_prx_db / 10)
    sinr = useful_pw / (noise_pw + interference_pw)
    return float(10 * log10(sinr))


def _overlap_fraction(t_s: int, t_e: int, col_s: int, col_e: int) -> float:
    if col_s < t_e < col_e:
        return (t_e - col_s) / max(t_e - t_s, 1)
    return (col_e - col_s) / max(t_e - t_s, 1)


def _update_shadowing(ctx: SimContext, t: int):
    if t >= ctx.shadowing_next_tick:
        ctx.shadowing_sample_index     += 1
        ctx.shadowing_next_tick         = t + ctx.shadowing_coherence_time_tick_duration


# ─────────────────────────────────────────────────────────────────────────────
# UE state machine
# ─────────────────────────────────────────────────────────────────────────────

def run_ue_statemachine(ue, t: int, ctx: SimContext):
    """
    Process one UE at tick t.  Modifies ue state in-place.
    Called for every UE in the main while loop when t == ue.get_state_duration().
    """
    uid    = ue.get_ue_id()
    bundle: UEMADRLBundle = ctx.madrl_ctrl.bundle(uid)
    Q_ue   = ctx.madrl_ctrl.Q(uid)    # per-UE buffer limit [MADRL]

    state = ue.get_state()

    # ── IDLE ──────────────────────────────────────────────────────────────────
    if state == 'IDLE':
        if ue.get_n_packets() > 0:
            ue.update_num_tx(ctx.enable_print)
            bo_dur = _backoff_duration(ue, ctx)
            ctx._go_in_backoff_fn(ue, t, bo_dur, ctx.enable_print)
            ctx.bo_phase_count[uid] += 1
        else:
            ctx._go_in_idle_fn(ue, t, ctx.enable_print)

    # ── BACK-OFF ───────────────────────────────────────────────────────────────
    elif state == 'BO':
        _handle_bo(ue, uid, t, bundle, Q_ue, ctx)

    # ── TX_ACK ────────────────────────────────────────────────────────────────
    elif state == 'TX_ACK':
        _handle_tx_ack(ue, uid, t, bundle, Q_ue, ctx)

    # ── TX_DATA ───────────────────────────────────────────────────────────────
    elif state == 'TX_DATA':
        wad = ctx.t_ack_tick + 2 * ctx.max_prop_delay_tick
        ctx._go_in_wait_ack_fn(ue, t, wad, ctx.enable_print)
        ue.first_entry = True

    # ── WAIT_ACK ──────────────────────────────────────────────────────────────
    elif state == 'WAIT_ACK':
        _handle_wait_ack(ue, uid, t, bundle, Q_ue, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# BO handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_bo(ue, uid: int, t: int, bundle, Q_ue: int, ctx: SimContext):
    go_tx = False

    if not ctx.star_topology:
        if (ue.get_state_duration() == ue.get_state_final_tick()
                and not ue.get_reception_during_bo_bool()):
            go_tx = True
        else:
            ue.set_reception_during_bo_bool(False)
            (drx_st, drx_en, drx_sz, pid_rx, drx_uid) = ctx._find_data_ue_fn(
                ctx.simulator_timing_structure, uid, t)

            for idx in range(len(drx_uid)):
                _update_shadowing(ctx, t)
                tx_ue = ctx.ue_array[drx_uid[idx]]

                colliding = ctx._check_collision_fn(
                    input_simulator_timing_structure=ctx.simulator_timing_structure,
                    input_ue_id=uid,
                    input_t_start_rx=drx_st,
                    input_t_end_rx=drx_en,
                    input_tx=None,
                    input_ue_id_rx=drx_uid[idx],
                    ues_colliding=[])
                sinr_db = _compute_sinr(
                    tx_ue, ue, ctx, drx_st, drx_en,
                    f"UE_{drx_uid[idx]}", uid,
                    colliding, ue.ues_interfering_at_ue, "ue_ue")

                buf_has_own = ue.check_generated_packet_present()
                max_buf = Q_ue + 1 if buf_has_own else Q_ue
                success = (sinr_db >= ctx.sinr_th_db
                           and len(ue.ul_buffer.buffer_packet_list) < max_buf)

                if success:
                    _forward_rx_packet_bo(ue, uid, t, idx, tx_ue,
                                          drx_sz, pid_rx, drx_uid, ctx)
                else:
                    ue.set_state_duration(ue.get_state_final_tick())
                    bundle.n_collisions_last += 1   # [MADRL] count failed RX as collision

                if len(ctx.simulator_timing_structure[f'UE_{uid}']['DATA_RX']
                       [f'UE_{drx_uid[idx]}']) > 1:
                    ctx._remove_fn(ctx.simulator_timing_structure,
                                   f'UE_{drx_uid[idx]}', 'DATA_RX', f'UE_{uid}')
    else:
        if ue.get_state_duration() == ue.get_state_final_tick():
            go_tx = True

    # forwarded packets during BO → go TX_ACK first
    if ue.packet_forward and t == ue.get_state_final_tick():
        ctx._go_in_tx_ack_fn(ue, t, len(ue.data_rx_at_ue_ue_id_list) * ctx.t_ack_tick,
                              ctx.enable_print)
        _schedule_ack_from_ue(ue, uid, t, ctx)
        ue.ues_colliding_at_ue.clear()
        ue.data_rx_at_ue_ue_id_list.clear()
        ue.ues_interfering_at_ue.clear()
        ue.previous_state = 'BO'
        ue.packet_forward = False

    elif go_tx and not ue.forward_in_bo:
        ctx._go_in_tx_data_fn(ue, t, ctx.enable_print)
        ue.ues_colliding_at_ue.clear()
        ue.ues_interfering_at_ue.clear()
        ue.check_last_round = True
        ue.update_n_data_tx(ctx.enable_print)

        if not ctx.star_topology:
            # ── [MADRL] routing decision replaces TB ──────────────────────────
            choose_next_action_madrl(ue, bundle, ctx.hop_limit, ctx.enable_print)
            _schedule_data_tx_mesh(ue, uid, t, ctx)
        else:
            _schedule_data_tx_star(ue, uid, t, ctx)

        # ── [MADRL] BO phase counter + T_step trigger ─────────────────────────
        ctx.bo_phase_count[uid] += 1
        if ctx.bo_phase_count[uid] % ctx.T_STEP_BO == 0:
            losses = madrl_step_update(
                ue, bundle, ctx.hop_limit,
                ack_received=ctx.ack_received.get(uid, False),
                packet_at_bs=ctx.packets_at_bs.get(uid, False),
                done=False)
            ctx.ack_received[uid]  = False
            ctx.packets_at_bs[uid] = False
            if ctx.enable_print and losses[0] is not None:
                print(f"[MADRL] UE {uid}  r_loss={losses[0]:.4f}"
                      f"  cw_loss={losses[1]:.4f}  buf_loss={losses[2]:.4f}"
                      f"  W={bundle.W_current}  Q={bundle.Q_current}")


# ─────────────────────────────────────────────────────────────────────────────
# TX_ACK handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_tx_ack(ue, uid: int, t: int, bundle, Q_ue: int, ctx: SimContext):
    ue.list_data_rx_during_wait_ack.clear()
    ue.list_data_generated_during_wait_ack.clear()
    for k in ue.dict_data_rx_during_wait_ack:
        ue.dict_data_rx_during_wait_ack[k].clear()
    ue.list_data_rx_from_ue_id.clear()
    for k in ue.dict_data_rx_during_bo:
        ue.dict_data_rx_during_bo[k].clear()

    if ue.previous_state == 'WAIT_ACK':
        if not ue.check_generated_packet_present():
            ue.new_action_bool = True
        bo_dur = _backoff_duration(ue, ctx)
        ctx._go_in_backoff_fn(ue, t, bo_dur, ctx.enable_print)
        ctx.bo_phase_count[uid] += 1        # [MADRL]
        ue.first_bo_entry = True

    elif ue.previous_state == 'BO':
        ue.check_last_round = True
        ctx._go_in_tx_data_fn(ue, t, ctx.enable_print)
        ue.update_n_data_tx(ctx.enable_print)

        # [MADRL] routing decision
        choose_next_action_madrl(ue, bundle, ctx.hop_limit, ctx.enable_print)
        _schedule_data_tx_mesh(ue, uid, t, ctx)

        # [MADRL] BO phase counter + T_step trigger
        ctx.bo_phase_count[uid] += 1
        if ctx.bo_phase_count[uid] % ctx.T_STEP_BO == 0:
            losses = madrl_step_update(
                ue, bundle, ctx.hop_limit,
                ack_received=ctx.ack_received.get(uid, False),
                packet_at_bs=ctx.packets_at_bs.get(uid, False),
                done=False)
            ctx.ack_received[uid]  = False
            ctx.packets_at_bs[uid] = False


# ─────────────────────────────────────────────────────────────────────────────
# WAIT_ACK handler  (most complex state)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_wait_ack(ue, uid: int, t: int, bundle, Q_ue: int, ctx: SimContext):
    remain = False
    go_bo  = False
    go_idle = False

    if ue.first_entry:
        ue.copy_buffer_packet_list = deepcopy(ue.ul_buffer.buffer_packet_list)
        ue.first_entry = False

    timed_out = (ue.get_state_duration() == ue.get_state_final_tick()
                 and not ue.get_reception_during_wait_bool())

    if timed_out:
        ue.set_retransmission_packets(True)
        bundle.n_collisions_last += 1      # [MADRL] timeout counts as collision

    else:
        # ── ACK received during WAIT_ACK ──────────────────────────────────────
        if ue.ack_rx_during_wait_ack:
            ue.ack_rx_during_wait_ack = False
            ue.set_reception_during_wait_bool(False)
            (ack_st, ack_en, ack_srcs, ack_dests, ack_ids) = ctx._find_ack_ue_fn(
                ctx.simulator_timing_structure, uid, t)
            ue.ack_rx_with_success = False

            for idx in range(len(ack_srcs)):
                src    = ack_srcs[idx]
                dest   = ack_dests[idx]
                is_bs  = (src == 'BS')
                tx_node = ctx.bs if is_bs else ctx.ue_array[int(src[3:])]
                los    = 'bs_ue' if is_bs else 'ue_ue'

                _update_shadowing(ctx, t)
                colliding = ctx._check_collision_fn(
                    input_simulator_timing_structure=ctx.simulator_timing_structure,
                    input_ue_id=uid,
                    input_t_start_rx=ack_st,
                    input_t_end_rx=ack_en,
                    input_tx=src,
                    input_ue_id_rx=uid,
                    ues_colliding=[])
                sinr_db = _compute_sinr(
                    tx_node, ue, ctx, ack_st, ack_en,
                    src, uid, colliding, ue.ues_interfering_at_ue, los)

                if sinr_db >= ctx.sinr_th_db and dest == uid:
                    ue.ack_rx_with_success = True
                    # [MADRL] successful ACK
                    ctx.ack_received[uid]  = True
                    bundle.n_tx_last      += 1

                    # obs update
                    if is_bs:
                        if ue.get_broadcast_bool():
                            ue.set_temp_obs_broadcast(-1, 0)
                        elif ue.get_last_action() == 0:
                            if (ue.get_unicast_rx_address() in (src[3:], src)):
                                ue.update_neighbor_table_unicast_success(0)
                    else:
                        src_idx = ue.neighbour_table.index(src[3:]) \
                                  if src[3:] in ue.neighbour_table else -1
                        bs_seen = ctx.ue_array[int(src[3:])].obs[0][-1] \
                                  if len(ctx.ue_array[int(src[3:])].obs[0]) else 0
                        if ue.get_broadcast_bool() and src_idx >= 0:
                            ue.set_temp_obs_broadcast(src_idx, 0, bs_seen)
                        elif ue.get_last_action() == 0:
                            if (ue.get_unicast_rx_address() in (src[3:], src)):
                                ue.update_neighbor_table_unicast_success(0, bs_seen)

                    # remove ACKed packets
                    _remove_acked_packets(ue, uid, t, src, is_bs, ack_ids, idx, ctx)

                    if len(ue.ul_buffer.buffer_packet_list) > 0 and \
                            not ue.list_data_rx_during_wait_ack:
                        remain = True

                else:
                    bundle.n_collisions_last += 1   # [MADRL]
                    if ue.get_state_duration() != ue.get_state_final_tick():
                        remain = True
                    else:
                        remain = False
                        ue.set_retransmission_packets(True)

            if ue.ack_rx_with_success and ue.get_last_action() == 0:
                ue.unicast_handling_no_reward_no_neighbor_update()
                remain = False

            for idx in range(len(ack_srcs)):
                if len(ctx.simulator_timing_structure
                       [f'UE_{uid}']['ACK_RX'][ack_srcs[idx]]) > 1:
                    ctx._remove_fn(ctx.simulator_timing_structure,
                                   ack_srcs[idx], 'ACK_RX', f'UE_{uid}')
            ack_ids.clear()

        # ── DATA received during WAIT_ACK ─────────────────────────────────────
        if ue.data_rx_during_wait_ack and not ctx.star_topology:
            ue.data_rx_during_wait_ack = False
            (drx_st, drx_en, drx_sz, pid_rx, drx_uid) = ctx._find_data_ue_fn(
                ctx.simulator_timing_structure, uid, t)

            for idx in range(len(drx_uid)):
                _update_shadowing(ctx, t)
                tx_ue = ctx.ue_array[drx_uid[idx]]
                colliding = ctx._check_collision_fn(
                    input_simulator_timing_structure=ctx.simulator_timing_structure,
                    input_ue_id=uid,
                    input_t_start_rx=drx_st,
                    input_t_end_rx=drx_en,
                    input_tx=None,
                    input_ue_id_rx=drx_uid[idx],
                    ues_colliding=[])
                sinr_db = _compute_sinr(
                    tx_ue, ue, ctx, drx_st, drx_en,
                    f"UE_{drx_uid[idx]}", uid,
                    colliding, ue.ues_interfering_at_ue, "ue_ue")

                buf_has_own = ue.check_generated_packet_present()
                max_buf = Q_ue + 1 if buf_has_own else Q_ue
                success = (sinr_db >= ctx.sinr_th_db
                           and len(ue.ul_buffer.buffer_packet_list) < max_buf)

                if success:
                    _forward_rx_packet_wait_ack(ue, uid, t, idx, tx_ue,
                                                drx_sz, pid_rx, drx_uid, ctx)
                else:
                    remain = True
                    if len(ue.ul_buffer.buffer_packet_list) >= Q_ue:
                        bundle.n_drop_q += 1   # [MADRL] buffer overflow

                if len(ctx.simulator_timing_structure
                       [f'UE_{uid}']['DATA_RX'][f'UE_{drx_uid[idx]}']) > 1:
                    ctx._remove_fn(ctx.simulator_timing_structure,
                                   f'UE_{drx_uid[idx]}', 'DATA_RX', f'UE_{uid}')

                if not ue.ack_rx_during_wait_ack:
                    if ue.get_state_duration() != ue.get_state_final_tick():
                        remain = True
                    else:
                        remain = False
                        ue.set_retransmission_packets(True)

    # retransmission limit check
    if len(ue.ul_buffer.buffer_packet_list) > 0:
        ue.set_retransmission_packets(True)
    else:
        ue.set_retransmission_packets(False)

    if ue.get_broadcast_bool():
        ue.ack_rx_during_wait_ack = False
        ue.data_rx_during_wait_ack = False
        remain = (t != ue.get_state_final_tick())

    if remain:
        ue.set_state_duration(ue.get_state_final_tick())
        return

    # ── decide next state ─────────────────────────────────────────────────────
    if len(ue.list_data_generated_during_wait_ack) > 0 or \
            len(ue.list_data_rx_from_ue_id) > 0:
        _transition_wait_ack_to_tx_ack(ue, uid, t, bundle, Q_ue, ctx)

    elif len(ue.list_data_generated_during_wait_ack) == 0:
        ue.ues_colliding_at_ue.clear()
        ue.ues_interfering_at_ue.clear()

        if ue.get_broadcast_bool():
            ue.check_remove_packet(ctx.enable_print)
            ue.set_retransmission_packets(
                len(ue.ul_buffer.buffer_packet_list) > 0)

        ue.energy_consumed -= ctx.power_ack * (
            ue.get_state_final_tick() - t) * ctx.simulator_tick_duration_s

        if not ue.get_retransmission_packets():
            # successful TX → try new data
            if ue.is_there_a_new_data(t, ctx.max_n_packets_to_be_forwarded):
                ue.update_num_tx(ctx.enable_print)
                ue.check_last_round = False
                ue.check_num_tx()
                ue.new_action_bool = True
            _handle_broadcast_outcome(ue, ctx)
            go_bo = True

        elif ue.get_retransmission_packets():
            go_bo = True
            ue.update_num_tx(ctx.enable_print)
            for p in ue.ul_buffer.buffer_packet_list:
                p.set_retransmission_packets(True)

            _handle_broadcast_outcome(ue, ctx)
            _handle_unicast_failure(ue, ctx)

            if not ue.check_num_tx():
                # [MADRL] max-retx drop
                bundle.n_drop_r += 1
                if not ue.check_generated_packet_present():
                    if ue.is_there_a_new_data(t, ctx.max_n_packets_to_be_forwarded):
                        ue.update_num_tx(
                            input_packet_id=ue.ul_buffer.get_last_packet().get_id())
                        ue.check_last_round = False
                        ue.new_action_bool  = True
                    else:
                        go_idle = True
                        go_bo   = False
            else:
                if ue.reception_ack_during_wait:
                    ue.reception_ack_during_wait = False
                    _maybe_generate_new_packet(ue, t, ctx)
        else:
            go_idle = True

    if go_bo:
        ue.reception_ack_during_wait = False
        _check_new_action_for_bo(ue)
        bo_dur = _backoff_duration(ue, ctx)
        ctx._go_in_backoff_fn(ue, t, bo_dur, ctx.enable_print)
        ctx.bo_phase_count[uid] += 1    # [MADRL]
        ue.list_ack_sent_from_bs.clear()
        ue.ues_colliding_at_ue.clear()
        ue.ues_interfering_at_ue.clear()

    elif go_idle:
        ue.reception_ack_during_wait = False
        ctx._go_in_idle_fn(ue, t, ctx.enable_print)
        ue.list_ack_sent_from_bs.clear()
        ue.ues_colliding_at_ue.clear()
        ue.ues_interfering_at_ue.clear()


# ─────────────────────────────────────────────────────────────────────────────
# BS state machine
# ─────────────────────────────────────────────────────────────────────────────

def run_bs_statemachine(bs, t: int, ctx: SimContext):
    """Called when t == bs.get_state_duration() (event-driven)."""
    if bs.get_state() in ('IDLE', 'RX'):
        _handle_bs_rx(bs, t, ctx)
    elif bs.get_state() == 'TX_ACK':
        ctx._go_rx_ack_bs_fn(bs, t, ctx.tot_simulation_time_tick + 1,
                             ctx.enable_print)


def _handle_bs_rx(bs, t: int, ctx: SimContext):
    bs.rx_data = False
    (drx_st, drx_en, pids_rx, drx_uid) = ctx._find_data_bs_fn(
        ctx.simulator_timing_structure, t)
    (ack_st, ack_en, ack_dests, ack_ids, ack_txs, n_sim) = ctx._find_ack_bs_fn(
        ctx.simulator_timing_structure, t)

    if drx_st is not None and drx_uid:
        for idx in range(len(drx_uid)):
            _update_shadowing(ctx, t)
            tx_ue = ctx.ue_array[drx_uid[idx]]
            colliding = ctx._check_collision_bs_fn(
                input_simulator_timing_structure=ctx.simulator_timing_structure,
                input_ue_id=drx_uid[idx],
                input_t_start_rx=drx_st,
                input_t_end_rx=drx_en,
                ues_colliding=[])
            sinr_db = _compute_sinr(
                tx_ue, bs, ctx, drx_st, drx_en,
                f"UE_{drx_uid[idx]}", drx_uid[idx],
                colliding, ctx.ues_interfering_at_bs, "bs_ue")

            if sinr_db >= ctx.sinr_th_db:
                bs.sequence_number_of_packet_rx = pids_rx[idx]
                pkts_rx = _process_bs_rx_success(bs, tx_ue, drx_uid[idx],
                                                 pids_rx, idx, t, ctx)
                bs.rx_data = True

                # [MADRL] signal delivery to routing agents
                src_id = tx_ue.get_ue_id()
                ctx.packets_at_bs[src_id] = True
                # credit the generator UE if this was a relay hop
                for p in tx_ue.buffer_packet_sent:
                    if p.get_data_rx_from_ue() is not None:
                        gen = p.get_generated_by_ue()
                        ctx.packets_at_bs[gen] = True

            if len(ctx.simulator_timing_structure['BS']['DATA_RX']
                   [f'UE_{drx_uid[idx]}']) > 1:
                ctx._remove_fn(ctx.simulator_timing_structure,
                               f'UE_{drx_uid[idx]}', 'DATA_RX', 'BS')

    # BS ACK-RX cleanup (unchanged logic)
    if ack_st is not None and ack_txs:
        for idx in range(len(ack_txs)):
            if ack_txs[idx] is not None:
                for pid in ack_ids:
                    if (len(ctx.simulator_timing_structure['BS']['ACK_RX']
                            [ack_txs[idx]]) > 1 and
                            ctx.simulator_timing_structure['BS']['ACK_RX']
                            [ack_txs[idx]][:, 3][1] == pid):
                        ctx._remove_fn(ctx.simulator_timing_structure,
                                       ack_txs[idx], 'ACK_RX', 'BS')
        ack_ids.clear()

    # BS sends ACK after full burst received
    if bs.end_of_rx_for_ack_tx == t:
        _bs_send_ack(bs, t, ctx)
    else:
        ctx._go_rx_ack_bs_fn(bs, t, ctx.tot_simulation_time_tick + 1,
                             ctx.enable_print)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-helpers
# ─────────────────────────────────────────────────────────────────────────────

def _backoff_duration(ue, ctx: SimContext) -> int:
    W   = ctx.madrl_ctrl.W(ue.get_ue_id())   # [MADRL] per-UE W
    exp = pow(2, ue.get_ul_buffer().get_first_packet().get_num_tx())
    delay = random.randint(1, exp * W)
    if ctx.star_topology:
        return delay * ctx.t_backoff_tick
    return (ue.get_data_duration_tick() + ctx.max_prop_delay_tick
            + delay * ctx.t_backoff_tick)


def _schedule_data_tx_mesh(ue, uid: int, t: int, ctx: SimContext):
    st = t
    for pkt in ue.ul_buffer.buffer_packet_list:
        td = round((pkt.packet_size * 8e-9) / ctx.bs.get_bit_rate_gbits()
                   / ctx.simulator_tick_duration_s)
        for other in ctx.ue_array:
            if other != ue:
                ctx._insert_fn(ctx.simulator_timing_structure,
                               st + ue.get_prop_delay_to_ue_tick(other.get_ue_id()),
                               st + td + ue.get_prop_delay_to_ue_tick(other.get_ue_id()),
                               pkt.packet_size, pkt.packet_id,
                               f'UE_{uid}', 'DATA_RX', f'UE_{other.get_ue_id()}')
        ctx._insert_fn(ctx.simulator_timing_structure,
                       st + ue.get_prop_delay_to_bs_tick(),
                       st + td + ue.get_prop_delay_to_bs_tick(),
                       pkt.packet_size, pkt.packet_id,
                       f'UE_{uid}', 'DATA_RX', 'BS')
        st += td


def _schedule_data_tx_star(ue, uid: int, t: int, ctx: SimContext):
    for pkt in ue.ul_buffer.buffer_packet_list:
        ctx._insert_fn(ctx.simulator_timing_structure,
                       t + ue.get_prop_delay_to_bs_tick(),
                       ue.get_state_duration() + ue.get_prop_delay_to_bs_tick(),
                       pkt.packet_size, pkt.packet_id,
                       f'UE_{uid}', 'DATA_RX', 'BS')


def _schedule_ack_from_ue(ue, uid: int, t: int, ctx: SimContext):
    j = -1
    for ue_id in ue.data_rx_at_ue_ue_id_list:
        j += 1
        for idx in range(len(ue.dict_data_rx_during_bo[ue_id])):
            for other in ctx.ue_array:
                if other != ue:
                    ctx._insert_fn(ctx.simulator_timing_structure,
                                   t + ue.get_prop_delay_to_ue_tick(other.get_ue_id())
                                   + j * ctx.t_ack_tick,
                                   t + ue.get_prop_delay_to_ue_tick(other.get_ue_id())
                                   + ctx.t_ack_tick + j * ctx.t_ack_tick,
                                   ue_id, ue.dict_data_rx_during_bo[ue_id][idx],
                                   f'UE_{uid}', 'ACK_RX', f'UE_{other.get_ue_id()}')
            ctx._insert_fn(ctx.simulator_timing_structure,
                           t + ue.get_prop_delay_to_bs_tick() + j * ctx.t_ack_tick,
                           t + ue.get_prop_delay_to_bs_tick()
                           + ctx.t_ack_tick + j * ctx.t_ack_tick,
                           ue_id, ue.dict_data_rx_during_bo[ue_id][idx],
                           f'UE_{uid}', 'ACK_RX', 'BS')


def _forward_rx_packet_bo(ue, uid, t, idx, tx_ue,
                           drx_sz, pid_rx, drx_uid, ctx: SimContext):
    """Handle forwarding of a packet received during BO."""
    ue.packet_forwarding.append(pid_rx[idx])
    Q_ue = ctx.madrl_ctrl.Q(uid)
    buf_has_own = ue.check_generated_packet_present()
    max_buf = Q_ue + 1 if buf_has_own else Q_ue

    if drx_sz[idx] > 0 and len(ue.ul_buffer.buffer_packet_list) < max_buf:
        for p in tx_ue.buffer_packet_sent:
            if p.packet_id == pid_rx[idx]:
                if p.address in (str(uid), "-1"):
                    ue.designated_rx = True
                    break
        if ue.designated_rx:
            ue.designated_rx = False
            for n_p in range(len(tx_ue.buffer_packet_sent)):
                _try_enqueue_forwarded_packet(
                    ue, uid, t, tx_ue, n_p, pid_rx, idx,
                    drx_uid, ctx, phase='bo')
            if ue.forward_in_bo:
                ue.forward_in_bo = False
                ue.data_rx_at_ue_ue_id_list.append(drx_uid[idx])
                if np.sum(ue.obs[1]) > 0:
                    adj = -1 if uid < drx_uid[idx] else 0
                    ue.set_obs_update(drx_uid[idx] + adj, 0)
            else:
                ue.set_state_duration(ue.get_state_final_tick())
        else:
            ue.set_state_duration(ue.get_state_final_tick())
    else:
        ue.set_state_duration(ue.get_state_final_tick())


def _forward_rx_packet_wait_ack(ue, uid, t, idx, tx_ue,
                                  drx_sz, pid_rx, drx_uid, ctx: SimContext):
    """Handle forwarding of a packet received during WAIT_ACK."""
    ue.packet_forwarding.append(pid_rx[idx])
    Q_ue = ctx.madrl_ctrl.Q(uid)
    buf_has_own = ue.check_generated_packet_present()
    max_buf = Q_ue + 1 if buf_has_own else Q_ue

    if drx_sz[idx] > 0 and len(ue.ul_buffer.buffer_packet_list) < max_buf:
        for p in tx_ue.buffer_packet_sent:
            if p.packet_id == pid_rx[idx]:
                if p.address in (str(uid), "-1"):
                    ue.designated_rx = True
                    break
        if ue.designated_rx:
            ue.designated_rx = False
            for n_p in range(len(tx_ue.buffer_packet_sent)):
                _try_enqueue_forwarded_packet(
                    ue, uid, t, tx_ue, n_p, pid_rx, idx,
                    drx_uid, ctx, phase='wait_ack')
            if ue.forward_in_wait_ack:
                ue.forward_in_wait_ack = False
                if np.sum(ue.obs[1]) > 0:
                    adj = -1 if uid < drx_uid[idx] else 0
                    ue.set_obs_update(drx_uid[idx] + adj, 0)
            else:
                pass  # remain handled by caller


def _try_enqueue_forwarded_packet(ue, uid, t, tx_ue, n_p,
                                   pid_rx, idx, drx_uid, ctx, phase):
    """Common packet-enqueue logic for BO and WAIT_ACK forwarding."""
    if pid_rx[idx] != tx_ue.buffer_packet_sent[n_p].packet_id:
        return
    Q_ue = ctx.madrl_ctrl.Q(uid)
    buf_has_own = ue.check_generated_packet_present()
    max_buf = Q_ue + 1 if buf_has_own else Q_ue
    if len(ue.ul_buffer.buffer_packet_list) >= max_buf:
        return

    pkt = tx_ue.buffer_packet_sent[n_p]
    already_q = any(
        p.get_generated_by_ue() == pkt.get_generated_by_ue() and
        p.get_packet_id_generator() == pkt.get_packet_id_generator()
        for p in ue.ul_buffer.buffer_packet_list)
    gen_by_self = (pkt.get_generated_by_ue() == uid)
    hop_exceeded = (pkt.get_hop_count() >= ctx.hop_limit)

    if gen_by_self and ue.obs[0][-1] == 0:
        ue.next_action = 3

    if not already_q and not gen_by_self and not hop_exceeded:
        fwd_flag = phase == 'bo'
        if fwd_flag:
            ue.forward_in_bo   = True
            ue.packet_forward  = True   # FIX-C: was missing; _handle_bo checks packet_forward to send ACK
        else:
            ue.forward_in_wait_ack = True
        ue.n_forwarding += 1
        ue.add_new_packet(
            current_tick=t,
            input_enable_print=ctx.enable_print,
            input_data_to_be_forwarded_bool=True,
            input_packet_size_bytes=pkt.packet_size,
            input_simulation_tick_duration=ctx.simulator_tick_duration_s,
            data_rx_from_ue=drx_uid[idx],
            packet_id_rx_from_ue=pkt.get_id(),
            packet_generated_by_ue=pkt.get_generated_by_ue(),
            packet_id_generator=pkt.get_packet_id_generator(),
            packet_hop_count=pkt.get_hop_count(),
            packet_address=(ue.get_unicast_rx_address()
                            if not ue.get_broadcast_bool() else "-1"),
            generation_time=pkt.get_generated_by_ue_time_instant_tick())
        ue.update_num_tx(
            input_packet_id=ue.ul_buffer.get_last_packet().get_id())
        if phase == 'bo':
            ue.dict_data_rx_during_bo[drx_uid[idx]].append(pkt.get_id())
        else:
            ue.list_data_generated_during_wait_ack.append(
                ue.ul_buffer.get_last_packet().get_id())
            ue.list_data_rx_during_wait_ack.append(pkt.get_id())
            ue.dict_data_rx_during_wait_ack[drx_uid[idx]].append(pkt.get_id())
            ue.list_data_rx_from_ue_id.append(drx_uid[idx])
        if pkt.get_id() not in ue.dict_ack_sent_from_ue[drx_uid[idx]]:
            ue.dict_ack_sent_from_ue[drx_uid[idx]].append(pkt.get_id())

    elif already_q and not gen_by_self:
        if phase == 'bo':
            ue.dict_data_rx_during_bo[drx_uid[idx]].append(pkt.get_id())
            if drx_uid[idx] not in ue.data_rx_at_ue_ue_id_list:
                ue.data_rx_at_ue_ue_id_list.append(drx_uid[idx])
        else:
            ue.dict_data_rx_during_wait_ack[drx_uid[idx]].append(pkt.get_id())
            ue.list_data_rx_from_ue_id.append(drx_uid[idx])
        if pkt.get_id() not in ue.dict_ack_sent_from_ue[drx_uid[idx]]:
            ue.dict_ack_sent_from_ue[drx_uid[idx]].append(pkt.get_id())


def _remove_acked_packets(ue, uid, t, src, is_bs, ack_ids, idx, ctx):
    """Remove from buffer the packets confirmed by this ACK."""
    ack_pid = ack_ids[idx]
    for p in range(len(ue.ul_buffer.buffer_packet_list)):
        if ue.ul_buffer.buffer_packet_list[p].get_id() == ack_pid:
            ue.ul_buffer.buffer_packet_list[p].set_ack_rx(True)

    to_remove = deepcopy(ue.ul_buffer.buffer_packet_list)
    for pkt in to_remove:
        if pkt.packet_id != ack_pid:
            continue
        if is_bs:
            if (pkt.get_id() not in ue.list_data_generated_during_wait_ack
                    and pkt.get_id() in ue.list_ack_sent_from_bs):
                if not ue.get_broadcast_bool():
                    ue.remove_packet(pkt.get_id(), ctx.enable_print)
                    ue.packets_sent -= 1
                else:
                    ue.packets_to_be_removed["BS"].append(pkt.get_id())
                ue.list_ack_sent_from_bs.remove(pkt.get_id())
        else:
            tx_ue = ctx.ue_array[int(src[3:])]
            if (not pkt.get_data_unicast()
                    and pkt.get_id() not in ue.list_data_generated_during_wait_ack
                    and pkt.get_id() in tx_ue.dict_ack_sent_from_ue[uid]):
                if not ue.get_broadcast_bool():
                    ue.remove_packet(pkt.get_id(), ctx.enable_print)
                    ue.packets_sent -= 1
                else:
                    ue.packets_to_be_removed[str(tx_ue.get_ue_id())].append(pkt.get_id())
                tx_ue.dict_ack_sent_from_ue[uid].remove(pkt.get_id())


def _transition_wait_ack_to_tx_ack(ue, uid, t, bundle, Q_ue, ctx):
    """Transition to TX_ACK to forward data received during WAIT_ACK."""
    if ue.get_broadcast_bool():
        ue.check_remove_packet(ctx.enable_print)
        ue.set_retransmission_packets(len(ue.ul_buffer.buffer_packet_list) > 0)

    for k in ctx.ue_array:
        if k != ue:
            ue.dict_ack_sent_from_ue[k.get_ue_id()].clear()

    ue.energy_consumed -= ctx.power_ack * (
        ue.get_state_final_tick() - t) * ctx.simulator_tick_duration_s
    ue.list_ack_sent_from_bs.clear()
    ue.ues_colliding_at_ue.clear()
    ue.ues_interfering_at_ue.clear()

    copy_dict = deepcopy(ue.dict_data_rx_during_wait_ack)
    ue.previous_state = 'WAIT_ACK'

    for pid in ue.list_data_generated_during_wait_ack:
        ue.update_num_tx(pid)
        ue.check_num_tx()

    unique_ids = list(dict.fromkeys(ue.list_data_rx_from_ue_id))
    ctx._go_in_tx_ack_fn(ue, t, len(unique_ids) * ctx.t_ack_tick, ctx.enable_print)

    j = -1
    for ue_id in unique_ids:
        j += 1
        for pidx in range(len(ue.dict_data_rx_during_wait_ack[ue_id])):
            for other in ctx.ue_array:
                if other != ue:
                    ctx._insert_fn(
                        ctx.simulator_timing_structure,
                        t + ue.get_prop_delay_to_ue_tick(other.get_ue_id()) + j * ctx.t_ack_tick,
                        t + ue.get_prop_delay_to_ue_tick(other.get_ue_id())
                        + ctx.t_ack_tick + j * ctx.t_ack_tick,
                        ue_id, ue.dict_data_rx_during_wait_ack[ue_id][pidx],
                        f'UE_{uid}', 'ACK_RX', f'UE_{other.get_ue_id()}')
            if copy_dict[ue_id][pidx] not in ue.dict_ack_sent_from_ue[ue_id]:
                ue.dict_ack_sent_from_ue[ue_id].append(copy_dict[ue_id][pidx])
            ctx._insert_fn(
                ctx.simulator_timing_structure,
                t + ue.get_prop_delay_to_bs_tick() + j * ctx.t_ack_tick,
                t + ue.get_prop_delay_to_bs_tick() + ctx.t_ack_tick + j * ctx.t_ack_tick,
                ue_id, ue.dict_data_rx_during_wait_ack[ue_id][pidx],
                f'UE_{uid}', 'ACK_RX', 'BS')

    for pkt in ue.ul_buffer.buffer_packet_list:
        retx = pkt.get_id() not in ue.list_data_generated_during_wait_ack
        pkt.set_retransmission_packets(retx)
        if retx:
            ue.update_num_tx(pkt.get_id(), ctx.enable_print)

    _handle_unicast_failure(ue, ctx)
    _handle_broadcast_outcome(ue, ctx)

    if not ue.check_num_tx():
        bundle.n_drop_r += 1   # [MADRL]
        if not ue.check_generated_packet_present():
            if ue.is_there_a_new_data(t, ctx.max_n_packets_to_be_forwarded):
                ue.update_num_tx(ue.ul_buffer.get_last_packet().get_id(), ctx.enable_print)
                ue.check_last_round = False
                if not ue.get_retransmission_packets():
                    ue.check_num_tx()
                ue.new_action_bool = True
    elif ue.reception_ack_during_wait:
        ue.reception_ack_during_wait = False
        _maybe_generate_new_packet(ue, t, ctx)


def _handle_broadcast_outcome(ue, ctx: SimContext):
    if ue.get_broadcast_bool():
        if np.sum(ue.temp_obs[1]) == 0:
            ue.broadcast_handling_failure_no_reward(ctx.TTL)
            ue.new_action_bool = True
            ue.reset_temp_obs()
        else:
            ue.broadcast_handling_no_reward(ctx.TTL)
            ue.set_last_action(None)
            ue.set_broadcast_bool(False)
            ue.new_action_bool = True
            ue.reset_temp_obs()


def _handle_unicast_failure(ue, ctx: SimContext):
    if ue.get_last_action() == 0:
        ue.unicast_handling_failure_no_reward(ctx.TTL)
        ue.new_action_bool = True


def _check_new_action_for_bo(ue):
    counter = sum(1 for p in ue.ul_buffer.buffer_packet_list
                  if p.get_retransmission_packets() and ue.get_last_action() is not None)
    if ue.check_generated_packet_present() or counter == 0:
        ue.new_action_bool = True


def _maybe_generate_new_packet(ue, t, ctx: SimContext):
    has_own = any(not p.get_data_to_be_forwarded_bool()
                  for p in ue.ul_buffer.buffer_packet_list)
    if not has_own:
        if ue.is_there_a_new_data(t, ctx.max_n_packets_to_be_forwarded):
            ue.update_num_tx(ue.ul_buffer.get_last_packet().get_id(), ctx.enable_print)
            ue.check_last_round = False
            ue.check_num_tx()


def _process_bs_rx_success(bs, tx_ue, ue_id, pids_rx, idx, t, ctx):
    """Process a successful DATA reception at the BS and return packet count."""
    pkts = 0
    if not (bs.end_of_rx_for_ack_tx is None or
            tx_ue.end_data_tx + tx_ue.get_prop_delay_to_bs_tick() > bs.end_of_rx_for_ack_tx):
        pass
    bs.end_of_rx_for_ack_tx = max(
        bs.end_of_rx_for_ack_tx or 0,
        tx_ue.end_data_tx + tx_ue.get_prop_delay_to_bs_tick())
    if ue_id not in bs.id_ues_data_rx:
        bs.id_ues_data_rx.append(ue_id)

    ack_pid = None
    for n_p in range(len(tx_ue.buffer_packet_sent)):
        pkt = tx_ue.buffer_packet_sent[n_p]
        if (pkt.get_id() != pids_rx[idx] or
                pkt.address not in ('BS', '-1')):
            continue
        if pkt.get_id() in bs.packet_id_received[ue_id]:
            ack_pid = pkt.get_id()
            tx_ue.list_ack_sent_from_bs.append(pkt.get_id())
            continue
        bs.packet_id_received[ue_id].append(pkt.get_id())
        tx_ue.set_ack_packet_id_ue(pkt.get_id())
        ack_pid = pkt.get_id()
        tx_ue.list_ack_sent_from_bs.append(pkt.get_id())
        pkts += 1

        if pkt.get_data_rx_from_ue() is None:
            bs.update_n_data_rx_from_ues(ue_id, pkts)
            if not ctx.star_topology:
                lat = (t + ctx.t_ack_tick + tx_ue.get_prop_delay_to_bs_tick()
                       - tx_ue.packet_generation_instant) * ctx.simulator_tick_duration_s
                tx_ue.latency_ue.append(lat)
        else:
            gen_id  = pkt.get_generated_by_ue()
            gen_pid = pkt.get_packet_id_generator()
            for gu in ctx.ue_array:
                if gu.get_ue_id() == gen_id:
                    gu.set_ack_packet_id_ue(gen_pid)
            already = gen_pid in bs.packet_id_received.get(gen_id, [])
            if not already:
                bs.packet_id_received[gen_id].append(gen_pid)
                bs.update_n_data_rx_from_ues(gen_id, 1)
                for gu in ctx.ue_array:
                    if gu.get_ue_id() == gen_id and not ctx.star_topology:
                        lat = (t + ctx.t_ack_tick + gu.get_prop_delay_to_bs_tick()
                               - pkt.get_generated_by_ue_time_instant_tick()
                               ) * ctx.simulator_tick_duration_s
                        gu.latency_ue.append(lat)

    if ack_pid is not None:
        bs.temp_packet_id_received[ue_id].append(ack_pid)
        bs.rx_data = True
    bs.update_n_data_rx(
        tx_ue.get_traffic_type(), pkts, ctx.enable_print)
    return pkts


def _bs_send_ack(bs, t: int, ctx: SimContext):
    for idx, ue_id in enumerate(bs.id_ues_data_rx):
        no_none = any(v is not None
                      for v in bs.temp_packet_id_received[ue_id])
        if bs.temp_packet_id_received[ue_id] and no_none:
            bs.packet_rx = True
            tx_ue = ctx.ue_array[ue_id]          # only the transmitting UE
            for i, pid in enumerate(bs.temp_packet_id_received[ue_id]):
                ctx._insert_fn(
                    ctx.simulator_timing_structure,
                    bs.end_of_rx_for_ack_tx
                    + tx_ue.get_prop_delay_to_bs_tick() + idx * ctx.t_ack_tick,
                    bs.end_of_rx_for_ack_tx
                    + tx_ue.get_prop_delay_to_bs_tick()
                    + ctx.t_ack_tick + idx * ctx.t_ack_tick,
                    ue_id, pid, 'BS', 'ACK_RX', f'UE_{ue_id}')
        bs.temp_packet_id_received[ue_id] = []

    if bs.packet_rx:
        ctx._go_in_tx_ack_bs_fn(bs, t, len(bs.id_ues_data_rx) * ctx.t_ack_tick,
                                ctx.enable_print)
        bs.packet_rx = False
    else:
        ctx._go_rx_ack_bs_fn(bs, t, ctx.tot_simulation_time_tick + 1, ctx.enable_print)
    bs.id_ues_data_rx.clear()

    # clean up stale entries in BS timing structure
    for ue in ctx.ue_array:
        for typ in ('ACK_RX', 'DATA_RX'):
            k = f'UE_{ue.get_ue_id()}'
            while len(ctx.simulator_timing_structure['BS'][typ][k]) > 1:
                r = ctx.simulator_timing_structure['BS'][typ][k][1]
                if r[1] <= bs.get_end_tx_ack() or r[0] <= bs.get_end_tx_ack():
                    ctx._remove_fn(ctx.simulator_timing_structure, k, typ, 'BS')
                else:
                    break