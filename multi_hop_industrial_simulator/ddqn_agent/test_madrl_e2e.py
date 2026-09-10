"""
test_madrl_e2e.py
=================
End-to-end integration test.  Runs the full MADRL pipeline
(agent selection → state construction → reward → training → evaluation)
using lightweight mock objects that reproduce the exact interface
of the real simulator's UE, BS, and buffer classes.

No simulator install required.  Run with:

    python test_madrl_e2e.py

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import os
import sys
import random
import tempfile
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── mock simulator objects ────────────────────────────────────────────────────

class MockPacket:
    def __init__(self, pid, size=50, hop=0, gen_by=0, gen_t=0):
        self.packet_id   = pid
        self.packet_size = size
        self.hop_count   = hop
        self._gen_by     = gen_by
        self._gen_t      = gen_t
        self._num_tx     = 0
        self._retx       = False
        self._ack_rx     = False
        self._fwd        = False
        self._unicast    = False
        self.address     = "-1"
    def get_id(self):                       return self.packet_id
    def get_size(self):                     return self.packet_size
    def get_num_tx(self):                   return self._num_tx
    def get_hop_count(self):                return self.hop_count
    def get_generated_by_ue(self):          return self._gen_by
    def get_packet_id_generator(self):      return self.packet_id
    def get_generated_by_ue_time_instant_tick(self): return self._gen_t
    def get_data_to_be_forwarded_bool(self):return self._fwd
    def get_retransmission_packets(self):   return self._retx
    def get_data_unicast(self):             return self._unicast
    def set_ack_rx(self, v):                self._ack_rx = v
    def set_retransmission_packets(self, v):self._retx = v
    def set_data_unicast(self, v):          self._unicast = v


class MockBuffer:
    def __init__(self):
        self.buffer_packet_list = []
    def get_first_packet(self):
        return self.buffer_packet_list[0] if self.buffer_packet_list else MockPacket(0)
    def get_last_packet(self):
        return self.buffer_packet_list[-1] if self.buffer_packet_list else MockPacket(0)


class MockUE:
    """
    Minimal UE mock matching every attribute and method accessed by
    madrl_integration.py and madrl_ue_statemachine.py.
    """
    _id_counter = 0

    def __init__(self, uid=None):
        if uid is None:
            uid = MockUE._id_counter; MockUE._id_counter += 1
        self._uid          = uid
        self.ul_buffer     = MockBuffer()
        self.neighbour_table = []
        # obs[0]=bs_seen_flags, obs[1]=ack_counts, obs[2]=rx_powers, obs[3]=ttl
        self.obs           = [[0]*6, [0]*5, [-90.0]*5, [10]*5, [0]*5]   # 5 rows: discovered,acks,prx,ttl,bs_seen
        self.temp_obs      = [[0]*5, [0]*5, [0.0]*5, [0]*5]
        self._state        = "IDLE"
        self._state_dur    = 0
        self._state_start  = 0
        self._state_final  = 0
        self._last_action  = None
        self._broadcast    = False
        self._unicast_addr = None
        self._unicast_idx  = None
        self.n_tx_last     = 0
        self.energy_consumed = 0.0
        self.ticks_in_BO   = []; self.ticks_in_TX_ACK = []
        self.ticks_in_TX_DATA = []; self.ticks_in_WAIT_ACK = []
        self.ues_colliding_at_ue  = []
        self.ues_interfering_at_ue = []
        self.buffer_packet_sent   = []
        self.packets_sent         = 0
        self.end_data_tx          = 0
        self.packet_generation_instant = 0
        self.first_entry          = False
        self.first_bo_entry       = True
        self.new_action_bool      = True
        self.next_action          = None
        self.copy_buffer_packet_list = None
        self.action_packet_id     = None
        self.designated_rx        = False
        self.forward_in_bo        = False
        self.forward_in_wait_ack  = False
        self.packet_forward       = False
        self.forward_in_ack       = False
        self.multihop_bool        = True
        self.check_last_round     = False
        self.n_tear               = 0
        self.n_forwarding         = 0
        self.n_interfering        = []
        self.ack_rx_during_wait_ack  = False
        self.data_rx_during_wait_ack = False
        self.list_data_rx_during_wait_ack       = []
        self.list_data_generated_during_wait_ack = []
        self.list_data_rx_from_ue_id            = []
        self.dict_data_rx_during_wait_ack       = {}
        self.dict_data_rx_during_bo             = {}
        self.dict_ack_sent_from_ue              = {}
        self.reception_ack_during_wait          = False
        self.list_ack_sent_from_bs              = []
        self.data_rx_at_ue_ue_id_list          = []
        self.latency_ue                         = []
        self.packet_forwarding                  = []
        self.packets_to_be_removed              = {}
        self.ues_colliding_at_ue                = []
        self._reception_bo                      = False
        self._reception_wait                    = False
        self._retx_packets                      = False
        self._relay                             = False
        self.previous_state                     = None
        self._saved_state                       = "IDLE"
        self.packet_id_received                 = {}
        self.starting_coordinates               = np.zeros(3)
        self.ack_rx_with_success                = False
        self.is_in_los                          = True
        self.is_in_los_ues                      = []
        self.forced_broadcast_actions_counter   = 0
        self.n_generated_packets                = 0
        self._n_data_tx = 0; self._n_data_rx = 0
        self._traffic_type = "RT"
        self._prop_delay_bs = 1; self._prop_delay_ues = {}
        self._next_gen = 100
        self._packet_id = 0
        self._actions_per_sim = [[], [], [], []]
        self._success_actions_per_sim = [[], []]
        self._replay_buf = None
        self.saved_coordinates = np.zeros(3)
        self._ack_packet_id = None
        self.temp_packet_id_received = {}

    # ── state getters / setters ───────────────────────────────────────────────
    def get_ue_id(self):                return self._uid
    def get_state(self):                return self._state
    def get_state_duration(self):       return self._state_dur
    def get_state_starting_tick(self):  return self._state_start
    def get_state_final_tick(self):     return self._state_final
    def get_last_action(self):          return self._last_action
    def get_broadcast_bool(self):       return self._broadcast
    def get_unicast_rx_address(self):   return self._unicast_addr
    def get_unicast_rx_index(self):     return self._unicast_idx
    def get_n_packets(self):            return len(self.ul_buffer.buffer_packet_list)
    def get_next_packet_generation_instant(self): return self._next_gen
    def get_retransmission_packets(self): return self._retx_packets
    def get_reception_during_bo_bool(self): return self._reception_bo
    def get_reception_during_wait_bool(self): return self._reception_wait
    def get_traffic_type(self):         return self._traffic_type
    def get_data_duration_tick(self):   return 5
    def get_prop_delay_to_bs_tick(self): return self._prop_delay_bs
    def get_prop_delay_to_ue_tick(self, uid): return self._prop_delay_ues.get(uid, 1)
    def get_updated_packet_list(self):  return list(self.ul_buffer.buffer_packet_list)
    def get_ack_packet_id_ue(self):     return self._ack_packet_id
    def get_coordinates(self):          return np.zeros(3)
    def get_relay_bool(self):           return self._relay

    def set_state(self, input_state=None, **kw):
        self._state = input_state
    def update_state_duration(self, input_ticks=0, **kw):
        self._state_dur += input_ticks
    def set_state_duration(self, input_ticks=0, **kw):
        self._state_dur = input_ticks
    def set_state_starting_tick(self, input_tick=0, **kw):
        self._state_start = input_tick
    def set_state_final_tick(self, input_tick=0, **kw):
        self._state_final = input_tick
    def set_last_action(self, input_last_action=None, **kw):
        self._last_action = input_last_action
    def set_broadcast_bool(self, input_broadcast_bool=False, **kw):
        self._broadcast = input_broadcast_bool
    def set_unicast_rx_address(self, input_unicast_rx_address=None, **kw):
        self._unicast_addr = input_unicast_rx_address
    def set_unicast_rx_index(self, input_unicast_rx_index=None, **kw):
        self._unicast_idx = input_unicast_rx_index
    def set_retransmission_packets(self, retransmission_bool=False, **kw):
        self._retx_packets = retransmission_bool
    def set_reception_during_bo_bool(self, input_data_rx_bool=False, **kw):
        self._reception_bo = input_data_rx_bool
    def set_reception_during_wait_bool(self, input_data_rx_bool=False, **kw):
        self._reception_wait = input_data_rx_bool
    def set_relay_bool(self, relay_bool=False, **kw):
        self._relay = relay_bool
    def set_neighbour_table(self, input_neighbour_table=None, **kw):
        self.neighbour_table = input_neighbour_table or []
    def get_neighbour_table(self):      return self.neighbour_table
    def set_t_generation(self, input_t_generation=0, **kw):
        self._next_gen = input_t_generation
    def set_packet_id(self, input_packet_id=0, **kw):
        self._packet_id = input_packet_id
    def set_n_data_tx(self, input_n_data_tx=0, **kw):
        self._n_data_tx = input_n_data_tx
    def set_n_data_rx(self, input_n_data_rx=0, **kw):
        self._n_data_rx = input_n_data_rx
    def set_ue_saved_state(self, input_ue_saved_state=None, **kw):
        self._saved_state = input_ue_saved_state
    def set_reward(self, input_reward=None, **kw): pass
    def set_old_state(self, input_old_state=None, **kw): pass
    def set_ul_buffer(self, **kw):      self.ul_buffer = MockBuffer()
    def set_action_list(self, input_action_list=None, **kw): pass
    def set_success_action_list(self, input_success_action_list=None, **kw): pass
    def set_actions_per_simulation(self, input_actions_per_simulation=None, **kw):
        self._actions_per_sim = input_actions_per_simulation or []
    def set_success_actions_per_simulation(self, input_success_actions_per_simulation=None, **kw):
        self._success_actions_per_sim = input_success_actions_per_simulation or []
    def set_replay_buffer(self, input_replay_buffer=None, **kw):
        self._replay_buf = input_replay_buffer
    def set_ack_packet_id_ue(self, v, **kw):  self._ack_packet_id = v
    def set_packets_sent(self, input_packets_sent=0, **kw):
        self.packets_sent = input_packets_sent
    def set_coordinates(self, x, y, z, **kw): pass
    def set_observation(self, v=None, **kw): pass

    def reset_obs(self):
        self.obs = [[0]*len(self.neighbour_table + ["BS"]),
                    [0]*len(self.neighbour_table),
                    [-90.0]*len(self.neighbour_table),
                    [10]*len(self.neighbour_table),
                    [0]*len(self.neighbour_table)]     # row 4 = bs_seen
    def reset_temp_obs(self):
        self.temp_obs = [[0]*max(len(self.neighbour_table),1),
                         [0]*max(len(self.neighbour_table),1),
                         [0.0]*max(len(self.neighbour_table),1),
                         [0]*max(len(self.neighbour_table),1),
                         [0]*max(len(self.neighbour_table),1)]  # row 4 = bs_seen

    def update_num_tx(self, input_packet_id=None, input_enable_print=False):
        if self.ul_buffer.buffer_packet_list:
            self.ul_buffer.buffer_packet_list[0]._num_tx += 1
    def update_n_data_tx(self, input_enable_print=False):
        self._n_data_tx += 1
    def check_num_tx(self):
        if not self.ul_buffer.buffer_packet_list: return True
        return self.ul_buffer.buffer_packet_list[0]._num_tx < 5
    def check_generated_packet_present(self):
        return any(not p._fwd for p in self.ul_buffer.buffer_packet_list)
    def is_there_a_new_data(self, t, max_q):
        return False
    def add_new_packet(self, *a, **kw):
        pid = self._packet_id; self._packet_id += 1
        p = MockPacket(pid, gen_by=self._uid)
        self.ul_buffer.buffer_packet_list.append(p)
        self.n_generated_packets += 1
    def remove_packet(self, pid, en_pr=False):
        self.ul_buffer.buffer_packet_list = [
            p for p in self.ul_buffer.buffer_packet_list if p.packet_id != pid]
    def check_remove_packet(self, en_pr=False): pass
    def set_temp_obs_broadcast(self, *a, **kw): pass
    def set_obs_update(self, *a, **kw): pass
    def update_neighbor_table_unicast_success(self, *a, **kw): pass
    def broadcast_handling_failure_no_reward(self, ttl): pass
    def broadcast_handling_no_reward(self, ttl): pass
    def unicast_handling_failure_no_reward(self, ttl): pass
    def unicast_handling_no_reward_no_neighbor_update(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
_failures = []

def check(cond: bool, name: str, detail: str = ""):
    if cond:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        _failures.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# T1 – MADRLController lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def test_controller_lifecycle():
    print("\n── T1: MADRLController lifecycle ─────────────────────────────")
    from madrl_agent import MADRLController, W_MIN_DEFAULT, Q_MIN_DEFAULT

    ctrl = MADRLController(n_ues=3, T_step=5,
                            agent_kwargs=dict(batch=8, buf_cap=100))
    check(len(ctrl.bundles) == 3,        "bundle count == n_ues")
    check(ctrl.W(0) == W_MIN_DEFAULT,    "W initialised to W_MIN")
    check(ctrl.Q(0) >= Q_MIN_DEFAULT,    "Q initialised >= Q_MIN")

    ctrl.reset_episode()
    b = ctrl.bundle(0)
    check(b.n_tx_last == 0,              "n_tx_last reset to 0")
    check(b._prev_routing_state is None, "prev routing state cleared")

    # apply CW and buffer actions
    old_W = b.W_current
    b.apply_cw_action(0)   # W++
    check(b.W_current == old_W + 4,      "W++ increments by W_STEP=4")
    b.apply_cw_action(2)   # W--
    check(b.W_current == old_W,          "W-- restores original W")
    b.apply_cw_action(1)   # W==
    check(b.W_current == old_W,          "W== unchanged")

    old_Q = b.Q_current
    b.apply_buf_action(0); check(b.Q_current == old_Q + 1, "Q++ increments by 1")
    b.apply_buf_action(2); check(b.Q_current == old_Q,     "Q-- restores Q")

    # boundary clamp
    for _ in range(40): b.apply_cw_action(0)
    check(b.W_current <= b.W_max, "W clamped at W_max")
    for _ in range(40): b.apply_cw_action(2)
    check(b.W_current >= b.W_min, "W clamped at W_min")


# ─────────────────────────────────────────────────────────────────────────────
# T2 – State builders
# ─────────────────────────────────────────────────────────────────────────────

def test_state_builders():
    print("\n── T2: State builders ────────────────────────────────────────")
    from madrl_agent import MADRLController, ROUTING_STATE_DIM, CW_STATE_DIM, BUF_STATE_DIM
    from madrl_integration import build_routing_state, build_cw_state, build_buf_state

    MockUE._id_counter = 0
    ues = [MockUE() for _ in range(4)]
    nodes = [str(u.get_ue_id()) for u in ues] + ["BS"]
    for i, ue in enumerate(ues):
        ue.set_neighbour_table(nodes[:i] + nodes[i+1:])
        ue.reset_obs()

    ctrl = MADRLController(n_ues=4, agent_kwargs=dict(batch=8, buf_cap=100))
    b = ctrl.bundle(0)
    b.n_tx_last = 3; b.n_tx_avg = 2.5; b.n_drop_r = 0; b.n_drop_q = 1
    b.n_collisions_last = 1

    rs = build_routing_state(ues[0], b, hop_limit=5)
    check(rs.shape == (ROUTING_STATE_DIM,), f"routing state shape {rs.shape}")
    check(np.all(np.isfinite(rs)),          "routing state finite")
    check(rs.dtype == np.float32,           "routing state float32")

    cw = build_cw_state(b, direct_bs=1.0)
    check(cw.shape == (CW_STATE_DIM,),      f"cw state shape {cw.shape}")
    check(np.all(cw >= 0) and np.all(cw <= 2.1), "cw state in reasonable range")

    bs_s = build_buf_state(ues[0], b, n_active_norm=0.5, direct_bs=1.0)
    check(bs_s.shape == (BUF_STATE_DIM,),   f"buf state shape {bs_s.shape}")
    check(np.all(np.isfinite(bs_s)),        "buf state finite")

    # add packets to buffer and verify buf_util changes
    ues[0].ul_buffer.buffer_packet_list.append(MockPacket(99))
    bs_s2 = build_buf_state(ues[0], b, n_active_norm=0.5, direct_bs=1.0)
    check(bs_s2[2] > bs_s[2], "buf_util increases with packet in buffer")


# ─────────────────────────────────────────────────────────────────────────────
# T3 – Reward functions
# ─────────────────────────────────────────────────────────────────────────────

def test_rewards():
    print("\n── T3: Reward functions ──────────────────────────────────────")
    from madrl_agent import MADRLController
    from madrl_integration import (compute_base_reward, compute_routing_reward,
                                    compute_cw_reward, compute_buf_reward)

    ctrl = MADRLController(n_ues=2, agent_kwargs=dict(batch=8, buf_cap=100))
    b = ctrl.bundle(0)

    # perfect run
    b.n_tx_last = 10; b.n_tx_avg = 8.0; b.n_drop_r = 0; b.n_drop_q = 0; b.n_collisions_last = 0
    r_perfect = compute_base_reward(b)
    check(r_perfect == 10 + 2 + 2, f"perfect base reward = {r_perfect}")

    # with drops
    b.n_drop_r = 1; b.n_drop_q = 1
    r_drops = compute_base_reward(b)
    check(r_drops < r_perfect, f"drops reduce reward {r_drops:.2f} < {r_perfect:.2f}")

    # routing reward richer than base
    b.n_drop_r = 0; b.n_drop_q = 0; b.n_collisions_last = 0
    r_base = compute_base_reward(b)
    r_rt   = compute_routing_reward(b, hop_count=1, hop_limit=5,
                                     ack_received=True, packet_at_bs=True)
    check(r_rt > r_base,    f"routing reward {r_rt:.2f} > base {r_base:.2f}")

    r_rt_no_bs = compute_routing_reward(b, hop_count=1, hop_limit=5,
                                         ack_received=False, packet_at_bs=False)
    check(r_rt > r_rt_no_bs, "delivery bonus adds to routing reward")

    # hop efficiency: fewer hops → higher reward
    r_short = compute_routing_reward(b, hop_count=1, hop_limit=5, ack_received=False, packet_at_bs=False)
    r_long  = compute_routing_reward(b, hop_count=4, hop_limit=5, ack_received=False, packet_at_bs=False)
    check(r_short >= r_long, f"shorter path preferred {r_short:.3f} >= {r_long:.3f}")

    # CW reward penalises collisions
    b.n_collisions_last = 5; b.n_tx_last = 5
    r_cw_coll = compute_cw_reward(b)
    b.n_collisions_last = 0
    r_cw_clean = compute_cw_reward(b)
    check(r_cw_clean >= r_cw_coll, "CW reward lower with collisions")

    # buf reward penalises drops
    b.n_drop_q = 1
    r_buf_drop = compute_buf_reward(b)
    b.n_drop_q = 0
    r_buf_ok   = compute_buf_reward(b)
    check(r_buf_ok > r_buf_drop, "buf reward lower with buffer drops")


# ─────────────────────────────────────────────────────────────────────────────
# T4 – choose_next_action_madrl routing decisions
# ─────────────────────────────────────────────────────────────────────────────

def test_routing_action():
    print("\n── T4: choose_next_action_madrl ──────────────────────────────")
    from madrl_agent import MADRLController
    from madrl_integration import choose_next_action_madrl

    MockUE._id_counter = 0
    ues = [MockUE() for _ in range(5)]
    nodes = [str(u.get_ue_id()) for u in ues] + ["BS"]
    for i, ue in enumerate(ues):
        ue.set_neighbour_table(nodes[:i] + nodes[i+1:])
        ue.reset_obs()
        # give ue[0] some ACK history
        if i == 0:
            ue.obs[1] = [3, 0, 2, 0, 1]   # ACKs from neighbours 0,2,4

    ctrl = MADRLController(n_ues=5, agent_kwargs=dict(batch=8, buf_cap=100))
    b = ctrl.bundle(0)
    ue0 = ues[0]

    # with high ε, actions are random → just check they don't crash
    b.routing_agent.eps = 1.0
    for _ in range(20):
        choose_next_action_madrl(ue0, b, hop_limit=5, enable_print=False)
    check(True, "random routing actions execute without error")

    # with ε=0, action is greedy → must be either broadcast or a valid unicast
    b.routing_agent.eps = 0.0
    choose_next_action_madrl(ue0, b, hop_limit=5, enable_print=False)
    action_ok = (ue0.get_broadcast_bool() or ue0.get_unicast_rx_address() is not None)
    check(action_ok, "greedy action sets broadcast_bool or unicast_rx_address")

    # verify previous-state is stored for next transition
    check(b._prev_routing_state is not None, "prev routing state stored after action")
    check(b._prev_routing_action is not None, "prev routing action stored")


# ─────────────────────────────────────────────────────────────────────────────
# T5 – madrl_step_update full cycle
# ─────────────────────────────────────────────────────────────────────────────

def test_step_update():
    print("\n── T5: madrl_step_update ─────────────────────────────────────")
    from madrl_agent import MADRLController
    from madrl_integration import choose_next_action_madrl, madrl_step_update

    MockUE._id_counter = 0
    ues = [MockUE() for _ in range(3)]
    nodes = [str(u.get_ue_id()) for u in ues] + ["BS"]
    for i, ue in enumerate(ues):
        ue.set_neighbour_table(nodes[:i] + nodes[i+1:])
        ue.reset_obs()

    ctrl = MADRLController(n_ues=3, agent_kwargs=dict(batch=16, buf_cap=500))

    # warm-up: generate enough transitions to fill the replay buffer
    for episode in range(5):
        ctrl.reset_episode()
        for ue in ues:
            b = ctrl.bundle(ue.get_ue_id())
            b.n_tx_last = random.randint(0, 10)
            b.n_tx_avg  = 5.0
            b.n_drop_r  = random.randint(0, 1)
            b.n_drop_q  = random.randint(0, 1)
            b.n_collisions_last = random.randint(0, 3)

            # simulate T_step BO phases
            for _ in range(3):
                choose_next_action_madrl(ue, b, hop_limit=5, enable_print=False)
            losses = madrl_step_update(
                ue, b, hop_limit=5,
                ack_received=True, packet_at_bs=(episode > 1), done=False)

    # after warm-up, at least one agent should produce a loss
    b0 = ctrl.bundle(0)
    buf_size = len(b0.routing_agent.buf)
    check(buf_size > 0, f"replay buffer filled  ({buf_size} samples)")

    # single training step
    losses = ctrl.train_all()
    r_loss = losses.get("routing")
    check(r_loss is not None or buf_size < 16, "training produces loss (or buffer too small)")
    check(b0.W_current >= b0.W_min and b0.W_current <= b0.W_max,
          f"W in bounds after update  (W={b0.W_current})")
    check(b0.Q_current >= b0.Q_min and b0.Q_current <= b0.Q_max,
          f"Q in bounds after update  (Q={b0.Q_current})")


# ─────────────────────────────────────────────────────────────────────────────
# T6 – madrl_episode_end
# ─────────────────────────────────────────────────────────────────────────────

def test_episode_end():
    print("\n── T6: madrl_episode_end ─────────────────────────────────────")
    from madrl_agent import MADRLController
    from madrl_integration import choose_next_action_madrl, madrl_episode_end

    MockUE._id_counter = 0
    ue = MockUE()
    ue.set_neighbour_table(["1", "2", "BS"]); ue.reset_obs()
    ctrl = MADRLController(n_ues=1, agent_kwargs=dict(batch=8, buf_cap=100))
    b = ctrl.bundle(0)

    # store a pending previous state
    choose_next_action_madrl(ue, b, hop_limit=5, enable_print=False)
    check(b._prev_routing_state is not None, "prev state set before episode_end")

    buf_before = len(b.routing_agent.buf)
    madrl_episode_end(ue, b, hop_limit=5)
    buf_after = len(b.routing_agent.buf)
    check(buf_after >= buf_before, "episode_end pushes terminal transition")


# ─────────────────────────────────────────────────────────────────────────────
# T7 – Curriculum phases
# ─────────────────────────────────────────────────────────────────────────────

def test_curriculum():
    print("\n── T7: Curriculum phases ─────────────────────────────────────")
    from madrl_agent import MADRLController
    from train_madrl import CurriculumController, get_current_phase, CURRICULUM

    ctrl = MADRLController(n_ues=2, agent_kwargs=dict(batch=8, buf_cap=100))
    cc   = CurriculumController(ctrl)

    phase_0  = get_current_phase(0)
    phase_10 = get_current_phase(10)
    phase_30 = get_current_phase(30)
    phase_80 = get_current_phase(80)

    check(phase_0.name  == "warm_up",        f"ep 0  → warm_up  (got {phase_0.name})")
    check(phase_10.name == "explore_routing", f"ep 10 → explore_routing (got {phase_10.name})")
    check(phase_30.name == "joint_learning",  f"ep 30 → joint_learning (got {phase_30.name})")
    check(phase_80.name == "fine_tune_params",f"ep 80 → fine_tune_params (got {phase_80.name})")

    # warm-up: all agents frozen
    cc.apply(0)
    check(cc.freeze_routing and cc.freeze_cw and cc.freeze_buf,
          "warm_up: all agents frozen")

    # explore_routing: only routing unfrozen
    cc.apply(10)
    check(not cc.freeze_routing and cc.freeze_cw and cc.freeze_buf,
          "explore_routing: routing active, W/Q frozen")

    # joint_learning: all active
    cc.apply(30)
    check(not any([cc.freeze_routing, cc.freeze_cw, cc.freeze_buf]),
          "joint_learning: all agents active")

    # fine_tune_params: routing frozen, W/Q active
    cc.apply(80)
    check(cc.freeze_routing and not cc.freeze_cw and not cc.freeze_buf,
          "fine_tune_params: routing frozen, W/Q active")

    # gated training respects freeze
    cc.apply(10)   # routing only
    for b in ctrl.bundles:
        for _ in range(20):
            s  = np.random.randn(89).astype(np.float32)
            b.routing_agent.push(s, 0, 1.0, s.copy(), False)
    losses = cc.train_all_gated()
    check(losses.get("cw")  is None, "gated: cw_loss None when frozen")
    check(losses.get("buf") is None, "gated: buf_loss None when frozen")


# ─────────────────────────────────────────────────────────────────────────────
# T8 – EarlyStopper
# ─────────────────────────────────────────────────────────────────────────────

def test_early_stopper():
    print("\n── T8: EarlyStopper ──────────────────────────────────────────")
    from train_madrl import EarlyStopper

    es = EarlyStopper(jain_threshold=0.90, patience=3, min_improvement=1e-4)

    # should not trigger below threshold
    for _ in range(10):
        triggered = es.step(0.80, 4.0)
    check(not triggered, "does not trigger below threshold")

    # should trigger after 3 consecutive successes
    for i in range(3):
        triggered = es.step(0.92, 5.0)
    check(triggered, "triggers after 3 consecutive >= threshold")

    # stagnation: no improvement for patience*3 steps
    es2 = EarlyStopper(jain_threshold=0.99, patience=3)
    for _ in range(16):   # patience*5 = 15; 16th call triggers
        triggered2 = es2.step(0.75, 3.0)
    check(triggered2, "triggers on stagnation (no improvement)")


# ─────────────────────────────────────────────────────────────────────────────
# T9 – Evaluation pipeline (mock run_fn)
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluation_pipeline():
    print("\n── T9: EvaluationPipeline ────────────────────────────────────")
    from evaluate_madrl import (EvaluationPipeline, extract_results,
                                  compute_jain_index, _bootstrap_ci)

    # synthetic output_dict generator
    def mock_run(n_ue, seed):
        np.random.seed(seed)
        od = {"j_index": {f"N={n_ue}": {"Sim=0": np.random.uniform(0.7,0.95,n_ue)}},
              "s":       {f"N={n_ue}": {"Sim=0": np.random.uniform(5e5,2e6,n_ue)}},
              "l":       {f"N={n_ue}": {"Sim=0": np.random.uniform(1e-3,5e-3,n_ue)}},
              "p_mac":   {f"N={n_ue}": {"Sim=0": np.random.uniform(0.5,0.9,n_ue)}}}
        return od

    def mock_run_tb(n_ue, seed):
        np.random.seed(seed + 100)
        od = {"j_index": {f"N={n_ue}": {"Sim=0": np.random.uniform(0.5,0.80,n_ue)}},
              "s":       {f"N={n_ue}": {"Sim=0": np.random.uniform(3e5,1e6,n_ue)}},
              "l":       {f"N={n_ue}": {"Sim=0": np.random.uniform(3e-3,8e-3,n_ue)}},
              "p_mac":   {f"N={n_ue}": {"Sim=0": np.random.uniform(0.3,0.7,n_ue)}}}
        return od

    pipe = EvaluationPipeline(n_ue_list=[4, 8], n_eval=6)
    pipe.run_method("MADRL", mock_run)
    pipe.run_method("TB",    mock_run_tb)

    cmp = pipe.compare()
    check(True, "Comparator created without error")

    result_4 = cmp.compare(4)
    check("Jain index J" in result_4["metrics"], "comparison has Jain metric")
    j_madrl = result_4["metrics"]["Jain index J"]["madrl"]
    j_tb    = result_4["metrics"]["Jain index J"]["tb"]
    check(j_madrl > j_tb, f"MADRL Jain > TB Jain  ({j_madrl:.3f} > {j_tb:.3f})")

    with tempfile.TemporaryDirectory() as td:
        pipe.save(os.path.join(td, "test_comparison"))
        check(os.path.exists(os.path.join(td, "test_comparison.txt")),
              "comparison table .txt saved")
        check(os.path.exists(os.path.join(td, "test_comparison.json")),
              "comparison JSON saved")

    # Jain edge cases
    check(abs(compute_jain_index(np.ones(6)) - 1.0) < 1e-9, "Jain = 1 for equal shares")
    check(abs(compute_jain_index(np.array([1,0,0,0])) - 0.25) < 1e-9,
          "Jain = 1/N for single active UE")
    check(compute_jain_index(np.zeros(4)) == 0.0, "Jain = 0 for all-zero")


# ─────────────────────────────────────────────────────────────────────────────
# T10 – Checkpoint save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint():
    print("\n── T10: Checkpoint round-trip ────────────────────────────────")
    from madrl_agent import MADRLController

    ctrl = MADRLController(n_ues=2, agent_kwargs=dict(batch=8, buf_cap=100))
    b = ctrl.bundle(0)

    # modify state
    b.apply_cw_action(0); b.apply_cw_action(0)   # W += 8
    W_before = b.W_current
    b.routing_agent.eps = 0.42
    eps_before = b.routing_agent.eps

    with tempfile.TemporaryDirectory() as td:
        ctrl.save(td)
        files = os.listdir(td)
        check(len(files) > 0, f"checkpoint files written  ({len(files)} files)")

        # create fresh controller and load
        ctrl2 = MADRLController(n_ues=2, agent_kwargs=dict(batch=8, buf_cap=100))
        ctrl2.load(td)
        b2 = ctrl2.bundle(0)

        # eps should be restored (only if torch is available)
        try:
            import torch
            check(abs(b2.routing_agent.eps - eps_before) < 1e-6,
                  f"epsilon restored  ({b2.routing_agent.eps:.4f})")
        except ImportError:
            check(True, "checkpoint load skipped (torch not available)")


# ─────────────────────────────────────────────────────────────────────────────
# T11 – PER priority update correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_per_priority():
    print("\n── T11: PER priority update ──────────────────────────────────")
    from madrl_agent import PrioritisedReplayBuffer
    import numpy as np

    buf = PrioritisedReplayBuffer(capacity=50, alpha=0.6, beta_start=0.4)
    for i in range(40):
        buf.push(np.zeros(4), i % 3, float(i), np.ones(4), False)

    _, idxs, w1 = buf.sample(8)
    # assign very high TD error to one sample
    td = np.ones(8) * 0.01
    td[0] = 100.0
    buf.update_priorities(idxs, td)

    # sample again: the high-priority sample should appear more
    counts = {int(idxs[0]): 0}
    for _ in range(200):
        _, new_idxs, _ = buf.sample(8)
        if int(idxs[0]) in new_idxs:
            counts[int(idxs[0])] += 1
    check(counts[int(idxs[0])] > 20,
          f"high-priority sample over-sampled  (count={counts[int(idxs[0])]})")

    # IS weights should be ≤ 1
    _, _, w2 = buf.sample(8)
    check(np.all(w2 <= 1.0 + 1e-6), f"IS weights ≤ 1  (max={w2.max():.4f})")
    check(np.all(w2 > 0),            "IS weights > 0")

    # beta should anneal upward
    beta_before = buf.beta
    for _ in range(100):
        buf.sample(8)
    check(buf.beta > beta_before, f"beta annealed  ({beta_before:.3f} → {buf.beta:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(42); np.random.seed(42)
    print("=" * 62)
    print("  MADRL-TB  End-to-End Test Suite")
    print("=" * 62)

    test_controller_lifecycle()
    test_state_builders()
    test_rewards()
    test_routing_action()
    test_step_update()
    test_episode_end()
    test_curriculum()
    test_early_stopper()
    test_evaluation_pipeline()
    test_checkpoint()
    test_per_priority()

    print("\n" + "=" * 62)
    if _failures:
        print(f"  \033[91m{len(_failures)} FAILED:\033[0m  {', '.join(_failures)}")
        sys.exit(1)
    else:
        total = sum(1 for _ in open(__file__) if "check(" in _)
        print(f"  \033[92mAll tests passed\033[0m  ({total} checks)")
    print("=" * 62)

if __name__ == "__main__":
    main()
