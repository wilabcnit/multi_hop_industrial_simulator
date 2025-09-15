import math
import psutil
import os
import sys
import numpy as np
from scipy import constants
import random
import json
from math import log10
from copy import deepcopy
from datetime import datetime

from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.network.ue import Ue

from multi_hop_industrial_simulator.utils.plot_data import plot_factory, plot_snr, plot_scenario_2d, write_data

from multi_hop_industrial_simulator.utils.check_success import check_collision
from multi_hop_industrial_simulator.utils.check_success import check_collision_bs

from multi_hop_industrial_simulator.utils.compute_distance_m import compute_distance_m
from multi_hop_industrial_simulator.utils.compute_propagation_delays import compute_propagation_delays
from multi_hop_industrial_simulator.utils.compute_simulation_outputs import compute_simulator_outputs
from multi_hop_industrial_simulator.utils.compute_snr_threshold import compute_snr_threshold_db

from multi_hop_industrial_simulator.utils.instantiate_bs import instantiate_bs
from multi_hop_industrial_simulator.utils.read_input_file import read_input_file
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs
from multi_hop_industrial_simulator.env.geometry import Geometry
from multi_hop_industrial_simulator.env.distribution import Distribution
from multi_hop_industrial_simulator.env.machine import Machine
from multi_hop_industrial_simulator.utils.instantiate_ues import instantiate_ues
from multi_hop_industrial_simulator.utils.compute_simulator_tick_duration import compute_simulator_tick_duration
from multi_hop_industrial_simulator.channel_models.THz_channel import THzChannel
from multi_hop_industrial_simulator.utils.check_for_neighbours import check_for_neighbours

from multi_hop_industrial_simulator.utils.set_ues_los_condition import set_ues_los_condition

from collections import deque

from multi_hop_industrial_simulator.utils.choose_next_action import choose_next_action_tb_no_RL

import gc
gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

""" 
    Utility functions for modifying the UEs states
"""


def go_in_idle(input_ue: Ue, current_tick: int, input_enable_print: bool):
    input_ue.set_state(input_state='IDLE')
    input_ue.set_state_duration(input_ticks=ue.get_next_packet_generation_instant())
    # Compute energy spent
    ue.energy_consumed += power_idle * (ue.get_state_duration() - t) * simulator_tick_duration_s
    if input_enable_print:
        print('UE ', input_ue.get_ue_id(), ' goes in IDLE from t = ', current_tick, ' until t = ',
              input_ue.get_state_duration())


def get_backoff_duration(input_ue: Ue, input_contention_window_int: int, input_t_backoff_tick: int,
                         input_max_prop_delay_tick: int):
    # The BO duration of a given UE is a function of its own retransmission pattern
    exp_backoff_factor = pow(2, input_ue.get_ul_buffer().get_first_packet().get_num_tx())
    delay_tick = random.randint(1, exp_backoff_factor * input_contention_window_int)
    data_duration_tick = input_ue.get_data_duration_tick()
    if star_topology:
        return delay_tick * input_t_backoff_tick
    else:
        return data_duration_tick + input_max_prop_delay_tick + delay_tick * input_t_backoff_tick


def go_in_backoff(input_ue: Ue, current_tick: int, input_backoff_duration_tick: int, input_enable_print: bool):
    if input_ue.get_state() == 'WAIT_ACK':
        input_ue.ticks_in_WAIT_ACK.append(current_tick - input_ue.get_state_starting_tick())
    elif input_ue.get_state() == 'TX_ACK':
        input_ue.ticks_in_TX_ACK.append(current_tick - input_ue.get_state_starting_tick())
    input_ue.set_state(input_state='BO')
    input_ue.update_state_duration(input_ticks=input_backoff_duration_tick)
    input_ue.set_state_starting_tick(input_tick=current_tick)
    input_ue.set_state_final_tick(input_tick=input_ue.get_state_duration())
    # Compute energy spent
    ue.energy_consumed += power_bo * (ue.get_state_duration() - t) * simulator_tick_duration_s
    if input_enable_print:
        print('UE ', input_ue.get_ue_id(), ' goes in BO from t = ', current_tick, ' until t = ',
              input_ue.get_state_duration())


def go_in_tx_data(input_ue: Ue, current_tick: int, input_enable_print: bool):
    if input_ue.get_state() == 'BO':
        input_ue.ticks_in_BO.append(current_tick - input_ue.get_state_starting_tick())
    elif input_ue.get_state() == 'TX_ACK':
        input_ue.ticks_in_TX_ACK.append(current_tick - input_ue.get_state_starting_tick())
    data_duration_tick = 0  # Duration of its last buffered data
    data_size_bytes = 0
    # Enlarge the duration of the data transmission if the UE has to also forward some receptions from other UEs
    ue_packet_list = input_ue.get_updated_packet_list()
    packet_ids = np.empty(len(ue_packet_list), dtype=int)
    packet_num_tx = np.empty(len(ue_packet_list), dtype=int)
    if input_enable_print:
        packet_ids[0] = input_ue.get_ul_buffer().get_first_packet().get_id()
        packet_num_tx[0] = input_ue.get_ul_buffer().get_first_packet().get_num_tx()
    # creation of a buffer with all the 'Packets' sent from that UE to the BS/UEs
    ue.buffer_packet_sent.clear()
    ue.end_data_tx = current_tick
    index = 0
    for packet in ue_packet_list:
        if packet.get_retransmission_packets() is False:
            packet.hop_count += 1
            if enable_print:
                print('UE  ', input_ue.get_ue_id(), 'TX Packet with ID ', packet.get_id(), ' has a hop count = ',
                      packet.hop_count)
            ue.packets_sent += 1

        if packet.get_data_to_be_forwarded_bool() is True:
            ue.set_relay_bool(relay_bool=True)
            packet.set_data_unicast(input_data_unicast=False)
            # NOTE: This UE can receive data of different size from UEs belonging to other traffic types

            data_duration_tick += packet.get_packet_duration_tick()
            data_size_bytes += packet.get_size()

            if input_enable_print:
                packet_ids[index] = packet.get_id()
                packet_num_tx[index] = packet.get_num_tx()
        else:
            packet.set_data_unicast(input_data_unicast=False)
            data_duration_tick += packet.get_packet_duration_tick()
            data_size_bytes += packet.get_size()
            if input_enable_print:
                packet_ids[index] = packet.get_id()
                packet_ids[index] = packet.get_id()
                packet_num_tx[index] = packet.get_num_tx()
        ue.buffer_packet_sent.append(packet)
        index += 1
    ue.end_data_tx += data_duration_tick
    input_ue.set_state(input_state='TX_DATA')
    input_ue.update_state_duration(input_ticks=data_duration_tick)
    input_ue.set_state_starting_tick(input_tick=current_tick)
    input_ue.set_state_final_tick(input_tick=input_ue.get_state_duration())

    if input_enable_print:
        print('UE ', input_ue.get_ue_id(), ' goes in TX DATA from t = ', current_tick, ' until t = ',
              input_ue.get_state_duration())
        for packet_id, packet_num_tx in zip(packet_ids, packet_num_tx):
            print('UE ', input_ue.get_ue_id(), ' is going to transmit packet with ID ', packet_id,
                  ' that has a number of tx attempts = ', packet_num_tx, " at t = ", current_tick)

    # Compute energy spent
    ue.energy_consumed += power_tx * (ue.get_state_duration() - t) * simulator_tick_duration_s

    return data_size_bytes


def go_in_tx_ack(input_ue: Ue, current_tick: int, input_ack_duration_tick: int, input_enable_print: bool):
    if input_ue.get_state() == 'BO':
        input_ue.ticks_in_BO.append(current_tick - input_ue.get_state_starting_tick())
    elif input_ue.get_state() == 'WAIT_ACK':
        input_ue.ticks_in_WAIT_ACK.append(current_tick - input_ue.get_state_starting_tick())
    input_ue.set_state(input_state='TX_ACK')
    input_ue.update_state_duration(input_ticks=input_ack_duration_tick)
    input_ue.set_state_starting_tick(input_tick=current_tick)
    input_ue.set_state_final_tick(input_tick=input_ue.get_state_duration())
    if input_enable_print:
        print('UE ', input_ue.get_ue_id(), ' goes in TX ACK from t = ', current_tick, ' until t = ',
              input_ue.get_state_duration())

    # Compute energy spent
    ue.energy_consumed += power_tx * (ue.get_state_duration() - t) \
                          * simulator_tick_duration_s


def go_in_tx_ack_bs(input_bs: BS, current_tick: int, input_ack_duration_tick: int, input_enable_print: bool):
    input_bs.set_state(input_state='TX_ACK')
    input_bs.update_state_duration(input_ticks=input_ack_duration_tick)
    input_bs.set_start_tx_ack(input_start_tx_ack=current_tick)
    input_bs.set_end_tx_ack(input_end_tx_ack=input_bs.get_state_duration())
    if input_enable_print:
        print('The BS goes in TX_ACK from t =', current_tick, ' to t = ', input_bs.get_state_duration())


def go_in_wait_ack(input_ue: Ue, current_tick: int, input_wait_ack_duration_tick: int,
                   input_enable_print: bool = True):
    if input_ue.get_state() == 'TX_DATA':
        input_ue.ticks_in_TX_DATA.append(current_tick - input_ue.get_state_starting_tick())
    elif input_ue.get_state() == 'TX_ACK':
        input_ue.ticks_in_TX_ACK.append(current_tick - input_ue.get_state_starting_tick())
    input_ue.set_state(input_state='WAIT_ACK')
    input_ue.update_state_duration(input_ticks=input_wait_ack_duration_tick)
    input_ue.set_state_starting_tick(input_tick=current_tick)
    input_ue.set_state_final_tick(input_tick=input_ue.get_state_duration())
    if input_enable_print:
        print('UE ', input_ue.get_ue_id(), ' goes in WAIT_ACK from t = ', current_tick, ' until t = ',
              input_ue.get_state_duration())

    # Compute energy spent
    ue.energy_consumed += power_ack * (ue.get_state_duration() - t) * simulator_tick_duration_s


def go_rx_ack_bs(input_bs: BS, current_tick: int, input_rx_duration_tick: int, input_enable_print: bool = True):
    input_bs.set_state(input_state='RX')
    input_bs.set_state_duration(input_ticks=input_rx_duration_tick)


def create_simulator_timing_structure(input_n_ue: int, input_simulation_duration_tick: int):
    output_simulator_timing_structure = {}
    for i in range(0, input_n_ue):
        ue_key = 'UE_' + str(i)
        ue_value = {'DATA_RX': {}, 'ACK_RX': {}}
        for j in range(0, input_n_ue):
            if j != i:
                ue_value['DATA_RX']['UE_' + str(j)] = np.array([[input_simulation_duration_tick + 1] * 4],
                                                               dtype=int)  # starting tick, ending tick, size
                ue_value['ACK_RX']['UE_' + str(j)] = np.array([[input_simulation_duration_tick + 1] * 4],
                                                              dtype=int)  # starting tick, ending tick, size
        ue_value['DATA_RX']['BS'] = np.array([[input_simulation_duration_tick + 1] * 4], dtype=int)
        # starting tick, ending tick, size
        ue_value['ACK_RX']['BS'] = np.array([[input_simulation_duration_tick + 1] * 4], dtype=int)
        # starting tick, ending tick, size
        output_simulator_timing_structure[ue_key] = ue_value

    bs_key = 'BS'
    bs_value = {'DATA_RX': {}, 'ACK_RX': {}}
    for ue_index in range(input_n_ue):
        bs_value['DATA_RX']['UE_' + str(ue_index)] = np.array([[input_simulation_duration_tick + 1] * 4],
                                                              dtype=int)  # starting tick, ending tick, size
        bs_value['ACK_RX']['UE_' + str(ue_index)] = np.array([[input_simulation_duration_tick + 1] * 4],
                                                             dtype=int)  # starting tick, ending tick, size
    output_simulator_timing_structure[bs_key] = bs_value
    return output_simulator_timing_structure


def reset_simulator_timing_structure(output_simulator_timing_structure: dict, input_simulation_duration_tick: int):
    for key_ext, value_ext in output_simulator_timing_structure.items():
        for key_int, value_int in value_ext['DATA_RX'].items():
            value_ext['DATA_RX'][key_int] = np.array([[input_simulation_duration_tick + 1] * 4],
                                                     dtype=int)  # starting tick, ending tick, size
        for key_int, value_int in value_ext['ACK_RX'].items():
            value_ext['ACK_RX'][key_int] = np.array([[input_simulation_duration_tick + 1] * 4],
                                                    dtype=int)  # starting tick, ending tick, size


def insert_item_in_timing_structure(input_simulator_timing_structure: dict, input_starting_tick: int,
                                    input_final_tick: int, input_third_field: int, input_fourth_field: int,
                                    # Size for data, and UE ID for ACK
                                    input_tx_key: str, input_type_key: str, input_rx_key: str):
    new_addition = np.array([input_starting_tick, input_final_tick, input_third_field, input_fourth_field])
    input_simulator_timing_structure[input_rx_key][input_type_key][input_tx_key] = (
        np.vstack([input_simulator_timing_structure[input_rx_key][input_type_key][input_tx_key], new_addition]))


def remove_item_in_timing_structure(input_simulator_timing_structure: dict, input_tx_key: str, input_type_key: str,
                                    input_rx_key: str):
    # Always remove the second row, that is, the most recent reception from this specific TX
    input_simulator_timing_structure[input_rx_key][input_type_key][input_tx_key] = np.delete(
        input_simulator_timing_structure[input_rx_key][input_type_key][input_tx_key], 1, axis=0)


def find_data_rx_times_tick(input_simulator_timing_structure: dict, input_ue_id: int, current_tick: int):
    output_data_rx_at_ue_starting_tick = None
    output_data_rx_at_ue_ending_tick = None
    output_data_rx_at_ue_size_bytes = list()
    output_data_rx_at_ue_packet_id = list()
    ue_id = list()
    index = 0
    # need to return a list of UEs, packet IDs and packet sizes otherwise it is not possible to detect
    #  correctly the reception of two or more packets perfectly overlapped
    for ue_key_ext in input_simulator_timing_structure.keys():
        if ue_key_ext == f'UE_{input_ue_id}':
            for ue_key_int in input_simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                # Find the row with the minimum rx time
                min_index = np.argmin(input_simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][:, 1])
                min_row = input_simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][min_index, :]
                if min_row[1] == current_tick:
                    index += 1
                    output_data_rx_at_ue_starting_tick = min_row[0]
                    output_data_rx_at_ue_ending_tick = min_row[1]
                    output_data_rx_at_ue_size_bytes.append(min_row[2])
                    output_data_rx_at_ue_packet_id.append(min_row[3])
                    ue_id.append(int(ue_key_int[3:]))
            break

    return output_data_rx_at_ue_starting_tick, output_data_rx_at_ue_ending_tick, output_data_rx_at_ue_size_bytes, \
        output_data_rx_at_ue_packet_id, ue_id


def find_data_rx_times_at_bs_tick(input_simulator_timing_structure: dict, current_tick: int):
    output_data_rx_at_bs_starting_tick = None
    output_data_rx_at_bs_ending_tick = None
    output_data_rx_packet_id = list()
    ue_id = list()

    for ue_key_int in input_simulator_timing_structure['BS']['DATA_RX'].keys():
        min_index = np.argmin(input_simulator_timing_structure['BS']['DATA_RX'][ue_key_int][:, 1])
        min_row = input_simulator_timing_structure['BS']['DATA_RX'][ue_key_int][min_index, :]
        if min_row[1] == current_tick:
            output_data_rx_at_bs_starting_tick = min_row[0]
            output_data_rx_at_bs_ending_tick = min_row[1]
            output_data_rx_at_bs_size_bytes = min_row[2]
            output_data_rx_packet_id.append(min_row[3])
            # Pick-up the UE ID so that the traffic type can be inferred
            ue_id.append(int(ue_key_int[3:]))
            # break

    return output_data_rx_at_bs_starting_tick, output_data_rx_at_bs_ending_tick, output_data_rx_packet_id, ue_id

def find_ack_rx_times_at_bs_tick(input_simulator_timing_structure: dict, current_tick: int):
    output_ack_rx_at_bs_starting_tick = None
    output_ack_rx_at_bs_ending_tick = None
    output_ack_rx_at_bs_recipient_id_int = list()
    output_ack_rx_transmitter_id_str = list()
    output_ack_rx_packet_id = list()
    n_ack_rx_simultaneously = None
    # need to return a list of UEs, list of ACKs for given packet IDs and otherwise it is not possible to
    #  detect correctly the reception of two or more ACKs perfectly overlapped
    for ue_key_ext in input_simulator_timing_structure.keys():
        if ue_key_ext == 'BS':
            for ue_key_int in input_simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                # Check if an ACK reception ends at current tick and return the starting and ending tick
                min_index = np.argmin(input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][:, 1])
                min_row = input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][min_index, :]
                min_tick_rx = np.min(input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][:, 1])
                if min_row[1] == current_tick:
                    output_ack_rx_at_bs_starting_tick = min_row[0]
                    output_ack_rx_at_bs_ending_tick = min_row[1]

                    # Pick-up the ID of the transmitter of the ACk
                    if ue_key_int not in output_ack_rx_transmitter_id_str:
                        output_ack_rx_at_bs_recipient_id_int.append(min_row[2])
                        output_ack_rx_transmitter_id_str.append(ue_key_int)
                    n_ack_rx_simultaneously = np.count_nonzero(
                        input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][:, 1] == min_tick_rx)

                    if n_ack_rx_simultaneously == 1:
                        # simultaneously from the same UE
                        output_ack_rx_packet_id.append(min_row[3])
                    else:
                        index = 0
                        for packet_id in input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][:, 3]:
                            if min_tick_rx == input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][index, 1]:
                                output_ack_rx_packet_id.append(packet_id)
                            index += 1

            break

    return (output_ack_rx_at_bs_starting_tick, output_ack_rx_at_bs_ending_tick, output_ack_rx_at_bs_recipient_id_int,
            output_ack_rx_packet_id, output_ack_rx_transmitter_id_str, n_ack_rx_simultaneously)

def find_ack_rx_times_tick(input_simulator_timing_structure: dict, input_ue_id: int, current_tick: int):
    output_ack_rx_at_ue_starting_tick = None
    output_ack_rx_at_ue_ending_tick = None
    output_ack_rx_packet_id = list() # list of ack packet id
    output_ack_rx_sources = list() # list of ack sources
    output_ack_rx_dest = list() # list of ack dest

    # need to return a list of UEs, list of ACKs for given packet IDs and otherwise it is not possible to
    #  detect correctly the reception of two or more ACKs perfectly overlapped
    for ue_key_ext in input_simulator_timing_structure.keys():
        if ue_key_ext == f'UE_{input_ue_id}' or ue_key_ext == 'BS':
            for ue_key_int in input_simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                # Check if an ACK reception ends at current tick and return the starting and ending tick
                min_index = np.argmin(input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][:, 1])
                min_row = input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][min_index, :]
                min_tick_rx = np.min(input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][:, 1])
                if min_row[1] == current_tick:
                    output_ack_rx_at_ue_starting_tick = min_row[0]
                    output_ack_rx_at_ue_ending_tick = min_row[1]

                    for index in range(len(input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int])):
                        if min_tick_rx == input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][index, 1]:
                            row = input_simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][index, :]
                            output_ack_rx_dest.append(row[2])
                            output_ack_rx_sources.append(ue_key_int)
                            output_ack_rx_packet_id.append(row[3])

            break

    return (output_ack_rx_at_ue_starting_tick, output_ack_rx_at_ue_ending_tick, output_ack_rx_sources,
            output_ack_rx_dest, output_ack_rx_packet_id)


sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

# Constants
c = constants.speed_of_light

# Pick-up inputs
inputs = read_inputs('inputs.yaml')
scenario_name = inputs.get('scenario').get('name')
scenario_file_name = inputs.get('scenario').get('input_file_name')
initial_seed = inputs.get('simulation').get('initial_seed')
final_seed = inputs.get('simulation').get('final_seed')
ue_distribution_type = inputs.get('ue').get('ue_spatial_distribution')
ue_distribution = inputs.get('ue').get('ue_spatial_distribution')
initial_number_of_ues = inputs.get('simulation').get('initial_number_of_ues')
step_number_of_ues = inputs.get('simulation').get('step_number_of_ues')
final_number_of_ues = inputs.get('simulation').get('final_number_of_ues')
input_g_node_bs = inputs['bs']['number_of_bs']
ue_starting_state = inputs.get('ue').get('ue_starting_state')
bs_starting_state = inputs.get('bs').get('bs_starting_state')
payload = inputs.get('traffic_rt').get('payload')
period = inputs.get('traffic_rt').get('period')
on_collection_nrt_s = inputs.get('traffic_nrt').get('collection_on_duration')
standby_collection_nrt_s = inputs.get('traffic_nrt').get('collection_standby_duration')
optimization_nrt_s = inputs.get('traffic_nrt').get('optimization_duration')
ack_size_bytes = inputs.get('aloha_protocol').get('ack_size_bytes')
contention_window_int = inputs.get('aloha_protocol').get('contention_window_int')
max_n_retx_per_packet = inputs.get('aloha_protocol').get('max_n_retx_per_packet')
max_n_packets_to_be_forwarded = inputs.get('aloha_protocol').get('max_n_packets_to_be_forwarded')
bit_rate_gbits = inputs.get('radio').get('bit_rate_gbits')
apply_fading = inputs.get('channel').get('apply_fading')
n_simulations = inputs.get('simulation').get('n_simulations')
p_succ_phy = inputs.get('channel').get('p_succ_phy')
simulation_time_s = inputs.get('simulation').get('tot_simulation_time_s')
enable_print = inputs.get('simulation').get('enable_print')
star_topology = inputs.get('simulation').get('star_topology')
shadowing_coherence_time_ms = inputs.get('channel').get('shadowing_coherence_time_ms')
carrier_frequency_ghz = inputs.get('radio').get('carrier_frequency_ghz')
bandwidth_ghz = inputs.get('radio').get('bandwidth_ghz')
antenna_gain_model = inputs.get('channel').get('antenna_gain_model')
use_huawei_measurements = inputs.get('channel').get('use_huawei_measurements')
shadowing_coherence_time_s = shadowing_coherence_time_ms / 1000
bandwidth_hz = bandwidth_ghz * 1e9
power_bo = inputs.get('power_consumed_ue').get('backoff')
power_idle = inputs.get('power_consumed_ue').get('idle')
power_tx = inputs.get('power_consumed_ue').get('tx_data')
power_ack = inputs.get('power_consumed_ue').get('wait_ack')
machine_kind = inputs.get('scenario').get('machine')
machine_type = inputs.get('scenario').get('machine_type')
snr_th_db = inputs.get('channel').get('snr_th_db')
sir_th_db = inputs.get('channel').get('sir_th_db')
sinr_th_db = inputs.get('channel').get('sinr_th_db')
modulation_order = inputs.get('radio').get('modulation_order')
noise_figure_ue_db = inputs.get('ue').get('ue_noise_figure_db')
noise_figure_ue = 10 ** (noise_figure_ue_db / 10)
noise_figure_bs_db = inputs.get('bs').get('bs_noise_figure_db')
noise_figure_bs = 10 ** (noise_figure_bs_db / 10)

broadcast_ampl_factor_no_change = inputs.get('rl').get('router').get(
    'alfa_broad_no_change')  # (minimum = 0.5 to keep the reward between 1 and 0)
broadcast_ampl_factor_change = inputs.get('rl').get('router').get('alfa_broad_change')
unicast_ampl_factor_no_ack = inputs.get('rl').get('router').get(
    'alfa_uni_no_ack')  # (minimum = 0.5 to keep the reward between 1 and 0)
unicast_ampl_factor_ack = inputs.get('rl').get('router').get('alfa_uni_ack')
energy_factor = inputs.get('rl').get('router').get('energy_factor')
TTL = inputs.get('rl').get('router').get('TTL')  # mi serve
n_actions = inputs.get('rl').get('agent').get('n_actions')
batch_size = inputs.get('rl').get('agent').get('batch_size')
discount_factor = inputs.get('rl').get('agent').get('discount_factor')

n_simulations_for_training = inputs.get('rl').get('agent').get('n_simulations_for_training')
max_len_replay_buffer = inputs.get('rl').get('agent').get('max_len_replay_buffer')
best_score = inputs.get('rl').get('agent').get('best_score')

mobility_obstacle = inputs.get('simulation').get('mobility_obstacle')
step_size = inputs.get('simulation').get('mobility_step_size')
mobility_spawn = inputs.get('simulation').get('mobility_spawn')
mobility_shuffle = inputs.get('simulation').get('mobility_shuffle')
mobility_changes = inputs.get('simulation').get('mobility_changes')
ack_tx_bs_seen = 0
hop_limit = inputs.get('aloha_protocol').get('hop_limit')

# Initializations
phy_success = False  # True if a transmission is successful at PHY layer, False otherwise
mac_success = False  # True if a transmission is successful at MAC layer, False otherwise
output_dict = {}  # Dictionary where keys are the metrics to be computed and the values contain the corresponding output
# for each UE in the form (N x M), where N is the number of times we loop over a different number of UEs
# and M is the number of simulations
output_keys = ["p_mac", "s_ue", "s", "l", "e", "j_index"]

# Loop over the output keys and initialize matrices for each key
n_simulated_ues = abs((final_number_of_ues - initial_number_of_ues) // step_number_of_ues) + 1
for output_key in output_keys:
    # Ensure the top-level key exists in the dictionary
    if output_key not in output_dict:
        output_dict[output_key] = {}

    for n_ue_index, n_ue in enumerate(range(initial_number_of_ues, final_number_of_ues + 1, step_number_of_ues)):
        # Ensure the N={n_ue} key exists
        if f"N={n_ue}" not in output_dict[output_key]:
            output_dict[output_key][f"N={n_ue}"] = {}

        for n_sim in range(n_simulations):
            # Initialize the array with the correct shape
            output_dict[output_key][f"N={n_ue}"][f"Sim={n_sim}"] = np.zeros(n_ue)


# Pick up input scenario
if scenario_name in ['grid']:
    scenario_sheet_name = inputs.get('scenario').get('input_sheet_names').get(scenario_name)
    scenario_df = read_input_file(file_name=scenario_file_name, sheet_name=scenario_sheet_name, reset_index=True)
else:
    exit("The input scenario is not recognized")

# Initialize geometry environment
geometry_class = Geometry(scenario_df=scenario_df)

# Initialize distribution environment
distribution_class = Distribution(ue_distribution_type=ue_distribution_type,
                                  machine_distribution_type=scenario_name,
                                  scenario_df=scenario_df,
                                  tot_number_of_ues=initial_number_of_ues)

# Distribute machines
machine_array = distribution_class.distribute_machines(scenario_df=scenario_df)

# Compute the cluster density
machine_area_sqr_m = 0  # Sum of the area covered by the machines
average_machine_height_m = 0  # Average height of the machines
for machine in machine_array:
    machine_area_sqr_m = machine_area_sqr_m + machine.get_machine_size() ** 2
    average_machine_height_m = average_machine_height_m + machine.get_machine_height()
area_scenario_sqr_m = geometry_class.get_factory_length() * geometry_class.get_factory_width()
clutter_density = machine_area_sqr_m / area_scenario_sqr_m
if len(machine_array) > 0:
    average_machine_height_m = average_machine_height_m / len(machine_array)

# Compute the tick duration
simulator_tick_duration_s = compute_simulator_tick_duration(input_params_dict=inputs)

# Instantiate the BS
bs = instantiate_bs(input_params_dict=inputs, simulator_tick_duration=simulator_tick_duration_s,
                    starting_state=bs_starting_state, bit_rate_gbits=bit_rate_gbits)

# Set the position of the gNB
distribution_class.distribute_bs(bs=bs)

# Instantiate the channel
thz_channel = THzChannel(params=inputs)

# Compute the SNR threshold
payload_fq = inputs.get('traffic_fq').get('payload')
snr_threshold_db = compute_snr_threshold_db(input_payload_bytes=payload_fq, input_p_succ_phy=p_succ_phy)
print("SNR THRESHOLD: ", snr_threshold_db)

# Compute the simulation duration
tot_simulation_time_tick = math.ceil(simulation_time_s / simulator_tick_duration_s)

# Compute the shadowing samples
n_shadowing_samples = math.ceil(simulation_time_s / shadowing_coherence_time_s)  # Number of shadowing samples
thz_channel.set_shadowing_sample_db(input_n_shadowing_samples=n_shadowing_samples)
shadowing_coherence_time_tick_duration = shadowing_coherence_time_s / simulator_tick_duration_s

# Set all the timings of the aloha protocol
t_ack_ns = round((ack_size_bytes * 8 * 1e-9) / bs.get_bit_rate_gbits(), 11)  # 1.6 ns
t_ack_tick = round(t_ack_ns / simulator_tick_duration_s)
t_backoff_tick = t_ack_tick
t_idle_tick = t_ack_tick
print("T_ack [s]: ", t_ack_ns)
print("T_ack_tick: ", t_ack_tick)
print("SNR threshold: ", snr_th_db)
# check d_max
for machine in machine_array:
    distance = np.sqrt((machine.x_max - bs.x) ** 2 + (machine.y_max - bs.y) ** 2 +
                       (machine.z_center - bs.z) ** 2)

"""
    Loop over an given number of UEs (n_ues)
"""
for seed in range(initial_seed, final_seed + 1):
    print("*************** Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)

    for n_ue_index, n_ue in enumerate(range(initial_number_of_ues, final_number_of_ues + 1, step_number_of_ues)):
        print("*************** Number of UEs: ", n_ue)

        single_y_data = {}
        single_z_data = {}
        single_l_data = {}
        single_e_data = {}
        single_ue_coordinates = {}

        # Update the number of UEs
        distribution_class.set_number_of_ues(input_n_ues=n_ue)

        """
            Repeat the simulation for a given number of times (n_simulations)
        """
        ue_coordinates = dict()
        copy_ue_coordinates_dict = dict()
        for ue in range(n_ue):
            ue_coordinates[ue] = list()
            single_y_data[ue] = []
            single_z_data[ue] = []
            single_l_data[ue] = []
            single_e_data[ue] = []
            single_ue_coordinates[ue] = []
            copy_ue_coordinates_dict[ue] = []

        ue_array = instantiate_ues(input_params_dict=inputs, tot_number_of_ues=n_ue, starting_state=ue_starting_state,
                                   t_state_tick=t_idle_tick, simulator_tick_duration=simulator_tick_duration_s,
                                   bit_rate_gbits=bit_rate_gbits, max_n_retx_per_packet=max_n_retx_per_packet)

        # Distribute the UEs in the environment
        distribution_class.distribute_ues(ue_array=ue_array, machine_array=machine_array, bs=bs,
                                          simulator_tick_duration=simulator_tick_duration_s,
                                          factory_length=geometry_class.get_factory_length(),
                                          factory_width=geometry_class.get_factory_width(),
                                          factory_height=geometry_class.get_factory_height())

        for ue in ue_array:
            ue.starting_coordinates = ue.get_coordinates()

        # Compute the propagation delays from UEs to BS and among UEs
        compute_propagation_delays(ue_array=ue_array, bs=bs, input_simulator_tick_duration_s=simulator_tick_duration_s)

        # Compute the maximum propagation delay, i.e., the diagonal of the paralleliped modeling the factory
        max_prop_delay_tick_diagonal = round((math.sqrt(geometry_class.get_factory_length() ** 2 +
                                                        geometry_class.get_factory_width() ** 2 +
                                                        geometry_class.get_factory_height() ** 2) / c) / simulator_tick_duration_s)

        # max_prop_delay_tick = min(max_prop_delay_tick_diagonal, max_prop_delay_tick_tx_range)
        max_prop_delay_tick = max_prop_delay_tick_diagonal

        print("The max propagation delay is", max_prop_delay_tick)

        # Set UE LoS/NLoS condition
        if ue_distribution_type != "Grid":

            for i in range(0, len(ue_array)):
                ue_array[i].is_in_los = set_ues_los_condition(ue=ue_array[i], bs=bs, machine_array=machine_array,
                                                                  link='ue_bs')

            for j in range(0, len(ue_array)):
                for i in range(0, len(ue_array)):
                    los_condition = set_ues_los_condition(ue=ue_array[j], bs=ue_array[i],
                                                              machine_array=machine_array,
                                                              link='ue_ue')
                    ue_array[j].is_in_los_ues.append(los_condition)

            # Method to ensure that in case of UEs' Uniform distribution they can have at least one neighbour
            # to reach the BS

            check_for_neighbours(ue_array=ue_array, machine_array=machine_array, bs=bs,
                                 input_snr_threshold_db=snr_th_db,
                                 input_shadowing_sample_index=0, input_thz_channel=thz_channel,
                                 input_carrier_frequency_ghz=carrier_frequency_ghz, input_bandwidth_hz=bandwidth_hz,
                                 input_apply_fading=apply_fading, input_clutter_density=clutter_density,
                                 antenna_gain_model=antenna_gain_model, use_huawei_measurements=use_huawei_measurements,
                                 input_average_clutter_height_m=average_machine_height_m)

            compute_propagation_delays(ue_array=ue_array, bs=bs,
                                       input_simulator_tick_duration_s=simulator_tick_duration_s)

        else:

            for i in range(0, len(ue_array)):
                ue_array[i].is_in_los = set_ues_los_condition(ue=ue_array[i], bs=bs, machine_array=machine_array,
                                                                  link='ue_bs')

            for j in range(0, len(ue_array)):
                for i in range(0, len(ue_array)):
                    los_condition = set_ues_los_condition(ue=ue_array[j], bs=ue_array[i], machine_array=machine_array,
                                                  link='ue_ue')
                    ue_array[j].is_in_los_ues.append(los_condition)

        print("The max propagation delay is", max_prop_delay_tick)

        # Initialization of the RL parameter
        nodes_list = [str(ue.get_ue_id()) for ue in ue_array] + ['BS']

        tx_results = {"UE_" + str(i): {} for i in range(n_ue)}

        for ue_index, ue in enumerate(ue_array):
            ue.set_neighbour_table(input_neighbour_table=nodes_list[:ue_index] + nodes_list[ue_index + 1:])
            tx_results["UE_" + str(ue.get_ue_id())] = {j: 0 for j in ue.get_neighbour_table()}
            ue.forced_broadcast_actions_counter = 0

        for n_simulation in range(n_simulations):
            print("***** Simulation number: ", n_simulation)

            # At each new simulation, reset the machine distribution and UE's distribution
            # Distribute machines
            machine_array = distribution_class.distribute_machines(scenario_df=scenario_df)

            # Create the dictionary containing the info about starting
            # and ending times of all transmissions in the mesh network
            # at the RX side
            simulator_timing_structure = create_simulator_timing_structure(
                input_n_ue=n_ue, input_simulation_duration_tick=tot_simulation_time_tick)

            # Re-initialization of the timing dictionary
            reset_simulator_timing_structure(output_simulator_timing_structure=simulator_timing_structure,
                                             input_simulation_duration_tick=tot_simulation_time_tick)

            # Re-initializations of counters
            t = 0
            shadowing_sample_index = 0
            shadowing_next_tick = t + shadowing_coherence_time_tick_duration  # Next tick where the shadowing sample
            # should be changed
            bs.set_n_data_rx(input_n_data_rx=0)
            bs.set_n_data_rx_rt(input_n_data_rx_rt=0)
            bs.set_n_data_rx_cn(input_n_data_rx_cn=0)
            bs.set_n_data_rx_nrt(input_n_data_rx_nrt=0)
            bs.set_n_data_rx_fq(input_n_data_rx_fq=0)
            bs.set_state(input_state=bs_starting_state)
            bs.set_state_duration(input_ticks=tot_simulation_time_tick + 1)  # Up to the end of the simulation
            bs.set_state_starting_tick(input_tick=0)
            bs.set_state_final_tick(input_tick=tot_simulation_time_tick + 1)
            bs.packet_rx = False
            ue_coordinates_list = list()
            bs.end_of_rx_for_ack_tx = None
            bs.sequence_number_of_packet_rx = 0
            bs.rx_data = False
            bs.id_ues_data_rx = list()
            for ue in ue_array:
                ue.set_n_data_tx(input_n_data_tx=0)
                ue.set_n_data_rx(input_n_data_rx=0)
                ue.set_state(input_state=ue_starting_state)
                ue.set_t_generation(input_t_generation=0)  # Since UEs are not synchronized by definition of ALOHA,
                # they can start generating data from the same instant
                ue.set_state_duration(input_ticks=t_idle_tick)
                ue.set_state_starting_tick(input_tick=0)
                ue.set_state_final_tick(input_tick=t_idle_tick)
                ue.energy_consumed = 0
                ue.energy_consumed += power_idle * (ue.get_state_final_tick()) * simulator_tick_duration_s
                ue.set_packet_id(input_packet_id=0)
                bs.set_n_data_rx_from_ues(input_ue_id=ue.get_ue_id(), input_n_data_rx=0)
                bs.packet_id_received[ue.get_ue_id()] = ([])
                bs.temp_packet_id_received[ue.get_ue_id()] = ([])

                ue.set_retransmission_packets(retransmission_bool=False)
                ue.set_relay_bool(relay_bool=False)
                ue.reset_temp_obs()
                ue.set_old_state(input_old_state=None)
                ue.reset_obs()
                ue.set_packets_sent(input_packets_sent=0)
                ue.set_reward(input_reward=[])
                ue.set_last_action(input_last_action=None)
                ue.set_unicast_rx_address(input_unicast_rx_address=None)
                ue.set_unicast_rx_index(input_unicast_rx_index=None)
                ue.set_broadcast_bool(input_broadcast_bool=False)
                ue.new_action_bool = True
                ue.first_entry = False
                ue.action_packet_id = None
                ue.forward_in_wait_ack = False
                ue.check_last_round = False
                ue.set_action_list(input_action_list=[])
                ue.set_success_action_list(input_success_action_list=[])
                ue.set_reception_during_bo_bool(
                    input_data_rx_bool=False)  # True when the UE has received something during BO
                ue.set_reception_during_wait_bool(input_data_rx_bool=False)
                ue.set_ul_buffer()

                ue.n_tear = 0
                ue.ack_rx_during_wait_ack = False
                ue.data_rx_during_wait_ack = False
                ue.list_data_rx_during_wait_ack = list()
                ue.list_data_generated_during_wait_ack = list()
                ue.list_data_rx_from_ue_id = list()
                ue.dict_data_rx_during_wait_ack = dict()
                ue.dict_data_rx_during_bo = dict()
                ue.reception_ack_during_wait = False
                ue.list_ack_sent_from_bs = list()
                ue.dict_ack_sent_from_ue = dict()
                ue.buffer_packet_sent = list()
                ue.ues_colliding_at_ue = list()
                ue.data_rx_at_ue_ue_id_list = list()
                ue.latency_ue = list()

                ue.n_generated_packets = 0

                ue.forward_in_bo = False
                ue.packet_forward = False
                ue.forward_in_ack = False
                ue.multihop_bool = True
                ue.end_data_tx = 0

                for other_ue in ue_array:
                    if other_ue != ue:
                        ue.packet_id_received[other_ue.get_ue_id()] = ([])
                        ue.dict_data_rx_during_wait_ack[other_ue.get_ue_id()] = ([])
                        ue.dict_data_rx_during_bo[other_ue.get_ue_id()] = ([])
                        ue.dict_ack_sent_from_ue[other_ue.get_ue_id()] = ([])

                ue_coordinates_list.append(ue.starting_coordinates)
                copy_ue_coordinates_dict[ue.get_ue_id()].append(ue.starting_coordinates.tolist())

            ues_colliding_at_bs = list()
            ues_interfering_at_bs = list()

            ues_no_phy_colliding_at_bs = list()
            ues_no_phy_colliding_pck_type_at_bs = list()
            ues_no_phy_colliding_at_ue = list()
            ues_no_phy_colliding_pck_type_at_ue = list()

            # Initialization of the RL parameter
            nodes_list = [str(ue.get_ue_id()) for ue in ue_array] + ['BS']  # mi serve -> lista per tab vicini

            for ue_index, ue in enumerate(ue_array):
                ue.set_replay_buffer(input_replay_buffer=deque(maxlen=max_len_replay_buffer))
                # lista di stringhe
                ue.set_neighbour_table(input_neighbour_table=nodes_list[:ue_index] + nodes_list[
                                                                                     ue_index + 1:])  # mi serve: tab vicini per ogni UE
                # ue.obs[0]: neighbour_table,
                # ue.obs[1]: lista ack ricevuti per ogni vicino,
                # ue.obs[2]: potenza ricevuta dall'ultima tx per ogni vicino
                # ue.obs[3]: TTL di ogni vicino
                ue.reset_obs()
                ue.reset_temp_obs()  # ***
                ue.set_last_action(input_last_action=None)
                ue.set_broadcast_bool(input_broadcast_bool=False)
                ue.new_action_bool = True
                ue.next_action = None
                ue.first_bo_entry = True
                ue.first_entry = False
                ue.copy_buffer_packet_list = None  # mi serve
                ue.action_packet_id = None  # mi serve
                ue.designated_rx = False  # mi serve (per unicast)
                ue.set_actions_per_simulation(input_actions_per_simulation=[[], [], [], []])
                ue.set_success_actions_per_simulation(input_success_actions_per_simulation=[[], []])
                ue.saved_coordinates = ue.get_coordinates()
                keys = nodes_list[:ue_index] + nodes_list[ue_index + 1:]
                ue.packets_to_be_removed = {key: [] for key in keys}
                ue.forward_in_bo = False
                ue.packet_forward = False
                ue.forward_in_ack = False
                ue.multihop_bool = True

            packet_already_in_queue = False
            packet_generated_by_ue_itself = False
            packet_out_of_hop_limit = False

            """
                Simulation starts for a given simulation time  
            """

            # Code to plot the factory in 2D or 3D
            plot_factory(factory_length=geometry_class.get_factory_length(),
                         factory_width=geometry_class.get_factory_width(),
                         factory_height=geometry_class.get_factory_height(),
                         machine_list=machine_array,
                         ue_list=ue_array,
                         bs=bs,
                         distribution_class=distribution_class,
                         scenario_name=scenario_name,
                         save_file=f'./multi_hop_industrial_simulator/results/plot_{scenario_name}_scenario_3d.png')
            plot_scenario_2d(factory_length=geometry_class.get_factory_length(),
                             factory_width=geometry_class.get_factory_width(),
                             factory_height=geometry_class.get_factory_height(),
                             machine_list=machine_array,
                             ue_list=ue_array,
                             bs=bs,
                             distribution_class=distribution_class,
                             scenario_name=scenario_name,
                             save_file=f'./multi_hop_industrial_simulator/results/plot_{scenario_name}_scenario_2d.png',
                             )

            t_change = 0
            next_t_change = 0
            for ue in ue_array:
                d_from_bs = np.sqrt((ue.x - bs.x) ** 2 + (ue.y - bs.y) ** 2 + (ue.z - bs.z) ** 2)

                for other_ue in ue_array:
                    if ue != other_ue:
                        d_from_ue = np.sqrt((ue.x - other_ue.x) ** 2 + (ue.y - other_ue.y) ** 2 +
                                            (ue.z - other_ue.z) ** 2)

            while t <= tot_simulation_time_tick:
                # Mobility code: there are pre-defined instant of time with different types of movement in the environment ->
                # it's important to update the simulator timing structure to save the old state of each UE and update
                # the current state based on the new coordinates
                # We need to do that because after the movement it could happen that a UE is no more able to rx a packet
                # from another UE because it is out of its range
                copy_simulator_timing_structure = deepcopy(simulator_timing_structure)
                if t == next_t_change and t > 0:
                    t_change = t

                    if mobility_obstacle: # Move obstacles in a clockwise manner in the environment with a fix step size
                        # Machine movement:
                        pilot_x_min = pilot_y_min = 1.25
                        pilot_x_max = pilot_y_max = 18.75
                        min_x, max_x = machine_array[0].x_center, machine_array[0].x_center
                        min_y, max_y = machine_array[0].y_center, machine_array[0].y_center

                        for machine in machine_array:
                            if pilot_x_min <= machine.x_center < min_x:
                                min_x = machine.x_center
                            if pilot_x_max >= machine.x_center > max_x:
                                max_x = machine.x_center
                            if pilot_y_min <= machine.y_center < min_y:
                                min_y = machine.y_center
                            if pilot_y_max >= machine.y_center > max_y:
                                max_y = machine.y_center

                        for machine in machine_array:
                            new_x, new_y = machine.move_machine(machine.x_center, machine.y_center, step_size,
                                                        min_x, max_x, min_y, max_y)
                            machine.set_coordinates(new_x, new_y, machine.z_center)
                            machine = Machine(x_center=new_x, y_center=new_y, z_center=machine.z_center,
                                machine_size=machine.get_machine_size(),
                                max_number_of_ues=machine.get_max_number_of_ues())

                    elif mobility_spawn: # some obstacles appear and disappear within the environment ->
                        # it could happen that sometimes there are links established within the environment
                        # and other times there are not
                        # Just 2 movement check if the tick is before the half
                        if t_change < 0.5 * tot_simulation_time_tick:
                            machine_array[8].set_coordinates(3.25, machine_array[8].y_center, machine_array[8].z_center)
                            machine_array[9].set_coordinates(machine_array[9].x_center, 7.75, machine_array[9].z_center)
                            machine_array[10].set_coordinates(16.75, machine_array[10].y_center, machine_array[10].z_center)

                        elif t_change > 0.5 * tot_simulation_time_tick and t_change < tot_simulation_time_tick:
                            machine_array[8].set_coordinates(16.75, machine_array[8].y_center, machine_array[8].z_center)
                            machine_array[9].set_coordinates(machine_array[9].x_center, 12.25, machine_array[9].z_center)
                            machine_array[10].set_coordinates(3.25, machine_array[10].y_center, machine_array[10].z_center)

                        else:
                            machine_array[8].set_coordinates(22, machine_array[8].y_center, machine_array[8].z_center)
                            machine_array[9].set_coordinates(machine_array[9].x_center, 22, machine_array[9].z_center)
                            machine_array[10].set_coordinates(22, machine_array[10].y_center, machine_array[10].z_center)

                        machine_array[8] = Machine(x_center=machine_array[8].x_center,
                                                    y_center=machine_array[8].y_center,
                                                    z_center=machine_array[8].z_center,
                                                    machine_size=machine_array[8].get_machine_size(),
                                                    max_number_of_ues=machine_array[8].get_max_number_of_ues())

                        machine_array[9] = Machine(x_center=machine_array[9].x_center,
                                                    y_center=machine_array[9].y_center,
                                                    z_center=machine_array[9].z_center,
                                                    machine_size=machine_array[9].get_machine_size(),
                                                    max_number_of_ues=machine_array[9].get_max_number_of_ues())

                        machine_array[10] = Machine(x_center=machine_array[10].x_center,
                                                    y_center=machine_array[10].y_center,
                                                    z_center=machine_array[10].z_center,
                                                    machine_size=machine_array[10].get_machine_size(),
                                                    max_number_of_ues=machine_array[10].get_max_number_of_ues())

                    elif mobility_shuffle: # UEs coordinates are randomly exchanged.
                        # The result is that each UE occupies a different position after the shuffle

                        for ue in ue_array:
                            for ue_key_ext in simulator_timing_structure.keys():
                                if ue_key_ext == f'UE_{ue.get_ue_id()}':
                                    # Loop over the DATA receptions from other UEs
                                    for ue_key_int in simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                                        # check the value of the final_state_tick
                                        if len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int]) > 0:
                                            for array_index in range(
                                                    len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int])):
                                                if simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                    array_index][0] \
                                                        != tot_simulation_time_tick + 1 and \
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] != tot_simulation_time_tick + 1:

                                                    for other_ue in ue_array:
                                                        if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                            prop_delay_tick = ue.get_prop_delay_to_ue_tick(
                                                                other_ue.get_ue_id())

                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][0] = \
                                                                simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                    ue_key_int][
                                                                    array_index][0] - prop_delay_tick
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][1] = \
                                                                simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                    ue_key_int][
                                                                    array_index][1] - prop_delay_tick

                                                    if ue_key_int == 'BS':

                                                        prop_delay_tick = ue.get_prop_delay_to_bs_tick()

                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][0] - prop_delay_tick
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][1] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][1] - prop_delay_tick

                                    for ue_key_int in simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                                        # check the value of the final_state_tick
                                        if len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int]) > 0:
                                            for array_index in range(
                                                    len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int])):
                                                if simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                    array_index][0] \
                                                        != tot_simulation_time_tick + 1 and \
                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][0] != tot_simulation_time_tick + 1:

                                                    for other_ue in ue_array:
                                                        if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                            prop_delay_tick = ue.get_prop_delay_to_ue_tick(
                                                                other_ue.get_ue_id())

                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][0] = \
                                                                simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                    ue_key_int][
                                                                    array_index][0] - prop_delay_tick
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][1] = \
                                                                simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                    ue_key_int][
                                                                    array_index][1] - prop_delay_tick

                        for ue_key_ext in simulator_timing_structure.keys():
                            if ue_key_ext == 'BS':
                                # Loop over the DATA receptions from other UEs
                                for ue_key_int in simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                                    # check the value of the final_state_tick
                                    if len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int]) > 0:
                                        for array_index in range(
                                                len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int])):
                                            if \
                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][array_index][
                                                0] \
                                                    != tot_simulation_time_tick + 1 and \
                                                    simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                        array_index][0] != tot_simulation_time_tick + 1:

                                                for other_ue in ue_array:
                                                    if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                        prop_delay_tick = other_ue.get_prop_delay_to_bs_tick()

                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][0] - prop_delay_tick
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][1] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][1] - prop_delay_tick

                                for ue_key_int in simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                                    # check the value of the final_state_tick
                                    if len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int]) > 0:
                                        for array_index in range(
                                                len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int])):
                                            if \
                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][array_index][0] \
                                                    != tot_simulation_time_tick + 1 and \
                                                    simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                        array_index][0] != tot_simulation_time_tick + 1:

                                                for other_ue in ue_array:
                                                    if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                        prop_delay_tick = other_ue.get_prop_delay_to_bs_tick()

                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][0] = \
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][0] - prop_delay_tick
                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][1] = \
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][1] - prop_delay_tick

                        ###### Random change in UEs coordinates ########
                        random.shuffle(ue_coordinates_list)

                        index = 0
                        for ue in ue_array:
                            ue.set_coordinates(ue_coordinates_list[index][0], ue_coordinates_list[index][1],
                                               ue_coordinates_list[index][2])
                            copy_ue_coordinates_dict[ue.get_ue_id()].append(ue.get_coordinates().tolist())

                            index += 1

                    # re-plot the reference scenario after the movement

                    # plot_scenario_2d(factory_length=geometry_class.get_factory_length(),
                    #                  factory_width=geometry_class.get_factory_width(),
                    #                  factory_height=geometry_class.get_factory_height(),
                    #                  machine_list=machine_array,
                    #                  ue_list=ue_array,
                    #                  bs=bs,
                    #                  distribution_class=distribution_class,
                    #                  scenario_name=scenario_name,
                    #                  save_file=f'./multi_hop_industrial_simulator/results/plot_{scenario_name}_scenario_2d.png',
                    #                  )
                    for i in range(0, len(ue_array)):
                        ue_array[i].is_in_los_ues.clear()

                    # Update UE LoS/NLoS condition after the movement

                    for i in range(0, len(ue_array)):
                        ue_array[i].is_in_los = set_ues_los_condition(ue=ue_array[i], bs=bs,
                                                                          machine_array=machine_array,
                                                                          link='ue_bs')

                    for j in range(0, len(ue_array)):
                        for i in range(0, len(ue_array)):
                            los_condition = set_ues_los_condition(ue=ue_array[j], bs=ue_array[i],
                                                                      machine_array=machine_array,
                                                                      link='ue_ue')
                            ue_array[j].is_in_los_ues.append(los_condition)

                    for ue in ue_array:
                        tx_rx_distance_m = compute_distance_m(tx=ue, rx=bs)
                        shadowing_sample_index = 0
                        snr_db = thz_channel.get_3gpp_snr_db(
                            tx=ue, rx=bs,
                            carrier_frequency_ghz=carrier_frequency_ghz,
                            tx_rx_distance_m=tx_rx_distance_m,
                            apply_fading=apply_fading,
                            bandwidth_hz=bandwidth_hz,
                            clutter_density=clutter_density,
                            input_shadowing_sample_index=shadowing_sample_index,
                            antenna_gain_model=antenna_gain_model,
                            use_huawei_measurements=use_huawei_measurements,
                            input_average_clutter_height_m=average_machine_height_m,
                            los_cond='bs_ue')
                        if snr_db > sinr_th_db:
                            success = True
                        else:
                            success = False

                    for ue in ue_array:
                        for other_ue in ue_array:
                            if other_ue != ue:
                                tx_rx_distance_m = compute_distance_m(tx=ue, rx=other_ue)

                                shadowing_sample_index = 0
                                snr_db = thz_channel.get_3gpp_snr_db(
                                    tx=ue, rx=other_ue,
                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                    tx_rx_distance_m=tx_rx_distance_m,
                                    apply_fading=apply_fading,
                                    bandwidth_hz=bandwidth_hz,
                                    clutter_density=clutter_density,
                                    input_shadowing_sample_index=shadowing_sample_index,
                                    antenna_gain_model=antenna_gain_model,
                                    use_huawei_measurements=use_huawei_measurements,
                                    input_average_clutter_height_m=average_machine_height_m,
                                    los_cond='ue_ue')
                                if snr_db > sinr_th_db:
                                    success = True
                                else:
                                    success = False

                    if mobility_shuffle: # Update propagation delays
                        compute_propagation_delays(ue_array=ue_array, bs=bs,
                                                   input_simulator_tick_duration_s=simulator_tick_duration_s)

                        for ue in ue_array:
                            for ue_key_ext in simulator_timing_structure.keys():
                                if ue_key_ext == f'UE_{ue.get_ue_id()}':
                                    # Loop over the DATA receptions from other UEs
                                    for ue_key_int in simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                                        # check the value of the final_state_tick
                                        if len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int]) > 0:
                                            for array_index in range(
                                                    len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int])):
                                                if simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                    array_index][0] \
                                                        != tot_simulation_time_tick + 1 and \
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] != tot_simulation_time_tick + 1:

                                                    for other_ue in ue_array:
                                                        if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                            prop_delay_tick = ue.get_prop_delay_to_ue_tick(
                                                                other_ue.get_ue_id())

                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][0] = \
                                                                simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                    ue_key_int][
                                                                    array_index][0] + prop_delay_tick
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][1] = \
                                                                simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                    ue_key_int][
                                                                    array_index][1] + prop_delay_tick

                                                            if simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][0] < next_t_change:
                                                                remove_item_in_timing_structure(
                                                                    input_simulator_timing_structure=copy_simulator_timing_structure,
                                                                    input_rx_key=f'UE_{ue.get_ue_id()}',
                                                                    input_type_key='DATA_RX',
                                                                    input_tx_key=f'UE_{other_ue.get_ue_id()}')

                                                            else:
                                                                if enable_print:
                                                                    print(" NEW start_tick = ",
                                                                          simulator_timing_structure[ue_key_ext]
                                                                          ['DATA_RX'][ue_key_int][array_index][0])
                                                                    print("NEW end tick = ",
                                                                          simulator_timing_structure[ue_key_ext]
                                                                          ['DATA_RX'][ue_key_int][array_index][1])

                                                    if ue_key_int == 'BS':

                                                        prop_delay_tick = ue.get_prop_delay_to_bs_tick()

                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][0] + prop_delay_tick
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][1] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][1] + prop_delay_tick

                                                        if \
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] < next_t_change:
                                                            remove_item_in_timing_structure(
                                                                input_simulator_timing_structure=copy_simulator_timing_structure,
                                                                input_rx_key=f'UE_{ue.get_ue_id()}',
                                                                input_type_key='DATA_RX',
                                                                input_tx_key='BS')
                                                        else:
                                                            if enable_print:
                                                                print(" NEW start_tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['DATA_RX'][ue_key_int][array_index][0])
                                                                print("NEW end tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['DATA_RX'][ue_key_int][array_index][1])

                                    for ue_key_int in simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                                        # check the value of the final_state_tick
                                        if len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int]) > 0:
                                            for array_index in range(
                                                    len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int])):
                                                if simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                    array_index][0] \
                                                        != tot_simulation_time_tick + 1 and \
                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][0] != tot_simulation_time_tick + 1:

                                                    for other_ue in ue_array:
                                                        if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                            prop_delay_tick = ue.get_prop_delay_to_ue_tick(
                                                                other_ue.get_ue_id())

                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][0] = \
                                                                simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                    ue_key_int][
                                                                    array_index][0] + prop_delay_tick
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][1] = \
                                                                simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                    ue_key_int][
                                                                    array_index][1] + prop_delay_tick

                                                            if simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][0] < next_t_change:
                                                                remove_item_in_timing_structure(
                                                                    input_simulator_timing_structure=copy_simulator_timing_structure,
                                                                    input_rx_key=f'UE_{ue.get_ue_id()}',
                                                                    input_type_key='ACK_RX',
                                                                    input_tx_key=f'UE_{other_ue.get_ue_id()}')
                                                            else:
                                                                if enable_print:
                                                                    print(" NEW start_tick = ",
                                                                          simulator_timing_structure[ue_key_ext]
                                                                          ['ACK_RX'][ue_key_int][array_index][0])
                                                                    print("NEW end tick = ",
                                                                          simulator_timing_structure[ue_key_ext]
                                                                          ['ACK_RX'][ue_key_int][array_index][1])

                                                    if ue_key_int == 'BS':

                                                        prop_delay_tick = ue.get_prop_delay_to_bs_tick()

                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][0] = \
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][0] + prop_delay_tick
                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][1] = \
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][1] + prop_delay_tick

                                                        if simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][0] < next_t_change:
                                                            remove_item_in_timing_structure(
                                                                input_simulator_timing_structure=copy_simulator_timing_structure,
                                                                input_rx_key=f'UE_{ue.get_ue_id()}',
                                                                input_type_key='ACK_RX',
                                                                input_tx_key='BS')
                                                        else:
                                                            if enable_print:
                                                                print(" NEW start_tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['ACK_RX'][ue_key_int][array_index][0])
                                                                print("NEW end tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['ACK_RX'][ue_key_int][array_index][1])
                        for ue_key_ext in simulator_timing_structure.keys():
                            if ue_key_ext == 'BS':
                                # Loop over the DATA receptions from other UEs
                                for ue_key_int in simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                                    # check the value of the final_state_tick
                                    if len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int]) > 0:
                                        for array_index in range(
                                                len(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int])):
                                            if \
                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][array_index][
                                                0] \
                                                    != tot_simulation_time_tick + 1 and \
                                                    simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                        array_index][0] != tot_simulation_time_tick + 1:

                                                for other_ue in ue_array:
                                                    if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                        prop_delay_tick = other_ue.get_prop_delay_to_bs_tick()

                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][0] + prop_delay_tick
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][1] = \
                                                            simulator_timing_structure[ue_key_ext]['DATA_RX'][
                                                                ue_key_int][
                                                                array_index][1] + prop_delay_tick
                                                        if \
                                                        simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                            array_index][0] < next_t_change:
                                                            remove_item_in_timing_structure(
                                                                input_simulator_timing_structure=copy_simulator_timing_structure,
                                                                input_rx_key='BS',
                                                                input_type_key='DATA_RX',
                                                                input_tx_key=f'UE_{other_ue.get_ue_id()}')
                                                        else:
                                                            if enable_print:
                                                                print(" NEW start_tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['DATA_RX'][ue_key_int][array_index][0])
                                                                print("NEW end tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['DATA_RX'][ue_key_int][array_index][1])

                                for ue_key_int in simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                                    # check the value of the final_state_tick
                                    if len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int]) > 0:
                                        for array_index in range(
                                                len(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int])):
                                            if \
                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][array_index][0] \
                                                    != tot_simulation_time_tick + 1 and \
                                                    simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                        array_index][0] != tot_simulation_time_tick + 1:

                                                for other_ue in ue_array:
                                                    if ue_key_int == f'UE_{other_ue.get_ue_id()}':

                                                        prop_delay_tick = other_ue.get_prop_delay_to_bs_tick()

                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][0] = \
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][0] + prop_delay_tick
                                                        simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][1] = \
                                                            simulator_timing_structure[ue_key_ext]['ACK_RX'][
                                                                ue_key_int][
                                                                array_index][1] + prop_delay_tick

                                                        if simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                            array_index][0] < next_t_change:
                                                            remove_item_in_timing_structure(
                                                                input_simulator_timing_structure=copy_simulator_timing_structure,
                                                                input_rx_key='BS',
                                                                input_type_key='ACK_RX',
                                                                input_tx_key=f'UE_{other_ue.get_ue_id()}')
                                                        else:
                                                            if enable_print:
                                                                print(" NEW start_tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['ACK_RX'][ue_key_int][array_index][0])
                                                                print("NEW end tick = ",
                                                                      simulator_timing_structure[ue_key_ext]
                                                                      ['ACK_RX'][ue_key_int][array_index][1])

                        simulator_timing_structure = deepcopy(copy_simulator_timing_structure)

                # End Mobility

                for ue in ue_array:
                    ue.data_rx_during_wait_ack = False
                    ue.ack_rx_during_wait_ack = False
                    ue.set_ue_saved_state(input_ue_saved_state=ue.get_state())
                    # Update the timing structure for this UE (if necessary)
                    new_bo_tick = tot_simulation_time_tick + 1
                    new_bo_ue_id = None
                    new_wait_ack_tick = tot_simulation_time_tick + 1
                    new_wait_ack_ue_id = None
                    remove_ack_rx_keys = False
                    remove_data_rx_keys = False
                    # Update the simulator timing structure of each UE as soon as a new event happened
                    for ue_key_ext in simulator_timing_structure.keys():
                        if ue_key_ext == f'UE_{ue.get_ue_id()}':
                            # Loop over the DATA receptions from other UEs
                            for ue_key_int in simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                                min_rx_tick = np.min(
                                    simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][:, 1])
                                if min_rx_tick == t:
                                    # Shrink only the BO duration
                                    if ue.get_state() == 'BO':
                                        for user in ue_array:
                                            if ue_key_int == f'UE_{user.get_ue_id()}':
                                                # check that the UE is not a relay -> if not reduce BO and RX its DATA
                                                new_bo_tick = min(new_bo_tick, min_rx_tick)

                                            else:
                                                # if the UE from which it will RX DATA is a RELAY -> it continues with BO
                                                new_bo_tick = new_bo_tick

                                    new_bo_ue_id = ue_key_int

                            min_rx_tick1 = list()
                            min_rx_tick2 = list()
                            new_wait_ack_ue_id1 = list()
                            new_wait_ack_ue_id2 = list()
                            new_wait_ack_ue_id = list()
                            for ue_key_int in simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                                min_rx_tick1.append(
                                    np.min(simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][:, 1]))
                            for ue_key_int in simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                                min_rx_tick2.append(
                                    np.min(simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][:, 1]))
                            min1 = np.min(min_rx_tick1)
                            min2 = np.min(min_rx_tick2)
                            min_rx_tick = np.minimum(min1, min2)
                            if min_rx_tick == t:
                                # Shrink only the WAIT_ACK duration
                                if ue.get_state() == 'WAIT_ACK':
                                    new_wait_ack_tick = min(new_wait_ack_tick, min_rx_tick)
                                    if new_wait_ack_tick == min1:
                                        ue.data_rx_during_wait_ack = True

                                    if new_wait_ack_tick == min2:
                                        ue.ack_rx_during_wait_ack = True

                                for ue_key_int in simulator_timing_structure[ue_key_ext]['ACK_RX'].keys():
                                    if min_rx_tick in simulator_timing_structure[ue_key_ext]['ACK_RX'][ue_key_int][
                                                      :, 1]:
                                        new_wait_ack_ue_id.append((ue_key_int, 'ack'))
                                        remove_ack_rx_keys = True
                                for ue_key_int in simulator_timing_structure[ue_key_ext]['DATA_RX'].keys():
                                    if min_rx_tick in simulator_timing_structure[ue_key_ext]['DATA_RX'][ue_key_int][
                                                      :, 1]:
                                        new_wait_ack_ue_id.append((ue_key_int, 'data'))
                                        remove_data_rx_keys = True

                            # Update BO duration (if any)
                            if (ue.get_state_starting_tick() < new_bo_tick <= ue.get_state_final_tick() and
                                    ue.get_state() == 'BO'):
                                ue.set_state_duration(input_ticks=new_bo_tick)

                                if ue.get_state_duration() == ue.get_state_final_tick():
                                    # Special case: reception when the BO ends
                                    ue.set_reception_during_bo_bool(input_data_rx_bool=True)
                            elif new_bo_ue_id is not None and ue.data_rx_during_wait_ack is False:
                                # Update the timing structure to reset this reception
                                remove_item_in_timing_structure(
                                    input_simulator_timing_structure=simulator_timing_structure,
                                    input_rx_key=f'UE_{ue.get_ue_id()}',
                                    input_type_key='DATA_RX',
                                    input_tx_key=new_bo_ue_id)

                            # Update WAIT_ACK duration (if any)
                            if (ue.get_state_starting_tick() < new_wait_ack_tick <= ue.get_state_final_tick() and
                                    ue.get_state() == 'WAIT_ACK'):
                                ue.set_state_duration(input_ticks=new_wait_ack_tick)
                                if ue.get_state_duration() == ue.get_state_final_tick():
                                    # Special case: reception when the WAIT_ACK ends
                                    ue.set_reception_during_wait_bool(input_data_rx_bool=True)
                            elif len(new_wait_ack_ue_id) > 0:
                                # Update the timing structure to reset this reception
                                for ue_id, type in new_wait_ack_ue_id:
                                    if type == 'ack':
                                        remove_item_in_timing_structure(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_rx_key=f'UE_{ue.get_ue_id()}',
                                            input_type_key='ACK_RX',
                                            input_tx_key=ue_id)

                                    if type == 'data' and ue.data_rx_during_wait_ack is True:
                                        remove_item_in_timing_structure(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_rx_key=f'UE_{ue.get_ue_id()}',
                                            input_type_key='DATA_RX', input_tx_key=ue_id)
                                ue.data_rx_during_wait_ack = False
                            break

                # Update timing structure for the BS
                new_rx_tick = tot_simulation_time_tick + 1
                for ue_key_int in simulator_timing_structure['BS']['DATA_RX'].keys():
                    min_rx_tick = np.min(simulator_timing_structure['BS']['DATA_RX'][ue_key_int][:, 1])
                    if min_rx_tick == t:
                        # Shrink only the RX duration
                        if bs.get_state() == 'RX':
                            new_rx_tick = min(min_rx_tick, new_rx_tick)

                for ue_key_int in simulator_timing_structure['BS']['ACK_RX'].keys():
                    min_rx_tick = np.min(simulator_timing_structure['BS']['ACK_RX'][ue_key_int][:, 1])
                    if min_rx_tick == t:
                        # Shrink only the RX duration
                        if bs.get_state() == 'RX':
                            new_rx_tick = min(min_rx_tick, new_rx_tick)

                # Update RX duration (if any)
                if (bs.get_state_starting_tick() < new_rx_tick <= bs.get_state_final_tick() and
                        bs.get_state() == 'RX'):
                    bs.set_state_duration(input_ticks=new_rx_tick)

                """
                    Loop over UEs to make actions based on their state
                """
                for ue in ue_array:

                    """
                        Update UEs queue if the current instant is equal to the instant of packet generation
                    """
                    packet_generation_instant = ue.get_next_packet_generation_instant()
                    if t == packet_generation_instant:
                        if len(ue.ul_buffer.buffer_packet_list) < max_n_packets_to_be_forwarded + 1:
                            ue.add_new_packet(current_tick=packet_generation_instant, input_enable_print=enable_print)
                            ue.packet_generation_instant = packet_generation_instant
                            # print("UE ", ue.get_ue_id(), " packet generation: ", packet_generation_instant)

                    """
                        Based on the UE state, make the corresponding action and update the future state
                    """
                    if t == ue.get_state_duration():
                        """
                            UEs' STATE
                        """
                        if ue.get_state() == 'IDLE':
                            # Check queue, if there is a data then go to BO, otherwise go to IDLE until next generation
                            if ue.get_n_packets() > 0:
                                ue.update_num_tx(input_enable_print=enable_print)
                                backoff_duration_tick = get_backoff_duration(input_ue=ue,
                                                                             input_contention_window_int=
                                                                             contention_window_int,
                                                                             input_t_backoff_tick=t_backoff_tick,
                                                                             input_max_prop_delay_tick=max_prop_delay_tick)
                                go_in_backoff(input_ue=ue, current_tick=t,
                                              input_backoff_duration_tick=backoff_duration_tick,
                                              input_enable_print=enable_print)

                            else:
                                # Remain in IDLE
                                go_in_idle(input_ue=ue, current_tick=t, input_enable_print=enable_print)

                                if enable_print:
                                    print("Energy for UE ", ue.get_ue_id(), " = ",
                                          ue.energy_consumed / simulator_tick_duration_s,
                                          " computed from t = ", t, " to t = ", ue.get_state_duration())

                        elif ue.get_state() == 'BO':
                            go_in_tx_data_bool = False
                            if star_topology is False:
                                go_in_tx_data_bool = False

                                # Check if during BO the UE has received some data
                                if (ue.get_state_duration() == ue.get_state_final_tick() and
                                        ue.get_reception_during_bo_bool() is False):
                                    # The UE has not received anything during BO, it should go to TX_DATA
                                    go_in_tx_data_bool = True
                                else:

                                    # The UE has received a data during BO. Check if the reception is successful
                                    # Find the corresponding transmission
                                    ue.set_reception_during_bo_bool(input_data_rx_bool=False)
                                    (data_rx_at_ue_starting_tick, data_rx_at_ue_ending_tick, data_rx_at_ue_size_bytes,
                                     packet_id_rx_from_ue, data_rx_at_ue_ue_id) = (
                                        find_data_rx_times_tick(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            current_tick=t,
                                            input_ue_id=ue.get_ue_id()))

                                    for index in range(len(data_rx_at_ue_ue_id)):
                                        # Compute the tx-rx distance
                                        tx_rx_distance_m = compute_distance_m(tx=ue_array[data_rx_at_ue_ue_id[index]], rx=ue)

                                        # Check if the shadowing sample should be changed
                                        if t >= shadowing_next_tick:
                                            shadowing_sample_index = shadowing_sample_index + 1
                                            shadowing_next_tick = t + shadowing_coherence_time_tick_duration

                                        data_rx_power = thz_channel.get_3gpp_prx_db(
                                            tx=ue_array[data_rx_at_ue_ue_id[index]], rx=ue,
                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                            tx_rx_distance_m=tx_rx_distance_m,
                                            apply_fading=apply_fading,
                                            bandwidth_hz=bandwidth_hz,
                                            clutter_density=clutter_density,
                                            input_shadowing_sample_index=shadowing_sample_index,
                                            antenna_gain_model=antenna_gain_model,
                                            use_huawei_measurements=use_huawei_measurements,
                                            input_average_clutter_height_m=average_machine_height_m,
                                            los_cond='ue_ue')

                                        # Compute the SNR between the current receiving UE and the transmitting UE
                                        snr_db = thz_channel.get_3gpp_snr_db(
                                            tx=ue_array[data_rx_at_ue_ue_id[index]], rx=ue,
                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                            tx_rx_distance_m=tx_rx_distance_m,
                                            apply_fading=apply_fading,
                                            bandwidth_hz=bandwidth_hz,
                                            clutter_density=clutter_density,
                                            input_shadowing_sample_index=shadowing_sample_index,
                                            antenna_gain_model=antenna_gain_model,
                                            use_huawei_measurements=use_huawei_measurements,
                                            input_average_clutter_height_m=average_machine_height_m,
                                            los_cond='ue_ue')
                                        
                                        sir_dB = None
                                        n_interferers = 0

                                        # this method takes in input both the current UE_ID that has received a data and both the
                                        # ID of the UE that has sent the data
                                        # -> need to check if there is another UE != from these two UEs that has TX a DATA or an ACK
                                        ue.ues_colliding_at_ue.clear()

                                        ue.ues_colliding_at_ue = check_collision(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_ue_id=ue.get_ue_id(),
                                            input_t_start_rx=data_rx_at_ue_starting_tick,
                                            input_t_end_rx=data_rx_at_ue_ending_tick, input_tx=None,
                                            input_ue_id_rx=data_rx_at_ue_ue_id[index], ues_colliding=ue.ues_colliding_at_ue)

                                        # capture effect -> we are in BO so the reference UE can RX
                                        #  DATA/ACK only from other UEs

                                        useful_rx_power_db = data_rx_power
                                        add_interferer = True
                                        if len(ue.ues_interfering_at_ue) > 0:
                                            for i in range(len(ue.ues_interfering_at_ue)):
                                                if f'UE_{data_rx_at_ue_ue_id[index]}' == \
                                                        ue.ues_interfering_at_ue[i][0] and \
                                                        t == ue.ues_interfering_at_ue[i][1]:
                                                    add_interferer = False
                                                    # the useful user will become an interferer for the next reception,
                                                    # so save the ID and the current ending tick of this reception
                                        if add_interferer is True:
                                            ue.ues_interfering_at_ue.append((f'UE_{data_rx_at_ue_ue_id[index]}',
                                                                             data_rx_at_ue_starting_tick,
                                                                             data_rx_at_ue_ending_tick))
                                        interference_rx_power = 0
                                        if len(ue.ues_colliding_at_ue) > 0:
                                            for user in ue_array:
                                                for i in range(len(ue.ues_colliding_at_ue)):
                                                    if f'UE_{user.get_ue_id()}' == ue.ues_colliding_at_ue[i][0] and \
                                                            user.get_ue_id() != ue.get_ue_id() and \
                                                            user.get_ue_id() != data_rx_at_ue_ue_id[index]:
                                                        # to compute the portion of data overlapped:
                                                        # t_j = (t_end_current - t_start_interferer) /
                                                        # (t_end_current - t_start_current)
                                                        if ue.ues_colliding_at_ue[i][1] < data_rx_at_ue_ending_tick < \
                                                                ue.ues_colliding_at_ue[i][2]:
                                                            t_overlap = ((data_rx_at_ue_ending_tick -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (data_rx_at_ue_ending_tick -
                                                                          data_rx_at_ue_starting_tick))
                                                        else:
                                                            t_overlap = ((ue.ues_colliding_at_ue[i][2] -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (data_rx_at_ue_ending_tick -
                                                                          data_rx_at_ue_starting_tick))
                                                        n_interferers += 1
                                                        tx_rx_distance_m = compute_distance_m(tx=user, rx=ue)
                                                        interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                            tx=user, rx=ue,
                                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                                            tx_rx_distance_m=tx_rx_distance_m,
                                                            apply_fading=apply_fading,
                                                            bandwidth_hz=bandwidth_hz,
                                                            clutter_density=clutter_density,
                                                            input_shadowing_sample_index=shadowing_sample_index,
                                                            antenna_gain_model=antenna_gain_model,
                                                            use_huawei_measurements=use_huawei_measurements,
                                                            input_average_clutter_height_m=average_machine_height_m,
                                                            los_cond='ue_ue')
                                            for i in range(len(ue.ues_colliding_at_ue)):
                                                if 'BS' == ue.ues_colliding_at_ue[i][0] and 'BS' != data_rx_at_ue_ue_id[index]:
                                                    # to compute the portion of data overlapped:
                                                    # t_j = (t_end_current - t_start_interferer) /
                                                    # (t_end_current - t_start_current)
                                                    if ue.ues_colliding_at_ue[i][1] < data_rx_at_ue_ending_tick < \
                                                            ue.ues_colliding_at_ue[i][2]:
                                                        t_overlap = ((data_rx_at_ue_ending_tick -
                                                                      ue.ues_colliding_at_ue[i][1]) /
                                                                     (data_rx_at_ue_ending_tick -
                                                                      data_rx_at_ue_starting_tick))
                                                    else:
                                                        t_overlap = ((ue.ues_colliding_at_ue[i][2] -
                                                                      ue.ues_colliding_at_ue[i][1]) /
                                                                     (data_rx_at_ue_ending_tick -
                                                                      data_rx_at_ue_starting_tick))
                                                    n_interferers += 1
                                                    tx_rx_distance_m = compute_distance_m(tx=bs, rx=ue)
                                                    interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                        tx=bs, rx=ue,
                                                        carrier_frequency_ghz=carrier_frequency_ghz,
                                                        tx_rx_distance_m=tx_rx_distance_m,
                                                        apply_fading=apply_fading,
                                                        bandwidth_hz=bandwidth_hz,
                                                        clutter_density=clutter_density,
                                                        input_shadowing_sample_index=shadowing_sample_index,
                                                        antenna_gain_model=antenna_gain_model,
                                                        use_huawei_measurements=use_huawei_measurements,
                                                        input_average_clutter_height_m=average_machine_height_m,
                                                        los_cond='bs_ue')

                                        if len(ue.ues_interfering_at_ue) > 0:
                                            # for the interfering users (whose that before where useful user),
                                            # I have to check if their ending tick of ACK or DATA is between the
                                            # staring and the ending tick of the actual RX DATA/ACK
                                            # If Yes -> it is an interferer
                                            # If No -> remove from the list of interferers.
                                            copy_of_list = deepcopy(ue.ues_interfering_at_ue)
                                            for user in ue_array:
                                                for i in range(len(copy_of_list)):
                                                    if user.get_ue_id() != ue.get_ue_id() and \
                                                            user.get_ue_id() != data_rx_at_ue_ue_id[index]:
                                                        if f'UE_{user.get_ue_id()}' == copy_of_list[i][0]:
                                                            if data_rx_at_ue_starting_tick < copy_of_list[i][2]:

                                                                # to compute the portion of data overlapped:

                                                                if data_rx_at_ue_starting_tick > copy_of_list[i][1]:
                                                                    t_overlap = ((copy_of_list[i][2] -
                                                                                  data_rx_at_ue_starting_tick) /
                                                                                 (data_rx_at_ue_ending_tick -
                                                                                  data_rx_at_ue_starting_tick))
                                                                else:
                                                                    t_overlap = ((copy_of_list[i][2] -
                                                                                  copy_of_list[i][1]) /
                                                                                 (data_rx_at_ue_ending_tick -
                                                                                  data_rx_at_ue_starting_tick))
                                                                n_interferers += 1
                                                                tx_rx_distance_m = compute_distance_m(tx=user, rx=ue)

                                                                interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                                    tx=user, rx=ue,
                                                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                                                    tx_rx_distance_m=tx_rx_distance_m,
                                                                    apply_fading=apply_fading,
                                                                    bandwidth_hz=bandwidth_hz,
                                                                    clutter_density=clutter_density,
                                                                    input_shadowing_sample_index=shadowing_sample_index,
                                                                    antenna_gain_model=antenna_gain_model,
                                                                    use_huawei_measurements=use_huawei_measurements,
                                                                    input_average_clutter_height_m=average_machine_height_m,
                                                                    los_cond='ue_ue')
                                                            elif t >= copy_of_list[i][2]:
                                                                ue.ues_interfering_at_ue.remove(
                                                                    (f'UE_{user.get_ue_id()}', copy_of_list[i][1],
                                                                     copy_of_list[i][2]))
                                            for i in range(len(copy_of_list)):
                                                if 'BS' != data_rx_at_ue_ue_id[index]:
                                                    if 'BS' == copy_of_list[i][0]:

                                                        if data_rx_at_ue_starting_tick < copy_of_list[i][2]:
                                                            # to compute the portion of data overlapped:

                                                            if data_rx_at_ue_starting_tick > copy_of_list[i][1]:
                                                                t_overlap = ((copy_of_list[i][2] -
                                                                              data_rx_at_ue_starting_tick) /
                                                                             (data_rx_at_ue_ending_tick -
                                                                              data_rx_at_ue_starting_tick))
                                                            else:
                                                                t_overlap = ((copy_of_list[i][2] -
                                                                              copy_of_list[i][1]) /
                                                                             (data_rx_at_ue_ending_tick -
                                                                              data_rx_at_ue_starting_tick))
                                                            n_interferers += 1
                                                            tx_rx_distance_m = compute_distance_m(tx=bs, rx=ue)

                                                            interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                                tx=bs, rx=ue,
                                                                carrier_frequency_ghz=carrier_frequency_ghz,
                                                                tx_rx_distance_m=tx_rx_distance_m,
                                                                apply_fading=apply_fading,
                                                                bandwidth_hz=bandwidth_hz,
                                                                clutter_density=clutter_density,
                                                                input_shadowing_sample_index=shadowing_sample_index,
                                                                antenna_gain_model=antenna_gain_model,
                                                                use_huawei_measurements=use_huawei_measurements,
                                                                input_average_clutter_height_m=average_machine_height_m,
                                                                los_cond='bs_ue')
                                                        elif t >= copy_of_list[i][2]:
                                                            ue.ues_interfering_at_ue.remove((copy_of_list[i][0],
                                                                                             copy_of_list[i][1],
                                                                                             copy_of_list[i][2]))
                                        if interference_rx_power == 0:
                                            sinr_db = snr_db

                                        else:

                                            noise_power_dbw = thz_channel.get_thermal_noise_power_dbw(
                                                input_noise_figure=noise_figure_ue, bandwidth_hz=bandwidth_hz)
                                            noise_power = 10 ** (noise_power_dbw / 10)
                                            noise_plus_interference = noise_power + interference_rx_power
                                            useful_rx_power = 10 ** (useful_rx_power_db / 10)
                                            sinr = useful_rx_power / noise_plus_interference
                                            sinr_db = 10 * log10(sinr)

                                        if sinr_db >= sinr_th_db:
                                            success = True
                                        else:
                                            success = False

                                        ue.n_interfering.append(n_interferers)

                                        if success and ((len(ue.ul_buffer.buffer_packet_list) < \
                                             max_n_packets_to_be_forwarded + 1 and ue.check_generated_packet_present() is True) \
                                            or (
                                            (len(ue.ul_buffer.buffer_packet_list) < \
                                             max_n_packets_to_be_forwarded and ue.check_generated_packet_present() is False))):

                                            ue.packet_forwarding.append(packet_id_rx_from_ue[index])

                                            counter = 0
                                            for packet in ue.ul_buffer.buffer_packet_list:
                                                if packet.get_generated_by_ue() == ue.ue_id:
                                                    counter += 1
                                            if counter > 0:
                                                total_buffer_size = max_n_packets_to_be_forwarded + 1
                                            else:
                                                total_buffer_size = max_n_packets_to_be_forwarded

                                            if data_rx_at_ue_size_bytes[index] > 0 and len(ue.ul_buffer.buffer_packet_list) < \
                                                    total_buffer_size:

                                                for user in ue_array:
                                                    if data_rx_at_ue_ue_id[index] == user.get_ue_id():
                                                        if len(user.buffer_packet_sent) > 0:
                                                            for packet in user.buffer_packet_sent:
                                                                if packet.packet_id == packet_id_rx_from_ue[index]:
                                                                    # -1 -> pacch rx in broadcast
                                                                    # or if I have rx a packet in unicast from the UE
                                                                    # that has selected that corresponding UE as relay
                                                                    if packet.address == str(
                                                                            ue.get_ue_id()) or packet.address == "-1":
                                                                        ue.designated_rx = True
                                                                        break

                                                            if ue.designated_rx:

                                                                # reset the action variables
                                                                if enable_print:
                                                                    print("UE ", ue.get_ue_id(),
                                                                          " has received a packet from UE ",
                                                                          data_rx_at_ue_ue_id[index])

                                                                ue.designated_rx = False  # mi serve

                                                                first_entry_in_loop = True
                                                                for n_packet in range(len(user.buffer_packet_sent)):
                                                                    if packet_id_rx_from_ue[index] == user.buffer_packet_sent[n_packet].packet_id:
                                                                        if len(ue.ul_buffer.buffer_packet_list) < total_buffer_size:
                                                                            # Check if the packet is already in the queue
                                                                            packet_already_in_queue = False
                                                                            for packet in ue.ul_buffer.buffer_packet_list:
                                                                                if (packet.get_generated_by_ue() ==
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_generated_by_ue() and
                                                                                        packet.get_packet_id_generator() ==
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_packet_id_generator()):
                                                                                    packet_already_in_queue = True

                                                                            # Check if the packet is generated by the UE itself
                                                                            packet_generated_by_ue_itself = False
                                                                            if user.buffer_packet_sent[
                                                                                n_packet].get_generated_by_ue() == ue.get_ue_id():
                                                                                packet_generated_by_ue_itself = True

                                                                            # If the packet is generated by the UE itself, force a broadcast action
                                                                            if packet_generated_by_ue_itself and ue.obs[0][-1] == 0:
                                                                                ue.next_action = 3
                                                                            # Check if the packet exceeded the hop limit
                                                                            packet_out_of_hop_limit = False
                                                                            if user.buffer_packet_sent[
                                                                                n_packet].get_hop_count() >= hop_limit:
                                                                                packet_out_of_hop_limit = True

                                                                            if (packet_already_in_queue is False and
                                                                                    packet_generated_by_ue_itself is False
                                                                                    and packet_out_of_hop_limit is False):
                                                                                if first_entry_in_loop:
                                                                                    first_entry_in_loop = False
                                                                                    ue.forward_in_bo = True
                                                                                    ue.packet_forward = True

                                                                                if enable_print:
                                                                                    print("UE ", ue.get_ue_id(),
                                                                                          " is forwarding packet: ",
                                                                                          user.buffer_packet_sent[
                                                                                              n_packet].get_id(),
                                                                                          " received from UE: ",
                                                                                          data_rx_at_ue_ue_id[index],
                                                                                          "with hop_count: ",
                                                                                          user.buffer_packet_sent[
                                                                                              n_packet].hop_count)
                                                                                ue.n_forwarding += 1
                                                                                # Successful data reception, add the data in the queue and transmit the ACK
                                                                                ue.add_new_packet(current_tick=t,
                                                                                                  input_enable_print=enable_print,
                                                                                                  input_data_to_be_forwarded_bool=True,
                                                                                                  input_packet_size_bytes=user.buffer_packet_sent[
                                                                                                      n_packet].packet_size,
                                                                                                  input_simulation_tick_duration=simulator_tick_duration_s,
                                                                                                  data_rx_from_ue=data_rx_at_ue_ue_id[index],
                                                                                                  packet_id_rx_from_ue=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_id(),
                                                                                                  packet_generated_by_ue=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_generated_by_ue(),
                                                                                                  packet_id_generator=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_packet_id_generator(),
                                                                                                  packet_hop_count=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_hop_count(),
                                                                                                  packet_address=(
                                                                                                      ue.get_unicast_rx_address() if ue.get_broadcast_bool() is False else "-1"), generation_time=user.buffer_packet_sent[
                                                                                                      n_packet].get_generated_by_ue_time_instant_tick())

                                                                                ue.update_num_tx(
                                                                                    input_packet_id=ue.ul_buffer.get_last_packet().get_id())

                                                                                ue.dict_data_rx_during_bo[data_rx_at_ue_ue_id[index]].append(user.buffer_packet_sent[n_packet].get_id())

                                                                                if user.buffer_packet_sent[
                                                                                    n_packet].get_id() not in \
                                                                                        ue.dict_ack_sent_from_ue[
                                                                                            data_rx_at_ue_ue_id[index]]:
                                                                                    ue.dict_ack_sent_from_ue[
                                                                                        data_rx_at_ue_ue_id[index]].append(
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_id())
                                                                            if (packet_already_in_queue is True and
                                                                                    packet_generated_by_ue_itself is False):
                                                                                ue.packet_forward = True
                                                                                ue.dict_data_rx_during_bo[
                                                                                    data_rx_at_ue_ue_id[index]].append(
                                                                                    user.buffer_packet_sent[
                                                                                        n_packet].get_id())

                                                                                if data_rx_at_ue_ue_id[
                                                                                    index] not in ue.data_rx_at_ue_ue_id_list:
                                                                                    ue.data_rx_at_ue_ue_id_list.append(
                                                                                        data_rx_at_ue_ue_id[index])

                                                                                if user.buffer_packet_sent[
                                                                                    n_packet].get_id() not in \
                                                                                        ue.dict_ack_sent_from_ue[
                                                                                            data_rx_at_ue_ue_id[index]]:
                                                                                    ue.dict_ack_sent_from_ue[
                                                                                        data_rx_at_ue_ue_id[
                                                                                            index]].append(
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_id())


                                                                if ue.forward_in_bo:

                                                                    ue.forward_in_bo = False

                                                                    if data_rx_at_ue_ue_id[
                                                                        index] not in ue.data_rx_at_ue_ue_id_list:
                                                                        ue.data_rx_at_ue_ue_id_list.append(data_rx_at_ue_ue_id[index])

                                                                    if np.sum(ue.obs[1]) > 0:
                                                                        if ue.get_ue_id() < data_rx_at_ue_ue_id[index]:
                                                                            ue.set_obs_update(
                                                                                input_data_rx_at_ue_tx_index=
                                                                                data_rx_at_ue_ue_id[index] - 1,
                                                                                input_rx_power=data_rx_power)

                                                                        else:
                                                                            ue.set_obs_update(
                                                                                input_data_rx_at_ue_tx_index=
                                                                                data_rx_at_ue_ue_id[index],
                                                                                input_rx_power=data_rx_power)

                                                                else:
                                                                    ue.set_state_duration(
                                                                        input_ticks=ue.get_state_final_tick())
                                                            else:
                                                                ue.set_state_duration(input_ticks=ue.get_state_final_tick())

                                                    else:
                                                        # Remain in BO until the end
                                                        ue.set_state_duration(input_ticks=ue.get_state_final_tick())
                                            else:
                                                # Remain in BO until the end
                                                ue.set_state_duration(input_ticks=ue.get_state_final_tick())
                                        else:

                                            # Remain in BO until the end
                                            ue.set_state_duration(input_ticks=ue.get_state_final_tick())
                                        if len(simulator_timing_structure[f'UE_{ue.get_ue_id()}']['DATA_RX'][
                                                   f'UE_{data_rx_at_ue_ue_id[index]}']) > 1:
                                            # Update the timing structure to reset this reception
                                            remove_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_rx_key=f'UE_{ue.get_ue_id()}',
                                                input_type_key='DATA_RX',
                                                input_tx_key=f'UE_{data_rx_at_ue_ue_id[index]}')

                            else:
                                if ue.get_state_duration() == ue.get_state_final_tick():
                                    go_in_tx_data_bool = True
                                else:
                                    # Remain in BO until the end
                                    ue.set_state_duration(input_ticks=ue.get_state_final_tick())
                            # If the UE has received during BO packets from other UEs to be forwarded, at the end of BO
                            # has to go in ACK TX for that UEs
                            if ue.packet_forward:
                                if t == ue.get_state_final_tick():

                                    go_in_tx_ack(input_ue=ue,
                                                 input_ack_duration_tick=len(ue.data_rx_at_ue_ue_id_list) * t_ack_tick,
                                                 current_tick=t,
                                                 input_enable_print=enable_print)

                                    j = - 1

                                    for ue_id in ue.data_rx_at_ue_ue_id_list:

                                        j += 1
                                        # Update the timing structure for each UE
                                        for index in range(len(ue.dict_data_rx_during_bo[ue_id])):
                                            for other_ue in ue_array:
                                                if other_ue != ue:
                                                    insert_item_in_timing_structure(
                                                        input_simulator_timing_structure=simulator_timing_structure,
                                                        input_starting_tick=t + ue.get_prop_delay_to_ue_tick(
                                                            input_ue_id=other_ue.get_ue_id()) + j * t_ack_tick,
                                                        input_final_tick=
                                                        t + ue.get_prop_delay_to_ue_tick(
                                                            input_ue_id=other_ue.get_ue_id()) + t_ack_tick + j * t_ack_tick,
                                                        input_third_field=ue_id,
                                                        input_fourth_field=
                                                        ue.dict_data_rx_during_bo[ue_id][index],
                                                        input_rx_key=f'UE_{other_ue.get_ue_id()}',
                                                        input_type_key='ACK_RX',
                                                        input_tx_key=f'UE_{ue.get_ue_id()}',
                                                    )

                                            # Update the timing structure for the BS
                                            insert_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_starting_tick=t + ue.get_prop_delay_to_bs_tick() + j * t_ack_tick,
                                                input_final_tick=
                                                t + ue.get_prop_delay_to_bs_tick() + t_ack_tick + j * t_ack_tick,
                                                input_third_field=ue_id,
                                                input_fourth_field=
                                                ue.dict_data_rx_during_bo[ue_id][index],
                                                input_rx_key='BS',
                                                input_type_key='ACK_RX',
                                                input_tx_key=f'UE_{ue.get_ue_id()}',
                                            )

                                    ue.ues_colliding_at_ue.clear()
                                    ue.data_rx_at_ue_ue_id_list.clear()
                                    if len(ue.ues_interfering_at_ue) > 0:
                                        ue.ues_interfering_at_ue.clear()

                                    ue.previous_state = 'BO'
                                    ue.packet_forward = False

                                else:
                                    # Remain in BO until the end
                                    ue.set_state_duration(input_ticks=ue.get_state_final_tick())
                            elif go_in_tx_data_bool and ue.forward_in_bo is False:
                                # if the UE after BO has not RX anything, it has to transmit all the packets it has in the queue
                                tx_data_size_bytes = go_in_tx_data(input_ue=ue, current_tick=t,
                                                                   input_enable_print=enable_print)
                                ue.ues_colliding_at_ue.clear()
                                if len(ue.ues_interfering_at_ue) > 0:
                                    ue.ues_interfering_at_ue.clear()
                                ue.check_last_round = True
                                ue.update_n_data_tx(input_enable_print=enable_print)
                                if star_topology is False:
                                    packet_id_to_be_sent = -1
                                    for packet in ue.ul_buffer.buffer_packet_list:
                                        if packet.get_data_unicast() is False or \
                                                packet.get_data_to_be_forwarded_bool() is False:
                                            packet_id_to_be_sent = packet.get_id()

                                    if packet_id_to_be_sent != -1:
                                        choose_next_action_tb_no_RL(input_ue=ue, input_enable_print=enable_print)

                                        starting_tick = t
                                        # Update the timing structure for UEs
                                        # fourth field is the ID of the packet received from a UE
                                        for packet in ue.ul_buffer.buffer_packet_list:
                                            t_data_ns = round(
                                                (packet.packet_size * 8 * 1e-9) / bs.get_bit_rate_gbits(),
                                                11)  # 1.6 ns
                                            t_data_tick = round(t_data_ns / simulator_tick_duration_s)
                                            for other_ue in ue_array:
                                                if other_ue != ue:
                                                    insert_item_in_timing_structure(
                                                        input_simulator_timing_structure=simulator_timing_structure,
                                                        input_starting_tick=starting_tick + ue.get_prop_delay_to_ue_tick(
                                                            input_ue_id=other_ue.get_ue_id()),
                                                        input_final_tick=starting_tick + t_data_tick +
                                                                         ue.get_prop_delay_to_ue_tick(
                                                            input_ue_id=other_ue.get_ue_id()),
                                                        input_third_field=packet.packet_size,
                                                        input_fourth_field=packet.packet_id,
                                                        input_rx_key=f'UE_{other_ue.get_ue_id()}',
                                                        input_type_key='DATA_RX',
                                                        input_tx_key=f'UE_{ue.get_ue_id()}')
                                            starting_tick += t_data_tick

                                    starting_tick = t
                                    for packet in ue.ul_buffer.buffer_packet_list:
                                        t_data_ns = round((packet.packet_size * 8 * 1e-9) / bs.get_bit_rate_gbits(), 11)
                                        t_data_tick = round(t_data_ns / simulator_tick_duration_s)
                                        insert_item_in_timing_structure(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_starting_tick=starting_tick + ue.get_prop_delay_to_bs_tick(),
                                            input_final_tick=
                                            starting_tick + t_data_tick + ue.get_prop_delay_to_bs_tick(),
                                            input_third_field=packet.packet_size,
                                            input_fourth_field=packet.packet_id,
                                            input_rx_key='BS',
                                            input_type_key='DATA_RX',
                                            input_tx_key=f'UE_{ue.get_ue_id()}')
                                        starting_tick += t_data_tick
                                else:
                                    starting_tick = t
                                    for packet in ue.ul_buffer.buffer_packet_list:
                                        t_data_ns = round((packet.packet_size * 8 * 1e-9) / bs.get_bit_rate_gbits(),11)
                                        t_data_tick = round(t_data_ns / simulator_tick_duration_s)
                                        # Update the timing dictionary with the RX instants at the BS
                                        insert_item_in_timing_structure(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_starting_tick=t + ue.get_prop_delay_to_bs_tick(),
                                            input_final_tick=
                                            ue.get_state_duration() + ue.get_prop_delay_to_bs_tick(),
                                            input_third_field=packet.packet_size,
                                            input_fourth_field=packet.packet_id,
                                            input_rx_key='BS',
                                            input_type_key='DATA_RX',
                                            input_tx_key=f'UE_{ue.get_ue_id()}')
                                        starting_tick += t_data_tick

                        elif ue.get_state() == 'TX_ACK':
                            # After ACK transmission go in TX_DATA
                            ue.list_data_rx_during_wait_ack.clear()
                            ue.list_data_generated_during_wait_ack.clear()
                            for other_ue in ue.dict_data_rx_during_wait_ack:
                                ue.dict_data_rx_during_wait_ack[other_ue].clear()
                            ue.list_data_rx_from_ue_id.clear()

                            for other_ue in ue.dict_data_rx_during_bo:
                                ue.dict_data_rx_during_bo[other_ue].clear()

                            # If the UE before was in WAIT ACK, then it has to go back to BO to try another DATA transmission

                            if ue.previous_state == 'WAIT_ACK':

                                if ue.check_generated_packet_present() is False:
                                    ue.new_action_bool = True

                                # Go in BO to avoid synchronism with other UEs when there are collisions and no ACKs are received
                                backoff_duration_tick = get_backoff_duration(input_ue=ue,
                                                                             input_contention_window_int=
                                                                             contention_window_int,
                                                                             input_t_backoff_tick=t_backoff_tick,
                                                                             input_max_prop_delay_tick=
                                                                             max_prop_delay_tick)
                                go_in_backoff(input_ue=ue, current_tick=t,
                                              input_backoff_duration_tick=backoff_duration_tick,
                                              input_enable_print=enable_print)

                                ue.first_bo_entry = True

                            # If the UE before was in BO, then it has to go in DATA TX and select the proper next hop
                            elif ue.previous_state == 'BO':
                                ue.check_last_round = True
                                tx_data_size_bytes = go_in_tx_data(input_ue=ue, current_tick=t,
                                                                   input_enable_print=enable_print)
                                ue.update_n_data_tx(input_enable_print=enable_print)

                                # Select the proper action : Broadcast or Unicast based on the neighbor table status

                                choose_next_action_tb_no_RL(input_ue=ue, input_enable_print=enable_print)

                                starting_tick = t
                                # Update the timing structure for UEs
                                for packet in ue.ul_buffer.buffer_packet_list:
                                    t_data_ns = round((packet.packet_size * 8 * 1e-9) / bs.get_bit_rate_gbits(),11)
                                    t_data_tick = round(t_data_ns / simulator_tick_duration_s)
                                    for other_ue in ue_array:
                                        if other_ue != ue:
                                            insert_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_starting_tick=starting_tick + ue.get_prop_delay_to_ue_tick(
                                                    input_ue_id=other_ue.get_ue_id()),
                                                input_final_tick=starting_tick + t_data_tick + ue.get_prop_delay_to_ue_tick(
                                                    input_ue_id=other_ue.get_ue_id()),
                                                input_third_field=packet.packet_size,
                                                input_fourth_field=packet.packet_id,
                                                input_rx_key=f'UE_{other_ue.get_ue_id()}',
                                                input_type_key='DATA_RX',
                                                input_tx_key=f'UE_{ue.get_ue_id()}'
                                            )
                                    starting_tick += t_data_tick

                                starting_tick = t
                                for packet in ue.ul_buffer.buffer_packet_list:
                                    t_data_ns = round((packet.packet_size * 8 * 1e-9) / bs.get_bit_rate_gbits(),11)
                                    t_data_tick = round(t_data_ns / simulator_tick_duration_s)
                                    # Update the timing structure for the BS
                                    insert_item_in_timing_structure(
                                        input_simulator_timing_structure=simulator_timing_structure,
                                        input_starting_tick=starting_tick + ue.get_prop_delay_to_bs_tick(),
                                        input_final_tick=
                                        starting_tick + t_data_tick + ue.get_prop_delay_to_bs_tick(),
                                        input_third_field=packet.packet_size,
                                        input_fourth_field=packet.packet_id,
                                        input_rx_key='BS',
                                        input_type_key='DATA_RX',
                                        input_tx_key=f'UE_{ue.get_ue_id()}')
                                    starting_tick += t_data_tick

                        elif ue.get_state() == 'TX_DATA':
                            # Go in WAIT_ACK and update the state duration
                            wait_ack_duration_tick = t_ack_tick + 2 * max_prop_delay_tick
                            go_in_wait_ack(input_ue=ue, current_tick=t,
                                           input_wait_ack_duration_tick=wait_ack_duration_tick,
                                           input_enable_print=enable_print)

                            ue.first_entry = True

                        elif ue.get_state() == 'WAIT_ACK':
                            remain_in_wait_ack = False  # True if the UE has to remain in WAIT_ACK
                            go_in_bo_bool = False  # True if the UE has to go to BO
                            go_in_idle_bool = False  # True if the UE has to go to IDLE
                            ack_received_old_packet = True
                            ack_received = True

                            if ue.first_entry is True:
                                ue.copy_buffer_packet_list = deepcopy(ue.ul_buffer.buffer_packet_list)
                                ue.first_entry = False

                            # Check if during WAIT_ACK the UE has received some ACKs
                            if (ue.get_state_duration() == ue.get_state_final_tick() and
                                    ue.get_reception_during_wait_bool() is False):
                                # The WAIT_ACK state is finished without receiving an ACK
                                # Update the number of transmission attempts and remove all packets that have reached
                                # their maximum number of transmission attempts
                                if enable_print:
                                    print("UE ", ue.get_ue_id(), " has not received an ACK or a DATA during WAIT_ACK.")
                                ue.set_retransmission_packets(retransmission_bool=True)

                            else:
                                # with BUFFER implementation a UE can receive an ACK for more than one
                                #  packet of the buffer -> in the simulator structure there are more than one fields
                                #  with the same starting and ending ticks but different packet IDs
                                # the UE can have received a DATA or an ACK during WAIT ACK
                                if ue.ack_rx_during_wait_ack is True:
                                    ue.ack_rx_during_wait_ack = False

                                    # The UE has received an ACK during WAIT_ACK. Check if the reception is successful
                                    # Find the corresponding transmission

                                    # ack_rx_at_ue_tx_id_str: it's a list of string indicating the IDs of the UE that is
                                    # transmitting the ACK (or the BS)
                                    # ack_rx_at_ue_rx_id_int: ID of the UE that is receiving the ACK
                                    # ack_rx_id: list of the packet ID for which the ACK is sent
                                    ue.set_reception_during_wait_bool(input_data_rx_bool=False)
                                    (ack_rx_at_ue_starting_tick, ack_rx_at_ue_ending_tick,ack_rx_sources, ack_rx_dest,
                                     ack_rx_id) = (find_ack_rx_times_tick(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            current_tick=t,
                                            input_ue_id=ue.get_ue_id()))
                                    ue.ack_rx_with_success = False

                                    for index in range(len(ack_rx_sources)):

                                        sorg_ack = ack_rx_sources[index]
                                        dest_ack = ack_rx_dest[index]

                                        if sorg_ack is None:
                                            print("current tick: ", t)

                                        # Understand who is the transmitter: UE ID or BS
                                        if star_topology is False:
                                            if sorg_ack.startswith('UE'):
                                                tx = ue_array[int(sorg_ack[3:])]
                                                los_cond = 'ue_ue'
                                            else:
                                                tx = bs
                                                los_cond = 'bs_ue'
                                        else:
                                            tx = bs
                                            los_cond = 'bs_ue'

                                        # Compute the tx-rx distance
                                        tx_rx_distance_m = compute_distance_m(tx=tx, rx=ue)

                                        # Check if the shadowing sample should be changed
                                        if t >= shadowing_next_tick:
                                            shadowing_sample_index = shadowing_sample_index + 1
                                            shadowing_next_tick = t + shadowing_coherence_time_tick_duration

                                        ack_rx_power = thz_channel.get_3gpp_prx_db(
                                            tx=tx, rx=ue,
                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                            tx_rx_distance_m=tx_rx_distance_m,
                                            apply_fading=apply_fading,
                                            bandwidth_hz=bandwidth_hz,
                                            clutter_density=clutter_density,
                                            input_shadowing_sample_index=shadowing_sample_index,
                                            antenna_gain_model=antenna_gain_model,
                                            use_huawei_measurements=use_huawei_measurements,
                                            input_average_clutter_height_m=average_machine_height_m,
                                            los_cond=los_cond)

                                        # Compute the SNR between the current receiving UE and the transmitting UE
                                        snr_db = thz_channel.get_3gpp_snr_db(
                                            tx=tx, rx=ue,
                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                            tx_rx_distance_m=tx_rx_distance_m,
                                            apply_fading=apply_fading,
                                            bandwidth_hz=bandwidth_hz,
                                            clutter_density=clutter_density,
                                            input_shadowing_sample_index=shadowing_sample_index,
                                            antenna_gain_model=antenna_gain_model,
                                            use_huawei_measurements=use_huawei_measurements,
                                            input_average_clutter_height_m=average_machine_height_m,
                                            los_cond=los_cond)

                                        sir_dB = None
                                        n_interferers = 0

                                        # this method takes in input both the current UE_ID that has received a data
                                        # and both the ID of the UE that has sent the data
                                        # -> need to check if there is another UE != from these two UEs that has TX a
                                        # DATA or an ACK
                                        ue.ues_colliding_at_ue.clear()

                                        ue.ues_colliding_at_ue = check_collision(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_ue_id=ue.get_ue_id(),
                                            input_t_start_rx=ack_rx_at_ue_starting_tick,
                                            input_t_end_rx=ack_rx_at_ue_ending_tick, input_tx=sorg_ack,
                                            input_ue_id_rx=ue.get_ue_id(),
                                            ues_colliding=ue.ues_colliding_at_ue)

                                        useful_rx_power_db = ack_rx_power
                                        add_interferer = True
                                        if len(ue.ues_interfering_at_ue) > 0:
                                            for i in range(len(ue.ues_interfering_at_ue)):
                                                if sorg_ack == \
                                                        ue.ues_interfering_at_ue[i][0] and \
                                                        t == ue.ues_interfering_at_ue[i][1]:
                                                    add_interferer = False
                                                    # the useful user will become an interferer for the next reception,
                                                    # so save the ID and the current ending tick of this reception
                                        if add_interferer is True:
                                            ue.ues_interfering_at_ue.append((sorg_ack, ack_rx_at_ue_starting_tick,
                                                                             ack_rx_at_ue_ending_tick))
                                        interference_rx_power = 0
                                        if len(ue.ues_colliding_at_ue) > 0:
                                            for user in ue_array:
                                                for i in range(len(ue.ues_colliding_at_ue)):
                                                    if f'UE_{user.get_ue_id()}' == ue.ues_colliding_at_ue[i][0] and \
                                                            user.get_ue_id() != ue.get_ue_id() and \
                                                            f'UE_{user.get_ue_id()}' != sorg_ack:
                                                        # to compute the portion of data overlapped:
                                                        # t_j = (t_end_current - t_start_interferer) /
                                                        # (t_end_current - t_start_current)
                                                        if ue.ues_colliding_at_ue[i][1] < ack_rx_at_ue_ending_tick < \
                                                                ue.ues_colliding_at_ue[i][2]:
                                                            t_overlap = ((ack_rx_at_ue_ending_tick -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (ack_rx_at_ue_ending_tick -
                                                                          ack_rx_at_ue_starting_tick))
                                                        else:
                                                            t_overlap = ((ue.ues_colliding_at_ue[i][2] -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (ack_rx_at_ue_ending_tick -
                                                                          ack_rx_at_ue_starting_tick))
                                                        n_interferers += 1
                                                        tx_rx_distance_m = compute_distance_m(tx=user, rx=ue)
                                                        interference_rx_power +=  t_overlap * thz_channel.get_3gpp_prx_lin(
                                                            tx=user, rx=ue,
                                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                                            tx_rx_distance_m=tx_rx_distance_m,
                                                            apply_fading=apply_fading,
                                                            bandwidth_hz=bandwidth_hz,
                                                            clutter_density=clutter_density,
                                                            input_shadowing_sample_index=shadowing_sample_index,
                                                            antenna_gain_model=antenna_gain_model,
                                                            use_huawei_measurements=use_huawei_measurements,
                                                            input_average_clutter_height_m=average_machine_height_m,
                                                            los_cond='ue_ue')
                                            for i in range(len(ue.ues_colliding_at_ue)):
                                                if 'BS' == ue.ues_colliding_at_ue[i][0]:
                                                    if 'BS' != sorg_ack:
                                                        # to compute the portion of data overlapped:
                                                        # t_j = (t_end_current - t_start_interferer) /
                                                        # (t_end_current - t_start_current)
                                                        if ue.ues_colliding_at_ue[i][1] < ack_rx_at_ue_ending_tick < \
                                                                ue.ues_colliding_at_ue[i][2]:
                                                            t_overlap = ((ack_rx_at_ue_ending_tick -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (ack_rx_at_ue_ending_tick -
                                                                          ack_rx_at_ue_starting_tick))
                                                        else:
                                                            t_overlap = ((ue.ues_colliding_at_ue[i][2] -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (ack_rx_at_ue_ending_tick -
                                                                          ack_rx_at_ue_starting_tick))
                                                        n_interferers += 1
                                                        tx_rx_distance_m = compute_distance_m(tx=bs, rx=ue)
                                                        interference_rx_power +=  t_overlap * thz_channel.get_3gpp_prx_lin(
                                                            tx=bs, rx=ue,
                                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                                            tx_rx_distance_m=tx_rx_distance_m,
                                                            apply_fading=apply_fading,
                                                            bandwidth_hz=bandwidth_hz,
                                                            clutter_density=clutter_density,
                                                            input_shadowing_sample_index=shadowing_sample_index,
                                                            antenna_gain_model=antenna_gain_model,
                                                            use_huawei_measurements=use_huawei_measurements,
                                                            input_average_clutter_height_m=average_machine_height_m,
                                                            los_cond='bs_ue')

                                        if len(ue.ues_interfering_at_ue) > 0:
                                            # for the interfering users (whose that before where useful user),
                                            # I have to check if their ending tick of ACK or DATA is between the
                                            # staring and the ending tick of the actual RX DATA/ACK
                                            # If Yes -> it is an interferer
                                            # If No -> remove from the list of interferers.
                                            copy_of_list = deepcopy(ue.ues_interfering_at_ue)
                                            for user in ue_array:
                                                for i in range(len(copy_of_list)):
                                                    if user.get_ue_id() != ue.get_ue_id() and \
                                                            f'UE_{user.get_ue_id()}' != sorg_ack:
                                                        if f'UE_{user.get_ue_id()}' == copy_of_list[i][0]:
                                                            if ack_rx_at_ue_starting_tick < copy_of_list[i][1]:
                                                                # to compute the portion of data overlapped:

                                                                if ack_rx_at_ue_starting_tick > copy_of_list[i][1]:
                                                                    t_overlap = ((copy_of_list[i][2] -
                                                                                  ack_rx_at_ue_starting_tick) /
                                                                                 (ack_rx_at_ue_ending_tick -
                                                                                  ack_rx_at_ue_starting_tick))
                                                                else:
                                                                    t_overlap = ((copy_of_list[i][2] -
                                                                                  copy_of_list[i][1]) /
                                                                                 (ack_rx_at_ue_ending_tick -
                                                                                  ack_rx_at_ue_starting_tick))
                                                                n_interferers += 1
                                                                tx_rx_distance_m = compute_distance_m(tx=user, rx=ue)
                                                                interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                                    tx=user, rx=ue,
                                                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                                                    tx_rx_distance_m=tx_rx_distance_m,
                                                                    apply_fading=apply_fading,
                                                                    bandwidth_hz=bandwidth_hz,
                                                                    clutter_density=clutter_density,
                                                                    input_shadowing_sample_index=shadowing_sample_index,
                                                                    antenna_gain_model=antenna_gain_model,
                                                                    use_huawei_measurements=use_huawei_measurements,
                                                                    input_average_clutter_height_m=average_machine_height_m,
                                                                    los_cond='ue_ue')
                                                            elif t >= copy_of_list[i][2]:
                                                                ue.ues_interfering_at_ue.remove(
                                                                    (f'UE_{user.get_ue_id()}', copy_of_list[i][1],
                                                                     copy_of_list[i][2]))
                                            for i in range(len(copy_of_list)):
                                                if 'BS' != sorg_ack:
                                                    if 'BS' == copy_of_list[i][0] and \
                                                            ack_rx_at_ue_starting_tick < \
                                                            copy_of_list[i][1]:
                                                        # to compute the portion of data overlapped:

                                                        if ack_rx_at_ue_starting_tick > copy_of_list[i][1]:
                                                            t_overlap = ((copy_of_list[i][2] -
                                                                          ack_rx_at_ue_starting_tick) /
                                                                         (ack_rx_at_ue_ending_tick -
                                                                          ack_rx_at_ue_starting_tick))
                                                        else:
                                                            t_overlap = ((copy_of_list[i][2] -
                                                                          copy_of_list[i][1]) /
                                                                         (ack_rx_at_ue_ending_tick -
                                                                          ack_rx_at_ue_starting_tick))
                                                        n_interferers += 1
                                                        tx_rx_distance_m = compute_distance_m(tx=bs, rx=ue)
                                                        interference_rx_power += t_overlap * thz_channel.get_3gpp_prx_lin(
                                                            tx=bs, rx=ue,
                                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                                            tx_rx_distance_m=tx_rx_distance_m,
                                                            apply_fading=apply_fading,
                                                            bandwidth_hz=bandwidth_hz,
                                                            clutter_density=clutter_density,
                                                            input_shadowing_sample_index=shadowing_sample_index,
                                                            antenna_gain_model=antenna_gain_model,
                                                            use_huawei_measurements=use_huawei_measurements,
                                                            input_average_clutter_height_m=average_machine_height_m,
                                                            los_cond='bs_ue')
                                                    elif 'BS' == copy_of_list[i][0] and \
                                                            t >= copy_of_list[i][2]:
                                                        ue.ues_interfering_at_ue.remove((f'BS', copy_of_list[i][1],
                                                                                         copy_of_list[i][2]))
                                        if interference_rx_power == 0:
                                            sinr_db = snr_db

                                        else:
                                            noise_power_dbw = thz_channel.get_thermal_noise_power_dbw(
                                                input_noise_figure=noise_figure_ue, bandwidth_hz=bandwidth_hz)
                                            noise_power = 10 ** (noise_power_dbw / 10)
                                            noise_plus_interference = noise_power + interference_rx_power
                                            useful_rx_power = 10 ** (useful_rx_power_db / 10)
                                            sinr = useful_rx_power / noise_plus_interference
                                            sinr_db = 10 * log10(sinr)


                                        if sinr_db >= sinr_th_db:
                                            success = True
                                        else:
                                            success = False

                                        ue.n_interfering.append(n_interferers)

                                        if success and ack_rx_dest[index] == ue.get_ue_id():
                                            # An ACK intended for this UE has been successfully received,
                                            # so discard the corresponding data

                                            ue.ack_rx_with_success = True

                                            remain_in_wait_ack = False
                                            if enable_print:
                                                print('UE ', ue.get_ue_id(), ' has received an ACK at t = ', t)
                                                if tx != bs:
                                                    print("ACK transmitter = ", tx.get_ue_id())
                                                else:
                                                    print("ACK transmitter = BS")
                                            # when there was a BURST transmission, the UE can receive an ACK
                                            #  for multiple packet IDs. Need to check based on n_ack_rx_simultaneously.
                                            ack_received = False
                                            indeces_to_be_removed = list()
                                            for packet in range(len(ue.ul_buffer.buffer_packet_list)):
                                                if ue.ul_buffer.buffer_packet_list[packet].get_id() == ack_rx_id[index]:
                                                    for i in ue.packets_to_be_removed:
                                                        if ue.ul_buffer.buffer_packet_list[packet].get_id() in \
                                                                ue.packets_to_be_removed[i]:
                                                            ack_received = True

                                                    if tx == bs:
                                                        if ue.get_broadcast_bool() is True:
                                                            ue.set_temp_obs_broadcast(input_ack_rx_at_ue_tx_index=-1,
                                                                                      input_rx_power=ack_rx_power)

                                                        elif ue.get_last_action() == 0:
                                                            # if the action is unicast towards BS ->
                                                            # take RX_address saved during TX and compare with the ID
                                                            # of the ACK sender

                                                            if (ue.get_unicast_rx_address() == sorg_ack[3:] or
                                                                    ue.get_unicast_rx_address() ==
                                                                    sorg_ack):

                                                                if enable_print:
                                                                    print("UE ", ue.get_ue_id(),
                                                                          " has transmitted in unicast with"
                                                                          " success to ", ue.unicast_rx_address)

                                                                ue.update_neighbor_table_unicast_success(
                                                                            input_rx_power=ack_rx_power)

                                                    else:

                                                        ack_tx_bs_seen = \
                                                        ue_array[int(sorg_ack[3:])].obs[0][-1]
                                                        if ue.get_broadcast_bool() is True:

                                                            ue.set_temp_obs_broadcast(
                                                                input_ack_rx_at_ue_tx_index=ue.neighbour_table.index(
                                                                    sorg_ack[3:]),
                                                                input_rx_power=ack_rx_power,
                                                                input_bs_seen=ack_tx_bs_seen)

                                                            if enable_print:
                                                                print("UE ", ue.get_ue_id(),
                                                                      " has transmitted in broadcast and it has received an ACK from",
                                                                          sorg_ack[3:])

                                                        elif ue.get_last_action() == 0:
                                                            if (ue.get_unicast_rx_address() == sorg_ack[3:] or
                                                                    ue.get_unicast_rx_address() ==
                                                                    sorg_ack):
                                                                if enable_print:
                                                                    print("UE ", ue.get_ue_id(),
                                                                          " has transmitted in unicast with"
                                                                          " success to ", ue.unicast_rx_address)

                                                                ue.update_neighbor_table_unicast_success(
                                                                    input_rx_power=ack_rx_power,
                                                                    input_bs_seen=ack_tx_bs_seen)

                                            # if the ACK for that packet has not been received yet:
                                            if ack_received is False:
                                                ue.reception_ack_during_wait = True

                                                # Before removing the packet, check if the ID of the packet successfully
                                                # transmitted is different with respect to the ID contained in the
                                                # previously received ACK
                                                # If the UE is a relay, it has to remove all the packets in the queue

                                                if tx == bs:
                                                    # Compute the latency for the star topology
                                                    if star_topology is True:
                                                        latency = (t - ue.packet_generation_instant) * \
                                                                  simulator_tick_duration_s
                                                        ue.latency_ue.append(latency)

                                                    for packet in range(len(ue.ul_buffer.buffer_packet_list)):
                                                        if ue.ul_buffer.buffer_packet_list[packet].packet_id == ack_rx_id[index]:
                                                            ue.ul_buffer.buffer_packet_list[packet].set_ack_rx(
                                                                ack_rx=True)

                                                    packet_to_be_removed = deepcopy(ue.ul_buffer.buffer_packet_list)

                                                    for packet in packet_to_be_removed:
                                                        if packet.packet_id == ack_rx_id[index]:
                                                            if packet.get_id() not in ue.list_data_generated_during_wait_ack \
                                                                    and packet.get_id() in ue.list_ack_sent_from_bs:

                                                                if ue.get_broadcast_bool() is False:
                                                                    ue.packet_id_success = packet.get_id()
                                                                    ue.remove_packet(packet_id=packet.get_id(),
                                                                                     input_enable_print=enable_print)
                                                                    ue.packets_sent -= 1
                                                                else:
                                                                    ue.packets_to_be_removed["BS"].append(
                                                                        packet.get_id())

                                                                ue.list_ack_sent_from_bs.remove(packet.get_id())

                                                    ue.set_relay_bool(relay_bool=False)
                                                else:

                                                    for packet in range(len(ue.ul_buffer.buffer_packet_list)):

                                                        if ue.ul_buffer.buffer_packet_list[
                                                            packet].get_data_unicast() is False:
                                                            if ue.ul_buffer.buffer_packet_list[packet].packet_id == ack_rx_id[index]:
                                                                ue.ul_buffer.buffer_packet_list[packet].set_ack_rx(
                                                                    ack_rx=True)

                                                    packet_to_be_removed = deepcopy(ue.ul_buffer.buffer_packet_list)

                                                    for packet in packet_to_be_removed:
                                                        if packet.packet_id == ack_rx_id[index]:
                                                            if packet.get_data_unicast() is False and packet.get_id() not \
                                                                    in ue.list_data_generated_during_wait_ack and \
                                                                    packet.get_id() in tx.dict_ack_sent_from_ue[
                                                                ue.get_ue_id()]:

                                                                if ue.get_broadcast_bool() is False:
                                                                    ue.packet_id_success = packet.get_id()
                                                                    ue.remove_packet(packet_id=packet.get_id(),
                                                                                     input_enable_print=enable_print)
                                                                    ue.packets_sent -= 1

                                                                else:
                                                                    ue.packets_to_be_removed[str(tx.get_ue_id())].append(
                                                                        packet.get_id())
                                                                # # Reset the bool variable for being a relay to false once the ACK
                                                                # has been sent
                                                                tx.dict_ack_sent_from_ue[ue.get_ue_id()].remove(
                                                                    packet.get_id())


                                                if len(ue.ul_buffer.buffer_packet_list) > 0 and len(
                                                        ue.list_data_rx_during_wait_ack) == 0:
                                                    remain_in_wait_ack = True

                                            else:
                                                if ue.get_state_duration() != ue.get_state_final_tick():
                                                    # Reception is successful, but the UE has already received that ACK
                                                    remain_in_wait_ack = True
                                                else:
                                                    remain_in_wait_ack = False


                                        else:
                                            if ue.get_state_duration() != ue.get_state_final_tick():
                                                # Reception is successful, but the UE has already received that ACK
                                                remain_in_wait_ack = True
                                            else:
                                                remain_in_wait_ack = False
                                                # The UE has not received an ACK because of PHY or MAC problems
                                                ue.set_retransmission_packets(retransmission_bool=True)

                                    if ue.ack_rx_with_success is True and ue.get_last_action() == 0:
                                        ue.unicast_handling_no_reward_no_neighbor_update()
                                        remain_in_wait_ack = False

                                    for index in range(len(ack_rx_sources)):
                                        if len(simulator_timing_structure[f'UE_{ue.get_ue_id()}']['ACK_RX'][ack_rx_sources[index]]) > 1:

                                            remove_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_rx_key=f'UE_{ue.get_ue_id()}',
                                                input_type_key='ACK_RX',
                                                input_tx_key=ack_rx_sources[index])

                                    ack_rx_id.clear()

                                if ue.data_rx_during_wait_ack is True and star_topology is False:
                                    ue.data_rx_during_wait_ack = False
                                    (data_rx_at_ue_starting_tick, data_rx_at_ue_ending_tick, data_rx_at_ue_size_bytes,
                                     packet_id_rx_from_ue, data_rx_at_ue_ue_id) = (
                                        find_data_rx_times_tick(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            current_tick=t,
                                            input_ue_id=ue.get_ue_id()))

                                    for index in range(len(data_rx_at_ue_ue_id)):
                                        # Compute the tx-rx distance
                                        tx_rx_distance_m = compute_distance_m(tx=ue_array[data_rx_at_ue_ue_id[index]], rx=ue)

                                        # Check if the shadowing sample should be changed
                                        if t >= shadowing_next_tick:
                                            shadowing_sample_index = shadowing_sample_index + 1
                                            shadowing_next_tick = t + shadowing_coherence_time_tick_duration

                                        data_rx_power = thz_channel.get_3gpp_prx_db(
                                            tx=ue_array[data_rx_at_ue_ue_id[index]], rx=ue,
                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                            tx_rx_distance_m=tx_rx_distance_m,
                                            apply_fading=apply_fading,
                                            bandwidth_hz=bandwidth_hz,
                                            clutter_density=clutter_density,
                                            input_shadowing_sample_index=shadowing_sample_index,
                                            antenna_gain_model=antenna_gain_model,
                                            use_huawei_measurements=use_huawei_measurements,
                                            input_average_clutter_height_m=average_machine_height_m,
                                            los_cond='ue_ue')

                                        # Compute the SNR between the current receiving UE and the transmitting UE
                                        snr_db = thz_channel.get_3gpp_snr_db(
                                            tx=ue_array[data_rx_at_ue_ue_id[index]], rx=ue,
                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                            tx_rx_distance_m=tx_rx_distance_m,
                                            apply_fading=apply_fading,
                                            bandwidth_hz=bandwidth_hz,
                                            clutter_density=clutter_density,
                                            input_shadowing_sample_index=shadowing_sample_index,
                                            antenna_gain_model=antenna_gain_model,
                                            use_huawei_measurements=use_huawei_measurements,
                                            input_average_clutter_height_m=average_machine_height_m,
                                            los_cond='ue_ue')

                                        sir_dB = None
                                        n_interferers = 0

                                        # this method takes in input both the current UE_ID that has received a data and both the
                                        # ID of the UE that has sent the data
                                        # -> need to check if there is another UE != from these two UEs that has TX a DATA or an ACK
                                        ue.ues_colliding_at_ue.clear()

                                        ue.ues_colliding_at_ue = check_collision(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_ue_id=ue.get_ue_id(),
                                            input_t_start_rx=data_rx_at_ue_starting_tick,
                                            input_t_end_rx=data_rx_at_ue_ending_tick, input_tx=None,
                                            input_ue_id_rx=data_rx_at_ue_ue_id[index],
                                            ues_colliding=ue.ues_colliding_at_ue)

                                        useful_rx_power_db = data_rx_power
                                        add_interferer = True
                                        if len(ue.ues_interfering_at_ue) > 0:
                                            for i in range(len(ue.ues_interfering_at_ue)):
                                                if f'UE_{data_rx_at_ue_ue_id[index]}' == \
                                                        ue.ues_interfering_at_ue[i][0] and \
                                                        t == ue.ues_interfering_at_ue[i][1]:
                                                    add_interferer = False
                                                    # the useful user will become an interferer for the next reception,
                                                    # so save the ID and the current ending tick of this reception
                                        if add_interferer is True:
                                            ue.ues_interfering_at_ue.append((f'UE_{data_rx_at_ue_ue_id[index]}',
                                                                             data_rx_at_ue_starting_tick,
                                                                             data_rx_at_ue_ending_tick))
                                        interference_rx_power = 0
                                        if len(ue.ues_colliding_at_ue) > 0:
                                            for user in ue_array:
                                                for i in range(len(ue.ues_colliding_at_ue)):
                                                    if (user.get_ue_id() != ue.get_ue_id() and user.get_ue_id() !=
                                                            data_rx_at_ue_ue_id[index] and
                                                            f'UE_{user.get_ue_id()}' == ue.ues_colliding_at_ue[i][0]):
                                                        # to compute the portion of data overlapped:
                                                        # t_j = (t_end_current - t_start_interferer) /
                                                        # (t_end_current - t_start_current)
                                                        if ue.ues_colliding_at_ue[i][1] < data_rx_at_ue_ending_tick < \
                                                                ue.ues_colliding_at_ue[i][2]:
                                                            t_overlap = ((data_rx_at_ue_ending_tick -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (data_rx_at_ue_ending_tick -
                                                                          data_rx_at_ue_starting_tick))
                                                        else:
                                                            t_overlap = ((ue.ues_colliding_at_ue[i][2] -
                                                                          ue.ues_colliding_at_ue[i][1]) /
                                                                         (data_rx_at_ue_ending_tick -
                                                                          data_rx_at_ue_starting_tick))
                                                        n_interferers += 1
                                                        tx_rx_distance_m = compute_distance_m(tx=user, rx=ue)
                                                        interference_rx_power +=  t_overlap * thz_channel.get_3gpp_prx_lin(
                                                            tx=user, rx=ue,
                                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                                            tx_rx_distance_m=tx_rx_distance_m,
                                                            apply_fading=apply_fading,
                                                            bandwidth_hz=bandwidth_hz,
                                                            clutter_density=clutter_density,
                                                            input_shadowing_sample_index=shadowing_sample_index,
                                                            antenna_gain_model=antenna_gain_model,
                                                            use_huawei_measurements=use_huawei_measurements,
                                                            input_average_clutter_height_m=average_machine_height_m,
                                                            los_cond='ue_ue')

                                            for i in range(len(ue.ues_colliding_at_ue)):
                                                if 'BS' == ue.ues_colliding_at_ue[i][0] and 'BS' != data_rx_at_ue_ue_id[index]:
                                                    # to compute the portion of data overlapped:

                                                    if ue.ues_colliding_at_ue[i][1] < data_rx_at_ue_ending_tick < \
                                                            ue.ues_colliding_at_ue[i][2]:
                                                        t_overlap = ((data_rx_at_ue_ending_tick -
                                                                      ue.ues_colliding_at_ue[i][1]) /
                                                                     (data_rx_at_ue_ending_tick -
                                                                      data_rx_at_ue_starting_tick))
                                                    else:
                                                        t_overlap = ((ue.ues_colliding_at_ue[i][2] -
                                                                      ue.ues_colliding_at_ue[i][1]) /
                                                                     (data_rx_at_ue_ending_tick -
                                                                      data_rx_at_ue_starting_tick))
                                                    n_interferers += 1
                                                    tx_rx_distance_m = compute_distance_m(tx=bs, rx=ue)
                                                    interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                        tx=bs, rx=ue,
                                                        carrier_frequency_ghz=carrier_frequency_ghz,
                                                        tx_rx_distance_m=tx_rx_distance_m,
                                                        apply_fading=apply_fading,
                                                        bandwidth_hz=bandwidth_hz,
                                                        clutter_density=clutter_density,
                                                        input_shadowing_sample_index=shadowing_sample_index,
                                                        antenna_gain_model=antenna_gain_model,
                                                        use_huawei_measurements=use_huawei_measurements,
                                                        input_average_clutter_height_m=average_machine_height_m,
                                                        los_cond='bs_ue')

                                        if len(ue.ues_interfering_at_ue) > 0:
                                            # for the interfering users (whose that before where useful user),
                                            # I have to check if their ending tick of ACK or DATA is between the
                                            # staring and the ending tick of the actual RX DATA/ACK
                                            # If Yes -> it is an interferer
                                            # If No -> remove from the list of interferers.
                                            copy_of_list = deepcopy(ue.ues_interfering_at_ue)
                                            for user in ue_array:
                                                for i in range(len(copy_of_list)):
                                                    if user.get_ue_id() != ue.get_ue_id() and \
                                                            user.get_ue_id() != data_rx_at_ue_ue_id[index]:
                                                        if f'UE_{user.get_ue_id()}' == copy_of_list[i][0]:

                                                            if data_rx_at_ue_starting_tick < copy_of_list[i][2]:
                                                                # to compute the portion of data overlapped:

                                                                if data_rx_at_ue_starting_tick > copy_of_list[i][1]:
                                                                    t_overlap = ((copy_of_list[i][2] -
                                                                                  data_rx_at_ue_starting_tick) /
                                                                                 (data_rx_at_ue_ending_tick -
                                                                                  data_rx_at_ue_starting_tick))
                                                                else:
                                                                    t_overlap = ((copy_of_list[i][2] -
                                                                                  copy_of_list[i][1]) /
                                                                                 (data_rx_at_ue_ending_tick -
                                                                                  data_rx_at_ue_starting_tick))
                                                                n_interferers += 1
                                                                tx_rx_distance_m = compute_distance_m(tx=user, rx=ue)

                                                                interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                                    tx=user, rx=ue,
                                                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                                                    tx_rx_distance_m=tx_rx_distance_m,
                                                                    apply_fading=apply_fading,
                                                                    bandwidth_hz=bandwidth_hz,
                                                                    clutter_density=clutter_density,
                                                                    input_shadowing_sample_index=shadowing_sample_index,
                                                                    antenna_gain_model=antenna_gain_model,
                                                                    use_huawei_measurements=use_huawei_measurements,
                                                                    input_average_clutter_height_m=average_machine_height_m,
                                                                    los_cond='ue_ue')
                                                            elif t >= copy_of_list[i][2]:
                                                                ue.ues_interfering_at_ue.remove(
                                                                    (copy_of_list[i][0], copy_of_list[i][1], copy_of_list[i][2]))

                                            for i in range(len(copy_of_list)):
                                                if 'BS' != data_rx_at_ue_ue_id[index]:
                                                    if 'BS' == copy_of_list[i][0]:

                                                        if data_rx_at_ue_starting_tick < copy_of_list[i][2]:
                                                            # to compute the portion of data overlapped:

                                                            if data_rx_at_ue_starting_tick > copy_of_list[i][1]:
                                                                t_overlap = ((copy_of_list[i][2] -
                                                                              data_rx_at_ue_starting_tick) /
                                                                             (data_rx_at_ue_ending_tick -
                                                                              data_rx_at_ue_starting_tick))
                                                            else:
                                                                t_overlap = ((copy_of_list[i][2] -
                                                                              copy_of_list[i][1]) /
                                                                             (data_rx_at_ue_ending_tick -
                                                                              data_rx_at_ue_starting_tick))
                                                            n_interferers += 1
                                                            tx_rx_distance_m = compute_distance_m(tx=bs, rx=ue)

                                                            interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                                tx=bs, rx=ue,
                                                                carrier_frequency_ghz=carrier_frequency_ghz,
                                                                tx_rx_distance_m=tx_rx_distance_m,
                                                                apply_fading=apply_fading,
                                                                bandwidth_hz=bandwidth_hz,
                                                                clutter_density=clutter_density,
                                                                input_shadowing_sample_index=shadowing_sample_index,
                                                                antenna_gain_model=antenna_gain_model,
                                                                use_huawei_measurements=use_huawei_measurements,
                                                                input_average_clutter_height_m=average_machine_height_m,
                                                                los_cond='bs_ue')
                                                        elif t >= copy_of_list[i][2]:
                                                            ue.ues_interfering_at_ue.remove(
                                                                (copy_of_list[i][0], copy_of_list[i][1], copy_of_list[i][2]))

                                        if interference_rx_power == 0:
                                            sinr_db = snr_db

                                        else:
                                            noise_power_dbw = thz_channel.get_thermal_noise_power_dbw(
                                                input_noise_figure=noise_figure_ue, bandwidth_hz=bandwidth_hz)
                                            noise_power = 10 ** (noise_power_dbw / 10)
                                            noise_plus_interference = noise_power + interference_rx_power
                                            useful_rx_power = 10 ** (useful_rx_power_db / 10)
                                            sinr = useful_rx_power / noise_plus_interference
                                            sinr_db = 10 * log10(sinr)

                                        if sinr_db >= sinr_th_db:
                                            success = True
                                        else:
                                            success = False

                                        ue.n_interfering.append(n_interferers)

                                        # Before adding a new packet -> check if n_packets_buffer < max_n_packets_to_be_forwarded + 1

                                        if success and ((len(ue.ul_buffer.buffer_packet_list) < \
                                             max_n_packets_to_be_forwarded + 1 and ue.check_generated_packet_present() is True) \
                                            or (
                                            (len(ue.ul_buffer.buffer_packet_list) < \
                                             max_n_packets_to_be_forwarded and ue.check_generated_packet_present() is False))):

                                            ue.packet_forwarding.append(packet_id_rx_from_ue[index])

                                            counter = 0
                                            for packet in ue.ul_buffer.buffer_packet_list:
                                                if packet.get_generated_by_ue() == ue.ue_id:
                                                    counter += 1
                                            if counter > 0:
                                                total_buffer_size = max_n_packets_to_be_forwarded + 1
                                            else:
                                                total_buffer_size = max_n_packets_to_be_forwarded

                                            if data_rx_at_ue_size_bytes[index] > 0 and len(ue.ul_buffer.buffer_packet_list) < \
                                                    total_buffer_size:

                                                for user in ue_array:
                                                    if data_rx_at_ue_ue_id[index] == user.get_ue_id():
                                                        if len(user.buffer_packet_sent) > 0:
                                                            for packet in user.buffer_packet_sent:
                                                                if packet.packet_id == packet_id_rx_from_ue[index]:
                                                                    if packet.address == str(
                                                                            ue.get_ue_id()) or packet.address == "-1":
                                                                        ue.designated_rx = True
                                                                        break

                                                            if ue.designated_rx:

                                                                # reset the action variables
                                                                if enable_print:
                                                                    print("UE ", ue.get_ue_id(),
                                                                          " has received a packet from UE ",
                                                                          data_rx_at_ue_ue_id[index])

                                                                ue.designated_rx = False

                                                                for n_packet in range(len(user.buffer_packet_sent)):
                                                                    if packet_id_rx_from_ue[index] == user.buffer_packet_sent[n_packet].packet_id:
                                                                        if len(ue.ul_buffer.buffer_packet_list) < total_buffer_size:
                                                                            # Check if the packet is already in the queue
                                                                            packet_already_in_queue = False
                                                                            for packet in ue.ul_buffer.buffer_packet_list:
                                                                                if (packet.get_generated_by_ue() ==
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_generated_by_ue() and
                                                                                        packet.get_packet_id_generator() ==
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_packet_id_generator()):
                                                                                    packet_already_in_queue = True

                                                                            # Check if the packet is generated by the UE itself
                                                                            packet_generated_by_ue_itself = False
                                                                            if user.buffer_packet_sent[
                                                                                n_packet].get_generated_by_ue() == ue.get_ue_id():
                                                                                packet_generated_by_ue_itself = True

                                                                            # If the packet is generated by the UE itself, force a broadcast action
                                                                            if packet_generated_by_ue_itself and ue.obs[0][-1] == 0: #and (np.sum(ue.obs[-1]) == 0):
                                                                                ue.next_action = 3
                                                                            # Check if the packet exceeded the hop limit
                                                                            packet_out_of_hop_limit = False
                                                                            if user.buffer_packet_sent[
                                                                                n_packet].get_hop_count() >= hop_limit:
                                                                                packet_out_of_hop_limit = True

                                                                            if (packet_already_in_queue is False and packet_generated_by_ue_itself is False
                                                                                    and packet_out_of_hop_limit is False):
                                                                                ue.forward_in_wait_ack = True
                                                                                if enable_print:
                                                                                    print("UE ", ue.get_ue_id(),
                                                                                          " is forwarding packet: ",
                                                                                          user.buffer_packet_sent[
                                                                                              n_packet].get_id(),
                                                                                          " received from UE: ",
                                                                                          data_rx_at_ue_ue_id[index],
                                                                                          "with hop_count: ",
                                                                                          user.buffer_packet_sent[
                                                                                              n_packet].hop_count)
                                                                                ue.n_forwarding += 1
                                                                                # Successful data reception, add the data in the queue and transmit the ACK

                                                                                ue.add_new_packet(current_tick=t,
                                                                                                  input_enable_print=enable_print,
                                                                                                  input_data_to_be_forwarded_bool=True,
                                                                                                  input_packet_size_bytes=user.buffer_packet_sent[
                                                                                                      n_packet].packet_size,
                                                                                                  input_simulation_tick_duration=simulator_tick_duration_s,
                                                                                                  data_rx_from_ue=data_rx_at_ue_ue_id[index],
                                                                                                  packet_id_rx_from_ue=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_id(),
                                                                                                  packet_generated_by_ue=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_generated_by_ue(),
                                                                                                  packet_id_generator=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_packet_id_generator(),
                                                                                                  packet_hop_count=
                                                                                                  user.buffer_packet_sent[
                                                                                                      n_packet].get_hop_count(),
                                                                                                  packet_address=(
                                                                                                      ue.get_unicast_rx_address() if ue.get_broadcast_bool() is False else "-1"), generation_time=user.buffer_packet_sent[
                                                                                                      n_packet].get_generated_by_ue_time_instant_tick())

                                                                                ue.list_data_generated_during_wait_ack.append(
                                                                                    ue.ul_buffer.get_last_packet().get_id())
                                                                                ue.list_data_rx_during_wait_ack.append(
                                                                                    user.buffer_packet_sent[n_packet].get_id())
                                                                                ue.dict_data_rx_during_wait_ack[
                                                                                    data_rx_at_ue_ue_id[index]].append(
                                                                                    user.buffer_packet_sent[n_packet].get_id())
                                                                                ue.list_data_rx_from_ue_id.append(
                                                                                    data_rx_at_ue_ue_id[index])

                                                                                if user.buffer_packet_sent[
                                                                                    n_packet].get_id() not in \
                                                                                        ue.dict_ack_sent_from_ue[
                                                                                            data_rx_at_ue_ue_id[index]]:
                                                                                    ue.dict_ack_sent_from_ue[
                                                                                        data_rx_at_ue_ue_id[index]].append(
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_id())

                                                                            if (packet_already_in_queue is True and
                                                                                    packet_generated_by_ue_itself is False):

                                                                                ue.dict_data_rx_during_wait_ack[
                                                                                    data_rx_at_ue_ue_id[index]].append(
                                                                                    user.buffer_packet_sent[
                                                                                        n_packet].get_id())
                                                                                ue.list_data_rx_from_ue_id.append(
                                                                                    data_rx_at_ue_ue_id[index])
                                                                                if user.buffer_packet_sent[
                                                                                    n_packet].get_id() not in \
                                                                                        ue.dict_ack_sent_from_ue[
                                                                                            data_rx_at_ue_ue_id[index]]:
                                                                                    ue.dict_ack_sent_from_ue[
                                                                                        data_rx_at_ue_ue_id[index]].append(
                                                                                        user.buffer_packet_sent[
                                                                                            n_packet].get_id())


                                                                if ue.forward_in_wait_ack:
                                                                    ue.forward_in_wait_ack = False
                                                                    if np.sum(ue.obs[1]) > 0:
                                                                        if ue.get_ue_id() < data_rx_at_ue_ue_id[index]:
                                                                            ue.set_obs_update(
                                                                                input_data_rx_at_ue_tx_index=data_rx_at_ue_ue_id[index] - 1,
                                                                                input_rx_power=data_rx_power)

                                                                        else:
                                                                            ue.set_obs_update(
                                                                                input_data_rx_at_ue_tx_index=data_rx_at_ue_ue_id[index],
                                                                                input_rx_power=data_rx_power)
                                                                else:
                                                                    remain_in_wait_ack = True
                                                            else:
                                                                remain_in_wait_ack = True

                                            else:
                                                remain_in_wait_ack = True


                                        else:

                                            remain_in_wait_ack = True
                                        if len(simulator_timing_structure[f'UE_{ue.get_ue_id()}']['DATA_RX'][
                                                   f'UE_{data_rx_at_ue_ue_id[index]}']) > 1:
                                            # Update the timing structure to reset this reception
                                            remove_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_rx_key=f'UE_{ue.get_ue_id()}',
                                                input_type_key='DATA_RX',
                                                input_tx_key=f'UE_{data_rx_at_ue_ue_id[index]}')
                                        if ue.ack_rx_during_wait_ack is False:
                                            if ue.get_state_duration() != ue.get_state_final_tick():
                                                remain_in_wait_ack = True
                                            else:
                                                remain_in_wait_ack = False
                                                ue.set_retransmission_packets(retransmission_bool=True)

                            if len(ue.ul_buffer.buffer_packet_list) > 0:
                                ue.set_retransmission_packets(retransmission_bool=True)
                            else:
                                ue.set_retransmission_packets(retransmission_bool=False)

                            if ue.get_broadcast_bool() is True:
                                ue.ack_rx_during_wait_ack = False
                                ue.data_rx_during_wait_ack = False
                                if t != ue.get_state_final_tick():
                                    remain_in_wait_ack = True
                                else:
                                    remain_in_wait_ack = False

                            # Update the future state
                            if remain_in_wait_ack:
                                # Remain in WAIT_ACK until the end of the slot
                                ue.set_state_duration(input_ticks=ue.get_state_final_tick())
                            elif (len(ue.list_data_generated_during_wait_ack) > 0 or
                                  len(ue.list_data_rx_from_ue_id) > 0) and remain_in_wait_ack is False and \
                                    star_topology is False:

                                if ue.get_broadcast_bool():
                                    ue.check_remove_packet(input_enable_print=enable_print)
                                    if len(ue.ul_buffer.buffer_packet_list) > 0:
                                        ue.set_retransmission_packets(retransmission_bool=True)
                                    else:
                                        ue.set_retransmission_packets(retransmission_bool=False)

                                for other_ue in ue_array:
                                    if other_ue != ue:
                                        ue.dict_ack_sent_from_ue[other_ue.get_ue_id()].clear()
                                # Reduce the energy consumed if the UE has stopped the WAIT_ACK before its end
                                # Compute energy spent
                                ue.energy_consumed -= power_ack * (
                                        ue.get_state_final_tick() - t) * simulator_tick_duration_s

                                ue.list_ack_sent_from_bs.clear()
                                ue.ues_colliding_at_ue.clear()

                                if len(ue.ues_interfering_at_ue) > 0:
                                    ue.ues_interfering_at_ue.clear()
                                copy_of_dictonary = deepcopy(ue.dict_data_rx_during_wait_ack)
                                ue.previous_state = 'WAIT_ACK'

                                index = 0
                                for packet_id in ue.list_data_generated_during_wait_ack:
                                    ue.update_num_tx(input_packet_id=packet_id)
                                    ue.check_num_tx()

                                copy_data_rx_ue_id = list()
                                for user_id in ue.list_data_rx_from_ue_id:
                                    if user_id not in copy_data_rx_ue_id:
                                        copy_data_rx_ue_id.append(user_id)
                                # if a DATA to be forwarded has been received, the UE has to go in ACK TX for that UE
                                go_in_tx_ack(input_ue=ue, input_ack_duration_tick=len(copy_data_rx_ue_id) * t_ack_tick,
                                             current_tick=t, input_enable_print=enable_print)

                                j = - 1

                                for ue_id in copy_data_rx_ue_id:

                                    j += 1

                                    for index in range(len(ue.dict_data_rx_during_wait_ack[ue_id])):
                                        for other_ue in ue_array:
                                            if other_ue != ue:
                                                insert_item_in_timing_structure(
                                                    input_simulator_timing_structure=simulator_timing_structure,
                                                    input_starting_tick=t + ue.get_prop_delay_to_ue_tick(
                                                        input_ue_id=other_ue.get_ue_id()) + j * t_ack_tick,
                                                    input_final_tick=
                                                    t + ue.get_prop_delay_to_ue_tick(
                                                        input_ue_id=other_ue.get_ue_id()) + t_ack_tick + j * t_ack_tick,
                                                    input_third_field=ue_id,
                                                    input_fourth_field=
                                                    ue.dict_data_rx_during_wait_ack[ue_id][index],
                                                    input_rx_key=f'UE_{other_ue.get_ue_id()}',
                                                    input_type_key='ACK_RX',
                                                    input_tx_key=f'UE_{ue.get_ue_id()}',
                                                )
                                        for i in range(len(copy_of_dictonary[ue_id])):
                                            if copy_of_dictonary[ue_id][i] not in ue.dict_ack_sent_from_ue[ue_id]:
                                                ue.dict_ack_sent_from_ue[ue_id].append(copy_of_dictonary[ue_id][i])

                                        # Update the timing structure for the BS
                                        insert_item_in_timing_structure(
                                            input_simulator_timing_structure=simulator_timing_structure,
                                            input_starting_tick=t + ue.get_prop_delay_to_bs_tick() + j * t_ack_tick,
                                            input_final_tick=
                                            t + ue.get_prop_delay_to_bs_tick() + t_ack_tick + j * t_ack_tick,
                                            input_third_field=ue_id,
                                            input_fourth_field=
                                            ue.dict_data_rx_during_wait_ack[ue_id][index],
                                            input_rx_key='BS',
                                            input_type_key='ACK_RX',
                                            input_tx_key=f'UE_{ue.get_ue_id()}',
                                        )

                                go_in_bo_bool = False
                                go_in_idle_bool = False

                                # if the buffer of the UE has a different length with respect to the one of the new
                                # generated data during wait_ack:
                                # 1. the old packets have to be retransmitted -> update the num_tx() and check if the max
                                # number of retransmission has not been reached
                                # if len(ue.list_data_generated_during_wait_ack) != len(ue.ul_buffer.buffer_packet_list):
                                for packet in ue.ul_buffer.buffer_packet_list:
                                    if packet.get_id() not in ue.list_data_generated_during_wait_ack:
                                        packet.set_retransmission_packets(retransmission_bool=True)
                                        ue.update_num_tx(input_packet_id=packet.get_id(),
                                                         input_enable_print=enable_print)
                                    else:
                                        packet.set_retransmission_packets(retransmission_bool=False)

                                if ue.get_last_action() == 0:

                                    if enable_print:
                                        print("UE ", ue.get_ue_id(), " has transmitted in unicast with "
                                                                     "no success towards ",
                                              ue.unicast_rx_address)
                                    ue.unicast_handling_failure_no_reward(input_ttl=TTL)
                                    ue.new_action_bool = True

                                if ue.get_broadcast_bool() is True:

                                    if np.sum(ue.temp_obs[1]) == 0:

                                        if enable_print:
                                            print("UE ", ue.get_ue_id(),
                                                  " has transmitted in broadcast with no success")
                                        ue.broadcast_handling_failure_no_reward(input_ttl=TTL)

                                        ue.new_action_bool = True

                                        ue.reset_temp_obs()

                                    else:
                                        if enable_print:
                                            print("UE ", ue.get_ue_id(),
                                                  " has transmitted in broadcast with success")

                                        ue.broadcast_handling_no_reward(input_ttl=TTL)
                                        ue.set_last_action(input_last_action=None)
                                        ue.set_broadcast_bool(input_broadcast_bool=False)

                                        ue.new_action_bool = True
                                        ue.reset_temp_obs()

                                if ue.check_num_tx() is False:  # Reached the maximum number of retransmissions ->
                                    # discard that packet and generate a new one if full queue
                                    if ue.check_generated_packet_present() is False:
                                        if ue.is_there_a_new_data(input_current_tick=t,
                                                                  max_n_packets_to_be_forwarded=max_n_packets_to_be_forwarded) \
                                                is True:
                                            ue.update_num_tx(
                                                input_packet_id=ue.ul_buffer.get_last_packet().get_id(),
                                                input_enable_print=enable_print)
                                            ue.check_last_round = False

                                            ue.new_action_bool = True
                                            if ue.get_retransmission_packets() is False:
                                                ue.check_num_tx()

                                # if the UE has received a new ACK for itself -> even if it has to send an ACK and
                                # become a RELAY, it will generate a new packet for itself
                                elif ue.reception_ack_during_wait is True:
                                    ue.reception_ack_during_wait = False
                                    if ue.check_generated_packet_present() is False:
                                        if ue.is_there_a_new_data(input_current_tick=t,
                                                                  max_n_packets_to_be_forwarded=max_n_packets_to_be_forwarded) is \
                                                True:
                                            ue.update_num_tx(
                                                input_packet_id=ue.ul_buffer.get_last_packet().get_id(),
                                                input_enable_print=enable_print)
                                            ue.check_last_round = False

                                            ue.check_num_tx()

                            elif len(ue.list_data_generated_during_wait_ack) == 0:
                                ue.ues_colliding_at_ue.clear()
                                ue.ues_interfering_at_ue.clear()
                                if ue.get_broadcast_bool():
                                    ue.check_remove_packet(input_enable_print=enable_print)
                                    if len(ue.ul_buffer.buffer_packet_list) > 0:
                                        ue.set_retransmission_packets(retransmission_bool=True)
                                    else:
                                        ue.set_retransmission_packets(retransmission_bool=False)

                                ue.energy_consumed -= power_ack * (
                                        ue.get_state_final_tick() - t) * simulator_tick_duration_s

                                if ue.get_retransmission_packets() is False:
                                    # the UE is not retransmitting anything -> The UE will generate a new packet if Full-queue
                                    if ue.is_there_a_new_data(input_current_tick=t,
                                                              max_n_packets_to_be_forwarded=max_n_packets_to_be_forwarded) \
                                            is True:
                                        ue.update_num_tx(input_enable_print=enable_print)
                                        ue.check_last_round = False

                                        ue.check_num_tx()
                                        ue.new_action_bool = True

                                    if ue.get_broadcast_bool() is True:
                                        if np.sum(ue.temp_obs[1]) == 0:

                                            if enable_print:
                                                print("UE ", ue.get_ue_id(),
                                                      " has transmitted in broadcast with no success")
                                            ue.broadcast_handling_failure_no_reward(input_ttl=TTL)

                                            ue.new_action_bool = True

                                            ue.reset_temp_obs()

                                        else:
                                            if enable_print:
                                                print("UE ", ue.get_ue_id(),
                                                      " has transmitted in broadcast with success")

                                            ue.broadcast_handling_no_reward(input_ttl=TTL)
                                            ue.set_last_action(input_last_action=None)
                                            ue.set_broadcast_bool(input_broadcast_bool=False)

                                            ue.new_action_bool = True
                                            ue.reset_temp_obs()

                                    # The queue is full so go in BO
                                    go_in_bo_bool = True
                                elif ue.get_retransmission_packets() is True:
                                    # The UE is forwarding a packet
                                    go_in_bo_bool = True
                                    ue.update_num_tx(input_enable_print=enable_print)

                                    for packet in ue.ul_buffer.buffer_packet_list:
                                        packet.set_retransmission_packets(retransmission_bool=True)

                                    # Computation of the reward in case no ACK received and the maximum number of
                                    # retransmissions is reached
                                    if ue.get_last_action() == 0:
                                        if enable_print:
                                            print("UE ", ue.get_ue_id(), " has transmitted in unicast with "
                                                                         "no success towards ",
                                                  ue.unicast_rx_address)

                                        ue.unicast_handling_failure_no_reward(input_ttl=TTL)

                                        ue.new_action_bool = True

                                    if ue.get_broadcast_bool() is True:
                                        if np.sum(ue.temp_obs[1]) == 0:

                                            if enable_print:
                                                print("UE ", ue.get_ue_id(),
                                                      " has transmitted in broadcast with no success")
                                            ue.broadcast_handling_failure_no_reward(input_ttl=TTL)

                                            ue.new_action_bool = True
                                            ue.reset_temp_obs()

                                        else:
                                            if enable_print:
                                                print("UE ", ue.get_ue_id(),
                                                      " has transmitted in broadcast with success")

                                            ue.set_last_action(input_last_action=None)
                                            ue.set_broadcast_bool(input_broadcast_bool=False)
                                            ue.broadcast_handling_no_reward(input_ttl=TTL)

                                            ue.new_action_bool = True
                                            ue.reset_temp_obs()

                                    if ue.check_num_tx() is False:  # Reached the maximum number of retransmissions ->
                                        # discard that packet and generate a new one if full queue
                                        if ue.check_generated_packet_present() is False:
                                            if ue.is_there_a_new_data(input_current_tick=t,
                                                                      max_n_packets_to_be_forwarded=max_n_packets_to_be_forwarded) \
                                                    is True:
                                                ue.update_num_tx(
                                                    input_packet_id=ue.ul_buffer.get_last_packet().get_id(),
                                                    input_enable_print=enable_print)
                                                ue.check_last_round = False

                                                if ue.get_retransmission_packets() is False:
                                                    ue.check_num_tx()

                                                ue.new_action_bool = True

                                            else:
                                                # The queue is empy so go in IDLE until the next data generation
                                                go_in_idle_bool = True
                                    else:
                                        if ue.reception_ack_during_wait is True:
                                            ue.reception_ack_during_wait = False
                                            new_packet_generation = True
                                            for packet in ue.ul_buffer.buffer_packet_list:
                                                if packet.get_data_to_be_forwarded_bool() is False:
                                                    new_packet_generation = False  # there is already a packet generated from
                                                    # the UE itself in the buffer, so no need to generate a new one
                                            if new_packet_generation is True:
                                                new_packet_generation = False
                                                if ue.is_there_a_new_data(input_current_tick=t,
                                                                          max_n_packets_to_be_forwarded=
                                                                          max_n_packets_to_be_forwarded) is True:
                                                    ue.update_num_tx(
                                                        input_packet_id=ue.ul_buffer.get_last_packet().get_id(),
                                                        input_enable_print=enable_print)
                                                    ue.check_last_round = False

                                                    ue.check_num_tx()

                                else:
                                    # The queue is empy so go in IDLE until the next data generation
                                    go_in_idle_bool = True

                            if go_in_bo_bool:
                                if ue.reception_ack_during_wait is True:
                                    ue.reception_ack_during_wait = False

                                counter_new_act = 0
                                for packet in ue.ul_buffer.buffer_packet_list:

                                    if packet.get_retransmission_packets() is True and ue.get_last_action() is not None:
                                        counter_new_act += 1

                                if ue.check_generated_packet_present():
                                    ue.new_action_bool = True

                                if counter_new_act == 0:
                                    ue.new_action_bool = True

                                backoff_duration_tick = get_backoff_duration(input_ue=ue,
                                                                             input_contention_window_int=
                                                                             contention_window_int,
                                                                             input_t_backoff_tick=t_backoff_tick,
                                                                             input_max_prop_delay_tick=
                                                                             max_prop_delay_tick)
                                go_in_backoff(input_ue=ue, current_tick=t,
                                              input_backoff_duration_tick=backoff_duration_tick,
                                              input_enable_print=enable_print)
                                ue.list_ack_sent_from_bs.clear()
                                ue.ues_colliding_at_ue.clear()
                                if len(ue.ues_interfering_at_ue) > 0:
                                    ue.ues_interfering_at_ue.clear()

                            elif go_in_idle_bool:
                                if ue.reception_ack_during_wait is True:
                                    ue.reception_ack_during_wait = False
                                go_in_idle(input_ue=ue, current_tick=t, input_enable_print=enable_print)
                                ue.list_ack_sent_from_bs.clear()
                                ue.ues_colliding_at_ue.clear()
                                if len(ue.ues_interfering_at_ue) > 0:
                                    ue.ues_interfering_at_ue.clear()

                # ********************** BS STATES *****************************
                if t == bs.get_state_duration():

                    if bs.get_state() == 'RX':
                        bs.rx_data = False
                        # The BS has received a DATA
                        # Find the corresponding transmission
                        data_rx_at_bs_starting_tick, data_rx_at_bs_ending_tick, output_data_rx_packet_id, data_rx_at_bs_ue_id = (
                            find_data_rx_times_at_bs_tick(
                                input_simulator_timing_structure=simulator_timing_structure,
                                current_tick=t))

                        # The BS has received an ACK
                        # Find the corresponding transmission

                        (ack_rx_at_bs_starting_tick, ack_rx_at_bs_ending_tick, ack_rx_at_bs_rx_id_int, ack_rx_id,
                         ack_rx_at_bs_tx_id_str, n_ack_rx_simultaneously) = find_ack_rx_times_at_bs_tick(
                            input_simulator_timing_structure=simulator_timing_structure, current_tick=t)

                        if data_rx_at_bs_starting_tick is not None and data_rx_at_bs_ending_tick is not None and \
                                len(data_rx_at_bs_ue_id) > 0:

                            for index in range(len(data_rx_at_bs_ue_id)):
                                # Compute the tx-rx distance
                                tx_rx_distance_m = compute_distance_m(tx=ue_array[data_rx_at_bs_ue_id[index]], rx=bs)

                                # Check if the shadowing sample should be changed
                                if t >= shadowing_next_tick:
                                    shadowing_sample_index = shadowing_sample_index + 1
                                    shadowing_next_tick = t + shadowing_coherence_time_tick_duration

                                data_rx_power = thz_channel.get_3gpp_prx_db(
                                    tx=ue_array[data_rx_at_bs_ue_id[index]], rx=bs,
                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                    tx_rx_distance_m=tx_rx_distance_m,
                                    apply_fading=apply_fading,
                                    bandwidth_hz=bandwidth_hz,
                                    clutter_density=clutter_density,
                                    input_shadowing_sample_index=shadowing_sample_index,
                                    antenna_gain_model=antenna_gain_model,
                                    use_huawei_measurements=use_huawei_measurements,
                                    input_average_clutter_height_m=average_machine_height_m,
                                    los_cond='bs_ue')

                                # Compute the SNR between the current receiving UE and the transmitting UE
                                snr_db = thz_channel.get_3gpp_snr_db(
                                    tx=ue_array[data_rx_at_bs_ue_id[index]], rx=bs,
                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                    tx_rx_distance_m=tx_rx_distance_m,
                                    apply_fading=apply_fading,
                                    bandwidth_hz=bandwidth_hz,
                                    clutter_density=clutter_density,
                                    input_shadowing_sample_index=shadowing_sample_index,
                                    antenna_gain_model=antenna_gain_model,
                                    use_huawei_measurements=use_huawei_measurements,
                                    input_average_clutter_height_m=average_machine_height_m,
                                    los_cond='bs_ue')

                                sir_dB = None
                                n_interferers = 0

                                # this method takes in input both the current UE_ID that has received a data and both the
                                # ID of the UE that has sent the data
                                # -> need to check if there is another UE != from these two UEs that has TX a DATA or an ACK
                                ues_colliding_at_bs.clear()
                                ues_colliding_at_bs = check_collision_bs(
                                    input_simulator_timing_structure=simulator_timing_structure,
                                    input_ue_id=data_rx_at_bs_ue_id[index],
                                    input_t_start_rx=data_rx_at_bs_starting_tick,
                                    input_t_end_rx=data_rx_at_bs_ending_tick, ues_colliding=ues_colliding_at_bs)

                                if f'UE_{data_rx_at_bs_ue_id[index]}' in ues_colliding_at_bs:
                                    ues_colliding_at_bs.remove((f'UE_{data_rx_at_bs_ue_id[index]}',
                                                                data_rx_at_bs_starting_tick, data_rx_at_bs_ending_tick))

                                success_at_bs = False

                                useful_rx_power_db = data_rx_power

                                add_interferer = True
                                if len(ues_interfering_at_bs) > 0:
                                    for i in range(len(ues_interfering_at_bs)):
                                        if data_rx_at_bs_ue_id[index] == \
                                                ues_interfering_at_bs[i][0] and \
                                                t == ues_interfering_at_bs[i][1]:
                                            add_interferer = False
                                            # the useful user will become an interferer for the next reception,
                                            # so save the ID and the current ending tick of this reception
                                if add_interferer is True:
                                    ues_interfering_at_bs.append((data_rx_at_bs_ue_id[index],
                                                                  data_rx_at_bs_starting_tick,
                                                                  data_rx_at_bs_ending_tick))

                                interference_rx_power = 0
                                if len(ues_colliding_at_bs) > 0:
                                    for user in ue_array:
                                        for i in range(len(ues_colliding_at_bs)):
                                            if f'UE_{user.get_ue_id()}' == ues_colliding_at_bs[i][0] and \
                                                    user.get_ue_id() != data_rx_at_bs_ue_id[index]:
                                                if ues_colliding_at_bs[i][1] < data_rx_at_bs_ending_tick < \
                                                        ues_colliding_at_bs[i][2]:
                                                    t_overlap = ((data_rx_at_bs_ending_tick -
                                                                  ues_colliding_at_bs[i][1]) /
                                                                 (data_rx_at_bs_ending_tick -
                                                                  data_rx_at_bs_starting_tick))
                                                else:
                                                    t_overlap = ((ues_colliding_at_bs[i][2] -
                                                                  ues_colliding_at_bs[i][1]) /
                                                                 (data_rx_at_bs_ending_tick -
                                                                  data_rx_at_bs_starting_tick))
                                                n_interferers += 1
                                                tx_rx_distance_m = compute_distance_m(tx=user, rx=bs)
                                                interference_rx_power +=  t_overlap * thz_channel.get_3gpp_prx_lin(
                                                    tx=user, rx=bs,
                                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                                    tx_rx_distance_m=tx_rx_distance_m,
                                                    apply_fading=apply_fading,
                                                    bandwidth_hz=bandwidth_hz,
                                                    clutter_density=clutter_density,
                                                    input_shadowing_sample_index=shadowing_sample_index,
                                                    antenna_gain_model=antenna_gain_model,
                                                    use_huawei_measurements=use_huawei_measurements,
                                                    input_average_clutter_height_m=average_machine_height_m,
                                                    los_cond='bs_ue')


                                if len(ues_interfering_at_bs) > 0:
                                    # for the interfering users (whose that before where useful user),
                                    # I have to check if their ending tick of ACK or DATA is between the
                                    # staring and the ending tick of the actual RX DATA/ACK
                                    # If Yes -> it is an interferer
                                    # If No -> remove from the list of interferers.
                                    copy_of_list = deepcopy(ues_interfering_at_bs)
                                    for user in ue_array:
                                        for i in range(len(copy_of_list)):
                                            if user.get_ue_id() != data_rx_at_bs_ue_id[index]:
                                                if user.get_ue_id() == copy_of_list[i][0]:
                                                    if data_rx_at_bs_starting_tick < \
                                                            copy_of_list[i][2]:
                                                        if data_rx_at_bs_starting_tick > copy_of_list[i][1]:
                                                            t_overlap = ((copy_of_list[i][2] -
                                                                          data_rx_at_bs_starting_tick) /
                                                                         (data_rx_at_bs_ending_tick -
                                                                          data_rx_at_bs_starting_tick))
                                                        else:
                                                            t_overlap = ((copy_of_list[i][2] -
                                                                          copy_of_list[i][1]) /
                                                                         (data_rx_at_bs_ending_tick -
                                                                          data_rx_at_bs_starting_tick))
                                                        n_interferers += 1
                                                        tx_rx_distance_m = compute_distance_m(tx=user, rx=bs)

                                                        interference_rx_power += t_overlap *  thz_channel.get_3gpp_prx_lin(
                                                            tx=user, rx=bs,
                                                            carrier_frequency_ghz=carrier_frequency_ghz,
                                                            tx_rx_distance_m=tx_rx_distance_m,
                                                            apply_fading=apply_fading,
                                                            bandwidth_hz=bandwidth_hz,
                                                            clutter_density=clutter_density,
                                                            input_shadowing_sample_index=shadowing_sample_index,
                                                            antenna_gain_model=antenna_gain_model,
                                                            use_huawei_measurements=use_huawei_measurements,
                                                            input_average_clutter_height_m=average_machine_height_m,
                                                            los_cond='bs_ue')
                                                    else:
                                                        ues_interfering_at_bs.remove((user.get_ue_id(),
                                                                                      copy_of_list[i][1],
                                                                                      copy_of_list[i][2]))
                                if interference_rx_power == 0:
                                    sinr_db = snr_db

                                else:
                                    noise_power_dbw = thz_channel.get_thermal_noise_power_dbw(input_noise_figure=noise_figure_bs, bandwidth_hz=bandwidth_hz)
                                    noise_power = 10 ** (noise_power_dbw / 10)
                                    noise_plus_interference = noise_power + interference_rx_power
                                    useful_rx_power = 10 ** (useful_rx_power_db / 10)
                                    sinr = useful_rx_power / noise_plus_interference
                                    sinr_db = 10 * log10(sinr)

                                if sinr_db >= sinr_th_db:
                                    success = True
                                else:
                                    success = False

                                if success:
                                    packets_received = 0
                                    packets_received_relay = 0
                                    packets_received_at_bs = 0

                                    # A DATA has been successfully received, so go in TX_ACK
                                    # Go in TX_ACK
                                    # the BS has to WAIT the end of a burst TX from a UE, before going in ACK TX
                                    bs.sequence_number_of_packet_rx = output_data_rx_packet_id[index]

                                    # Find the traffic type of the successful UE
                                    successful_ue_traffic_type = None
                                    ack_packet_id = None
                                    for ue in ue_array:
                                        if ue.get_ue_id() == data_rx_at_bs_ue_id[index]:
                                            if enable_print:
                                                print("Successful transmission from UE: ", data_rx_at_bs_ue_id[index], " at t = ", t)
                                            successful_ue_traffic_type = ue.get_traffic_type()

                                            # Save the id of the UE and the ID of the packet received, to avoid counting
                                            # multiple reception
                                            buffer_size_tick = 0
                                            if len(ue.buffer_packet_sent) > 0:

                                                # Save in a list the ID of the UE for which the BS has received a DATA (before going in ACK TX)
                                                if ue.get_ue_id() not in bs.id_ues_data_rx:
                                                    bs.id_ues_data_rx.append(ue.get_ue_id())
                                                # update the end tick of BS RX only if the new one is greater
                                                if (bs.end_of_rx_for_ack_tx is None or ue.end_data_tx +
                                                        ue.get_prop_delay_to_bs_tick() > bs.end_of_rx_for_ack_tx):
                                                    bs.end_of_rx_for_ack_tx = ue.end_data_tx + ue.get_prop_delay_to_bs_tick()


                                                for n_packet in range(len(ue.buffer_packet_sent)):
                                                    # 1) check if the UE is a relay or not
                                                    # 2a) If relay -> check if the packet has been forwarded and if the BS has
                                                    # already received it
                                                    # 2b) If not relay -> check if the BS has already RX that packet, because maybe
                                                    # forwarded from another UE

                                                    # need to process only the current packet received from whose in the burst:
                                                    if (bs.sequence_number_of_packet_rx == ue.buffer_packet_sent[n_packet].get_id()) and (ue.buffer_packet_sent[n_packet].address == 'BS' or ue.buffer_packet_sent[n_packet].address == "-1"):
                                                        if ue.buffer_packet_sent[n_packet].get_id() not in \
                                                                bs.packet_id_received[ue.get_ue_id()]:
                                                            # if the packet of the UE has not been forwarded
                                                            if ue.buffer_packet_sent[n_packet].get_data_rx_from_ue() is \
                                                                    None:
                                                                bs.packet_id_received[ue.get_ue_id()]. \
                                                                    append(ue.buffer_packet_sent[n_packet].get_id())
                                                                ue.set_ack_packet_id_ue(
                                                                    ue.buffer_packet_sent[n_packet].get_id())
                                                                ack_packet_id = ue.get_ack_packet_id_ue()
                                                                ue.list_ack_sent_from_bs.append(
                                                                    ue.buffer_packet_sent[n_packet].get_id())
                                                                packets_received = 1
                                                                packets_received_at_bs += 1
                                                                bs.update_n_data_rx_from_ues(input_ue_id=data_rx_at_bs_ue_id[index],
                                                                                             packets_received=packets_received)
                                                                # Latency computation for the packet generated by the reference UE:
                                                                # add the time needed to transmit the ACK +
                                                                # the propagation delay between the BS and the UE
                                                                if star_topology is False:
                                                                    latency = (t + t_ack_tick + ue.get_prop_delay_to_bs_tick() -
                                                                               ue.packet_generation_instant) * \
                                                                              simulator_tick_duration_s
                                                                    ue.latency_ue.append(latency)
                                                                    # print("Latency for UE ", ue.get_ue_id(), " = ", latency,
                                                                    #       "tick -> Packet generated at t = ",
                                                                    #       ue.packet_generation_instant)

                                                            else:
                                                                if enable_print:
                                                                    print("Packet: ",
                                                                          ue.buffer_packet_sent[n_packet].get_id(),
                                                                          "is Packet: ", ue.buffer_packet_sent[
                                                                              n_packet].get_packet_id_generator(),
                                                                          " generated from ",
                                                                          ue.buffer_packet_sent[
                                                                              n_packet].get_generated_by_ue())
                                                                bs.packet_id_received[ue.get_ue_id()].append(
                                                                    ue.buffer_packet_sent[n_packet].get_id())
                                                                ue.set_ack_packet_id_ue(
                                                                    ue.buffer_packet_sent[n_packet].get_id())
                                                                ack_packet_id = ue.get_ack_packet_id_ue()
                                                                ue.list_ack_sent_from_bs.append(
                                                                    ue.buffer_packet_sent[n_packet].get_id())


                                                                # Check if for that UE, the BS has already counted the reception ->
                                                                # if yes, decrease the number of packets received,
                                                                # if not, keep it as it is

                                                                user = ue.buffer_packet_sent[
                                                                    n_packet].get_generated_by_ue()
                                                                packet_id = ue.buffer_packet_sent[
                                                                    n_packet].get_packet_id_generator()

                                                                if enable_print:
                                                                        print("The BS has received packet ", packet_id,
                                                                              " generated from UE ", user)
                                                                for i in ue_array:
                                                                    if i.get_ue_id() == user:
                                                                        i.set_ack_packet_id_ue(packet_id)

                                                                packet_already_count = False
                                                                for packet in range(len(bs.packet_id_received[user])):
                                                                    if bs.packet_id_received[user][packet] == packet_id:
                                                                        packet_already_count = True
                                                                if packet_already_count is False:
                                                                    if enable_print:
                                                                        print("BS doesn't count for that RX, so it counts now.")
                                                                    bs.packet_id_received[user].append(packet_id)

                                                                    packets_received_relay = 1
                                                                    packets_received_at_bs += 1
                                                                    bs.update_n_data_rx_from_ues(
                                                                        input_ue_id=ue.buffer_packet_sent[
                                                                            n_packet].get_generated_by_ue(),
                                                                        packets_received=packets_received_relay)
                                                                    if star_topology is False:
                                                                        # In case of multi-hop, for the latency computation
                                                                        # it is necessary to take into account to the
                                                                        # packet generation instant of the source UE
                                                                        for i in ue_array:
                                                                            if i.get_ue_id() == user:
                                                                                latency = (t + t_ack_tick +
                                                                                           i.get_prop_delay_to_bs_tick() -
                                                                                           ue.buffer_packet_sent[n_packet].get_generated_by_ue_time_instant_tick()) * \
                                                                                          simulator_tick_duration_s
                                                                                i.latency_ue.append(latency)
                                                                                # print("Latency for UE ", i.get_ue_id(), " = ", latency,
                                                                                #       "tick -> Packet generated at t = ",
                                                                                #       i.packet_generation_instant)

                                                                else:
                                                                    if enable_print:
                                                                        print(" BS has already count that RX")

                                                        else:
                                                            ack_packet_id = ue.buffer_packet_sent[n_packet].get_id() # Andrea
                                                            ue.list_ack_sent_from_bs.append(
                                                                ue.buffer_packet_sent[n_packet].get_id())

                                            else:
                                                ue.set_ack_packet_id_ue(output_data_rx_packet_id[index])
                                                ack_packet_id = ue.get_ack_packet_id_ue()
                                                ue.list_ack_sent_from_bs.append(output_data_rx_packet_id[index])
                                                if enable_print:
                                                    print("Currently, UE ", ue.get_ue_id(),
                                                          " has already removed the packet "
                                                          "from the queue.")

                                    if successful_ue_traffic_type is not None:

                                        bs.update_n_data_rx(input_ue_traffic_type=successful_ue_traffic_type,
                                                            packets_received=packets_received_at_bs,
                                                            input_enable_print=enable_print)

                                    else:
                                        sys.exit('UE traffic type not yet supported '
                                                 'when controlling the successful transmitting UE at the BS')

                                    bs.temp_packet_id_received[data_rx_at_bs_ue_id[index]].append(ack_packet_id)


                                    bs.rx_data = True

                                if len(simulator_timing_structure[f'BS']['DATA_RX'][f'UE_{data_rx_at_bs_ue_id[index]}'])\
                                        > 1:
                                    # Update the timing structure to reset this reception
                                    remove_item_in_timing_structure(
                                        input_simulator_timing_structure=simulator_timing_structure,
                                        input_rx_key='BS',
                                        input_type_key='DATA_RX',
                                        input_tx_key=f'UE_{data_rx_at_bs_ue_id[index]}'
                                    )

                        # The BS has to check when it receives an ACK and update the timing structure
                        if ack_rx_at_bs_starting_tick is not None and ack_rx_at_bs_ending_tick is not None:

                            for index in range(len(ack_rx_at_bs_tx_id_str)):
                                ue_id = None
                                tx = None
                                # Compute the tx-rx distance
                                for ue in ue_array:
                                    if f'UE_{ue.get_ue_id()}' == ack_rx_at_bs_tx_id_str[index]:
                                        ue_id = ue.get_ue_id()
                                        tx = ue
                                tx_rx_distance_m = compute_distance_m(tx=tx, rx=bs)

                                # Check if the shadowing sample should be changed
                                if t >= shadowing_next_tick:
                                    shadowing_sample_index = shadowing_sample_index + 1
                                    shadowing_next_tick = t + shadowing_coherence_time_tick_duration

                                ack_rx_power = thz_channel.get_3gpp_prx_db(
                                    tx=tx, rx=bs,
                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                    tx_rx_distance_m=tx_rx_distance_m,
                                    apply_fading=apply_fading,
                                    bandwidth_hz=bandwidth_hz,
                                    clutter_density=clutter_density,
                                    input_shadowing_sample_index=shadowing_sample_index,
                                    antenna_gain_model=antenna_gain_model,
                                    use_huawei_measurements=use_huawei_measurements,
                                    input_average_clutter_height_m=average_machine_height_m,
                                    los_cond='bs_ue')

                                # Compute the SNR between the current receiving UE and the transmitting UE
                                snr_db = thz_channel.get_3gpp_snr_db(
                                    tx=tx, rx=bs,
                                    carrier_frequency_ghz=carrier_frequency_ghz,
                                    tx_rx_distance_m=tx_rx_distance_m,
                                    apply_fading=apply_fading,
                                    bandwidth_hz=bandwidth_hz,
                                    clutter_density=clutter_density,
                                    input_shadowing_sample_index=shadowing_sample_index,
                                    antenna_gain_model=antenna_gain_model,
                                    use_huawei_measurements=use_huawei_measurements,
                                    input_average_clutter_height_m=average_machine_height_m,
                                    los_cond='bs_ue')


                                # this method takes in input both the current UE_ID that has received a data and both the
                                # ID of the UE that has sent the data
                                # -> need to check if there is another UE != from these two UEs that has TX a DATA or an ACK
                                ues_colliding_at_bs.clear()
                                ues_colliding_at_bs = check_collision_bs(
                                    input_simulator_timing_structure=simulator_timing_structure,
                                    input_ue_id=ue_id,
                                    input_t_start_rx=ack_rx_at_bs_starting_tick,
                                    input_t_end_rx=ack_rx_at_bs_ending_tick, ues_colliding=ues_colliding_at_bs)

                                if f'UE_{ue_id}' in ues_colliding_at_bs:
                                    ues_colliding_at_bs.remove(f'UE_{ue_id}')

                                add_interferer = True
                                if len(ues_interfering_at_bs) > 0:
                                    for i in range(len(ues_interfering_at_bs)):
                                        if ue_id == \
                                                ues_interfering_at_bs[i][0] and \
                                                t == ues_interfering_at_bs[i][1]:
                                            add_interferer = False
                                            # the useful user will become an interferer for the next reception,
                                            # so save the ID and the current ending tick of this reception
                                if add_interferer is True:
                                    ues_interfering_at_bs.append((ue_id,
                                                                  ack_rx_at_bs_starting_tick,
                                                                  ack_rx_at_bs_ending_tick))

                                # If the BS has RX a data has already removed ACK with no phy collision
                                if ack_rx_at_bs_tx_id_str[index] is not None:
                                    for packet_id in ack_rx_id:
                                        if len(simulator_timing_structure['BS']['ACK_RX'][ack_rx_at_bs_tx_id_str[index]]) > 1:
                                            if simulator_timing_structure['BS']['ACK_RX'][ack_rx_at_bs_tx_id_str[index]][:, 3][1] == packet_id:

                                            # Update the timing structure to reset this reception
                                                remove_item_in_timing_structure(
                                                    input_simulator_timing_structure=simulator_timing_structure,
                                                    input_rx_key='BS',
                                                    input_type_key='ACK_RX',
                                                    input_tx_key=ack_rx_at_bs_tx_id_str[index]
                                                )
                            ack_rx_id.clear()

                        if bs.end_of_rx_for_ack_tx == t:

                            #Update the simulator timing structure for ACK_RX
                            for index in range(len(bs.id_ues_data_rx)):
                                ue_id = bs.id_ues_data_rx[index]
                                # if the BS has received some data from the ue_id contained within the list
                                bool_elements_none = all(None == bs.temp_packet_id_received[ue_id][i] for i in
                                                         range(len(bs.temp_packet_id_received[ue_id]))) # if all None -> no acks to be sent
                                if len(bs.temp_packet_id_received[ue_id]) > 0 and bool_elements_none is False:
                                    bs.packet_rx = True
                                    for i in range(len(bs.temp_packet_id_received[ue_id])):
                                        # update the timing structure
                                        for ue in ue_array:
                                            insert_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_starting_tick=bs.end_of_rx_for_ack_tx + ue.get_prop_delay_to_bs_tick()
                                                                    + index * t_ack_tick,
                                                input_final_tick=bs.end_of_rx_for_ack_tx + ue.get_prop_delay_to_bs_tick() +
                                                                 t_ack_tick + index * t_ack_tick,
                                                input_third_field=ue_id,
                                                input_fourth_field=bs.temp_packet_id_received[ue_id][i],
                                                input_rx_key=f'UE_{ue.get_ue_id()}',
                                                input_type_key='ACK_RX',
                                                input_tx_key='BS',
                                            )
                                bs.temp_packet_id_received[ue_id] = ([])

                            if bs.packet_rx is True:
                                go_in_tx_ack_bs(input_bs=bs, current_tick=t,
                                                input_ack_duration_tick=len(bs.id_ues_data_rx) * t_ack_tick,
                                                input_enable_print=enable_print)
                                bs.packet_rx = False
                            else:
                                go_rx_ack_bs(input_bs=bs, current_tick=t,
                                             input_rx_duration_tick=tot_simulation_time_tick + 1,
                                             input_enable_print=enable_print)
                            bs.id_ues_data_rx.clear()

                            for ue in ue_array:

                                new_list = deepcopy(simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}'])
                                for i in range(len(new_list)):
                                    if len(simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}']) > 1:

                                        if (simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}'][:, 1][
                                            1] <= bs.get_end_tx_ack() or
                                                simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}'][:, 0][
                                            1] <= bs.get_end_tx_ack()):
                                            # Update the timing structure to reset this reception
                                            remove_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_rx_key='BS',
                                                input_type_key='ACK_RX',
                                                input_tx_key=f'UE_{ue.get_ue_id()}'
                                            )

                                new_list = deepcopy(simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}'])
                                for i in range(len(new_list)):
                                    if len(simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}']) > 1:

                                        if (simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}'][:, 1][
                                            1] <= bs.get_end_tx_ack() or
                                                simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}'][:, 0][
                                            1] <= bs.get_end_tx_ack()):
                                            # Update the timing structure to reset this reception
                                            remove_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_rx_key='BS',
                                                input_type_key='DATA_RX',
                                                input_tx_key=f'UE_{ue.get_ue_id()}'
                                            )

                        elif bs.sequence_number_of_packet_rx > 0 and bs.end_of_rx_for_ack_tx == t:
                            # Update the simulator timing structure for ACK_RX
                            for index in range(len(bs.id_ues_data_rx)):
                                ue_id = bs.id_ues_data_rx[index]
                                # if the BS has received some data from the ue_id contained within the list
                                bool_elements_none = all(None == bs.temp_packet_id_received[ue_id][i] for i in
                                                         range(len(bs.temp_packet_id_received[
                                                             ue_id])))  # if all None -> no acks to be sent
                                if len(bs.temp_packet_id_received[ue_id]) > 0 and bool_elements_none is False:
                                    bs.packet_rx = True
                                    for i in range(len(bs.temp_packet_id_received[ue_id])):
                                        # update the timing structure
                                        for ue in ue_array:
                                            insert_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_starting_tick=bs.end_of_rx_for_ack_tx + ue.get_prop_delay_to_bs_tick()
                                                                    + index * t_ack_tick,
                                                input_final_tick=bs.end_of_rx_for_ack_tx + ue.get_prop_delay_to_bs_tick() +
                                                                 t_ack_tick + index * t_ack_tick,
                                                input_third_field=ue_id,
                                                input_fourth_field=bs.temp_packet_id_received[ue_id][i],
                                                input_rx_key=f'UE_{ue.get_ue_id()}',
                                                input_type_key='ACK_RX',
                                                input_tx_key='BS',
                                            )
                                bs.temp_packet_id_received[ue_id] = ([])
                            if bs.packet_rx is True:
                                go_in_tx_ack_bs(input_bs=bs, current_tick=t,
                                                input_ack_duration_tick=len(bs.id_ues_data_rx) * t_ack_tick,
                                                input_enable_print=enable_print)
                                bs.packet_rx = False
                            else:
                                go_rx_ack_bs(input_bs=bs, current_tick=t,
                                             input_rx_duration_tick=tot_simulation_time_tick + 1,
                                             input_enable_print=enable_print)
                            bs.id_ues_data_rx.clear()

                            for ue in ue_array:
                                new_list = deepcopy(simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}'])
                                for i in range(len(new_list)):

                                    if len(simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}']) > 1:

                                        if (simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}'][:, 1][
                                            1] <= bs.get_end_tx_ack() or
                                                simulator_timing_structure['BS']['ACK_RX'][f'UE_{ue.get_ue_id()}'][:, 0][
                                            1] <= bs.get_end_tx_ack()):
                                            # Update the timing structure to reset this reception
                                            remove_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_rx_key='BS',
                                                input_type_key='ACK_RX',
                                                input_tx_key=f'UE_{ue.get_ue_id()}'
                                            )

                                new_list = deepcopy(simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}'])

                                for i in range(len(new_list)):

                                    if len(simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}']) > 1:

                                        if (simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}'][:, 1][
                                            1] <= bs.get_end_tx_ack() or
                                                simulator_timing_structure['BS']['DATA_RX'][f'UE_{ue.get_ue_id()}'][:, 0][
                                            1] <= bs.get_end_tx_ack()):
                                            # Update the timing structure to reset this reception
                                            remove_item_in_timing_structure(
                                                input_simulator_timing_structure=simulator_timing_structure,
                                                input_rx_key='BS',
                                                input_type_key='DATA_RX',
                                                input_tx_key=f'UE_{ue.get_ue_id()}'
                                            )

                        else:  # Reception is not successful, so remain in RX
                            go_rx_ack_bs(input_bs=bs, current_tick=t,
                                         input_rx_duration_tick=tot_simulation_time_tick + 1,
                                         input_enable_print=enable_print)


                    elif bs.get_state() == 'TX_ACK':
                        go_rx_ack_bs(input_bs=bs, current_tick=t, input_rx_duration_tick=tot_simulation_time_tick + 1,
                                     input_enable_print=enable_print)

                # Move time forward based on the next occurring event ->
                # this allows to reduce computation time because we don't do a tick by tick simulation, but there is a
                # jump to the next event (new state of a UE/BS, a reception or the end of a transmission and so on)
                t_states_ues_tick = [ue.get_state_duration() for ue in ue_array]  # All UEs state duration
                t_generations_ues_tick = [ue.get_next_packet_generation_instant() for ue in
                                          ue_array]  # All UEs state duration
                t_state_bs_tick = bs.get_state_duration()  # BS state duration

                if mobility_obstacle or mobility_spawn:
                    if step_size == 4.5:
                        next_t_change = t_change + math.floor(tot_simulation_time_tick / 2)
                    elif step_size == 2.25:
                        next_t_change = t_change + math.floor(tot_simulation_time_tick / 3)
                elif mobility_shuffle:
                    next_t_change = t_change + math.floor(tot_simulation_time_tick / (mobility_changes + 1))
                else:
                    next_t_change = t_change + math.floor(tot_simulation_time_tick + 1)
                if t == next_t_change:
                    print("t: ", t)
                    t_change = next_t_change
                final_rx_times = np.inf  # Store the minimum final tick of a reception
                for key_ext, values_ext in simulator_timing_structure.items():
                    for key_int, values_int in values_ext['DATA_RX'].items():
                        min_index = np.argmin(values_ext['DATA_RX'][key_int][:, 1])
                        min_row = values_ext['DATA_RX'][key_int][min_index, :]
                        final_rx_times = min(final_rx_times, min_row[1])
                    for key_int, values_int in values_ext['ACK_RX'].items():
                        min_index = np.argmin(values_ext['ACK_RX'][key_int][:, 1])
                        min_row = values_ext['ACK_RX'][key_int][min_index, :]
                        final_rx_times = min(final_rx_times, min_row[1])
                # Min(UEs state durations, BS state duration, generation times, reception times)
                t_next = min(t_state_bs_tick, min(t_states_ues_tick),
                             min(t_generations_ues_tick), final_rx_times, next_t_change)
                t = t_next  # Updated time

            """
            Compute simulation outputs
            """
            if enable_print:
                print("Simulation ended at t = ", tot_simulation_time_tick)


            compute_simulator_outputs(ue_array=ue_array, bs=bs, simulation_time_s=simulation_time_s, inputs_dict=inputs,
                                      output_dict=output_dict,
                                      output_n_ue=n_ue, output_n_sim=n_simulation)

            gc.collect()
            print(f"[DEBUG] Memory usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2):.2f} MB")


"""
    Plot and save results
"""
x_data = [n_ue for n_ue in range(initial_number_of_ues, final_number_of_ues + 1, step_number_of_ues)]

# Initialize a dictionary to store the averages
averaged_data = {}

for metric, users_data in output_dict.items():
    averaged_data[metric] = []
    for user_count, simulations in users_data.items():
        # Collect all simulation arrays for this user count
        sim_averages = []
        for sim_key, sim_array in simulations.items():
            if metric == 'l':
                # Filter out zero values
                non_zero_values = sim_array[sim_array != 0]
                if len(non_zero_values) > 0:
                    sim_average = np.mean(non_zero_values)
                else:
                    # Handle cases where all values are zero
                    sim_average = 0
            else:
                # Average over the simulation's outputs normally
                sim_average = np.mean(sim_array)
            sim_averages.append(sim_average)

        # Average across simulations
        overall_average = np.mean(sim_averages)
        averaged_data[metric].append(overall_average)

# Print output
print("p_mac: ", averaged_data['p_mac'])
print("S_net: ", averaged_data['s'])
print("Latency: ", averaged_data['l'])
print("Energy: ", averaged_data['e'])
print("Jain Index: ", averaged_data['j_index'])

averaged_ticks_BO = 0
BO_instances= 0
averaged_ticks_TX_ACK= 0
TX_ACK_instances= 0
averaged_ticks_TX_DATA= 0
TX_DATA_instances= 0
averaged_ticks_WAIT_ACK= 0
WAIT_ACK_instances = 0
averaged_forced_broadcast_actions = 0
averaged_n_forwarding = 0
tot_av_interferers = 0
for ue in ue_array:
    averaged_ticks_BO += np.sum(ue.ticks_in_BO)
    BO_instances += len(ue.ticks_in_BO)
    averaged_ticks_TX_ACK += np.sum(ue.ticks_in_TX_ACK)
    TX_ACK_instances += len(ue.ticks_in_TX_ACK)
    averaged_ticks_TX_DATA += np.sum(ue.ticks_in_TX_DATA)
    TX_DATA_instances += len(ue.ticks_in_TX_DATA)
    averaged_ticks_WAIT_ACK += np.sum(ue.ticks_in_WAIT_ACK)
    WAIT_ACK_instances += len(ue.ticks_in_WAIT_ACK)
    averaged_forced_broadcast_actions += ue.forced_broadcast_actions_counter
    averaged_n_forwarding += ue.n_forwarding
    # print("Average number of interferers per UE ", ue.get_ue_id(), " = ", np.mean(ue.n_interfering))
    tot_av_interferers += np.mean(ue.n_interfering)

print("Ticks in BO: ", averaged_ticks_BO / BO_instances)
print("Ticks in TX_ACK: ", averaged_ticks_TX_ACK / TX_ACK_instances)
print("Ticks in TX_DATA: ", averaged_ticks_TX_DATA / TX_DATA_instances)
print("Ticks in WAIT_ACK: ", averaged_ticks_WAIT_ACK / WAIT_ACK_instances)
print("Average Forced Broadcast Actions per Simulation: ", averaged_forced_broadcast_actions / n_simulations / len(ue_array))

print("Average N forwarding per Simulation: ", averaged_n_forwarding / n_simulations / len(ue_array))
print("Average number of interferers = ", tot_av_interferers / len(ue_array))

# Get current date
current_date = datetime.now()

# Format the date components
year = current_date.year
month = current_date.month
day = current_date.day
final_file_name_output_1 = 'multi_hop_industrial_simulator/results/' + f'{year}_{month:02d}_{day:02d}_' + 'TB_output_dict' + '_final_UEs_' + str(final_number_of_ues) + '_Payload_' + str(payload_fq)+ '_contention_window_' + str(contention_window_int) + '_buffer_size_' + str(max_n_packets_to_be_forwarded) + '_TTL_' + str(TTL) + '.txt'
final_file_name_output_2 = 'multi_hop_industrial_simulator/results/' + f'{year}_{month:02d}_{day:02d}_' + 'TB_averaged_data' + '_final_UEs_' + str(final_number_of_ues) +'_Payload_' + str(payload_fq)+ '_contention_window_' + str(contention_window_int) + '_buffer_size_' + str(max_n_packets_to_be_forwarded) + '_TTL_' + str(TTL) + '.txt'
final_file_name_output_3 = 'multi_hop_industrial_simulator/results/' + f'{year}_{month:02d}_{day:02d}_' + 'TB_inputs' + '_final_UEs_' + str(final_number_of_ues) + '_Payload_' + str(payload_fq)+ '_contention_window_' + str(contention_window_int) + '_buffer_size_' + str(max_n_packets_to_be_forwarded) + '_TTL_' + str(TTL) + '.txt'

# Plot and save simulation output
with open(final_file_name_output_1, "w") as file:
    file.write(json.dumps(output_dict, indent=0, default=str))  # Adding indentation for readability
with open(final_file_name_output_2, "w") as file:
    file.write(json.dumps(averaged_data, indent=0, default=str))  # Adding indentation for readability
with open(final_file_name_output_3, "w") as file:
    file.write(json.dumps(inputs, indent=0, default=str))  # Adding indentation for readability

