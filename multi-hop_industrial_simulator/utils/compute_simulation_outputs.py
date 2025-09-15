"""
    Utility scripts to compute the outputs of simulations
"""
import sys
import numpy as np

from timessim.network.bs import BS
from numpy import ndarray

# Function to compute the outputs of the simulator
def compute_simulator_outputs(ue_array: ndarray, bs: BS, simulation_time_s: float, inputs_dict: dict, output_dict: dict,
                              output_n_ue: int, output_n_sim: int):
    # Metrics
    payload_fq_bytes = inputs_dict.get('traffic_fq').get('payload')

    # Loop over UEs
    for ue_index, ue in enumerate(ue_array):
        # If the simulation stopped before storing a successful data reception at the BS,
        # reduce the counter of successful data reception by 1 (if it is not 1)
        if ue.get_n_data_tx() != (bs.get_n_data_rx_from_ues(input_ue_id=ue.get_ue_id()) + ue.get_n_data_discarded()):
            packet_tx_ue = ue.get_n_data_tx()

            for packet in ue.ul_buffer.get_packet_list():
                if packet.get_data_to_be_forwarded_bool() is False and packet.get_id() not in \
                        bs.packet_id_received[ue.get_ue_id()] and ue.check_last_round is True:
                    packet_tx_ue -= 1
                if packet_tx_ue == (bs.get_n_data_rx_from_ues(input_ue_id=ue.get_ue_id()) + ue.get_n_data_discarded()):
                    break
            ue.set_n_data_tx(packet_tx_ue)

        # Prints
        print("UE: ", ue.get_ue_id())
        print("DATA TX:", ue.get_n_data_tx())
        print("DATA RX AT BS: ", bs.get_n_data_rx_from_ues(input_ue_id=ue.get_ue_id()))
        # print("DATA DISCARDED: ", ue.get_n_data_discarded())
        # print("Percentage of packets discarded: ", ue.get_packet_discarded_perc())

        # Compute pmac and update the output dictionary
        if ue.get_n_data_tx() > 0:
            # Compute the pmac
            p_mac = bs.get_n_data_rx_from_ues(input_ue_id=ue.get_ue_id()) / ue.get_n_data_tx()
            output_dict['p_mac'][f"N={output_n_ue}"][f"Sim={output_n_sim}"][ue_index] = p_mac
            # output_dict['Discarded_packets_percentage'][f"N={output_n_ue}"][f"Sim={output_n_sim}"][ue_index] = ue.get_packet_discarded_perc()

        # Compute the per-user network throughput, and average latency and update the output dictionary
        s_ue = (payload_fq_bytes * 8 * bs.get_n_data_rx_from_ues(input_ue_id=ue.get_ue_id())
                * 1e-9 / simulation_time_s)
        output_dict['s_ue'][f"N={output_n_ue}"][f"Sim={output_n_sim}"][ue_index] = s_ue
        if len(ue.latency_ue) != 0:
            output_dict['l'][f"N={output_n_ue}"][f"Sim={output_n_sim}"][ue_index] = np.mean(ue.latency_ue)

        # Compute the energy consumption and update the output dictionary
        if ue.get_n_data_tx() != 0:
            output_dict['e'][f"N={output_n_ue}"][f"Sim={output_n_sim}"][ue_index] = (
                    ue.energy_consumed / ue.get_n_data_tx())

    # Compute the network throughput, and Jain index and update the output dictionary
    output_dict['s'][f"N={output_n_ue}"][f"Sim={output_n_sim}"] = (
            payload_fq_bytes * 8 * bs.get_n_data_rx_fq() * 1e-9 / simulation_time_s)
    s_array = output_dict['s_ue'][f"N={output_n_ue}"][f"Sim={output_n_sim}"]
    if np.sum(s_array) == 0:
        output_dict['j_index'][f"N={output_n_ue}"][f"Sim={output_n_sim}"] = 0
    else:
        output_dict['j_index'][f"N={output_n_ue}"][f"Sim={output_n_sim}"] = (
                np.sum(s_array) ** 2 / (len(s_array) * np.sum(s_array ** 2)))
    print("PACCH_RX with success at the BS: ", bs.get_n_data_rx_fq())

    # Print intermediate results
    for key, values in output_dict.items():
        print("Metric ", key, '= ', values)
