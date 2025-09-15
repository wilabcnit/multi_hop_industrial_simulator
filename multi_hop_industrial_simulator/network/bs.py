"""
gNodeB class
"""
from multi_hop_industrial_simulator.traffic_models.traffic_model import TrafficModel
from multi_hop_industrial_simulator.network.packet import Packet
from multi_hop_industrial_simulator.network.bs_buffer import BsBuffer
import copy as cp
import sys
from multi_hop_industrial_simulator.network.ue import Ue
from typing import List
import numpy as np
import math


class BS(TrafficModel):
    """

    Attributes
    ----------
    params: dict
        Dictionary with all inputs
    traffic_type: str
        {traffic_rt, traffic_cn, traffic_nrt}
    starting_state : str
        Initial state of the BS.
    """

    def __init__(self, params, traffic_type: str, starting_state: str, input_full_queue: bool):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.params = params
        self.carrier_frequency = 0  # Carrier frequency used for a transmission in a given instant
        self.channel = 0  # Current channel used
        self.state = starting_state
        self.t_state = 0  # Duration of the current state
        self.t_starting_state = 0  # Beginning instant of the current state
        self.t_final_state = 0  # Ending instant of the current state
        self.t_start_tx_ack = 0  # Starting tick of ACK transmission - Used for collision computation
        self.t_end_tx_ack = 0  # Ending tick of ACK transmission - Used for collision computation
        self.n_data_rx = 0  # Number of data correctly received at the BS
        self.n_data_rx_rt = 0  # Number of data from RT UEs correctly received at the BS
        self.n_data_rx_cn = 0  # Number of data from CN UEs correctly received at the BS
        self.n_data_rx_nrt = 0  # Number of data from NRT UEs correctly received at the BS
        self.n_data_rx_fq = 0  # Number of data from Full queue UEs correctly received at the BS
        self.n_data_rx_from_ues = {}  # UE_x : # of receptions
        self.t_generation_optimization = 0
        self.traffic_type = traffic_type
        self.packet = Packet()
        self. id_machine ='False'
        self.dl_buffer = BsBuffer()
        self.n_generated_packets = 0  # Number of data generated
        self.n_discarded_packets = 0  # Number of discarded data
        self.t_generation = self.t_generation_optimization
        self.bit_rate_gbits = 0  # Gbit/s
        self.is_in_los = False
        self.rx_data = False
        self.is_low_channel_condition_with_bs = 0  # True when the link with the BS is in Low condition, False otherwise
        # new dictionary to keep track of the number of packets colliding at BS from a given UE
        self.packet_number_colliding = dict()
        # sequence number of the first packet received at the BS
        self.sequence_number_of_packet_rx = 0
        self.packet_rx = False
        self.end_of_rx_for_ack_tx = None

        self.id_ues_data_rx = list()
        self.end_bo_rreq_rx_tick = None
        self.dict_rreq = dict()
        self.tx_also_rreply = False
        self.tx_also_ack = False
        self.time_shift = 0

        self.transceiver_params = {
            "Transmit power": params.get('bs').get('bs_transmit_power_dbm'),  # dBm
            "Number of antennas": params.get('bs').get('bs_number_of_antennas'),
            "Antenna efficiency": params.get('bs').get('bs_antenna_efficiency'),
            "Antenna gain": params.get('bs').get('bs_gain_db'),
            "Noise figure": params.get('bs').get('bs_noise_figure_db'),  # dB
        }
        self.packet_id_received = dict()
        self.temp_packet_id_received = dict()
        super(BS, self).__init__(input_full_queue=input_full_queue)

        self.traffic_model = {
            "traffic_rt":
                super().get_time_periodicity,
            "traffic_cn":
                super().get_time_periodicity,
            "traffic_nrt": super().get_exp_distribution
        }

    # Set the BS coordinates
    def set_coordinates(self, x_input: float, y_input: float, z_input: float):
        self.x = x_input
        self.y = y_input
        self.z = z_input

    # Get the BS coordinates
    def get_coordinates(self):
        return np.array([self.x, self.y, self.z])

    # Set the carrier frequency
    def set_carrier_frequency(self, input_carrier_frequency: float):
        self.carrier_frequency = input_carrier_frequency

    # Get the carrier frequency
    def get_carrier_frequency(self):
        return self.carrier_frequency

    # Get the state of the BS
    def get_state(self):
        return self.state

    # Set the state of the BS
    def set_state(self, input_state: str):
        self.state = input_state

    # Update the state duration of the BS
    def update_state_duration(self, input_ticks: int):
        self.t_state += input_ticks

    # Set the duration of the state of the BS
    def set_state_duration(self, input_ticks: int):
        self.t_state = input_ticks

    # Get the duration of the state of the BS
    def get_state_duration(self):
        return self.t_state

    # Set the starting tick of the state of the BS
    def set_state_starting_tick(self, input_tick: int):
        self.t_starting_state = input_tick

    # Get the starting tick of the state of the BS
    def get_state_starting_tick(self):
        return self.t_starting_state

    # Set the final tick of the state of the BS
    def set_state_final_tick(self, input_tick: int):
        self.t_final_state = input_tick

    # Get the final tick of the state of the BS
    def get_state_final_tick(self):
        return self.t_final_state

    # Set the packet size for the packet of the BS
    def set_new_packet_size(self, packet_size: int):
        self.packet.set_size(packet_size)

    # Get the packet size for the packet of the BS
    def get_new_packet_size(self):
        return self.packet.get_size()

    # Get the next packet generation instant for the packet of the BS
    def get_next_packet_generation_instant_bs(self):
        return self.t_generation_optimization

    # Set the LOS/NLOS condition of the BS
    def set_channel_condition_with_bs(self, is_low_channel_condition_bool: bool):
        self.is_low_channel_condition_with_bs = is_low_channel_condition_bool

    # Get the LOS/NLOS condition of the BS
    def get_channel_condition_with_bs(self):
        return self.is_low_channel_condition_with_bs

    # Method for adding a new packet in the BS buffer
    def add_new_packet(self, current_tick: int):
        """
            Add a new packet in the queue and compute the next generation instant, if the buffer is not full.
            Otherwise, just discard the packet.
        """
        new_packet = cp.copy(self.packet)
        new_packet.set_generation_time(current_tick)
        self.n_generated_packets += 1

        # Check if successfully added to the ul buffer
        if not (self.dl_buffer.add_packet(new_packet)):
            self.n_discarded_packets += 1

        if self.traffic_type in self.traffic_model:  # To avoid errors in the UE initialization
            self.t_generation_optimization += self.traffic_model[self.traffic_type]()
            self.packet.set_generation_time(self.t_generation_optimization)
        else:
            sys.exit('Traffic model ' + str(self.traffic_type) + ' not supported')

        self.packet.set_id(self.packet.get_id() + 1)

    # Get the LOS/NLOS condition of a given UE
    def get_los_condition(self, ue):  # NOTE: This is something the UE may not know in practise,
        # so do not use it to change its behavior during simulation
        return ue.is_in_los

    # Get the bit rate value
    def get_bit_rate_gbits(self):
        return self.bit_rate_gbits

    # Set the bit rate value
    def set_bit_rate_gbits(self, input_bit_rate_gbits: float):
        self.bit_rate_gbits = input_bit_rate_gbits

    # Set the starting tick of the ACK transmission
    def set_start_tx_ack(self, input_start_tx_ack: int):
        self.t_start_tx_ack = input_start_tx_ack

    # Get the starting tick of the ACK transmission
    def get_start_tx_ack(self):
        return self.t_start_tx_ack

    # Set the ending tick of the ACK transmission
    def set_end_tx_ack(self, input_end_tx_ack: int):
        self.t_end_tx_ack = input_end_tx_ack

    # Get the ending tick of the ACK transmission
    def get_end_tx_ack(self):
        return self.t_end_tx_ack

    # Set the number of data successfully received at BS
    def set_n_data_rx(self, input_n_data_rx: int):
        self.n_data_rx = input_n_data_rx

    # Get the number of data successfully received at BS
    def get_n_data_rx(self):
        return self.n_data_rx

    # Set the number of data successfully received for RT traffic at BS
    def set_n_data_rx_rt(self, input_n_data_rx_rt: int):
        self.n_data_rx_rt = input_n_data_rx_rt

    # Get the number of data successfully received for RT traffic at BS
    def get_n_data_rx_rt(self):
        return self.n_data_rx_rt

    # Set the number of data successfully received for CN traffic at BS
    def set_n_data_rx_cn(self, input_n_data_rx_cn: int):
        self.n_data_rx_cn = input_n_data_rx_cn

    # Get the number of data successfully received for CN traffic at BS
    def get_n_data_rx_cn(self):
        return self.n_data_rx_cn

    # Set the number of data successfully received for NRT traffic at BS
    def set_n_data_rx_nrt(self, input_n_data_rx_nrt: int):
        self.n_data_rx_nrt = input_n_data_rx_nrt

    # Get the number of data successfully received for NRT traffic at BS
    def get_n_data_rx_nrt(self):
        return self.n_data_rx_nrt

    # Set the number of data successfully received for FQ traffic at BS
    def set_n_data_rx_fq(self, input_n_data_rx_fq: int):
        self.n_data_rx_fq = input_n_data_rx_fq

    # Get the number of data successfully received for FQ traffic at BS
    def get_n_data_rx_fq(self):
        return self.n_data_rx_fq

    # Update the number of data successfully received at BS
    def update_n_data_rx(self, input_ue_traffic_type: str, packets_received: int, input_enable_print: bool = False):
        self.n_data_rx += packets_received
        if input_enable_print:
            print('The BS has updated n_data_rx to ', self.n_data_rx)

        if input_ue_traffic_type == "traffic_rt":
            self.n_data_rx_rt += packets_received
            if input_enable_print:
                print('The BS has updated n_data_rx_rt to ', self.n_data_rx_rt)
        elif input_ue_traffic_type == "traffic_cn":
            self.n_data_rx_cn += packets_received
            if input_enable_print:
                print('The BS has updated n_data_rx_cn to ', self.n_data_rx_cn)
        elif input_ue_traffic_type == "traffic_nrt":
            self.n_data_rx_nrt += packets_received
            if input_enable_print:
                print('The BS has updated n_data_rx_nrt to ', self.n_data_rx_nrt)
        elif input_ue_traffic_type == "traffic_fq":
            self.n_data_rx_fq += packets_received
            if input_enable_print:
                print('The BS has updated n_data_rx_fq to ', self.n_data_rx_fq)
        else:
            sys.exit("Unknown traffic type when updating n_data_rx at the BS")

    # Set the number of data received from a given UE
    def set_n_data_rx_from_ues(self, input_ue_id: int, input_n_data_rx: int):
        self.n_data_rx_from_ues[f'UE_{input_ue_id}'] = input_n_data_rx

    # Get the number of data received from a given UE
    def get_n_data_rx_from_ues(self, input_ue_id: int):
        return self.n_data_rx_from_ues[f'UE_{input_ue_id}']

    # Update the number of data received from a given UE
    def update_n_data_rx_from_ues(self, input_ue_id: int, packets_received: int):
        self.n_data_rx_from_ues[f'UE_{input_ue_id}'] += packets_received



