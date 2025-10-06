"""
gNodeB class
"""
from multi_hop_industrial_simulator.traffic_models.traffic_model import TrafficModel
from multi_hop_industrial_simulator.network.packet import Packet
from multi_hop_industrial_simulator.network.bs_buffer import BsBuffer
import copy as cp
import sys
import numpy as np

class BS(TrafficModel):
    """ """

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

    def set_coordinates(self, x_input: float, y_input: float, z_input: float):
        """

        Args:
          x_input: float: x coordinate of the BS
          y_input: float: y coordinate of the BS
          z_input: float: z coordinate of the BS

        Set the BS coordinates

        """
        self.x = x_input
        self.y = y_input
        self.z = z_input

    def get_coordinates(self):
        """
        Returns: get the BS coordinates
        """
        return np.array([self.x, self.y, self.z])

    def set_carrier_frequency(self, input_carrier_frequency: float):
        """

        Args:
          input_carrier_frequency: float: frequency of the carrier

        Set the carrier frequency

        """
        self.carrier_frequency = input_carrier_frequency

    def get_carrier_frequency(self):
        """
        Returns: the carrier frequency
        """
        return self.carrier_frequency

    def get_state(self):
        """
        Returns: state of the BS
        """
        return self.state

    def set_state(self, input_state: str):
        """

        Args:
          input_state: str: current state of the BS

        Returns:
            Set the state of the BS

        """
        self.state = input_state

    def update_state_duration(self, input_ticks: int):
        """

        Args:
          input_ticks: int: duration in ticks of the NEW BS state

        Update the state duration of the BS

        """
        self.t_state += input_ticks

    def set_state_duration(self, input_ticks: int):
        """

        Args:
          input_ticks: int: duration in ticks of the current BS state

        Returns:
            duration of the state of the BS

        """
        self.t_state = input_ticks

    def get_state_duration(self):
        """
        Returns: the state duration of the BS
        """
        return self.t_state

    def set_state_starting_tick(self, input_tick: int):
        """

        Args:
          input_tick: int: current tick

        Returns:
            starting tick of the state of the BS

        """
        self.t_starting_state = input_tick

    def get_state_starting_tick(self):
        """
        Returns: the starting tick of the state of the BS
        """
        return self.t_starting_state

    def set_state_final_tick(self, input_tick: int):
        """

        Args:
          input_tick: int: current tick

        Returns:
            final tick of the state of the BS

        """
        self.t_final_state = input_tick

    def get_state_final_tick(self):
        """
        Returns: the final tick of the state of the BS
        """
        return self.t_final_state

    def set_new_packet_size(self, packet_size: int):
        """

        Args:
          packet_size: int: size of the packet (bytes)

        Set the packet size for the packet of the BS

        """
        self.packet.set_size(packet_size)

    def get_new_packet_size(self):
        """
        Returns: packet size of the packet of the BS
        """
        return self.packet.get_size()

    # Get the next packet generation instant for the packet of the BS
    def get_next_packet_generation_instant_bs(self):
        """
        Returns: the next packet generation instant of the BS
        """
        return self.t_generation_optimization

    def set_channel_condition_with_bs(self, is_low_channel_condition_bool: bool):
        """

        Args:
          is_low_channel_condition_bool: bool: True, if the channel condition is low (BS' height is lower than average
          clutter height); False, otherwise

        Returns:
            Set the LOS/NLOS condition of the BS with respect to a UE

        """
        self.is_low_channel_condition_with_bs = is_low_channel_condition_bool

    def get_channel_condition_with_bs(self):
        """
        Returns: the channel condition of the BS
        """
        return self.is_low_channel_condition_with_bs

    # Method for adding a new packet in the BS buffer
    def add_new_packet(self, current_tick: int):
        """Add a new packet in the queue and compute the next generation instant, if the buffer is not full.
            Otherwise, just discard the packet.

        Args:
          current_tick: int: current tick of the simulation

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

    def get_los_condition(self, ue):  # NOTE: This is something the UE may not know in practise,
        """

        Args:
          ue: class

        Returns:
            LOS or NLOS condition of a given UE

        """

        return ue.is_in_los

    def get_bit_rate_gbits(self):
        """
        Returns: the bit rate of the BS
        """
        return self.bit_rate_gbits

    def set_bit_rate_gbits(self, input_bit_rate_gbits: float):
        """

        Args:
          input_bit_rate_gbits: float: input value of the Bit Rate in Gbit/s

        Set the bit rate value of the BS in Gbit/s

        """
        self.bit_rate_gbits = input_bit_rate_gbits

    def set_start_tx_ack(self, input_start_tx_ack: int):
        """

        Args:
          input_start_tx_ack: int: input value of the starting tick of ACK transmission from BS

        Set the starting tick of the ACK transmission

        """
        self.t_start_tx_ack = input_start_tx_ack

    def get_start_tx_ack(self):
        """
        Returns: the starting tick of the ACK transmission from BS
        """
        return self.t_start_tx_ack

    def set_end_tx_ack(self, input_end_tx_ack: int):
        """

        Args:
          input_end_tx_ack: int: input value of the ending tick of ACK transmission from BS

        Set the ending tick of the ACK transmission

        """
        self.t_end_tx_ack = input_end_tx_ack

    def get_end_tx_ack(self):
        """
        Returns: the ending tick of the ACK transmission from BS
        """
        return self.t_end_tx_ack

    def set_n_data_rx(self, input_n_data_rx: int):
        """

        Args:
          input_n_data_rx: int: number of data received at BS

        Set the number of data received at BS

        """
        self.n_data_rx = input_n_data_rx

    def get_n_data_rx(self):
        """
        Returns: the number of data received at BS
        """
        return self.n_data_rx

    def set_n_data_rx_rt(self, input_n_data_rx_rt: int):
        """

        Args:
          input_n_data_rx_rt: int: number of data received at BS for RT traffic

        Set the number of data received at BS for RT traffic

        """
        self.n_data_rx_rt = input_n_data_rx_rt

    def get_n_data_rx_rt(self):
        """
        Returns: Get the number of data received at BS for RT traffic
        """
        return self.n_data_rx_rt

    def set_n_data_rx_cn(self, input_n_data_rx_cn: int):
        """

        Args:
          input_n_data_rx_cn: int: number of data received at BS for CN traffic

        Set the number of data received at BS for CN traffic

        """
        self.n_data_rx_cn = input_n_data_rx_cn

    def get_n_data_rx_cn(self):
        """
        Returns: Get the number of data received at BS for CN traffic
        """
        return self.n_data_rx_cn

    def set_n_data_rx_nrt(self, input_n_data_rx_nrt: int):
        """

        Args:
          input_n_data_rx_nrt: int: number of data received at BS for NRT traffic

        Set the number of data received at BS for NRT traffic

        """
        self.n_data_rx_nrt = input_n_data_rx_nrt

    def get_n_data_rx_nrt(self):
        """
        Returns: Get the number of data received at BS for NRT traffic
        """
        return self.n_data_rx_nrt

    def set_n_data_rx_fq(self, input_n_data_rx_fq: int):
        """

        Args:
          input_n_data_rx_fq: int: number of data received at BS for FQ traffic

        Set the number of data received at BS for FQ traffic

        """
        self.n_data_rx_fq = input_n_data_rx_fq

    def get_n_data_rx_fq(self):
        """
         Returns: the number of data received at BS for FQ traffic
         """
        return self.n_data_rx_fq

    def update_n_data_rx(self, input_ue_traffic_type: str, packets_received: int, input_enable_print: bool = False):
        """

        Args:
          input_ue_traffic_type: str: type of traffic of the UE
          packets_received: int: number of packets received from that UE
          input_enable_print: bool:  (Default value = False)

        Update the number of data successfully received at BS from that UE

        """
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

    def set_n_data_rx_from_ues(self, input_ue_id: int, input_n_data_rx: int):
        """

        Args:
          input_ue_id: int: ID of the UE from which the data is received with success
          input_n_data_rx: int: number of data received at BS from that UE

        Set the number of data received from a given UE

        """
        self.n_data_rx_from_ues[f'UE_{input_ue_id}'] = input_n_data_rx

    def get_n_data_rx_from_ues(self, input_ue_id: int):
        """

        Args:
          input_ue_id: int: ID of the UE from which the BS wants to know how many data it has received

        Returns:
            the number of data received from that UE

        """
        return self.n_data_rx_from_ues[f'UE_{input_ue_id}']

    def update_n_data_rx_from_ues(self, input_ue_id: int, packets_received: int):
        """

        Args:
          input_ue_id: int: ID of the UE from which the BS has received new packets
          packets_received: int: number of packets received from that UE

        Update the number of data received from that UE

        """
        self.n_data_rx_from_ues[f'UE_{input_ue_id}'] += packets_received



