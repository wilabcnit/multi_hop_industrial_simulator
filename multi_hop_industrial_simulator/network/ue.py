import sys
import copy as cp

import numpy as np

from multi_hop_industrial_simulator.traffic_models.traffic_model import TrafficModel
from multi_hop_industrial_simulator.network.ue_buffer import UeBuffer
from multi_hop_industrial_simulator.network.packet import Packet

################################## RL Implementation ##################################
from multi_hop_industrial_simulator.utils.utils_for_tb_ualoha_with_dqn import ttl_reset, select_input_DRL
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs

inputs = read_inputs('inputs.yaml')
DDQN = inputs.get('rl').get('agent').get('DDQN')
_3DQN = inputs.get('rl').get('agent').get('3DQN')
Rainbow_DQN = inputs.get('rl').get('agent').get('Rainbow_DQN')
DDQN_new_state = inputs.get('rl').get('agent').get('New_state')
DRL_input_nodes_number = inputs.get('rl').get('agent').get('DRL_nodes_number')
DRL_input_type_state = inputs.get('rl').get('agent').get('DRL_type_state')
Q_min = inputs.get('rl').get('router').get('max_n_packets_to_be_forwarded_min')
Q_max = inputs.get('rl').get('router').get('max_n_packets_to_be_forwarded_max')
W_min = inputs.get('rl').get('router').get('contention_window_int_min')
W_max = inputs.get('rl').get('router').get('contention_window_int_max')
############################## End RL Implementation ##################################


""""

  User Equipment (UE) class

    This class represents a user equipment (UE) in the network simulation.

    It handles:
    - Initialization of UE parameters and neighbor table
    - Management of UE state and state duration
    - Setting coordinates and transmission parameters (unicast/broadcast mode, transmission address)
    - Packet operations (add/remove packets, update number of transmissions)
    - Determination of link conditions (LoS/NLoS)
    - Reward handling (set and get reward values)

"""

class Ue(TrafficModel):
    """ """
    def __init__(self, params, ue_id: int, traffic_type: str, starting_state: str, t_state_tick: int,
                 input_full_queue: bool):
        super(Ue, self).__init__(input_full_queue=input_full_queue)

        self.packet_info_dict = {
            'ue_id': 0,  # The ID of the recipient of this packet
            'packet_id': 0,  # Packet ID
            'generation_instant': 0,  # Packet generation instant
        }

        self.traffic_model = {
            "traffic_rt":
                super().get_time_periodicity,
            "traffic_cn":
                super().get_time_periodicity,
            "traffic_nrt": super().get_time_periodicity,
            "traffic_fq": super().get_time_periodicity
        }

        self.transceiver_params = {
            "Transmit power": params.get('ue').get('ue_transmit_power_dbm'),  # dBm
            "Number of antennas": params.get('ue').get('ue_number_of_antennas'),
            "Antenna efficiency": params.get('ue').get('ue_antenna_efficiency'),
            "Antenna gain": params.get('ue').get('ue_gain_db'),
            "Noise figure": params.get('ue').get('ue_noise_figure_db'),  # dB
        }

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.params = params
        self.ue_id = ue_id
        self.ul_buffer = UeBuffer(max_buffer_size=self.params.get('ue').get('max_buffer_size'))
        self.packet = Packet()  # Packet that is appended in the queue with periodicity equal to t_generation
        self.state = starting_state
        self.vectState = starting_state
        self.traffic_type = traffic_type
        self.t_generation = 0
        self.t_state = t_state_tick  # Duration of the current state
        self.t_starting_state = 0  # Beginning instant of the current state
        self.t_final_state = 0  # Ending instant of the current state
        self.t_start_data_tx = 0  # Starting tick of data transmission - Used for collision computation
        self.t_end_data_tx = 0  # Ending tick of data transmission - Used for collision computation
        self.t_start_ack_tx = 0  # Starting tick of ACK transmission - Used for collision computation
        self.t_end_ack_tx = 0  # Ending tick of ACK transmission - Used for collision computation
        self.carrier_frequency_ghz = self.params.get('radio').get('carrier_frequency_ghz')
        self.channel = 0
        self.distance_from_bs_m = 0  # UE-BS 3D distance
        self.n_generated_packets = 0  # Number of data generated
        self.n_discarded_packets = 0  # Number of discarded data
        self.is_in_los = 0  # True when the link with the BS is in LoS, False otherwise
        self.is_low_channel_condition_with_bs = 0  # True when the link with the BS is in Low condition, False otherwise
        self.is_in_los_ues = []  # True when the link with the i-th UE is in LoS (i is the list index), False otherwise
        self.is_low_channel_condition_with_ues = []  # True when the link with the i-th UE is Low condition
        # (i is the list index), False otherwise
        self.id_machine = 'False'
        self.prop_delay_to_bs_s = 0  # Propagation delay from UE to BS in seconds
        self.prop_delay_to_bs_tick = 0  # Propagation delay from UE to BS in ticks
        self.prop_delay_to_ues_s = {}  # Dictionary of propagation delays to other UEs in seconds
        self.prop_delay_to_ues_tick = {}  # Dictionary of propagation delays from UE to other UEs in ticks
        self.data_duration_s = 0  # Duration of TX_DATA state in seconds
        self.data_duration_tick = 0  # Duration of TX_DATA state in ticks
        self.collection_on = False
        self.collection_standby = False
        self.optimization = False
        self.nrt_state = None
        self.status_nrt_change = 0
        self.wait_rx_optimization = 0
        self.buffer_dimension_list = list()
        # Set Parameters for MAC protocol
        self.n_data_tx = 0  # Number of data packets transmitted by the node
        self.n_data_collided = 0  # number of data packets collided
        self.n_data_discarded = 0  # number of data discarded
        self.n_data_rx = 0  # Number of data packets successfully transmitted by the UE
        self.n_rx_phy = 0
        self.ue_saved_state = None
        self.bit_rate_gbits = 0  # Gbit/s
        self.energy_consumed = 0
        self.mac_success = False  # True when the UE has transmitted data with success, False otherwise
        # (used to know the type of ACK it will receive (Positive or Negative)
        self.reception_during_bo_bool = False  # True when the UE has received something during BO
        self.reception_during_rreq_rx_bool = False  # True when the UE has received something during RREQ RX
        self.reception_during_wait_bool = False  # True when the UE has received something during WAIT_ACK
        self.max_n_retx_per_packet = None  # Maximum number of retxs per packet
        self.relay = False  # variable set to True when the UE is acting as a relay for another UE
        self.packets_sent = 0  # variable that is set to the number of packets sent, when the UE is a relay
        self.packet_id_success = None  # id of the last packet transmitted with success from the UE
        self.packet_id_received = dict()
        self.packet_generated = dict()
        self.packet_forwarding = list()
        self.packet_id_ack = None
        self.retransmission_of_packets = False
        self.n_tear = 0
        self.ack_rx_during_wait_ack = False
        self.data_rx_during_wait_ack = False
        self.list_data_rx_during_wait_ack = list()
        self.list_data_generated_during_wait_ack = list()
        self.list_data_rx_from_ue_id = list()
        self.dict_data_rx_during_wait_ack = dict()
        self.dict_data_rx_during_bo = dict()
        self.reception_ack_during_wait = False
        self.list_ack_sent_from_bs = list()
        self.dict_ack_sent_from_ue = dict()
        self.buffer_packet_sent = list()
        self.packet_generation_instant = -1
        self.latency_ue = list()
        self.check_last_round = False
        self.single_ue_output_dict = {}
        self.discarded_packets = 0

        #############################RL Implementation#############################
        self.first_bo_entry = False
        self.previous_state = None # 0: IDLE; 1: BO; 2: TX_ACK; 3: TX_DATA; 4: WAIT_ACK;

        self.last_action = None
        self.obs = None
        self.temp_obs = None
        self.old_state = None
        self.reward = None
        self.simulations_reward = None
        self.model = None
        self.replay_buffer = None
        self.neighbour_table = None
        self.epsilon = None
        self.best_weights = None
        self.best_score = None
        self.unicast_rx_address = None
        self.copy_unicast_rx_address = None
        self.unicast_rx_index = None
        self.broadcast_bool = False
        self.action_list = None
        self.actions_per_simulation = None
        self.success_action_list = None
        self.success_actions_per_simulation = None
        self.tx_broad_list = list()
        self.data_discard_bool = False
        self.first_entry = False
        self.copy_buffer_packet_list = None
        self.new_action_bool = False
        self.action_packet_id = None
        self.forward_in_wait_ack = False
        self.forward_in_bo = False
        self.packet_forward = False
        self.designated_rx = False
        self.check_last_round = False
        # self.packet_id_rx = list()
        self.saved_coordinates = None
        self.starting_coordinates = None
        self.target_model = None
        self.packets_to_be_removed = dict()
        self.DRL_state = None
        self.actions_since_last_ttl_reset = list()
        self.Rainbow_DQN = False
        self.packets_discarded_full_queue = 0
        self.packets_discarded_max_rtx = 0

        ##########################End RL Implementation############################

        ########################## Multihop implementation ########################
        self.multihop_bool = False
        self.next_action = None
        self.ues_colliding_at_ue = list()
        self.ues_interfering_at_ue = list()
        self.ues_colliding_at_bs = list()
        self.packet_number_colliding = list()
        self.ack_rx_with_success = False
        self.data_rx_at_ue_ue_id_list = list()
        ########################## End Multihop implementation ########################
        self.ticks_in_BO = list()
        self.ticks_in_TX_DATA = list()
        self.ticks_in_TX_ACK = list()
        self.ticks_in_WAIT_ACK = list()
        self.ticks_in_IDLE = list()
        self.forced_broadcast_actions_counter = 0
        # Buffer transmission
        # self.n_ack_rx_simultaneously = None  # number of ACKs received from a UE simultaneously, it happens when there
        # was a buffer TX

        self.end_data_tx = 0

        self.n_forwarding = 0
        self.n_interfering = list()

        ################################ RL Q and W ####################################
        self.Q_and_W_enabled = False
        self.Q_and_W_current_state = list()
        self.Q_and_W_previous_state = list()
        self.Q_and_W_saved_state_Q = list()
        self.Q_and_W_saved_state_W = list()
        self.Q_and_W_latencies = list()
        self.Q_and_W_pcks_tx_with_success = list()
        self.Q_and_W_rtx_pcks_tx_with_success = list()
        self.Q_and_W_buffer_length = 0
        self.Q_and_W_contention_window = 0
        self.Q_and_W_previous_latency = 0
        self.Q_and_W_tx_data_counter = 0
        self.Q_and_W_acks_rx_per_step_counter = 0
        self.Q_and_W_pcks_tx_per_step_counter = 0
        self.Q_and_W_dropped_pcks_per_step_counter = 0
        self.Q_and_W_max_latency = 2.4 * 1e-6
        self.Q_and_W_forbidden_action = False
        self.Q_and_W_last_action = None # -1 means not to append to the replay buffer


        ################################## Only W RL ####################################
        self.W_enabled = False
        self.W_model = None
        self.W_replay_buffer = None
        self.W_target_model = None
        self.W_current_state = list()
        self.W_previous_state = list()
        self.W_pcks_tx_with_success = list()
        self.W_rtx_pcks_tx_with_success = list()
        self.W_dropped_pcks_per_step_counter = 0
        self.W_not_added_pcks_per_step_counter = 0
        self.W_forbidden_action = False
        self.W_action_list = None
        self.W_reward = None
        self.W_simulations_reward = None
        self.W_last_action = None # -1 means not to append to the replay buffer
        self.W_start_tick_params_window = 0

        ################################## Only W RL ####################################
        self.Q_enabled = False
        self.Q_model = None
        self.Q_replay_buffer = None
        self.Q_target_model = None
        self.Q_current_state = list()
        self.Q_previous_state = list()
        self.Q_pcks_tx_with_success = list()
        self.Q_rtx_pcks_tx_with_success = list()
        self.Q_buffer_utilization = list()
        self.Q_dropped_pcks_per_step_counter = 0
        self.Q_not_added_pcks_per_step_counter = 0
        self.Q_forbidden_action = False
        self.Q_action_list = None
        self.Q_reward = None
        self.Q_simulations_reward = None
        self.Q_last_action = None # -1 means not to append to the replay buffer
        self.Q_start_tick_params_window = 0

        ############## AODV ################
        self.end_control_plane = False
        self.rreq_dict = dict()
        self.rreply_received = False
        self.forwarding_rreq = False
        self.end_of_bo_rreq = None
        self.rreq_received = False
        self.unicast_address = None
        self.new_control_plane = False
        self.time_shift = 0 # shift in time for RREQ forward
        self.n_rreq_forwarded = 0 # number of RREQ to be forwarded
        self.relay_rreq = dict()
        self.source_rreq = list()
        self.rreq_tx = 0
        self.dict_rreq_rx = dict()
        self.forward_rreply = False
        self.new_rreply_dest = list()
        self.relay_list_for_rreply = dict()
        self.current_buffer_size = 0

    def set_packet_info(self, packet_info: dict):
        """

        Args:
          packet_info: dict: dictionary containing the information about the packet

        Set the packet info dictionary

        """
        self.packet_info_dict.update(packet_info)

    # Add a new packet in the queue and compute the next generation instant if the buffer is not full
    def add_new_packet(self, current_tick: int, input_enable_print: bool = False,
                       input_data_to_be_forwarded_bool: bool = None, input_packet_size_bytes: int = None,
                       input_simulation_tick_duration: int = None, data_rx_from_ue: int = None,
                       packet_id_rx_from_ue: int = None, packet_generated_by_ue: int = None,
                       packet_id_generator: int = None, packet_hop_count: int = None, packet_address: int = None, generation_time: int = None):
        """Add a new packet in the queue and compute the next generation instant, if the buffer is not full.
            Otherwise, just discard the packet.

        Args:
          current_tick: int: current tick of the simulation
          input_enable_print: bool:  (Default value = False)
          input_data_to_be_forwarded_bool: bool:  True, if the data is forwarded; False, otherwise. Default value = None
          input_packet_size_bytes: int: size in bytes of the packet to be added in the queue (Default value = None)
          input_simulation_tick_duration: int: duration of the simulation tick (Default value = None)
          data_rx_from_ue: int: ID of the UE from which DATA has been received (Default value = None)
          packet_id_rx_from_ue: int: ID of the packet received (Default value = None)
          packet_generated_by_ue: int: ID of the source UE of the DATA (Default value = None)
          packet_id_generator: int: original ID of the packet generated by the source UE (Default value = None)
          packet_hop_count: int: current hop of that packet (Default value = None)
          packet_address: int: address of the packet destination: If Broadcast = -1; else: UE_ID or BS if Unicast
          generation_time: int: instant of generation of the packet (Default value = None)


        """
        """
            For NRT Nodes, set the right payload to be added depending on NRT states
        """
        if self.traffic_type == "traffic_nrt":
            if self.nrt_state == 'collection_on':
                self.packet.packet_size = self.packet.get_size_on()
            elif self.nrt_state == 'collection_standby':
                self.packet.packet_size = self.packet.get_size_standby()
            elif self.nrt_state == 'optimization':
                print("UE enters in optimization and has to receive data from the BS -> WAIT FOR DATA.")
            print("UE: ", self.ue_id, " is in ", self.nrt_state)

        new_packet = cp.copy(self.packet)
        new_packet.set_generation_time(current_tick)
        new_packet.set_max_n_retx(input_max_n_retx=self.max_n_retx_per_packet)
        new_packet.set_ue_id(input_ue_id=self.ue_id)
        if input_data_to_be_forwarded_bool is not None:
            new_packet.set_data_to_be_forwarded_bool(input_data_to_be_forwarded_bool=input_data_to_be_forwarded_bool)
            data_duration_s = round((input_packet_size_bytes * 8 * 1e-9) /
                                    self.bit_rate_gbits, 11)
            data_duration_tick = int(round(data_duration_s / input_simulation_tick_duration))
            new_packet.set_packet_duration_s(input_packet_duration_s=data_duration_s)
            new_packet.set_packet_duration_tick(input_packet_duration_tick=data_duration_tick)
            new_packet.set_data_rx_from_ue(data_rx_from_ue)
            new_packet.set_packet_id_rx_from_ue(packet_id_rx_from_ue)

            if self.multihop_bool:
                new_packet.set_generated_by_ue(packet_generated_by_ue)
                new_packet.set_packet_id_generator(packet_id_generator)
                new_packet.set_hop_count(packet_hop_count)
                new_packet.set_generated_by_ue_time_instant_tick(generation_time)

            self.packet_generated[self.packet.get_id()] = True
        else:
            new_packet.set_packet_duration_s(input_packet_duration_s=self.data_duration_s)
            new_packet.set_packet_duration_tick(input_packet_duration_tick=self.data_duration_tick)
            self.packet_generated[self.packet.get_id()] = False
            if self.multihop_bool:
                new_packet.set_generated_by_ue(self.ue_id)
                new_packet.set_packet_id_generator(self.packet.get_id())
                new_packet.set_generated_by_ue_time_instant_tick(generation_time)

        if self.multihop_bool:
            new_packet.set_address(packet_address)

        self.n_generated_packets += 1

        # Check if successfully added to the ul buffer
        if not (self.ul_buffer.add_packet(new_packet)):
            self.n_discarded_packets += 1

        self.buffer_dimension_list.append(self.ul_buffer.buffer_size)

        if generation_time is None:
            new_packet.set_generated_by_ue_time_instant_tick(current_tick)
        """
            Compute the right t_generation depending on the current NRT status (collection or optimization); 
            While for RT and Camera Nodes (all periodic traffic) simply use "get time periodicity" function
        """
        if self.traffic_type in self.traffic_model:  # To avoid errors in the UE initialization
            if self.traffic_type == 'traffic_nrt':
                # ATTENTION: If the next t_generation will be in a NEW NRT state ->
                # next t_generation = instant when the new state begins + new traffic model
                if self.nrt_state == 'collection_on' or self.nrt_state == 'collection_standby':
                    self.t_generation += self.traffic_model[self.traffic_type]()
                    if self.t_generation >= self.status_nrt_change:
                        if self.nrt_state == 'collection_on':
                            self.t_generation = self.status_nrt_change + \
                                                self.traffic_model[self.traffic_type]()
                        elif self.nrt_state == 'collection_standby':
                            # After standby, the node goes in optimization and wait for the packet from the BS
                            self.t_generation = self.status_nrt_change + self.wait_rx_optimization + \
                                                self.traffic_model[self.traffic_type]()
            else:
                self.t_generation += self.traffic_model[self.traffic_type]()

            self.packet.set_generation_time(self.t_generation)
        else:
            sys.exit('Traffic model ' + str(self.traffic_type) + ' not supported when adding a new packet')

        if input_enable_print:
            print('New packet with ID ', self.packet.get_id(), ', generated by UE',
                  self.get_ue_id(), ' at tick: ', current_tick)

        # Set ID of the new packet
        self.packet.set_id(self.packet.get_id() + 1)

        # Set the sequence number for all the packets within the buffer of the UE -> the first packet must have the
        # highest sequence number -> in this way the BS knows for how long it has to stay in RX before ACK TX

        index = 0
        for packet in self.ul_buffer.buffer_packet_list:
            packet.set_sequence_number(input_sequence_number=len(self.ul_buffer.buffer_packet_list) - index)
            # print("Packet ", packet.packet_id, " has sequence number: ", packet.get_sequence_number())
            index += 1

    # Gert the list of buffer dimensions
    def get_ue_buffer_list(self):
        return self.buffer_dimension_list

    # Remove a packet from the queue
    def remove_packet(self, packet_id: int = None, input_enable_print: bool = False):
        """

        Args:
          packet_id: int: ID of the packet to be removed (Default value = None)
          input_enable_print: bool:  (Default value = False)

        Remove the packet from the queue

        """
        # Either remove only the first packet or all those having being received from other UEs
        if packet_id is not None:

            for packet in self.get_updated_packet_list():
                if (packet.get_id() == packet_id):
                    if self.Q_and_W_enabled:
                        self.Q_and_W_pcks_tx_with_success[-1] += 1
                        self.Q_and_W_rtx_pcks_tx_with_success[-1].append(packet.get_num_tx() - 1)
                    if self.Q_enabled:
                        self.Q_pcks_tx_with_success[-1] += 1
                        self.Q_rtx_pcks_tx_with_success[-1].append(packet.get_num_tx() - 1)
                    if self.W_enabled:
                        self.W_pcks_tx_with_success[-1] += 1
                        self.W_rtx_pcks_tx_with_success[-1].append(packet.get_num_tx() - 1)

            self.ul_buffer.remove_data(packet_id=packet_id)

            if input_enable_print:
                print('UE ', self.get_ue_id(), ' has removed packet with ID ', packet_id)
        else:
            for packet in self.get_updated_packet_list():
                if (packet.get_data_to_be_forwarded_bool() or
                        packet.get_id() == self.ul_buffer.get_first_packet().get_id()):

                    if self.Q_and_W_enabled:
                        self.Q_and_W_pcks_tx_with_success[-1] += 1
                        self.Q_and_W_rtx_pcks_tx_with_success[-1].append(packet.get_num_tx() - 1)
                    if self.Q_enabled:
                        self.Q_pcks_tx_with_success[-1] += 1
                        self.Q_rtx_pcks_tx_with_success[-1].append(packet.get_num_tx() - 1)
                    if self.W_enabled:
                        self.W_pcks_tx_with_success[-1] += 1
                        self.W_rtx_pcks_tx_with_success[-1].append(packet.get_num_tx() - 1)

                    self.ul_buffer.remove_data(packet_id=packet.get_id())

                    if input_enable_print:
                        print('UE ', self.get_ue_id(), ' has removed packet with ID ', packet.get_id())

    def set_coordinates(self, x_input, y_input, z_input):
        """

        Args:
          x_input: x coordinate of UE
          y_input: y coordinate of UE
          z_input: z coordinate of UE

        Set the 3D coordinates of the UE

        """
        self.x = x_input
        self.y = y_input
        self.z = z_input

    def get_coordinates(self):
        """
        Returns the 3D coordinates of the UE
        """
        return np.array([self.x, self.y, self.z])

    def get_next_packet_generation_instant(self):
        """
        Returns: the next packet generation instant in ticks
        """
        return self.t_generation

    def get_state(self):
        """
        Returns: the current state of the UE
        """
        return self.state

    def set_state(self, input_state: str):
        """

        Args:
          input_state: str: current state of the UE

        Set the current state of the UE

        """
        self.state = input_state

    def set_start_data_tx(self, input_start_data_tx: int):
        """

        Args:
          input_start_data_tx: int: input value of the tick corresponding to the instant when the DATA transmission starts

        Set the starting tick of data transmission

        """
        self.t_start_data_tx = input_start_data_tx

    def get_start_data_tx(self):
        """
        Returns: starting tick of data transmission
        """
        return self.t_start_data_tx

    def set_start_ack_tx(self, input_start_ack_tx: int):
        """

        Args:
          input_start_ack_tx: int: input value of the tick corresponding to the instant when the ACK transmission starts

        Set the starting tick of ACK transmission

        """
        self.t_start_ack_tx = input_start_ack_tx

    def get_start_ack_tx(self):
        """
        Returns: starting tick of ACK transmission
        """
        return self.t_start_ack_tx

    def set_end_ack_tx(self, input_end_ack_tx: int):
        """

        Args:
          input_end_ack_tx: int: input value of the tick corresponding to the instant when the ACK transmission ends

        Set the ending tick of ACK transmission

        """
        self.t_end_ack_tx = input_end_ack_tx

    def get_end_ack_tx(self):
        """
        Returns: ending tick of ACK transmission
        """
        return self.t_end_ack_tx

    def set_end_data_tx(self, input_end_data_tx: int):
        """

        Args:
          input_end_data_tx: int: input value corresponding to the instant when DATA transmission ends

        Set the ending tick of DATA transmission

        """
        self.t_end_data_tx = input_end_data_tx

    def get_end_data_tx(self):
        """
         Returns: ending tick of DATA transmission
        """
        return self.t_end_data_tx

    def set_state_duration(self, input_ticks: int):
        """

        Args:
          input_ticks: int: input value corresponding to the duration in ticks of the next state of the UE

        Set the current state duration in ticks of the UE

        """
        self.t_state = input_ticks

    def update_state_duration(self, input_ticks: int):
        """

        Args:
          input_ticks: int: input value corresponding to the duration in ticks of the next state of the UE

        Update the current state duration

        """
        self.t_state += input_ticks

    def get_state_duration(self):
        """
         Returns: get the current state duration in ticks
         """
        return self.t_state

    def set_state_starting_tick(self, input_tick: int):
        """

        Args:
          input_tick: int: input value in ticks corresponding to the starting instant of the UE's state

        Set the starting tick of the current state

        """
        self.t_starting_state = input_tick

    def get_state_starting_tick(self):
        """
         Returns: the starting tick of the current state
         """
        return self.t_starting_state

    def set_state_final_tick(self, input_tick: int):
        """

        Args:
          input_tick: int: input value in ticks corresponding to the final instant of the UE's state

        Set the final tick of the current state

        """
        self.t_final_state = input_tick

    def get_state_final_tick(self):
        """
         Returns: the final tick of the current state
         """
        return self.t_final_state

    def get_ue_id(self):
        """
         Returns: the ID of the UE
         """
        return self.ue_id

    def set_ue_id(self, ue_id):
        """

        Args:
          ue_id: ID of the current UE

        Set the UE ID

        """
        self.ue_id = ue_id

    def get_bit_rate_gbits(self):
        """
         Returns: the bit rate of the current UE in Gbit/s
         """
        return self.bit_rate_gbits

    def set_bit_rate_gbits(self, input_bit_rate_gbits: float):
        """

        Args:
          input_bit_rate_gbits: float: bit rate of the current UE in Gbit/s

        Set the bit rate of the UE

        """
        self.bit_rate_gbits = input_bit_rate_gbits

    def get_data_duration_s(self):
        """
         Returns: the duration in seconds of the DATA packet
         """
        return self.data_duration_s

    def set_data_duration_s(self, input_data_duration_s: float):
        """

        Args:
          input_data_duration_s: float: input value of the DATA packet duration in seconds

        Set the duration in seconds of the DATA packet

        """
        self.data_duration_s = input_data_duration_s

    def get_data_duration_tick(self):
        """
        Returns: the duration in ticks of the DATA packet
        """
        return self.data_duration_tick

    def set_data_duration_tick(self, input_data_duration_tick: float):
        """

        Args:
          input_data_duration_tick: float: input value of the DATA packet duration in ticks

        Set the duration in ticks of the DATA packet

        """
        self.data_duration_tick = input_data_duration_tick

    """
        Methods referring to the next packet that will be added in the queue at tick "t_generation" 
    """

    def set_new_packet_size(self, packet_size: int):
        """

        Args:
            packet_size (int): The size of the packet in bytes.

        Set the size of the current packet.

        """
        self.packet.set_size(packet_size)

    def get_new_packet_size(self) -> int:
        """
        Get the size of the current packet.

        Returns:
            int: The size of the packet in bytes.
        """
        return self.packet.get_size()

    def set_new_packet_data_to_be_sent(self, input_data_to_be_sent: int):
        """
        Args:
            input_data_to_be_sent (int): The amount of data (in bytes) to be sent in the packet.

        Set the amount of data in the packet to be sent.

        """
        self.packet.set_data_to_be_sent(input_data_to_be_sent)

    def get_new_packet_data_to_be_sent(self) -> int:
        """
        Get the amount of data in the packet to be sent.

        Returns:
            int: The amount of data (in bytes) to be sent in the packet.
        """
        return self.packet.get_data_to_be_sent()

    def set_new_packet_size_on(self, packet_size: int):
        """
        Args:
            packet_size (int): The packet size (in bytes) when in "on" mode.

        Set the packet size in "on" mode.
        """
        self.packet.set_size_on(packet_size)

    def get_new_packet_size_on(self) -> int:
        """
        Get the packet size in "on" mode.

        Returns:
            int: The packet size (in bytes) when in "on" mode.
        """
        return self.packet.get_size_on()

    def get_new_packet_data_to_be_sent_on(self) -> int:
        """
        Get the amount of data to be sent in "on" mode.

        Returns:
            int: The amount of data (in bytes) to be sent when in "on" mode.
        """
        return self.packet.get_data_to_be_sent()

    def set_new_packet_size_standby(self, packet_size: int):
        """

        Args:
            packet_size (int): The packet size (in bytes) when in standby mode.

        Set the packet size in standby mode.
        """
        self.packet.set_size_standby(packet_size)

    def get_new_packet_size_standby(self) -> int:
        """
        Get the packet size in standby mode.

        Returns:
            int: The packet size (in bytes) when in standby mode.
        """
        return self.packet.get_size_standby()

    """
        End of these methods
    """
    def get_last_packet_id(self):
        """
        Get the ID of the last packet added to the queue.

        Returns:
            int: The ID of the last packet, starting from 0.
        """
        return self.packet.get_id() - 1

    def get_updated_packet_list(self):
        """
        Get the list of packets currently in the uplink buffer.

        Returns:
            list: A list of packet objects in the UE's uplink buffer.
        """
        return self.ul_buffer.get_packet_list()

    def get_packet_list_size(self):
        """
        Get the number of packets currently in the queue.

        Returns:
            int: The number of packets stored in the uplink buffer.
        """
        return self.ul_buffer.get_n_packets()

    def get_ul_buffer(self):
        """
        Get the uplink buffer object of the UE.

        Returns:
            object: The uplink buffer instance associated with the UE.
        """
        return self.ul_buffer

    def set_carrier_frequency(self, input_carrier_frequency_ghz: float):
        """

        Args:
            input_carrier_frequency_ghz (float): Carrier frequency in GHz.

        Set the carrier frequency for the UE.

        """
        self.carrier_frequency_ghz = input_carrier_frequency_ghz

    def get_carrier_frequency_ghz(self):
        """
        Get the carrier frequency of the UE.

        Returns:
            float: The carrier frequency in GHz.
        """
        return self.carrier_frequency_ghz

    def set_channel(self, input_channel: int):
        """

        Args:
            input_channel (int): The channel number to assign.

        Set the channel index for the UE.

        """
        self.channel = input_channel

    def get_channel(self):
        """
        Get the channel index used by the UE.

        Returns:
            int: The channel number assigned to the UE.
        """
        return self.channel

    def set_ue_saved_state(self, input_ue_saved_state: str):
        """

        Args:
            input_ue_saved_state (str): The state to save for the UE.

        Set the saved state string for the UE.

        """
        self.ue_saved_state = input_ue_saved_state

    def get_ue_saved_state(self):
        """
        Get the saved state of the UE.

        Returns:
            str: The string representing the UE's saved state.
        """
        return self.ue_saved_state

    def get_current_packet_transmission_number(self):  # Current packet is the one that the UE is transmitting
        """
        Get the transmission counter of the first packet in the queue.

        Returns:
            int: The number of times the current packet has been transmitted.
        """
        return self.ul_buffer.get_first_packet().get_transmission_counter()

    def increment_current_packet_transmission_number(self):
        """
        Increment the transmission counter of the first packet in the queue.

        """
        self.ul_buffer.get_first_packet().increment_transmission_counter()

    def get_los_condition(self):  # NOTE: This is something the UE may not know in practise,
        """
        Get the line-of-sight (LOS) condition with the BS.

        Returns:
            bool: True if UE is in LOS with the BS, False otherwise.
        """
        # so do not use it to change its behavior during simulation
        return self.is_in_los

    def set_los_condition(self, is_in_los: bool):
        """

        Args:
            is_in_los (bool): True if LOS condition exists, False otherwise.

        Set the LOS condition with the BS.

        """
        self.is_in_los = is_in_los

    def set_channel_condition_with_bs(self, is_low_channel_condition_bool: bool):
        """

        Args:
            is_low_channel_condition_bool (bool): True if the UE and the BS are below the average clutter height, False otherwise.

        Set the channel condition between the UE and the BS.

        """
        self.is_low_channel_condition_with_bs = is_low_channel_condition_bool

    def get_channel_condition_with_bs(self):
        """
        Get the channel condition between the UE and the BS.

        Returns:
            bool: True if the channel condition is low, False otherwise.
        """
        return self.is_low_channel_condition_with_bs

    def set_los_condition_ue_ue(self, is_in_los: bool):
        """

        Args:
            is_in_los (bool): True if LOS condition exists, False otherwise.

        Set the LOS condition between this UE and another UE.

        """
        self.is_in_los_ues.append(is_in_los)

    def set_channel_condition_with_ue(self, is_low_channel_condition_bool: bool):
        """

        Args:
            is_low_channel_condition_bool (bool): True if the UEs are below the average clutter height, False otherwise.

        Set the channel condition between this UE and another UE.

        """
        self.is_low_channel_condition_with_ues.append(is_low_channel_condition_bool)

    def get_channel_condition_with_ue(self, ue_index: int):
        """
        Get the channel condition between this UE and another UE.

        Args:
            ue_index (int): The index of the other UE.

        Returns:
            bool: True if the channel condition is low, False otherwise.
        """
        return self.is_low_channel_condition_with_ues[ue_index]

    def get_distance_from_bs_m(self):  # NOTE: This is something the UE may not know in practise,
        """
        Get the distance from the UE to the BS in meters.

        Returns:
            float: The UE–BS distance in meters.
        """
        # so do not use it to change its behavior during simulation
        return self.distance_from_bs_m

    def set_distance_from_bs(self, ue_bs_distance_m: float):
        """

        Args:
            ue_bs_distance_m (float): Distance between UE and BS in meters.

        Set the distance from the UE to the BS.

        """
        self.distance_from_bs_m = ue_bs_distance_m

    def update_num_tx(self, input_packet_id: int = None, input_enable_print: bool = False):
        """
        Update the number of transmissions for the first packet in the buffer and
        for any packets that need to be forwarded.

        Args:
            input_packet_id (int, optional): The ID of the packet whose transmission
                counter should be updated. If None, updates all packets. Default is None.
            input_enable_print (bool, optional): Whether to print a message when updating
                the transmission counter. Default is False.

        Returns:
            None
        """
        if input_packet_id is not None:
            packet = self.ul_buffer.get_packet_by_id(packet_id=input_packet_id)
            current_num_tx = packet.get_num_tx()
            packet.set_num_tx(input_num_tx=current_num_tx + 1, input_enable_print=input_enable_print)
        else:
            for packet in self.get_updated_packet_list():
                current_num_tx = packet.get_num_tx()
                packet.set_num_tx(input_num_tx=current_num_tx + 1, input_enable_print=input_enable_print)

    def check_num_tx(self):
        """
        Check whether there are packets that have reached the maximum number of
        retransmissions and remove them.

        Returns:
            bool: True if there is still data to transmit (i.e., some packets have not
            reached their retransmission limit), False otherwise.
        """
        data_to_transmit = False
        packets_list = list()
        for packet in self.get_updated_packet_list():
            if packet.get_num_tx() <= packet.get_max_n_retx() + 1:
                data_to_transmit = True
            else:
                if self.Q_and_W_enabled:
                    self.Q_and_W_dropped_pcks_per_step_counter += 1
                if self.Q_enabled:
                    self.Q_dropped_pcks_per_step_counter += 1
                if self.W_enabled:
                    self.W_dropped_pcks_per_step_counter += 1
                if packet.get_data_to_be_forwarded_bool():
                    packets_list.append(packet)
                    packet.set_packet_discarded(True)
                else:
                    packets_list.append(packet)

        for packet in packets_list:
            data_to_transmit = False
            if packet.get_data_to_be_forwarded_bool() is False:
                self.update_n_data_discarded()
            self.remove_packet(packet_id=packet.get_id())
            self.discarded_packets += 1
            self.packets_discarded_max_rtx += 1
            self.packets_sent -= 1

        return data_to_transmit

    def update_n_data_rx(self, input_tick: int, input_enable_print: bool = False):
        """

        Args:
            input_tick (int): The current simulation tick.
            input_enable_print (bool, optional): Whether to print a debug message
                about the update. Default is False.

        Update the number of successfully received data packets by the UE.

        """
        self.n_data_rx += 1
        if input_enable_print:
            print('UE ', self.get_ue_id(), ' has updated n_data_rx to ',
                  self.get_n_data_rx(), 'at t = ', input_tick, ' because it has transmitted packet ',
                  self.packet.get_id(), ' with success')

    def set_n_data_tx(self, input_n_data_tx: int):
        """

        Args:
            input_n_data_tx (int): The total number of transmitted packets.

        Set the total number of data packets transmitted by the UE.

        """
        self.n_data_tx = input_n_data_tx

    def update_n_data_tx(self, input_enable_print: bool = False):
        """

        Args:
            input_enable_print (bool, optional): Whether to print a debug message
                about the update. Default is False.

        Update the number of transmitted data packets by the UE.

        """
        for packet in self.get_updated_packet_list():
            if self.multihop_bool:
                if packet.get_retransmission_packets() is False and packet.get_data_to_be_forwarded_bool() is False:
                    self.n_data_tx += 1
                    if input_enable_print:
                        print('UE ', self.get_ue_id(), ' has updated n_data_tx to ',
                              self.n_data_tx)
                    break
            else:
                if packet.get_retransmission_packets() is False and packet.get_data_unicast() is False:
                    if packet.get_id() == self.ul_buffer.get_first_packet().get_id() or packet.get_data_to_be_forwarded_bool() is False:
                        self.n_data_tx += 1
                        if input_enable_print:
                            print('UE ', self.get_ue_id(), ' has updated n_data_tx to ',
                                  self.n_data_tx)
                    break

    def get_n_data_tx(self):
        """
        Get the total number of data packets transmitted by the UE.

        Returns:
            int: The number of transmitted data packets.
        """
        return self.n_data_tx

    def set_n_data_rx(self, input_n_data_rx: int):
        """

        Args:
            input_n_data_rx (int): The total number of received packets.

        Set the total number of data packets received by the UE.

        """
        self.n_data_rx = input_n_data_rx

    def get_n_data_rx(self):
        """
        Get the total number of data packets received by the UE.

        Returns:
            int: The number of received data packets.
        """
        return self.n_data_rx

    def set_n_data_discarded(self, input_n_data_discarded: int):
        """

        Args:
            input_n_data_discarded (int): The number of discarded packets.

        Set the total number of data packets discarded by the UE.

        """
        self.n_data_discarded = input_n_data_discarded

    def update_n_data_discarded(self):
        """
        Increment the number of discarded data packets by one.

        """
        self.n_data_discarded += 1

    def get_n_data_discarded(self):
        """
        Get the total number of data packets discarded by the UE.

        Returns:
            int: The number of discarded data packets.
        """
        return self.n_data_discarded

    def is_there_a_new_data(self, input_current_tick: int, max_n_packets_to_be_forwarded: int):
        """
        Check whether a new data packet should be generated and added to the queue.

        Args:
            input_current_tick (int): The current simulation tick.
            max_n_packets_to_be_forwarded (int): The maximum number of packets
                that can be forwarded.

        Returns:
            bool: True if a new packet was generated or the buffer is not empty,
            False otherwise.
        """
        if self.traffic_type == 'traffic_fq':
            if len(self.ul_buffer.buffer_packet_list) < max_n_packets_to_be_forwarded + 1:
                if self.multihop_bool:
                    self.add_new_packet(current_tick=input_current_tick, input_enable_print=False,
                                        packet_address=(self.get_unicast_rx_address() if self.get_broadcast_bool() is False else "-1"))
                else:
                    self.add_new_packet(current_tick=input_current_tick, input_enable_print=False)
                self.packet_generation_instant = input_current_tick
                return True
        elif self.ul_buffer.is_there_any_data():
            return True
        else:
            return False

    def get_n_packets(self):
        """
        Get the number of packets in the uplink buffer.

        Returns:
            int: The number of packets in the buffer.
        """
        return self.ul_buffer.get_n_packets()

    def get_traffic_type(self):
        """
        Get the traffic type configured for the UE.

        Returns:
            str: The traffic type (e.g., "traffic_fq").
        """
        return self.traffic_type

    def set_t_generation(self, input_t_generation: int):
        """

        Args:
            input_t_generation (int): The tick at which the next packet will be generated.

        Set the time instant for the next packet generation.

        """
        self.t_generation = input_t_generation

    def set_packet_id(self, input_packet_id: int or str):
        """

        Args:
            input_packet_id (int or str): The packet ID, either integer or string.

        Set the ID of the current packet.

        """
        self.packet.set_id(input_packet_id=input_packet_id)

    def set_reception_during_rreq_rx_bool(self, input_rreq_rx_bool: bool):
        """

        Args:
            input_rreq_rx_bool (bool): True if reception is occurring during RREQ, False otherwise.

        Set whether the UE is receiving during RREQ reception.

        """
        self.reception_during_rreq_rx_bool = input_rreq_rx_bool

    def set_reception_during_bo_bool(self, input_data_rx_bool: bool):
        """

        Args:
            input_data_rx_bool (bool): True if reception is occurring during BO, False otherwise.

        Set whether the UE is receiving during backoff (BO).

        """
        self.reception_during_bo_bool = input_data_rx_bool

    def set_reception_during_wait_bool(self, input_data_rx_bool: bool):
        """

        Args:
            input_data_rx_bool (bool): True if reception is occurring during WAIT, False otherwise.

        Set whether the UE is receiving during the WAIT state.

        """
        self.reception_during_wait_bool = input_data_rx_bool

    def get_reception_during_bo_bool(self):
        """
        Get whether the UE is receiving during backoff (BO).

        Returns:
            bool: True if reception is occurring during BO, False otherwise.
        """
        return self.reception_during_bo_bool

    def get_reception_during_rreq_rx_bool(self):
        """
        Get whether the UE is receiving during RREQ reception.

        Returns:
            bool: True if reception is occurring during RREQ, False otherwise.
        """
        return self.reception_during_rreq_rx_bool

    def get_reception_during_wait_bool(self):
        """
        Get whether the UE is receiving during WAIT state.

        Returns:
            bool: True if reception is occurring during WAIT, False otherwise.
        """
        return self.reception_during_wait_bool

    def set_prop_delay_to_bs_s(self, input_prop_delay_to_bs_s: float):
        """

        Args:
            input_prop_delay_to_bs_s (float): Propagation delay in seconds.

        Set the propagation delay to the BS in seconds.

        """
        self.prop_delay_to_bs_s = input_prop_delay_to_bs_s

    def get_prop_delay_to_bs_s(self):
        """
        Get the propagation delay to the BS in seconds.

        Returns:
            float: Propagation delay in seconds.
        """
        return self.prop_delay_to_bs_s

    def set_prop_delay_to_bs_tick(self, input_prop_delay_to_bs_tick: int):
        """
        Set the propagation delay to the BS in simulation ticks.

        Args:
            input_prop_delay_to_bs_tick (int): Propagation delay in ticks.

        Returns:
            None
        """
        self.prop_delay_to_bs_tick = input_prop_delay_to_bs_tick

    def get_prop_delay_to_bs_tick(self):
        """
        Get the propagation delay to the BS in simulation ticks.

        Returns:
            int: Propagation delay in ticks.
        """
        return self.prop_delay_to_bs_tick

    def add_prop_delay_to_ue_s(self, input_ue_id: int, input_prop_delay_to_ue_s: float):
        """
        Add propagation delay to a specific UE in seconds.

        Args:
            input_ue_id (int): The ID of the UE.
            input_prop_delay_to_ue_s (float): Propagation delay in seconds.

        Returns:
            None
        """
        self.prop_delay_to_ues_s[f'UE_{input_ue_id}'] = input_prop_delay_to_ue_s

    def get_prop_delay_to_ue_s(self, input_ue_id: int):
        """
        Get propagation delay to a specific UE in seconds.

        Args:
            input_ue_id (int): The ID of the UE.

        Returns:
            float: Propagation delay in seconds.
        """
        if f'UE_{input_ue_id}' in self.prop_delay_to_ues_s:
            return self.prop_delay_to_ues_s[f'UE_{input_ue_id}']
        else:
            sys.exit(f'UE {input_ue_id} not found in prop_delay_to_ues_s of UE {self.ue_id}')

    def add_prop_delay_to_ue_tick(self, input_ue_id: int, input_prop_delay_to_ue_tick: float):
        """
        Add propagation delay to a specific UE in ticks.

        Args:
            input_ue_id (int): The ID of the UE.
            input_prop_delay_to_ue_tick (float): Propagation delay in ticks.

        Returns:
            None
        """
        self.prop_delay_to_ues_tick[f'UE_{input_ue_id}'] = input_prop_delay_to_ue_tick

    def get_prop_delay_to_ue_tick(self, input_ue_id: int):
        """
        Get propagation delay to a specific UE in ticks.

        Args:
            input_ue_id (int): The ID of the UE.

        Returns:
            float: Propagation delay in ticks.
        """
        if f'UE_{input_ue_id}' in self.prop_delay_to_ues_tick:
            return self.prop_delay_to_ues_tick[f'UE_{input_ue_id}']
        else:
            sys.exit(f'UE {input_ue_id} not found in prop_delay_to_ues_tick of UE {self.ue_id}')

    def get_packet_size_bytes(self, input_packet_id: int = None):
        """
        Get the packet size in bytes.

        Args:
            input_packet_id (int, optional): The ID of the packet. If None, the first packet
                in the buffer is used. Default is None.

        Returns:
            int: Packet size in bytes.
        """
        if input_packet_id is not None:
            return self.ul_buffer.get_packet_by_id(packet_id=input_packet_id).get_size()
        else:
            return self.ul_buffer.get_first_packet().get_size()

    def set_max_n_retx_per_packet(self, input_max_n_retx_per_packet: int):
        """
        Set the maximum number of retransmissions allowed per packet.

        Args:
            input_max_n_retx_per_packet (int): Maximum retransmission attempts.

        Returns:
            None
        """
        self.max_n_retx_per_packet = input_max_n_retx_per_packet

    def get_max_n_retx_per_packet(self):
        """
        Get the maximum number of retransmissions allowed per packet.

        Returns:
            int: Maximum retransmission attempts.
        """
        return self.max_n_retx_per_packet

    def set_relay_bool(self, relay_bool: bool):
        """
        Set whether the UE is acting as a relay.

        Args:
            relay_bool (bool): True if the UE is a relay, False otherwise.

        Returns:
            None
        """
        self.relay = relay_bool

    def get_relay_bool(self):
        """
        Get whether the UE is acting as a relay.

        Returns:
            bool: True if the UE is a relay, False otherwise.
        """
        return self.relay

    def set_data_rx_from_ue(self, data_rx_from_ue: int):
        """
        Set the ID of the UE from which data was received.

        Args:
            data_rx_from_ue (int): ID of the transmitting UE.

        Returns:
            None
        """
        self.packet.set_data_rx_from_ue(data_rx_from_ue=data_rx_from_ue)

    def get_data_rx_from_ue(self):
        """
        Get the ID of the UE from which the packet was received.

        Returns:
            int: ID of the transmitting UE.
        """
        return self.packet.get_data_rx_from_ue()

    def set_packet_id_rx_from_ue(self, packet_id_rx_from_ue: int):
        """
        Set the packet ID received from another UE.

        Args:
            packet_id_rx_from_ue (int): Packet ID received.

        Returns:
            None
        """
        self.packet.set_packet_id_rx_from_ue(packet_id_rx_from_ue=packet_id_rx_from_ue)

    def get_packet_id_rx_from_ue(self):
        """
        Get the packet ID received from another UE.

        Returns:
            int: Packet ID received from another UE.
        """
        return self.packet.get_packet_id_rx_from_ue()

    def set_ack_packet_id_ue(self, packet_id: int):
        """
        Set the acknowledged packet ID for this UE.

        Args:
            packet_id (int): Packet ID acknowledged.

        Returns:
            None
        """
        self.packet_id_ack = packet_id

    def get_ack_packet_id_ue(self):
        """
        Get the acknowledged packet ID for this UE.

        Returns:
            int: Acknowledged packet ID.
        """
        return self.packet_id_ack

    def set_retransmission_packets(self, retransmission_bool: bool):
        """
        Set whether retransmission of packets is enabled.

        Args:
            retransmission_bool (bool): True if retransmissions are enabled, False otherwise.

        Returns:
            None
        """
        self.retransmission_of_packets = retransmission_bool

    def get_retransmission_packets(self):
        """
        Get whether retransmission of packets is enabled.

        Returns:
            bool: True if retransmissions are enabled, False otherwise.
        """
        return self.retransmission_of_packets

    def set_ul_buffer(self):
        """
        Initialize and set the uplink buffer for the UE.

        Returns:
            None
        """
        self.ul_buffer = UeBuffer(max_buffer_size=self.params.get('ue').get('max_buffer_size'))

    def set_packets_sent(self, input_packets_sent: int):
        """
        Set the number of packets sent by the UE.

        Args:
            input_packets_sent (int): Number of sent packets.

        Returns:
            None
        """
        self.packets_sent = input_packets_sent

    def get_packets_sent(self):
        """
        Get the number of packets sent by the UE.

        Returns:
            int: Number of sent packets.
        """
        return self.packets_sent

    def set_last_action(self, input_last_action):
        """
        Set the last action taken by the UE.

        Args:
            input_last_action: Action performed (type may vary).

        Returns:
            None
        """
        self.last_action = input_last_action

    def get_last_action(self):
        """
        Get the last action taken by the UE.

        Returns:
            object: Last action performed (type may vary).
        """
        return self.last_action

    def set_temp_obs(self, input_temp_obs: np.ndarray):
        """
        Set the temporary observation state.

        Args:
            input_temp_obs (np.ndarray): The temporary observation matrix.

        Returns:
            None
        """
        self.temp_obs = cp.deepcopy(input_temp_obs)

    def reset_temp_obs(self):
        """
        Reset the temporary observation to an empty matrix.

        Returns:
            None
        """
        self.temp_obs = np.zeros((5, len(self.neighbour_table)), dtype=np.float32)

    def set_temp_obs_broadcast(self, input_ack_rx_at_ue_tx_index, input_rx_power, input_bs_seen: int = 0):
        """
        Update temporary observation after a successful broadcast.

        Args:
            input_ack_rx_at_ue_tx_index: Index of the transmitting UE for the ACK.
            input_rx_power: Received signal power.
            input_bs_seen (int, optional): Indicator if BS is seen. Default is 0.

        Returns:
            None
        """
        self.temp_obs[0][input_ack_rx_at_ue_tx_index] = 1 # neighbor discovered
        self.temp_obs[1][input_ack_rx_at_ue_tx_index] += 1 # count of a successful ACK reception
        self.temp_obs[2][input_ack_rx_at_ue_tx_index] = input_rx_power # received power
        self.temp_obs[3][input_ack_rx_at_ue_tx_index] = 0 # TTL value
        self.temp_obs[4][input_ack_rx_at_ue_tx_index] = input_bs_seen # if the neighbor sees the BS or not

    def set_obs_update(self, input_data_rx_at_ue_tx_index, input_rx_power):
        """
        Update the observation after a successful data reception.

        Args:
            input_data_rx_at_ue_tx_index: Index of the transmitting UE for the data.
            input_rx_power: Received signal power.

        Returns:
            None
        """
        self.obs[0][input_data_rx_at_ue_tx_index] = 1
        self.obs[2][input_data_rx_at_ue_tx_index] = input_rx_power
        self.obs[3][input_data_rx_at_ue_tx_index] = 0

    def get_temp_obs(self):
        """
        Get the temporary observation matrix.

        Returns:
            np.ndarray: The temporary observation state.
        """
        return self.temp_obs

    def set_obs(self, input_obs: np.ndarray):
        """
        Set the observation matrix.

        Args:
            input_obs (np.ndarray): Observation matrix.

        Returns:
            None
        """
        self.obs = cp.deepcopy(input_obs)

    def reset_obs(self):
        """
        Reset the observation matrix to an empty state.

        Returns:
            None
        """
        self.obs = np.zeros((5, len(self.neighbour_table)), dtype=np.float32)

    def get_obs(self):
        """
        Get the observation matrix.

        Returns:
            np.ndarray: The observation state.
        """
        return self.obs

    def set_old_state(self, input_old_state):
        """
        Set the old state of the UE.

        Args:
            input_old_state: The previous state to store (type may vary).

        Returns:
            None
        """
        self.old_state = cp.deepcopy(input_old_state)

    def get_old_state(self):
        """
        Get the old state of the UE.

        Returns:
            object: The previously stored state.
        """
        return self.old_state

    def set_reward(self, input_reward: list):
        """
        Set the reward values for the UE.

        Args:
            input_reward (list): A list of reward values.

        Returns:
            None
        """
        self.reward = input_reward

    def get_reward(self):
        """
        Get the reward values of the UE.

        Returns:
            list: The list of rewards.
        """
        return self.reward

    def append_reward(self, input_reward: float):
        """
        Append a new reward value to the list.

        Args:
            input_reward (float): Reward value to add.

        Returns:
            None
        """
        self.reward.append(input_reward)

    def get_last_reward(self):
        """
        Get the most recent reward value.

        Returns:
            float: The last reward value.
        """
        return self.reward[-1]

    def set_simulations_reward(self, input_simulations_reward: list):
        """
        Set the reward values for multiple simulations.

        Args:
            input_simulations_reward (list): A list of simulation rewards.

        Returns:
            None
        """
        self.simulations_reward = input_simulations_reward

    def get_simulations_reward(self):
        """
        Get the simulation rewards.

        Returns:
            list: The list of simulation rewards.
        """
        return self.simulations_reward

    def get_last_simulations_reward(self):
        """
        Get the last simulation reward value.

        Returns:
            float: The most recent simulation reward.
        """
        return self.simulations_reward[-1]

    def append_simulations_reward(self, input_simulations_reward: float):
        """
        Append a new simulation reward value.

        Args:
            input_simulations_reward (float): Simulation reward value to add.

        Returns:
            None
        """
        self.simulations_reward.append(input_simulations_reward)

    def set_W_reward(self, input_reward: list):
        """
        Set the W reward values.

        Args:
            input_reward (list): A list of W reward values.

        Returns:
            None
        """
        self.W_reward = input_reward

    def get_W_reward(self):
        """
        Get the W reward values.

        Returns:
            list: The list of W rewards.
        """
        return self.W_reward

    def append_W_reward(self, input_reward: float):
        """
        Append a new W reward value.

        Args:
            input_reward (float): W reward value to add.

        Returns:
            None
        """
        self.W_reward.append(input_reward)

    def get_last_W_reward(self):
        """
        Get the last W reward value.

        Returns:
            float: The most recent W reward.
        """
        return self.W_reward[-1]

    def set_W_simulations_reward(self, input_simulations_reward: list):
        """
        Set the W simulation rewards.

        Args:
            input_simulations_reward (list): A list of W simulation rewards.

        Returns:
            None
        """
        self.W_simulations_reward = input_simulations_reward

    def get_W_simulations_reward(self):
        """
        Get the W simulation rewards.

        Returns:
            list: The list of W simulation rewards.
        """
        return self.W_simulations_reward

    def get_last_W_simulations_reward(self):
        """
        Get the last W simulation reward.

        Returns:
            float: The most recent W simulation reward.
        """
        return self.W_simulations_reward[-1]

    def append_W_simulations_reward(self, input_simulations_reward: float):
        """
        Append a new W simulation reward value.

        Args:
            input_simulations_reward (float): W simulation reward value to add.

        Returns:
            None
        """
        self.W_simulations_reward.append(input_simulations_reward)

    def set_Q_reward(self, input_reward: list):
        """
        Set the Q reward values.

        Args:
            input_reward (list): A list of Q reward values.

        Returns:
            None
        """
        self.Q_reward = input_reward

    def get_Q_reward(self):
        """
        Get the Q reward values.

        Returns:
            list: The list of Q rewards.
        """
        return self.Q_reward

    def append_Q_reward(self, input_reward: float):
        """
        Append a new Q reward value.

        Args:
            input_reward (float): Q reward value to add.

        Returns:
            None
        """
        self.Q_reward.append(input_reward)

    def get_last_Q_reward(self):
        """
        Get the last Q reward value.

        Returns:
            float: The most recent Q reward.
        """
        return self.Q_reward[-1]

    def set_Q_simulations_reward(self, input_simulations_reward: list):
        """
        Set the Q simulation rewards.

        Args:
            input_simulations_reward (list): A list of Q simulation rewards.

        Returns:
            None
        """
        self.Q_simulations_reward = input_simulations_reward

    def get_Q_simulations_reward(self):
        """
        Get the Q simulation rewards.

        Returns:
            list: The list of Q simulation rewards.
        """
        return self.Q_simulations_reward

    def get_last_Q_simulations_reward(self):
        """
        Get the last Q simulation reward.

        Returns:
            float: The most recent Q simulation reward.
        """
        return self.Q_simulations_reward[-1]

    def append_Q_simulations_reward(self, input_simulations_reward: float):
        """
        Append a new Q simulation reward value.

        Args:
            input_simulations_reward (float): Q simulation reward value to add.

        Returns:
            None
        """
        self.Q_simulations_reward.append(input_simulations_reward)

    def set_model(self, input_model):
        """
        Set the model.

        Args:
            input_model: The model to set (type may vary, e.g., ML model).

        Returns:
            None
        """
        self.model = input_model

    def get_model(self):
        """
        Get the model.

        Returns:
            object: The stored model.
        """
        return self.model

    def set_target_model(self, input_target_model):
        """
        Set the target model.

        Args:
            input_target_model: The target model (type may vary).

        Returns:
            None
        """
        self.target_model = input_target_model

    def get_target_model(self):
        """
        Get the target model.

        Returns:
            object: The stored target model.
        """
        return self.target_model

    def set_W_model(self, input_model):
        """
        Set the W model.

        Args:
            input_model: The W model (type may vary).

        Returns:
            None
        """
        self.W_model = input_model

    def get_W_model(self):
        """
        Get the W model.

        Returns:
            object: The stored W model.
        """
        return self.W_model

    def set_W_target_model(self, input_target_model):
        """
        Set the W target model.

        Args:
            input_target_model: The W target model (type may vary).

        Returns:
            None
        """
        self.W_target_model = input_target_model

    def get_W_target_model(self):
        """
        Get the W target model.

        Returns:
            object: The stored W target model.
        """
        return self.W_target_model

    def set_Q_model(self, input_model):
        """
        Set the Q model.

        Args:
            input_model: The Q model (type may vary).

        Returns:
            None
        """
        self.Q_model = input_model

    def get_Q_model(self):
        """
        Get the Q model.

        Returns:
            object: The stored Q model.
        """
        return self.Q_model

    def set_Q_target_model(self, input_target_model):
        """
        Set the Q target model.

        Args:
            input_target_model: The Q target model (type may vary).

        Returns:
            None
        """
        self.Q_target_model = input_target_model

    def get_Q_target_model(self):
        """
        Get the Q target model.

        Returns:
            object: The stored Q target model.
        """
        return self.Q_target_model

    def set_replay_buffer(self, input_replay_buffer):
        """
        Set the replay buffer.

        Args:
            input_replay_buffer: The replay buffer (list or custom object).

        Returns:
            None
        """
        self.replay_buffer = input_replay_buffer

    def get_replay_buffer(self):
        """
        Get the replay buffer.

        Returns:
            object: The stored replay buffer.
        """
        return self.replay_buffer

    def get_last_replay_instance(self):
        """
        Get the last replay buffer instance.

        Returns:
            object: The most recent entry in the replay buffer.
        """
        return self.replay_buffer[-1]

    def append_replay_buffer(self, input_replay_buffer_instance):
        """
        Append a new instance to the replay buffer.

        Args:
            input_replay_buffer_instance: The instance to add.

        Returns:
            None
        """
        self.replay_buffer.append(input_replay_buffer_instance)

    def drop_last_replay_instance(self):
        """
        Remove the last instance from the replay buffer.

        """
        self.replay_buffer.pop()

    def set_W_replay_buffer(self, input_replay_buffer):
        """
        Set the W replay buffer.

        Args:
            input_replay_buffer: The W replay buffer (list or custom object).

        Returns:
            None
        """
        self.W_replay_buffer = input_replay_buffer

    def get_W_replay_buffer(self):
        """
        Get the W replay buffer.

        Returns:
            object: The stored W replay buffer.
        """
        return self.W_replay_buffer

    def get_last_W_replay_instance(self):
        """
        Get the last W replay buffer instance.

        Returns:
            object: The most recent entry in the W replay buffer.
        """
        return self.W_replay_buffer[-1]

    def append_W_replay_buffer(self, input_replay_buffer_instance):
        """
        Append a new instance to the W replay buffer.

        Args:
            input_replay_buffer_instance: The instance to add.

        Returns:
            None
        """
        self.W_replay_buffer.append(input_replay_buffer_instance)

    def drop_last_W_replay_instance(self):
        """
        Remove the last instance from the W replay buffer.

        """
        self.W_replay_buffer.pop()

    def set_Q_replay_buffer(self, input_replay_buffer):
        """
        Set the Q replay buffer.

        Args:
            input_replay_buffer: The Q replay buffer (list or custom object).

        Returns:
            None
        """
        self.Q_replay_buffer = input_replay_buffer

    def get_Q_replay_buffer(self):
        """
        Get the Q replay buffer.

        Returns:
            object: The stored Q replay buffer.
        """
        return self.Q_replay_buffer

    def get_last_Q_replay_instance(self):
        """
        Get the last Q replay buffer instance.

        Returns:
            object: The most recent entry in the Q replay buffer.
        """
        return self.Q_replay_buffer[-1]

    def append_Q_replay_buffer(self, input_replay_buffer_instance):
        """
        Append a new instance to the Q replay buffer.

        Args:
            input_replay_buffer_instance: The instance to add.

        Returns:
            None
        """
        self.Q_replay_buffer.append(input_replay_buffer_instance)

    def drop_last_Q_replay_instance(self):
        """
        Remove the last instance from the Q replay buffer.

        """
        self.Q_replay_buffer.pop()

    #################### Other Methods ####################

    def set_neighbour_table(self, input_neighbour_table: list):
        """
        Set the UE’s neighbour table.

        Args:
            input_neighbour_table (list): List containing information about neighbouring UEs.

        Returns:
            None
        """
        self.neighbour_table = input_neighbour_table

    def get_neighbour_table(self):
        """
        Get the current neighbour table.

        Returns:
            list: List of neighbouring UEs.
        """
        return self.neighbour_table

    def set_env(self, input_env):
        """
        Set the simulation environment for the UE.

        Args:
            input_env: The environment object in which this UE operates.

        Returns:
            None
        """
        self.env = input_env

    def get_env(self):
        """
        Get the simulation environment associated with the UE.

        Returns:
            object: The environment instance.
        """
        return self.env

    def set_epsilon(self, input_epsilon: float):
        """
        Set the epsilon value used in the epsilon-greedy policy.

        Args:
            input_epsilon (float): The epsilon value (probability of random exploration).

        Returns:
            None
        """
        self.epsilon = input_epsilon

    def get_epsilon(self):
        """
        Get the epsilon value used in the epsilon-greedy policy.

        Returns:
            float: The epsilon value.
        """
        return self.epsilon

    def set_best_weights(self, input_best_weights):
        """
        Set the best model weights found so far.

        Args:
            input_best_weights: The best weights of the model.

        Returns:
            None
        """
        self.best_weights = input_best_weights

    def get_best_weights(self):
        """
        Get the best weights of the model.

        Returns:
            object: The stored best weights.
        """
        return self.best_weights

    def set_best_score(self, input_best_score: float):
        """
        Set the best score achieved by the model.

        Args:
            input_best_score (float): The best performance score achieved.

        Returns:
            None
        """
        self.best_score = input_best_score

    def get_best_score(self):
        """
        Get the best score achieved by the model.

        Returns:
            float: The best performance score.
        """
        return self.best_score

    def set_unicast_rx_address(self, input_unicast_rx_address):
        """
        Set the unicast RX address of the UE.

        Args:
            input_unicast_rx_address: The RX address for unicast communication.

        Returns:
            None
        """
        self.unicast_rx_address = input_unicast_rx_address

    def get_unicast_rx_address(self):
        """
        Get the unicast RX address of the UE.

        Returns:
            object: The unicast RX address.
        """
        return self.unicast_rx_address

    def set_unicast_rx_index(self, input_unicast_rx_index):
        """
        Set the unicast RX index.

        Args:
            input_unicast_rx_index: The index representing the unicast RX target.

        Returns:
            None
        """
        self.unicast_rx_index = input_unicast_rx_index

    def get_unicast_rx_index(self):
        """
        Get the unicast RX index.

        Returns:
            int: The unicast RX index.
        """
        return self.unicast_rx_index

    def set_broadcast_bool(self, input_broadcast_bool: bool):
        """
        Set the broadcast mode flag.

        Args:
            input_broadcast_bool (bool): True if broadcast mode is enabled, False otherwise.

        Returns:
            None
        """
        self.broadcast_bool = input_broadcast_bool

    def get_broadcast_bool(self):
        """
        Get the broadcast mode flag.

        Returns:
            bool: True if broadcast mode is active, False otherwise.
        """
        return self.broadcast_bool

    def set_action_list(self, input_action_list: list):
        """
        Set the list of actions performed by the UE.

        Args:
            input_action_list (list): List containing performed actions.

        Returns:
            None
        """
        self.action_list = input_action_list

    def get_action_list(self):
        """
        Get the list of actions performed by the UE.

        Returns:
            list: The list of actions.
        """
        return self.action_list

    def set_W_action_list(self, input_action_list: list):
        """
        Set the list of actions for the W learning model.

        Args:
            input_action_list (list): List of actions for W model.

        Returns:
            None
        """
        self.W_action_list = input_action_list

    def get_W_action_list(self):
        """
        Get the list of actions for the W learning model.

        Returns:
            list: The W model’s action list.
        """
        return self.W_action_list

    def set_Q_action_list(self, input_action_list: list):
        """
        Set the list of actions for the Q learning model.

        Args:
            input_action_list (list): List of actions for Q model.

        Returns:
            None
        """
        self.Q_action_list = input_action_list

    def get_Q_action_list(self):
        """
        Get the list of actions for the Q learning model.

        Returns:
            list: The Q model’s action list.
        """
        return self.Q_action_list

    def append_action_list(self, input_action):
        """
        Append an action to the main action list.

        Args:
            input_action: The action to append.

        Returns:
            None
        """
        self.action_list.append(input_action)

    def append_W_action_list(self, input_action):
        """
        Append an action to the W model’s action list.

        Args:
            input_action: The action to append.

        Returns:
            None
        """
        self.W_action_list.append(input_action)

    def append_Q_action_list(self, input_action):
        """
        Append an action to the Q model’s action list.

        Args:
            input_action: The action to append.

        Returns:
            None
        """
        self.Q_action_list.append(input_action)

    def set_success_action_list(self, input_success_action_list: list):
        """
        Set the list of successful actions taken by the UE.

        Args:
            input_success_action_list (list): List of successful actions.

        Returns:
            None
        """
        self.success_action_list = input_success_action_list

    def get_success_action_list(self):
        """
        Get the list of successful actions taken by the UE.

        Returns:
            list: The success action list.
        """
        return self.success_action_list

    def append_success_action_list(self, input_success_action):
        """
        Append a successful action to the list.

        Args:
            input_success_action: The successful action to append.

        Returns:
            None
        """
        self.success_action_list.append(input_success_action)

    def set_actions_per_simulation(self, input_actions_per_simulation: list):
        """
        Set the recorded actions per simulation run.

        Args:
            input_actions_per_simulation (list): Actions per simulation list.

        Returns:
            None
        """
        self.actions_per_simulation = input_actions_per_simulation

    def get_actions_per_simulation(self):
        """
        Get the actions per simulation data.

        Returns:
            list: The actions per simulation.
        """
        return self.actions_per_simulation

    def append_actions_per_simulation(self):
        """
        Count the number of each action type in the last simulation
        and append them to the per-simulation record.

        Returns:
            None
        """
        self.actions_per_simulation[0].append(self.action_list.count(0))
        self.actions_per_simulation[1].append(self.action_list.count(1))
        self.actions_per_simulation[2].append(self.action_list.count(2))
        self.actions_per_simulation[3].append(self.action_list.count(3))

    def set_success_actions_per_simulation(self, input_success_actions_per_simulation: list):
        """
        Set the successful actions recorded per simulation.

        Args:
            input_success_actions_per_simulation (list): Successful actions list.

        Returns:
            None
        """
        self.success_actions_per_simulation = input_success_actions_per_simulation

    def get_success_actions_per_simulation(self):
        """
        Get the successful actions recorded per simulation.

        Returns:
            list: Successful actions per simulation.
        """
        return self.success_actions_per_simulation

    def append_success_actions_per_simulation(self):
        """
        Append the counts of successful actions taken in the last simulation
        to the per-simulation success record.

        Returns:
            None
        """
        self.success_actions_per_simulation[0].append(self.success_action_list.count(0))
        self.success_actions_per_simulation[1].append(self.success_action_list.count(1))

    def set_tx_broad_list(self, input_tx_broad_list: list):
        """
        Set the list of broadcast transmissions.

        Args:
            input_tx_broad_list (list): List of broadcast transmissions.

        Returns:
            None
        """
        self.tx_broad_list = input_tx_broad_list

    def get_tx_broad_list(self):
        """
        Get the list of broadcast transmissions.

        Returns:
            list: Broadcast transmission list.
        """
        return self.tx_broad_list

    def set_data_discard_bool(self, input_data_discard_bool: bool):
        """
        Set the flag indicating whether data was discarded.

        Args:
            input_data_discard_bool (bool): True if data was discarded, False otherwise.

        Returns:
            None
        """
        self.data_discard_bool = input_data_discard_bool

    def get_data_discard_bool(self):
        """
        Get the flag indicating whether data was discarded.

        Returns:
            bool: True if data was discarded, False otherwise.
        """
        return self.data_discard_bool

    def set_saved_coordinates(self, input_saved_coordinates):
        """
        Set the saved coordinates of the UE.

        Args:
            input_saved_coordinates: The coordinates to store.

        Returns:
            None
        """
        self.saved_coordinates = input_saved_coordinates

    def get_saved_coordinates(self):
        """
        Get the saved coordinates of the UE.

        Returns:
            object: The saved coordinates.
        """
        return self.saved_coordinates

    def reset_complete_actions_since_last_ttl_reset(self, input_neighbour_number: int):
        """
        Reset the action counters since the last TTL reset for all neighbours.

        Args:
            input_neighbour_number (int): The number of neighbours.

        Returns:
            None
        """
        self.actions_since_last_ttl_reset = [0 for _ in range(input_neighbour_number)]

    def reset_actions_since_last_ttl_reset(self, input_neighbour_index: int):
        """
        Reset the action counter since the last TTL reset for a specific neighbour.

        Args:
            input_neighbour_index (int): Index of the neighbour.

        Returns:
            None
        """
        self.actions_since_last_ttl_reset[input_neighbour_index] = 0

    def increment_actions_since_last_ttl_reset(self, input_neighbour_index: int):
        """
        Increment the action counter for a specific neighbour since the last TTL reset.

        Args:
            input_neighbour_index (int): Index of the neighbour.

        Returns:
            None
        """
        self.actions_since_last_ttl_reset[input_neighbour_index] += 1

    def increment_all_actions_since_last_ttl_reset(self):
        """
        Increment all action counters since the last TTL reset.

        Returns:
            None
        """
        for i in range(len(self.actions_since_last_ttl_reset)):
            self.actions_since_last_ttl_reset[i] += 1

    def check_num_tx_RL(self):
        """
        Check whether there are packets that have reached the maximum number of retransmissions (RL version).

        Returns:
            bool: True if there is at least one packet still eligible for transmission, False otherwise.
        """
        data_to_transmit = False
        packets_list = list()
        for packet in self.get_updated_packet_list():
            if packet.get_data_to_be_forwarded_bool() or packet.get_id() == self.ul_buffer.get_first_packet().get_id():
                if packet.get_num_tx() <= packet.get_max_n_retx() + 1:
                    data_to_transmit = True
                else:
                    if packet.get_data_to_be_forwarded_bool():
                        packets_list.append(packet)
                        packet.set_packet_discarded(True)
                    else:
                        packets_list.append(packet)

        for packet in packets_list:
            if packet.get_data_to_be_forwarded_bool() is False:
                self.update_n_data_discarded()

            self.remove_packet(packet_id=packet.get_id())
            self.packets_sent -= 1

        return data_to_transmit

    def check_rtx(self):
        """
        Check whether there are packets that have reached the maximum number of retransmissions.

        Returns:
            bool: True if at least one packet is still valid for transmission, False otherwise.
        """
        data_to_transmit = False
        for packet in self.get_updated_packet_list():
            if packet.get_data_to_be_forwarded_bool() or packet.get_id() == self.ul_buffer.get_first_packet().get_id():
                if packet.get_num_tx() <= packet.get_max_n_retx() + 1:
                    data_to_transmit = True
        return data_to_transmit

    def check_generated_packet_present(self):
        """
        Check if there is at least one packet generated by the UE in the queue.

        Returns:
            bool: True if there is a packet generated by this UE, False otherwise.
        """
        for packet in self.get_updated_packet_list():
            if packet.get_data_to_be_forwarded_bool() is False:
                return True
        return False

    def check_remove_packet(self, input_enable_print=False):
        """
        Remove packets marked for deletion from the UE buffer.

        Args:
            input_enable_print (bool): If True, enables debug print statements. Defaults to False.

        Returns:
            None
        """
        for txs in self.packets_to_be_removed.keys():
            for packet_id in self.packets_to_be_removed[txs]:
                for buffer_packet in self.ul_buffer.buffer_packet_list:
                    if packet_id == buffer_packet.get_id():
                        self.remove_packet(packet_id=packet_id, input_enable_print=input_enable_print)
                        self.packets_sent -= 1
                        if self.Q_and_W_enabled:
                            self.Q_and_W_acks_rx_per_step_counter += 1
            self.packets_to_be_removed[txs] = []

    def update_neighbours_forwarding(self, input_rx_power, input_tx_str, input_bs_seen):
        """
        Update neighbour observation data after a successful unicast transmission.

        Args:
            input_rx_power: Received power value of the unicast transmission.
            input_tx_str: The neighbour identifier (string) or 'BS' for base station.
            input_bs_seen: Indicator showing whether the base station was visible
                           (meaning that the TX can directly reach the BS via one hop).

        Returns:
            None
        """
        if input_tx_str == "BS":
            index = -1
        else:
            index = self.neighbour_table.index(input_tx_str)

        self.obs[0][index] = 1
        self.obs[1][index] += 1
        self.obs[2][index] = input_rx_power
        self.obs[3][index] = 0
        self.obs[4][index] = input_bs_seen

        self.unicast_rx_index = index
        self.unicast_rx_address = self.neighbour_table[index]

    def check_action_packet_id(self):
        """
        Check whether the action packet has exceeded its retransmission limit.

        Returns:
            bool: True if the packet should be discarded, False otherwise.
        """
        packet_to_discard = False
        for packet in self.get_updated_packet_list():
            if packet.get_id() == self.action_packet_id:
                if packet.get_num_tx() > packet.get_max_n_retx() + 1:
                    packet_to_discard = True
                break
        return packet_to_discard

    def check_present_action_packet_id(self):
        """
        Check whether the packet corresponding to the current action is still in the queue.

        Returns:
            bool: True if the action packet is still in the buffer, False otherwise.
        """
        action_packet_present = False
        for packet in self.get_updated_packet_list():
            if packet.get_id() == self.action_packet_id:
                action_packet_present = True
                break
        return action_packet_present

    def unicast_handling_failure_v2(self, input_ttl, input_unicast_ampl_factor_no_ack, input_energy_factor,
                                    input_max_n_retx_per_packet):
        """
        Handle a unicast failure in the RL environment.

        Args:
            input_ttl: Time-to-live threshold for the packet.
            input_unicast_ampl_factor_no_ack: Amplification factor used when no ACK is received.
            input_energy_factor: Energy cost factor for transmissions.
            input_max_n_retx_per_packet: Maximum retransmissions allowed per packet.

        Returns:
            None
        """
        self.append_action_list(input_action=0)
        tx_index = self.get_unicast_rx_index()
        self.set_old_state(input_old_state=self.get_obs())
        self.obs[3][tx_index] += 1
        if self.obs[3][tx_index] > input_ttl:
            self.obs = ttl_reset(self.obs, tx_index)
            self.reset_actions_since_last_ttl_reset(tx_index)

        reward = input_unicast_ampl_factor_no_ack - input_energy_factor * input_max_n_retx_per_packet
        self.append_reward(reward)
        if not DDQN_new_state:
            self.append_replay_buffer(
                input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0],
                                              False])
        else:
            if not self.Rainbow_DQN:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                  select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                                   DRL_input_type_state,
                                                                   self.actions_since_last_ttl_reset),
                                                  False])
            else:
                self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                        select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                         DRL_input_type_state, self.actions_since_last_ttl_reset),
                                        False), priority=1.0)

        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    def broadcast_handling_failure_v2(self, input_ttl, input_broadcast_ampl_factor_change,
                                      input_broadcast_ampl_factor_no_change, input_energy_factor,
                                      input_max_n_retx_per_packet):
        """
        Handle a broadcast failure in the RL environment.

        Args:
            input_ttl: Time-to-live threshold for the packet.
            input_broadcast_ampl_factor_change: Amplification factor when broadcast leads to state change.
            input_broadcast_ampl_factor_no_change: Amplification factor when no change occurs.
            input_energy_factor: Energy cost factor for broadcast transmissions.
            input_max_n_retx_per_packet: Maximum retransmissions allowed per packet.

        Returns:
            None
        """
        self.set_old_state(input_old_state=self.get_obs())
        self.append_action_list(input_action=1)
        self.obs[3] += 1
        for i in range(len(self.obs[3])):
            if self.obs[3][i] > input_ttl:
                self.obs = ttl_reset(self.obs, i)
                self.reset_actions_since_last_ttl_reset(i)
            if self.obs[0][i] == 0:
                self.obs[3][i] = 0

        if np.array_equal(self.old_state[0], self.obs[0]) is False:
            reward = input_broadcast_ampl_factor_change * (
                    input_broadcast_ampl_factor_change - input_energy_factor * input_max_n_retx_per_packet)
            self.append_reward(reward)
            if not DDQN_new_state:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0],
                                                  False])
            else:
                if not self.Rainbow_DQN:
                    self.append_replay_buffer(
                        input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                      select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                                      False])
                else:
                    self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                            select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                             DRL_input_type_state, self.actions_since_last_ttl_reset),
                                            False), priority=1.0)
        else:
            reward = input_broadcast_ampl_factor_no_change * (
                    input_broadcast_ampl_factor_no_change - input_energy_factor * input_max_n_retx_per_packet)
            self.append_reward(reward)
            if not DDQN_new_state:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0],
                                                  False])
            else:
                if not self.Rainbow_DQN:
                    self.append_replay_buffer(
                        input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                      select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                                      False])

                else:
                    self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                            select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                             DRL_input_type_state, self.actions_since_last_ttl_reset),
                                            False), priority=1.0)


        self.set_broadcast_bool(input_broadcast_bool=False)
        self.set_last_action(input_last_action=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    def unicast_handling_v2(self, input_rx_power, input_unicast_ampl_factor_ack, input_energy_factor,
                            input_bs_seen: int = 0):
        """
        Handles a successful unicast transmission in the old RL version.
        Updates observations, computes the reward, and appends experience to the replay buffer.

        Args:
            input_rx_power (float): Received power of the unicast packet.
            input_unicast_ampl_factor_ack (float): Amplification factor for the ACK response of the unicast.
            input_energy_factor (float): Energy penalty or cost factor applied to transmission attempts.
            input_bs_seen (int): Indicates whether the BS is seen from the UE or not (Default value = 0).

        Returns:
            None
        """
        self.set_old_state(input_old_state=self.get_obs())
        self.append_action_list(input_action=0)

        considered_packet = None
        for packet in self.copy_buffer_packet_list:
            if packet.get_id() == self.action_packet_id:
                considered_packet = packet
                break

        if considered_packet is None:
            for packet in self.copy_buffer_packet_list:
                if packet.generated_by_ue == self.ue_id:
                    considered_packet = packet
                    break

        self.obs[0][self.get_unicast_rx_index()] = 1
        self.obs[1][self.get_unicast_rx_index()] += 1
        self.obs[2][self.get_unicast_rx_index()] = input_rx_power
        self.obs[3][self.get_unicast_rx_index()] = 0
        self.obs[4][self.get_unicast_rx_index()] = input_bs_seen

        reward = input_unicast_ampl_factor_ack - input_energy_factor * (considered_packet.get_num_tx() - 1)
        self.append_reward(reward)
        if not DDQN_new_state:
            self.append_replay_buffer(
                input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0], False])
        else:
            if not self.Rainbow_DQN:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                  select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                                   DRL_input_type_state,
                                                                   self.actions_since_last_ttl_reset),
                                                  False])
            else:
                self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                        select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                         DRL_input_type_state, self.actions_since_last_ttl_reset),
                                        False), priority=1.0)

        self.append_success_action_list(input_success_action=self.get_last_action())

        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    def unicast_handling_v2_no_neighbour_update(self, input_unicast_ampl_factor_ack, input_energy_factor):
        """
        Handles a successful unicast transmission (old RL version) without updating the neighbor table.
        Computes and logs reward but does not modify neighbor information.

        Args:
            input_unicast_ampl_factor_ack (float): Amplification factor for the ACK response of the unicast.
            input_energy_factor (float): Energy penalty or cost factor applied to transmission attempts.

        Returns:
            None
        """
        self.set_old_state(input_old_state=self.get_obs())
        self.append_action_list(input_action=0)

        considered_packet = None
        for packet in self.copy_buffer_packet_list:
            if packet.get_id() == self.action_packet_id:
                considered_packet = packet
                break

        if considered_packet is None:
            for packet in self.copy_buffer_packet_list:
                if packet.generated_by_ue == self.ue_id:
                    considered_packet = packet
                    break

        reward = input_unicast_ampl_factor_ack - input_energy_factor * (considered_packet.get_num_tx() - 1)
        self.append_reward(reward)
        if not DDQN_new_state:
            self.append_replay_buffer(
                input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0], False])
        else:
            if not self.Rainbow_DQN:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                  select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                                   DRL_input_type_state,
                                                                   self.actions_since_last_ttl_reset),
                                                  False])
            else:
                self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                        select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                         DRL_input_type_state, self.actions_since_last_ttl_reset),
                                        False), priority=1.0)

        self.append_success_action_list(input_success_action=self.get_last_action())

        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    def broadcast_handling_v2(self, input_ttl, input_broadcast_ampl_factor_change,
                              input_broadcast_ampl_factor_no_change, input_energy_factor):
        """
        Handles a successful broadcast transmission (old RL version).
        Updates neighbor states, TTL counters, and computes rewards depending on whether neighbor states changed.

        Args:
            input_ttl (int): Time-to-live threshold for broadcast entries.
            input_broadcast_ampl_factor_change (float): Amplification factor applied when neighbor table changes.
            input_broadcast_ampl_factor_no_change (float): Amplification factor applied when neighbor table remains the same.
            input_energy_factor (float): Energy penalty or cost factor applied to transmission attempts.

        Returns:
            None
        """
        self.set_old_state(input_old_state=self.get_obs())
        self.append_action_list(input_action=1)

        considered_packet = None
        for packet in self.copy_buffer_packet_list:
            if packet.get_id() == self.action_packet_id:
                considered_packet = packet
                break

        if considered_packet is None:
            for packet in self.copy_buffer_packet_list:
                if packet.generated_by_ue == self.ue_id:
                    considered_packet = packet
                    break

        for i in range(len(self.obs[0])):
            if self.temp_obs[1][i] >= 1:
                self.obs[0][i] = 1
                self.obs[1][i] += self.temp_obs[1][i]
                self.obs[2][i] = self.temp_obs[2][i]
                self.obs[3][i] = 0
                self.obs[4][i] = self.temp_obs[4][i]
            elif self.temp_obs[1][i] == 0 and self.old_state[0][i] == 1:
                self.obs[3][i] += 1

            if self.obs[3][i] > input_ttl:
                self.obs = ttl_reset(self.obs, i)
                self.reset_actions_since_last_ttl_reset(i)

        if np.array_equal(self.old_state[0], self.obs[0]) is False:
            reward = input_broadcast_ampl_factor_change * (
                    input_broadcast_ampl_factor_change - input_energy_factor * (considered_packet.get_num_tx() - 1))
            self.append_reward(reward)
            if not DDQN_new_state:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0],
                                                  False])
            else:
                if not self.Rainbow_DQN:
                    self.append_replay_buffer(
                        input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                      select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                                       DRL_input_type_state,
                                                                       self.actions_since_last_ttl_reset),
                                                      False])
                else:
                    self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                            select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                             DRL_input_type_state, self.actions_since_last_ttl_reset),
                                            False), priority=1.0)

            self.append_success_action_list(input_success_action=self.get_last_action())
        else:
            reward = input_broadcast_ampl_factor_no_change * (
                    input_broadcast_ampl_factor_no_change - input_energy_factor * (considered_packet.get_num_tx() - 1))
            self.append_reward(reward)
            if not DDQN_new_state:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0],
                                                  False])
            else:
                if not self.Rainbow_DQN:
                    self.append_replay_buffer(
                        input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                      select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                                       DRL_input_type_state,
                                                                       self.actions_since_last_ttl_reset),
                                                      False])
                else:
                    self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                            select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number,
                                                             DRL_input_type_state, self.actions_since_last_ttl_reset),
                                            False), priority=1.0)

        self.set_last_action(input_last_action=None)
        self.set_broadcast_bool(input_broadcast_bool=False)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    def unicast_handling_failure_no_reward(self, input_ttl):
        """
        Handles a unicast transmission failure without applying any reward.
        Increments TTL counters and resets entries that exceed the TTL.

        Args:
            input_ttl (int): Time-to-live threshold for unicast entries.

        Returns:
            None
        """
        tx_index = self.get_unicast_rx_index()
        self.set_old_state(input_old_state=self.get_obs())
        self.obs[3][tx_index] += 1
        if self.obs[3][tx_index] > input_ttl:
            self.obs = ttl_reset(self.obs, tx_index)
            self.unicast_address = None

        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.set_old_state(input_old_state=None)
        self.copy_unicast_rx_address = None
        self.new_action_bool = True

    def broadcast_handling_failure_no_reward(self, input_ttl):
        """
        Handles a broadcast transmission failure without applying any reward.
        Increments TTL counters and resets expired entries.

        Args:
            input_ttl (int): Time-to-live threshold for broadcast entries.

        Returns:
            None
        """
        self.set_old_state(input_old_state=self.get_obs())
        self.obs[3] += 1
        for i in range(len(self.obs[3])):
            if self.obs[3][i] > input_ttl:
                self.obs = ttl_reset(self.obs, i)
            if self.obs[0][i] == 0:
                self.obs[3][i] = 0

        self.set_broadcast_bool(input_broadcast_bool=False)
        self.set_last_action(input_last_action=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    def unicast_handling_no_reward(self, input_rx_power, input_reset_vars, input_bs_seen: int = 0):
        """
        Handles a successful unicast transmission without computing or applying rewards.
        Updates the neighbor table and optionally resets related variables.

        Args:
            input_rx_power (float): Received power of the unicast packet.
            input_reset_vars (bool): Whether to reset transmission-related state variables.
            input_bs_seen (int): Indicates whether the BS is seen from the UE or not (Default value = 0).

        Returns:
            None
        """
        self.set_old_state(input_old_state=self.get_obs())

        self.obs[0][self.get_unicast_rx_index()] = 1
        self.obs[1][self.get_unicast_rx_index()] += 1
        self.obs[2][self.get_unicast_rx_index()] = input_rx_power
        self.obs[3][self.get_unicast_rx_index()] = 0
        self.obs[4][self.get_unicast_rx_index()] = input_bs_seen

        if input_reset_vars is True:
            self.set_last_action(input_last_action=None)
            self.set_unicast_rx_index(input_unicast_rx_index=None)
            self.set_unicast_rx_address(input_unicast_rx_address=None)
            self.set_old_state(input_old_state=None)
            self.new_action_bool = True

    def broadcast_handling_no_reward(self, input_ttl):
        """
        Handles a successful broadcast transmission without computing or applying rewards.
        Updates neighbor states and TTL counters.

        Args:
            input_ttl (int): Time-to-live threshold for broadcast entries.

        Returns:
            None
        """
        self.set_old_state(input_old_state=self.get_obs())

        for i in range(len(self.obs[0])):
            if self.temp_obs[1][i] >= 1:
                self.obs[0][i] = 1
                self.obs[1][i] += self.temp_obs[1][i]
                self.obs[2][i] = self.temp_obs[2][i]
                self.obs[3][i] = 0
                self.obs[4][i] = self.temp_obs[4][i]
            elif self.temp_obs[1][i] == 0 and self.old_state[0][i] == 1:
                self.obs[3][i] += 1

            if self.obs[3][i] > input_ttl:
                self.obs = ttl_reset(self.obs, i)

        self.set_last_action(input_last_action=None)
        self.set_broadcast_bool(input_broadcast_bool=False)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    def update_neighbor_table_unicast_success(self, input_rx_power, input_bs_seen=0):
        """
        Updates the neighbor table after a successful unicast transmission.

        Args:
            input_rx_power (float): Received power of the unicast packet.
            input_bs_seen (int): Indicates whether the BS is seen from the UE or not (Default value = 0).

        Returns:
            None
        """
        self.obs[0][self.get_unicast_rx_index()] = 1
        self.obs[1][self.get_unicast_rx_index()] += 1
        self.obs[2][self.get_unicast_rx_index()] = input_rx_power
        self.obs[3][self.get_unicast_rx_index()] = 0
        self.obs[4][self.get_unicast_rx_index()] = input_bs_seen

    def unicast_handling_no_reward_no_neighbor_update(self):
        """
        Handles a unicast success without reward and without updating the neighbor table.
        Only resets internal state variables.

        Returns:
            None
        """
        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.new_action_bool = True

    def reward_computation_for_only_Q(self, input_goal_oriented: str = None):
        """
        Computes reward based on queue-related (Q) parameter only.

        Args:
            input_goal_oriented (str): Reward mode ("S&L", "S", or "L").
                                       Determines which set of hyperparameters to use (Default value = None).

        Returns:
            float: Computed reward value based on queue performance.
        """
        if self.Q_forbidden_action:
            self.Q_forbidden_action = False
            reward = -1.0
        else:
            if input_goal_oriented == "S&L":
                # Hyperparameters
                alpha = 0.8     # PCK_TX with success
                beta = 0.3      # RTX discarded
                delta = 0.2     # PCK discarded max rtx
                gamma = 0       # PCK discarded full queue

            elif input_goal_oriented == "S":
                # Hyperparameters
                alpha = 0.5     # PCK_TX with success
                beta = 0      # RTX discarded
                delta = 0.25     # PCK discarded max rtx
                gamma = 0.25       # PCK discarded full queue

            elif input_goal_oriented == "L":
                # Hyperparameters
                alpha = 0  # PCK_TX with success
                beta = -0.8  # RTX discarded
                delta = 0.2  # PCK discarded max rtx
                gamma = 0       # PCK discarded full queue

            reward = ( alpha * self.Q_current_state[0] -
                          beta * self.Q_current_state[1] +
                          delta * (1 if self.Q_dropped_pcks_per_step_counter == 0 else 0) +
                          gamma * (1 if self.Q_not_added_pcks_per_step_counter == 0 else 0)  # Penalize full queue
                        )

        return reward

    def reward_computation_for_only_W(self, input_goal_oriented: str = None):
        """
        Computes reward based on contention window-related (W) performance metrics only.

        Args:
            input_goal_oriented (str): Reward mode ("S&L", "S", or "L").
                                       Determines which set of hyperparameters to use (Default value = None).

        Returns:
            float: Computed reward value based on contention window performance.
        """
        if self.W_forbidden_action:
            self.W_forbidden_action = False  # Reset flag
            reward = -1.0  # Strong penalty for forbidden action

        else:
            if input_goal_oriented == "S&L":
                # Hyperparameters
                alpha = 0.8     # PCK_TX with success
                beta = 0.3      # RTX discarded
                delta = 0.2     # PCK discarded max rtx
                gamma = 0       # PCK discarded full queue

            elif input_goal_oriented == "S":
                # Hyperparameters
                alpha = 0.5     # PCK_TX with success
                beta = 0      # RTX discarded
                delta = 0.25     # PCK discarded max rtx
                gamma = 0.25       # PCK discarded full queue

            elif input_goal_oriented == "L":
                # Hyperparameters
                alpha = 0       # PCK_TX with success
                beta = -0.8     # RTX discarded
                delta = 0.2     # PCK discarded max rtx
                gamma = 0       # PCK discarded full queue

            reward = ( alpha * self.W_current_state[0] -
                          beta * self.W_current_state[1] +
                          delta * (1 if self.W_dropped_pcks_per_step_counter == 0 else 0) +
                          gamma * (1 if self.Q_not_added_pcks_per_step_counter == 0 else 0)  # Penalize full queue
                       )# )

        return reward

    def reward_computation_for_Q_and_W(self):
        """
        Computes reward using both queue (Q) and contention window (W) parameters setting.
        Considers transmission success, latency, packet drops, and efficiency.

        Returns:
            float: Combined reward value considering both Q and W objectives.
        """
        if self.Q_and_W_forbidden_action:
            self.Q_and_W_forbidden_action = False  # Reset flag
            reward = -1.0  # Strong penalty for forbidden action

        else:
            # Extract state variables
            ack_success_ratio = self.Q_and_W_current_state[0]  # p_mac
            buffer_norm = self.Q_and_W_current_state[1]  # Q_norm
            contention_norm = self.Q_and_W_current_state[2]  # W_norm
            total_success_ratio = self.Q_and_W_current_state[3]  # T_success
            packet_drop_rate = self.Q_and_W_current_state[4]  # P_drop
            avg_latency_norm = self.Q_and_W_current_state[5]  # L_norm
            transmission_utilization = self.Q_and_W_current_state[6]  # T_util

            # Reward computation weights S = 1.905, tx = 5, p_mac = 0.9008
            alpha = 0.4  # Weight for ACK success ratio
            beta = 0.3  # Weight for total successful transmissions
            gamma = 0.2  # Penalty for packet drops
            delta = 0.2  # Penalty for latency
            eta = 0.2  # Reward for transmission efficiency

            # Compute reward
            reward = (
                    alpha * ack_success_ratio +  # Encourage ACK success
                    beta * total_success_ratio -  # Encourage total successful transmissions
                    gamma * packet_drop_rate -  # Penalize dropped packets
                    delta * avg_latency_norm +  # Penalize high latency
                    eta * transmission_utilization  # Encourage optimal use of transmissions
            )

        self.Q_and_W_saved_state_Q[-1].append(self.Q_and_W_buffer_length)
        self.Q_and_W_saved_state_W[-1].append(self.Q_and_W_contention_window)

        reward = max(reward, -1.0)  # Ensure reward is at least -1.0
        self.append_reward(reward)

        return reward




