import sys
import copy as cp

import numpy as np

from timessim.traffic_models.traffic_model import TrafficModel
from timessim.network.ue_buffer import UeBuffer
from timessim.network.packet import Packet

##################################RL Implementation##################################
from timessim.utils.utils_for_tb_ualoha_with_dqn import ttl_reset, select_input_DRL, compute_normalized_linear_interpolation
from timessim.scheduler.DQN_agent_rl_mesh import PrioritizedReplayBuffer
from timessim.utils.read_inputs import read_inputs

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
##############################End RL Implementation##################################

class Ue(TrafficModel):
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

    # Set the packet info dictionary
    def set_packet_info(self, packet_info: dict):
        self.packet_info_dict.update(packet_info)

    # Add a new packet in the queue and compute the next generation instant if the buffer is not full
    def add_new_packet(self, current_tick: int, input_enable_print: bool = False,
                       input_data_to_be_forwarded_bool: bool = None, input_packet_size_bytes: int = None,
                       input_simulation_tick_duration: int = None, data_rx_from_ue: int = None,
                       packet_id_rx_from_ue: int = None, packet_generated_by_ue: int = None,
                       packet_id_generator: int = None, packet_hop_count: int = None, packet_address: int = None, generation_time: int = None):
        """
            Add a new packet in the queue and compute the next generation instant, if the buffer is not full.
            Otherwise, just discard the packet.
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

    # Set the 3D coordinates of the UE
    def set_coordinates(self, x_input, y_input, z_input):
        self.x = x_input
        self.y = y_input
        self.z = z_input

    # Get the 3D coordinates of the UE
    def get_coordinates(self):
        return np.array([self.x, self.y, self.z])

    # Set the next packet generation instant
    def get_next_packet_generation_instant(self):
        return self.t_generation

    # Get the current state of the UE
    def get_state(self):
        return self.state

    # Set the current state of the UE
    def set_state(self, input_state: str):
        self.state = input_state

    # Get the starting tick of data transmission
    def set_start_data_tx(self, input_start_data_tx: int):
        self.t_start_data_tx = input_start_data_tx

    # Set the starting tick of data transmission
    def get_start_data_tx(self):
        return self.t_start_data_tx

    # Set the starting tick of ACK transmission
    def set_start_ack_tx(self, input_start_ack_tx: int):
        self.t_start_ack_tx = input_start_ack_tx

    # Get the starting tick of ACK transmission
    def get_start_ack_tx(self):
        return self.t_start_ack_tx

    # Set the ending tick of ACK transmission
    def set_end_ack_tx(self, input_end_ack_tx: int):
        self.t_end_ack_tx = input_end_ack_tx

    # Get the ending tick of ACK transmission
    def get_end_ack_tx(self):
        return self.t_end_ack_tx

    # Set the ending tick of data transmission
    def set_end_data_tx(self, input_end_data_tx: int):
        self.t_end_data_tx = input_end_data_tx

    # Get the ending tick of data transmission
    def get_end_data_tx(self):
        return self.t_end_data_tx

    # Set the current state duration
    def set_state_duration(self, input_ticks: int):
        self.t_state = input_ticks

    # Update the current state duration
    def update_state_duration(self, input_ticks: int):
        self.t_state += input_ticks

    # Get the current state duration
    def get_state_duration(self):
        return self.t_state

    # Set the starting tick of the current state
    def set_state_starting_tick(self, input_tick: int):
        self.t_starting_state = input_tick

    # Get the starting tick of the current state
    def get_state_starting_tick(self):
        return self.t_starting_state

    # Set the ending tick of the current state
    def set_state_final_tick(self, input_tick: int):
        self.t_final_state = input_tick

    # Get the ending tick of the current state
    def get_state_final_tick(self):
        return self.t_final_state

    # Get the UE ID
    def get_ue_id(self):
        return self.ue_id

    # Set the UE ID
    def set_ue_id(self, ue_id):
        self.ue_id = ue_id

    # Get bit rate in Gbit/s
    def get_bit_rate_gbits(self):
        return self.bit_rate_gbits

    # Set bit rate in Gbit/s
    def set_bit_rate_gbits(self, input_bit_rate_gbits: float):
        self.bit_rate_gbits = input_bit_rate_gbits

    # Get the data duration in seconds
    def get_data_duration_s(self):
        return self.data_duration_s

    # Set the data duration in seconds
    def set_data_duration_s(self, input_data_duration_s: float):
        self.data_duration_s = input_data_duration_s

    # Get the data duration in ticks
    def get_data_duration_tick(self):
        return self.data_duration_tick

    # Set the data duration in ticks
    def set_data_duration_tick(self, input_data_duration_tick: float):
        self.data_duration_tick = input_data_duration_tick

    """
        Methods referring to the next packet that will be added in the queue at tick "t_generation" 
    """

    # Set the new packet arrival time
    def set_new_packet_arrival_time(self, input_arrival_time: int):
        self.packet.set_arrival_time(input_arrival_time)

    # Set the new packet ID
    def set_new_packet_ue_id(self):
        self.packet.set_ue_id(self.ue_id)

    # Set the new packet size
    def set_new_packet_size(self, packet_size: int):
        self.packet.set_size(packet_size)

    # Get the new packet size
    def get_new_packet_size(self):
        return self.packet.get_size()

    # Set the new packet to be sent
    def set_new_packet_data_to_be_sent(self, input_data_to_be_sent: int):
        self.packet.set_data_to_be_sent(input_data_to_be_sent)

    # Get the new packet to be sent
    def get_new_packet_data_to_be_sent(self):
        return self.packet.get_data_to_be_sent()

    # Set the new packet size on
    def set_new_packet_size_on(self, packet_size: int):
        self.packet.set_size_on(packet_size)

    # Get the new packet size on
    def get_new_packet_size_on(self):
        return self.packet.get_size_on()

    # Get the new packet to be sent on
    def get_new_packet_data_to_be_sent_on(self):
        return self.packet.get_data_to_be_sent()

    # Set the new packet size standby
    def set_new_packet_size_standby(self, packet_size: int):
        self.packet.set_size_standby(packet_size)

    # Get the new packet size standby
    def get_new_packet_size_standby(self):
        return self.packet.get_size_standby()

    """
        End of these methods
    """

    # Get the last packet ID added in the queue
    def get_last_packet_id(self):
        return self.packet.get_id() - 1

    # Get the list of packets in the queue
    def get_updated_packet_list(self):
        return self.ul_buffer.get_packet_list()

    # Get the number of packets in the queue
    def get_packet_list_size(self):
        return self.ul_buffer.get_n_packets()

    # Get the UE buffer
    def get_ul_buffer(self):
        return self.ul_buffer

    # Set the carrier frequency in GHz
    def set_carrier_frequency(self, input_carrier_frequency_ghz: float):
        self.carrier_frequency_ghz = input_carrier_frequency_ghz

    # Get the carrier frequency in GHz
    def get_carrier_frequency_ghz(self):
        return self.carrier_frequency_ghz

    # Set the channel
    def set_channel(self, input_channel: int):
        self.channel = input_channel

    # Get the channel
    def get_channel(self):
        return self.channel

    # Set the UE saved state
    def set_ue_saved_state(self, input_ue_saved_state: str):
        self.ue_saved_state = input_ue_saved_state

    # Get the UE saved state
    def get_ue_saved_state(self):
        return self.ue_saved_state

    # Get the first packet transmission number
    def get_current_packet_transmission_number(self):  # Current packet is the one that the UE is transmitting
        return self.ul_buffer.get_first_packet().get_transmission_counter()

    # Increment the first packet transmission number
    def increment_current_packet_transmission_number(self):
        self.ul_buffer.get_first_packet().increment_transmission_counter()

    # Get the first packet priority
    def get_current_packet_priority(self):
        return self.ul_buffer.get_first_packet().get_priority()

    # Get the first packet ID
    def get_current_packet_id(self):
        return self.ul_buffer.get_first_packet().get_id()

    # Get the LOS condition with the BS
    def get_los_condition(self):  # NOTE: This is something the UE may not know in practise,
        # so do not use it to change its behavior during simulation
        return self.is_in_los

    # Set the LOS condition with the BS
    def set_los_condition(self, is_in_los: bool):
        self.is_in_los = is_in_los

    # Set the channel condition with the BS
    def set_channel_condition_with_bs(self, is_low_channel_condition_bool: bool):
        self.is_low_channel_condition_with_bs = is_low_channel_condition_bool

    # Get the channel condition with the BS
    def get_channel_condition_with_bs(self):
        return self.is_low_channel_condition_with_bs

    # Get the LOS condition with the UE
    def set_los_condition_ue_ue(self, is_in_los: bool):
        self.is_in_los_ues.append(is_in_los)

    # Set the channel condition with the UE
    def set_channel_condition_with_ue(self, is_low_channel_condition_bool: bool):
        self.is_low_channel_condition_with_ues.append(is_low_channel_condition_bool)

    # Get the channel condition with the UE
    def get_channel_condition_with_ue(self, ue_index: int):
        return self.is_low_channel_condition_with_ues[ue_index]

    # Get the distance from the BS in meters
    def get_distance_from_bs_m(self):  # NOTE: This is something the UE may not know in practise,
        # so do not use it to change its behavior during simulation
        return self.distance_from_bs_m

    # Set the distance from the BS in meters
    def set_distance_from_bs(self, ue_bs_distance_m: float):
        self.distance_from_bs_m = ue_bs_distance_m

    # Get the discarded packets percentage
    def get_packet_discarded_perc(self):
        return self.discarded_packets / self.n_generated_packets

    # Get the total packet generated by the UE
    def get_tot_packet(self):
        return self.n_generated_packets

    # Get the number of packets discarded by the UE
    def get_discarded(self):
        return self.n_discarded_packets

     # Get the number of antennas of the tr-rx
    def get_n_antennas(self):
        return self.transceiver_params.get("Number of antennas")

    # Get the transmission power of the tr-rx
    def get_tx_power(self):
        return self.transceiver_params.get("Transmit power")

    # Get the TX antenna efficiency
    def get_tx_antenna_efficiency(self):
        return self.transceiver_params.get("Antenna efficiency")

    # Get the noise figure in dB
    def get_noise_figure_db(self):
        return self.transceiver_params.get("Noise figure")

    # Set the mac success bool
    def set_mac_success_bool(self, input_mac_success: bool):
        self.mac_success = input_mac_success

    # Get the mac success bool
    def get_mac_success_bool(self):
        return self.mac_success

    # Update the number of transmissions for the first packet and all those to be forwarded
    def update_num_tx(self, input_packet_id: int = None, input_enable_print: bool = False):
        # Increment by 1 the number of Txs for the first packet and all those to be forwarded
        if input_packet_id is not None:
            packet = self.ul_buffer.get_packet_by_id(packet_id=input_packet_id)
            current_num_tx = packet.get_num_tx()
            packet.set_num_tx(input_num_tx=current_num_tx + 1, input_enable_print=input_enable_print)
        else:
            for packet in self.get_updated_packet_list():
                current_num_tx = packet.get_num_tx()
                packet.set_num_tx(input_num_tx=current_num_tx + 1, input_enable_print=input_enable_print)

    # Check whether there are some packets that have reached the maximum number of retransmissions and remove them
    def check_num_tx(self):
        # Checks whether there are some packets that have reached the maximum number of retransmissions and remove them
        # Returns True only if either the first packet or those that have to be forwarded has not reached that limit
        data_to_transmit = False
        packets_list = list()
        for packet in self.get_updated_packet_list():
            # if packet.get_data_to_be_forwarded_bool() or packet.get_id() == self.ul_buffer.get_first_packet().get_id():
            if packet.get_num_tx() <= packet.get_max_n_retx() + 1:
                data_to_transmit = True
                # break
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
            # print("UE ", self.get_ue_id(), " has removed packet: ", packet.get_id())
            self.packets_sent -= 1

        return data_to_transmit

    # Update the number of data packets successfully received by the UE
    def update_n_data_rx(self, input_tick: int, input_enable_print: bool = False):
        self.n_data_rx += 1
        if input_enable_print:
            print('UE ', self.get_ue_id(), ' has updated n_data_rx to ',
                  self.get_n_data_rx(), 'at t = ', input_tick, ' because it has transmitted packet ',
                  self.packet.get_id(), ' with success')

    # Set the number of data packets transmitted by the UE
    def set_n_data_tx(self, input_n_data_tx: int):
        self.n_data_tx = input_n_data_tx

    # Update the number of data packets transmitted by the UE
    def update_n_data_tx(self, input_enable_print: bool = False):
        # Update n_data_tx for each packet in the queue
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

    # Get the number of data packets transmitted by the UE
    def get_n_data_tx(self):
        return self.n_data_tx

    # Set the number of data packets received by the UE
    def set_n_data_rx(self, input_n_data_rx: int):
        self.n_data_rx = input_n_data_rx

    # Get the number of data packets received by the UE
    def get_n_data_rx(self):
        return self.n_data_rx

    # Set the number of data packets discarded by the UE
    def set_n_data_discarded(self, input_n_data_discarded: int):
        self.n_data_discarded = input_n_data_discarded

    # Update the number of data packets discarded by the UE
    def update_n_data_discarded(self):
        self.n_data_discarded += 1

    # Get the number of data packets discarded by the UE
    def get_n_data_discarded(self):
        return self.n_data_discarded

    # Check whether there is a new packet to be added in the queue, generated by the UE
    def is_there_a_new_data(self, input_current_tick: int, max_n_packets_to_be_forwarded: int):
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

    # Get the number of packets in the queue
    def get_n_packets(self):
        return self.ul_buffer.get_n_packets()

    # Get the traffic type
    def get_traffic_type(self):
        return self.traffic_type

    # Set the next packet generation instant
    def set_t_generation(self, input_t_generation: int):
        self.t_generation = input_t_generation

    # Set the packet ID
    def set_packet_id(self, input_packet_id: int or str):
        self.packet.set_id(input_packet_id=input_packet_id)

    # Set the reception during RREQ reception boolean
    def set_reception_during_rreq_rx_bool(self, input_rreq_rx_bool: bool):
        self.reception_during_rreq_rx_bool = input_rreq_rx_bool

    # Set the reception during BO boolean
    def set_reception_during_bo_bool(self, input_data_rx_bool: bool):
        self.reception_during_bo_bool = input_data_rx_bool

    # Set the reception during WAIT boolean
    def set_reception_during_wait_bool(self, input_data_rx_bool: bool):
        self.reception_during_wait_bool = input_data_rx_bool

    # Get the reception during BO boolean
    def get_reception_during_bo_bool(self):
        return self.reception_during_bo_bool

    # Get the reception during RREQ reception boolean
    def get_reception_during_rreq_rx_bool(self):
        return self.reception_during_rreq_rx_bool

    # Get the reception during WAIT boolean
    def get_reception_during_wait_bool(self):
        return self.reception_during_wait_bool

    # Set the propagation delay to the BS in seconds
    def set_prop_delay_to_bs_s(self, input_prop_delay_to_bs_s: float):
        self.prop_delay_to_bs_s = input_prop_delay_to_bs_s

    # Get the propagation delay to the BS in seconds
    def get_prop_delay_to_bs_s(self):
        return self.prop_delay_to_bs_s

    # Set the propagation delay to the BS in ticks
    def set_prop_delay_to_bs_tick(self, input_prop_delay_to_bs_tick: int):
        self.prop_delay_to_bs_tick = input_prop_delay_to_bs_tick

    # Get the propagation delay to the BS in ticks
    def get_prop_delay_to_bs_tick(self):
        return self.prop_delay_to_bs_tick

    # Add the propagation delay to a specific UE in seconds
    def add_prop_delay_to_ue_s(self, input_ue_id: int, input_prop_delay_to_ue_s: float):
        self.prop_delay_to_ues_s[f'UE_{input_ue_id}'] = input_prop_delay_to_ue_s

    # Get the propagation delay to a specific UE in seconds
    def get_prop_delay_to_ue_s(self, input_ue_id: int):
        if f'UE_{input_ue_id}' in self.prop_delay_to_ues_s:
            return self.prop_delay_to_ues_s[f'UE_{input_ue_id}']
        else:
            sys.exit(f'UE {input_ue_id} not found in prop_delay_to_ues_s of UE {self.ue_id}')

    # Add the propagation delay to a specific UE in ticks
    def add_prop_delay_to_ue_tick(self, input_ue_id: int, input_prop_delay_to_ue_tick: float):
        self.prop_delay_to_ues_tick[f'UE_{input_ue_id}'] = input_prop_delay_to_ue_tick

    # Get the propagation delay to a specific UE in ticks
    def get_prop_delay_to_ue_tick(self, input_ue_id: int):
        if f'UE_{input_ue_id}' in self.prop_delay_to_ues_tick:
            return self.prop_delay_to_ues_tick[f'UE_{input_ue_id}']
        else:
            sys.exit(f'UE {input_ue_id} not found in prop_delay_to_ues_tick of UE {self.ue_id}')

    # Get the packet size in bytes
    def get_packet_size_bytes(self, input_packet_id: int = None):
        if input_packet_id is not None:
            return self.ul_buffer.get_packet_by_id(packet_id=input_packet_id).get_size()
        else:
            return self.ul_buffer.get_first_packet().get_size()

    # Set the maximum number of retransmissions per packet
    def set_max_n_retx_per_packet(self, input_max_n_retx_per_packet: int):
        self.max_n_retx_per_packet = input_max_n_retx_per_packet

    # Get the maximum number of retransmissions per packet
    def get_max_n_retx_per_packet(self):
        return self.max_n_retx_per_packet

    # Set the relay boolean
    def set_relay_bool(self, relay_bool: bool):
        self.relay = relay_bool

    # Get the relay boolean
    def get_relay_bool(self):
        return self.relay

    # Set the ID of the UE from which the packet has been received
    def set_data_rx_from_ue(self, data_rx_from_ue: int):
        self.packet.set_data_rx_from_ue(data_rx_from_ue=data_rx_from_ue)

    # Get the ID of the UE from which this packet has been received
    def get_data_rx_from_ue(self):
        return self.packet.get_data_rx_from_ue()

    # Set the packet ID received from a UE
    def set_packet_id_rx_from_ue(self, packet_id_rx_from_ue: int):
        self.packet.set_packet_id_rx_from_ue(packet_id_rx_from_ue=packet_id_rx_from_ue)

    # Get the packet ID received from a UE
    def get_packet_id_rx_from_ue(self):
        return self.packet.get_packet_id_rx_from_ue()

    # Set the packet id acknowledged at the UE
    def set_ack_packet_id_ue(self, packet_id: int):
        self.packet_id_ack = packet_id

    # Get the packet id acknowledged at the UE
    def get_ack_packet_id_ue(self):
        return self.packet_id_ack

    # Set the boolean for retransmission of packets
    def set_retransmission_packets(self, retransmission_bool: bool):
        self.retransmission_of_packets = retransmission_bool

    # Get the boolean for retransmission of packets
    def get_retransmission_packets(self):
        return self.retransmission_of_packets

    # Set the UE uplink buffer
    def set_ul_buffer(self):
        self.ul_buffer = UeBuffer(max_buffer_size=self.params.get('ue').get('max_buffer_size'))

    # Set the number of packets sent
    def set_packets_sent(self, input_packets_sent: int):
        self.packets_sent = input_packets_sent

    # Get the number of packets sent
    def get_packets_sent(self):
        return self.packets_sent

    # Set the last action
    def set_last_action(self, input_last_action):
        self.last_action = input_last_action

    # Get the last action
    def get_last_action(self):
        return self.last_action

    # Set the temporary observation
    def set_temp_obs(self, input_temp_obs: np.ndarray):
        self.temp_obs = cp.deepcopy(input_temp_obs)

    # Reset the temporary observation
    def reset_temp_obs(self):
        self.temp_obs = np.zeros((5, len(self.neighbour_table)), dtype=np.float32)

    # Update the temporary observation after a successful broadcast
    def set_temp_obs_broadcast(self, input_ack_rx_at_ue_tx_index, input_rx_power,input_bs_seen: int=0):
        self.temp_obs[0][input_ack_rx_at_ue_tx_index] = 1  # neighbour discovered
        self.temp_obs[1][input_ack_rx_at_ue_tx_index] += 1  # count of a successful received ack
        self.temp_obs[2][input_ack_rx_at_ue_tx_index] = input_rx_power  # received power to implement
        self.temp_obs[3][input_ack_rx_at_ue_tx_index] = 0  # neighbour TTL
        self.temp_obs[4][input_ack_rx_at_ue_tx_index] = input_bs_seen  # BS seen

    # Update the observation after a successful data reception
    def set_obs_update(self, input_data_rx_at_ue_tx_index, input_rx_power):
        self.obs[0][input_data_rx_at_ue_tx_index] = 1  # neighbour discovered
        self.obs[2][input_data_rx_at_ue_tx_index] = input_rx_power  # received power to implement
        self.obs[3][input_data_rx_at_ue_tx_index] = 0  # neighbour TTL

    # Get the temporary observation
    def get_temp_obs(self):
        return self.temp_obs

    # Set the observation
    def set_obs(self, input_obs: np.ndarray):
        self.obs = cp.deepcopy(input_obs)

    # Reset the observation
    def reset_obs(self):
        self.obs = np.zeros((5, len(self.neighbour_table)), dtype=np.float32)

    # Get the observation
    def get_obs(self):
        return self.obs

    # Set the old state
    def set_old_state(self, input_old_state):
        self.old_state = cp.deepcopy(input_old_state)

    # Get the old state
    def get_old_state(self):
        return self.old_state

    # Set the reward
    def set_reward(self, input_reward: list):
        self.reward = input_reward

    # Get the reward
    def get_reward(self):
        return self.reward

    # Append a new reward value
    def append_reward(self, input_reward: float):
        self.reward.append(input_reward)

    # Get the last reward value
    def get_last_reward(self):
        return self.reward[-1]

    # Set the simulations reward
    def set_simulations_reward(self, input_simulations_reward: list):
        self.simulations_reward = input_simulations_reward

    # Get the simulations reward
    def get_simulations_reward(self):
        return self.simulations_reward

    # Get the last simulations reward
    def get_last_simulations_reward(self):
        return self.simulations_reward[-1]

    # Append a new simulations reward value
    def append_simulations_reward(self, input_simulations_reward: float):
        self.simulations_reward.append(input_simulations_reward)

    #################### W Methods ####################

    # Set the W reward
    def set_W_reward(self, input_reward: list):
        self.W_reward = input_reward

    # Get the W reward
    def get_W_reward(self):
        return self.W_reward

    # Append a new W reward value
    def append_W_reward(self, input_reward: float):
        self.W_reward.append(input_reward)

    # Get the last W reward value
    def get_last_W_reward(self):
        return self.W_reward[-1]

    # Set the W simulations reward
    def set_W_simulations_reward(self, input_simulations_reward: list):
        self.W_simulations_reward = input_simulations_reward

    # Get the W simulations reward
    def get_W_simulations_reward(self):
        return self.W_simulations_reward

    # Get the last W simulations reward
    def get_last_W_simulations_reward(self):
        return self.W_simulations_reward[-1]

    # Append a new W simulations reward value
    def append_W_simulations_reward(self, input_simulations_reward: float):
        self.W_simulations_reward.append(input_simulations_reward)

    #################### Q Methods ####################

    # Set the Q reward
    def set_Q_reward(self, input_reward: list):
        self.Q_reward = input_reward

    # Get the Q reward
    def get_Q_reward(self):
        return self.Q_reward

    # Append a new Q reward value
    def append_Q_reward(self, input_reward: float):
        self.Q_reward.append(input_reward)

    # Get the last Q reward value
    def get_last_Q_reward(self):
        return self.Q_reward[-1]

    # Set the Q simulations reward
    def set_Q_simulations_reward(self, input_simulations_reward: list):
        self.Q_simulations_reward = input_simulations_reward

    # Get the Q simulations reward
    def get_Q_simulations_reward(self):
        return self.Q_simulations_reward

    # Get the last Q simulations reward
    def get_last_Q_simulations_reward(self):
        return self.Q_simulations_reward[-1]

    # Append a new Q simulations reward value
    def append_Q_simulations_reward(self, input_simulations_reward: float):
        self.Q_simulations_reward.append(input_simulations_reward)

    #################### DNN Methods ####################

    # Set the model
    def set_model(self, input_model):
        self.model = input_model

    # Get the model
    def get_model(self):
        return self.model

    # Set the target model
    def set_target_model(self, input_target_model):
        self.target_model = input_target_model

    # Get the target model
    def get_target_model(self):
        return self.target_model

    # Set the W model
    def set_W_model(self, input_model):
        self.W_model = input_model

    # Get the W model
    def get_W_model(self):
        return self.W_model

    # Set the W target model
    def set_W_target_model(self, input_target_model):
        self.W_target_model = input_target_model

    # Get the W target model
    def get_W_target_model(self):
        return self.W_target_model

    # Set the Q model
    def set_Q_model(self, input_model):
        self.Q_model = input_model

    # Get the Q model
    def get_Q_model(self):
        return self.Q_model

    # Set the Q target model
    def set_Q_target_model(self, input_target_model):
        self.Q_target_model = input_target_model

    # Get the Q target model
    def get_Q_target_model(self):
        return self.Q_target_model

    # Set the replay buffer
    def set_replay_buffer(self, input_replay_buffer):
        self.replay_buffer = input_replay_buffer

    # Get the replay buffer
    def get_replay_buffer(self):
        return self.replay_buffer

    # Get the last replay buffer instance
    def get_last_replay_instance(self):
        return self.replay_buffer[-1]

    # Append a new instance to the replay buffer
    def append_replay_buffer(self, input_replay_buffer_instance):
        self.replay_buffer.append(input_replay_buffer_instance)

    # Remove the last instance from the replay buffer
    def drop_last_replay_instance(self):
        self.replay_buffer.pop()

    # Set the W replay buffer
    def set_W_replay_buffer(self, input_replay_buffer):
        self.W_replay_buffer = input_replay_buffer

    # Get the W replay buffer
    def get_W_replay_buffer(self):
        return self.W_replay_buffer

    # Get the last W replay buffer instance
    def get_last_W_replay_instance(self):
        return self.W_replay_buffer[-1]

    # Append a new instance to the W replay buffer
    def append_W_replay_buffer(self, input_replay_buffer_instance):
        self.W_replay_buffer.append(input_replay_buffer_instance)

    # Remove the last instance from the W replay buffer
    def drop_last_W_replay_instance(self):
        self.W_replay_buffer.pop()

    # Set the Q replay buffer
    def set_Q_replay_buffer(self, input_replay_buffer):
        self.Q_replay_buffer = input_replay_buffer

    # Get the Q replay buffer
    def get_Q_replay_buffer(self):
        return self.Q_replay_buffer

    # Get the last Q replay buffer instance
    def get_last_Q_replay_instance(self):
        return self.Q_replay_buffer[-1]

    # Append a new instance to the Q replay buffer
    def append_Q_replay_buffer(self, input_replay_buffer_instance):
        self.Q_replay_buffer.append(input_replay_buffer_instance)

    # Remove the last instance from the Q replay buffer
    def drop_last_Q_replay_instance(self):
        self.Q_replay_buffer.pop()

    #################### Other Methods ####################

    # Set the neighbour table
    def set_neighbour_table(self, input_neighbour_table: list):
        self.neighbour_table = input_neighbour_table

    # Get the neighbour table
    def get_neighbour_table(self):
        return self.neighbour_table

    # Set the environment
    def set_env(self, input_env):
        self.env = input_env

    # Get the environment
    def get_env(self):
        return self.env

    ############### DNN Hyperparameters Methods ###############

    # Set epsilon, for epsilon-greedy policy
    def set_epsilon(self, input_epsilon: float):
        self.epsilon = input_epsilon

    # Get epsilon
    def get_epsilon(self):
        return self.epsilon

    # Set the best weights of the model
    def set_best_weights(self, input_best_weights):
        self.best_weights = input_best_weights

    # get the best weights of the model
    def get_best_weights(self):
        return self.best_weights

    # Set the best score of the model
    def set_best_score(self, input_best_score: float):
        self.best_score = input_best_score

    # Get the best score of the model
    def get_best_score(self):
        return self.best_score

    ############### Actions Methods ###############

    # Set the unicast RX address
    def set_unicast_rx_address(self, input_unicast_rx_address):
        self.unicast_rx_address = input_unicast_rx_address

    # Get the unicast RX address
    def get_unicast_rx_address(self):
        return self.unicast_rx_address

    # Set the unicast RX index
    def set_unicast_rx_index(self, input_unicast_rx_index):
        self.unicast_rx_index = input_unicast_rx_index

    # Get the unicast RX index
    def get_unicast_rx_index(self):
        return self.unicast_rx_index

    # Set the broadcast boolean
    def set_broadcast_bool(self, input_broadcast_bool: bool):
        self.broadcast_bool = input_broadcast_bool

    # Get the broadcast boolean
    def get_broadcast_bool(self):
        return self.broadcast_bool

    # Set the action list
    def set_action_list(self, input_action_list: list):
        self.action_list = input_action_list

    # Get the action list
    def get_action_list(self):
        return self.action_list

    # Set the W action list
    def set_W_action_list(self, input_action_list: list):
        self.W_action_list = input_action_list

    # Get the W action list
    def get_W_action_list(self):
        return self.W_action_list

    # Set the Q action list
    def set_Q_action_list(self, input_action_list: list):
        self.Q_action_list = input_action_list

    # Get the Q action list
    def get_Q_action_list(self):
        return self.Q_action_list

    # Append a new action to the action list
    def append_action_list(self, input_action):
        self.action_list.append(input_action)

    # Append a new action to the W action list
    def append_W_action_list(self, input_action):
        self.W_action_list.append(input_action)

    # Append a new action to the Q action list
    def append_Q_action_list(self, input_action):
        self.Q_action_list.append(input_action)

    # Set the success action list
    def set_success_action_list(self, input_success_action_list: list):
        self.success_action_list = input_success_action_list

    # Get the success action list
    def get_success_action_list(self):
        return self.success_action_list

    # Append a new action to the success action list
    def append_success_action_list(self, input_success_action):
        self.success_action_list.append(input_success_action)

    # Set the actions per simulation
    def set_actions_per_simulation(self, input_actions_per_simulation: list):
        self.actions_per_simulation = input_actions_per_simulation

    # Get the actions per simulation
    def get_actions_per_simulation(self):
        return self.actions_per_simulation

    # Append the number of each action taken in the last simulation to the actions_per_simulation list
    def append_actions_per_simulation(self):
        self.actions_per_simulation[0].append(self.action_list.count(0))
        self.actions_per_simulation[1].append(self.action_list.count(1))
        self.actions_per_simulation[2].append(self.action_list.count(2))
        self.actions_per_simulation[3].append(self.action_list.count(3))

    # Set the success actions per simulation
    def set_success_actions_per_simulation(self, input_success_actions_per_simulation: list):
        self.success_actions_per_simulation = input_success_actions_per_simulation

    # Get the success actions per simulation
    def get_success_actions_per_simulation(self):
        return self.success_actions_per_simulation

    # Append the number of each successful action taken in the last simulation to the success_actions_per_simulation list
    def append_success_actions_per_simulation(self):
        self.success_actions_per_simulation[0].append(self.success_action_list.count(0))
        self.success_actions_per_simulation[1].append(self.success_action_list.count(1))

    # Set the TX broadcast list
    def set_tx_broad_list(self, input_tx_broad_list: list):
        self.tx_broad_list = input_tx_broad_list

    # Get the TX broadcast list
    def get_tx_broad_list(self):
        return self.tx_broad_list

    # Set the data discard boolean
    def set_data_discard_bool(self, input_data_discard_bool: bool):
        self.data_discard_bool = input_data_discard_bool

    # Get the data discard boolean
    def get_data_discard_bool(self):
        return self.data_discard_bool

    # Set the saved coordinates
    def set_saved_coordinates(self, input_saved_coordinates):
        self.saved_coordinates = input_saved_coordinates

    # Get the saved coordinates
    def get_saved_coordinates(self):
        return self.saved_coordinates

    # Reset all the actions since last TTL reset
    def reset_complete_actions_since_last_ttl_reset(self, input_neighbour_number:int):
        self.actions_since_last_ttl_reset = [0 for _ in range(input_neighbour_number)]

    # Reset the actions since last TTL reset for a specific neighbour
    def reset_actions_since_last_ttl_reset(self, input_neighbour_index:int):
        self.actions_since_last_ttl_reset[input_neighbour_index] = 0

    # Increment the actions since last TTL reset for a specific neighbour
    def increment_actions_since_last_ttl_reset(self, input_neighbour_index:int):
        self.actions_since_last_ttl_reset[input_neighbour_index] += 1

    # Increment all the actions since last TTL reset
    def increment_all_actions_since_last_ttl_reset(self):
        for i in range(len(self.actions_since_last_ttl_reset)):
            self.actions_since_last_ttl_reset[i] += 1

    # Check whether there are some packets that have reached the maximum number of retransmissions and remove them
    def check_num_tx_RL(self):
        # Checks whether there are some packets that have reached the maximum number of retransmissions and remove them
        # Returns True only if either the first packet or those that have to be forwarded has not reached that limit
        data_to_transmit = False
        packets_list = list()
        for packet in self.get_updated_packet_list():
            if packet.get_data_to_be_forwarded_bool() or packet.get_id() == self.ul_buffer.get_first_packet().get_id():
                if packet.get_num_tx() <= packet.get_max_n_retx() + 1:
                    data_to_transmit = True
                    # break
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

    # Check whether there are some packets that have reached the maximum number of retransmissions
    def check_rtx(self):
        # Checks whether there are some packets that have reached the maximum number of retransmissions
        # Returns True only if either the first packet or those that have to be forwarded has not reached that limit
        data_to_transmit = False
        for packet in self.get_updated_packet_list():
            if packet.get_data_to_be_forwarded_bool() or packet.get_id() == self.ul_buffer.get_first_packet().get_id():
                if packet.get_num_tx() <= packet.get_max_n_retx() + 1:
                    data_to_transmit = True
                    # break

        return data_to_transmit

    # Check whether there are some packets generated by the UE in the queue
    def check_generated_packet_present(self):
        for packet in self.get_updated_packet_list():
            if packet.get_data_to_be_forwarded_bool() is False:
                return True
        return False

    # Check whether there are some packets to be removed from the queue
    def check_remove_packet(self, input_enable_print=False):
        for txs in self.packets_to_be_removed.keys():
            for packet_id in self.packets_to_be_removed[txs]:
                for buffer_packet in self.ul_buffer.buffer_packet_list:
                    if packet_id == buffer_packet.get_id():
                        self.remove_packet(packet_id=packet_id, input_enable_print=input_enable_print)
                        self.packets_sent -= 1
                        # Increase the counter for the acks received
                        if self.Q_and_W_enabled:
                            self.Q_and_W_acks_rx_per_step_counter += 1
                        #ue.packet_id_success = packet_id

            self.packets_to_be_removed[txs] = []

    # Update the neighbours table after a successful unicast
    def update_neighbours_forwarding(self, input_rx_power, input_tx_str, input_bs_seen):
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

    # Check the action packet ID for packets that have reached the maximum number of retransmissions
    def check_action_packet_id(self):
        packet_to_discard = False
        for packet in self.get_updated_packet_list():
            if packet.get_id() == self.action_packet_id:
                if packet.get_num_tx() > packet.get_max_n_retx() + 1:
                    packet_to_discard = True
                break

        return packet_to_discard

    # Check whether the action packet ID is still present in the queue
    def check_present_action_packet_id(self):
        action_packet_present = False
        for packet in self.get_updated_packet_list():
            if packet.get_id() == self.action_packet_id:
                # if packet.get_num_tx() == 5:
                action_packet_present = True
                break

        return action_packet_present

    # Handling of a unicast failure (Old RL version)
    def unicast_handling_failure_v2(self, input_ttl, input_unicast_ampl_factor_no_ack, input_energy_factor,
                                    input_max_n_retx_per_packet):
        self.append_action_list(input_action=0)
        tx_index = self.get_unicast_rx_index()
        self.set_old_state(input_old_state=self.get_obs())
        self.obs[3][tx_index] += 1
        if self.obs[3][tx_index] > input_ttl:
            self.obs = ttl_reset(self.obs, tx_index)
            self.reset_actions_since_last_ttl_reset(tx_index)

        reward = input_unicast_ampl_factor_no_ack - input_energy_factor * input_max_n_retx_per_packet
        # print("unicast failure reward:",reward)
        self.append_reward(reward)
        if not DDQN_new_state:
            self.append_replay_buffer(
                input_replay_buffer_instance=[self.old_state[0], self.get_last_action(), reward, self.obs[0],
                                              False])  # modificare condizione done
        else:
            if not self.Rainbow_DQN:
                self.append_replay_buffer(
                    input_replay_buffer_instance=[self.DRL_state, self.get_last_action(), reward,
                                                  select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                                  False])  # modificare condizione done
            else:
                self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                        select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                        False), priority= 1.0)


        # Uncomment with proper working flow of the simulation
        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    # Handling of a broadcast failure (Old RL version)
    def broadcast_handling_failure_v2(self, input_ttl, input_broadcast_ampl_factor_change,
                                      input_broadcast_ampl_factor_no_change, input_energy_factor,
                                      input_max_n_retx_per_packet):
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

    # unicast success handling (Old RL version)
    def unicast_handling_v2(self, input_rx_power, input_unicast_ampl_factor_ack, input_energy_factor, input_bs_seen: int=0):
        self.set_old_state(input_old_state=self.get_obs())
        self.append_action_list(input_action=0)

        considered_packet = None
        for packet in self.copy_buffer_packet_list:
            if packet.get_id() == self.action_packet_id:
                considered_packet = packet
                break

        if considered_packet == None:
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
                                                  select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                                  False])
            else:
                self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                        select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                        False), priority= 1.0)

        self.append_success_action_list(input_success_action=self.get_last_action())

        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    # unicast success handling without neighbour update (Old RL version)
    def unicast_handling_v2_no_neighbour_update(self, input_unicast_ampl_factor_ack, input_energy_factor):
        self.set_old_state(input_old_state=self.get_obs())
        self.append_action_list(input_action=0)

        considered_packet = None
        for packet in self.copy_buffer_packet_list:
            if packet.get_id() == self.action_packet_id:
                considered_packet = packet
                break

        if considered_packet == None:
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
                                                  select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                                  False])
            else:
                self.replay_buffer.add((self.DRL_state, self.get_last_action(), reward,
                                        select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
                                        False), priority= 1.0)

        self.append_success_action_list(input_success_action=self.get_last_action())

        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.set_old_state(input_old_state=None)
        self.new_action_bool = True

    # broadcast success handling (Old RL version)
    def broadcast_handling_v2(self, input_ttl, input_broadcast_ampl_factor_change,
                              input_broadcast_ampl_factor_no_change, input_energy_factor):
        self.set_old_state(input_old_state=self.get_obs())
        self.append_action_list(input_action=1)

        considered_packet = None
        for packet in self.copy_buffer_packet_list:
            if packet.get_id() == self.action_packet_id:
                considered_packet = packet
                break

        if considered_packet == None:
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
                                                      select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
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
                                                      select_input_DRL(self.obs[1], self.obs[2], DRL_input_nodes_number, DRL_input_type_state, self.actions_since_last_ttl_reset),
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


    # Handling of a unicast failure without reward
    def unicast_handling_failure_no_reward(self, input_ttl):

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

    # Handling of a broadcast failure without reward
    def broadcast_handling_failure_no_reward(self, input_ttl):

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

    # Handling of a unicast success without reward
    def unicast_handling_no_reward(self, input_rx_power, input_reset_vars, input_bs_seen: int=0):

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

    # Handling of a broadcast success without reward
    def broadcast_handling_no_reward(self, input_ttl):

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

    # Update the neighbour table after a successful unicast tx
    def update_neighbor_table_unicast_success(self, input_rx_power, input_bs_seen = 0):
        self.obs[0][self.get_unicast_rx_index()] = 1
        self.obs[1][self.get_unicast_rx_index()] += 1
        self.obs[2][self.get_unicast_rx_index()] = input_rx_power
        self.obs[3][self.get_unicast_rx_index()] = 0
        self.obs[4][self.get_unicast_rx_index()] = input_bs_seen

    # Handling of a unicast success without reward and without neighbour update
    def unicast_handling_no_reward_no_neighbor_update(self):
        self.set_last_action(input_last_action=None)
        self.set_unicast_rx_index(input_unicast_rx_index=None)
        self.set_unicast_rx_address(input_unicast_rx_address=None)
        self.new_action_bool = True

    # Reward computation for W only
    def reward_computation_for_only_Q(self, input_goal_oriented:str = None):

        # Forbidden action check
        if self.Q_forbidden_action:
            self.Q_forbidden_action = False  # Reset flag
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

    # Reward computation for W only
    def reward_computation_for_only_W(self, input_goal_oriented:str = None):

        # Forbidden action check
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

    # Reward computation for Q and W
    def reward_computation_for_Q_and_W(self):

        # Forbidden action check
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



