import operator
import numpy as np

from multi_hop_industrial_simulator.network.packet import Packet


class BsBuffer:
    """
        This class implements a BS buffer, i.e., a collection of packets.
    """

    def __init__(self):
        self.buffer_packet_list = []  # Contains all generated and not yet sent packets
        self.buffer_packet_id_set = set()  # Contains the ids of the packets in 'buffer_packet_list'
        self.sent_packets_latency = []
        self.pending_packets_list = []
        self.pending_packet_id_set = set()  # Contains the ids of the packets in 'pending_packet_list'

        self.buffer_size = 0  # Number of bits in the buffer
        self.n_packets = 0  # Number of packets in the buffer
        self.pending = False
        self.max_buffer_size = 10000000000000000000000  # bytes

    def add_packet(self, packet: Packet):
        """
            Add a new packet to the buffer, updating the buffer size, if the buffer is not full
        """
        if self.buffer_size + packet.packet_size < self.max_buffer_size:
            self.buffer_packet_list.append(packet)
            self.buffer_packet_id_set.add(packet.get_id())
            self.buffer_size += packet.packet_size
            print("Packet size added to BS buffer: ", packet.packet_size)
            self.n_packets += 1
            return True
        else:
            return False
