import operator

from multi_hop_industrial_simulator.network.packet import Packet


class UeBuffer:
    """This class implements a user buffer, i.e., a collection of packets."""

    def __init__(self, max_buffer_size: int):
        self.buffer_packet_list = []  # Contains all generated and not yet sent packets
        self.buffer_packet_id_set = set()  # Contains the ids of the packets in 'buffer_packet_list'
        self.pending_packets_list = []
        self.pending_packet_id_set = set()  # Contains the ids of the packets in 'pending_packet_list'

        self.buffer_size = 0  # Number of bits in the buffer
        self.n_packets = 0  # Number of packets in the buffer
        self.pending = False
        self.max_buffer_size = max_buffer_size  # bytes

    def add_packet(self, packet: Packet):
        """
        Add a new packet to the buffer, updating the buffer size, if the buffer is not full.

        Args:
            packet (Packet): The packet object to add to the buffer.

        Returns:
            bool: True if the packet was successfully added, False if the buffer is full.
        """
        if self.buffer_size + packet.packet_size < self.max_buffer_size:
            self.buffer_packet_list.append(packet)
            self.buffer_packet_id_set.add(packet.get_id())
            self.buffer_size += packet.packet_size
            self.n_packets += 1
            return True
        else:
            return False

    def remove_data(self, packet_id: int = None):
        """
        Remove a packet from the buffer by its ID and update buffer statistics.

        Args:
            packet_id (int, optional): The ID of the packet to remove. Defaults to None.

        Returns:
            None
        """
        idx_packet = self.find_packet_by_id(packet_id)
        self.buffer_size -= self.buffer_packet_list[idx_packet].get_size()
        self.n_packets -= 1
        self.buffer_packet_id_set.remove(packet_id)
        del self.buffer_packet_list[idx_packet]


    def order_buffer(self, priority_metric):
        """
        Order the packets in the buffer according to a given priority metric.

        Args:
            priority_metric (str): The packet attribute name to sort by.

        Returns:
            None
        """
        self.buffer_packet_list = sorted(
            self.buffer_packet_list, reverse=False, key=operator.attrgetter(priority_metric)
        )

    def find_packet_by_id(self, packet_id: int):
        """
        Find the index of a packet in the buffer by its ID.

        Args:
            packet_id (int): The ID of the packet to find.

        Returns:
            int: Index of the packet in the buffer list if found, otherwise None.
        """
        for idx in range(self.n_packets):
            if self.buffer_packet_list[idx].get_id() == packet_id:
                return idx

    def get_packet_by_id(self, packet_id: int):
        """
        Retrieve a packet object from the buffer by its ID.

        Args:
            packet_id (int): The ID of the packet to retrieve.

        Returns:
            Packet: The packet object with the specified ID.
        """
        return self.buffer_packet_list[self.find_packet_by_id(packet_id)]

    def get_first_packet(self):
        """
        Get the first packet currently stored in the buffer.

        Returns:
            Packet: The first packet in the buffer.
        """
        return self.buffer_packet_list[0]

    def get_last_packet(self):
        """
        Get the last packet currently stored in the buffer.

        Returns:
            Packet: The last packet in the buffer.
        """
        return self.buffer_packet_list[-1]

    def get_buffer_size(self):
        """
        Get the current total buffer size in Mb (or bytes).

        Returns:
            float: The total size of all packets in the buffer.
        """
        return self.buffer_size

    def get_n_packets(self):
        """
        Get the number of packets currently stored in the buffer.

        Returns:
            int: The number of packets in the buffer.
        """
        return self.n_packets

    def get_packet_list(self):
        """
        Get the list of all packets currently in the buffer.

        Returns:
            list[Packet]: List of packet objects.
        """
        return self.buffer_packet_list

    def get_first_packet_id(self):
        """
        Get the ID of the first packet currently in the buffer.

        Returns:
            int: Packet ID of the first packet.
        """
        return self.buffer_packet_list[0].get_id()

    def get_last_packet_id(self):
        """
        Get the ID of the last packet currently in the buffer.

        Returns:
            int: Packet ID of the last packet.
        """
        return self.buffer_packet_list[-1].get_id()

    def remove_old_data(self):
        """
        Remove packets from the buffer that were not transmitted (e.g., due to learning or scheduling errors).

        Args:
            None

        Returns:
            int: Number of packets removed from the buffer.
        """
        packets_removed = list()
        for p_id in self.buffer_packet_id_set:
            for packet in self.buffer_packet_list:
                if packet.get_id() == p_id:
                    del self.buffer_packet_list[self.find_packet_by_id(packet.get_id())]
                    packets_removed.append(packet.get_id())
                    self.buffer_size -= packet.get_data_to_be_sent()
                    if self.buffer_size < 0:
                        self.buffer_size = 0
                    self.n_packets -= 1
        n_discarded_packets = len(packets_removed)
        for pack_id in packets_removed:
            self.buffer_packet_id_set.remove(pack_id)
        return n_discarded_packets

    def is_there_any_data(self):
        """
        Check whether there is at least one packet stored in the buffer.

        Returns:
            bool: True if at least one packet is in the buffer, False otherwise.
        """
        if self.get_n_packets() > 0:
            return True
        else:
            return False
