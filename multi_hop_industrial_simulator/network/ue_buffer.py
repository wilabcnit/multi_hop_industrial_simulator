import operator

from timessim.network.packet import Packet


class UeBuffer:
    """
        This class implements a user buffer, i.e., a collection of packets.
    """

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
            Add a new packet to the buffer, updating the buffer size, if the buffer is not full
        """
        if self.buffer_size + packet.packet_size < self.max_buffer_size:
            self.buffer_packet_list.append(packet)
            self.buffer_packet_id_set.add(packet.get_id())
            self.buffer_size += packet.packet_size
            # print("Packet size added: ", packet.packet_size)
            self.n_packets += 1
            return True
        else:
            return False
    # Remove the DATA from the UE buffer, finding the packet to be removed by ID
    def remove_data(self, packet_id: int = None):
        idx_packet = self.find_packet_by_id(packet_id)
        # Update buffer size and number of total packets
        self.buffer_size -= self.buffer_packet_list[idx_packet].get_size()
        self.n_packets -= 1
        self.buffer_packet_id_set.remove(packet_id)
        del self.buffer_packet_list[idx_packet]

    # remove
    def schedule_data(self, priority_metric: str = None, packet_id: int = None, tx_size: float = None, time=None):
        """
            Schedule tx_size Mb or specific packet_id packet from this buffer. Return 0 if not empty, 1 otherwise.
        """
        if time is not None:
            for p in self.buffer_packet_list:
                p.update_elapsed_time(time)
        if self.buffer_size == 0:
            return 1
        if priority_metric is not None:
            self.order_buffer(priority_metric)

        # Transmit packets
        if packet_id is not None:
            # Add packet to the pending ones
            if packet_id in self.buffer_packet_id_set and packet_id not in self.pending_packet_id_set:
                self.pending_packets_list.append(
                    self.buffer_packet_list[self.find_packet_by_id(packet_id)]
                )
                self.pending_packet_id_set.add(packet_id)
                self.pending = True
            else:
                print(
                    "Attention ! The packet with id {} "
                    "is not in the buffer and it cannot be scheduled for transmission".format(packet_id))
        # Transmit specific amount of data
        elif tx_size is not None:
            # Partial packet transmission is missing
            pkt_index = 0
            # Loop while there is space to be allocated, and data to be sent.
            while tx_size > 0 and pkt_index < len(self.buffer_packet_list):
                pending_packet = self.buffer_packet_list[pkt_index]
                # Update the pending packet size
                scheduled_size = pending_packet.schedule_pending(tx_size)
                if pending_packet.get_id() not in self.pending_packet_id_set:
                    self.pending_packets_list.append(pending_packet)
                    self.pending_packet_id_set.add(pending_packet.get_id())
                tx_size -= scheduled_size
                self.pending = True
                pkt_index += 1

        if self.buffer_size:
            return 0
        return 1

    def order_buffer(self, priority_metric):
        # Order packets by priority and qos_latency requirement to choose the one that needs resources
        self.buffer_packet_list = sorted(self.buffer_packet_list, reverse=False,
                                         key=operator.attrgetter(priority_metric))

    def find_packet_by_id(self, packet_id: int):
        """
            Return the list index of the packet with id == 'packet_id'.
        """
        for idx in range(self.n_packets):
            if self.buffer_packet_list[idx].get_id() == packet_id:
                return idx

    def get_packet_by_id(self, packet_id: int):
        return self.buffer_packet_list[self.find_packet_by_id(packet_id)]

    # Return the first packet of the UE buffer
    def get_first_packet(self):
        return self.buffer_packet_list[0]

    # Return the last packet of the UE buffer
    def get_last_packet(self):
        return self.buffer_packet_list[-1]

    # Return the buffer size
    def get_buffer_size(self):
        return self.buffer_size

    # Return the number of packets in the buffer
    def get_n_packets(self):
        return self.n_packets

    # Return the list of packets in the UE buffer
    def get_packet_list(self):
        return self.buffer_packet_list

    # Return the ID of the first packet of the UE buffer
    def get_first_packet_id(self):
        return self.buffer_packet_list[0].get_id()

    # Return the ID of the last packet of the UE buffer
    def get_last_packet_id(self):
        return self.buffer_packet_list[-1].get_id()

    def remove_old_data(self):
        """
            Remove packets from the buffer if they have not been transmitted because of errors in scheduling (due to
            the learning time needed).
        """
        packets_removed = list()
        for p_id in self.buffer_packet_id_set:
            for packet in self.buffer_packet_list:
                if packet.get_id() == p_id:
                    del self.buffer_packet_list[self.find_packet_by_id(packet.get_id())]
                    packets_removed.append(packet.get_id())
                    # print('Packet {} discarded '.format(packet.get_id()))
                    self.buffer_size -= packet.get_data_to_be_sent()
                    if self.buffer_size < 0:
                        self.buffer_size = 0
                    self.n_packets -= 1
        n_discarded_packets = len(packets_removed)
        for pack_id in packets_removed:
            self.buffer_packet_id_set.remove(pack_id)
        return n_discarded_packets

    # Return the True if there is at least a packet in the UE buffer
    def is_there_any_data(self):
        if self.get_n_packets() > 0:
            return True
        else:
            return False
