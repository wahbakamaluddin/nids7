from scapy.packet import Packet

from nids.helper.other import constants
from nids.helper.features.context import PacketDirection, get_packet_flow_key
from nids.helper.features.flag_count import FlagCount
from nids.helper.features.flow_bytes import FlowBytes
from nids.helper.features.packet_count import PacketCount
from nids.helper.features.packet_length import PacketLength
from nids.helper.features.packet_time import PacketTime
from nids.helper.other.utils import get_statistics


class Flow:
    """This class summarizes the values of the features of the network flows"""

    def __init__(self, packet: Packet, direction: PacketDirection):
        """This method initializes an object from the Flow class.

        Args:
            packet (Any): A packet from the network.
            direction (Enum): The direction the packet is going ove the wire.
        """

        (
            self.src_ip,
            self.dest_ip,
            self.src_port,
            self.dest_port,
        ) = get_packet_flow_key(packet, direction)

        # Initialize flow properties with the first packet
        self.packets = [(packet, direction)]  # Add the first packet
        self.flow_interarrival_time = []
        self.start_timestamp = packet.time
        self.latest_timestamp = packet.time  # Initialize latest_timestamp too
        self.protocol = packet.proto

        # Initialize window sizes
        self.init_window_size = {PacketDirection.FORWARD: 0, PacketDirection.REVERSE: 0}
        if "TCP" in packet:
            # Set initial window size based on the first packet's direction
            self.init_window_size[direction] = packet["TCP"].window

        # Initialize active/idle tracking
        self.start_active = packet.time
        self.last_active = 0
        self.active = []
        self.idle = []

        # NEW: TCP termination flag tracking (to match Java CICFlowMeter)
        self.fwd_fin_count = 0
        self.bwd_fin_count = 0
        self.has_rst = False  # RST triggers immediate termination
        
        self.forward_bulk_last_timestamp = 0
        self.forward_bulk_start_tmp = 0
        self.forward_bulk_count = 0
        self.forward_bulk_count_tmp = 0
        self.forward_bulk_duration = 0
        self.forward_bulk_packet_count = 0
        self.forward_bulk_size = 0
        self.forward_bulk_size_tmp = 0
        self.backward_bulk_last_timestamp = 0
        self.backward_bulk_start_tmp = 0
        self.backward_bulk_count = 0
        self.backward_bulk_count_tmp = 0
        self.backward_bulk_duration = 0
        self.backward_bulk_packet_count = 0
        self.backward_bulk_size = 0
        self.backward_bulk_size_tmp = 0

    def get_data(self, include_fields=None) -> dict:
        """This method obtains the values of the features extracted from each flow.

        Note:
            Only some of the network data plays well together in this list.
            Time-to-live values, window values, and flags cause the data to
            separate out too much.

        Returns:
           list: returns a List of values to be outputted into a csv file.

        """

        flow_bytes = FlowBytes(self)
        flag_count = FlagCount(self)
        packet_count = PacketCount(self)
        packet_length = PacketLength(self)
        packet_time = PacketTime(self)
        flow_iat = get_statistics(self.flow_interarrival_time)
        forward_iat = get_statistics(
            packet_time.get_packet_iat(PacketDirection.FORWARD)
        )
        backward_iat = get_statistics(
            packet_time.get_packet_iat(PacketDirection.REVERSE)
        )
        active_stat = get_statistics(self.active)
        idle_stat = get_statistics(self.idle)

        data = {
            "Bwd Packet Length Std": packet_length.get_std(PacketDirection.REVERSE),
            "Bwd Packet Length Mean": packet_length.get_mean(PacketDirection.REVERSE),
            "Bwd Packet Length Max": packet_length.get_max(PacketDirection.REVERSE),
            "Total Length of Fwd Packets": packet_length.get_total(PacketDirection.FORWARD),
            "Fwd Packet Length Max": packet_length.get_max(PacketDirection.FORWARD),
            "Fwd Packet Length Mean": packet_length.get_mean(PacketDirection.FORWARD),
            "Fwd IAT Std": forward_iat["std"] * 1_000_000,
            "Total Fwd Packets": packet_count.get_total(PacketDirection.FORWARD),
            "Fwd Packet Length Std": packet_length.get_std(PacketDirection.FORWARD),
            "Flow IAT Max": flow_iat["max"] * 1_000_000,
            "Flow Bytes/s": flow_bytes.get_rate(),
            "Flow IAT Std": flow_iat["std"] * 1_000_000,
            "Bwd Packet Length Min": packet_length.get_min(PacketDirection.REVERSE),
            "Fwd IAT Total": forward_iat["total"] * 1_000_000
        }

        return data

    def add_packet(self, packet: Packet, direction: PacketDirection) -> None:
        """Adds a packet to the current list of packets.

        Args:
            packet: Packet to be added to a flow
            direction: The direction the packet is going in that flow

        """
        self.packets.append((packet, direction))

        # Calculate interarrival time using the previous latest_timestamp
        # This check prevents adding a 0 IAT for the very first packet added after init
        if len(self.packets) > 1:
            self.flow_interarrival_time.append(packet.time - self.latest_timestamp)

        # Update latest timestamp
        self.latest_timestamp = max(packet.time, self.latest_timestamp) # Compare the current packet time with the existing latest_timestamp to ensure it always reflects the most recent packet time.

        # Track TCP termination flags
        if "TCP" in packet:
            tcp_flags = str(packet["TCP"].flags)
            
            # Track RST flag (immediate termination)
            if "R" in tcp_flags:
                self.has_rst = True
            
            # Track FIN flags per direction (Java: waits for both directions)
            if "F" in tcp_flags:
                if direction == PacketDirection.FORWARD:
                    self.fwd_fin_count += 1
                else:
                    self.bwd_fin_count += 1       

        # Update flow bulk and subflow stats
        self.update_flow_bulk(packet, direction)
        self.update_subflow(packet)

        # Update initial window size if not already set for this direction
        if "TCP" in packet and self.init_window_size[direction] == 0:
            self.init_window_size[direction] = packet["TCP"].window

        # Note: start_timestamp and protocol are set in __init__

    def is_bidirectional_fin(self) -> bool:
        """Check if both directions have sent FIN (matches Java behavior).
        
        Returns:
            bool: True if both forward and backward FINs received
        """
        return self.fwd_fin_count > 0 and self.bwd_fin_count > 0

    def should_terminate(self) -> bool:
        """Check if flow should be terminated based on TCP flags.
        
        Matches Java CICFlowMeter logic:
        - RST: immediate termination
        - FIN: terminate when both directions have FIN
        
        Returns:
            bool: True if flow should be terminated
        """
        return self.has_rst or self.is_bidirectional_fin()

    def update_subflow(self, packet: Packet):
        """Update subflow

        Args:
            packet: Packet to be parse as subflow

        """
        last_timestamp = (
            self.latest_timestamp if self.latest_timestamp != 0 else packet.time
        )
        # If time gap between packets is larger than CLUMP_TIMEOUT, update active/idle
        if (packet.time - last_timestamp) > constants.CLUMP_TIMEOUT:
            # Pass
            self.update_active_idle(packet.time)

    def update_active_idle(self, current_time):
        """Adds a packet to the current list of packets.

        Args:
            packet: Packet to be update active time

        """
        # if time gap > ACTIVE_TIMEOUT, add the duration as idle time
        if (current_time - self.last_active) > constants.ACTIVE_TIMEOUT:
            # Update active time as the duration between last_active and start_active
            duration = abs(self.last_active - self.start_active)
            if duration > 0:
                self.active.append(duration)
            # Update idle time as the duration between current_time and last_active
            self.idle.append(current_time - self.last_active)
            # Reset active tracking as current_time
            self.start_active = current_time
            self.last_active = current_time
        else:
            self.last_active = current_time

    def update_flow_bulk(self, packet: Packet, direction: PacketDirection):
        """Update bulk flow

        Args:
            packet: Packet to be parse as bulk

        """
        payload_size = len(PacketCount.get_payload(packet))
        if payload_size == 0:
            return
        if direction == PacketDirection.FORWARD:
            if self.backward_bulk_last_timestamp > self.forward_bulk_start_tmp:
                self.forward_bulk_start_tmp = 0
            # Initialize new bulk
            if self.forward_bulk_start_tmp == 0:
                self.forward_bulk_start_tmp = packet.time
                self.forward_bulk_last_timestamp = packet.time
                self.forward_bulk_count_tmp = 1
                self.forward_bulk_size_tmp = payload_size
            else:
                # If time gap too large, start new bulk for a fresh features
                if (
                    packet.time - self.forward_bulk_last_timestamp
                ) > constants.CLUMP_TIMEOUT:
                    self.forward_bulk_start_tmp = packet.time
                    self.forward_bulk_last_timestamp = packet.time
                    self.forward_bulk_count_tmp = 1
                    self.forward_bulk_size_tmp = payload_size
                # If within bulk time, update bulk features
                else:  # Add to bulk
                    self.forward_bulk_count_tmp += 1
                    self.forward_bulk_size_tmp += payload_size
                    # Update bulk features when reaching BULK_BOUND
                    if self.forward_bulk_count_tmp == constants.BULK_BOUND:
                        self.forward_bulk_count += 1
                        self.forward_bulk_packet_count += self.forward_bulk_count_tmp
                        self.forward_bulk_size += self.forward_bulk_size_tmp
                        self.forward_bulk_duration += (
                            packet.time - self.forward_bulk_start_tmp
                        )
                    
                    elif self.forward_bulk_count_tmp > constants.BULK_BOUND:
                        self.forward_bulk_packet_count += 1
                        self.forward_bulk_size += payload_size
                        self.forward_bulk_duration += (
                            packet.time - self.forward_bulk_last_timestamp
                        )
                    self.forward_bulk_last_timestamp = packet.time
        else:
            if self.forward_bulk_last_timestamp > self.backward_bulk_start_tmp:
                self.backward_bulk_start_tmp = 0
            if self.backward_bulk_start_tmp == 0:
                self.backward_bulk_start_tmp = packet.time
                self.backward_bulk_last_timestamp = packet.time
                self.backward_bulk_count_tmp = 1
                self.backward_bulk_size_tmp = payload_size
            else:
                if (
                    packet.time - self.backward_bulk_last_timestamp
                ) > constants.CLUMP_TIMEOUT:
                    self.backward_bulk_start_tmp = packet.time
                    self.backward_bulk_last_timestamp = packet.time
                    self.backward_bulk_count_tmp = 1
                    self.backward_bulk_size_tmp = payload_size
                else:  # Add to bulk
                    self.backward_bulk_count_tmp += 1
                    self.backward_bulk_size_tmp += payload_size
                    if self.backward_bulk_count_tmp == constants.BULK_BOUND:
                        self.backward_bulk_count += 1
                        self.backward_bulk_packet_count += self.backward_bulk_count_tmp
                        self.backward_bulk_size += self.backward_bulk_size_tmp
                        self.backward_bulk_duration += (
                            packet.time - self.backward_bulk_start_tmp
                        )
                    elif self.backward_bulk_count_tmp > constants.BULK_BOUND:
                        self.backward_bulk_packet_count += 1
                        self.backward_bulk_size += payload_size
                        self.backward_bulk_duration += (
                            packet.time - self.backward_bulk_last_timestamp
                        )
                    self.backward_bulk_last_timestamp = packet.time

    @property
    def duration(self):
        return self.latest_timestamp - self.start_timestamp
