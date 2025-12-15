"""
Packet Parser Module (Component 2)

The packet parser converts raw packets into flow-based features by aggregating 
packets into bidirectional flows. Each flow is uniquely identified using the 
5-tuple method (source IP, source port, destination IP, destination port, 
and transport protocol).

This is a widely adopted standard in network traffic analysis.
"""

import threading
from typing import Dict, Optional, Tuple, Callable

from scapy.packet import Packet

from nids.helper.other.constants import FLOW_TIMEOUT, PACKETS_PER_GC
from nids.helper.features.context import PacketDirection, get_packet_flow_key
from nids.helper.other.flow import Flow


# Type alias for flow key (5-tuple)
FlowKey = Tuple[Tuple[str, str, int, int], int]  # ((src_ip, dst_ip, src_port, dst_port), count)


class PacketParser:
    """
    Converts raw packets into bidirectional flows.
    
    This is the second component in the NIDS pipeline. It receives raw packets
    from the Packet Capturer and aggregates them into bidirectional flows
    identified by the 5-tuple method.
    
    Each flow is stored using Python's dictionary data type, where each key is 
    a 5-tuple (source IP, source port, destination IP, destination port, and 
    transport protocol), and the corresponding value contains the Flow object.
    
    Attributes:
        flows: Dictionary mapping flow keys to Flow objects
        flow_callback: Callback function when a flow is ready for feature extraction
    """
    
    def __init__(
        self,
        flow_callback: Callable[[Flow], None],
        flow_timeout: float = FLOW_TIMEOUT
    ):
        """
        Initialize the PacketParser.
        
        Args:
            flow_callback: Function called when a flow is complete and ready
                          for feature extraction
            flow_timeout: Time in seconds before a flow expires (default: 120s)
        """
        self.flow_callback = flow_callback
        self.flow_timeout = flow_timeout
        
        # Flow storage: key is ((src_ip, dst_ip, src_port, dst_port), count)
        self.flows: Dict[FlowKey, Flow] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._packets_processed = 0
        self._flows_created = 0
        self._flows_completed = 0
    
    def parse_packet(self, packet: Packet) -> Optional[Flow]:
        """
        Parse a packet and assign it to a flow.
        
        This method implements the 5-tuple flow identification:
        - Packets are assigned to existing flows if they match
        - New flows are created for unmatched packets
        - Expired flows are finalized and forwarded for feature extraction
        
        Args:
            packet: Raw packet from the Packet Capturer
            
        Returns:
            Flow object if assigned, None if packet is invalid
        """
        # Only process TCP and UDP packets
        if "TCP" not in packet and "UDP" not in packet:
            return None
        
        self._packets_processed += 1
        count = 0
        direction = PacketDirection.FORWARD
        
        try:
            packet_flow_key = get_packet_flow_key(packet, direction)
        except Exception:
            return None
        
        # Check for existing flow (forward direction)
        with self._lock:
            flow = self.flows.get((packet_flow_key, count))
        
        # If no forward flow exists, check reverse direction
        if flow is None:
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            with self._lock:
                flow = self.flows.get((packet_flow_key, count))
        
        # Create new flow if none exists
        if flow is None:
            direction = PacketDirection.FORWARD
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = Flow(packet, direction)
            
            with self._lock:
                self.flows[(packet_flow_key, count)] = flow
                self._flows_created += 1
        
        # Handle expired flows or FIN packets
        elif (packet.time - flow.latest_timestamp) > self.flow_timeout:
            expired = self.flow_timeout
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                expired += self.flow_timeout
                
                with self._lock:
                    flow = self.flows.get((packet_flow_key, count))
                
                if flow is None:
                    flow = Flow(packet, direction)
                    with self._lock:
                        self.flows[(packet_flow_key, count)] = flow
                        self._flows_created += 1
                    break
        
        # Add packet to flow
        flow.add_packet(packet, direction)
        
        # Check for flow termination conditions
        if flow.should_terminate() or self._packets_processed % PACKETS_PER_GC == 0:
            self.garbage_collect(packet.time)
        
        return flow
    
    def garbage_collect(self, latest_time: Optional[float] = None) -> None:
        """
        Clean up expired or terminated flows.
        
        Expired flows are forwarded to the Feature Extractor via the callback.
        
        Args:
            latest_time: Current timestamp for timeout calculation
        """
        with self._lock:
            keys = list(self.flows.keys())
        
        for key in keys:
            with self._lock:
                flow = self.flows.get(key)
            
            if not flow:
                continue
            
            # Check if flow should be terminated
            should_terminate = (
                flow.should_terminate()  # RST or bidirectional FIN
                or (latest_time is not None and 
                    latest_time - flow.latest_timestamp >= self.flow_timeout)
                or flow.duration >= self.flow_timeout
            )
            
            if not should_terminate:
                continue
            
            # Remove flow from storage
            with self._lock:
                if key in self.flows:
                    del self.flows[key]
                    self._flows_completed += 1
            
            # Forward flow to Feature Extractor
            if self.flow_callback:
                self.flow_callback(flow)
    
    def flush_all_flows(self) -> None:
        """
        Flush all remaining flows.
        
        Called when stopping capture to process any incomplete flows.
        """
        with self._lock:
            flows = list(self.flows.values())
            self.flows.clear()
        
        for flow in flows:
            self._flows_completed += 1
            if self.flow_callback:
                self.flow_callback(flow)
    
    def get_active_flow_count(self) -> int:
        """Get the number of currently active flows."""
        with self._lock:
            return len(self.flows)
    
    @property
    def packets_processed(self) -> int:
        """Get the total number of packets processed."""
        return self._packets_processed
    
    @property
    def flows_created(self) -> int:
        """Get the total number of flows created."""
        return self._flows_created
    
    @property
    def flows_completed(self) -> int:
        """Get the total number of flows completed."""
        return self._flows_completed
    
    def reset_statistics(self) -> None:
        """Reset parser statistics."""
        self._packets_processed = 0
        self._flows_created = 0
        self._flows_completed = 0
