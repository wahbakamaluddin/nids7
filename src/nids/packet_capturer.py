"""
Packet Capturer Module (Component 1)

The packet capturer is responsible for capturing live network traffic from 
the device's network interface. It is implemented using the Scapy library, 
which allows low-level packet sniffing and handling.

The captured packets are then forwarded to the Packet Parser for processing.
"""

import threading
from typing import Callable, Optional

from scapy.sendrecv import AsyncSniffer
from scapy.packet import Packet


class PacketCapturer:
    """
    Captures live network traffic from a network interface using Scapy.
    
    This is the first component in the NIDS pipeline. It captures raw packets
    and forwards them to the next component (Packet Parser) via a callback.
    
    Attributes:
        interface: Network interface to capture from (e.g., 'eth0', 'wlan0')
        packet_callback: Callback function to handle captured packets
        bpf_filter: Berkeley Packet Filter expression for filtering packets
    """
    
    def __init__(
        self,
        interface: str,
        packet_callback: Callable[[Packet], None],
        bpf_filter: str = "ip and (tcp or udp)"
    ):
        """
        Initialize the PacketCapturer.
        
        Args:
            interface: Network interface name to capture from
            packet_callback: Function called for each captured packet
            bpf_filter: BPF filter expression (default: IP packets with TCP/UDP)
        """
        self.interface = interface
        self.packet_callback = packet_callback
        self.bpf_filter = bpf_filter
        
        self._sniffer: Optional[AsyncSniffer] = None
        self._is_capturing = False
        self._lock = threading.Lock()
        
        # Statistics
        self._packets_captured = 0
    
    def start(self) -> None:
        """
        Start capturing packets from the network interface.
        
        Raises:
            PermissionError: If insufficient permissions to capture packets
            OSError: If the network interface is not available
        """
        with self._lock:
            if self._is_capturing:
                return
            
            self._sniffer = AsyncSniffer(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._handle_packet,
                store=False,  # Don't store packets in memory
            )
            
            self._sniffer.start()
            self._is_capturing = True
    
    def stop(self) -> None:
        """
        Stop capturing packets.
        
        This method safely stops the packet capture and cleans up resources.
        """
        with self._lock:
            if not self._is_capturing:
                return
            
            if self._sniffer:
                self._sniffer.stop()
                self._sniffer = None
            
            self._is_capturing = False
    
    def _handle_packet(self, packet: Packet) -> None:
        """
        Internal handler for captured packets.
        
        Increments statistics and forwards packet to the registered callback.
        
        Args:
            packet: The captured packet
        """
        self._packets_captured += 1
        
        # Forward packet to the next component (Packet Parser)
        if self.packet_callback:
            self.packet_callback(packet)
    
    @property
    def is_capturing(self) -> bool:
        """Check if packet capture is currently active."""
        return self._is_capturing
    
    @property
    def packets_captured(self) -> int:
        """Get the total number of packets captured."""
        return self._packets_captured
    
    def reset_statistics(self) -> None:
        """Reset capture statistics."""
        self._packets_captured = 0
