"""
Feature Extractor Module (Component 3)

The feature extractor extracts flow features from network packets and performs 
calculations for certain features of the flow. The features extracted and 
calculated are designed to match the features that the machine learning model 
of the anomaly detector was trained on.

The flow features are stored using Python's dictionary data type, where each key 
is a 5-tuple (source IP, source port, destination IP, destination port, and 
transport protocol), and the corresponding value contains the set of selected 
flow features.
"""

from typing import Dict, List, Optional, Callable, Any

from .helper.other.flow import Flow
from nids.helper.features.context import PacketDirection
from nids.helper.features.flag_count import FlagCount
from nids.helper.features.flow_bytes import FlowBytes
from nids.helper.features.packet_count import PacketCount
from nids.helper.features.packet_length import PacketLength
from nids.helper.features.packet_time import PacketTime
from nids.helper.other.utils import get_statistics


class FeatureExtractor:
    """
    Extracts features from network flows.
    
    This is the third component in the NIDS pipeline. It receives completed 
    flows from the Packet Parser and extracts a set of features that will be 
    used by the Anomaly Detector for classification.
    
    The features are designed to match the CICFlowMeter feature set, which is 
    widely used for network intrusion detection.
    
    Attributes:
        feature_callback: Callback function when features are extracted
    """
    
    def __init__(
        self,
        feature_callback: Callable[[Dict[str, Any], Flow], None],
    ):
        """
        Initialize the FeatureExtractor.
        
        Args:
            feature_callback: Function called with extracted features and flow
            include_fields: Optional list of specific feature names to extract.
                           If None, extracts the default feature set.
        """
        self.feature_callback = feature_callback

        # Statistics
        self._flows_processed = 0
        self._features_extracted = 0
    
    def extract_features(self, flow: Flow) -> Dict[str, Any]:
        """
        Extract features from a completed flow.
        
        This method calculates various statistical features from the flow's 
        packets, including packet lengths, inter-arrival times, byte rates, 
        and flag counts.
        
        Args:
            flow: Completed Flow object from the Packet Parser
            
        Returns:
            Dictionary mapping feature names to their calculated values
        """
        self._flows_processed += 1
        
        # Initialize feature calculators
        flow_bytes = FlowBytes(flow)
        flag_count = FlagCount(flow)
        packet_count = PacketCount(flow)
        packet_length = PacketLength(flow)
        packet_time = PacketTime(flow)
        
        # Calculate inter-arrival time statistics
        flow_iat = get_statistics(flow.flow_interarrival_time)
        forward_iat = get_statistics(
            packet_time.get_packet_iat(PacketDirection.FORWARD)
        )
        backward_iat = get_statistics(
            packet_time.get_packet_iat(PacketDirection.REVERSE)
        )
        
        # Calculate active/idle statistics
        active_stat = get_statistics(flow.active)
        idle_stat = get_statistics(flow.idle)
        
        # Build complete feature dictionary
        all_features = {
            # Backward packet length features
            "Bwd Packet Length Std": packet_length.get_std(PacketDirection.REVERSE),
            "Bwd Packet Length Mean": packet_length.get_mean(PacketDirection.REVERSE),
            "Bwd Packet Length Max": packet_length.get_max(PacketDirection.REVERSE),
            "Bwd Packet Length Min": packet_length.get_min(PacketDirection.REVERSE),
            
            # Forward packet length features
            "Total Length of Fwd Packets": packet_length.get_total(PacketDirection.FORWARD),
            "Fwd Packet Length Max": packet_length.get_max(PacketDirection.FORWARD),
            "Fwd Packet Length Mean": packet_length.get_mean(PacketDirection.FORWARD),
            "Fwd Packet Length Std": packet_length.get_std(PacketDirection.FORWARD),
            "Fwd Packet Length Min": packet_length.get_min(PacketDirection.FORWARD),
            
            # Packet count features
            "Total Fwd Packets": packet_count.get_total(PacketDirection.FORWARD),
            "Total Backward Packets": packet_count.get_total(PacketDirection.REVERSE),
            
            # Inter-arrival time features (converted to microseconds)
            "Flow IAT Max": flow_iat["max"] * 1_000_000,
            "Flow IAT Min": flow_iat["min"] * 1_000_000,
            "Flow IAT Mean": flow_iat["mean"] * 1_000_000,
            "Flow IAT Std": flow_iat["std"] * 1_000_000,
            "Fwd IAT Total": forward_iat["total"] * 1_000_000,
            "Fwd IAT Mean": forward_iat["mean"] * 1_000_000,
            "Fwd IAT Std": forward_iat["std"] * 1_000_000,
            "Fwd IAT Max": forward_iat["max"] * 1_000_000,
            "Fwd IAT Min": forward_iat["min"] * 1_000_000,
            "Bwd IAT Total": backward_iat["total"] * 1_000_000,
            "Bwd IAT Mean": backward_iat["mean"] * 1_000_000,
            "Bwd IAT Std": backward_iat["std"] * 1_000_000,
            "Bwd IAT Max": backward_iat["max"] * 1_000_000,
            "Bwd IAT Min": backward_iat["min"] * 1_000_000,
            
            # Flow byte rate features
            "Flow Bytes/s": flow_bytes.get_rate(),
            "Flow Packets/s": packet_count.get_rate(),
            
            # Flow metadata (useful for logging/reporting)
            "Src IP": flow.src_ip,
            "Dst IP": flow.dest_ip,
            "Src Port": flow.src_port,
            "Dst Port": flow.dest_port,
            "Protocol": flow.protocol,
            "Flow Duration": flow.duration * 1_000_000,  # microseconds
        }
        
        features = all_features
        
        self._features_extracted += len(features)
        
        # Forward features to the Feature Mapper
        if self.feature_callback:
            self.feature_callback(features, flow)
        
        return features
    
    @classmethod
    def get_available_features(cls) -> List[str]:
        """
        Get a list of all available features that can be extracted.
        
        Returns:
            List of feature names
        """
        return cls.DEFAULT_FEATURES.copy()
    
    @property
    def flows_processed(self) -> int:
        """Get the total number of flows processed."""
        return self._flows_processed
    
    @property
    def features_extracted(self) -> int:
        """Get the total number of features extracted."""
        return self._features_extracted
    
    def reset_statistics(self) -> None:
        """Reset extractor statistics."""
        self._flows_processed = 0
        self._features_extracted = 0
