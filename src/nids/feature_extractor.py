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

            # Flow-level features (selected 6 out of 6)
            'Flow Duration': packet_time.get_duration(),
            'Flow Packets/s': packet_count.get_rate(),
            'Flow Bytes/s': flow_bytes.get_rate(),
            'Flow IAT Mean': flow_iat["mean"],
            'Flow IAT Max': flow_iat["max"],
            'Flow IAT Std': flow_iat["std"],
            
            # Forward features (selected 11 out of 11)
            'Fwd Header Length': flow_bytes.get_forward_header_bytes(),
            'Fwd IAT Total': forward_iat["total"],
            'Fwd IAT Mean': forward_iat["mean"],
            'Fwd IAT Max': forward_iat["max"],
            'Fwd IAT Std': forward_iat["std"],
            'Fwd Packet Length Min': packet_length.get_min(PacketDirection.FORWARD),
            'Fwd Packet Length Max': packet_length.get_max(PacketDirection.FORWARD),
            'Fwd Packet Length Mean': packet_length.get_mean(PacketDirection.FORWARD),
            'Fwd Packet Length Std': packet_length.get_std(PacketDirection.FORWARD),
            'Subflow Fwd Bytes': packet_length.get_total(PacketDirection.FORWARD),  # From subflow_fwd_byts
            'Total Fwd Packets': packet_count.get_total(PacketDirection.FORWARD),
            'Total Length of Fwd Packets': packet_length.get_total(PacketDirection.FORWARD),
            
            # Backward features (selected 6 out of 6)
            'Bwd Header Length': flow_bytes.get_reverse_header_bytes(),
            'Bwd Packet Length Min': packet_length.get_min(PacketDirection.REVERSE),
            'Bwd Packet Length Max': packet_length.get_max(PacketDirection.REVERSE),
            'Bwd Packet Length Std': packet_length.get_std(PacketDirection.REVERSE),
            'Bwd Packets/s': packet_count.get_rate(PacketDirection.REVERSE),
            'Init_Win_bytes_backward': flow.init_window_size.get(PacketDirection.REVERSE,0),
            
            # Packet-level features (selected 7 out of 7)
            'Packet Length Mean': packet_length.get_mean(),
            'Packet Length Std': packet_length.get_std(),
            'Packet Length Variance': packet_length.get_var(),
            'Average Packet Size': packet_length.get_avg(),  # From pkt_size_avg
            'PSH Flag Count': flag_count.count("P"),
            'Init_Win_bytes_forward': flow.init_window_size.get(PacketDirection.FORWARD,0),
            'Max Packet Length': packet_length.get_max(),
                    
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
