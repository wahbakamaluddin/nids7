"""
NIDS Pipeline Module

This module orchestrates the five core components of the Network Intrusion 
Detection System:

1. Packet Capturer - Captures live network traffic
2. Packet Parser - Converts packets to bidirectional flows
3. Feature Extractor - Extracts flow features
4. Feature Mapper - Maps features to ML model input format
5. Anomaly Detector - Classifies traffic using ML models

The pipeline provides a clean separation between the Traffic Processing 
Pipeline (components 1-4) and the detection logic (component 5), allowing 
for flexible deployment scenarios where the anomaly detection module can 
be updated without modifying the traffic processing pipeline.
"""

import threading
import time
from typing import Optional, Callable, Dict, Any

from nids.packet_capturer import PacketCapturer
from nids.packet_parser import PacketParser
from nids.feature_extractor import FeatureExtractor
from nids.feature_mapper import FeatureMapper
from nids.anomaly_detector import AnomalyDetector, DetectionResult
from nids.helper.other.flow import Flow
from nids.helper.other.writer import CSVWriter


class NIDSPipeline:
    """
    Main NIDS pipeline that orchestrates all five components.
    
    This class connects the components in sequence:
    PacketCapturer -> PacketParser -> FeatureExtractor -> FeatureMapper -> AnomalyDetector
    
    Each component processes data and passes it to the next via callbacks,
    creating an efficient streaming pipeline for real-time intrusion detection.
    
    Attributes:
        interface: Network interface to capture from
        model_paths: Dictionary of paths to ML models
        output_mode: Output mode ('csv', 'url', etc.)
        output: Output destination
    """
    
    def __init__(
        self,
        interface: str,
        binary_model_path: Optional[str] = None,
        multi_class_model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        output_path: Optional[str] = None,
        detection_callback: Optional[Callable[[DetectionResult], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the NIDS pipeline.
        
        Args:
            interface: Network interface name to capture from
            binary_model_path: Path to binary classifier model
            multi_class_model_path: Path to multi-class classifier model
            scaler_path: Path to feature scaler
            output: Output destination (file path or URL)
            detection_callback: Optional callback for detection results
            log_callback: Optional callback for log messages
        """
        self.interface = interface
        self.output_path = output_path
        self.log_callback = log_callback
        
        # Statistics
        self._is_running = False
        self._start_time: Optional[float] = None
        
        # Initialize output writer
        self._output_writer =  CSVWriter(self.output_path)
        
        # Initialize Component 5: Anomaly Detector
        self.anomaly_detector = AnomalyDetector(
            detection_callback=self._on_detection
        )
        
        # Load ML models
        if any([binary_model_path, multi_class_model_path, scaler_path]):
            self.anomaly_detector.load_models(
                binary_model_path=binary_model_path,
                multi_class_model_path=multi_class_model_path,
                scaler_path=scaler_path
            )
        
        # Store user's detection callback
        self._user_detection_callback = detection_callback
        
        # Initialize Component 4: Feature Mapper
        self.feature_mapper = FeatureMapper(
            feature_callback=self._on_features_mapped,
            scaler=self.anomaly_detector.scaler
        )
        
        # Initialize Component 3: Feature Extractor
        self.feature_extractor = FeatureExtractor(
            feature_callback=self._on_features_extracted
        )
        
        # Initialize Component 2: Packet Parser
        self.packet_parser = PacketParser(
            flow_callback=self._on_flow_ready
        )
        
        # Initialize Component 1: Packet Capturer
        self.packet_capturer = PacketCapturer(
            interface=interface,
            packet_callback=self._on_packet_captured
        )
        
        # Garbage collection thread
        self._gc_thread: Optional[threading.Thread] = None
        self._gc_stop = threading.Event()
    
    def start(self) -> None:
        """
        Start the NIDS pipeline.
        
        This starts packet capture and begins processing the pipeline.
        """
        if self._is_running:
            return
        
        self._log(f"[*] Starting NIDS pipeline on interface {self.interface}")
        
        # Start garbage collection thread
        self._gc_stop.clear()
        self._gc_thread = threading.Thread(target=self._gc_worker, daemon=True)
        self._gc_thread.start()
        
        # Start packet capture
        self.packet_capturer.start()
        
        self._is_running = True
        self._start_time = time.time()
        
        self._log("[*] NIDS pipeline started successfully")
    
    def stop(self) -> None:
        """
        Stop the NIDS pipeline.
        
        This stops packet capture and flushes any remaining flows.
        """
        if not self._is_running:
            return
        
        self._log("[*] Stopping NIDS pipeline...")
        
        # Stop packet capture
        self.packet_capturer.stop()
        
        # Stop garbage collection
        self._gc_stop.set()
        if self._gc_thread:
            self._gc_thread.join(timeout=2.0)
        
        # Flush remaining flows
        self.packet_parser.flush_all_flows()
        
        # Clean up output writer
        if self._output_writer:
            try:
                del self._output_writer
                self._output_writer = None
            except Exception:
                pass
        
        self._is_running = False
        
        self._log(f"[*] NIDS pipeline stopped. Statistics:")
        self._log(f"    - Packets captured: {self.packet_capturer.packets_captured}")
        self._log(f"    - Flows processed: {self.packet_parser.flows_completed}")
        self._log(f"    - Attacks detected: {self.anomaly_detector.attacks_detected}")
    
    def _gc_worker(self) -> None:
        """Background garbage collection worker thread."""
        while not self._gc_stop.is_set():
            try:
                self.packet_parser.garbage_collect(time.time())
            except Exception:
                pass
            time.sleep(1.0)  # GC interval
    
    # Pipeline callbacks - each connects components in sequence
    
    def _on_packet_captured(self, packet) -> None:
        """
        Callback from Component 1 (Packet Capturer).
        Forwards packet to Component 2 (Packet Parser).
        """
        self.packet_parser.parse_packet(packet)
    
    def _on_flow_ready(self, flow: Flow) -> None:
        """
        Callback from Component 2 (Packet Parser).
        Forwards flow to Component 3 (Feature Extractor).
        """
        self.feature_extractor.extract_features(flow)
    
    def _on_features_extracted(self, features: Dict[str, Any], flow: Flow) -> None:
        """
        Callback from Component 3 (Feature Extractor).
        Forwards features to Component 4 (Feature Mapper).
        """
        self.feature_mapper.map_features(features, flow)
    
    def _on_features_mapped(self, features: Dict[str, Any], flow: Flow) -> None:
        """
        Callback from Component 4 (Feature Mapper).
        Forwards mapped features to Component 5 (Anomaly Detector).
        """
        self.anomaly_detector.detect(features, flow)
    
    def _on_detection(self, result: DetectionResult) -> None:
        """
        Callback from Component 5 (Anomaly Detector).
        Handles detection results - logging, output, and user callback.
        """
        # Log attacks
        if result.is_attack and result.flow_metadata:
            src_ip = result.flow_metadata.get('src_ip', 'Unknown')
            self._log(f"[!][!][!] Detected Attack: {result.prediction} from IP Address {src_ip}")
        
        # Write to output
        if self._output_writer and result.flow_metadata:
            output_data = result.flow_metadata.copy()
            output_data['Prediction'] = result.prediction
            self._output_writer.write(output_data)
        
        # Forward to user callback
        if self._user_detection_callback:
            self._user_detection_callback(result)
    
    def _log(self, message: str) -> None:
        """Internal logging method."""
        if self.log_callback:
            self.log_callback(message + "\n")
    
    # Properties for accessing statistics
    
    @property
    def is_running(self) -> bool:
        """Check if the pipeline is currently running."""
        return self._is_running
    
    @property
    def packets_captured(self) -> int:
        """Get total packets captured."""
        return self.packet_capturer.packets_captured
    
    @property
    def flows_processed(self) -> int:
        """Get total flows processed."""
        return self.packet_parser.flows_completed
    
    @property
    def active_flows(self) -> int:
        """Get number of currently active flows."""
        return self.packet_parser.get_active_flow_count()
    
    @property
    def attacks_detected(self) -> int:
        """Get total attacks detected."""
        return self.anomaly_detector.attacks_detected
    
    @property
    def uptime(self) -> float:
        """Get pipeline uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.
        
        Returns:
            Dictionary of statistics from all components
        """
        return {
            "is_running": self._is_running,
            "uptime_seconds": self.uptime,
            "packets_captured": self.packet_capturer.packets_captured,
            "packets_processed": self.packet_parser.packets_processed,
            "flows_created": self.packet_parser.flows_created,
            "flows_completed": self.packet_parser.flows_completed,
            "active_flows": self.packet_parser.get_active_flow_count(),
            "features_extracted": self.feature_extractor.flows_processed,
            "features_mapped": self.feature_mapper.features_mapped,
            "flows_analyzed": self.anomaly_detector.flows_analyzed,
            "attacks_detected": self.anomaly_detector.attacks_detected,
            "attack_counts": self.anomaly_detector.attack_counts,
            "detection_rate": self.anomaly_detector.get_detection_rate(),
        }
