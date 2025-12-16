"""
Anomaly Detector Module (Component 5)

The anomaly detector receives the mapped flow features and feeds them into 
the pre-trained machine learning model. The model analyzes the features and 
classifies the network flow into one of several categories:

- Normal Traffic
- DoS (Denial of Service)
- DDoS (Distributed Denial of Service)
- Port Scanning
- Brute Force
- Web Attacks
- Bots

The result is then used to determine if any action or alert is needed.
"""

from typing import Dict, Any, Optional, Callable, List
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np
import pandas as pd


class AttackCategory(Enum):
    """Enumeration of attack categories that can be detected."""
    BENIGN = auto()
    DOS = auto()
    DDOS = auto()
    PORT_SCAN = auto()
    BRUTE_FORCE = auto()
    WEB_ATTACK = auto()
    BOT = auto()
    UNKNOWN = auto()


@dataclass
class DetectionResult:
    """
    Result of anomaly detection for a network flow.
    
    Attributes:
        prediction: String label of the prediction
        category: AttackCategory enum value
        confidence: Confidence score (0.0 to 1.0)
        is_attack: Whether the flow is classified as an attack
        flow_metadata: Optional metadata about the flow
    """
    prediction: str
    confidence: float
    is_attack: bool
    flow_metadata: Optional[Dict[str, Any]] = None
    category: AttackCategory = AttackCategory.UNKNOWN


class AnomalyDetector:
    """
    Classifies network flows using machine learning models.
    
    This is the fifth and final component in the NIDS pipeline. It receives
    mapped features and uses pre-trained ML models to classify the traffic
    as normal or one of several attack categories.
    
    The detector supports a two-stage classification:
    1. Binary classifier: Normal vs Attack
    2. Multi-class classifier: Attack type classification
    
    Attributes:
        binary_model: Model for binary classification (Normal/Attack)
        multi_class_model: Model for attack type classification
        detection_callback: Callback function for detection results
    """

    # Mapping for binary classification (handles both string and int)
    BINARY_ATTACK_VALUES = {1, "1", "Attack"}
    BINARY_BENIGN_VALUES = {0, "0", "BENIGN"}
    MULTICLASS_LABEL_MAP = {
        0: "DoS",
        1: "DDoS", 
        2: "PortScan",
        3: "BruteForce",
        4: "WebAttack",
        5: "Bot",
    }

    # Mapping from prediction labels to attack categories
    CATEGORY_MAP = {
        "BENIGN" or 0: AttackCategory.BENIGN,
        "DoS": AttackCategory.DOS,
        "DDoS": AttackCategory.DDOS,
        "PortScan": AttackCategory.PORT_SCAN,
        "Port Scanning": AttackCategory.PORT_SCAN,
        "Brute Force": AttackCategory.BRUTE_FORCE,
        "Web Attack": AttackCategory.WEB_ATTACK,
        "Bot": AttackCategory.BOT,
    }
    
    def __init__(
        self,
        binary_model: Optional[Any] = None,
        multi_class_model: Optional[Any] = None,
        detection_callback: Optional[Callable[[DetectionResult], None]] = None
    ):
        """
        Initialize the AnomalyDetector.
        
        Args:
            binary_model: Pre-trained model for binary classification
            multi_class_model: Pre-trained model for multi-class classification
            detection_callback: Function called with each detection result
        """
        self.binary_model = binary_model
        self.multi_class_model = multi_class_model
        self.detection_callback = detection_callback
        
        # Statistics
        self._flows_analyzed = 0
        self._attacks_detected = 0
        self._attack_counts: Dict[str, int] = {}

    def _is_attack(self, binary_pred) -> bool:
        """Check if binary prediction indicates an attack."""
        return binary_pred in self.BINARY_ATTACK_VALUES
    
    def _normalize_multiclass_prediction(self, pred) -> str:
        """Convert numeric or string prediction to standard string label."""
        # Handle numeric predictions
        if isinstance(pred, (int, np.integer)):
            return self.MULTICLASS_LABEL_MAP.get(pred, "Unknown")
        # Handle string predictions
        return str(pred)

    def detect(
        self, 
        features: np.ndarray, 
        flow_metadata: Optional[Any] = None
    ) -> DetectionResult:
        """
        Analyze flow features and detect potential intrusions.
        
        This method implements the two-stage classification:
        1. Binary classification: Is this traffic normal or an attack?
        2. If attack: Multi-class classification to determine attack type
        
        Args:
            features: Mapped features from the Feature Mapper
            flow_metadata: Optional flow object or metadata for context
            
        Returns:
            DetectionResult containing the classification and metadata
        """
        self._flows_analyzed += 1
        
        # Default result
        prediction = "Unknown"
        category = AttackCategory.UNKNOWN
        confidence = 0.0
        is_attack = False
        
        try:
            X = features
        
            # Stage 1: Binary classification
            if self.binary_model is not None:
                binary_pred = self.binary_model.predict(X)[0]
                
                # Get confidence if model supports it
                if hasattr(self.binary_model, 'predict_proba'):
                    proba = self.binary_model.predict_proba(X)[0]
                    confidence = float(max(proba))
                
                if self._is_attack(binary_pred):
                    is_attack = True
                    
                    # Stage 2: Multi-class classification
                    if self.multi_class_model is not None:
                        prediction = self._classify_attack_type(X, flow_metadata)
                        category = self.CATEGORY_MAP.get(prediction, AttackCategory.UNKNOWN)
                        
                        # Get confidence from multi-class model
                        if hasattr(self.multi_class_model, 'predict_proba'):
                            proba = self.multi_class_model.predict_proba(X)[0]
                            confidence = float(max(proba))
                    else:
                        prediction = "Attack"
                        category = AttackCategory.UNKNOWN
                else:
                    prediction = "BENIGN"
                    category = AttackCategory.BENIGN
                    is_attack = False
            
        except Exception as e:
            print(f"[DEBUG] Detection error: {type(e).__name__}: {e}")  # Add this line
            prediction = "Error"
            category = AttackCategory.UNKNOWN
            confidence = 0.0
        
        # Update statistics
        if is_attack:
            self._attacks_detected += 1
            self._attack_counts[prediction] = self._attack_counts.get(prediction, 0) + 1
        
        # Create result
        result = DetectionResult(
            prediction=prediction,
            category=category,
            confidence=confidence,
            is_attack=is_attack,
            flow_metadata=self._extract_metadata(flow_metadata)
        )
        
        # Forward to callback
        if self.detection_callback:
            self.detection_callback(result)
        
        return result
    
    def _classify_attack_type(
        self, 
        X: np.ndarray, 
        flow_metadata: Optional[Any]
    ) -> str:
        """
        Classify the specific type of attack.
        
        Applies heuristics to improve classification accuracy.
        
        Args:
            X: Scaled feature array
            flow_metadata: Flow metadata for heuristic checks
            
        Returns:
            String label of the attack type
        """
        prediction = str(self.multi_class_model.predict(X)[0])
        prediction = self._normalize_multiclass_prediction(prediction)
        # Apply domain knowledge heuristics
        if flow_metadata is not None and hasattr(flow_metadata, 'dest_port'):
            dest_port = flow_metadata.dest_port
            
            # SSH/FTP ports with DoS/DDoS might actually be brute force
            if prediction in ("DoS", "DDoS"):
                if dest_port in (21, 22):  # FTP, SSH
                    prediction = "Brute Force"
        
        return prediction
    
    def _extract_metadata(self, flow_metadata: Any) -> Optional[Dict[str, Any]]:
        """
        Extract relevant metadata from flow object.
        
        Args:
            flow_metadata: Flow object or existing metadata dict
            
        Returns:
            Dictionary of metadata or None
        """
        if flow_metadata is None:
            return None
        
        if isinstance(flow_metadata, dict):
            return flow_metadata
        
        # Extract from Flow object
        try:
            return {
                "src_ip": getattr(flow_metadata, 'src_ip', None),
                "dst_ip": getattr(flow_metadata, 'dest_ip', None),
                "src_port": getattr(flow_metadata, 'src_port', None),
                "dst_port": getattr(flow_metadata, 'dest_port', None),
                "protocol": getattr(flow_metadata, 'protocol', None),
            }
        except Exception:
            return None
    
    def load_models(
        self,
        binary_model_path: Optional[str] = None,
        multi_class_model_path: Optional[str] = None,
    ) -> None:
        """
        Load pre-trained models from file paths.
        
        Args:
            binary_model_path: Path to binary classifier model
            multi_class_model_path: Path to multi-class classifier model
        """
        import joblib
        
        if binary_model_path:
            self.binary_model = joblib.load(binary_model_path)
        
        if multi_class_model_path:
            self.multi_class_model = joblib.load(multi_class_model_path)
    
    @property
    def flows_analyzed(self) -> int:
        """Get the total number of flows analyzed."""
        return self._flows_analyzed
    
    @property
    def attacks_detected(self) -> int:
        """Get the total number of attacks detected."""
        return self._attacks_detected
    
    @property
    def attack_counts(self) -> Dict[str, int]:
        """Get counts of detected attacks by type."""
        return self._attack_counts.copy()
    
    def get_detection_rate(self) -> float:
        """Get the attack detection rate (attacks / total flows)."""
        if self._flows_analyzed == 0:
            return 0.0
        return self._attacks_detected / self._flows_analyzed
    
    def reset_statistics(self) -> None:
        """Reset detector statistics."""
        self._flows_analyzed = 0
        self._attacks_detected = 0
        self._attack_counts.clear()
