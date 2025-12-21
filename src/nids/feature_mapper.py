"""
Feature Mapper Module (Component 4)

The feature mapper translates the extracted flow features into the format 
expected by the anomaly detection model. This includes standardizing feature 
names, normalizing values if needed, and ensuring consistent data types.

The mapping ensures compatibility between the real-time extracted features 
and the trained machine learning model's input format.
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd
import joblib


class FeatureMapper:
    """
    Maps extracted features to ML model input format.
    
    This is the fourth component in the NIDS pipeline. It receives raw features
    from the Feature Extractor and transforms them into the exact format 
    expected by the Anomaly Detector's ML model.
    
    Key responsibilities:
    - Feature name standardization
    - Value normalization/scaling
    - Data type consistency
    - Feature ordering
    
    Attributes:
        feature_callback: Callback function when features are mapped
        scaler: Optional sklearn scaler for feature normalization
        feature_order: List defining the order of features for the ML model
    """
    
    # Default feature order matching the trained model
    REQUIRED_FEATURES = [
        # "Bwd Packet Length Std",
        # "Bwd Packet Length Mean",
        # "Bwd Packet Length Max",
        # "Total Length of Fwd Packets",
        # "Fwd Packet Length Max",
        # "Fwd Packet Length Mean",
        # "Fwd IAT Std",
        # "Total Fwd Packets",
        # "Fwd Packet Length Std",
        # "Flow IAT Max",
        # "Flow Bytes/s",
        # "Flow IAT Std",
        # "Bwd Packet Length Min",
        # "Fwd IAT Total"
        
        # Flow-level
        'Flow Duration',
        'Flow Packets/s',
        'Flow Bytes/s',
        'Flow IAT Mean',
        'Flow IAT Max',
        'Flow IAT Std',
        
        # Forward features
        'Fwd Header Length',
        'Fwd IAT Total',
        'Fwd IAT Mean',
        'Fwd IAT Max',
        'Fwd IAT Std',
        'Fwd Packet Length Min',
        'Fwd Packet Length Max',
        'Fwd Packet Length Mean',
        'Fwd Packet Length Std',
        'Subflow Fwd Bytes',
        'Total Fwd Packets',
        'Total Length of Fwd Packets',
        
        # Backward features
        'Bwd Header Length',
        'Bwd Packet Length Min',
        'Bwd Packet Length Max',
        'Bwd Packet Length Std',
        'Bwd Packets/s',
        'Init_Win_bytes_backward',
        
        # Packet-level
        'Packet Length Mean',
        'Packet Length Std',
        'Packet Length Variance',
        'Average Packet Size',
        'PSH Flag Count',
        'Init_Win_bytes_forward',
        'Max Packet Length',
    ]
    
    def __init__(
        self,
        feature_callback: Callable[[Dict[str, Any], Any], None],
        scaler: Optional[Any] = None,
        ):
        """
        Initialize the FeatureMapper.
        
        Args:
            feature_callback: Function called with mapped features
            scaler: Optional sklearn scaler object for feature normalization
            feature_order: List defining the order of features for ML input.
                          If None, uses REQUIRED_FEATURES.
        """
        self.feature_callback = feature_callback
        self.scaler = scaler
        
        # Statistics
        self._features_mapped = 0
    
    def map_features(
        self, 
        features: Dict[str, Any], 
        flow: Any = None
    ):
        """
        Map extracted features to model input format.
        
        This method performs:
        1. Feature name standardization
        2. Feature ordering
        3. Optional value normalization
        4. Data type validation
        
        Args:
            features: Raw features from the Feature Extractor
            flow_metadata: Optional flow object or metadata to pass through
            
        Returns:
            Dictionary of mapped features ready for the Anomaly Detector
        """
        self._features_mapped += 1
        feature_array = self._to_array(features)

        scaled_array = self.scale(feature_array)
    
        # Forward to Anomaly Detector
        if self.feature_callback:
            self.feature_callback(scaled_array, flow)

    
    def _to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert features dictionary to ordered numpy array.
        
        Args:
            features: Features dictionary from Feature Extractor
            
        Returns:
            2D numpy array shaped (1, n_features)
            
        Raises:
            KeyError: If a required feature is missing
        """
        values = [features[k] for k in self.REQUIRED_FEATURES]
        return np.array([values], dtype=np.float64)
    
    def scale(self, features: np.ndarray) -> np.ndarray:
        """
        Scale feature values using the configured scaler.
        
        This is the dedicated scaling function that applies normalization
        to the feature array before it's sent to the Anomaly Detector.
        
        Args:
            features: 2D numpy array shaped (1, n_features)
            
        Returns:
            Scaled 2D numpy array with same shape
        """
        if self.scaler is None:
            return features
        
        return self.scaler.transform(features)
    
    def set_scaler(self, scaler: Any) -> None:
        """
        Set or update the feature scaler.
        
        Args:
            scaler: sklearn-compatible scaler object
        """
        self.scaler = scaler
    

    def load_scaler(self, scaler_path: str) -> None:
        """
        Load a scaler from file.
        
        Args:
            scaler_path: Path to the saved scaler file (.joblib)
        """
  
        self.scaler = joblib.load(scaler_path)

    @property
    def features_mapped(self) -> int:
        """Get the total number of feature sets mapped."""
        return self._features_mapped
    
    def reset_statistics(self) -> None:
        """Reset mapper statistics."""
        self._features_mapped = 0
