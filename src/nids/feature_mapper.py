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
    DEFAULT_FEATURE_ORDER = [
        "Bwd Packet Length Std",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Total Length of Fwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Mean",
        "Fwd IAT Std",
        "Total Fwd Packets",
        "Fwd Packet Length Std",
        "Flow IAT Max",
        "Flow Bytes/s",
        "Flow IAT Std",
        "Bwd Packet Length Min",
        "Fwd IAT Total"
    ]
    
    # Feature name mapping (extracted name -> model name)
    # Used when feature names differ between extraction and model
    FEATURE_NAME_MAP: Dict[str, str] = {
        # Add mappings here if feature names need translation
        # Example: "src_ip": "Source IP"
    }
    
    def __init__(
        self,
        feature_callback: Callable[[Dict[str, Any], Any], None],
        scaler: Optional[Any] = None,
        feature_order: Optional[List[str]] = None
    ):
        """
        Initialize the FeatureMapper.
        
        Args:
            feature_callback: Function called with mapped features
            scaler: Optional sklearn scaler object for feature normalization
            feature_order: List defining the order of features for ML input.
                          If None, uses DEFAULT_FEATURE_ORDER.
        """
        self.feature_callback = feature_callback
        self.scaler = scaler
        self.feature_order = feature_order or self.DEFAULT_FEATURE_ORDER
        
        # Statistics
        self._features_mapped = 0
    
    def map_features(
        self, 
        features: Dict[str, Any], 
        flow_metadata: Any = None
    ) -> Dict[str, Any]:
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
        
        # Step 1: Standardize feature names
        standardized = self._standardize_names(features)
        
        # Step 2: Order features according to model requirements
        ordered = self._order_features(standardized)
        
        # Step 3: Apply normalization if scaler is available
        if self.scaler is not None:
            ordered = self._normalize_features(ordered)
        
        # Step 4: Validate data types
        validated = self._validate_types(ordered)
        
        # Forward to Anomaly Detector
        if self.feature_callback:
            self.feature_callback(validated, flow_metadata)
        
        return validated
    
    def _standardize_names(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize feature names using the mapping dictionary.
        
        Args:
            features: Features with original names
            
        Returns:
            Features with standardized names
        """
        standardized = {}
        
        for key, value in features.items():
            # Use mapped name if available, otherwise keep original
            new_key = self.FEATURE_NAME_MAP.get(key, key)
            standardized[new_key] = value
        
        return standardized
    
    def _order_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Order features according to the model's expected input order.
        
        Args:
            features: Unordered features
            
        Returns:
            Ordered features dictionary
        """
        ordered = {}
        
        for feature_name in self.feature_order:
            if feature_name in features:
                ordered[feature_name] = features[feature_name]
            else:
                # Use default value (0) for missing features
                ordered[feature_name] = 0.0
        
        return ordered
    
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize feature values using the provided scaler.
        
        Args:
            features: Features with raw values
            
        Returns:
            Features with normalized values
        """
        if self.scaler is None:
            return features
        
        try:
            # Convert to array in the correct order
            values = np.array([[features[k] for k in self.feature_order]])
            
            # Apply scaling
            scaled_values = self.scaler.transform(values)
            
            # Convert back to dictionary
            normalized = {
                k: float(v) for k, v in 
                zip(self.feature_order, scaled_values[0])
            }
            
            return normalized
            
        except Exception:
            # If scaling fails, return original features
            return features
    
    def _validate_types(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and convert feature data types.
        
        Ensures all numeric features are proper floats.
        
        Args:
            features: Features to validate
            
        Returns:
            Features with validated types
        """
        validated = {}
        
        for key, value in features.items():
            if key in self.feature_order:
                # Ensure numeric features are floats
                try:
                    validated[key] = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    validated[key] = 0.0
            else:
                # Keep non-model features as-is (metadata)
                validated[key] = value
        
        return validated
    
    def get_feature_array(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert mapped features to a numpy array for ML input.
        
        Args:
            features: Mapped features dictionary
            
        Returns:
            2D numpy array shaped (1, n_features) for model prediction
        """
        values = [features.get(k, 0.0) for k in self.feature_order]
        return np.array([values])
    
    def set_scaler(self, scaler: Any) -> None:
        """
        Set or update the feature scaler.
        
        Args:
            scaler: sklearn-compatible scaler object
        """
        self.scaler = scaler
    
    @property
    def features_mapped(self) -> int:
        """Get the total number of feature sets mapped."""
        return self._features_mapped
    
    def reset_statistics(self) -> None:
        """Reset mapper statistics."""
        self._features_mapped = 0
