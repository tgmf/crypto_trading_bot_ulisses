# src/models/model_factory.py
"""
Factory for creating different trading models with optimal performance.
TODO: Add model registration and dynamic loading.
"""

import logging
import importlib.util
from typing import Dict, Type, Any

from ..core.param_manager import ParamManager

tf_spec = importlib.util.find_spec("tensorflow")
TF_AVAILABLE = tf_spec is not None

pymc_spec = importlib.util.find_spec("pymc")
PYMC_AVAILABLE = pymc_spec is not None

# Registry to track available model classes
AVAILABLE_MODELS: Dict[str, Type] = {}

# Try to import based on availability
if PYMC_AVAILABLE:
    try:
        from .bayesian_model import BayesianModel
        AVAILABLE_MODELS['bayesian'] = BayesianModel
    except ImportError:
        pass
        
    try:
        from .enhanced_bayesian_model import EnhancedBayesianModel
        AVAILABLE_MODELS['enhanced_bayesian'] = EnhancedBayesianModel
    except ImportError:
        pass

if TF_AVAILABLE:
    try:
        from .tf_bayesian_model import TFBayesianModel
        AVAILABLE_MODELS['tf_bayesian'] = TFBayesianModel
    except ImportError:
        pass

# Global model priorities for auto-selection
MODEL_PRIORITIES = [
    'enhanced_bayesian',  # First choice
    'bayesian',           # Second choice
    'tf_bayesian'         # Third choice
]

class ModelFactory:
    """Factory for creating trading models with optimal acceleration"""
    
    def __init__(self, params):
        """
        Initialize with configuration from ParamManager.
        
        Args:
            params: ParamManager instance containing model configuration
        """
        self.params = params or ParamManager.get_instance()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        
        _instance = None
        
    @classmethod
    def get_instance(cls, params=None):
        if cls._instance is None:
            cls._instance = cls(params)
        return cls._instance
        
    def create_model(self):
        """Create the model specified in configuration with optimal acceleration"""
        model_type = self.params.get('model', 'type', default='auto')
        self.logger.debug(f"Requested model type: {model_type}")
        
        # Auto-selection based on available models
        if model_type == 'auto':
            return self._auto_select_model()
            
        # Check if requested model is available
        if model_type in AVAILABLE_MODELS:
            self.logger.info(f"Creating {model_type} model")
            return AVAILABLE_MODELS[model_type](self.params)
        
        # Fallback to best available model if requested model not available
        self.logger.warning(f"Requested model type '{model_type}' not available, selecting best alternative")
        return self._auto_select_model()
    
    def _auto_select_model(self):
        """Automatically select the best model based on available implementations"""
        
        # Select first available model from priority list
        for model_type in MODEL_PRIORITIES:
            if model_type in AVAILABLE_MODELS:
                self.logger.info(f"Auto-selecting {model_type} model")
                return AVAILABLE_MODELS[model_type](self.params)
            
        # No model available
        self.logger.error("No suitable models available")
        raise ImportError("No suitable models available. Please install PyMC or TensorFlow.")
    
    def register_model(self, model_type: str, model_class: Type) -> None:
        """
        Register a new model type dynamically.
        
        This allows researchers to add new models at runtime.
        
        Args:
            model_type: String identifier for the model
            model_class: The model class to register
        """
        global AVAILABLE_MODELS, MODEL_PRIORITIES
    
        # Register the model
        AVAILABLE_MODELS[model_type] = model_class
        
        # Update priorities if specified
        if priority is not None and 0 <= priority < len(MODEL_PRIORITIES):
            if model_type not in MODEL_PRIORITIES:
                MODEL_PRIORITIES.insert(priority, model_type)
        elif model_type not in MODEL_PRIORITIES:
            # Add to end of list if no priority specified
            MODEL_PRIORITIES.append(model_type)
            
        self.logger.info(f"Registered new model type: {model_type}")