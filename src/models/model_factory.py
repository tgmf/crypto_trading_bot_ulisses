# src/models/model_factory.py
"""Factory for creating different trading models with optimal performance."""

import logging
import importlib.util
import time
import numpy as np

from .. import JAX_AVAILABLE
from ..utils.jax_config import benchmark_jax
from ..core.param_manager import ParamManager

# Check for available backends
jax_spec = importlib.util.find_spec("jax")
JAX_AVAILABLE = jax_spec is not None

tf_spec = importlib.util.find_spec("tensorflow")
TF_AVAILABLE = tf_spec is not None
TF_BAYESIAN_AVAILABLE = False

pymc_spec = importlib.util.find_spec("pymc")
PYMC_AVAILABLE = pymc_spec is not None
BAYESIAN_AVAILABLE = False
ENHANCED_AVAILABLE = False

# Try to import based on availability
if PYMC_AVAILABLE:
    try:
        from .bayesian_model import BayesianModel
        BAYESIAN_AVAILABLE = True
        
        try:
            from .enhanced_bayesian_model import EnhancedBayesianModel
            ENHANCED_AVAILABLE = True
        except ImportError:
            ENHANCED_AVAILABLE = False
            
    except ImportError:
        BAYESIAN_AVAILABLE = False

if TF_AVAILABLE:
    try:
        from .tf_bayesian_model import TFBayesianModel
        TF_BAYESIAN_AVAILABLE = True
    except ImportError:
        TF_BAYESIAN_AVAILABLE = False

class ModelFactory:
    """Factory for creating trading models with optimal acceleration"""
    
    def __init__(self, params):
        """
        Initialize with configuration from ParamManager.
        
        Args:
            params: ParamManager instance containing model configuration
        """
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Detect hardware acceleration capabilities
        self._configure_acceleration()
        
        # Log available models
        self.logger.info(f"Available models - PyMC: {PYMC_AVAILABLE} (Bayesian: {BAYESIAN_AVAILABLE}, "
                        f"Enhanced: {ENHANCED_AVAILABLE}), TensorFlow: {TF_BAYESIAN_AVAILABLE}")
    
    def _configure_acceleration(self):
        """
        Detect available acceleration and set appropriate flags in params.
        
        This sets multiple capability flags that models can check:
        - 'model', 'acceleration_type' - Primary acceleration type
        - 'model', 'jax_available' - Whether JAX is available
        - 'model', 'jax_performance' - JAX performance benchmark
        - 'model', 'tensorflow_gpu' - Whether TensorFlow GPU is available
        - 'model', 'pytensor_gpu' - Whether PyTensor GPU is available
        """
        # Initialize with no acceleration
        acceleration_type = "CPU-Only"
        
        # JAX Configuration
        if JAX_AVAILABLE:
            self.logger.info("JAX detected, running benchmarks...")
            self.params.set(True, 'model', 'jax_available')
            
            # Configure and benchmark JAX
            elapsed = benchmark_jax()
            self.params.set(elapsed, 'model', 'jax_performance')
            
            if elapsed and elapsed < 1.0:
                self.logger.info(f"JAX GPU acceleration detected (benchmark: {elapsed:.3f}s)")
                acceleration_type = "JAX-GPU"
            elif elapsed and elapsed < 3.0:
                self.logger.info(f"JAX acceleration detected (benchmark: {elapsed:.3f}s)")
                acceleration_type = "JAX-Accelerated"
            else:
                self.logger.info(f"JAX CPU mode detected (benchmark: {elapsed:.3f}s)")
                acceleration_type = "JAX-CPU"
        else:
            self.params.set(False, 'model', 'jax_available')
            self.params.set(None, 'model', 'jax_performance')
        
        # TensorFlow Configuration
        if TF_AVAILABLE:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    self.logger.info(f"TensorFlow GPU detected: {gpus}")
                    self.params.set(True, 'model', 'tensorflow_gpu')
                    if acceleration_type == "CPU-Only":
                        acceleration_type = "TensorFlow-GPU"
                else:
                    self.logger.info("TensorFlow CPU mode detected")
                    self.params.set(False, 'model', 'tensorflow_gpu')
                    if acceleration_type == "CPU-Only":
                        acceleration_type = "TensorFlow-CPU"
            except Exception as e:
                self.logger.warning(f"Error checking TensorFlow GPU: {e}")
                self.params.set(False, 'model', 'tensorflow_gpu')
        else:
            self.params.set(False, 'model', 'tensorflow_gpu')
        
        # PyTensor Configuration
        if PYMC_AVAILABLE:
            try:
                import pytensor
                if hasattr(pytensor, 'config') and pytensor.config.device == 'cuda':
                    self.logger.info("PyTensor GPU mode detected")
                    self.params.set(True, 'model', 'pytensor_gpu')
                    if acceleration_type == "CPU-Only":
                        acceleration_type = "PyTensor-GPU"
                else:
                    self.logger.info("PyTensor CPU mode detected")
                    self.params.set(False, 'model', 'pytensor_gpu')
                    if acceleration_type == "CPU-Only":
                        acceleration_type = "PyTensor-CPU"
            except Exception as e:
                self.logger.warning(f"Error checking PyTensor device: {e}")
                self.params.set(False, 'model', 'pytensor_gpu')
        else:
            self.params.set(False, 'model', 'pytensor_gpu')
        
        # Store primary acceleration type
        self.logger.info(f"Primary acceleration type: {acceleration_type}")
        self.params.set(acceleration_type, 'model', 'acceleration_type')
        
        # Set default JAX usage based on availability
        # Let models check and use JAX if beneficial
        self.params.set(JAX_AVAILABLE, 'model', 'use_jax')
    
    def create_model(self):
        """Create the model specified in configuration with optimal acceleration"""
        model_type = self.params.get('model', 'type', default='auto')
        
        # Auto-selection based on available models
        if model_type == 'auto':
            return self._auto_select_model()
            
        # Explicit model creation based on requested type
        if model_type == 'enhanced_bayesian' and ENHANCED_AVAILABLE:
            self.logger.info("Creating Enhanced Bayesian model")
            return EnhancedBayesianModel(self.params)
            
        elif model_type == 'bayesian' and BAYESIAN_AVAILABLE:
            self.logger.info("Creating standard Bayesian model")
            return BayesianModel(self.params)
            
        elif model_type == 'tf_bayesian' and TF_BAYESIAN_AVAILABLE:
            self.logger.info("Creating TensorFlow-based Bayesian model")
            return TFBayesianModel(self.params)
            
        elif model_type == 'quantum' and TF_BAYESIAN_AVAILABLE:
            self.logger.info("Creating quantum-inspired TensorFlow model")
            self.params.set(True, 'model', 'quantum_inspired') 
            return TFBayesianModel(self.params)
        
        # Fallback to best available model if requested model not available
        self.logger.warning(f"Requested model type '{model_type}' not available, selecting best alternative")
        return self._auto_select_model()
    
    def _auto_select_model(self):
        """Automatically select the best model based on available implementations"""
        # First priority: enhanced bayesian if available
        if ENHANCED_AVAILABLE:
            self.logger.info("Auto-selecting Enhanced Bayesian model")
            return EnhancedBayesianModel(self.params)
            
        # Second priority: standard bayesian model
        elif BAYESIAN_AVAILABLE:
            self.logger.info("Auto-selecting standard Bayesian model")
            return BayesianModel(self.params)
            
        # Third priority: tensorflow model
        elif TF_BAYESIAN_AVAILABLE:
            self.logger.info("Auto-selecting TensorFlow-based Bayesian model")
            return TFBayesianModel(self.params)
            
        # No model available
        else:
            raise ImportError("No suitable models available. Please install PyMC or TensorFlow.")