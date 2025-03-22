# src/models/model_factory.py
"""Factory for creating different trading models with optimal performance."""

import logging
import importlib.util
import time
import numpy as np

# Import the JAX configuration utility
from ..utils.jax_config import configure_jax, benchmark_jax

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
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check and configure hardware acceleration
        self.acceleration_type = self._check_acceleration()
        
        # Log available models and acceleration
        self.logger.info(f"Hardware acceleration: {self.acceleration_type}")
        self.logger.info(f"Available models - PyMC: {PYMC_AVAILABLE} (Bayesian: {BAYESIAN_AVAILABLE}, "
                        f"Enhanced: {ENHANCED_AVAILABLE}), TensorFlow: {TF_BAYESIAN_AVAILABLE}")
    
    def _check_acceleration(self):
        """Check what type of acceleration is available."""
        # Configure JAX if available
        if JAX_AVAILABLE:
            configure_jax()
            elapsed = benchmark_jax()
            
            if elapsed and elapsed < 1.0:
                return "JAX-GPU"
            elif elapsed and elapsed < 3.0:
                return "JAX-Accelerated"
            else:
                return "JAX-CPU"
        
        # Check TensorFlow GPU
        elif TF_AVAILABLE:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    return "TensorFlow-GPU"
                else:
                    return "TensorFlow-CPU"
            except:
                return "TensorFlow-CPU"
        
        # Check PyTensor
        elif PYMC_AVAILABLE:
            try:
                import pytensor
                if pytensor.config.device == 'cuda':
                    return "PyTensor-GPU"
                else:
                    return "PyTensor-CPU"
            except:
                return "CPU-Only"
        
        return "CPU-Only"
    
    def create_model(self):
        """Create a model based on configuration and available acceleration"""
        model_type = self.config.get('model', {}).get('type', 'auto')
        
        # Auto-selection based on acceleration
        if model_type == 'auto':
            if self.acceleration_type in ['JAX-GPU', 'JAX-Accelerated'] and PYMC_AVAILABLE:
                self.logger.info("Auto-selecting JAX-accelerated Bayesian model")
                return self._create_jax_accelerated_model()
            elif self.acceleration_type == 'TensorFlow-GPU' and TF_BAYESIAN_AVAILABLE:
                self.logger.info("Auto-selecting TensorFlow-based Bayesian model")
                return TFBayesianModel(self.config)
            elif BAYESIAN_AVAILABLE:
                self.logger.info("Auto-selecting standard Bayesian model")
                return BayesianModel(self.config)
            else:
                raise ImportError("No suitable models available. Please install PyMC or TensorFlow.")
        
        # Explicit model selection
        if model_type in ['bayesian', 'enhanced_bayesian']:
            if model_type == 'enhanced_bayesian' and ENHANCED_AVAILABLE:
                self.logger.info("Creating Enhanced Bayesian model")
                model = EnhancedBayesianModel(self.config)
            elif BAYESIAN_AVAILABLE:
                if model_type == 'enhanced_bayesian':
                    self.logger.warning("Enhanced Bayesian model not available, falling back to standard Bayesian model")
                else:
                    self.logger.info("Creating standard Bayesian model")
                model = BayesianModel(self.config)
            elif TF_BAYESIAN_AVAILABLE:
                self.logger.warning("PyMC not available, falling back to TensorFlow-based model")
                model = TFBayesianModel(self.config)
            else:
                raise ImportError("No suitable models available. PyMC is required for Bayesian models.")
            
            # Apply JAX acceleration if available
            if JAX_AVAILABLE and self.acceleration_type in ['JAX-GPU', 'JAX-Accelerated']:
                self.logger.info(f"Applying JAX acceleration to {model.__class__.__name__}")
                self._apply_jax_acceleration(model)
            
            return model
                
        # Handle TensorFlow-based models
        elif model_type in ['tf_bayesian', 'quantum']:
            if TF_BAYESIAN_AVAILABLE:
                if model_type == 'quantum':
                    self.logger.info("Creating quantum-inspired TensorFlow model")
                else:
                    self.logger.info("Creating TensorFlow-based Bayesian model")
                return TFBayesianModel(self.config)
            elif BAYESIAN_AVAILABLE:
                self.logger.warning("TensorFlow not available, falling back to PyMC-based Bayesian model")
                return BayesianModel(self.config)
            else:
                raise ImportError("No suitable models available. TensorFlow or PyMC is required.")
                
        # Unknown model type
        else:
            self.logger.warning(f"Unknown model type: {model_type}, trying to select best available")
            # Try TF first, then PyMC
            if TF_BAYESIAN_AVAILABLE:
                return TFBayesianModel(self.config)
            elif BAYESIAN_AVAILABLE:
                return BayesianModel(self.config)
            else:
                raise ImportError("No suitable models available. Please install TensorFlow or PyMC.")
    
    def _create_jax_accelerated_model(self):
        """Create a Bayesian model with JAX acceleration."""
        # Use the standard Bayesian model class but with JAX-specific customization
        model = BayesianModel(self.config)
        self._apply_jax_acceleration(model)
        return model
    
    def _apply_jax_acceleration(self, model):
        """Apply JAX acceleration to a PyMC model."""
        # Monkey patch the model's build_model method to use JAX
        original_build_model = model.build_model
        
        def jax_accelerated_build_model(X_train, y_train):
            import pymc as pm
            import pytensor
            
            # Set pytensor to use 32-bit precision for better JAX compatibility
            pytensor.config.compute_test_value = "off"
            pytensor.config.floatX = "float32"
            
            # Adjust y_train to be 0, 1, 2 instead of -1, 0, 1 for ordered logistic
            y_train_adj = y_train + 1
            
            # Create PyMC model with JAX-specific optimizations
            with pm.Model() as jax_model:
                # Priors for unknown model parameters
                alpha = pm.Normal("alpha", mu=0, sigma=10, shape=2)
                betas = pm.Normal("betas", mu=0, sigma=2, shape=X_train.shape[1])
                
                # Linear predictor
                eta = pm.math.dot(X_train, betas)
                
                # Ordered logistic regression likelihood
                p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_train_adj)
                
                # Sample with JAX acceleration
                self.logger.info("Starting JAX-accelerated MCMC sampling")
                start_time = time.time()
                
                trace = pm.sample(
                    draws=800,
                    tune=1000,
                    chains=4,
                    cores=1,  # Use 1 core to avoid conflicts with JAX
                    target_accept=0.9,
                    return_inferencedata=True,
                    compute_convergence_checks=False,
                    discard_tuned_samples=True 
                )
                
                sampling_time = time.time() - start_time
                self.logger.info(f"JAX-accelerated sampling completed in {sampling_time:.2f} seconds")
            
            model.model = jax_model
            model.trace = trace
            return jax_model, trace
        
        # Replace the original build_model with our JAX-accelerated version
        model.build_model = jax_accelerated_build_model
        model.using_jax_acceleration = True
        
        return model