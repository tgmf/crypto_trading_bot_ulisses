#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating different trading models.
"""

import logging
import importlib.util

# Check for TensorFlow availability
tf_spec = importlib.util.find_spec("tensorflow")
TF_AVAILABLE = tf_spec is not None

# Check for PyMC availability
pymc_spec = importlib.util.find_spec("pymc")
PYMC_AVAILABLE = pymc_spec is not None

# Import models conditionally based on available dependencies
BAYESIAN_AVAILABLE = False
ENHANCED_AVAILABLE = False
TF_BAYESIAN_AVAILABLE = False

# Try to import PyMC-based models if PyMC is available
if PYMC_AVAILABLE:
    try:
        from .bayesian_model import BayesianModel
        BAYESIAN_AVAILABLE = True
        
        # Try to import enhanced model which also depends on PyMC
        try:
            from .enhanced_bayesian_model import EnhancedBayesianModel
            ENHANCED_AVAILABLE = True
        except ImportError:
            ENHANCED_AVAILABLE = False
            
    except ImportError:
        BAYESIAN_AVAILABLE = False

# Try to import TensorFlow-based models if TensorFlow is available
if TF_AVAILABLE:
    try:
        from .tf_bayesian_model import TFBayesianModel
        TF_BAYESIAN_AVAILABLE = True
    except ImportError:
        TF_BAYESIAN_AVAILABLE = False

class ModelFactory:
    """Factory for creating trading models"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Log available models
        self.logger.info(f"Available models - PyMC: {PYMC_AVAILABLE} (Bayesian: {BAYESIAN_AVAILABLE}, "
                        f"Enhanced: {ENHANCED_AVAILABLE}), TensorFlow: {TF_BAYESIAN_AVAILABLE}")
    
    def create_model(self):
        """Create a model based on configuration and available libraries"""
        model_type = self.config.get('model', {}).get('type', 'auto')
        
        # Auto-select best available model
        if model_type == 'auto':
            if TF_BAYESIAN_AVAILABLE:
                self.logger.info("Auto-selecting TensorFlow-based Bayesian model")
                return TFBayesianModel(self.config)
            elif BAYESIAN_AVAILABLE:
                self.logger.info("Auto-selecting standard Bayesian model")
                return BayesianModel(self.config)
            else:
                raise ImportError("No suitable models available. Please install TensorFlow or PyMC.")
        
        # Handle PyMC-based models
        if model_type in ['bayesian', 'enhanced_bayesian']:
            if model_type == 'enhanced_bayesian' and ENHANCED_AVAILABLE:
                self.logger.info("Creating Enhanced Bayesian model")
                return EnhancedBayesianModel(self.config)
            elif BAYESIAN_AVAILABLE:
                if model_type == 'enhanced_bayesian':
                    self.logger.warning("Enhanced Bayesian model not available, falling back to standard Bayesian model")
                else:
                    self.logger.info("Creating standard Bayesian model")
                return BayesianModel(self.config)
            elif TF_BAYESIAN_AVAILABLE:
                self.logger.warning("PyMC not available, falling back to TensorFlow-based model")
                return TFBayesianModel(self.config)
            else:
                raise ImportError("No suitable models available. PyMC is required for Bayesian models.")
                
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