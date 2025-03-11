#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating different trading models.
"""

import logging
from .bayesian_model import BayesianModel
from .tf_bayesian_model import TFBayesianModel
from .enhanced_bayesian_model import EnhancedBayesianModel

class ModelFactory:
    """Factory for creating trading models"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_model(self):
        """Create a model based on configuration"""
        model_type = self.config.get('model', {}).get('type', 'bayesian')
        
        if model_type == 'bayesian':
            self.logger.info("Creating Bayesian model")
            return BayesianModel(self.config)
        elif model_type == 'tf_bayesian':
            self.logger.info("Creating TensorFlow-based Bayesian model")
            return TFBayesianModel(self.config)
        elif model_type == 'quantum':
            self.logger.info("Quantum-inspired model requested, using TensorFlow-based model instead")
            return TFBayesianModel(self.config)
        elif model_type == 'enhanced_bayesian':
            self.logger.info("Creating Enhanced Bayesian model")
            return EnhancedBayesianModel(self.config)
        else:
            self.logger.warning(f"Unknown model type: {model_type}, using TensorFlow-based Bayesian model")
            return TFBayesianModel(self.config)