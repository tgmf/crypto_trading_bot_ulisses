#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating different trading models.
"""

import logging
from .bayesian_model import BayesianModel

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
        elif model_type == 'quantum':
            self.logger.info("Quantum-inspired model not yet implemented")
            # Fall back to Bayesian for now
            return BayesianModel(self.config)
        else:
            self.logger.warning(f"Unknown model type: {model_type}, using Bayesian model")
            return BayesianModel(self.config)