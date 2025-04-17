#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for custom feature generators
"""

import pandas as pd
import numpy as np
import logging

class FeatureGeneratorBase:
    """Base class for all feature generators"""
    
    def __init__(self, params):
        """
        Initialize with parameter manager
        
        Args:
            params: ParamManager instance
        """
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate(self, df):
        """
        Generate features for this feature group
        
        Args:
            df (DataFrame): DataFrame with price data
            
        Returns:
            DataFrame: DataFrame with added features
        """
        raise NotImplementedError("Subclasses must implement generate()")