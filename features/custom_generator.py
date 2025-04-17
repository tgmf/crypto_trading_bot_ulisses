#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom feature generator example
"""

from base import FeatureGeneratorBase
import pandas as pd
import numpy as np

class ExampleGenerator(FeatureGeneratorBase):
    """Example custom feature generator"""
    
    def generate(self, df):
        """Generate custom features"""
        # Always make a copy to avoid modifying the original
        df = df.copy()
        
        # Get parameters from configuration
        window = self.params.get('model', 'features', 'example', 'window', default=10)
        
        # Generate features
        df['example_feature'] = df['close'].rolling(window=window).mean() / df['close']
        
        self.logger.debug(f"Generated custom features with window={window}")
        return df
    
"""
Don't forget to add your new feature to the configuration file.
    model:
    features:
        # Other features...
        
        # Custom feature (defined in /features/example.py or /features/example_generator.py)
        example:
            enabled: true
            window: 15
"""