#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for trading strategies.

This module defines the interface that all trading strategies must implement.
Strategies are responsible for generating trading signals based on market data
and model predictions, while the backtest engine handles execution simulation,
performance evaluation, and reporting.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    Strategies translate market data and model predictions into actual trading decisions
    including position direction and size.
    """
    
    def __init__(self, params=None):
        """
        Initialize the strategy with configuration parameters.
        
        Args:
            params: Configuration parameters for the strategy
        """
        self.params = params or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.initialize_from_params()
        
    def initialize_from_params(self):
        """Initialize strategy parameters from config"""
        # Default implementation does nothing
        pass
        
    @abstractmethod
    def calculate_signals(self, df, model_predictions=None):
        """
        Calculate trading signals based on data and predictions.
        
        Args:
            df (DataFrame): Market data
            model_predictions: Predictions from a model (optional)
            
        Returns:
            DataFrame or array-like: Trading signals
        """
        pass
        
    @abstractmethod
    def calculate_position_sizes(self, signals, df):
        """
        Calculate position sizes based on signals.
        
        Args:
            signals: Trading signals from calculate_signals
            df (DataFrame): Market data
            
        Returns:
            array-like: Position sizes (-1 to 1 for full short to full long)
        """
        pass
        
    def initialize(self, df):
        """
        Initialize the strategy with data. 
        
        Called once before backtesting begins.
        
        Args:
            df (DataFrame): Market data
            
        Returns:
            bool: Success flag
        """
        # Default implementation does nothing
        return True
        
    def before_backtest(self, df):
        """
        Perform setup operations before backtest runs.
        
        Args:
            df (DataFrame): Market data
            
        Returns:
            DataFrame: Potentially modified market data
        """
        # Default implementation returns unmodified dataframe
        return df
        
    def after_backtest(self, results):
        """
        Perform operations after backtest completes.
        
        Args:
            results (DataFrame): Backtest results
            
        Returns:
            DataFrame: Potentially enhanced results
        """
        # Default implementation returns unmodified results
        return results
        
    def on_bar(self, index, bar_data, position, equity):
        """
        Called for each bar during backtesting (optional).
        
        Allows strategies to implement custom per-bar logic beyond
        the pre-calculated signals. This can be used for dynamic
        position adjustments or risk management.
        
        Args:
            index: Current bar index
            bar_data: Data for the current bar
            position: Current position
            equity: Current equity
            
        Returns:
            float: Adjusted position or None to use existing position
        """
        # Default implementation makes no adjustments
        return None