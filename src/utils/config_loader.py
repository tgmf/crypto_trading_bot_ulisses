#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration management utilities.
"""

import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

class ConfigLoader:
    """Loads and manages configuration from files and environment variables"""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize with path to config file"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = {}
        
        # Load environment variables
        load_dotenv()
        
        # Load config file
        self._load_config()
        
        # Override with environment variables
        self._override_from_env()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config from {self.config_path}: {str(e)}")
            self.config = {}
    
    def _override_from_env(self):
        """Override configuration with environment variables"""
        # Environment
        if 'ENVIRONMENT' in os.environ:
            self.config['environment'] = os.environ['ENVIRONMENT']
        
        # Initialize exchanges config if not present
        if 'exchanges' not in self.config:
            self.config['exchanges'] = {}
        
        # Binance API Keys
        if 'BINANCE_API_KEY' in os.environ and 'BINANCE_API_SECRET' in os.environ:
            self.config['exchanges']['binance'] = {
                'api_key': os.environ['BINANCE_API_KEY'],
                'api_secret': os.environ['BINANCE_API_SECRET']
            }
            
            if 'BINANCE_TESTNET' in os.environ:
                self.config['exchanges']['binance']['testnet'] = (
                    os.environ['BINANCE_TESTNET'].lower() == 'true'
                )
        
        # Kraken API Keys
        if 'KRAKEN_API_KEY' in os.environ and 'KRAKEN_API_SECRET' in os.environ:
            self.config['exchanges']['kraken'] = {
                'api_key': os.environ['KRAKEN_API_KEY'],
                'api_secret': os.environ['KRAKEN_API_SECRET']
            }
        
        # KuCoin API Keys
        if all(k in os.environ for k in ['KUCOIN_API_KEY', 'KUCOIN_API_SECRET', 'KUCOIN_PASSPHRASE']):
            self.config['exchanges']['kucoin'] = {
                'api_key': os.environ['KUCOIN_API_KEY'],
                'api_secret': os.environ['KUCOIN_API_SECRET'],
                'passphrase': os.environ['KUCOIN_PASSPHRASE']
            }
            
            if 'KUCOIN_TESTNET' in os.environ:
                self.config['exchanges']['kucoin']['testnet'] = (
                    os.environ['KUCOIN_TESTNET'].lower() == 'true'
                )
        
        # FTX API Keys - Removed as FTX is no longer operational
        
        # Backtesting parameters
        if 'BACKTEST_FEE_RATE' in os.environ:
            if 'backtesting' not in self.config:
                self.config['backtesting'] = {}
            self.config['backtesting']['fee_rate'] = float(os.environ['BACKTEST_FEE_RATE'])
        
        if 'BACKTEST_MIN_PROFIT' in os.environ:
            if 'backtesting' not in self.config:
                self.config['backtesting'] = {}
            self.config['backtesting']['min_profit_target'] = float(os.environ['BACKTEST_MIN_PROFIT'])
        
        if 'BACKTEST_SLIPPAGE' in os.environ:
            if 'backtesting' not in self.config:
                self.config['backtesting'] = {}
            self.config['backtesting']['slippage'] = float(os.environ['BACKTEST_SLIPPAGE'])
        
        # Model parameters
        if 'MODEL_TYPE' in os.environ:
            if 'model' not in self.config:
                self.config['model'] = {}
            self.config['model']['type'] = os.environ['MODEL_TYPE']
            
    def get(self, *keys, default=None):
        """
        Get a nested configuration value using a sequence of keys.
        
        Example:
            config_loader.get('incremental_training', 'chunk_size', default=10000)
            config_loader.get('exchanges', 'binance', 'api_key')
        
        Args:
            *keys: Sequence of keys to navigate nested dictionaries
            default: Value to return if the path doesn't exist
            
        Returns:
            The value at the specified path, or the default if not found
        """
        result = self.config
        try:
            for key in keys:
                result = result[key]
            return result
        except (KeyError, TypeError):
            return default
    
    def get_config(self):
        """Get the loaded configuration"""
        return self.config
    
    def set(self, *keys, value):
        """
        Set a nested configuration value.
        
        Example:
            config_loader.set('training', 'memory_efficient', value=True)
        
        Args:
            *keys: Sequence of keys to navigate the nested dictionaries
            value: Value to set at the specified path
        """
        if not keys:
            return
            
        # Navigate to the nested dictionary
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        
    def update_config(self, updates):
        """
        Update multiple config values at once.
        
        Args:
            updates: Dictionary of updates to apply
            
        Example:
            config_loader.update_config({
                'training': {'memory_efficient': True},
                'model': {'type': 'enhanced_bayesian'}
            })
        """
        def update_nested(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested(target[key], value)
                else:
                    target[key] = value
        
        update_nested(self.config, updates)