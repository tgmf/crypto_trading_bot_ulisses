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
        
        # Exchange API Keys (for authenticated requests)
        if 'BINANCE_API_KEY' in os.environ and 'BINANCE_API_SECRET' in os.environ:
            if 'exchanges' not in self.config:
                self.config['exchanges'] = {}
            
            self.config['exchanges']['binance'] = {
                'api_key': os.environ['BINANCE_API_KEY'],
                'api_secret': os.environ['BINANCE_API_SECRET']
            }
            
            if 'BINANCE_TESTNET' in os.environ:
                self.config['exchanges']['binance']['testnet'] = (
                    os.environ['BINANCE_TESTNET'].lower() == 'true'
                )
        
        # FTX API Keys
        if 'FTX_API_KEY' in os.environ and 'FTX_API_SECRET' in os.environ:
            if 'exchanges' not in self.config:
                self.config['exchanges'] = {}
            
            self.config['exchanges']['ftx'] = {
                'api_key': os.environ['FTX_API_KEY'],
                'api_secret': os.environ['FTX_API_SECRET']
            }
            
            if 'FTX_SUBACCOUNT' in os.environ:
                self.config['exchanges']['ftx']['subaccount'] = os.environ['FTX_SUBACCOUNT']
        
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
    
    def get_config(self):
        """Get the loaded configuration"""
        return self.config