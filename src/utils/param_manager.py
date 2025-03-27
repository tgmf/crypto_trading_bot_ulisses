#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter management system for centralized configuration with multi-source precedence.
"""

import yaml
import os
import logging
from pathlib import Path
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

class ParamManager:
    """
    Centralized parameter management with multi-source precedence and nested access.
    
    Responsibilities:
    - Loading configuration from files, environment variables, and CLI arguments
    - Establishing clear parameter precedence (CLI > env vars > config file > defaults)
    - Providing type-safe parameter access with path notation
    - Supporting required parameters and validation
    - Tracking parameter access for optimization
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config_path=None, cli_args=None, env_vars=True, force_new=False):
        """
        Get or create the singleton instance of ParamManager.
        
        Args:
            config_path: Path to config file (required for first initialization)
            cli_args: Command line arguments
            env_vars: Whether to load environment variables
            force_new: Force creation of a new instance
            
        Returns:
            ParamManager: The singleton parameter manager instance
            
        Raises:
            ValueError: If config_path is None during first initialization
        """
        if cls._instance is None or force_new:
            if config_path is None and cls._instance is None:
                config_path = 'config/config.yaml'  # Default path as fallback
                
            cls._instance = cls(config_path=config_path, cli_args=cli_args, env_vars=env_vars)
            
        return cls._instance
    
    def __init__(self, config_path='config/config.yaml', cli_args=None, env_vars=True):
        """
        Initialize parameter manager with various configuration sources.
        
        Args:
            config_path: Path to the config file (str or Path)
            cli_args: Command line arguments (typically from argparse)
            env_vars: Whether to load environment variables (boolean)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = {}
        
        # Track parameter access for optimization and debugging
        self.access_registry = {}
        
        # Load configuration in order of increasing precedence
        self._load_defaults()
        
        if config_path:
            self._load_from_config(config_path)
            
        if env_vars:
            self._load_from_env()
            
        if cli_args:
            self._load_from_cli(cli_args)
        
        self.logger.debug("ParamManager initialized with multi-source configuration")
    
    def _load_defaults(self):
        """Load hardcoded default values (lowest precedence)"""
        defaults = {
            'data': {
                'symbols': ['BTC/USDT'],
                'timeframes': ['1h'],
                'exchanges': ['binance'],
                'start_date': '2020-01-01',
                'path': 'data',
                'raw': {
                    'path': 'data/raw'
                }
            },
            'model': {
                'type': 'enhanced_bayesian',
                'feature_cols': [
                    'bb_pos', 'RSI_14', 'MACDh_12_26_9', 'trend_strength', 
                    'volatility', 'volume_ratio', 'range', 'macd_hist_diff'
                ]
            },
            'training': {
                'test_size': 0.2,
                'memory_efficient': True,
                'max_memory_mb': 8000,
                'incremental_training': {
                    'chunk_size': 50000,
                    'overlap': 5000,
                    'checkpoint_frequency': 1
                }
            },
            'backtesting': {
                'fee_rate': 0.0006,
                'min_profit_target': 0.008,
                'exit_threshold': 0.03,
                'no_trade_threshold': 0.96,
                'min_position_change': 0.025,
                'results': {
                    'path': 'data/backtest_results'
                }
            },
            'walk_forward': {
                'train_window': 180,
                'test_window': 30,
                'step_size': 30
            }
        }
        
        self.params.update(defaults)
        self.logger.debug("Loaded default parameters")
    
    def _load_from_config(self, config_path):
        """
        Load configuration from YAML file (middle precedence)
        
        Args:
            config_path: Path to the config file
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return
                
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            if config:
                self._update_nested(self.params, config)
                self.logger.info(f"Loaded configuration from {config_path}")
            else:
                self.logger.warning(f"Empty or invalid config file: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables (higher precedence)"""
        # Environment variable mapping to parameter paths
        env_mapping = {
            'SYMBOLS': ('data', 'symbols'),
            'TIMEFRAMES': ('data', 'timeframes'),
            'EXCHANGE': ('data', 'exchanges', 0),  # First exchange
            'MODEL_TYPE': ('model', 'type'),
            'BACKTEST_FEE_RATE': ('backtesting', 'fee_rate'),
            'BACKTEST_MIN_PROFIT': ('backtesting', 'min_profit_target'),
            'BACKTEST_SLIPPAGE': ('backtesting', 'slippage'),
            'TEST_SIZE': ('training', 'test_size'),
            'MAX_MEMORY': ('training', 'max_memory_mb'),
            'CHUNK_SIZE': ('training', 'incremental_training', 'chunk_size'),
            'NO_TRADE_THRESHOLD': ('backtesting', 'no_trade_threshold'),
            'MIN_POSITION_CHANGE': ('backtesting', 'min_position_change')
        }
        
        # Process environment variables
        for env_var, param_path in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert to appropriate type
                if env_var in ['SYMBOLS', 'TIMEFRAMES']:
                    value = value.split()  # Split space-separated values into list
                elif env_var in ['BACKTEST_FEE_RATE', 'BACKTEST_MIN_PROFIT', 'BACKTEST_SLIPPAGE', 
                                'TEST_SIZE', 'NO_TRADE_THRESHOLD', 'MIN_POSITION_CHANGE']:
                    value = float(value)
                elif env_var in ['MAX_MEMORY', 'CHUNK_SIZE']:
                    value = int(value)
                
                # Set the parameter
                self.set(value, *param_path)
                
        # Handle exchange API credentials for multiple exchanges
        supported_exchanges = ['BINANCE', 'KRAKEN', 'KUCOIN']
        
        # Initialize exchanges configuration if not exists
        if 'exchanges' not in self.params:
            self.params['exchanges'] = {}
        
        # Process each exchange's API credentials
        for exchange in supported_exchanges:
            api_key_var = f"{exchange}_API_KEY"
            api_secret_var = f"{exchange}_API_SECRET"
            testnet_var = f"{exchange}_TESTNET"
            passphrase_var = f"{exchange}_PASSPHRASE"  # For exchanges like KuCoin
            
            if api_key_var in os.environ and api_secret_var in os.environ:
                exchange_name = exchange.lower()
                
                # Create config for this exchange
                self.params['exchanges'][exchange_name] = {
                    'api_key': os.environ[api_key_var],
                    'api_secret': os.environ[api_secret_var]
                }
                
                # Add testnet setting if provided
                if testnet_var in os.environ:
                    self.params['exchanges'][exchange_name]['testnet'] = (
                        os.environ[testnet_var].lower() == 'true'
                    )
                
                # Add passphrase for exchanges that require it
                if passphrase_var in os.environ:
                    self.params['exchanges'][exchange_name]['passphrase'] = os.environ[passphrase_var]
                    
                self.logger.debug(f"Loaded API credentials for {exchange_name}")
        
        # Ensure the primary exchange (from EXCHANGE env var) exists in the list of exchanges
        primary_exchange = os.environ.get('EXCHANGE', '').lower()
        if primary_exchange and primary_exchange not in self.params.get('data', {}).get('exchanges', []):
            # Add to exchanges list if not already there
            if isinstance(self.params.get('data', {}).get('exchanges', []), list):
                self.params['data']['exchanges'].insert(0, primary_exchange)
            else:
                self.params['data']['exchanges'] = [primary_exchange]
                
        self.logger.debug("Loaded parameters from environment variables")
    
    def _load_from_cli(self, args):
        """
        Load configuration from command line arguments (highest precedence)
        
        Args:
            args: Command line arguments (typically from argparse)
        """
        # Convert argparse Namespace to dictionary if needed
        if not isinstance(args, dict):
            args = vars(args)
            
        # Map CLI arguments to parameter paths
        cli_mapping = {
            'symbols': ('data', 'symbols'),
            'timeframes': ('data', 'timeframes'),
            'exchange': ('data', 'exchanges', 0),  # Set as first exchange
            'model': ('model', 'type'),
            'test_size': ('training', 'test_size'),
            'cv': ('training', 'use_cv'),
            'walk_forward': ('backtesting', 'use_walk_forward'),
            'position_sizing': ('backtesting', 'use_position_sizing'),
            'no_trade_threshold': ('backtesting', 'no_trade_threshold'),
            'min_position_change': ('backtesting', 'min_position_change'),
            'chunk_size': ('training', 'incremental_training', 'chunk_size'),
            'overlap': ('training', 'incremental_training', 'overlap'),
            'max_memory': ('training', 'max_memory_mb'),
            'checkpoint_freq': ('training', 'incremental_training', 'checkpoint_frequency'),
            'resume_from': ('training', 'incremental_training', 'resume_from')
        }
        
        # Process CLI arguments
        for arg_name, param_path in cli_mapping.items():
            if arg_name in args and args[arg_name] is not None:
                self.set(args[arg_name], *param_path)
        
        # Special handling for the first exchange
        if 'exchange' in args and args['exchange']:
            if isinstance(self.params['data']['exchanges'], list):
                # Replace the first exchange or add it
                if len(self.params['data']['exchanges']) > 0:
                    self.params['data']['exchanges'][0] = args['exchange']
                else:
                    self.params['data']['exchanges'].append(args['exchange'])
            else:
                # Create a new list
                self.params['data']['exchanges'] = [args['exchange']]
        
        self.logger.debug("Loaded parameters from CLI arguments")
    
    def get(self, *keys, default=None, required=False):
        """
        Get parameter with nested path, optional default, and requirement checking.
        
        Args:
            *keys: Sequence of keys forming the path to the parameter
            default: Default value if parameter not found
            required: Whether the parameter is required (raises error if missing)
            
        Returns:
            The requested parameter value or default
            
        Raises:
            ValueError: If a required parameter is missing
        """
        # Track parameter access for optimization
        access_key = '.'.join(str(k) for k in keys)
        self.access_registry[access_key] = self.access_registry.get(access_key, 0) + 1
        
        # Navigate through nested dictionaries
        value = self.params
        try:
            for key in keys:
                if isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
                    # Handle list indexing
                    value = value[key]
                elif isinstance(value, dict) and key in value:
                    # Handle dictionary lookup
                    value = value[key]
                else:
                    # Key not found in current level
                    raise KeyError(f"Key '{key}' not found in {value}")
                    
            return value
        except (KeyError, TypeError, IndexError):
            if required:
                raise ValueError(f"Required parameter '{access_key}' not found")
            return default
    
    def set(self, value, *keys):
        """
        Set parameter value with nested path.
        
        Args:
            value: Value to set
            *keys: Sequence of keys forming the path to the parameter
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not keys:
            return False
            
        # Navigate to the parent node
        parent = self.params
        for key in keys[:-1]:
            # Create path if it doesn't exist
            if isinstance(parent, dict):
                if key not in parent:
                    parent[key] = {}
                parent = parent[key]
            elif isinstance(parent, list) and isinstance(key, int) and 0 <= key < len(parent):
                parent = parent[key]
            else:
                # Can't create path
                return False
                
        # Set the value
        last_key = keys[-1]
        if isinstance(parent, dict):
            parent[last_key] = value
            return True
        elif isinstance(parent, list) and isinstance(last_key, int) and 0 <= last_key < len(parent):
            parent[last_key] = value
            return True
            
        return False
    
    def _update_nested(self, target, source):
        """
        Recursively update a nested dictionary without completely overwriting nested structures.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursively update nested dictionary
                self._update_nested(target[key], value)
            else:
                # Update or add value
                target[key] = value
    
    def get_all(self):
        """
        Get complete configuration for inspection or serialization.
        
        Returns:
            dict: Deep copy of all parameters
        """
        return copy.deepcopy(self.params)
    
    def export_active_config(self, filename=None):
        """
        Export the active configuration for reproducibility.
        
        Args:
            filename: Optional filename to save config to
            
        Returns:
            str: Path to saved config file if filename provided, else None
        """
        active_config = self.get_all()
        
        # Add metadata
        active_config['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'most_accessed_params': self.get_most_accessed_params(10)
        }
        
        # Export to file if requested
        if filename:
            try:
                with open(filename, 'w') as f:
                    yaml.dump(active_config, f, default_flow_style=False)
                self.logger.info(f"Exported active configuration to {filename}")
                return filename
            except Exception as e:
                self.logger.error(f"Failed to export configuration: {e}")
                
        return None
    
    def get_most_accessed_params(self, limit=10):
        """
        Get the most frequently accessed parameters.
        
        Args:
            limit: Maximum number of parameters to return
            
        Returns:
            list: Top accessed parameters with their access counts
        """
        sorted_params = sorted(
            self.access_registry.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_params[:limit]
    
    def register_derived_parameter(self, compute_func, *param_paths, cache=True):
        """
        Register a parameter that is computed from other parameters.
        
        Args:
            compute_func: Function that computes the derived parameter
            *param_paths: Paths to parameters this computation depends on
            cache: Whether to cache the computed value
            
        Returns:
            function: Getter function for the derived parameter
        """
        # Implementation for computing derived parameters
        # Based on a function that uses other parameters as inputs
        if cache:
            cached_value = None
            last_inputs = None
            
            def getter():
                nonlocal cached_value, last_inputs
                # Get current input values
                current_inputs = tuple(self.get(*path.split('.')) for path in param_paths)
                
                # Recompute only if inputs changed
                if current_inputs != last_inputs:
                    cached_value = compute_func(*current_inputs)
                    last_inputs = current_inputs
                    
                return cached_value
        else:
            # Always recompute
            def getter():
                inputs = [self.get(*path.split('.')) for path in param_paths]
                return compute_func(*inputs)
                
        return getter
    
    def flatten(self, prefix=''):
        """
        Create a flattened version of the parameters for easier access.
        
        Args:
            prefix: Optional prefix for all keys
            
        Returns:
            dict: Flattened dictionary with dot-notation keys
        """
        result = {}
        
        def _flatten(obj, current_path=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    _flatten(value, new_path)
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    new_path = f"{current_path}[{i}]"
                    _flatten(value, new_path)
            else:
                result[current_path] = obj
                
        _flatten(self.params)
        
        # Apply prefix if provided
        if prefix:
            return {f"{prefix}.{k}" if k else prefix: v for k, v in result.items()}
            
        return result