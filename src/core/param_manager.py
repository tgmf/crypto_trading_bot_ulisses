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
import re
from datetime import datetime
from ..config.param_schemas import get_schema_for_parameter

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
    def get_instance(cls, base_config_path=None, cli_args=None, env_vars=True, force_new=False):
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
            if base_config_path is None and cls._instance is None:
                base_config_path = 'config/base.yaml'  # Default path as fallback
                
            cls._instance = cls(base_config_path, cli_args, env_vars)
            
        return cls._instance
    
    def __init__(self, base_config_path='config/base.yaml', cli_args=None, env_vars=True):
        """
        Initialize parameter manager with various configuration sources.
    
        Loading algorithm:
        1. Load base config
        2. Extract parameters from CLI that have their own config modules
        3. Load corresponding config modules dynamically
        4. Apply environment variables
        5. Apply remaining CLI arguments
        
        Args:
            base_config_path: Path to the base config file (str or Path)
            cli_args: Command line arguments (typically from argparse)
            env_vars: Whether to load environment variables (boolean)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = {}
        
        # Track parameter access for optimization and debugging
        self.access_registry = {}
        
        # Store loaded files to prevent circular references
        self._loaded_files = set()
        
        # Load configuration in order of increasing precedence
        # 1. Load base config
        if base_config_path:
            self._load_from_config(base_config_path)
    
        # 2. Load environment-specific config based on ENVIRONMENT env var
        environment = os.getenv('ENVIRONMENT', 'development')
        env_config = f"config/environment/{environment}.yaml"
        self._load_from_config(env_config)
        
        # 3. Load modular configurations based on base.yaml settings 
        # (if no CLI arguments override them)
        self._load_modular_configs_from_base()
        
        # 4. Load modular configurations based on CLI arguments
        if cli_args:
            self._load_modular_configs_from_cli(cli_args)
        
        # 5. Load environment variables
        if env_vars:
            self._load_from_env()
        
        # 6. Load remaining CLI arguments
        if cli_args:
            self._load_from_cli(cli_args)
            
        # Resolve any parameter references
        self._resolve_parameter_references()
        
        self.logger.debug(f"ParamManager initialized for environment: {environment}")
    
    def _load_from_config(self, config_path):
        """
        Load configuration from YAML file with support for includes and references
        
        Args:
            config_path: Path to the config file
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return
                
            # Prevent circular includes
            if str(config_path.resolve()) in self._loaded_files:
                self.logger.warning(f"Skipping already loaded config: {config_path}")
                return
                
            self._loaded_files.add(str(config_path.resolve()))
                
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            if config:
                self._update_nested(self.params, config)
                self.logger.info(f"Configuration from {config_path} — ✔")
            else:
                self.logger.warning(f"Empty or invalid config file: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            
    def _load_modular_configs_from_base(self):
        """
        Dynamically load ALL modular configurations based on base.yaml settings.
        
        This method scans the entire config structure and loads all applicable
        module configs based on parameters defined in base.yaml.
        
        It handles any parameter that follows the pattern:
        - parameter.type = 'value'
        
        And looks for corresponding config files in:
        - config/parameters/value.yaml
        """
        # Get available configurations
        available_configs = self.discover_available_configs()
    
        # Flatten the parameter structure for easier matching
        flattened_params = self.flatten()
    
        # Process each available config directory
        for module_dir, yaml_files in available_configs.items():
            # Skip base and environment configs - they're handled separately
            if module_dir in ['base', 'environment']:
                continue
            
            # Convert module directory to parameter prefix
            # For example: 'models' -> 'model', 'strategies' -> 'strategy'
            param_prefix = module_dir[:-1] if module_dir.endswith('s') else module_dir
        
            # Look for a parameter that specifies which config to load
            param_key = f"{param_prefix}.type"
            
            if param_key in flattened_params:
                module_value = flattened_params[param_key]
                self.logger.debug(f"Found parameter {param_key}={module_value} - looking for config file")
                
                # Try to find a matching config file
                possible_filenames = [
                    f"{module_value}.yaml",
                    f"{module_value.lower()}.yaml" if isinstance(module_value, str) else f"{module_value}.yaml",
                    f"{module_value.replace('-', '_')}.yaml" if isinstance(module_value, str) else f"{module_value}.yaml"
                ]
                
                config_file = None
                for filename in possible_filenames:
                    if filename in yaml_files:
                        config_file = Path('config') / module_dir / filename
                        break
                        
                # Load the config if found
                if config_file and config_file.exists():
                    self.logger.info(f"Loading module config: {config_file}")
                    self._load_from_config(config_file)
                else:
                    self.logger.debug(f"No config file found for {param_key}={module_value} in {module_dir}/")
            else:
                # Look for direct parameter match with module name
                # For example: if there's a 'data' parameter and 'data' module directory
                if param_prefix in self.params:
                    # Try to find default config
                    if 'default.yaml' in yaml_files:
                        config_file = Path('config') / module_dir / 'default.yaml'
                        self.logger.info(f"Loading default config for {param_prefix}: {config_file}")
                        self._load_from_config(config_file)
                        
        self.logger.debug("Loaded all modular configurations from base config")
            
    def _load_modular_configs_from_cli(self, args):
        """
        Dynamically load modular configurations based on CLI arguments.
        
        This method:
        1. Scans config directory to discover available modules
        2. Checks if any CLI args match module names
        3. Loads corresponding config files
        
        Args:
            args: Command line arguments
        """
        # Convert argparse Namespace to dictionary if needed
        if not isinstance(args, dict):
            args = vars(args)
    
        self.logger.debug(f"Processing CLI args for modular configs: {args}")
    
        # Use discover_available_configs to find all available module configs
        available_configs = self.discover_available_configs()
    
        # Skip 'base' and 'environment' modules
        if 'base' in available_configs:
            del available_configs['base']
        if 'environment' in available_configs:
            del available_configs['environment']
    
        # Create a mapping between possible CLI arg names and module directories
        arg_to_module_mapping = {}
        
        # Process each module directory
        for module_name in available_configs:
            # Map both singular and plural forms to the module name
            singular_name = module_name[:-1] if module_name.endswith('s') else module_name
            plural_name = f"{module_name}s" if not module_name.endswith('s') else module_name
            
            arg_to_module_mapping[singular_name] = module_name
            arg_to_module_mapping[plural_name] = module_name
        
        self.logger.debug(f"CLI arg to module mapping: {arg_to_module_mapping}")
        
        # Check each CLI arg against our mapping
        for arg_name, arg_value in args.items():
            if arg_value is None:
                continue
                
            # Check if this CLI arg maps to a module directory
            if arg_name in arg_to_module_mapping:
                module_dir = arg_to_module_mapping[arg_name]
                yaml_files = available_configs[module_dir]
                
                # Convert to lowercase for case-insensitive comparison
                value_lower = arg_value.lower() if isinstance(arg_value, str) else str(arg_value).lower()
                
                # Possible filename variations
                possible_filenames = [
                    f"{arg_value}.yaml",
                    f"{value_lower}.yaml",
                    f"{value_lower.replace('-', '_')}.yaml"
                ]
                
                # Find matching config file
                found_file = None
                for filename in possible_filenames:
                    if filename in yaml_files:
                        found_file = filename
                        break
                
                if found_file:
                    config_file = Path('config') / module_dir / found_file
                    self.logger.info(f"Loading module config for --{arg_name}={arg_value}: {config_file}")
                    self._load_from_config(config_file)
                else:
                    self.logger.warning(f"No config file found for --{arg_name}={arg_value} in {module_dir}/")
                    self.logger.debug(f"Available files in {module_dir}: {yaml_files}")
        
        self.logger.debug("Modular configurations based on CLI args — ✔")
        
    def discover_available_configs(self):
        """
        Scan the config directory structure and discover all available configuration modules.
        
        Returns:
            dict: Dictionary with module categories as keys and available configs as values
        """
        configs = {}
        config_root = Path('config')
        
        if not config_root.exists():
            self.logger.warning(f"Where is the config directory? Idk, but it's not '/{config_root}'")
            return configs
        
        # Scan base config
        base_file = config_root / 'base.yaml'
        if base_file.exists():
            configs['base'] = ['base.yaml']
        
        # Scan module directories
        for module_dir in [d for d in config_root.iterdir() if d.is_dir()]:
            module_name = module_dir.name
            yaml_files = [f.name for f in module_dir.glob('*.yaml')]
            
            if yaml_files:
                configs[module_name] = yaml_files
        
        return configs
    
    def _load_from_env(self):
        """Load configuration from environment variables (higher precedence)"""
        # Only map truly instance-specific or sensitive settings
        env_mapping = {
            # Resource limits (might vary by machine)
            'MAX_MEMORY': ('training', 'max_memory_mb'),
            'CHUNK_SIZE': ('training', 'incremental_training', 'chunk_size'),
        }
        
        # Process environment variables
        for env_var, param_path in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert to appropriate type
                if env_var in ['MAX_MEMORY', 'CHUNK_SIZE']:
                    value = int(value)
                
                # Set the parameter
                self.set(value, *param_path)
                
        # Handle exchange API credentials (these should definitely stay as env vars)
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
        
        # Ensure the primary exchange exists in the list of exchanges
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
    
        Apply all CLI args except those that were used to load module configs.
        
        Args:
            args: Command line arguments (typically from argparse)
        """
        # Convert argparse Namespace to dictionary if needed
        if not isinstance(args, dict):
            args = vars(args)
    
        # Identify module parameters (these were already used to load module configs)
        module_params = set()
        config_root = Path('config')
    
        if config_root.exists():
            # Get module directories and corresponding CLI argument names
            for module_dir in [d for d in config_root.iterdir() if d.is_dir()]:
                module_name = module_dir.name
                cli_arg_name = module_name[:-1] if module_name.endswith('s') else module_name
                module_params.add(cli_arg_name)
            
        # Map CLI arguments to parameter paths
        cli_mapping = {
            # Main parameters
            'model': ('model', 'type'),
            'max_memory': ('training', 'max_memory_mb'),
            
            # Data parameters
            'symbols': ('data', 'symbols'),
            'timeframes': ('data', 'timeframes'),
            'exchanges': ('data', 'exchanges'),  # Set as first exchange
            'start_date': ('data', 'start_date'),
            'end_date': ('data', 'end_date'),
            
            # Training parameters
            'test_size': ('training', 'test_size'),
            'cv': ('training', 'use_cv'),
            'chunk_size': ('training', 'incremental_training', 'chunk_size'),
            'overlap': ('training', 'incremental_training', 'overlap'),
            'checkpoint_freq': ('training', 'incremental_training', 'checkpoint_frequency'),
            'resume_from': ('training', 'incremental_training', 'resume_from'),
        
            # Backtesting parameters
            'no_trade_threshold': ('backtesting', 'no_trade_threshold'),
            'min_position_change': ('backtesting', 'min_position_change'),
            'min_profit_target': ('backtesting', 'min_profit_target'),
            'exit_threshold': ('backtesting', 'exit_threshold'),
            'exaggerate': ('backtesting', 'exaggerate'),
            'compound': ('backtesting', 'compound'),
            
            # Strategy parameters
            'strategy': ('strategy', 'type'),
            
            # Exchange parameters
            'exchange': ('exchange', 'type'),
            'fee_rate': ('exchange', 'fee_rate'),
        }
        
        # Process CLI arguments
        for arg_name, param_path in cli_mapping.items():
            if arg_name in args and args[arg_name] is not None and arg_name not in module_params:
                self.set(args[arg_name], *param_path)
    
        # Special handling for any other CLI args that don't have explicit mappings
        # but match parameter paths directly (using dot notation)
        for arg_name, arg_value in args.items():
            if '.' in arg_name and arg_value is not None and arg_name not in module_params:
                # Convert dot notation to parameter path
                param_path = arg_name.split('.')
                self.set(arg_value, *param_path)
        
        self.logger.debug("Loaded parameters from CLI arguments")
        
    def _resolve_parameter_references(self):
        """
        Resolve references between parameters (e.g., ${exchange.fee_rate})
        """
        def _resolve_value(value):
            if isinstance(value, str) and '${' in value:
                # Extract reference pattern
                reference_pattern = r'\${([\w\.]+)}'
                matches = re.findall(reference_pattern, value)
                
                for match in matches:
                    # Get referenced value
                    ref_keys = match.split('.')
                    ref_value = self.get(*ref_keys)
                    
                    if ref_value is not None:
                        # Replace reference with actual value
                        if isinstance(ref_value, (str, int, float, bool)):
                            placeholder = f"${{{match}}}"
                            if value == placeholder:  # Direct replacement
                                return ref_value
                            else:  # Embedded replacement
                                value = value.replace(placeholder, str(ref_value))
                
                return value
            else:
                return value
        
        def _scan_dict(d):
            for key, val in list(d.items()):
                if isinstance(val, dict):
                    _scan_dict(val)
                elif isinstance(val, list):
                    d[key] = [_resolve_value(item) if not isinstance(item, dict) 
                            else _scan_dict(item) for item in val]
                else:
                    d[key] = _resolve_value(val)
            return d
        
        # Process all parameters
        self.params = _scan_dict(self.params)
        self.logger.debug("Resolved parameter references")
    
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
    
        # Apply type conversion based on schema
        value = self._convert_value(value, keys)
            
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
    
    def _convert_value(self, value, param_path):
        """
        Convert a value to the correct type according to its schema.
        
        Args:
            value: The value to convert
            param_path: Path to the parameter as a tuple of keys
            
        Returns:
            The converted value
        """
        # Get schema for this parameter
        schema = get_schema_for_parameter(param_path)
        
        # If no schema defined, return value as is
        if not schema:
            return value
            
        schema_type = schema.get('type', '')
        
        try:
            if schema_type == 'integer':
                # Handle string representations like "10K", "2M", etc.
                if isinstance(value, str):
                    value = value.upper().strip()
                    if value.endswith('K'):
                        value = int(float(value[:-1]) * 1000)
                    elif value.endswith('M'):
                        value = int(float(value[:-1]) * 1000000)
                    elif value.endswith('G'):
                        value = int(float(value[:-1]) * 1000000000)
                    else:
                        value = int(value)
                else:
                    value = int(value)
                    
                # Apply min/max constraints
                if 'min' in schema and value < schema['min']:
                    self.logger.warning(f"Value {value} for {'.'.join(param_path)} is below minimum {schema['min']}, using minimum")
                    value = schema['min']
                if 'max' in schema and value > schema['max']:
                    self.logger.warning(f"Value {value} for {'.'.join(param_path)} is above maximum {schema['max']}, using maximum")
                    value = schema['max']
                    
            elif schema_type == 'float':
                value = float(value)
                
                # Apply min/max constraints
                if 'min' in schema and value < schema['min']:
                    self.logger.warning(f"Value {value} for {'.'.join(param_path)} is below minimum {schema['min']}, using minimum")
                    value = schema['min']
                if 'max' in schema and value > schema['max']:
                    self.logger.warning(f"Value {value} for {'.'.join(param_path)} is above maximum {schema['max']}, using maximum")
                    value = schema['max']
                    
            elif schema_type == 'boolean':
                if isinstance(value, str):
                    value = value.lower().strip()
                    if value in ('true', 'yes', '1', 'y'):
                        value = True
                    elif value in ('false', 'no', '0', 'n'):
                        value = False
                    else:
                        raise ValueError(f"Invalid boolean value: {value}")
                else:
                    value = bool(value)
                    
            elif schema_type == 'list_str':
                if isinstance(value, str):
                    # Handle comma or space-separated strings
                    if ',' in value:
                        value = [item.strip() for item in value.split(',')]
                    else:
                        value = [item.strip() for item in value.split()]
                elif not isinstance(value, list):
                    value = [str(value)]
                
                # Convert all items to strings
                value = [str(item) for item in value]
                
            elif schema_type == 'date':
                if isinstance(value, str):
                    # Try different date formats
                    for date_format in ('%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d/%m/%Y'):
                        try:
                            value = datetime.datetime.strptime(value, date_format).date()
                            break
                        except ValueError:
                            continue
                
            # Add more type conversions as needed
                
            return value
        except (ValueError, TypeError) as e:
            self.logger.error(f"Type conversion error for {'.'.join(param_path)}: {value} - {e}")
            # Return original value if conversion fails
            return value