"""
Parameter schema definitions for type conversion and validation.

This module defines the expected types and validation rules for parameters
used throughout the application. Add new parameter schemas here instead of
modifying the ParamManager class.
"""

from typing import Dict, Any, Tuple, List, Union
import datetime

SchemaType = Dict[str, Any]
ParameterPath = Tuple[str, ...]
SchemaRegistry = Dict[ParameterPath, SchemaType]

def get_parameter_schemas() -> SchemaRegistry:
    """
    Define the types and validation rules for parameters.
    
    Returns:
        Dict mapping parameter paths to their schema definitions
    """
    return {
        # Data parameters
        ('data', 'timeframes'): {
            'type': 'list_str',
            'description': 'Trading timeframes to analyze'
        },
        ('data', 'symbols'): {
            'type': 'list_str',
            'description': 'Trading symbols/pairs to use'
        },
        ('data', 'start_date'): {
            'type': 'date',
            'description': 'Start date for historical data'
        },
        ('data', 'exchanges'): {
            'type': 'list_str',
            'description': 'Exchanges to use for data and trading'
        },
        ('data', 'alternative_sources'): {
            'type': 'list_str',
            'description': 'Alternative data sources'
        },
        ('data', 'path'): {
            'type': 'string',
            'description': 'Base path for data storage'
        },
        ('data', 'raw', 'path'): {
            'type': 'string',
            'description': 'Path for raw data storage'
        },
        
        # Model parameters
        ('model', 'type'): {
            'type': 'string',
            'allowed_values': ['bayesian', 'enhanced_bayesian', 'tf_bayesian'],
            'description': 'Type of prediction model to use'
        },
        ('model', 'path'): {
            'type': 'string',
            'description': 'Path to store trained models'
        },
        
        # Strategy parameters
        ('strategy', 'type'): {
            'type': 'string',
            'allowed_values': ['quantum', 'mean_reversion', 'trend_following'],
            'description': 'Trading strategy to use'
        },
        ('strategy', 'path'): {
            'type': 'string',
            'description': 'Path to store strategy configurations'
        },
        
        # Training parameters
        ('training', 'memory_efficient'): {
            'type': 'boolean',
            'description': 'Whether to use memory-efficient training'
        },
        ('training', 'test_size'): {
            'type': 'float',
            'min': 0.01,
            'max': 0.5,
            'description': 'Fraction of data to use for testing'
        },
        ('training', 'max_memory_mb'): {
            'type': 'integer',
            'min': 1000,
            'description': 'Maximum memory usage in MB'
        },
        ('training', 'min_profit_target'): {
            'type': 'float',
            'min': 0.0,
            'description': 'Minimum profit target for training'
        },
        ('training', 'incremental_training', 'chunk_size'): {
            'type': 'integer',
            'min': 1000,
            'description': 'Number of samples to process in each training chunk'
        },
        ('training', 'incremental_training', 'overlap'): {
            'type': 'integer',
            'min': 0,
            'description': 'Number of overlapping samples between chunks'
        },
        ('training', 'incremental_training', 'checkpoint_frequency'): {
            'type': 'integer',
            'min': 1,
            'description': 'Frequency of saving model checkpoints (in chunks)'
        },
        
        # Backtesting parameters
        ('backtesting', 'exit_threshold'): {
            'type': 'float',
            'min': 0.0,
            'description': 'Threshold for exiting positions'
        },
        ('backtesting', 'stop_loss'): {
            'type': 'float',
            'min': 0.0,
            'max': 1.0,
            'description': 'Stop loss percentage'
        },
        ('backtesting', 'initial_balance'): {
            'type': 'float',
            'min': 0.0,
            'description': 'Initial balance for backtesting'
        },
        ('backtesting', 'min_position_change'): {
            'type': 'float',
            'min': 0.0,
            'description': 'Minimum change in position size to trigger a trade'
        },
        ('backtesting', 'exaggerate'): {
            'type': 'boolean',
            'description': 'Whether to exaggerate backtesting results'
        },
        ('backtesting', 'slippage'): {
            'type': 'float',
            'min': 0.0,
            'description': 'Slippage to apply in backtesting'
        },
        ('backtesting', 'leverage'): {
            'type': 'float',
            'min': 1.0,
            'description': 'Leverage to use in backtesting'
        },
        ('backtesting', 'max_open_trades'): {
            'type': 'integer',
            'min': 0,
            'description': 'Maximum number of open trades allowed'
        },
        ('backtesting', 'max_holding_time'): {
            'type': 'integer',
            'min': 0,
            'description': 'Maximum holding time for positions in ticks'
        },
        ('backtesting', 'max_open_time'): {
            'type': 'integer',
            'min': 0,
            'description': 'Maximum time to keep a position open in ticks'
        },
        ('backtesting', 'max_open_trades_per_symbol'): {
            'type': 'integer',
            'min': 0,
            'description': 'Maximum open trades per symbol'
        },
        ('backtesting', 'max_open_trades_per_side'): {
            'type': 'integer',
            'min': 0,
            'description': 'Maximum open trades per side (long/short)'
        },
        ('backtesting', 'max_open_trades_per_symbol_per_side'): {
            'type': 'integer',
            'min': 0,
            'description': 'Maximum open trades per symbol per side'
        },
        ('backtesting', 'results', 'path'): {
            'type': 'string',
            'description': 'Path to store backtesting results'
        },
        ('backtesting', 'test_sets', 'custom_test_set'): {
            'type': 'any',
            'description': 'Custom test set for backtesting'
        },
        ('backtesting', 'test_sets', 'path'): {
            'type': 'string',
            'description': 'Path to test set files'
        },
        
        # Walk-forward parameters
        ('walk_forward', 'train_window'): {
            'type': 'integer',
            'min': 1,
            'description': 'Training window size in ticks'
        },
        ('walk_forward', 'test_window'): {
            'type': 'integer',
            'min': 1,
            'description': 'Testing window size in ticks'
        },
        ('walk_forward', 'step_size'): {
            'type': 'integer',
            'min': 1,
            'description': 'Step size for walk-forward analysis in ticks'
        },
        ('walk_forward', 'min_train_samples'): {
            'type': 'integer',
            'min': 10,
            'description': 'Minimum number of samples required for training'
        },
        
        # Exchange parameters
        ('exchange', 'name'): {
            'type': 'string',
            'description': 'Exchange name to use for trading'
        },
        ('exchange', 'fee_rate'): {
            'type': 'float',
            'min': 0.0,
            'max': 0.1,
            'description': 'Exchange fee rate for transactions'
        },
        ('exchange', 'testnet'): {
            'type': 'boolean',
            'description': 'Whether to use the exchange testnet'
        },
        
        # These are commonly used in CLI args and env vars
        ('environment',): {
            'type': 'string',
            'allowed_values': ['development', 'testing', 'production'],
            'description': 'Application environment'
        },
        ('log_level',): {
            'type': 'string',
            'allowed_values': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], #TODO: Make log_level count
            'description': 'Logging level'
        },
        ('verbose',): {
            'type': 'boolean',
            'description': 'Enable verbose output'
        },
    }

def get_schema_for_parameter(parameter_path: Union[List[str], Tuple[str, ...]]) -> SchemaType:
    """
    Get the schema for a specific parameter path.
    
    Args:
        parameter_path: Path to the parameter as a list or tuple of keys
        
    Returns:
        Schema definition dict or empty dict if not found
    """
    schemas = get_parameter_schemas()
    path_tuple = tuple(parameter_path)
    return schemas.get(path_tuple, {})

def is_valid_parameter(parameter_path: Union[List[str], Tuple[str, ...]], value: Any) -> bool:
    """
    Check if a value is valid according to the schema for a parameter.
    
    Args:
        parameter_path: Path to the parameter
        value: Value to validate
        
    Returns:
        True if valid, False otherwise
    """
    schema = get_schema_for_parameter(parameter_path)
    if not schema:
        # No schema defined, assume valid
        return True
        
    schema_type = schema.get('type', '')
    
    if schema_type == 'integer':
        if not isinstance(value, int):
            return False
        if 'min' in schema and value < schema['min']:
            return False
        if 'max' in schema and value > schema['max']:
            return False
    
    elif schema_type == 'float':
        if not isinstance(value, (int, float)):
            return False
        if 'min' in schema and value < schema['min']:
            return False
        if 'max' in schema and value > schema['max']:
            return False
    
    elif schema_type == 'string':
        if not isinstance(value, str):
            return False
        if 'allowed_values' in schema and value not in schema['allowed_values']:
            return False
    
    elif schema_type == 'boolean':
        if not isinstance(value, bool):
            return False
    
    elif schema_type == 'list_str':
        if not isinstance(value, list):
            return False
        # Check if all items are strings
        if not all(isinstance(item, str) for item in value):
            return False
    
    elif schema_type == 'date':
        if not isinstance(value, (datetime.date, datetime.datetime)):
            return False
    
    return True

def convert_value_to_schema_type(value: Any, param_path: Union[List[str], Tuple[str, ...]]) -> Any:
    """
    Convert a value to the type specified in its schema.
    
    Args:
        value: The value to convert
        param_path: Path to the parameter
        
    Returns:
        The converted value or original if no conversion needed
    """
    schema = get_schema_for_parameter(param_path)
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
                value = schema['min']
            if 'max' in schema and value > schema['max']:
                value = schema['max']
                
        elif schema_type == 'float':
            value = float(value)
            
            # Apply min/max constraints
            if 'min' in schema and value < schema['min']:
                value = schema['min']
            if 'max' in schema and value > schema['max']:
                value = schema['max']
                
        elif schema_type == 'boolean':
            if isinstance(value, str):
                value = value.lower().strip()
                if value in ('true', 'yes', '1', 'y', 'on'):
                    value = True
                elif value in ('false', 'no', '0', 'n', 'off'):
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
                        
        elif schema_type == 'string' and isinstance(value, (int, float, bool)):
            # Convert primitive types to string if needed
            value = str(value)
            
    except Exception as e:
        # If conversion fails, return original value
        pass
        
    return value