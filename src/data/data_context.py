#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DataContext: Market data encapsulation with metadata and processing history.

This module provides a DataContext class that serves as a self-describing 
dataset container, tracking data state, transformations, and metadata.
The DataContext is a core abstraction layer for trading systems to ensure
proper data handling, validation, and reproducibility.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import hashlib
import copy
from typing import Dict, List, Optional, Tuple, Union, Any


class DataContext:
    """
    Encapsulates market data with metadata, processing history, and validation capabilities.
    Acts as a self-describing dataset that tracks its own state and transformations.
    
    Attributes:
        params: Parameter manager instance for configuration
        df: Pandas DataFrame containing the market data
        exchange: Exchange name (e.g., 'binance')
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Data timeframe (e.g., '1h')
        source: Data source identifier (e.g., 'raw', 'processed', 'test_set')
        processing_history: List of processing operations performed on the data
        split: Dictionary containing train/test/validation splits
    """
    
    def __init__(self, param_manager, df=None, exchange=None, symbol=None, 
                    timeframe=None, source=None):
        """
        Initialize with parameter manager and optional data and metadata.
        
        Args:
            param_manager: ParamManager instance for configuration
            df: Optional pandas DataFrame with market data
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h')
            source: Data source identifier (e.g., 'raw', 'processed', 'test_set')
        """
        self.params = param_manager
        self.df = df.copy() if df is not None else None
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.source = source or "unknown"
        self.processing_history = []
        self.split = {'train': None, 'test': None, 'validation': None}
        self.logger = logging.getLogger(__name__)
        
        # Generate creation timestamp for unique identification
        self.created_at = datetime.now().isoformat()
        
        # Calculate data hash if DataFrame is provided
        self._data_hash = self._calculate_hash() if df is not None else None
        
        # Add initialization to processing history
        self.add_processing_step("initialize", {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "rows": len(df) if df is not None else 0,
            "columns": list(df.columns) if df is not None else []
        })
        
        self.logger.debug(f"DataContext initialized for {symbol} {timeframe} "
                            f"from {source} with {len(df) if df is not None else 0} rows")
    
    def _calculate_hash(self) -> str:
        """
        Calculate a hash of the DataFrame to track changes.
        
        Returns:
            str: Hash string representing the current data state
        """
        if self.df is None or len(self.df) == 0:
            return "empty"
            
        # Use a sample of data to calculate hash for efficiency
        sample_size = min(1000, len(self.df))
        step = max(1, len(self.df) // sample_size)
        
        # Sample from beginning, middle and end for better representation
        sample_indices = list(range(0, len(self.df), step))
        if len(self.df) - 1 not in sample_indices:
            sample_indices.append(len(self.df) - 1)
            
        sample = self.df.iloc[sample_indices]
        
        # Create hash from a string representation of the data
        data_repr = (
            f"{str(sample.shape)}|"
            f"{sample.index[0]}|{sample.index[-1]}|"
            f"{str(sample.dtypes)}|"
            f"{sample.values.tobytes()[:10000]}"  # First 10KB of data
        )
        
        return hashlib.md5(data_repr.encode()).hexdigest()

    @classmethod
    def from_raw_data(cls, param_manager, exchange, symbol, timeframe):
        """
        Create DataContext from raw data directory with path from ParamManager.
        
        Args:
            param_manager: ParamManager instance
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h')
            
        Returns:
            DataContext: Instance with loaded data or None if file not found
            
        Raises:
            FileNotFoundError: If the specified data file doesn't exist
        """
        logger = logging.getLogger(__name__)
        
        # Get raw data directory from params or use default
        raw_path = param_manager.get('data', 'raw', 'path', default='data/raw')
        
        # Create safe symbol name for path
        symbol_safe = symbol.replace('/', '_')
        
        # Construct file path
        file_path = Path(f"{raw_path}/{exchange}/{symbol_safe}/{timeframe}.csv")
        
        if not file_path.exists():
            logger.warning(f"Raw data file not found: {file_path}")
            return None
            
        try:
            # Load the data
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            
            # Create and return DataContext
            context = cls(param_manager, df, exchange, symbol, timeframe, source="raw")
            
            logger.info(f"Loaded raw data for {symbol} {timeframe} with {len(df)} rows")
            return context
            
        except Exception as e:
            logger.error(f"Error loading raw data for {symbol} {timeframe}: {str(e)}")
            if param_manager.get('system', 'strict_mode', default=False):
                raise
            return None
    @classmethod
    def _load_from_path(cls, param_manager, exchange, symbol, timeframe, path, filename_pattern, source_name, strict_mode=None):
        """
        Internal helper method to load data from a file path with common error handling.
        
        Args:
            param_manager: ParamManager instance
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h')
            path: Base path to load from
            filename_pattern: Pattern for filename, with {} placeholders for timeframe
            source_name: Source name for logging and context
            strict_mode: Whether to raise exceptions or return None
            
        Returns:
            DataContext: Instance with loaded data or None if file not found
        """
        logger = logging.getLogger(__name__)
        
        # Create safe symbol name for path
        symbol_safe = symbol.replace('/', '_')
        
        # Construct file path
        file_path = Path(f"{path}/{exchange}/{symbol_safe}/{filename_pattern.format(timeframe=timeframe)}")
        
        if not file_path.exists():
            logger.warning(f"{source_name} file not found: {file_path}")
            return None
            
        try:
            # Load the data
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            
            # Create and return DataContext
            context = cls(param_manager, df, exchange, symbol, timeframe, source=source_name)
            
            logger.info(f"Loaded {source_name} data for {symbol} {timeframe} with {len(df)} rows")
            return context
            
        except Exception as e:
            logger.error(f"Error loading {source_name} data for {symbol} {timeframe}: {str(e)}")
            if strict_mode is None:
                strict_mode = param_manager.get('system', 'strict_mode', default=False)
            if strict_mode:
                raise
            return None

    @classmethod
    def from_raw_data(cls, param_manager, exchange, symbol, timeframe):
        """
        Create DataContext from raw data directory with path from ParamManager.
        
        Args:
            param_manager: ParamManager instance
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h')
            
        Returns:
            DataContext: Instance with loaded data or None if file not found
        """
        raw_path = param_manager.get('data', 'raw', 'path', default='data/raw')
        return cls._load_from_path(
            param_manager, exchange, symbol, timeframe,
            raw_path, "{timeframe}.csv", "raw"
        )

    @classmethod
    def from_processed_data(cls, param_manager, exchange, symbol, timeframe):
        """
        Create DataContext from processed data directory with path from ParamManager.
        
        Args:
            param_manager: ParamManager instance
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h')
            
        Returns:
            DataContext: Instance with loaded data or None if file not found
        """
        processed_path = param_manager.get('data', 'processed', 'path', default='data/processed')
        return cls._load_from_path(
            param_manager, exchange, symbol, timeframe,
            processed_path, "{timeframe}.csv", "processed", 
            strict_mode=True  # Keep existing behavior with strict mode always on
        )

    @classmethod
    def from_test_set(cls, param_manager, exchange, symbol, timeframe):
        """
        Create DataContext from test set directory with path from ParamManager.
        
        Args:
            param_manager: ParamManager instance
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h')
            
        Returns:
            DataContext: Instance with loaded data or None if file not found
        """
        test_sets_path = param_manager.get('backtesting', 'test_sets', 'path', default='data/test_sets')
        return cls._load_from_path(
            param_manager, exchange, symbol, timeframe,
            test_sets_path, "{timeframe}_test.csv", "test_set"
        )
        
    
    def validate(self, required_columns=None, min_rows=1):
        """
        Validate data meets specified requirements.
        
        Args:
            required_columns (list): Required column names
            min_rows (int): Minimum number of rows
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValueError: If validation fails and system.strict_mode is enabled
        """
        strict_mode = self.params.get('system', 'strict_mode', default=False)
        valid = True
        messages = []
        
        # Check if DataFrame exists
        if self.df is None:
            messages.append("DataFrame is None")
            valid = False
        else:
            # Check minimum rows
            if len(self.df) < min_rows:
                messages.append(f"DataFrame has {len(self.df)} rows, minimum required is {min_rows}")
                valid = False
                
            # Check required columns
            if required_columns:
                missing_cols = [col for col in required_columns if col not in self.df.columns]
                if missing_cols:
                    messages.append(f"Missing required columns: {missing_cols}")
                    valid = False
        
        # Add validation step to processing history
        result = "passed" if valid else "failed"
        self.add_processing_step("validate", {
            "required_columns": required_columns,
            "min_rows": min_rows,
            "result": result,
            "messages": messages
        })
        
        # Handle validation failures
        if not valid:
            error_msg = f"Data validation failed: {'; '.join(messages)}"
            self.logger.error(error_msg)
            
            if strict_mode:
                raise ValueError(error_msg)
                
        return valid
    
    def filter_by_date(self, start_date=None, end_date=None):
        """
        Filter data to specified date range.
        
        Args:
            start_date (str or datetime): Start date for filtering
            end_date (str or datetime): End date for filtering
            
        Returns:
            DataContext: Self for method chaining
            
        Raises:
            ValueError: If DataFrame is None or filtering yields empty result
        """
        if self.df is None:
            raise ValueError("Cannot filter: DataFrame is None")
            
        # Store original length for logging
        original_len = len(self.df)
        
        # Parse dates if provided as strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Apply filters
        if start_date is not None:
            self.df = self.df[self.df.index >= start_date]
            
        if end_date is not None:
            self.df = self.df[self.df.index <= end_date]
            
        # Check if result is empty
        if len(self.df) == 0:
            error_msg = "Date filtering resulted in empty DataFrame"
            self.logger.error(error_msg)
            
            if self.params.get('system', 'strict_mode', default=False):
                raise ValueError(error_msg)
        
        # Add to processing history
        self.add_processing_step("filter_by_date", {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "original_rows": original_len,
            "filtered_rows": len(self.df),
            "removed_rows": original_len - len(self.df)
        })
        
        self.logger.info(f"Filtered data from {original_len} to {len(self.df)} rows "
                            f"({(original_len - len(self.df))} removed)")
        
        # Update data hash
        self._data_hash = self._calculate_hash()
        
        return self
    
    def create_time_split(self, test_size=0.2, validation_size=0.0):
        """
        Create chronological train/test/validation splits.
        
        Args:
            test_size (float): Proportion of data for testing (0.0 to 1.0)
            validation_size (float): Proportion of data for validation (0.0 to 1.0)
            
        Returns:
            dict: Dictionary containing the splits
            
        Raises:
            ValueError: If DataFrame is None or splits are invalid
        """
        if self.df is None:
            raise ValueError("Cannot create split: DataFrame is None")
            
        # Validate split parameters
        if not 0.0 <= test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
            
        if not 0.0 <= validation_size < 1.0:
            raise ValueError(f"validation_size must be between 0.0 and 1.0, got {validation_size}")
            
        if test_size + validation_size >= 1.0:
            raise ValueError(f"Sum of test_size and validation_size must be less than 1.0, "
                                f"got {test_size + validation_size}")
        
        # Ensure the DataFrame is sorted by index
        df = self.df.sort_index()
        
        # Calculate split points
        total_rows = len(df)
        test_rows = int(total_rows * test_size)
        validation_rows = int(total_rows * validation_size)
        train_rows = total_rows - test_rows - validation_rows
        
        if train_rows <= 0:
            raise ValueError(f"Split would result in {train_rows} training samples, "
                                f"which is invalid. Adjust split parameters.")
        
        # Create splits (chronologically)
        self.split['train'] = df.iloc[:train_rows].copy()
        
        if validation_size > 0:
            self.split['validation'] = df.iloc[train_rows:train_rows+validation_rows].copy()
        else:
            self.split['validation'] = None
            
        if test_size > 0:
            self.split['test'] = df.iloc[-test_rows:].copy()
        else:
            self.split['test'] = None
        
        # Add to processing history
        self.add_processing_step("create_time_split", {
            "test_size": test_size,
            "validation_size": validation_size,
            "total_rows": total_rows,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
            "train_start": self.split['train'].index[0].isoformat() if self.split['train'] is not None else None,
            "train_end": self.split['train'].index[-1].isoformat() if self.split['train'] is not None else None,
            "validation_start": self.split['validation'].index[0].isoformat() if self.split['validation'] is not None else None,
            "validation_end": self.split['validation'].index[-1].isoformat() if self.split['validation'] is not None else None,
            "test_start": self.split['test'].index[0].isoformat() if self.split['test'] is not None else None,
            "test_end": self.split['test'].index[-1].isoformat() if self.split['test'] is not None else None
        })
        
        self.logger.info(f"Created time split: {train_rows} train, "
                            f"{validation_rows} validation, {test_rows} test rows")
        
        return self.split
    
    def create_subset(self, subset_index, total_subsets):
        """
        Create a subset of the data for segmented backtesting or analysis
        
        Args:
            subset_index (int): Which subset to use (0-indexed)
            total_subsets (int): Total number of subsets to divide data into
                
        Returns:
            DataContext: Self for method chaining
            
        Raises:
            ValueError: If subset_index is out of range or parameters are invalid
        """
        if self.df is None:
            raise ValueError("Cannot create subset: DataFrame is None")
            
        if subset_index < 0 or total_subsets <= 0:
            raise ValueError(f"Invalid parameters: subset_index={subset_index}, total_subsets={total_subsets}")
        
        # Store original length for logging
        original_len = len(self.df)
        
        # Calculate subset size and boundaries
        subset_size = len(self.df) // total_subsets
        if subset_size == 0:
            raise ValueError(f"Cannot create {total_subsets} subsets from {original_len} rows (too few rows)")
        
        # Ensure the dataframe is sorted by index
        self.df = self.df.sort_index()
        
        # Calculate start and end indices
        start_idx = subset_index * subset_size
        # For the last subset, include any remaining rows
        end_idx = len(self.df) if subset_index == total_subsets - 1 else (subset_index + 1) * subset_size
        
        if start_idx >= len(self.df):
            raise ValueError(f"Subset index {subset_index} is out of range for dataset with {len(self.df)} rows")
        
        # Extract the subset
        self.df = self.df.iloc[start_idx:end_idx]
        
        # Add to processing history
        self.add_processing_step("create_subset", {
            "subset_index": subset_index,
            "total_subsets": total_subsets,
            "original_rows": original_len,
            "subset_rows": len(self.df),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "subset_ratio": 1.0 / total_subsets
        })
        
        # Update source with subset information
        self.source = f"{self.source}_subset_{subset_index + 1}_of_{total_subsets}"
        
        self.logger.info(f"Created subset {subset_index + 1}/{total_subsets} with {len(self.df)}/{original_len} rows")
        
        # Update data hash
        self._data_hash = self._calculate_hash()
        
        return self
    
    @staticmethod
    def parse_subset_string(subset_str):
        """
        Parse a subset string in format "n/m" (e.g., "1/10")
        
        Args:
            subset_str (str): Subset string to parse
            
        Returns:
            tuple: (subset_index, total_subsets) or (None, None) if invalid
        """
        if not subset_str:
            return None, None
            
        try:
            parts = subset_str.split('/')
            if len(parts) == 2:
                subset_index = int(parts[0]) - 1  # Convert to 0-based index
                total_subsets = int(parts[1])
                if subset_index < 0 or total_subsets <= 0:
                    return None, None
                return subset_index, total_subsets
        except (ValueError, IndexError):
            pass
            
        return None, None
    
    def add_processing_step(self, operation, params=None):
        """
        Add operation to processing history with timestamp.
        
        Args:
            operation (str): Name of the operation performed
            params (dict): Parameters of the operation
            
        Returns:
            dict: The processing step that was added
        """
        # Create the step record
        step = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "params": params or {},
            "data_hash": self._data_hash,
            "step_id": len(self.processing_history)
        }
        
        # Add to history
        self.processing_history.append(step)
        
        return step
    
    def get_processing_history(self, as_dataframe=False):
        """
        Get processing history as list or DataFrame.
        
        Args:
            as_dataframe (bool): Whether to return as pandas DataFrame
            
        Returns:
            list or DataFrame: Processing history
        """
        if not as_dataframe:
            return self.processing_history
            
        # Convert to DataFrame for easier analysis
        if not self.processing_history:
            return pd.DataFrame()
            
        # Flatten the params dictionaries
        flat_history = []
        
        for step in self.processing_history:
            flat_step = {
                "step_id": step["step_id"],
                "operation": step["operation"],
                "timestamp": step["timestamp"],
                "data_hash": step["data_hash"]
            }
            
            # Add flattened parameters with operation prefix
            if step["params"]:
                for param_key, param_value in step["params"].items():
                    # Handle lists and complex objects
                    if isinstance(param_value, (list, dict)):
                        param_value = json.dumps(param_value)
                        
                    flat_step[f"param_{param_key}"] = param_value
                    
            flat_history.append(flat_step)
            
        return pd.DataFrame(flat_history)

    def __repr__(self):
        """String representation of the DataContext"""
        df_shape = f"{self.df.shape}" if self.df is not None else "None"
        
        return (f"DataContext(exchange='{self.exchange}', symbol='{self.symbol}', "
                f"timeframe='{self.timeframe}', source='{self.source}', "
                f"df_shape={df_shape}, processing_steps={len(self.processing_history)})")
                
    def __len__(self):
        """Return length of the DataFrame"""
        return 0 if self.df is None else len(self.df)
    
    # legacy
    # @classmethod
    # def from_processed_data(cls, param_manager, exchange, symbol, timeframe):
    #     """
    #     Create DataContext from processed data directory with path from ParamManager.
        
    #     Args:
    #         param_manager: ParamManager instance
    #         exchange: Exchange name (e.g., 'binance')
    #         symbol: Trading pair (e.g., 'BTC/USDT')
    #         timeframe: Data timeframe (e.g., '1h')
            
    #     Returns:
    #         DataContext: Instance with loaded data or None if file not found
            
    #     Raises:
    #         FileNotFoundError: If the specified data file doesn't exist
    #     """
    #     logger = logging.getLogger(__name__)
        
    #     # Get data directory from params or use default
    #     processed_path = param_manager.get('data', 'processed', 'path',
    #                                         default='data/processed')
        
    #     # Create safe symbol name for path
    #     symbol_safe = symbol.replace('/', '_')
        
    #     # Construct file path
    #     file_path = Path(f"{processed_path}/{exchange}/{symbol_safe}/{timeframe}.csv")
        
    #     if not file_path.exists():
    #         logger.warning(f"Processed data file not found: {file_path}")
    #         return None
            
    #     try:
    #         # Load the data
    #         df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            
    #         # Create and return DataContext
    #         context = cls(param_manager, df, exchange, symbol, timeframe, source="processed")
            
    #         logger.info(f"Loaded processed data for {symbol} {timeframe} with {len(df)} rows")
    #         return context
            
    #     except Exception as e:
    #         logger.error(f"Error loading processed data for {symbol} {timeframe}: {str(e)}")
    #         raise
    
    # @classmethod
    # def from_test_set(cls, param_manager, exchange, symbol, timeframe):
    #     """
    #     Create DataContext from test set directory with path from ParamManager.
        
    #     Args:
    #         param_manager: ParamManager instance
    #         exchange: Exchange name (e.g., 'binance')
    #         symbol: Trading pair (e.g., 'BTC/USDT')
    #         timeframe: Data timeframe (e.g., '1h')
            
    #     Returns:
    #         DataContext: Instance with loaded data or None if file not found
    #     """
    #     logger = logging.getLogger(__name__)
        
    #     # Get test set directory from params or use default
    #     test_sets_path = param_manager.get('backtesting', 'test_sets', 'path', 
    #                                         default='data/test_sets')
        
    #     # Create safe symbol name for path
    #     symbol_safe = symbol.replace('/', '_')
        
    #     # Construct file path
    #     file_path = Path(f"{test_sets_path}/{exchange}/{symbol_safe}/{timeframe}_test.csv")
        
    #     if not file_path.exists():
    #         logger.warning(f"Test set file not found: {file_path}")
    #         return None
            
    #     try:
    #         # Load the data
    #         df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            
    #         # Create and return DataContext
    #         context = cls(param_manager, df, exchange, symbol, timeframe, source="test_set")
            
    #         logger.info(f"Loaded test set for {symbol} {timeframe} with {len(df)} rows")
    #         return context
            
    #     except Exception as e:
    #         logger.error(f"Error loading test set for {symbol} {timeframe}: {str(e)}")
    #         if param_manager.get('system', 'strict_mode', default=False):
    #             raise
    #         return None