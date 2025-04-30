#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retracement Level Analysis Script

This script analyzes the probability of price reaching specific Fibonacci retracement levels
(38.2%, 50%, 61.8%) after pivot points. It identifies pivot highs and lows, calculates
retracement targets, and tracks whether price reaches these levels within a specified period.

Usage:
    python retracement_analysis.py --symbols "BTC/USDT" --timeframes "1h" --pivots "high" --look-forward 48
"""

import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from project modules
from src.core.param_manager import ParamManager
from src.data.data_context import DataContext
from src.visualization.visualization import VisualizationTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetracementAnalyzer:
    """
    Analyzes price retracements after pivot points and calculates probabilities
    of reaching specific Fibonacci levels.
    """
    
    def __init__(self, params):
        """
        Initialize with parameter manager
        
        Args:
            params: ParamManager instance
        """
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Retracement levels to analyze (Fibonacci)
        self.levels = [0.382, 0.5, 0.618]
        
        # Create output directory for results
        self.output_dir = Path("data/analysis/retracement")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_analysis(self):
        """
        Run the retracement analysis for configured symbols and timeframes
        
        Returns:
            dict: Analysis results
        """
        symbols = self.params.get('data', 'symbols', default=[])
        timeframes = self.params.get('data', 'timeframes', default=[])
        exchanges = self.params.get('data', 'exchanges', default=[])
        
        # Analysis settings
        pivot_type = self.params.get('analysis', 'retracement', 'pivot_type', default='both')
        look_forward = self.params.get('analysis', 'retracement', 'look_forward', default=48)
        pivot_window = self.params.get('analysis', 'retracement', 'pivot_window', default=5)
        max_pivots = self.params.get('analysis', 'retracement', 'max_pivots', default=None)
        sample_rate = self.params.get('analysis', 'retracement', 'sample_rate', default=1)
        pivot_method = self.params.get('analysis', 'retracement', 'pivot_method', default='simple')
        fractal_min_window = self.params.get('analysis', 'retracement', 'fractal_min_window', default='5')
        fractal_max_windows = self.params.get('analysis', 'retracement', 'fractal_max_windows', default='5')
        fractal_min_strength = self.params.get('analysis', 'retracement', 'fractal_min_strength', default=1)
        chunk_size = self.params.get('analysis', 'retracement', 'chunk_size', default=1000000)
        
        # Create a timestamp for the analysis run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        all_results = {}
        
        for exchange in exchanges:
            for symbol in symbols:
                for timeframe in timeframes:
                    self.logger.info(f"Analyzing retracements for {symbol} {timeframe}")
                    
                    try:
                        # Load data using DataContext
                        data_context = DataContext.from_raw_data(self.params, exchange, symbol, timeframe)
                        
                        if data_context is None or data_context.df is None:
                            self.logger.warning(f"No data found for {symbol} {timeframe}")
                            continue
                            
                        # Validate data has required columns
                        if not data_context.validate(required_columns=['open', 'high', 'low', 'close', 'volume']):
                            continue
                            
                        # Process data to identify pivot points and retracements
                        result_context = self._analyze_retracements(
                            data_context, 
                            pivot_type,
                            look_forward,
                            pivot_window,
                            max_pivots,
                            sample_rate,
                            pivot_method,
                            fractal_min_window,
                            fractal_max_windows,
                            fractal_min_strength,
                            chunk_size
                        )
                        
                        # Store results
                        safe_symbol = symbol.replace('/', '_')
                        key = f"{exchange}_{safe_symbol}_{timeframe}"
                        all_results[key] = {
                            'exchange': exchange,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'data': result_context.df,
                            'stats': self._calculate_stats(result_context.df),
                            'processing_history': result_context.get_processing_history()
                        }
                        
                        # Save results
                        csv_file = self.output_dir / f"{key}_retracement_analysis_{timestamp}.csv"
                        result_context.df.to_csv(csv_file)
                        self.logger.info(f"Saved analysis results to {csv_file}")
                        
                        # Create visualizations
                        self._create_visualizations(result_context, key, timestamp)
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        continue
                        
        # Create summary report
        if all_results:
            self._create_summary(all_results, timestamp)
        
        # Debug summary - only for successful results
        for key, result in all_results.items():
            if 'data' in result:
                df = result['data']
                symbol = result['symbol']
                timeframe = result['timeframe']
                
                # Debug verification checks
                if 'is_pivot_high' in df.columns:
                    total_pivots_high = df['is_pivot_high'].sum()
                    
                    # Check retracement levels and hits for each level
                    for level in self.levels:
                        level_str = f'{int(level*1000)}'
                        
                        # Check high pivots
                        retr_col_high = f'retracement_{level_str}_from_high'
                        reached_col_high = f'reached_{level_str}_from_high'
                        
                        if retr_col_high in df.columns and reached_col_high in df.columns:
                            levels_calculated_high = (~df[retr_col_high].isna() & df['is_pivot_high']).sum()
                            
                            # Convert to boolean for consistent handling
                            reached_values = df[reached_col_high].fillna(False).astype(bool)
                            targets_reached_high = reached_values.sum()
                            
                            self.logger.info(f"{symbol} {timeframe} - High pivots: {total_pivots_high}")
                            self.logger.info(f"{symbol} {timeframe} - {level*100}% levels calculated: {levels_calculated_high}")
                            self.logger.info(f"{symbol} {timeframe} - {level*100}% targets reached: {targets_reached_high}")
                            
                            if targets_reached_high > 0 and levels_calculated_high > 0:
                                hit_rate = targets_reached_high / levels_calculated_high
                                self.logger.info(f"{symbol} {timeframe} - {level*100}% hit rate: {hit_rate:.2%}")
            
        return all_results
    
    def _sample_pivots(self, df, max_pivots=None, sample_rate=1.0):
        """
        Sample pivots for processing large datasets
        
        Args:
            df: DataFrame with pivot points identified
            max_pivots: Maximum number of pivots to process
            sample_rate: Rate to sample the data (0.0-1.0)
            
        Returns:
            DataFrame with sampled pivot points
        """
        # Apply sampling rate to entire dataset if less than 1.0
        if sample_rate < 1.0:
            self.logger.info(f"Sampling data at rate: {sample_rate}")
            # Create a sampled copy
            sample_size = int(len(df) * sample_rate)
            df = df.sample(n=sample_size)
            
        # Count pivots
        pivot_high_count = df['is_pivot_high'].sum()
        pivot_low_count = df['is_pivot_low'].sum()
        
        self.logger.info(f"Found {pivot_high_count} pivot highs and {pivot_low_count} pivot lows")
        
        # If max_pivots specified, sample pivot points
        if max_pivots is not None and pivot_high_count > max_pivots:
            self.logger.info(f"Limiting to {max_pivots} pivot highs")
            
            # Get indices of pivot highs
            high_indices = df[df['is_pivot_high']].index
            
            # Calculate sampling interval
            sample_interval = len(high_indices) // max_pivots
            
            # Sample pivot highs systematically
            sampled_high_indices = high_indices[::sample_interval][:max_pivots]
            
            # Reset all pivot flags then set only sampled ones
            df['is_pivot_high'] = False
            df.loc[sampled_high_indices, 'is_pivot_high'] = True
            
        # Repeat for pivot lows if needed
        if max_pivots is not None and pivot_low_count > max_pivots:
            self.logger.info(f"Limiting to {max_pivots} pivot lows")
            
            # Get indices of pivot lows
            low_indices = df[df['is_pivot_low']].index
            
            # Calculate sampling interval
            sample_interval = len(low_indices) // max_pivots
            
            # Sample pivot lows systematically
            sampled_low_indices = low_indices[::sample_interval][:max_pivots]
            
            # Reset all pivot flags then set only sampled ones
            df['is_pivot_low'] = False
            df.loc[sampled_low_indices, 'is_pivot_low'] = True
            
        # Count final pivots
        final_high_count = df['is_pivot_high'].sum()
        final_low_count = df['is_pivot_low'].sum()
        self.logger.info(f"Processing {final_high_count} pivot highs and {final_low_count} pivot lows")
        
        return df
        
    def _analyze_retracements(self, data_context, pivot_type='both', look_forward=48, 
                                pivot_window=5, max_pivots=None, sample_rate=1.0, 
                                pivot_method='fractal', fractal_min_window=5, fractal_max_windows=5, 
                                fractal_min_strength=1, chunk_size=1000000):
        """
        Analyze retracements from pivot points
        
        Args:
            data_context: DataContext with price data
            pivot_type: Type of pivots to analyze ('high', 'low', or 'both')
            look_forward: Number of bars to look forward for retracement targets
            pivot_window: Window size for simple pivot detection
            max_pivots: Maximum number of pivots to process (total across all chunks)
            sample_rate: Rate to sample the data (0.0-1.0)
            pivot_method: Method for identifying pivot points ('simple' or 'fractal')
            fractal_min_window: Minimum window size for fractal pattern (if pivot_method='fractal')
            fractal_max_windows: Number of Fibonacci window sizes to use for fractal detection
            fractal_min_strength: Minimum strength for fractal pattern (if pivot_method='fractal')
            chunk_size: Number of rows to process in each chunk for large datasets
        
        Returns:
            DataContext: New context with retracement analysis
        """
        try:
            # Make a copy of the data to avoid modifying the original
            df = data_context.df.copy()
            
            # Create a new data context for results
            result_context = DataContext(
                self.params,
                df,
                data_context.exchange,
                data_context.symbol,
                data_context.timeframe,
                source="retracement_analysis"
            )
            
            # Add the analysis parameters to processing history
            result_context.add_processing_step("retracement_analysis_params", {
                "pivot_type": pivot_type,
                "look_forward": look_forward, 
                "pivot_window": pivot_window,
                "retracement_levels": self.levels,
                "max_pivots": max_pivots,
                "sample_rate": sample_rate,
                "pivot_method": pivot_method,
                "fractal_min_window": fractal_min_window,
                "fractal_max_windows": fractal_max_windows,
                "fractal_min_strength": fractal_min_strength,
                "chunk_size": chunk_size
            })
            
            # Determine whether to use chunking based on dataset size
            use_chunking = len(df) > chunk_size
            
            if use_chunking:
                self.logger.info(f"Large dataset detected ({len(df)} rows). Using chunked processing.")
                
                # Initialize empty result dataframe with the same columns
                result_df = pd.DataFrame(columns=df.columns)
                
                # Calculate total chunks
                total_chunks = (len(df) + chunk_size - 1) // chunk_size
                
                # If max_pivots is specified, distribute across chunks
                max_pivots_per_chunk = None
                if max_pivots is not None:
                    max_pivots_per_chunk = max_pivots // total_chunks
                    self.logger.info(f"Processing approximately {max_pivots_per_chunk} pivots per chunk")
                
                # Process data in chunks
                for i in range(0, len(df), chunk_size):
                    chunk_num = i // chunk_size + 1
                    self.logger.info(f"Processing chunk {chunk_num}/{total_chunks} (rows {i}-{min(i+chunk_size, len(df))})")
                    
                    # Extract chunk
                    chunk = df.iloc[i:i+chunk_size].copy()
                    
                    # 1. Identify pivot points using selected method
                    if pivot_method == 'fractal':
                        # Match parameter order with non-chunked mode
                        self._identify_fractals(
                            chunk,                 # df
                            pivot_type=pivot_type, # pivot_type
                            look_forward=look_forward, # look_forward
                            min_window=fractal_min_window, # min_window
                            max_windows=fractal_max_windows, # max_windows  
                            min_strength=fractal_min_strength, # min_strength
                            is_vectorized=True     # is_vectorized
                        )
                    else:
                        self._identify_pivots(chunk, pivot_type, pivot_window)
                    
                    # 1.5. Sample pivots if needed but keep track of original data
                    # We sample the pivots but keep all rows
                    if max_pivots_per_chunk is not None or sample_rate < 1.0:
                        chunk = self._sample_pivots(chunk, max_pivots=max_pivots_per_chunk, sample_rate=sample_rate)
                    
                    # Only process chunks with pivots
                    has_pivots = False
                    if pivot_type in ['high', 'both'] and 'is_pivot_high' in chunk.columns:
                        has_pivots = has_pivots or chunk['is_pivot_high'].sum() > 0
                    if pivot_type in ['low', 'both'] and 'is_pivot_low' in chunk.columns:
                        has_pivots = has_pivots or chunk['is_pivot_low'].sum() > 0
                    
                    if has_pivots:
                        # 2. Calculate retracement levels
                        self._calculate_retracement_levels(chunk, pivot_type)
                        
                        # 3. Check if price reaches each retracement level
                        self._check_retracement_targets(chunk, pivot_type, look_forward)
                    
                    # Add ALL rows from this chunk to the result dataframe, not just pivots
                    result_df = pd.concat([result_df, chunk])
                    
                    self.logger.info(f"Completed chunk {chunk_num}/{total_chunks}")
                
                # Replace the result_context dataframe with the concatenated chunks
                result_context.df = result_df
                
            else:
                # Process the entire dataset at once
                
                # 1. Identify pivot points using selected method
                self.logger.info(f"Identifying pivot points using {pivot_method} method...")
                
                if pivot_method == 'fractal':
                    # Keep parameter order consistent with chunked mode
                    self._identify_fractals(
                        result_context.df,         # df
                        pivot_type=pivot_type,    # pivot_type
                        look_forward=look_forward, # look_forward
                        min_window=fractal_min_window, # min_window
                        max_windows=fractal_max_windows, # max_windows
                        min_strength=fractal_min_strength, # min_strength
                        is_vectorized=True         # is_vectorized
                    )
                else:
                    self._identify_pivots(result_context.df, pivot_type, pivot_window)
                    
                result_context.add_processing_step("identify_pivots", {
                    "pivot_type": pivot_type,
                    "method": pivot_method,
                    "window": fractal_min_window if pivot_method == 'fractal' else pivot_window
                })
                
                # 1.5 Sample pivots if needed (when max_pivots is specified to limit pivot count)
                if max_pivots is not None or sample_rate < 1.0:
                    self.logger.info("Sampling pivot points...")
                    result_context.df = self._sample_pivots(
                        result_context.df, 
                        max_pivots=max_pivots,
                        sample_rate=sample_rate
                    )
                    result_context.add_processing_step("sample_pivots", {
                        "max_pivots": max_pivots,
                        "sample_rate": sample_rate
                    })
                
                # Add progress indicators
                if pivot_type in ['high', 'both'] and 'is_pivot_high' in result_context.df.columns:
                    high_count = result_context.df['is_pivot_high'].sum()
                    self.logger.info(f"Processing {high_count} pivot highs")
                
                if pivot_type in ['low', 'both'] and 'is_pivot_low' in result_context.df.columns:
                    low_count = result_context.df['is_pivot_low'].sum()
                    self.logger.info(f"Processing {low_count} pivot lows")
                
                # 2. Calculate retracement levels for each pivot
                self.logger.info("Calculating retracement levels...")
                self._calculate_retracement_levels(result_context.df, pivot_type)
                result_context.add_processing_step("calculate_retracement_levels", {
                    "levels": self.levels
                })
                
                # 3. Check if price reaches each retracement level
                self.logger.info("Checking if retracement targets are reached...")
                self._check_retracement_targets(result_context.df, pivot_type, look_forward)
                result_context.add_processing_step("check_retracement_targets", {
                    "look_forward": look_forward
                })
            
            return result_context
            
        except Exception as e:
            self.logger.error(f"Error in retracement analysis: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
                    
    def _identify_pivots(self, df, pivot_type='both', window=5):
        """Vectorized version of pivot detection"""
        # Initialize pivot columns
        df['is_pivot_high'] = False
        df['is_pivot_low'] = False
        
        half_window = window // 2
        
        if pivot_type in ['high', 'both']:
            # Use rolling window max for pivot highs
            roll_max = df['high'].rolling(window=window, center=True).max()
            df.loc[df['high'] == roll_max, 'is_pivot_high'] = True
            
        if pivot_type in ['low', 'both']:
            # Use rolling window min for pivot lows
            roll_min = df['low'].rolling(window=window, center=True).min()
            df.loc[df['low'] == roll_min, 'is_pivot_low'] = True
        
        # Clean up NaN values from rolling window - Use iloc instead of loc with numeric indices
        df.iloc[:half_window, df.columns.get_indexer(['is_pivot_high', 'is_pivot_low'])] = False
        df.iloc[-half_window:, df.columns.get_indexer(['is_pivot_high', 'is_pivot_low'])] = False

    def _identify_fractals(self, df, pivot_type='both', look_forward=48, min_window=5, max_windows=5, 
                            min_strength=1, is_vectorized=True):
        """
        Identify pivot points using Fibonacci window sizes for Bill Williams' fractal pattern
        
        Args:
            df: DataFrame with price data
            pivot_type: 'high', 'low', or 'both'
            look_forward: Look forward parameter for analysis
            min_window: Minimum window size (must be odd, >= 5)
            max_windows: Maximum number of Fibonacci window sizes to use
            min_strength: Minimum fractal strength to keep (number of window sizes that confirm the fractal)
            is_vectorized: Whether to use vectorized implementation (faster) or loop-based implementation
        
        Adds columns to the DataFrame:
            - is_pivot_high: Boolean indicating pivot high (bullish fractal) points
            - is_pivot_low: Boolean indicating pivot low (bearish fractal) points
            - fractal_strength_high: Integer indicating strength of high fractals (how many window sizes confirm)
            - fractal_strength_low: Integer indicating strength of low fractals (how many window sizes confirm)
        """
        self.logger.info("Identifying fractals")
        # Ensure min_window is valid (odd, at least 5)
        if min_window < 5:
            min_window = 5
        if min_window % 2 == 0:
            min_window += 1
    
        # Generate Fibonacci-like sequence for window sizes starting from min_window
        window_sizes = [min_window]  # Start with minimum window
        
        # Find the two closest Fibonacci-like values to start our sequence
        # Find the two starting values that would maintain the Fibonacci ratio (approx 1.618)
        a = min_window
        b = round(min_window * 1.618)  # Next value using golden ratio
        
        # Ensure b is odd (for fractal detection)
        if b % 2 == 0:
            # If it's even, decide which odd number is closer
            if (min_window * 1.618) < b:  # Actual value is less than rounded value
                b -= 1  # Go down to the odd number below
            else:
                b += 1  # Go up to the odd number above
        
        # Generate the remaining sequence
        while len(window_sizes) < max_windows:
            window_sizes.append(b)
            a, b = b, a + b  # Standard Fibonacci recursion formula
            if b % 2 == 0:
                b += 1  # Make sure all values are odd
        
        # Filter out window sizes larger than look_forward
        filtered_window_sizes = []
        for size in window_sizes:
            if size > look_forward:
                self.logger.info(f"Skipping window size {size} as it is larger than look_forward ({look_forward})")
                continue
            filtered_window_sizes.append(size)
        
        if not filtered_window_sizes:
            self.logger.warning("No valid window sizes after filtering! Using default window size of 5.")
            filtered_window_sizes = [5]
        
        self.logger.info(f"Using Fibonacci window sizes for fractals: {filtered_window_sizes}")
        
        # Initialize pivot and strength columns
        df['is_pivot_high'] = False
        df['is_pivot_low'] = False
        df['fractal_strength_high'] = 0
        df['fractal_strength_low'] = 0
        
        # Process each Fibonacci window size
        for window_size in filtered_window_sizes:
            # Window size must be odd
            if window_size % 2 == 0:
                window_size += 1
                
            self.logger.debug(f"Processing window size {window_size}")
            
            # Create temporary columns for this window size
            temp_high_col = f'temp_high_{window_size}'
            temp_low_col = f'temp_low_{window_size}'
            df[temp_high_col] = False
            df[temp_low_col] = False
            
            if is_vectorized:
                # Vectorized implementation for this window size
                self._identify_fractals_vectorized(df, window_size, pivot_type,
                                                        temp_high_col, temp_low_col)
            else:
                # Non-vectorized implementation
                self._identify_fractals_looped(df, window_size, pivot_type,
                                                    temp_high_col, temp_low_col)
        
        # Combine results from all window sizes
        if pivot_type in ['high', 'both']:
            for window_size in filtered_window_sizes:
                # Adjust to odd window size if it was originally even
                if window_size % 2 == 0:
                    window_size += 1
                    
                temp_high_col = f'temp_high_{window_size}'
                # Where this window size found a fractal, increment strength and mark as pivot
                mask = df[temp_high_col]
                df.loc[mask, 'fractal_strength_high'] += 1
                # Drop temporary column
                df.drop(temp_high_col, axis=1, inplace=True)
            
            # Mark as pivot high where strength meets minimum threshold
            df.loc[df['fractal_strength_high'] >= min_strength, 'is_pivot_high'] = True
            
        if pivot_type in ['low', 'both']:
            for window_size in filtered_window_sizes:
                # Adjust to odd window size if it was originally even
                if window_size % 2 == 0:
                    window_size += 1
                    
                temp_low_col = f'temp_low_{window_size}'
                # Where this window size found a fractal, increment strength and mark as pivot
                mask = df[temp_low_col]
                df.loc[mask, 'fractal_strength_low'] += 1
                # Drop temporary column
                df.drop(temp_low_col, axis=1, inplace=True)
            
            # Mark as pivot low where strength meets minimum threshold
            df.loc[df['fractal_strength_low'] >= min_strength, 'is_pivot_low'] = True
        
        # Log statistics
        high_count = df['is_pivot_high'].sum() if 'is_pivot_high' in df.columns else 0
        low_count = df['is_pivot_low'].sum() if 'is_pivot_low' in df.columns else 0
        
        self.logger.info(f"Found {high_count} high fractals and {low_count} low fractals with min strength {min_strength}")
        
        # Show distribution of fractal strengths
        if pivot_type in ['high', 'both'] and high_count > 0:
            strength_counts = df['fractal_strength_high'].value_counts().sort_index()
            self.logger.info(f"High fractal strength distribution: {strength_counts.to_dict()}")
            
        if pivot_type in ['low', 'both'] and low_count > 0:
            strength_counts = df['fractal_strength_low'].value_counts().sort_index()
            self.logger.info(f"Low fractal strength distribution: {strength_counts.to_dict()}")

    def _identify_fractals_vectorized(self, df, window_size, pivot_type, high_col, low_col):
        """
        Vectorized implementation to identify fractal patterns for a specific window size
        
        Args:
            df: DataFrame with price data
            window_size: Window size for fractal pattern
            pivot_type: 'high', 'low', or 'both'
            high_col: Column name to store high fractal results
            low_col: Column name to store low fractal results
        """
        half_window = window_size // 2
        
        # Create shifted columns for comparison
        price_data = pd.DataFrame(index=df.index)
        
        if pivot_type in ['high', 'both']:
            # For bullish (high) fractals
            price_data['high'] = df['high']
            for i in range(1, half_window + 1):
                price_data[f'high_m{i}'] = df['high'].shift(i)
                price_data[f'high_p{i}'] = df['high'].shift(-i)
            
            # Create mask for where center is higher than all others
            high_mask = True
            for i in range(1, half_window + 1):
                high_mask = high_mask & (price_data['high'] > price_data[f'high_m{i}']) & \
                            (price_data['high'] > price_data[f'high_p{i}'])
            
            # Apply mask to identify fractals
            df.loc[high_mask, high_col] = True
        
        if pivot_type in ['low', 'both']:
            # For bearish (low) fractals
            price_data['low'] = df['low']
            for i in range(1, half_window + 1):
                price_data[f'low_m{i}'] = df['low'].shift(i)
                price_data[f'low_p{i}'] = df['low'].shift(-i)
            
            # Create mask for where center is lower than all others
            low_mask = True
            for i in range(1, half_window + 1):
                low_mask = low_mask & (price_data['low'] < price_data[f'low_m{i}']) & \
                            (price_data['low'] < price_data[f'low_p{i}'])
            
            # Apply mask to identify fractals
            df.loc[low_mask, low_col] = True
        
        # Clear NaN values at the edges
        df.iloc[:half_window, df.columns.get_indexer([high_col, low_col])] = False
        df.iloc[-half_window:, df.columns.get_indexer([high_col, low_col])] = False

    def _identify_fractals_looped(self, df, window_size, pivot_type, high_col, low_col):
        """
        Non-vectorized implementation to identify fractal patterns for a specific window size
        (Slower but more memory efficient for very large datasets)
        
        Args:
            df: DataFrame with price data
            window_size: Window size for fractal pattern
            pivot_type: 'high', 'low', or 'both'
            high_col: Column name to store high fractal results
            low_col: Column name to store low fractal results
        """
        half_window = window_size // 2
        
        # Identify bullish (high) fractals
        if pivot_type in ['high', 'both']:
            for i in range(half_window, len(df) - half_window):
                # Center candle's high
                center_high = df.iloc[i]['high']
                
                # Check if center candle's high is higher than all surrounding candles
                is_fractal = True
                
                # Check left side (at least 2 bars)
                for j in range(1, half_window + 1):
                    if df.iloc[i-j]['high'] >= center_high:
                        is_fractal = False
                        break
                        
                # If left side checked out, check right side (at least 2 bars)
                if is_fractal:
                    for j in range(1, half_window + 1):
                        if df.iloc[i+j]['high'] >= center_high:
                            is_fractal = False
                            break
                
                # If it's a fractal, mark it
                if is_fractal:
                    df.loc[df.index[i], high_col] = True
        
        # Identify bearish (low) fractals
        if pivot_type in ['low', 'both']:
            for i in range(half_window, len(df) - half_window):
                # Center candle's low
                center_low = df.iloc[i]['low']
                
                # Check if center candle's low is lower than all surrounding candles
                is_fractal = True
                
                # Check left side (at least 2 bars)
                for j in range(1, half_window + 1):
                    if df.iloc[i-j]['low'] <= center_low:
                        is_fractal = False
                        break
                        
                # If left side checked out, check right side (at least 2 bars)
                if is_fractal:
                    for j in range(1, half_window + 1):
                        if df.iloc[i+j]['low'] <= center_low:
                            is_fractal = False
                            break
                
                # If it's a fractal, mark it
                if is_fractal:
                    df.loc[df.index[i], low_col] = True
    
    def _calculate_retracement_levels(self, df, pivot_type='both', min_range_percent=0.1):
        """
        Calculate retracement levels for each identified pivot point
        
        Args:
            df: DataFrame with price data and pivot points
            pivot_type: 'high', 'low', or 'both'
            min_range_percent: Minimum range size as percent of price (default 0.1%)
            
        Adds retracement level columns to the DataFrame
        """
        # Create retracement columns if they don't exist
        for level in self.levels:
            level_str = f'{int(level*1000)}'
            
            # For high pivots
            if pivot_type in ['high', 'both']:
                col_name = f'retracement_{level_str}_from_high'
                if col_name not in df.columns:
                    df[col_name] = np.nan
                    
            # For low pivots
            if pivot_type in ['low', 'both']:
                col_name = f'retracement_{level_str}_from_low'
                if col_name not in df.columns:
                    df[col_name] = np.nan
        
        # For pivot highs (retracement down)
        if pivot_type in ['high', 'both']:
            # Get indices of pivot highs
            pivot_high_indices = df[df['is_pivot_high']].index
            
            for idx in pivot_high_indices:
                pivot_high = df.loc[idx, 'high']
                
                # Find the last pivot low before this pivot high
                prev_pivot_lows = df[(df.index < idx) & (df['is_pivot_low'])]
                
                if not prev_pivot_lows.empty:
                    last_pivot_low = prev_pivot_lows.iloc[-1]['low']
                    price_range = pivot_high - last_pivot_low
                    
                    # Check if range is significant enough (at least min_range_percent of price)
                    min_range = pivot_high * min_range_percent / 100
                    
                    if price_range >= min_range:
                        # Calculate retracement levels
                        for level in self.levels:
                            retracement_price = pivot_high - (price_range * level)
                            col_name = f'retracement_{int(level*1000)}_from_high'
                            
                            # Add the retracement price at the pivot point
                            df.loc[idx, col_name] = retracement_price
                    else:
                        self.logger.debug(f"Skipping high pivot at {idx} - range too small: {price_range:.2f} < {min_range:.2f}")
        
        # For pivot lows (retracement up)
        if pivot_type in ['low', 'both']:
            # Get indices of pivot lows
            pivot_low_indices = df[df['is_pivot_low']].index
            
            for idx in pivot_low_indices:
                pivot_low = df.loc[idx, 'low']
                
                # Find the last pivot high before this pivot low
                prev_pivot_highs = df[(df.index < idx) & (df['is_pivot_high'])]
                
                if not prev_pivot_highs.empty:
                    last_pivot_high = prev_pivot_highs.iloc[-1]['high']
                    price_range = last_pivot_high - pivot_low
                    
                    # Check if range is significant enough (at least min_range_percent of price)
                    min_range = pivot_low * min_range_percent / 100
                    
                    if price_range >= min_range:
                        # Calculate retracement levels
                        for level in self.levels:
                            retracement_price = pivot_low + (price_range * level)
                            col_name = f'retracement_{int(level*1000)}_from_low'
                            
                            # Add the retracement price at the pivot point
                            df.loc[idx, col_name] = retracement_price
                    else:
                        self.logger.debug(f"Skipping low pivot at {idx} - range too small: {price_range:.2f} < {min_range:.2f}")
    
    def _check_retracement_targets(self, df, pivot_type='both', look_forward=48):
        """
        Check if price reaches each retracement level after pivot
        
        Args:
            df: DataFrame with price data and retracement levels
            pivot_type: 'high', 'low', or 'both'
            look_forward: Number of bars to look forward
                
        Returns:
            None - Updates df with target reached information
        """
        # Track hits for debugging
        hit_counts = {level: 0 for level in self.levels}
        
        # For pivot highs (retracement down)
        if pivot_type in ['high', 'both']:
            # Ensure all required columns exist for high pivots
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                retracement_col = f'retracement_{level_str}_from_high'
                reached_col = f'reached_{level_str}_from_high'
                bars_col = f'bars_to_{level_str}_from_high'
                
                # Create columns if they don't exist with proper data types
                if retracement_col not in df.columns:
                    self.logger.warning(f"Creating missing column {retracement_col}")
                    df[retracement_col] = np.nan
                
                if reached_col not in df.columns:
                    # Initialize with boolean values to avoid type warnings
                    df[reached_col] = False
                    
                if bars_col not in df.columns:
                    df[bars_col] = np.nan
            
            # Get indices of pivot highs
            pivot_high_indices = df[df['is_pivot_high']].index
            
            pivot_count = 0
            for idx in pivot_high_indices:
                pivot_count += 1
                pivot_loc = df.index.get_loc(idx)
                
                # Skip if we're too close to the end of the dataframe
                if pivot_loc + look_forward >= len(df):
                    continue
                
                # Get forward price data
                forward_df = df.iloc[pivot_loc+1:pivot_loc+look_forward+1]
                
                # Check each retracement level
                for level in self.levels:
                    level_str = f'{int(level*1000)}'
                    retracement_col = f'retracement_{level_str}_from_high'
                    reached_col = f'reached_{level_str}_from_high'
                    bars_col = f'bars_to_{level_str}_from_high'
                    
                    # Skip if the retracement level wasn't calculated for this pivot
                    if pd.isna(df.loc[idx, retracement_col]):
                        continue
                    
                    # Get the target price
                    target_price = df.loc[idx, retracement_col]
                    
                    # Check if the low price of any forward bar reaches the target
                    reached_mask = forward_df['low'] <= target_price
                    
                    if reached_mask.any():
                        # Target was reached
                        reached = True
                        
                        # Find first bar where target was reached
                        first_reach_idx = forward_df.index[reached_mask][0]
                        bars_to_reach = df.index.get_loc(first_reach_idx) - pivot_loc
                        
                        # Update hit counts for logging
                        hit_counts[level] += 1
                    else:
                        # Target was not reached
                        reached = False
                        bars_to_reach = None
                    
                    # Store the results - convert bool to int to match column type
                    df.loc[idx, reached_col] = reached
                    if reached:
                        df.loc[idx, bars_col] = bars_to_reach
            
            # Log hit counts for debugging
            if pivot_count > 0:
                self.logger.info(f"Pivot highs analyzed: {pivot_count}")
                for level, count in hit_counts.items():
                    self.logger.info(f"Level {level*100}% hit count: {count}")
        
        # For pivot lows (retracement up)
        if pivot_type in ['low', 'both']:
            # Ensure all required columns exist for low pivots
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                retracement_col = f'retracement_{level_str}_from_low'
                reached_col = f'reached_{level_str}_from_low'
                bars_col = f'bars_to_{level_str}_from_low'
                
                # Create columns if they don't exist with proper data types
                if retracement_col not in df.columns:
                    self.logger.warning(f"Creating missing column {retracement_col}")
                    df[retracement_col] = np.nan
                
                if reached_col not in df.columns:
                    # Initialize with boolean values to avoid type warnings
                    df[reached_col] = False
                    
                if bars_col not in df.columns:
                    df[bars_col] = np.nan
                    
            # Reset hit counts for pivot lows
            hit_counts = {level: 0 for level in self.levels}
            
            # Get indices of pivot lows
            pivot_low_indices = df[df['is_pivot_low']].index
            
            pivot_count = 0
            for idx in pivot_low_indices:
                pivot_count += 1
                pivot_loc = df.index.get_loc(idx)
                
                # Skip if we're too close to the end of the dataframe
                if pivot_loc + look_forward >= len(df):
                    continue
                
                # Get forward price data
                forward_df = df.iloc[pivot_loc+1:pivot_loc+look_forward+1]
                
                # Check each retracement level
                for level in self.levels:
                    level_str = f'{int(level*1000)}'
                    retracement_col = f'retracement_{level_str}_from_low'
                    reached_col = f'reached_{level_str}_from_low'
                    bars_col = f'bars_to_{level_str}_from_low'
                    
                    # Skip if the retracement level wasn't calculated for this pivot
                    if pd.isna(df.loc[idx, retracement_col]):
                        continue
                    
                    # Get the target price
                    target_price = df.loc[idx, retracement_col]
                    
                    # Check if the high price of any forward bar reaches the target
                    reached_mask = forward_df['high'] >= target_price
                    
                    if reached_mask.any():
                        # Target was reached
                        reached = True
                        
                        # Find first bar where target was reached
                        first_reach_idx = forward_df.index[reached_mask][0]
                        bars_to_reach = df.index.get_loc(first_reach_idx) - pivot_loc
                        
                        # Update hit counts for logging
                        hit_counts[level] += 1
                    else:
                        # Target was not reached
                        reached = False
                        bars_to_reach = None
                    
                    # Store the results
                    df.loc[idx, reached_col] = reached
                    if reached:
                        df.loc[idx, bars_col] = bars_to_reach
            
            # Log hit counts for debugging
            if pivot_count > 0:
                self.logger.info(f"Pivot lows analyzed: {pivot_count}")
                for level, count in hit_counts.items():
                    self.logger.info(f"Level {level*100}% hit count: {count}")
                    
    def _calculate_time_decay_stats(self, df):
        """
        Calculate how retracement probabilities change with time
        
        Args:
            df: DataFrame with retracement analysis results
            
        Returns:
            dict: Time-based probability statistics
        """
        # Define Fibonacci sequence of bar counts to check
        fibonacci_bars = [1, 2, 3, 5, 8, 13, 21, 34]
        
        time_decay_stats = {}
        
        # Process high pivots
        if 'is_pivot_high' in df.columns and df['is_pivot_high'].sum() > 0:
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                bars_col = f'bars_to_{level_str}_from_high'
                
                # Skip if the column doesn't exist
                if bars_col not in df.columns:
                    continue
                
                # Get all valid pivot highs with retracement levels calculated
                retr_col = f'retracement_{level_str}_from_high'
                valid_rows = df[df['is_pivot_high'] & ~df[retr_col].isna()]
                total_valid = len(valid_rows)
                
                if total_valid > 0:
                    # Calculate cumulative probabilities for each time threshold
                    for bar_count in fibonacci_bars:
                        # Count pivots that reached the level within this bar count
                        reached_within = valid_rows[valid_rows[bars_col] <= bar_count][bars_col].count()
                        
                        # Count pivots that reached the level after this bar count
                        reached_after = valid_rows[(valid_rows[bars_col] > bar_count) & ~valid_rows[bars_col].isna()][bars_col].count()
                        
                        # Calculate remaining unreached pivots (these reached the level after bar_count or never reached it)
                        remaining_unreached = total_valid - reached_within
                        
                        # Calculate probability of still reaching the target after bar_count bars
                        if remaining_unreached > 0:
                            prob_after = reached_after / remaining_unreached
                        else:
                            prob_after = 0.0
                        
                        # Store statistics
                        time_decay_stats[f'high_{level_str}_prob_after_{bar_count}_bars'] = prob_after
                        time_decay_stats[f'high_{level_str}_count_after_{bar_count}_bars'] = int(reached_after)
                        time_decay_stats[f'high_{level_str}_remaining_after_{bar_count}_bars'] = int(remaining_unreached)
        
        # Process low pivots (similar approach)
        if 'is_pivot_low' in df.columns and df['is_pivot_low'].sum() > 0:
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                bars_col = f'bars_to_{level_str}_from_low'
                
                # Skip if the column doesn't exist
                if bars_col not in df.columns:
                    continue
                
                # Get all valid pivot lows with retracement levels calculated
                retr_col = f'retracement_{level_str}_from_low'
                valid_rows = df[df['is_pivot_low'] & ~df[retr_col].isna()]
                total_valid = len(valid_rows)
                
                if total_valid > 0:
                    # Calculate cumulative probabilities for each time threshold
                    for bar_count in fibonacci_bars:
                        # Count pivots that reached the level within this bar count
                        reached_within = valid_rows[valid_rows[bars_col] <= bar_count][bars_col].count()
                        
                        # Count pivots that reached the level after this bar count
                        reached_after = valid_rows[(valid_rows[bars_col] > bar_count) & ~valid_rows[bars_col].isna()][bars_col].count()
                        
                        # Calculate remaining unreached pivots (these reached the level after bar_count or never reached it)
                        remaining_unreached = total_valid - reached_within
                        
                        # Calculate probability of still reaching the target after bar_count bars
                        if remaining_unreached > 0:
                            prob_after = reached_after / remaining_unreached
                        else:
                            prob_after = 0.0
                        
                        # Store statistics
                        time_decay_stats[f'low_{level_str}_prob_after_{bar_count}_bars'] = prob_after
                        time_decay_stats[f'low_{level_str}_count_after_{bar_count}_bars'] = int(reached_after)
                        time_decay_stats[f'low_{level_str}_remaining_after_{bar_count}_bars'] = int(remaining_unreached)
        
        return time_decay_stats
        
    def _calculate_stats(self, df):
        """
        Calculate retracement statistics with breakdown by fractal strength
        
        Args:
            df: DataFrame with retracement analysis
            
        Returns:
            dict: Dictionary with statistics
        """
        stats = {}
        
        # Statistics for pivot highs (retracement down)
        pivot_high_count = df['is_pivot_high'].sum()
        
        if pivot_high_count > 0:
            stats['pivot_high_count'] = int(pivot_high_count)
            
            # First calculate overall probabilities (as before)
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                reached_col = f'reached_{level_str}_from_high'
                bars_col = f'bars_to_{level_str}_from_high'
                
                # Skip if the column doesn't exist
                if reached_col not in df.columns:
                    continue
                
                # Only consider rows where the retracement level was calculated
                retr_col = f'retracement_{level_str}_from_high'
                valid_rows = df[df['is_pivot_high'] & ~df[retr_col].isna()]
                
                if len(valid_rows) > 0:
                    # Calculate probability of reaching this level
                    # Fix for FutureWarning: use infer_objects() instead of astype(bool)
                    reached_values = valid_rows[reached_col].fillna(False)
                    reached_values = reached_values.infer_objects(copy=False).astype(bool)
                    reached_count = reached_values.sum()
                    reached_prob = reached_count / len(valid_rows)
                    
                    # Calculate average bars to reach the level
                    bars_to_reach = valid_rows[bars_col].dropna()
                    avg_bars = bars_to_reach.mean() if len(bars_to_reach) > 0 else None
                    
                    stats[f'high_{level_str}_prob'] = reached_prob
                    stats[f'high_{level_str}_count'] = int(reached_count)
                    stats[f'high_{level_str}_avg_bars'] = avg_bars
            
            # Now calculate probabilities by fractal strength
            if 'fractal_strength_high' in df.columns:
                # Get unique strength values
                strength_values = sorted(df['fractal_strength_high'].unique())
                # Convert to Python integers to avoid numpy type issues
                strength_values = [int(s) for s in strength_values if s > 0]
                
                for strength in strength_values:
                    # Skip strength 0 (not a fractal)
                    if strength == 0:
                        continue
                        
                    # Calculate stats for this strength
                    strength_rows = df[df['fractal_strength_high'] == strength]
                    strength_pivot_count = strength_rows['is_pivot_high'].sum()
                    
                    if strength_pivot_count > 0:
                        stats[f'pivot_high_strength_{strength}_count'] = int(strength_pivot_count)
                        
                        for level in self.levels:
                            level_str = f'{int(level*1000)}'
                            reached_col = f'reached_{level_str}_from_high'
                            bars_col = f'bars_to_{level_str}_from_high'
                            
                            # Skip if the column doesn't exist
                            if reached_col not in df.columns:
                                continue
                            
                            # Only consider rows where the retracement level was calculated
                            retr_col = f'retracement_{level_str}_from_high'
                            valid_rows = strength_rows[strength_rows['is_pivot_high'] & ~strength_rows[retr_col].isna()]
                            
                            if len(valid_rows) > 0:
                                # Fix for FutureWarning: use infer_objects() instead of astype(bool)
                                reached_values = valid_rows[reached_col].fillna(False)
                                reached_values = reached_values.infer_objects(copy=False).astype(bool)
                                reached_count = reached_values.sum()
                                reached_prob = reached_count / len(valid_rows)
                                
                                # Calculate average bars to reach the level
                                bars_to_reach = valid_rows[bars_col].dropna()
                                avg_bars = bars_to_reach.mean() if len(bars_to_reach) > 0 else None
                                
                                stats[f'high_{level_str}_strength_{strength}_prob'] = reached_prob
                                stats[f'high_{level_str}_strength_{strength}_count'] = int(reached_count)
                                stats[f'high_{level_str}_strength_{strength}_avg_bars'] = avg_bars
                
                # Also calculate stats for strength >= X (cumulative)
                if strength_values:  # Check if we have any valid strength values
                    for min_strength in range(1, max(strength_values) + 1):
                        strength_rows = df[df['fractal_strength_high'] >= min_strength]
                        strength_pivot_count = strength_rows['is_pivot_high'].sum()
                        
                        if strength_pivot_count > 0:
                            stats[f'pivot_high_strength_ge_{min_strength}_count'] = int(strength_pivot_count)
                            
                            for level in self.levels:
                                level_str = f'{int(level*1000)}'
                                reached_col = f'reached_{level_str}_from_high'
                                bars_col = f'bars_to_{level_str}_from_high'
                                
                                # Skip if the column doesn't exist
                                if reached_col not in df.columns:
                                    continue
                                
                                # Only consider rows where the retracement level was calculated
                                retr_col = f'retracement_{level_str}_from_high'
                                valid_rows = strength_rows[strength_rows['is_pivot_high'] & ~strength_rows[retr_col].isna()]
                                
                                if len(valid_rows) > 0:
                                    # Fix for FutureWarning: use infer_objects() instead of astype(bool)
                                    reached_values = valid_rows[reached_col].fillna(False)
                                    reached_values = reached_values.infer_objects(copy=False).astype(bool)
                                    reached_count = reached_values.sum()
                                    reached_prob = reached_count / len(valid_rows)
                                    
                                    # Calculate average bars to reach the level
                                    bars_to_reach = valid_rows[bars_col].dropna()
                                    avg_bars = bars_to_reach.mean() if len(bars_to_reach) > 0 else None
                                    
                                    stats[f'high_{level_str}_strength_ge_{min_strength}_prob'] = reached_prob
                                    stats[f'high_{level_str}_strength_ge_{min_strength}_count'] = int(reached_count)
                                    stats[f'high_{level_str}_strength_ge_{min_strength}_avg_bars'] = avg_bars
        
        # Statistics for pivot lows (retracement up) - same approach
        pivot_low_count = df['is_pivot_low'].sum()
        
        if pivot_low_count > 0:
            stats['pivot_low_count'] = int(pivot_low_count)
            
            # First calculate overall probabilities (as before)
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                reached_col = f'reached_{level_str}_from_low'
                bars_col = f'bars_to_{level_str}_from_low'
                
                # Skip if the column doesn't exist
                if reached_col not in df.columns:
                    continue
                
                # Only consider rows where the retracement level was calculated
                retr_col = f'retracement_{level_str}_from_low'
                valid_rows = df[df['is_pivot_low'] & ~df[retr_col].isna()]
                
                if len(valid_rows) > 0:
                    # Fix for FutureWarning: use infer_objects() instead of astype(bool)
                    reached_values = valid_rows[reached_col].fillna(False)
                    reached_values = reached_values.infer_objects(copy=False).astype(bool)
                    reached_count = reached_values.sum()
                    reached_prob = reached_count / len(valid_rows)
                    
                    # Calculate average bars to reach the level
                    bars_to_reach = valid_rows[bars_col].dropna()
                    avg_bars = bars_to_reach.mean() if len(bars_to_reach) > 0 else None
                    
                    stats[f'low_{level_str}_prob'] = reached_prob
                    stats[f'low_{level_str}_count'] = int(reached_count)
                    stats[f'low_{level_str}_avg_bars'] = avg_bars
            
            # Now calculate probabilities by fractal strength
            if 'fractal_strength_low' in df.columns:
                # Get unique strength values
                strength_values = sorted(df['fractal_strength_low'].unique())
                # Convert to Python integers to avoid numpy type issues
                strength_values = [int(s) for s in strength_values if s > 0]
                
                for strength in strength_values:
                    # Skip strength 0 (not a fractal)
                    if strength == 0:
                        continue
                        
                    # Calculate stats for this strength
                    strength_rows = df[df['fractal_strength_low'] == strength]
                    strength_pivot_count = strength_rows['is_pivot_low'].sum()
                    
                    if strength_pivot_count > 0:
                        stats[f'pivot_low_strength_{strength}_count'] = int(strength_pivot_count)
                        
                        for level in self.levels:
                            level_str = f'{int(level*1000)}'
                            reached_col = f'reached_{level_str}_from_low'
                            bars_col = f'bars_to_{level_str}_from_low'
                            
                            # Skip if the column doesn't exist
                            if reached_col not in df.columns:
                                continue
                            
                            # Only consider rows where the retracement level was calculated
                            retr_col = f'retracement_{level_str}_from_low'
                            valid_rows = strength_rows[strength_rows['is_pivot_low'] & ~strength_rows[retr_col].isna()]
                            
                            if len(valid_rows) > 0:
                                # Fix for FutureWarning: use infer_objects() instead of astype(bool)
                                reached_values = valid_rows[reached_col].fillna(False)
                                reached_values = reached_values.infer_objects(copy=False).astype(bool)
                                reached_count = reached_values.sum()
                                reached_prob = reached_count / len(valid_rows)
                                
                                # Calculate average bars to reach the level
                                bars_to_reach = valid_rows[bars_col].dropna()
                                avg_bars = bars_to_reach.mean() if len(bars_to_reach) > 0 else None
                                
                                stats[f'low_{level_str}_strength_{strength}_prob'] = reached_prob
                                stats[f'low_{level_str}_strength_{strength}_count'] = int(reached_count)
                                stats[f'low_{level_str}_strength_{strength}_avg_bars'] = avg_bars
                
                # Also calculate stats for strength >= X (cumulative)
                if strength_values:  # Check if we have any valid strength values
                    for min_strength in range(1, max(strength_values) + 1):
                        strength_rows = df[df['fractal_strength_low'] >= min_strength]
                        strength_pivot_count = strength_rows['is_pivot_low'].sum()
                        
                        if strength_pivot_count > 0:
                            stats[f'pivot_low_strength_ge_{min_strength}_count'] = int(strength_pivot_count)
                            
                            for level in self.levels:
                                level_str = f'{int(level*1000)}'
                                reached_col = f'reached_{level_str}_from_low'
                                bars_col = f'bars_to_{level_str}_from_low'
                                
                                # Skip if the column doesn't exist
                                if reached_col not in df.columns:
                                    continue
                                
                                # Only consider rows where the retracement level was calculated
                                retr_col = f'retracement_{level_str}_from_low'
                                valid_rows = strength_rows[strength_rows['is_pivot_low'] & ~strength_rows[retr_col].isna()]
                                
                                if len(valid_rows) > 0:
                                    # Fix for FutureWarning: use infer_objects() instead of astype(bool)
                                    reached_values = valid_rows[reached_col].fillna(False)
                                    reached_values = reached_values.infer_objects(copy=False).astype(bool)
                                    reached_count = reached_values.sum()
                                    reached_prob = reached_count / len(valid_rows)
                                    
                                    # Calculate average bars to reach the level
                                    bars_to_reach = valid_rows[bars_col].dropna()
                                    avg_bars = bars_to_reach.mean() if len(bars_to_reach) > 0 else None
                                    
                                    stats[f'low_{level_str}_strength_ge_{min_strength}_prob'] = reached_prob
                                    stats[f'low_{level_str}_strength_ge_{min_strength}_count'] = int(reached_count)
                                    stats[f'low_{level_str}_strength_ge_{min_strength}_avg_bars'] = avg_bars
        
        return stats
        
    def _create_visualizations(self, result_context, key, timestamp):
        """
        Create visualizations for retracement analysis
        
        Args:
            result_context: DataContext with analysis results
            key: Key for the symbol and timeframe
            timestamp: Timestamp string
        """
        df = result_context.df
        
        # Create figure for retracement probabilities
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        
        # Plot for pivot highs (retracement down)
        if 'is_pivot_high' in df.columns and df['is_pivot_high'].sum() > 0:
            ax = axes[0]
            
            # Calculate probabilities for each level
            probs = []
            level_labels = []
            counts = []
            
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                reached_col = f'reached_{level_str}_from_high'
                
                # Skip if the column doesn't exist
                if reached_col not in df.columns:
                    continue
                
                # Only consider rows where the retracement level was calculated
                retr_col = f'retracement_{level_str}_from_high'
                valid_rows = df[df['is_pivot_high'] & ~df[retr_col].isna()]
                
                if len(valid_rows) > 0:
                    # Calculate probability of reaching this level
                    # Convert to boolean for consistent handling
                    reached_values = valid_rows[reached_col].fillna(False).astype(bool)
                    reached_count = reached_values.sum()
                    reached_prob = reached_count / len(valid_rows)
                    
                    probs.append(reached_prob)
                    level_labels.append(f'{level*100:.1f}%')
                    counts.append(int(reached_count))
            
            # Create bar chart
            bars = ax.bar(level_labels, probs, color='red', alpha=0.7)
            
            # Add count labels above bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f'n={count}',
                    ha='center', va='bottom'
                )
            
            ax.set_title(f'Probability of Price Reaching Retracement Levels from Pivot Highs\n({df["is_pivot_high"].sum()} pivots)')
            ax.set_xlabel('Retracement Level')
            ax.set_ylabel('Probability')
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            # Add percentage labels on bars
            for i, prob in enumerate(probs):
                ax.text(
                    i,
                    prob / 2,
                    f'{prob:.1%}',
                    ha='center', va='center',
                    color='white', fontweight='bold'
                )
        
        # Plot for pivot lows (retracement up)
        if 'is_pivot_low' in df.columns and df['is_pivot_low'].sum() > 0:
            ax = axes[1]
            
            # Calculate probabilities for each level
            probs = []
            level_labels = []
            counts = []
            
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                reached_col = f'reached_{level_str}_from_low'
                
                # Skip if the column doesn't exist
                if reached_col not in df.columns:
                    continue
                
                # Only consider rows where the retracement level was calculated
                retr_col = f'retracement_{level_str}_from_low'
                valid_rows = df[df['is_pivot_low'] & ~df[retr_col].isna()]
                
                if len(valid_rows) > 0:
                    # Calculate probability of reaching this level
                    # Convert to boolean for consistent handling
                    reached_values = valid_rows[reached_col].fillna(False).astype(bool)
                    reached_count = reached_values.sum()
                    reached_prob = reached_count / len(valid_rows)
                    
                    probs.append(reached_prob)
                    level_labels.append(f'{level*100:.1f}%')
                    counts.append(int(reached_count))
            
            # Create bar chart
            bars = ax.bar(level_labels, probs, color='green', alpha=0.7)
            
            # Add count labels above bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f'n={count}',
                    ha='center', va='bottom'
                )
            
            ax.set_title(f'Probability of Price Reaching Retracement Levels from Pivot Lows\n({df["is_pivot_low"].sum()} pivots)')
            ax.set_xlabel('Retracement Level')
            ax.set_ylabel('Probability')
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            # Add percentage labels on bars
            for i, prob in enumerate(probs):
                ax.text(
                    i,
                    prob / 2,
                    f'{prob:.1%}',
                    ha='center', va='center',
                    color='white', fontweight='bold'
                )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Add overall title
        symbol = result_context.symbol
        timeframe = result_context.timeframe
        plt.suptitle(f'Retracement Analysis for {symbol} {timeframe}', fontsize=16, y=0.98)
        
        # Save figure
        prob_file = self.output_dir / f"{key}_retracement_probabilities_{timestamp}.png"
        plt.savefig(prob_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create figure for time-to-target
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        
        # Plot for pivot highs (retracement down)
        if 'is_pivot_high' in df.columns and df['is_pivot_high'].sum() > 0:
            ax = axes[0]
            
            # Calculate average bars to target for each level
            avg_bars = []
            level_labels = []
            
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                bars_col = f'bars_to_{level_str}_from_high'
                
                # Skip if the column doesn't exist
                if bars_col not in df.columns:
                    continue
                
                # Only consider rows where the target was reached
                valid_bars = df[~df[bars_col].isna()][bars_col]
                
                if len(valid_bars) > 0:
                    avg_bars.append(valid_bars.mean())
                    level_labels.append(f'{level*100:.1f}%')
                else:
                    avg_bars.append(0)
                    level_labels.append(f'{level*100:.1f}%')
            
            # Create bar chart
            bars = ax.bar(level_labels, avg_bars, color='red', alpha=0.7)
            
            ax.set_title('Average Bars to Reach Retracement Level from Pivot Highs')
            ax.set_xlabel('Retracement Level')
            ax.set_ylabel('Average Bars')
            ax.grid(alpha=0.3)
            
            # Add value labels on bars
            for i, val in enumerate(avg_bars):
                if val > 0:
                    ax.text(
                        i,
                        val / 2,
                        f'{val:.1f}',
                        ha='center', va='center',
                        color='white', fontweight='bold'
                    )
        
        # Plot for pivot lows (retracement up)
        if 'is_pivot_low' in df.columns and df['is_pivot_low'].sum() > 0:
            ax = axes[1]
            
            # Calculate average bars to target for each level
            avg_bars = []
            level_labels = []
            
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                bars_col = f'bars_to_{level_str}_from_low'
                
                # Skip if the column doesn't exist
                if bars_col not in df.columns:
                    continue
                
                # Only consider rows where the target was reached
                valid_bars = df[~df[bars_col].isna()][bars_col]
                
                if len(valid_bars) > 0:
                    avg_bars.append(valid_bars.mean())
                    level_labels.append(f'{level*100:.1f}%')
                else:
                    avg_bars.append(0)
                    level_labels.append(f'{level*100:.1f}%')
            
            # Create bar chart
            bars = ax.bar(level_labels, avg_bars, color='green', alpha=0.7)
            
            ax.set_title('Average Bars to Reach Retracement Level from Pivot Lows')
            ax.set_xlabel('Retracement Level')
            ax.set_ylabel('Average Bars')
            ax.grid(alpha=0.3)
            
            # Add value labels on bars
            for i, val in enumerate(avg_bars):
                if val > 0:
                    ax.text(
                        i,
                        val / 2,
                        f'{val:.1f}',
                        ha='center', va='center',
                        color='white', fontweight='bold'
                    )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Add overall title
        plt.suptitle(f'Time to Target Analysis for {symbol} {timeframe}', fontsize=16, y=0.98)
        
        # Save figure
        time_file = self.output_dir / f"{key}_time_to_target_{timestamp}.png"
        plt.savefig(time_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Also create a visualization of example pivots with retracement levels
        self._visualize_example_pivots(result_context, key, timestamp)
        
        # Add the new market structure visualization
        look_forward = self.params.get('analysis', 'retracement', 'look_forward', default=48)
        self._visualize_pivot_context(result_context, key, timestamp, look_forward)
    
        # Add the new fractal strength probability visualization
        self._visualize_strength_probabilities(result_context, key, timestamp)
    
        # Add the new time decay probability visualization
        self._visualize_time_decay(result_context, key, timestamp)
        
    def _visualize_example_pivots(self, result_context, key, timestamp, num_examples=3):
        """
        Create visualizations of example pivots with retracement levels
        
        Args:
            result_context: DataContext with analysis results
            key: Key for the symbol and timeframe
            timestamp: Timestamp string
            num_examples: Number of example pivots to visualize
        """
        df = result_context.df
        
        # Create separate figures for pivot highs and lows
        if 'is_pivot_high' in df.columns and df['is_pivot_high'].sum() > 0:
            # Find pivots with calculated retracement levels
            valid_pivot_indices = []
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                retr_col = f'retracement_{level_str}_from_high'
                # Find pivots that have retracement levels calculated
                if retr_col in df.columns:
                    valid_indices = df[df['is_pivot_high'] & ~df[retr_col].isna()].index
                    valid_pivot_indices.extend(valid_indices.tolist())
            
            # Get unique indices
            valid_pivot_indices = list(set(valid_pivot_indices))
            
            if not valid_pivot_indices:
                self.logger.warning("No valid pivot highs with retracement levels found")
                return
                
            # Select random examples (if we have enough)
            if len(valid_pivot_indices) <= num_examples:
                example_indices = valid_pivot_indices
            else:
                example_indices = np.random.choice(valid_pivot_indices, size=num_examples, replace=False)
            
            for i, idx in enumerate(example_indices):
                loc = df.index.get_loc(idx)
                
                # Get a window around the pivot
                start_loc = max(0, loc - 20)
                end_loc = min(len(df), loc + 50)
                window_df = df.iloc[start_loc:end_loc]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Plot high-low bars (simpler alternative to candlesticks)
                for _, row in window_df.iterrows():
                    # Plot high-low line
                    ax.plot([row.name, row.name], [row['low'], row['high']], 
                        color='black', linewidth=1)
                    
                    # Add markers for open and close if desired
                    # Color the body based on if close > open
                    color = 'green' if row['close'] >= row['open'] else 'red'
                    # Plot open-close body
                    ax.plot([row.name, row.name], [row['open'], row['close']], 
                        color=color, linewidth=3, alpha=0.7)
                
                # Mark the pivot high
                pivot_high = df.loc[idx, 'high']
                ax.scatter([idx], [pivot_high], color='red', s=100, marker='^', label='Pivot High')
                
                # Find the last pivot low before this pivot high
                prev_pivot_lows = df[(df.index < idx) & (df['is_pivot_low'])]
                
                if not prev_pivot_lows.empty:
                    last_pivot_low_idx = prev_pivot_lows.index[-1]
                    last_pivot_low = prev_pivot_lows.iloc[-1]['low']
                    
                    # Mark the last pivot low
                    ax.scatter([last_pivot_low_idx], [last_pivot_low], color='green', s=100, marker='v', label='Previous Pivot Low')
                    
                    # Draw line between pivot points
                    ax.plot([last_pivot_low_idx, idx], [last_pivot_low, pivot_high], 'k--', alpha=0.5)
                    
                    # Plot retracement levels
                    for level in self.levels:
                        level_str = f'{int(level*1000)}'
                        retracement_col = f'retracement_{level_str}_from_high'
                        reached_col = f'reached_{level_str}_from_high'
                        bars_col = f'bars_to_{level_str}_from_high'
                        
                        # Skip if columns don't exist
                        if retracement_col not in df.columns or reached_col not in df.columns:
                            continue
                        
                        if not pd.isna(df.loc[idx, retracement_col]):
                            retracement_price = df.loc[idx, retracement_col]
                            
                            # Draw horizontal line for retracement level
                            ax.axhline(y=retracement_price, color='red', linestyle='--', alpha=0.5)
                            
                            # Add label with retracement percentage
                            ax.text(
                                window_df.index[0], 
                                retracement_price,
                                f"{level*100:.1f}% ({retracement_price:.2f})",
                                va='center'
                            )
                            
                            # If the level was reached, mark it
                            # Convert to boolean for consistent handling
                            reached_value = df.loc[idx, reached_col]
                            if isinstance(reached_value, (bool, np.bool_)) and reached_value:
                                if bars_col in df.columns and not pd.isna(df.loc[idx, bars_col]):
                                    bars_to_reach = df.loc[idx, bars_col]
                                    reached_idx = df.index[loc + int(bars_to_reach)]
                                    
                                    if reached_idx in window_df.index:
                                        # Use the exact retracement level price instead of the candle low
                                        # Mark the point where level was reached with a horizontal marker
                                        ax.scatter(
                                            [reached_idx], 
                                            [retracement_price],  # Using retracement price instead of candle low
                                            color='red', 
                                            s=80, 
                                            marker='o', 
                                            label=f'Reached {level*100:.1f}% in {int(bars_to_reach)} bars' if level == self.levels[0] else ""
                                        )
                                        
                                        # Draw a vertical line to the exact retracement level
                                        ax.plot([reached_idx, reached_idx], 
                                            [window_df.loc[reached_idx, 'low'], retracement_price],
                                            'r:', alpha=0.7, linewidth=1.5)
                
                # Set title and labels
                ax.set_title(f'Pivot High at {idx} with Retracement Levels')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.grid(alpha=0.3)
                ax.legend()
                
                # Format date axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                
                # Save figure
                example_file = self.output_dir / f"{key}_pivot_high_example_{i+1}_{timestamp}.png"
                plt.savefig(example_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # Repeat for pivot lows
        if 'is_pivot_low' in df.columns and df['is_pivot_low'].sum() > 0:
            # Find pivots with calculated retracement levels
            valid_pivot_indices = []
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                retr_col = f'retracement_{level_str}_from_low'
                # Find pivots that have retracement levels calculated
                if retr_col in df.columns:
                    valid_indices = df[df['is_pivot_low'] & ~df[retr_col].isna()].index
                    valid_pivot_indices.extend(valid_indices.tolist())
            
            # Get unique indices
            valid_pivot_indices = list(set(valid_pivot_indices))
            
            if not valid_pivot_indices:
                self.logger.warning("No valid pivot lows with retracement levels found")
                return
                
            # Select random examples (if we have enough)
            if len(valid_pivot_indices) <= num_examples:
                example_indices = valid_pivot_indices
            else:
                example_indices = np.random.choice(valid_pivot_indices, size=num_examples, replace=False)
            
            for i, idx in enumerate(example_indices):
                loc = df.index.get_loc(idx)
                
                # Get a window around the pivot
                start_loc = max(0, loc - 20)
                end_loc = min(len(df), loc + 50)
                window_df = df.iloc[start_loc:end_loc]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Plot high-low bars (simpler alternative to candlesticks)
                for _, row in window_df.iterrows():
                    # Plot high-low line
                    ax.plot([row.name, row.name], [row['low'], row['high']], 
                        color='black', linewidth=1)
                    
                    # Add markers for open and close if desired
                    # Color the body based on if close > open
                    color = 'green' if row['close'] >= row['open'] else 'red'
                    # Plot open-close body
                    ax.plot([row.name, row.name], [row['open'], row['close']], 
                        color=color, linewidth=3, alpha=0.7)
                
                # Mark the pivot low
                pivot_low = df.loc[idx, 'low']
                ax.scatter([idx], [pivot_low], color='green', s=100, marker='v', label='Pivot Low')
                
                # Find the last pivot high before this pivot low
                prev_pivot_highs = df[(df.index < idx) & (df['is_pivot_high'])]
                
                if not prev_pivot_highs.empty:
                    last_pivot_high_idx = prev_pivot_highs.index[-1]
                    last_pivot_high = prev_pivot_highs.iloc[-1]['high']
                    
                    # Mark the last pivot high
                    ax.scatter([last_pivot_high_idx], [last_pivot_high], color='red', s=100, marker='^', label='Previous Pivot High')
                    
                    # Draw line between pivot points
                    ax.plot([last_pivot_high_idx, idx], [last_pivot_high, pivot_low], 'k--', alpha=0.5)
                    
                    # Plot retracement levels
                    for level in self.levels:
                        level_str = f'{int(level*1000)}'
                        retracement_col = f'retracement_{level_str}_from_low'
                        reached_col = f'reached_{level_str}_from_low'
                        bars_col = f'bars_to_{level_str}_from_low'
                        
                        # Skip if columns don't exist
                        if retracement_col not in df.columns or reached_col not in df.columns:
                            continue
                        
                        if not pd.isna(df.loc[idx, retracement_col]):
                            retracement_price = df.loc[idx, retracement_col]
                            
                            # Draw horizontal line for retracement level
                            ax.axhline(y=retracement_price, color='green', linestyle='--', alpha=0.5)
                            
                            # Add label with retracement percentage
                            ax.text(
                                window_df.index[0], 
                                retracement_price,
                                f"{level*100:.1f}% ({retracement_price:.2f})",
                                va='center'
                            )
                            
                            # If the level was reached, mark it
                            # Convert to boolean for consistent handling
                            reached_value = df.loc[idx, reached_col]
                            if isinstance(reached_value, (bool, np.bool_)) and reached_value:
                                if bars_col in df.columns and not pd.isna(df.loc[idx, bars_col]):
                                    bars_to_reach = df.loc[idx, bars_col]
                                    reached_idx = df.index[loc + int(bars_to_reach)]
                                    
                                    if reached_idx in window_df.index:
                                        # Use the exact retracement level price instead of the candle high
                                        # Mark the point where level was reached with a horizontal marker
                                        ax.scatter(
                                            [reached_idx], 
                                            [retracement_price],  # Using retracement price instead of candle high
                                            color='green', 
                                            s=80, 
                                            marker='o', 
                                            label=f'Reached {level*100:.1f}% in {int(bars_to_reach)} bars' if level == self.levels[0] else ""
                                        )
                                        
                                        # Draw a vertical line to the exact retracement level
                                        ax.plot([reached_idx, reached_idx], 
                                            [window_df.loc[reached_idx, 'high'], retracement_price],
                                            'g:', alpha=0.7, linewidth=1.5)
                
                # Set title and labels
                ax.set_title(f'Pivot Low at {idx} with Retracement Levels')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.grid(alpha=0.3)
                ax.legend()
                
                # Format date axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                
                # Save figure
                example_file = self.output_dir / f"{key}_pivot_low_example_{i+1}_{timestamp}.png"
                plt.savefig(example_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    def _visualize_pivot_context(self, result_context, key, timestamp, look_forward, num_examples=3):
        """
        Create visualizations of market structure showing all pivots within a look-forward window
        
        Args:
            result_context: DataContext with analysis results
            key: Key for the symbol and timeframe
            timestamp: Timestamp string
            look_forward: Number of bars in forward window
            num_examples: Number of pivot examples to visualize
        """
        df = result_context.df
        
        # Select random starting points for visualization 
        # Make sure we have enough data after the starting point
        last_valid_idx = max(0, len(df) - look_forward - 1)
        valid_indices_range = range(last_valid_idx)
        
        if len(valid_indices_range) <= num_examples:
            start_locs = valid_indices_range
        else:
            start_locs = np.random.choice(valid_indices_range, size=num_examples, replace=False)
        
        # Convert integer positions to actual index values
        starting_indices = [df.index[loc] for loc in start_locs]    
        
        for i, start_idx in enumerate(starting_indices):
            start_loc = df.index.get_loc(start_idx)
            
            # Get a window from this starting point
            window_df = df.iloc[start_loc:start_loc + look_forward + 20]  # Add some extra bars for context
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 9))
            
            # Plot candlesticks
            for _, row in window_df.iterrows():
                # Plot high-low line
                ax.plot([row.name, row.name], [row['low'], row['high']], 
                        color='black', linewidth=1)
                
                # Color the body based on if close > open
                color = 'green' if row['close'] >= row['open'] else 'red'
                # Plot open-close body
                ax.plot([row.name, row.name], [row['open'], row['close']], 
                    color=color, linewidth=3, alpha=0.7)
            
            # Mark all pivot highs in the window
            if 'is_pivot_high' in window_df.columns:
                pivot_high_indices = window_df[window_df['is_pivot_high']].index
                pivot_high_prices = window_df.loc[pivot_high_indices, 'high']
                
                if not pivot_high_indices.empty:
                    ax.scatter(pivot_high_indices, pivot_high_prices, 
                            color='red', s=80, marker='^', label='Pivot Highs')
                    
                    # Add fractal strength if available
                    if 'fractal_strength_high' in window_df.columns:
                        for idx in pivot_high_indices:
                            strength = window_df.loc[idx, 'fractal_strength_high']
                            ax.text(idx, window_df.loc[idx, 'high'] * 1.001, 
                                f'S{strength}', ha='center', va='bottom',
                                fontsize=8, color='red')
            
            # Mark all pivot lows in the window
            if 'is_pivot_low' in window_df.columns:
                pivot_low_indices = window_df[window_df['is_pivot_low']].index
                pivot_low_prices = window_df.loc[pivot_low_indices, 'low']
                
                if not pivot_low_indices.empty:
                    ax.scatter(pivot_low_indices, pivot_low_prices, 
                            color='green', s=80, marker='v', label='Pivot Lows')
                    
                    # Add fractal strength if available
                    if 'fractal_strength_low' in window_df.columns:
                        for idx in pivot_low_indices:
                            strength = window_df.loc[idx, 'fractal_strength_low']
                            ax.text(idx, window_df.loc[idx, 'low'] * 0.999, 
                                f'S{strength}', ha='center', va='top',
                                fontsize=8, color='green')
            
            # Draw look-forward window
            look_forward_end = window_df.index[min(look_forward, len(window_df)-1)]
            ax.axvspan(start_idx, look_forward_end, alpha=0.1, color='blue')
            ax.axvline(x=start_idx, color='blue', linestyle='--', alpha=0.7, 
                    label=f'Look-forward window ({look_forward} bars)')
            
            # Set title and labels
            symbol = result_context.symbol
            timeframe = result_context.timeframe
            ax.set_title(f'Market Structure with Pivot Points - {symbol} {timeframe}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(alpha=0.3)
            ax.legend()
            
            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            fig.autofmt_xdate()
            
            # Save figure
            structure_file = self.output_dir / f"{key}_market_structure_{i+1}_{timestamp}.png"
            plt.savefig(structure_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Saved market structure visualization to {structure_file}")
    
    def _visualize_strength_probabilities(self, result_context, key, timestamp):
        """
        Create visualizations showing retracement probabilities by fractal strength
        
        Args:
            result_context: DataContext with analysis results
            key: Key for the symbol and timeframe
            timestamp: Timestamp string
        """
        df = result_context.df
        stats = self._calculate_stats(df)
        
        # Only proceed if we have fractal strength data
        if 'fractal_strength_high' not in df.columns and 'fractal_strength_low' not in df.columns:
            self.logger.warning("No fractal strength data available for visualization")
            return
        
        # Process high pivots
        if 'fractal_strength_high' in df.columns:
            # Get unique strength values (excluding 0)
            strength_values = sorted([s for s in df['fractal_strength_high'].unique() if s > 0])
            
            if strength_values:
                # Create figure - separate subplot for each retracement level
                fig, axes = plt.subplots(len(self.levels), 1, figsize=(12, 4*len(self.levels)), squeeze=False)
                
                # For each retracement level
                for i, level in enumerate(self.levels):
                    ax = axes[i, 0]
                    level_str = f'{int(level*1000)}'
                    
                    # Data for individual strengths
                    individual_probs = []
                    individual_counts = []
                    
                    for strength in strength_values:
                        prob_key = f'high_{level_str}_strength_{strength}_prob'
                        count_key = f'high_{level_str}_strength_{strength}_count'
                        
                        if prob_key in stats:
                            individual_probs.append(stats[prob_key])
                            individual_counts.append(stats[count_key])
                        else:
                            individual_probs.append(0)
                            individual_counts.append(0)
                    
                    # Data for cumulative strengths (>= X)
                    cumulative_probs = []
                    cumulative_counts = []
                    
                    for min_strength in range(1, max(strength_values) + 1):
                        prob_key = f'high_{level_str}_strength_ge_{min_strength}_prob'
                        count_key = f'high_{level_str}_strength_ge_{min_strength}_count'
                        
                        if prob_key in stats:
                            cumulative_probs.append(stats[prob_key])
                            cumulative_counts.append(stats[count_key])
                        else:
                            cumulative_probs.append(0)
                            cumulative_counts.append(0)
                    
                    # Create bar chart for individual strengths
                    x = np.arange(len(strength_values))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, individual_probs, width, 
                            label='Individual Strength', color='lightcoral')
                    
                    # Create bar chart for cumulative strengths
                    bars2 = ax.bar(x + width/2, cumulative_probs, width,
                            label='Strength >= X', color='darkred', alpha=0.7)
                    
                    # Add count labels above bars
                    for bars, counts in [(bars1, individual_counts), (bars2, cumulative_counts)]:
                        for j, (bar, count) in enumerate(zip(bars, counts)):
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height + 0.02,
                                f'n={count}',
                                ha='center', va='bottom',
                                fontsize=8
                            )
                    
                    # Add percentage labels on bars
                    for bars, probs in [(bars1, individual_probs), (bars2, cumulative_probs)]:
                        for j, (bar, prob) in enumerate(zip(bars, probs)):
                            height = bar.get_height()
                            if height > 0.05:  # Only add text if bar is tall enough
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2,
                                    height / 2,
                                    f'{prob:.1%}',
                                    ha='center', va='center',
                                    color='white', fontweight='bold',
                                    fontsize=8
                                )
                    
                    # Configure plot
                    ax.set_title(f'{level*100:.1f}% Retracement Probability by Fractal Strength (Pivot Highs)')
                    ax.set_xlabel('Fractal Strength')
                    ax.set_ylabel('Probability')
                    ax.set_ylim(0, 1.1)
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(s) for s in strength_values])
                    ax.grid(alpha=0.3)
                    ax.legend()
                    
                    # Add reference line for overall probability
                    overall_prob_key = f'high_{level_str}_prob'
                    if overall_prob_key in stats:
                        overall_prob = stats[overall_prob_key]
                        ax.axhline(y=overall_prob, color='red', linestyle='--', 
                                label=f'Overall: {overall_prob:.1%}')
                        ax.text(
                            max(x) + 0.5,
                            overall_prob,
                            f'Overall: {overall_prob:.1%}',
                            va='center',
                            fontsize=8,
                            color='red'
                        )
                
                # Adjust layout and save
                plt.tight_layout()
                symbol = result_context.symbol
                timeframe = result_context.timeframe
                
                # Increase top margin to prevent title overlap
                plt.subplots_adjust(top=0.92)  # Reduce from 0.99 to 0.92='o', color=color, alpha=alpha, label=f'{level*100:.1f}%')
                
                plt.suptitle(f'Retracement Analysis by Fractal Strength - {symbol} {timeframe} (Pivot Highs)', 
                        fontsize=16, y=0.98)
                
                strength_file = self.output_dir / f"{key}_high_strength_probabilities_{timestamp}.png"
                plt.savefig(strength_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                self.logger.info(f"Saved strength probability visualization to {strength_file}")
        
        # Process low pivots (very similar code)
        if 'fractal_strength_low' in df.columns:
            # Get unique strength values (excluding 0)
            strength_values = sorted([s for s in df['fractal_strength_low'].unique() if s > 0])
            
            if strength_values:
                # Create figure - separate subplot for each retracement level
                fig, axes = plt.subplots(len(self.levels), 1, figsize=(12, 4*len(self.levels)), squeeze=False)
                
                # For each retracement level
                for i, level in enumerate(self.levels):
                    ax = axes[i, 0]
                    level_str = f'{int(level*1000)}'
                    
                    # Data for individual strengths
                    individual_probs = []
                    individual_counts = []
                    
                    for strength in strength_values:
                        prob_key = f'low_{level_str}_strength_{strength}_prob'
                        count_key = f'low_{level_str}_strength_{strength}_count'
                        
                        if prob_key in stats:
                            individual_probs.append(stats[prob_key])
                            individual_counts.append(stats[count_key])
                        else:
                            individual_probs.append(0)
                            individual_counts.append(0)
                    
                    # Data for cumulative strengths (>= X)
                    cumulative_probs = []
                    cumulative_counts = []
                    
                    for min_strength in range(1, max(strength_values) + 1):
                        prob_key = f'low_{level_str}_strength_ge_{min_strength}_prob'
                        count_key = f'low_{level_str}_strength_ge_{min_strength}_count'
                        
                        if prob_key in stats:
                            cumulative_probs.append(stats[prob_key])
                            cumulative_counts.append(stats[count_key])
                        else:
                            cumulative_probs.append(0)
                            cumulative_counts.append(0)
                    
                    # Create bar chart for individual strengths
                    x = np.arange(len(strength_values))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, individual_probs, width, 
                            label='Individual Strength', color='lightgreen')
                    
                    # Create bar chart for cumulative strengths
                    bars2 = ax.bar(x + width/2, cumulative_probs, width,
                            label='Strength >= X', color='darkgreen', alpha=0.7)
                    
                    # Add count labels above bars
                    for bars, counts in [(bars1, individual_counts), (bars2, cumulative_counts)]:
                        for j, (bar, count) in enumerate(zip(bars, counts)):
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height + 0.02,
                                f'n={count}',
                                ha='center', va='bottom',
                                fontsize=8
                            )
                    
                    # Add percentage labels on bars
                    for bars, probs in [(bars1, individual_probs), (bars2, cumulative_probs)]:
                        for j, (bar, prob) in enumerate(zip(bars, probs)):
                            height = bar.get_height()
                            if height > 0.05:  # Only add text if bar is tall enough
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2,
                                    height / 2,
                                    f'{prob:.1%}',
                                    ha='center', va='center',
                                    color='white', fontweight='bold',
                                    fontsize=8
                                )
                    
                    # Configure plot
                    ax.set_title(f'{level*100:.1f}% Retracement Probability by Fractal Strength (Pivot Lows)')
                    ax.set_xlabel('Fractal Strength')
                    ax.set_ylabel('Probability')
                    ax.set_ylim(0, 1.1)
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(s) for s in strength_values])
                    ax.grid(alpha=0.3)
                    ax.legend()
                    
                    # Add reference line for overall probability
                    overall_prob_key = f'low_{level_str}_prob'
                    if overall_prob_key in stats:
                        overall_prob = stats[overall_prob_key]
                        ax.axhline(y=overall_prob, color='green', linestyle='--', 
                                label=f'Overall: {overall_prob:.1%}')
                        ax.text(
                            max(x) + 0.5,
                            overall_prob,
                            f'Overall: {overall_prob:.1%}',
                            va='center',
                            fontsize=8,
                            color='green'
                        )
                
                # Adjust layout and save
                plt.tight_layout()
                symbol = result_context.symbol
                timeframe = result_context.timeframe
    
                # Increase top margin to prevent title overlap
                plt.subplots_adjust(top=0.92)  # Reduce from 0.99 to 0.92'
                
                plt.suptitle(f'Retracement Analysis by Fractal Strength - {symbol} {timeframe} (Pivot Lows)', 
                        fontsize=16, y=0.98)
                
                strength_file = self.output_dir / f"{key}_low_strength_probabilities_{timestamp}.png"
                plt.savefig(strength_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                self.logger.info(f"Saved strength probability visualization to {strength_file}")
                
    def _visualize_time_decay(self, result_context, key, timestamp):
        """
        Create visualizations showing how retracement probabilities change with time
        
        Args:
            result_context: DataContext with analysis results
            key: Key for the symbol and timeframe
            timestamp: Timestamp string
        """
        df = result_context.df
        time_decay_stats = self._calculate_time_decay_stats(df)
        
        if not time_decay_stats:
            self.logger.warning("No time decay statistics available for visualization")
            return
        
        # Define Fibonacci sequence of bar counts
        fibonacci_bars = [0, 1, 2, 3, 5, 8, 13, 21, 34]
        
        # Create separate figures for high and low pivots
        for pivot_type in ['high', 'low']:
            # Check if we have data for this pivot type
            has_data = False
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                if f'{pivot_type}_{level_str}_prob_after_1_bars' in time_decay_stats:
                    has_data = True
                    break
                    
            if not has_data:
                continue
                
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Track original probability for each level
            original_probs = {}
        
            # Store all decay probabilities for each level
            all_decay_probs = {}
            
            # Plot line for each retracement level
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                
                # Get base probability for this level
                base_prob_key = f'{pivot_type}_{level_str}_prob'
                base_prob = 0  # Default if not found
                
                # Use original calculation stats
                original_stats = self._calculate_stats(df)
                if base_prob_key in original_stats:
                    base_prob = original_stats[base_prob_key]
                    original_probs[level] = base_prob
                
                # Data for time decay
                decay_probs = []
                
                # First point is the original probability (at bar 0)
                decay_probs.append(base_prob)
                
                # Get the probabilities for each time threshold
                for bar_count in fibonacci_bars[1:]:  # Skip 0
                    prob_key = f'{pivot_type}_{level_str}_prob_after_{bar_count}_bars'
                    if prob_key in time_decay_stats:
                        decay_probs.append(time_decay_stats[prob_key])
                    else:
                        # If missing, use previous value or 0
                        decay_probs.append(decay_probs[-1] if decay_probs else 0)
            
                # Store all probabilities for this level
                all_decay_probs[level] = decay_probs
                
                # Plot the time decay line
                color = 'red' if pivot_type == 'high' else 'green'
                alpha = 0.5 + 0.5 * (float(self.levels.index(level)) / len(self.levels))
                ax.plot(fibonacci_bars, decay_probs, 
                        marker='o', linewidth=2, markersize=8,
                        color=color, alpha=alpha, 
                        label=f'{level*100:.1f}% Level')
            
            # For each time threshold (except bar 0)
            for idx, x in enumerate(fibonacci_bars[1:], 1):
                # Skip first point (bars=0) as it's the base probability
                # Create a combined label with all three levels
                label_parts = []
                
                for level in self.levels:
                    prob = all_decay_probs[level][idx]
                    label_parts.append(f"{level*100:.1f}%: {prob:.1%}")
                
                # Join all parts with commas
                combined_label = ", ".join(label_parts)
                
                # Find average y position for label placement (aim for middle level)
                mid_level = self.levels[len(self.levels) // 2]
                y_pos = all_decay_probs[mid_level][idx]
                
                ax.annotate(
                    combined_label,
                    xy=(x, y_pos),  # Use y_pos for the y-coordinate
                    xytext=(x + 0.5, y_pos),  # Slightly to the right
                    fontsize=8,
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.2)
                )
            
            # Configure plot
            pivot_name = "Highs" if pivot_type == "high" else "Lows"
            symbol = result_context.symbol
            timeframe = result_context.timeframe
            
            ax.set_title(f'Retracement Probability Decay - {symbol} {timeframe} (Pivot {pivot_name})')
            ax.set_xlabel('Bars Since Pivot (No Retracement Yet)')
            ax.set_ylabel('Probability of Still Reaching Target')
            
            # Set x-axis to show Fibonacci bar counts plus extra space for annotations
            ax.set_xlim(0,  fibonacci_bars[-1] + 4)  # Add space on right for annotations
            ax.set_xticks(fibonacci_bars)
            ax.set_xticklabels([str(x) for x in fibonacci_bars])
            
            # Set y-axis range
            ax.set_ylim(0, 1.0)
            
            # Add grid and legend
            ax.grid(alpha=0.3)
            ax.legend(title="Retracement Levels")
            
            # Add reference line for random chance (0.5)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.text(fibonacci_bars[-1] - 2, 0.5, 'Random (50%)', va='center', fontsize=8, color='gray')
            
            # Add annotations explaining the chart
            textstr = '\n'.join([
                r'Interpretation:',
                r'- Shows probability of price reaching retracement target',
                r'  after not reaching it for X bars',
                r'- Initial point (0 bars): Overall probability',
                r'- Steeper drop = More urgent to act',
                r'- Flatter curve = More time to enter position',
                r'- The higher the probability, the more likely the target will be reached'
            ])
            
            # Place a text box in the top right
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
            
            # Save figure
            decay_file = self.output_dir / f"{key}_{pivot_type}_probability_decay_{timestamp}.png"
            plt.savefig(decay_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Saved time decay visualization to {decay_file}")
            
    def _create_summary(self, all_results, timestamp):
        """
        Create comprehensive summary report for all analyzed symbols and timeframes
        
        Args:
            all_results: Dictionary of analysis results
            timestamp: Timestamp string
        """
        # Create multiple summary DataFrames for different aspects
        overall_rows = []        # Basic retracement stats
        strength_rows = []       # Fractal strength distribution
        strength_prob_rows = []  # Retracement probabilities by strength
        time_decay_rows = []     # Time-based probability decay
        
        for key, result in all_results.items():
            exchange = result['exchange']
            symbol = result['symbol']
            timeframe = result['timeframe']
            stats = result['stats']
            
            # 1. OVERALL RETRACEMENT STATISTICS
            # Create rows for pivot highs
            if 'pivot_high_count' in stats:
                for level in self.levels:
                    level_str = f'{int(level*1000)}'
                    prob_key = f'high_{level_str}_prob'
                    count_key = f'high_{level_str}_count'
                    bars_key = f'high_{level_str}_avg_bars'
                    
                    if prob_key in stats:
                        row = {
                            'exchange': exchange,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'pivot_type': 'high',
                            'retracement_level': level,
                            'pivot_count': stats.get('pivot_high_count', 0),
                            'reached_count': stats.get(count_key, 0),
                            'probability': stats.get(prob_key, 0),
                            'avg_bars_to_reach': stats.get(bars_key, None)
                        }
                        overall_rows.append(row)
            
            # Create rows for pivot lows
            if 'pivot_low_count' in stats:
                for level in self.levels:
                    level_str = f'{int(level*1000)}'
                    prob_key = f'low_{level_str}_prob'
                    count_key = f'low_{level_str}_count'
                    bars_key = f'low_{level_str}_avg_bars'
                    
                    if prob_key in stats:
                        row = {
                            'exchange': exchange,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'pivot_type': 'low',
                            'retracement_level': level,
                            'pivot_count': stats.get('pivot_low_count', 0),
                            'reached_count': stats.get(count_key, 0),
                            'probability': stats.get(prob_key, 0),
                            'avg_bars_to_reach': stats.get(bars_key, None)
                        }
                        overall_rows.append(row)
            
            # 2. FRACTAL STRENGTH DISTRIBUTION
            # Get strength distribution for high pivots
            for strength_key in [k for k in stats.keys() if k.startswith('pivot_high_strength_') and not k.startswith('pivot_high_strength_ge_')]:
                strength = int(strength_key.split('_')[-2])
                row = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pivot_type': 'high',
                    'strength': strength,
                    'count': stats[strength_key]
                }
                strength_rows.append(row)

            # Get strength distribution for low pivots
            for strength_key in [k for k in stats.keys() if k.startswith('pivot_low_strength_') and not k.startswith('pivot_low_strength_ge_')]:
                strength = int(strength_key.split('_')[-2])
                row = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pivot_type': 'low',
                    'strength': strength,
                    'count': stats[strength_key]
                }
                strength_rows.append(row)
            
            # 3. RETRACEMENT PROBABILITIES BY STRENGTH
            # For high pivots
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                # Individual strength probabilities
                for strength_key in [k for k in stats.keys() if k.startswith(f'high_{level_str}_strength_') and not k.startswith(f'high_{level_str}_strength_ge_') and k.endswith('_prob')]:
                    strength = int(strength_key.split('_')[-2])
                    count_key = strength_key.replace('_prob', '_count')
                    bars_key = strength_key.replace('_prob', '_avg_bars')
                    
                    row = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pivot_type': 'high',
                        'retracement_level': level,
                        'strength_type': 'exact',
                        'strength': strength,
                        'count': stats.get(count_key, 0),
                        'probability': stats.get(strength_key, 0),
                        'avg_bars': stats.get(bars_key, None)
                    }
                    strength_prob_rows.append(row)
                
                # Cumulative strength probabilities
                for strength_key in [k for k in stats.keys() if k.startswith(f'high_{level_str}_strength_ge_') and k.endswith('_prob')]:
                    strength = int(strength_key.split('_')[-2])
                    count_key = strength_key.replace('_prob', '_count')
                    bars_key = strength_key.replace('_prob', '_avg_bars')
                    
                    row = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pivot_type': 'high',
                        'retracement_level': level,
                        'strength_type': 'cumulative',
                        'strength': strength,
                        'count': stats.get(count_key, 0),
                        'probability': stats.get(strength_key, 0),
                        'avg_bars': stats.get(bars_key, None)
                    }
                    strength_prob_rows.append(row)
            
            # For low pivots (same pattern)
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                # Individual strength probabilities
                for strength_key in [k for k in stats.keys() if k.startswith(f'low_{level_str}_strength_') and not k.startswith(f'low_{level_str}_strength_ge_') and k.endswith('_prob')]:
                    strength = int(strength_key.split('_')[-2])
                    count_key = strength_key.replace('_prob', '_count')
                    bars_key = strength_key.replace('_prob', '_avg_bars')
                    
                    row = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pivot_type': 'low',
                        'retracement_level': level,
                        'strength_type': 'exact',
                        'strength': strength,
                        'count': stats.get(count_key, 0),
                        'probability': stats.get(strength_key, 0),
                        'avg_bars': stats.get(bars_key, None)
                    }
                    strength_prob_rows.append(row)
                
                # Cumulative strength probabilities
                for strength_key in [k for k in stats.keys() if k.startswith(f'low_{level_str}_strength_ge_') and k.endswith('_prob')]:
                    strength = int(strength_key.split('_')[-2])
                    count_key = strength_key.replace('_prob', '_count')
                    bars_key = strength_key.replace('_prob', '_avg_bars')
                    
                    row = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pivot_type': 'low',
                        'retracement_level': level,
                        'strength_type': 'cumulative',
                        'strength': strength,
                        'count': stats.get(count_key, 0),
                        'probability': stats.get(strength_key, 0),
                        'avg_bars': stats.get(bars_key, None)
                    }
                    strength_prob_rows.append(row)
            
            # 4. TIME-BASED PROBABILITY DECAY
            # Calculate time decay statistics
            df = result['data']
            time_decay_stats = self._calculate_time_decay_stats(df)
            
            # Fibonacci bar counts used in time decay analysis
            fibonacci_bars = [1, 2, 3, 5, 8, 13, 21, 34]
            
            # For high pivots
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                
                for bar_count in fibonacci_bars:
                    prob_key = f'high_{level_str}_prob_after_{bar_count}_bars'
                    count_key = f'high_{level_str}_count_after_{bar_count}_bars'
                    remaining_key = f'high_{level_str}_remaining_after_{bar_count}_bars'
                    
                    if prob_key in time_decay_stats:
                        row = {
                            'exchange': exchange,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'pivot_type': 'high',
                            'retracement_level': level,
                            'bars_elapsed': bar_count,
                            'probability': time_decay_stats.get(prob_key, 0),
                            'count': time_decay_stats.get(count_key, 0),
                            'remaining_pivots': time_decay_stats.get(remaining_key, 0)
                        }
                        time_decay_rows.append(row)
            
            # For low pivots
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                
                for bar_count in fibonacci_bars:
                    prob_key = f'low_{level_str}_prob_after_{bar_count}_bars'
                    count_key = f'low_{level_str}_count_after_{bar_count}_bars'
                    remaining_key = f'low_{level_str}_remaining_after_{bar_count}_bars'
                    
                    if prob_key in time_decay_stats:
                        row = {
                            'exchange': exchange,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'pivot_type': 'low',
                            'retracement_level': level,
                            'bars_elapsed': bar_count,
                            'probability': time_decay_stats.get(prob_key, 0),
                            'count': time_decay_stats.get(count_key, 0),
                            'remaining_pivots': time_decay_stats.get(remaining_key, 0)
                        }
                        time_decay_rows.append(row)
        
        # Create DataFrames and save to CSV if we have data
        if overall_rows:
            overall_df = pd.DataFrame(overall_rows)
            overall_file = self.output_dir / f"retracement_summary_overall_{timestamp}.csv"
            overall_df.to_csv(overall_file, index=False)
            self.logger.info(f"Saved overall summary to {overall_file}")
        
        if strength_rows:
            strength_df = pd.DataFrame(strength_rows)
            strength_file = self.output_dir / f"retracement_summary_strength_distribution_{timestamp}.csv"
            strength_df.to_csv(strength_file, index=False)
            self.logger.info(f"Saved strength distribution summary to {strength_file}")
        
        if strength_prob_rows:
            strength_prob_df = pd.DataFrame(strength_prob_rows)
            strength_prob_file = self.output_dir / f"retracement_summary_strength_probabilities_{timestamp}.csv"
            strength_prob_df.to_csv(strength_prob_file, index=False)
            self.logger.info(f"Saved strength probabilities summary to {strength_prob_file}")
        
        if time_decay_rows:
            time_decay_df = pd.DataFrame(time_decay_rows)
            time_decay_file = self.output_dir / f"retracement_summary_time_decay_{timestamp}.csv"
            time_decay_df.to_csv(time_decay_file, index=False)
            self.logger.info(f"Saved time decay summary to {time_decay_file}")
        
        # Create consolidated summary file with key statistics
        self._create_consolidated_summary(all_results, timestamp)
        
        # Create visualization of overall summary
        if overall_rows:
            self._visualize_summary(pd.DataFrame(overall_rows), timestamp)
            
    def _create_consolidated_summary(self, all_results, timestamp):
        """
        Create a consolidated summary with key statistics in a single CSV
        
        Args:
            all_results: Dictionary of analysis results
            timestamp: Timestamp string
        """
        consolidated_rows = []
        
        for key, result in all_results.items():
            exchange = result['exchange']
            symbol = result['symbol']
            timeframe = result['timeframe']
            stats = result['stats']
            df = result['data']
            
            # Get base statistics for high and low pivots
            high_pivot_count = stats.get('pivot_high_count', 0)
            low_pivot_count = stats.get('pivot_low_count', 0)
            
            # Get fractal strength distribution
            high_strength_dist = {}
            low_strength_dist = {}
            
            for strength_key in [k for k in stats.keys() if k.startswith('pivot_high_strength_') and not k.startswith('pivot_high_strength_ge_')]:
                strength = int(strength_key.split('_')[-2])
                high_strength_dist[strength] = stats[strength_key]
                
            for strength_key in [k for k in stats.keys() if k.startswith('pivot_low_strength_') and not k.startswith('pivot_low_strength_ge_')]:
                strength = int(strength_key.split('_')[-2])
                low_strength_dist[strength] = stats[strength_key]
            
            # For each retracement level
            for level in self.levels:
                level_str = f'{int(level*1000)}'
                
                # Basic retracement probabilities
                high_prob = stats.get(f'high_{level_str}_prob', None)
                high_count = stats.get(f'high_{level_str}_count', None)
                high_avg_bars = stats.get(f'high_{level_str}_avg_bars', None)
                
                low_prob = stats.get(f'low_{level_str}_prob', None)
                low_count = stats.get(f'low_{level_str}_count', None)
                low_avg_bars = stats.get(f'low_{level_str}_avg_bars', None)
                
                # Get best fractal strength (exact) for high pivots
                high_best_strength = None
                high_best_prob = 0
                
                for strength in range(1, 6):  # Assuming max strength is 5
                    prob_key = f'high_{level_str}_strength_{strength}_prob'
                    if prob_key in stats and stats[prob_key] > high_best_prob and stats.get(f'high_{level_str}_strength_{strength}_count', 0) >= 30:
                        high_best_prob = stats[prob_key]
                        high_best_strength = strength
                
                # Get best fractal strength (exact) for low pivots
                low_best_strength = None
                low_best_prob = 0
                
                for strength in range(1, 6):  # Assuming max strength is 5
                    prob_key = f'low_{level_str}_strength_{strength}_prob'
                    if prob_key in stats and stats[prob_key] > low_best_prob and stats.get(f'low_{level_str}_strength_{strength}_count', 0) >= 30:
                        low_best_prob = stats[prob_key]
                        low_best_strength = strength
                
                # Get time decay information for high and low pivots
                time_decay_stats = self._calculate_time_decay_stats(df)
                
                # Get probability after 8 bars (informative timepoint)
                high_prob_after_8 = time_decay_stats.get(f'high_{level_str}_prob_after_8_bars', None)
                low_prob_after_8 = time_decay_stats.get(f'low_{level_str}_prob_after_8_bars', None)
                
                # Create consolidated row
                row = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'retracement_level': level,
                    'high_pivot_count': high_pivot_count,
                    'low_pivot_count': low_pivot_count,
                    'high_overall_prob': high_prob,
                    'high_reached_count': high_count,
                    'high_avg_bars': high_avg_bars,
                    'high_best_strength': high_best_strength,
                    'high_best_strength_prob': high_best_prob if high_best_strength else None,
                    'high_prob_after_8_bars': high_prob_after_8,
                    'low_overall_prob': low_prob,
                    'low_reached_count': low_count,
                    'low_avg_bars': low_avg_bars,
                    'low_best_strength': low_best_strength,
                    'low_best_strength_prob': low_best_prob if low_best_strength else None,
                    'low_prob_after_8_bars': low_prob_after_8,
                    'high_strength_distribution': str(high_strength_dist),
                    'low_strength_distribution': str(low_strength_dist)
                }
                consolidated_rows.append(row)
        
        if consolidated_rows:
            consolidated_df = pd.DataFrame(consolidated_rows)
            consolidated_file = self.output_dir / f"retracement_summary_consolidated_{timestamp}.csv"
            consolidated_df.to_csv(consolidated_file, index=False)
            self.logger.info(f"Saved consolidated summary to {consolidated_file}")
    
    def _visualize_summary(self, summary_df, timestamp):
        """
        Create visualization of the summary report
        
        Args:
            summary_df: DataFrame with summary data
            timestamp: Timestamp string
        """
        # Create separate plots for pivot highs and lows
        for pivot_type in ['high', 'low']:
            # Filter data
            pivot_data = summary_df[summary_df['pivot_type'] == pivot_type]
            
            if pivot_data.empty:
                continue
            
            # Group by symbol and timeframe
            symbols = pivot_data['symbol'].unique()
            timeframes = pivot_data['timeframe'].unique()
            
            # Create figure
            fig_width = max(12, 3 * len(symbols))
            fig_height = max(8, 2 * len(timeframes))
            fig, axes = plt.subplots(len(timeframes), 1, figsize=(fig_width, fig_height), squeeze=False)
            
            # Get level values
            levels = pivot_data['retracement_level'].unique()
            
            # Plot for each timeframe
            for i, tf in enumerate(timeframes):
                ax = axes[i, 0]
                
                # Filter data for this timeframe
                tf_data = pivot_data[pivot_data['timeframe'] == tf]
                
                # For each level, plot probabilities across symbols
                width = 0.8 / len(levels)
                
                for j, level in enumerate(sorted(levels)):
                    level_data = tf_data[tf_data['retracement_level'] == level]
                    
                    # Extract probabilities for each symbol
                    x = np.arange(len(symbols))
                    probs = []
                    
                    for symbol in symbols:
                        symbol_data = level_data[level_data['symbol'] == symbol]
                        if not symbol_data.empty:
                            probs.append(symbol_data['probability'].values[0])
                        else:
                            probs.append(0)
                    
                    # Plot bar for this level
                    offset = (j - len(levels) / 2 + 0.5) * width
                    color = 'red' if pivot_type == 'high' else 'green'
                    bars = ax.bar(x + offset, probs, width, label=f'{level*100:.1f}%', color=color, alpha=0.5 + 0.5*j/len(levels))
                    
                    # Add value labels
                    for k, bar in enumerate(bars):
                        height = bar.get_height()
                        if height > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height + 0.02,
                                f'{height:.2f}',
                                ha='center', va='bottom',
                                fontsize=8
                            )
                
                # Set labels and title
                ax.set_title(f'Retracement Probabilities for {tf} Timeframe (Pivot {pivot_type.capitalize()}s)')
                ax.set_xlabel('Symbol')
                ax.set_ylabel('Probability')
                ax.set_xticks(x)
                ax.set_xticklabels(symbols)
                ax.grid(alpha=0.3)
                ax.legend(title='Retracement Level')
                ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            # Save figure
            summary_plot_file = self.output_dir / f"retracement_summary_{pivot_type}_{timestamp}.png"
            plt.savefig(summary_plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Saved summary visualization to {summary_plot_file}")

def main():
    """Main function to run retracement analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze retracement levels')
    parser.add_argument('--config', type=str, default='config/base.yaml', help='Path to configuration file')
    parser.add_argument('--symbols', type=str, nargs='+', help='Specific symbols to process (e.g., BTC/USD ETH/USDT)')
    parser.add_argument('--timeframes', type=str, nargs='+', help='Specific timeframes to process (e.g., 1h 4h 1d)')
    parser.add_argument('--exchanges', type=str, help='Specific exchanges to use')
    parser.add_argument('--pivot-type', type=str, choices=['high', 'low', 'both'], default='both', help='Type of pivots to analyze')
    parser.add_argument('--look-forward', type=int, default=48, help='Number of bars to look forward for retracement targets')
    parser.add_argument('--pivot-window', type=int, default=5, help='Window size for pivot detection')
    
    args = parser.parse_args()
    
    # Initialize parameter manager
    params = ParamManager.get_instance(
        base_config_path=args.config,
        cli_args=args,
        env_vars=True
    )
    
    # Run retracement analysis
    analyzer = RetracementAnalyzer(params)
    results = analyzer.run_analysis()
    
    if results:
        logger.info("Retracement analysis completed successfully")
    else:
        logger.error("Retracement analysis failed")

if __name__ == "__main__":
    main()