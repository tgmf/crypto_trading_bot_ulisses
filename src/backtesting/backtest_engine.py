#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting engine for trading strategies with proper train-test separation.

This enhanced version maintains compatibility with the original backtest engine 
while adding support for proper train-test separation. It prioritizes using 
separate test sets when available for more reliable performance evaluation.

Key features:
- Automatically uses test data when available
- Tracks data source in results for transparency
- Supports both individual and multi-symbol backtesting
- Maintains original function signatures for backwards compatibility
"""

import logging
import pandas as pd
import numpy as np
import pickle
import arviz as az
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import json

from ..models.model_factory import ModelFactory
from ..core.result_logger import ResultLogger

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, params):
        """
        Initialize the backtest engine with paramsuration
        
        Args:
            params (dict): ParamManager instance containing backtesting parameters
        """
        self.params = params
        self.logger = logging.getLogger(__name__)
        # Extract fee rate from config or use default value
        # Extract fee rate and thresholds from params
        self.fee_rate = self.params.get('exchange', 'fee_rate', default=0.0006)
        self.exit_threshold = self.params.get('backtesting', 'exit_threshold', default=0.3)
        
        # Create result logger instance for consistent output
        self.result_logger = ResultLogger(params)
        
    def run_test(self):
        """
        Run backtest for a given model and dataset
        
        This function has been enhanced to prioritize using separate test data when available.
        It follows these steps:
        1. Check if a separate test set exists and use it if available
        2. Fall back to the full processed dataset if no test set exists
        3. Load the appropriate model
        4. Generate predictions
        5. Execute trading logic
        6. Save and visualize results with data source tracking
            
        Returns:
            tuple or bool: (backtest_results, metrics) if successful, False otherwise
        """
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        strategy = params.get('strategy', 'type', default='quantum')
        test_sets_path = self.params.get('backtesting', 'test_sets', 'path', default='data/test_sets')
        processed_path = self.params.get('data', 'processed', 'path', default='data/processed')
        
        self.logger.info(f"Running backtest for {exchange} {symbol} {timeframe}")
        
        try:
            # Check for test data first
            symbol_safe = symbol.replace('/', '_')
            test_file = Path(f"{test_sets_path}/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            # If test data exists, use it (preferred for proper evaluation)
            if test_file.exists():
                self.logger.info(f"Using held-out test data from {test_file}")
                df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                data_source = "test_set"  # Track the data source
            else:
                # Fall back to processed data with a warning
                self.logger.warning(f"No test set found at {test_file}. Using processed data instead. "
                        f"This may lead to overoptimistic results due to possible data leakage.")
                
                input_file = Path(f"{processed_path}/{exchange}/{symbol_safe}/{timeframe}.csv")
                
                if not input_file.exists():
                    self.logger.error(f"No processed data file found at {input_file}")
                    return False
                    
                df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                data_source = "full_data"  # Track the data source
            
            # Load model through the model factory
            model_factory = ModelFactory(self.params)
            model = model_factory.create_model()
            
            if not model.load_model():
                self.logger.error("Failed to load model for backtesting")
                return False
            
            # Make sure the dataframe is sorted by timestamp
            df = df.sort_index()
            # Get predictions from the model
            probabilities = model.predict_probabilities(df)
            
            if probabilities is None:
                self.logger.error("Failed to get predictions for backtesting")
                return False
            
            # TODO: implement switching between different strategies 
            # Run backtest with quantum-inspired approach
            backtest_results, metrics = self._run_quantum_backtest(df, probabilities)
        
            # Add data source to metrics
            metrics['data_source'] = data_source
            
            # Save and plot results
            self.result_logger.save_results(backtest_results, metrics)
            self.result_logger.plot_results(backtest_results, metrics)
            
            return backtest_results, metrics
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_quantum_backtest(self, df, probabilities, threshold=0.5, hedge_threshold=0.48):
        """
        Run backtest with quantum-inspired trading approach
        
        This implements a quantum-inspired trading strategy that allows for three distinct states:
        - Long (position=1): Betting on price increase
        - Short (position=-1): Betting on price decrease
        - Neutral (position=0): No position
        - Hedged (position=2): Both long and short simultaneously
        
        The approach uses probability thresholds to determine state transitions and manages
        position entries, exits, and hedging based on changing probability distributions.
        
        Args:
            df (DataFrame): Price data DataFrame with OHLCV data
            probabilities (ndarray): Probability array [P(short), P(no_trade), P(long)]
            threshold (float): Minimum probability to enter a position (default: 0.2)
            hedge_threshold (float): Threshold for considering hedging (default: 0.4)
            
        Returns:
            tuple: (df_backtest, metrics) - DataFrame with backtest results and performance statistics
        """
        df_backtest = df.copy()
        
        # Initialize columns for tracking positions and returns
        df_backtest['short_prob'] = probabilities[:, 0]  # Probability of short position
        df_backtest['no_trade_prob'] = probabilities[:, 1]  # Probability of no position
        df_backtest['long_prob'] = probabilities[:, 2]  # Probability of long position
        df_backtest['position'] = 0  # Position state: -1=short, 0=flat, 1=long, 2=hedged
        df_backtest['entry_price'] = np.nan  # Price at position entry
        df_backtest['exit_price'] = np.nan  # Price at position exit
        df_backtest['trade_return'] = np.nan  # Return of individual trades
        df_backtest['trade_duration'] = np.nan  # Duration of trades in bars
        
        # Simulate trading
        position = 0  # Start with no position
        entry_idx = 0  # Index of entry bar
        entry_price = 0  # Price at entry
        
        for i in range(1, len(df_backtest) - 1):
            current_price = df_backtest['close'].iloc[i]
            
            # Get current probabilities
            short_prob = probabilities[i, 0]  # Probability of profitable short opportunity
            no_trade_prob = probabilities[i, 1]  # Probability of no profitable opportunity
            long_prob = probabilities[i, 2]  # Probability of profitable long opportunity
            
            # Update position based on current state and probabilities
            if position == 0:  # Currently flat
                # Check for new position signals
                if long_prob > threshold:
                    # Enter long position when long probability exceeds threshold
                    position = 1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                
                elif short_prob > threshold:
                    # Enter short position when short probability exceeds threshold
                    position = -1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                
                # Handle quantum edge case: both probabilities high
                elif long_prob > hedge_threshold and short_prob > hedge_threshold:
                    # Enter hedged position if both signals are strong - this is unique to quantum approach
                    # Hedging means simultaneously holding long and short positions
                    position = 2  # Hedged
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
            
            elif position == 1:  # Currently long
                # Check for exit or hedge signals
                if long_prob < self.exit_threshold or short_prob > threshold:
                    # Exit long position if long probability drops or short probability rises
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Calculate return (accounting for fees)
                    # Return = (exit price / entry price) - 1 - round-trip fees
                    trade_return = (current_price / entry_price) - 1 - (self.fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Check for immediate reversal to short
                    if short_prob > threshold:
                        # If short signal is strong, immediately enter short position
                        position = -1
                        entry_idx = i
                        entry_price = current_price
                        df_backtest.loc[df_backtest.index[i], 'position'] = -1
                        df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    else:
                        position = 0
                
                elif short_prob > hedge_threshold:
                    # Add hedge to long position if short signal strengthens
                    position = 2  # Hedged
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                
                else:
                    # Stay in long position if no exit signals
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
            
            elif position == -1:  # Currently short
                # Check for exit or hedge signals
                if short_prob < self.exit_threshold or long_prob > threshold:
                    # Exit short position if short probability drops or long probability rises
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Calculate return (accounting for fees)
                    # Short return = 1 - (exit price / entry price) - round-trip fees
                    trade_return = 1 - (current_price / entry_price) - (self.fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Check for immediate reversal to long
                    if long_prob > threshold:
                        # If long signal is strong, immediately enter long position
                        position = 1
                        entry_idx = i
                        entry_price = current_price
                        df_backtest.loc[df_backtest.index[i], 'position'] = 1
                        df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    else:
                        position = 0
                
                elif long_prob > hedge_threshold:
                    # Add hedge to short position if long signal strengthens
                    position = 2  # Hedged
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                
                else:
                    # Stay in short position if no exit signals
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
            
            elif position == 2:  # Currently hedged
                # Check for removing hedge
                if long_prob < hedge_threshold and short_prob < hedge_threshold:
                    # Exit hedged position if both signals weaken
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Hedged positions typically have near-zero returns plus double fees
                    # We approximate this as the cost of the double hedging fees
                    trade_return = -(self.fee_rate * 4)  # Approximate cost of hedging
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    position = 0
                
                elif long_prob > threshold and short_prob < hedge_threshold:
                    # Convert hedge to pure long if long signal strengthens and short weakens
                    position = 1
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
                
                elif short_prob > threshold and long_prob < hedge_threshold:
                    # Convert hedge to pure short if short signal strengthens and long weakens
                    position = -1
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
                
                else:
                    # Stay hedged if signals remain conflicted
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
        
        # Close any open position at the end
        if position != 0:
            i = len(df_backtest) - 1
            current_price = df_backtest['close'].iloc[i]
            df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
            
            if position == 1:  # Long position
                trade_return = (current_price / entry_price) - 1 - (self.fee_rate * 2)
            elif position == -1:  # Short position
                trade_return = 1 - (current_price / entry_price) - (self.fee_rate * 2)
            else:  # Hedged position
                trade_return = -(self.fee_rate * 4)  # Approximate cost of hedging
            
            df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
            df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
        
        # Calculate strategy returns
        df_backtest['returns'] = df_backtest['close'].pct_change()
        
        # For simplicity, we'll just use position values (-1, 0, 1, 2) directly
        # For position=2 (hedged), we assume the return is approximately zero minus fees
        df_backtest['strategy_returns'] = 0.0
        
        long_mask = df_backtest['position'].shift(1) == 1
        short_mask = df_backtest['position'].shift(1) == -1
        hedged_mask = df_backtest['position'].shift(1) == 2
        
        df_backtest.loc[long_mask, 'strategy_returns'] = df_backtest.loc[long_mask, 'returns']
        df_backtest.loc[short_mask, 'strategy_returns'] = -df_backtest.loc[short_mask, 'returns']
        df_backtest.loc[hedged_mask, 'strategy_returns'] = 0  # Approximate hedged return as 0
        
        # Calculate cumulative returns
        df_backtest['cumulative_returns'] = (1 + df_backtest['returns']).cumprod() - 1
        df_backtest['strategy_cumulative'] = (1 + df_backtest['strategy_returns']).cumprod() - 1
        
        # Calculate trading statistics
        trades = df_backtest[~df_backtest['trade_return'].isna()]
        
        metrics = {
            'total_trades': len(trades),
            'long_trades': (trades['position'] == 1).sum(),
            'short_trades': (trades['position'] == -1).sum(),
            'hedged_trades': (trades['position'] == 2).sum(),
            'win_rate': (trades['trade_return'] > 0).mean() if len(trades) > 0 else 0,
            'avg_win': trades.loc[trades['trade_return'] > 0, 'trade_return'].mean() 
                        if (trades['trade_return'] > 0).any() else 0,
            'avg_loss': trades.loc[trades['trade_return'] < 0, 'trade_return'].mean() 
                        if (trades['trade_return'] < 0).any() else 0,
            'profit_factor': abs(trades.loc[trades['trade_return'] > 0, 'trade_return'].sum() / 
                                trades.loc[trades['trade_return'] < 0, 'trade_return'].sum()) 
                                if (trades['trade_return'] < 0).any() and 
                                    (trades['trade_return'] > 0).any() else float('inf'),
            'avg_trade_duration': trades['trade_duration'].mean() if len(trades) > 0 else 0,
            'final_return': df_backtest['strategy_cumulative'].iloc[-1]
        }
        
        return df_backtest, metrics
        
    def run_multi_test(self):
        """
        Run backtest across multiple symbols and timeframes using the latest trained model
        
        Returns:
            dict: Results dictionary or False if error
        """
        
        # Store symbols and timeframes in params
        symbols = self.params.get('data', 'symbols')
        timeframes = self.params.get('data', 'timeframes')
        exchanges = self.params.get('data', 'exchanges')
        
        self.logger.info(f"Running multi-symbol backtest for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Define a dictionary to store results
        all_results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Run backtest for each exchange/symbol/timeframe
        for exchange in exchanges:
            exchange_results = {}
            for symbol in symbols:
                symbol_results = {}
                for timeframe in timeframes:
                    try:
                        # Use the existing run_test method for each combination
                        self.logger.info(f"Testing {exchange} {symbol} {timeframe}")
                
                        # Update params for this specific symbol/timeframe for model loading
                        self.params.set([symbol], 'data', 'symbols')
                        self.params.set([timeframe], 'data', 'timeframes')
                
                        # Run the individual backtest
                        #TODO: run_test() called by multitest should not save and plot test results for each symbol/timeframe/exchange 
                        backtest_results, metrics = self.run_test(exchange, symbol, timeframe)
                
                        if backtest_results is False:
                            self.logger.warning(f"Backtest failed for {symbol} {timeframe}")
                            continue
                
                        # Ensure data_source is in metrics (in case run_test doesn't include it)
                        if 'data_source' not in metrics:
                            # Check if test file exists to determine source
                            symbol_safe = symbol.replace('/', '_')
                            test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
                            metrics['data_source'] = "test_set" if test_file.exists() else "full_data"
                    
                    # Store results and include data source from metrics
                        data_source = metrics.get('data_source', 'unknown')
                
                        symbol_results[timeframe] = {
                            'results': backtest_results,
                            'metrics': metrics,
                            'data_source': data_source
                        }
                
                    except Exception as e:
                        self.logger.error(f"Error during backtest of {symbol} {timeframe}: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        continue
                
                if symbol_results:
                    exchange_results[symbol] = symbol_results
                
            if exchange_results:
                all_results[exchange] = exchange_results
        
        # Create a summary report if we have results
        if all_results:
            self.result_logger.create_multi_summary(all_results, timestamp)
        else:
            self.logger.warning("No successful backtests to summarize")
        
        return all_results

    def run_test_on_test_set(self, custom_testset=None):
        """
        Run backtest strictly on test set data to ensure unbiased evaluation
        """
        symbol = self.params.get('data', 'symbols', 0)
        timeframe = self.params.get('data', 'timeframes', 0)
        exchange = self.params.get('data', 'exchanges', 0)
        testset_path = self.params.get('backtest', 'testsets', 'path')
        
        self.logger.info(f"Running backtest on test set for {exchange} {symbol} {timeframe}")
        
        try:
            # Load the test data specifically
            symbol_safe = symbol.replace('/', '_')
            test_file = Path(f"{testset_path}/{custom_testset}") or Path(f"{testset_path}/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            if not test_file.exists():
                self.logger.error(f"Test data file not found at {test_file}. Please run training first.")
                return False
                
            test_df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
            self.logger.info(f"Loaded test set with {len(test_df)} samples")
            
            # Load model
            model_factory = ModelFactory(self.params)
            model = model_factory.create_model()
            
            if not model.load_model():
                self.logger.error("Failed to load model for backtesting")
                return False
            
            # Run predictions using the trained model
            probabilities = model.predict_probabilities(test_df)
            
            if probabilities is None:
                self.logger.error("Failed to get predictions for backtesting")
                return False
            
            # Run backtest with quantum-inspired approach on test data only
            backtest_results, metrics = self._run_quantum_backtest(test_df, probabilities)
            
            metrics['data_source'] = "test_set"  # Track the data source
            
            # Save and plot results
            self.result_logger.save_results(backtest_results, metrics)
            self.result_logger.plot_results(backtest_results, metrics)
            
            # Return results
            return backtest_results, metrics
            
        except Exception as e:
            self.logger.error(f"Error during test set backtesting: {str(e)}")
            return False
        
    def run_position_sizing_test(self, no_trade_threshold=0.96, min_position_change=0.025):
        """
        Run a backtest with quantum position sizing instead of traditional binary signals.
        
        This method:
        1. Loads the appropriate test set or falls back to processed data
        2. Gets probability predictions from the model
        3. Applies quantum-inspired position sizing 
        4. Evaluates and visualizes performance
        
        Args:
            no_trade_threshold (float): Threshold for no_trade probability to ignore signals
            min_position_change (float): Minimum position change to avoid fee churn
                
        Returns:
            tuple or bool: (backtest_results, metrics, fig) if successful, False otherwise
        """
        
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        strategy = params.get('strategy', 'type', default='quantum')
        
        test_sets_path = self.params.get('backtesting', 'test_sets', 'path', default='data/test_sets')

        self.logger.info(f"Running position sizing test for {exchange} {symbol} {timeframe}")
        
        try:
            # Check for test data first
            symbol_safe = symbol.replace('/', '_')
            test_file = Path(f"{test_sets_path}/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            # If test data exists, use it (preferred for proper evaluation)
            if test_file.exists():
                self.logger.info(f"Using held-out test data from {test_file}")
                df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                data_source = "test_set"  # Track the data source
            else:
                # Fall back to processed data with a warning
                self.logger.warning(f"No test set found at {test_file}. Using processed data instead. "
                        f"This may lead to overoptimistic results due to possible data leakage.")
                
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                
                if not input_file.exists():
                    self.logger.error(f"No processed data file found at {input_file}")
                    return False
                    
                df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                data_source = "full_data"  # Track the data source
            
            # Load model through the model factory
            model_factory = ModelFactory(self.params)
            model = model_factory.create_model()
            
            if not model.load_model():
                self.logger.error("Failed to load model for backtesting")
                return False
            
            # Run backtest with position sizing (delegates to the model's method)
            # Check if the model class has the position sizing method
            if hasattr(model, 'run_backtest_with_position_sizing'):
                return_value = model.run_backtest_with_position_sizing(
                    df, 
                    no_trade_threshold=no_trade_threshold,
                    min_position_change=min_position_change
                )
                
                # Handle different return formats
                if isinstance(return_value, tuple):
                    if len(return_value) == 2:
                        # Model only returned results and metrics, no figure
                        results, metrics = return_value
                        fig = None
                    else:
                        # Model returned all three values
                        results, metrics, fig = return_value
                else:
                    # Model returned a single value or False
                    return return_value
                
                # Add data source information
                if isinstance(metrics, dict):
                    metrics['data_source'] = data_source
                
                return results, metrics, fig
            else:
                self.logger.error("Model does not support position sizing backtest")
                return False
        
        except Exception as e:
            self.logger.error(f"Error during position sizing test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


    def run_multi_position_sizing_test(self, symbols, timeframes, exchange='binance',
                                    no_trade_threshold=0.96, min_position_change=0.05):
        """
        Run position sizing tests across multiple symbols and timeframes
        
        Args:
            symbols (list): List of trading pair symbols
            timeframes (list): List of timeframes to test
            exchange (str): Exchange name
            no_trade_threshold (float): Threshold for no_trade probability
            min_position_change (float): Minimum position change threshold
                
        Returns:
            dict: Dictionary of results for all symbols and timeframes
        """
        self.logger.info(f"Running multi-symbol position sizing tests: {symbols}, {timeframes}")
        
        # Dictionary to store all results
        all_results = {}
        
        # Process each symbol
        for symbol in symbols:
            symbol_results = {}
            
            # Process each timeframe for this symbol
            for timeframe in timeframes:
                self.logger.info(f"Running position sizing test for {symbol} {timeframe}")
                
                try:
                    # Run position sizing test for this symbol/timeframe
                    results, metrics, fig = self.run_position_sizing_test(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        no_trade_threshold=no_trade_threshold,
                        min_position_change=min_position_change
                    )
                    
                    # Store results if successful
                    if results is not False:
                        symbol_results[timeframe] = {
                            'results': results,
                            'metrics': metrics,
                            'fig': fig
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error in position sizing test for {symbol} {timeframe}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
            
            # Store all timeframe results for this symbol
            if symbol_results:
                all_results[symbol] = symbol_results
        
        # Create a summary report comparing performance across symbols and timeframes
        if all_results:
            self._create_position_sizing_summary(all_results, symbols, timeframes)
            self.logger.info(f"Multi-symbol position sizing tests complete for {len(all_results)} symbols")
        else:
            self.logger.warning("No valid results were generated in multi-symbol position sizing tests")
        
        return all_results


    def _create_position_sizing_summary(self, all_results, symbols, timeframes):
        """
        Create summary report for multi-symbol position sizing tests
        
        Args:
            all_results (dict): Results dictionary
            symbols (list): List of symbols
            timeframes (list): List of timeframes
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a DataFrame for summary metrics
            summary_rows = []
            
            for symbol in symbols:
                if symbol not in all_results:
                    continue
                    
                for timeframe in timeframes:
                    if timeframe not in all_results[symbol]:
                        continue
                    
                    # Get performance metrics
                    metrics = all_results[symbol][timeframe]['metrics']
                    
                    summary_rows.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'data_source': metrics.get('data_source', 'unknown'),
                        'total_return': metrics.get('total_return', 0),
                        'alpha': metrics.get('alpha', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'profit_factor': metrics.get('profit_factor', 0),
                        'num_position_changes': metrics.get('num_position_changes', 0),
                        'fee_drag': metrics.get('fee_drag', 0)
                    })
            
            if not summary_rows:
                self.logger.warning("No summary data available")
                return False
            
            # Convert to DataFrame
            summary_df = pd.DataFrame(summary_rows)
            
            # Save summary
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_type = self.params.get('model', 'type')
            output_dir = Path(f"data/backtest_results/{symbol}/{timeframe}/{model_type}/summary")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_file = output_dir / f"pos_sizing_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            
            # Create summary visualizations
            self._plot_position_sizing_summary(summary_df, output_dir, timestamp)
            
            self.logger.info(f"Saved position sizing summary report to {summary_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating position sizing summary report: {str(e)}")
            return False


    def _plot_position_sizing_summary(self, summary_df, output_dir, timestamp):
        """
        Create summary visualizations for position sizing tests
        
        Args:
            summary_df (DataFrame): Summary DataFrame with performance metrics
            output_dir (Path): Directory to save plots
            timestamp (str): Timestamp string for filenames
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Plot 1: Returns by symbol
            plt.figure(figsize=(12, 6))
            
            # Group by symbol and calculate mean return
            symbol_returns = summary_df.groupby('symbol')['total_return'].mean().sort_values(ascending=False) * 100
            
            # Create bar chart
            plt.bar(symbol_returns.index, symbol_returns.values)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.title('Average Return by Symbol (Position Sizing)')
            plt.xlabel('Symbol')
            plt.ylabel('Return (%)')
            plt.xticks(rotation=45)
            
            # Add text annotations
            for i, v in enumerate(symbol_returns):
                plt.text(i, v + 1, f'{v:.2f}%',
                        ha='center', va='bottom',
                        color='green' if v > 0 else 'red')
            
            plt.tight_layout()
            returns_file = output_dir / f"pos_sizing_returns_by_symbol_{timestamp}.png"
            plt.savefig(returns_file)
            plt.close()
            
            # Plot 2: Sharpe ratio comparison
            plt.figure(figsize=(12, 6))
            
            # Group by symbol and calculate mean Sharpe
            symbol_sharpe = summary_df.groupby('symbol')['sharpe_ratio'].mean().sort_values(ascending=False)
            
            plt.bar(symbol_sharpe.index, symbol_sharpe.values)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='Min acceptable')
            
            plt.title('Average Sharpe Ratio by Symbol (Position Sizing)')
            plt.xlabel('Symbol')
            plt.ylabel('Sharpe Ratio')
            plt.xticks(rotation=45)
            plt.legend()
            
            # Add text annotations
            for i, v in enumerate(symbol_sharpe):
                plt.text(i, v + 0.1, f'{v:.2f}',
                        ha='center', va='bottom',
                        color='green' if v > 1 else 'red')
            
            plt.tight_layout()
            sharpe_file = output_dir / f"pos_sizing_sharpe_by_symbol_{timestamp}.png"
            plt.savefig(sharpe_file)
            plt.close()
            
            # Plot 3: Fee drag comparison
            plt.figure(figsize=(12, 6))
            
            # Group by symbol and calculate mean fee drag
            symbol_fees = summary_df.groupby('symbol')['fee_drag'].mean().sort_values() * 100
            
            plt.bar(symbol_fees.index, symbol_fees.values)
            
            plt.title('Fee Drag by Symbol (Position Sizing)')
            plt.xlabel('Symbol')
            plt.ylabel('Fee Drag (%)')
            plt.xticks(rotation=45)
            
            # Add text annotations
            for i, v in enumerate(symbol_fees):
                plt.text(i, v + 0.01, f'{v:.3f}%',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            fee_file = output_dir / f"pos_sizing_fee_drag_by_symbol_{timestamp}.png"
            plt.savefig(fee_file)
            plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating position sizing summary plots: {str(e)}")
            # Continue execution even if plotting fails
            return False