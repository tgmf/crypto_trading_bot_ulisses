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
from ..data.data_context import DataContext

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
        self.taker_fee_rate = self.params.get('exchange', 'taker_fee_rate', default=0.0006)
        self.exit_threshold = self.params.get('strategy', 'exit_threshold', default=0.3)
        
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
        custom_test_set = params.get('backtesting', 'test_sets', 'custom_test_set', default=None)
        test_set_only = params.get('backtesting', 'test_sets', 'test_set_only', default=False)
        
        # Check if subset testing is requested
        subset_str = self.params.get('backtesting', 'test_sets', 'subset', default=None)
        subset_index, total_subsets = DataContext.parse_subset_string(subset_str)
        
        self.logger.info(f"Running backtest for {exchange} {symbol} {timeframe}")
        
        try:
            # Handle custom test set if provided
            if custom_test_set and custom_test_set != 'None':
                # Load custom test set into a DataContext
                data_context = DataContext(self.params)
                
                # Parse the custom testset path
                test_file = Path(custom_test_set)
                if not test_file.exists():
                    self.logger.error(f"Custom test data file not found at {test_file}")
                    return False
                    
                # Load data into context
                data_context.df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                data_context.source = "custom_test_set"
                data_context.add_processing_step("load_custom_test_set", {
                    "file_path": str(test_file),
                    "rows": len(data_context.df)
                })
            else:
                # Try to load test set using DataContext
                data_context = DataContext.from_test_set(self.params, exchange, symbol, timeframe)
                
                if data_context is not None:
                    self.logger.info(f"Using held-out test data from test set")
                    data_context.add_processing_step("load_test_set", {
                        "source": "test_set"
                    })
                elif test_set_only:
                    # If test_set_only is True and no test set is found, return False
                    self.logger.error(f"No test set found for {symbol} {timeframe} and test_set_only=True")
                    return False
                else:
                    # Fall back to processed data with a warning
                    self.logger.warning(f"No test set found. Using processed data instead. "
                            f"This may lead to overoptimistic results due to possible data leakage.")
                    
                    data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
                    
            if data_context is None:
                self.logger.error(f"No processed data found for {symbol} {timeframe}")
                return False
                
            data_context.add_processing_step("load_processed_data", {
                "source": "processed_data",
                "warning": "Using full data instead of test set could lead to data leakage"
            })
            
            # Validate data has the required columns
            if not data_context.validate(required_columns=['open', 'high', 'low', 'close', 'volume']):
                self.logger.error("Data validation failed - missing required columns")
                return False
            
            # Apply subset if requested
            if subset_index is not None and total_subsets is not None:
                try:
                    data_context.create_subset(subset_index, total_subsets)
                except ValueError as e:
                    self.logger.error(f"Error creating subset: {str(e)}")
                    return False
                
            # Load model through the model factory
            model_factory = ModelFactory(self.params)
            model = model_factory.create_model()
            
            if not model.load_model():
                self.logger.error("Failed to load model for backtesting")
                data_context.add_processing_step("model_load_error", {
                    "error": "Failed to load model for backtesting"
                })
                return False
        
            data_context.add_processing_step("model_loaded", {
                "model_type": self.params.get('model', 'type', default='auto'),
                "strategy": strategy
            })
        
            # Make sure the dataframe is sorted by timestamp
            if not data_context.df.index.is_monotonic_increasing:
                data_context.df = data_context.df.sort_index()
                data_context.add_processing_step("sort_index", {
                    "reason": "Ensure chronological order for backtesting"
                })
        
            # Get predictions from the model
            self.logger.info(f"Generating predictions for {len(data_context.df)} samples")
            probabilities = model.predict_probabilities(data_context.df)
        
            if probabilities is None:
                self.logger.error("Failed to get predictions for backtesting")
                data_context.add_processing_step("prediction_error", {
                    "error": "Failed to get probability predictions from model"
                })
                return False
        
            data_context.add_processing_step("predictions_generated", {
                "predictions_shape": probabilities.shape,
                "data_shape": data_context.df.shape
            })
            
            # TODO: implement switching between different strategies 
            # Run backtest with quantum-inspired approach
            self.logger.info(f"Running {strategy} backtest with {len(data_context.df)} samples")
            backtest_results, metrics = self._run_quantum_backtest(data_context, probabilities)
        
            # Add data source to metrics
            metrics['data_source'] = data_context.source
            
            # Add processing history to metrics for full reproducibility
            metrics['processing_history'] = data_context.get_processing_history()
        
            # Generate execution ID for tracking
            is_test_data = data_context.source in ["test_set", "custom_test_set"]
            execution_suffix = "_test_set" if is_test_data else ""
            execution_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}_{timeframe}{execution_suffix}"
            metrics['execution_id'] = execution_id
        
            # Add final processing step
            data_context.add_processing_step("backtest_complete", {
                "execution_id": execution_id,
                "final_return": metrics.get('final_return', 0),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "total_trades": metrics.get('total_trades', 0)
            })
            
            # Set correct suffix for saving results
            if subset_index is not None:
                suffix = f"_subset_{subset_index + 1}_of_{total_subsets}"
                self.params.set(suffix, 'model', 'suffix')
            
            self.result_logger.save_results(backtest_results, metrics)
            self.result_logger.plot_results(backtest_results, metrics)
            
            # Reset suffix after saving
            if subset_index is not None:
                self.params.set(None, 'model', 'suffix')
            
            self.logger.info(f"Backtest completed successfully with execution ID: {execution_id}")
            return backtest_results, metrics
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(error_trace)
            
            # If we have a data context, record the error
            if locals().get('data_context') is not None:
                data_context.add_processing_step("backtest_error", {
                    "error": str(e),
                    "traceback": error_trace
                })
            
            return False
    
    def _run_quantum_backtest(self, data_context, probabilities, threshold=0.52, hedge_threshold=0.5):
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
            data_context (DataContext): Data context containing price DataFrame
            probabilities (ndarray): Probability array [P(short), P(no_trade), P(long)]
            threshold (float): Minimum probability to enter a position (default: 0.2)
            hedge_threshold (float): Threshold for considering hedging (default: 0.4)
            
        Returns:
            tuple: (df_backtest, metrics) - DataFrame with backtest results and performance statistics
        """
        
        fee_rate = self.taker_fee_rate
        stop_loss = self.params.get('strategy', 'stop_loss', default=0)
    
        # Get risk-reward ratio from params (default to 1:1)
        risk_reward_ratio = self.params.get('strategy', 'risk_reward_ratio', default=1)
        # Calculate take profit based on stop loss and R:R ratio
        take_profit = stop_loss * risk_reward_ratio
        
        # Record backtest parameters in data context
        data_context.add_processing_step("backtest_start", {
            "strategy": "quantum",
            "enter_threshold": threshold,
            "hedge_threshold": hedge_threshold,
            "exit_threshold": self.exit_threshold,
            "fee_rate": fee_rate,
            "stop_loss": stop_loss,
            "risk_reward_ratio": risk_reward_ratio,
            "take_profit": take_profit
        })
        
        df = data_context.df
        df_backtest = df.copy()
        threshold = self.params.get('strategy', 'enter_threshold', default=threshold)
        hedge_threshold = self.params.get('strategy', 'hedge_threshold', default=hedge_threshold)
        threshold_bias = self.params.get('strategy', 'threshold_bias', default=1)
        # Different thresholds for long and short
        long_threshold = threshold * threshold_bias # More conservative
        short_threshold = threshold / threshold_bias # More aggressive
        long_exit_threshold = self.exit_threshold * threshold_bias # More conservative
        short_exit_threshold = self.exit_threshold / threshold_bias # More aggressive
        
        # Initialize columns for tracking positions and returns
        df_backtest['short_prob'] = probabilities[:, 0]  # Probability of short position
        df_backtest['no_trade_prob'] = probabilities[:, 1]  # Probability of no position
        df_backtest['long_prob'] = probabilities[:, 2]  # Probability of long position
        df_backtest['position'] = 0  # Position state: -1=short, 0=flat, 1=long, 2=hedged
        df_backtest['entry_price'] = np.nan  # Price at position entry
        df_backtest['exit_price'] = np.nan  # Price at position exit
        df_backtest['trade_return'] = np.nan  # Return of individual trades
        df_backtest['trade_duration'] = np.nan  # Duration of trades in bars
    
        # New columns for fee tracking
        df_backtest['fee_impact'] = 0.0  # Fee amount for each trade
        df_backtest['total_fees_paid'] = 0.0  # Running total of fees paid
        df_backtest['closed_position_type'] = np.nan  # Type of closed position: long, short, hedged
        # Initialize exit_reason with object dtype to handle string values
        df_backtest['exit_reason'] = pd.Series(np.nan, index=df_backtest.index, dtype='object')  # Reason for position exit: signal, stop_loss, etc.

        # Add column to track if position matches target (only if target exists in df)
        if 'target' in df.columns:
            df_backtest['target_match'] = False
        
        # Simulate trading
        position = 0  # Start with no position
        entry_idx = 0  # Index of entry bar
        entry_price = 0  # Price at entry
        total_fees_paid = 0.0  # Track total fees paid
        trade_count = 0  # Count of executed trades
        stop_loss_exits = 0  # Count of stop loss exits
        take_profit_exits = 0  # Count of take profit exits
        
        data_context.add_processing_step("backtest_columns_initialized", {
            "position_states": "[-1=short, 0=flat, 1=long, 2=hedged]",
            "data_rows": len(df_backtest),
            "stop_loss_enabled": (stop_loss > 0),
            "take_profit_enabled": (take_profit > 0),
            "risk_reward_ratio": risk_reward_ratio
        })
        
        for i in range(1, len(df_backtest) - 1):
            current_price = df_backtest['close'].iloc[i]
            
            # Get current probabilities
            short_prob = probabilities[i, 0]  # Probability of profitable short opportunity
            no_trade_prob = probabilities[i, 1]  # Probability of no profitable opportunity
            long_prob = probabilities[i, 2]  # Probability of profitable long opportunity
            
            # Check if stop loss is triggered before other logic
            stop_loss_triggered = False
            take_profit_triggered = False
            
            if position == 1:  # Check stop loss and take profit for long position
                # Calculate current percentage loss
                current_return = (current_price / entry_price) - 1
                
                if stop_loss and current_return < -stop_loss:
                    # Stop loss triggered for long position
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "stop_loss"
                    
                    # Calculate return (accounting for fees)
                    trade_return = (current_price / entry_price) - 1 - (fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Calculate exit fee
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                    
                    # Record fee impact
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    
                    position = 0
                    stop_loss_triggered = True
                    stop_loss_exits += 1
                
                elif take_profit and current_return > take_profit:  # R:R based take profit check
                    # Take profit triggered for long position
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "take_profit"
                    
                    # Calculate return (accounting for fees)
                    trade_return = (current_price / entry_price) - 1 - (fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Calculate exit fee
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                    
                    # Record fee impact
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    
                    position = 0
                    take_profit_triggered = True
                    take_profit_exits += 1
                    
            elif position == -1:  # Check stop loss for short position
                # Calculate current percentage loss for short (reverse logic)
                current_return = 1 - (current_price / entry_price)
                
                if stop_loss and current_return < -stop_loss:
                    # Stop loss triggered for short position
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "stop_loss"
                    
                    # Calculate return (accounting for fees)
                    trade_return = 1 - (current_price / entry_price) - (fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Calculate exit fee
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                    
                    # Record fee impact
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    
                    position = 0
                    stop_loss_triggered = True
                    stop_loss_exits += 1
                
                elif take_profit and current_return > take_profit:  # R:R based take profit check
                    # Take profit triggered for short position
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "take_profit"
                    
                    # Calculate return (accounting for fees)
                    trade_return = 1 - (current_price / entry_price) - (fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Calculate exit fee
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                    
                    # Record fee impact
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    
                    position = 0
                    take_profit_triggered = True
                    take_profit_exits += 1
            
            # If stop loss was triggered, skip the rest of the logic for this iteration
            if stop_loss_triggered or take_profit_triggered:
                continue
            
            # # If a position was closed by stop loss in the previous bar, skip this bar too
            # if i > 1 and df_backtest.loc[df_backtest.index[i-1], 'exit_reason'] == "stop_loss":
            #     df_backtest.loc[df_backtest.index[i], 'position'] = 0  # Stay flat for this bar
            #     continue
            
            # Update position based on current state and probabilities
            if position == 0:  # Currently flat
                # Check for new position signals
                if long_prob > long_threshold:
                    # Enter long position when long probability exceeds threshold
                    position = 1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    trade_count += 1
                    
                    # Record entry fee
                    entry_fee = fee_rate
                    total_fees_paid += entry_fee
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = entry_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                
                elif short_prob > short_threshold:
                    # Enter short position when short probability exceeds threshold
                    position = -1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    trade_count += 1
                    
                    # Record entry fee
                    entry_fee = fee_rate
                    total_fees_paid += entry_fee
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = entry_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                
                # Handle quantum edge case: both probabilities high
                elif long_prob > hedge_threshold and short_prob > hedge_threshold:
                    # Enter hedged position if both signals are strong - this is unique to quantum approach
                    # Hedging means simultaneously holding long and short positions
                    position = 2  # Hedged
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    trade_count += 1
                    
                    # Record entry fee
                    entry_fee = fee_rate * 2
                    total_fees_paid += entry_fee
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = entry_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                
            elif position == 1:  # Currently long
                # Check for exit or hedge signals
                if long_prob < long_exit_threshold or short_prob > short_threshold:
                    # Exit long position if long probability drops or short probability rises
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "signal"

                    # Calculate return (accounting for fees)
                    # Return = (exit price / entry price) - 1 - round-trip fees
                    trade_return = (current_price / entry_price) - 1 - (fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                
                    # Calculate exit fee
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                
                    # Record fee impact for this exit
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    
                    # Check for immediate reversal to short
                    if short_prob > short_threshold:
                        # If short signal is strong, immediately enter short position
                        position = -1
                        entry_idx = i
                        entry_price = current_price
                        df_backtest.loc[df_backtest.index[i], 'position'] = -1
                        df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                        trade_count += 1
                    
                        # Add entry fee for new short position
                        reversal_fee = fee_rate
                        total_fees_paid += reversal_fee
                        df_backtest.loc[df_backtest.index[i], 'fee_impact'] += reversal_fee
                        df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    else:
                        position = 0
                
                elif long_prob > hedge_threshold and short_prob > hedge_threshold:
                    # Add hedge to long position if short signal strengthens
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    position = 2  # Hedged
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                
                    # Record fee for adding the short hedge
                    hedge_fee = fee_rate
                    total_fees_paid += hedge_fee
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = hedge_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                
                else:
                    # Stay in long position if no exit signals
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
            
            elif position == -1:  # Currently short
                # Check for exit or hedge signals
                if short_prob < short_exit_threshold or long_prob > long_threshold:
                    # Exit short position if short probability drops or long probability rises
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "signal"

                    # Calculate return (accounting for fees)
                    # Short return = 1 - (exit price / entry price) - round-trip fees
                    trade_return = 1 - (current_price / entry_price) - (fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                
                    # Calculate exit fee
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                
                    # Record fee impact for this exit
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    
                    # Check for immediate reversal to long
                    if long_prob > long_threshold:
                        # If long signal is strong, immediately enter long position
                        position = 1
                        entry_idx = i
                        entry_price = current_price
                        df_backtest.loc[df_backtest.index[i], 'position'] = 1
                        df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                        trade_count += 1
                        
                        # Add entry fee for new long position
                        reversal_fee = fee_rate
                        total_fees_paid += reversal_fee
                        df_backtest.loc[df_backtest.index[i], 'fee_impact'] += reversal_fee
                        df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    else:
                        position = 0
                
                elif long_prob > hedge_threshold and short_prob > hedge_threshold:
                    # Add hedge to short position if long signal strengthens
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    position = 2  # Hedged
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                
                    # Record fee for adding the long hedge
                    hedge_fee = fee_rate
                    total_fees_paid += hedge_fee
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = hedge_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                
                else:
                    # Stay in short position if no exit signals
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
            
            elif position == 2:  # Currently hedged
                # Check for removing hedge
                if long_prob < hedge_threshold and short_prob < hedge_threshold:
                    # Exit hedged position if both signals weaken
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "signal"
                                    
                    # Hedged positions typically have near-zero returns plus double fees
                    # We approximate this as the cost of the double hedging fees
                    trade_return = -(fee_rate * 4)  # Approximate cost of hedging
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                
                    # Calculate exit fee (double fee to exit both positions)
                    exit_fee = fee_rate * 2
                    total_fees_paid += exit_fee
                
                    # Record fee impact for this exit
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
                    
                    position = 0
                
                elif long_prob > threshold and short_prob < hedge_threshold:
                    # Convert hedge to pure long if long signal strengthens and short weakens
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = -1
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    position = 1
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
                
                    # Fee for exiting the short side
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
    
                    # Calculate return from closing the short side of the hedge
                    # Short return = 1 - (exit price / entry price) - exit fee
                    short_side_return = 1 - (current_price / entry_price) - (fee_rate * 2)
    
                    # Record partial close trade return (only for short side)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = short_side_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                
                elif short_prob > short_threshold and long_prob < hedge_threshold:
                    # Convert hedge to pure short if short signal strengthens and long weakens
                    df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = 1
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    position = -1
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
                
                    # Fee for exiting the long side
                    exit_fee = fee_rate
                    total_fees_paid += exit_fee
                    df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
                    df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
    
                    # Calculate return from closing the long side of the hedge
                    # Long return = (exit price / entry price) - 1 - exit fee
                    long_side_return = (current_price / entry_price) - 1 - (fee_rate * 2)
                    
                    # Record partial close trade return (only for long side)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = long_side_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                
                else:
                    # Stay hedged if signals remain conflicted
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
        
        # Close any open position at the end
        if position != 0:
            i = len(df_backtest) - 1
            current_price = df_backtest['close'].iloc[i]
            df_backtest.loc[df_backtest.index[i], 'closed_position_type'] = position
            df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
            df_backtest.loc[df_backtest.index[i], 'exit_reason'] = "end_of_data"
        
            if position == 1:  # Long position
                trade_return = (current_price / entry_price) - 1 - (fee_rate * 2)
            elif position == -1:  # Short position
                trade_return = 1 - (current_price / entry_price) - (fee_rate * 2)
            else:  # Hedged position
                trade_return = -(fee_rate * 4)  # Approximate cost of hedging
            
            exit_fee = fee_rate * abs(position)
            
            # Record final trade details
            total_fees_paid += exit_fee
            
            df_backtest.loc[df_backtest.index[i], 'fee_impact'] = exit_fee
            df_backtest.loc[df_backtest.index[i], 'total_fees_paid'] = total_fees_paid
            df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
            df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
        
        # Track completion of trading simulation
        data_context.add_processing_step("trading_simulation_complete", {
            "total_position_updates": len(df_backtest),
            "initial_position": 0,
            "final_position": position,
            "total_trades_entered": trade_count,
            "total_fees_paid": total_fees_paid,
            "stop_loss_exits": stop_loss_exits,
            "take_profit_exits": take_profit_exits
        })
        
        # Initialize strategy returns
        df_backtest['bar_return'] = 0.0
        
        # Apply realized returns from closed trades
        exit_mask = ~df_backtest['trade_return'].isna()
        df_backtest.loc[exit_mask, 'bar_return'] = df_backtest.loc[exit_mask, 'trade_return']
        
        # Calculate cumulative returns
        df_backtest['returns'] = df_backtest['close'].pct_change()
        df_backtest['cumulative_returns'] = (1 + df_backtest['returns']).cumprod() - 1
        df_backtest['strategy_cumulative'] = (1 + df_backtest['bar_return']).cumprod() - 1
    
        # Calculate fee drag ratio (cumulative fees / gross strategy value)
        if df_backtest['strategy_cumulative'].iloc[-1] > 0:
            gross_portfolio_value = 1 + df_backtest['strategy_cumulative'].iloc[-1]
            overall_fee_drag = total_fees_paid / gross_portfolio_value
        else:
            overall_fee_drag = 0.0
        
        # Calculate trading statistics
        trades = df_backtest[~df_backtest['trade_return'].isna()]

        # Count hedged positions (position=2) as both long and short 
        hedged_trades_count = len(trades[trades['closed_position_type'] == 2])
        
        # Record metrics for return calculation
        data_context.add_processing_step("return_calculation_complete", {
            "trades_closed": len(trades),
            "final_return": df_backtest['strategy_cumulative'].iloc[-1],
            "total_fee_drag": overall_fee_drag,
            "final_portfolio_value": 1 + df_backtest['strategy_cumulative'].iloc[-1] if len(df_backtest) > 0 else 1.0
        })
        
        # Calculate prediction accuracy if target exists
        target_accuracy = 0
        class_accuracy = {}
        
        # Calculate prediction accuracy if target exists
        if 'target' in df.columns:
            # For each step, check if position matches target
            for i in range(len(df_backtest)):
                # Get target and position, default to 0 if target is NaN
                target_val = df['target'].iloc[i] if not pd.isna(df['target'].iloc[i]) else 0
                position_val = df_backtest['position'].iloc[i]
                
                # Check if they match (only considering standard positions, not hedged)
                if position_val == target_val and position_val != 2:  # Ignore hedged positions (2)
                    df_backtest.loc[df_backtest.index[i], 'target_match'] = True
            
            # Calculate overall accuracy
            valid_samples = df_backtest[~df['target'].isna()]
            if len(valid_samples) > 0:
                target_accuracy = valid_samples['target_match'].mean()
                
                # Calculate class-specific accuracy
                class_accuracy = {}
                for target_value in [-1, 0, 1]:
                    target_class = valid_samples[df['target'] == target_value]
                    if len(target_class) > 0:
                        class_match = target_class['target_match'].mean()
                        class_accuracy[f'target_{target_value}_accuracy'] = class_match
                        class_accuracy[f'target_{target_value}_count'] = len(target_class)
                # Record target accuracy metrics
                data_context.add_processing_step("target_accuracy_calculated", {
                    "overall_accuracy": target_accuracy,
                    "class_accuracies": class_accuracy
                })
    
        # Analyze stop loss metrics
        if stop_loss_exits > 0:
            stop_loss_trades = trades[trades['exit_reason'] == 'stop_loss']
            avg_stop_loss_return = stop_loss_trades['trade_return'].mean() if len(stop_loss_trades) > 0 else 0
            self.logger.info(f"Stop loss was triggered {stop_loss_exits} times with average return: {avg_stop_loss_return:.2%}")
        else:
            self.logger.info("No stop loss exits occurred in this backtest")
                
        metrics = {
            'total_trades': trade_count,
            'long_trades': len(trades[trades['closed_position_type'] == 1]) + hedged_trades_count,
            'short_trades': len(trades[trades['closed_position_type'] == -1]) + hedged_trades_count,
            'hedged_trades': hedged_trades_count,
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
            'total_fees_paid': total_fees_paid,
            'total_fee_drag': overall_fee_drag,
            'final_return': df_backtest['strategy_cumulative'].iloc[-1],
            'target_match_accuracy': target_accuracy,
            'target_class_accuracy': class_accuracy,
            'stop_loss': stop_loss,
            'stop_loss_exits': stop_loss_exits,
            'stop_loss_exit_rate': stop_loss_exits / trade_count if trade_count > 0 else 0,
            'take_profit': take_profit,
            'take_profit_exits': take_profit_exits,
            'take_profit_exit_rate': take_profit_exits / trade_count if trade_count > 0 else 0,
            'risk_reward_ratio': risk_reward_ratio
        }
        
        # Add detailed stop loss metrics
        if stop_loss_exits > 0:
            stop_loss_trades = trades[trades['exit_reason'] == 'stop_loss']
            signal_trades = trades[trades['exit_reason'] == 'signal']
            
            metrics['stop_loss_avg_return'] = stop_loss_trades['trade_return'].mean() if len(stop_loss_trades) > 0 else 0
            metrics['signal_exit_avg_return'] = signal_trades['trade_return'].mean() if len(signal_trades) > 0 else 0
            
            # Compute max drawdown and drawdown statistics for trades that hit stop loss
            if len(stop_loss_trades) > 0:
                metrics['stop_loss_max_loss'] = stop_loss_trades['trade_return'].min()
                metrics['stop_loss_long_exits'] = len(stop_loss_trades[stop_loss_trades['closed_position_type'] == 1])
                metrics['stop_loss_short_exits'] = len(stop_loss_trades[stop_loss_trades['closed_position_type'] == -1])
        
    
        # Add detailed take profit metrics
        if take_profit_exits > 0:
            take_profit_trades = trades[trades['exit_reason'] == 'take_profit']
            
            metrics['take_profit_avg_return'] = take_profit_trades['trade_return'].mean() if len(take_profit_trades) > 0 else 0
            
            # Detailed stats for take profit exits
            metrics['take_profit_long_exits'] = len(take_profit_trades[take_profit_trades['closed_position_type'] == 1])
            metrics['take_profit_short_exits'] = len(take_profit_trades[take_profit_trades['closed_position_type'] == -1])
        
        # Log the take profit info
        self.logger.info(f"Take profit was triggered {take_profit_exits} times (R:R = {risk_reward_ratio})")
        
        # Log the accuracy
        self.logger.info(f"Position to target match accuracy: {target_accuracy:.2%}")
        for k, v in class_accuracy.items():
            if 'accuracy' in k:
                self.logger.info(f"{k}: {v:.2%}")
        
        # Final record of backtest completion with key metrics
        data_context.add_processing_step("backtest_complete", {
            "total_trades": metrics['total_trades'],
            "win_rate": metrics['win_rate'],
            "final_return": metrics['final_return'],
            "profit_factor": metrics['profit_factor'],
            "stop_loss_exits": stop_loss_exits,
            "stop_loss_pct": stop_loss,
            "take_profit_exits": take_profit_exits,
            "take_profit_pct": take_profit,
            "risk_reward_ratio": risk_reward_ratio
        })
        
        return df_backtest, metrics
        
    def run_multi_test(self):
        """
        Run backtest across multiple symbols and timeframes using the latest trained model
        
        This method systematically tests model performance across all configured symbols
        and timeframes, leveraging DataContext for proper data tracking and provenance.
        
        Returns:
            dict: Results dictionary organized by exchange, symbol, and timeframe
        """
        
        # Store symbols and timeframes in params
        symbols = self.params.get('data', 'symbols')
        timeframes = self.params.get('data', 'timeframes')
        exchanges = self.params.get('data', 'exchanges')
        
        self.logger.info(f"Running multi-symbol backtest for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Create a master DataContext for tracking the multi-test process
        master_context = DataContext(self.params)
        master_context.source = "multi_test"
        master_context.add_processing_step("multi_test_started", {
            "symbols": symbols,
            "timeframes": timeframes,
            "exchanges": exchanges,
            "timestamp": datetime.now().isoformat()
        })
        
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
                        # Log current test
                        self.logger.info(f"Testing {exchange} {symbol} {timeframe}")
                
                        # Update params for this specific symbol/timeframe for model loading
                        self.params.set([symbol], 'data', 'symbols')
                        self.params.set([timeframe], 'data', 'timeframes')
                        self.params.set([exchange], 'data', 'exchanges')
                        
                        # Run the individual backtest
                        #TODO: run_test() called by multitest should not save and plot test results for each symbol/timeframe/exchange 
                        backtest_results, metrics = self.run_test()
                
                        if backtest_results is False:
                            self.logger.warning(f"Backtest failed for {symbol} {timeframe}")
                            master_context.add_processing_step("test_failed", {
                                "exchange": exchange,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "reason": "Backtest execution failure"
                            })
                            continue
                    
                        # Record successful test in master context
                        master_context.add_processing_step("test_completed", {
                            "exchange": exchange,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "data_source": metrics.get('data_source', 'unknown'),
                            "final_return": metrics.get('final_return', 0),
                            "sharpe_ratio": metrics.get('sharpe_ratio', 0)
                        })
                    
                        # Store results in dictionary
                        symbol_results[timeframe] = {
                            'results': backtest_results,
                            'metrics': metrics,
                            'data_source': metrics.get('data_source', 'unknown')
                        }
                
                    except Exception as e:
                        self.logger.error(f"Error during backtest of {symbol} {timeframe}: {str(e)}")
                        import traceback
                        error_trace = traceback.format_exc()
                        self.logger.error(error_trace)
                    
                        # Record error in master context
                        master_context.add_processing_step("test_error", {
                            "exchange": exchange,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "error": str(e),
                            "traceback": error_trace
                        })
                        continue
                
                if symbol_results:
                    exchange_results[symbol] = symbol_results
                
            if exchange_results:
                all_results[exchange] = exchange_results
    
        # Record completion in master context
        successful_tests = sum(len(exchange_results[symbol]) 
                            for exchange in all_results 
                            for symbol in all_results[exchange])

        total_tests = len(exchanges) * len(symbols) * len(timeframes)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        master_context.add_processing_step("multi_test_complete", {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "successful_tests": successful_tests, 
            "total_tests": total_tests,
            "success_rate": success_rate
        })
        
        # Create a summary report if we have results
        if all_results:
            # Include master context processing history in summary
            summary_meta = {
                "processing_history": master_context.get_processing_history(),
                "success_rate": success_rate,
                "timestamp": timestamp
            }
            self.result_logger.create_multi_summary(all_results, timestamp, metadata=summary_meta)
        else:
            self.logger.warning("No successful backtests to summarize")
        
        return all_results
        
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
        
        self.logger.info(f"Running position sizing test for {exchange} {symbol} {timeframe}")
        
        try:
            # First try to load test set using DataContext
            data_context = DataContext.from_test_set(self.params, exchange, symbol, timeframe)
            
            if data_context is not None:
                self.logger.info(f"Using held-out test data for position sizing test")
                data_context.add_processing_step("load_test_set", {
                    "source": "test_set",
                    "test_type": "position_sizing"
                })
            else:
                # Fall back to processed data with a warning
                self.logger.warning(f"No test set found. Using processed data instead. "
                        f"This may lead to overoptimistic results due to possible data leakage.")
                
                data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
                
                if data_context is None:
                    self.logger.error(f"No processed data found for {symbol} {timeframe}")
                    return False
                    
                data_context.add_processing_step("load_processed_data", {
                    "source": "processed_data",
                    "warning": "Using full data instead of test set could lead to data leakage",
                    "test_type": "position_sizing"
                })
                
            # Validate data has the required columns
            if not data_context.validate(required_columns=['open', 'high', 'low', 'close', 'volume']):
                self.logger.error("Data validation failed - missing required columns")
                return False
            
            # Make sure the dataframe is sorted by timestamp
            if not data_context.df.index.is_monotonic_increasing:
                data_context.df = data_context.df.sort_index()
                data_context.add_processing_step("sort_index", {
                    "reason": "Ensure chronological order for backtesting"
                })
            
            # Load model through the model factory
            model_factory = ModelFactory(self.params)
            model = model_factory.create_model()
            
            if not model.load_model():
                self.logger.error("Failed to load model for position sizing test")
                data_context.add_processing_step("model_load_error", {
                    "error": "Failed to load model for position sizing test"
                })
                return False
        
            data_context.add_processing_step("model_loaded", {
                "model_type": self.params.get('model', 'type', default='auto'),
                "strategy": strategy
            })
            
            # Record position sizing parameters
            data_context.add_processing_step("position_sizing_params", {
                "no_trade_threshold": no_trade_threshold,
                "min_position_change": min_position_change
            })
        
            # Check if the model class has the position sizing method
            if not hasattr(model, 'run_backtest_with_position_sizing'):
                self.logger.error("Model does not support position sizing backtest")
                data_context.add_processing_step("position_sizing_error", {
                    "error": "Model does not support position sizing backtest"
                })
                return False
            
            # Run position sizing backtest, passing the DataContext
            # If the model doesn't accept DataContext directly, we can pass the DataFrame
            if 'data_context' in model.run_backtest_with_position_sizing.__code__.co_varnames:
                # Model accepts DataContext
                return_value = model.run_backtest_with_position_sizing(
                    data_context,
                    no_trade_threshold=no_trade_threshold,
                    min_position_change=min_position_change
                )
            else:
                # Model only accepts DataFrame
                return_value = model.run_backtest_with_position_sizing(
                    data_context.df,
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
                data_context.add_processing_step("backtest_error", {
                    "error": "Model returned invalid response format"
                })
                return return_value
                
            # Add data source information
            if isinstance(metrics, dict):
                metrics['data_source'] = data_context.source
                
                # Add processing history to metrics for full reproducibility
                metrics['processing_history'] = data_context.get_processing_history()
            
                # Generate execution ID for tracking
                execution_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}_{timeframe}_position_sizing"
                metrics['execution_id'] = execution_id
            
                # Add final processing step
                data_context.add_processing_step("position_sizing_complete", {
                    "execution_id": execution_id,
                    "final_return": metrics.get('total_return', 0),
                    "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                    "position_changes": metrics.get('num_position_changes', 0)
                })
            
            return results, metrics, fig
        
        except Exception as e:
            self.logger.error(f"Error during position sizing test: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
    def run_target_backtest(self):
        """
        Run a backtest using the actual targets rather than model predictions.
        
        This provides an "upper bound" of model performance - what would happen 
        if predictions were perfect. Useful for validating target quality and
        the trading strategy logic.
        
        Returns:
            tuple or bool: (backtest_results, metrics) if successful, False otherwise
        """
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        
        self.logger.info(f"Running target quality backtest for {exchange} {symbol} {timeframe}")
        
        try:
            # Create a DataContext for the target backtest
            data_context = DataContext.from_test_set(self.params, exchange, symbol, timeframe)
            
            if data_context is not None:
                self.logger.info(f"Using held-out test data for target backtest")
                data_context.add_processing_step("load_test_set", {
                    "source": "test_set",
                    "test_type": "target_backtest"
                })
            else:
                # Fall back to processed data with a warning
                self.logger.warning(f"No test set found. Using processed data instead. "
                        f"This may lead to overoptimistic results due to possible data leakage.")
                
                data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
                
                if data_context is None:
                    self.logger.error(f"No processed data found for {symbol} {timeframe}")
                    return False
                
                data_context.add_processing_step("load_processed_data", {
                    "source": "processed_data",
                    "warning": "Using full data instead of test set could lead to data leakage",
                    "test_type": "target_backtest"
                })
        
            # Make sure the dataframe is sorted by timestamp
            if not data_context.df.index.is_monotonic_increasing:
                data_context.df = data_context.df.sort_index()
                data_context.add_processing_step("sort_index", {
                    "reason": "Ensure chronological order for backtesting"
                })
            
            # Verify that target exists in the DataFrame
            if 'target' not in data_context.df.columns:
                self.logger.error("Missing 'target' column in data")
                data_context.add_processing_step("validation_failed", {
                    "reason": "Missing 'target' column in data"
                })
                return False
            
            df = data_context.df
            
            # Create "perfect predictions" from the actual targets
            # Convert from single target column (-1, 0, 1) to probability array format
            probabilities = np.zeros((len(df), 3))
            
            # For each row, set probability=1.0 for the correct class
            for i in range(len(df)):
                target_val = df['target'].iloc[i]
                if target_val == -1:  # Short
                    probabilities[i, 0] = 1.0
                elif target_val == 0:  # No trade
                    probabilities[i, 1] = 1.0
                elif target_val == 1:  # Long
                    probabilities[i, 2] = 1.0
        
            # Let's add a histogram of the probabilities we're using
            target_counts = df['target'].value_counts()
            data_context.add_processing_step("target_distribution", {
                "distribution": target_counts,
                "total_rows": len(df)
            })
            self.logger.info(f"Target distribution before running backtest: {target_counts.to_dict()}")
            
            # Record that we're using perfect predictions
            data_context.add_processing_step("perfect_predictions_generated", {
                "method": "target_to_probabilities",
                "description": "Converting targets to perfect probability predictions (p=1.0 for target class)"
            })
            
            # Run backtest with "perfect" predictions
            backtest_results, metrics = self._run_quantum_backtest(data_context, probabilities)
        
            # Add data source and label this as a target backtest
            metrics['data_source'] = data_context.source
            metrics['backtest_type'] = 'target_quality'
            metrics['strategy_type'] = 'quantum'
        
            # Add diagnostics to help identify the discrepancy between targets and trades
            # 1. Count transitions where strategy didn't follow perfect signals
            trade_mismatch = 0
            signal_ignored = 0
            expected_trades = 0
            signal_followed = 0
            
            for i in range(1, len(df)):
                # Get the target and actual position
                target = df['target'].iloc[i]
                position = backtest_results['position'].iloc[i]
            
                # Skip NaN targets
                if pd.isna(target):
                    continue
                
                # Check if we expect a trade here
                if target != 0:  # Should be in a position
                    expected_trades += 1
                    
                    if (target == 1 and position == 1) or (target == -1 and position == -1):
                        signal_followed += 1
                    elif position == 0:
                        signal_ignored += 1
                    else:
                        trade_mismatch += 1  # Position doesn't match target
            
            metrics['expected_trades'] = expected_trades
            metrics['signal_followed'] = signal_followed
            metrics['signal_ignored'] = signal_ignored
            metrics['trade_mismatch'] = trade_mismatch
            
            # Calculate trade efficiency
            if expected_trades > 0:
                metrics['target_efficiency'] = signal_followed / expected_trades
            else:
                metrics['target_efficiency'] = 0
                
            # More detailed diagnostics on position transitions
            if 'position' in backtest_results.columns:
                position_changes = (backtest_results['position'] != backtest_results['position'].shift(1)).sum()
                metrics['position_changes'] = position_changes
            
                # Check position stability - how often positions match targets
                # First create mask for valid targets (not NaN)
                valid_targets = ~df['target'].isna()
            
                # Then calculate matches only for valid targets
                target_position_match = (
                    (valid_targets & (df['target'] == 1) & (backtest_results['position'] == 1)) | 
                    (valid_targets & (df['target'] == -1) & (backtest_results['position'] == -1)) | 
                    (valid_targets & (df['target'] == 0) & (backtest_results['position'] == 0))
                )
            
                if valid_targets.sum() > 0:
                    metrics['target_position_match_rate'] = target_position_match.sum() / valid_targets.sum()
                else:
                    metrics['target_position_match_rate'] = 0
        
            # DIAGNOSTIC: Check position distribution in trades
            trades = backtest_results[~backtest_results['trade_return'].isna()]
            position_counts = trades['position'].value_counts().to_dict()
            self.logger.info(f"Target backtest position distribution: {position_counts}")
            
            # Check for unaccounted positions
            known_positions = [1, -1, 0, 2]  # Expected position values
            unknown_positions = [pos for pos in position_counts.keys() if pos not in known_positions]
            if unknown_positions:
                self.logger.warning(f"Found trades with unexpected position values: {unknown_positions}")
                for pos in unknown_positions:
                    metrics[f'unknown_pos_{pos}_trades'] = position_counts.get(pos, 0)
        
            # Add diagnostic metrics to data context
            data_context.add_processing_step("target_diagnostics", {
                "expected_trades": expected_trades,
                "signal_followed": signal_followed,
                "signal_ignored": signal_ignored,
                "trade_mismatch": trade_mismatch,
                "target_efficiency": metrics['target_efficiency'],
                "position_changes": metrics.get('position_changes', 0),
                "target_position_match_rate": metrics.get('target_position_match_rate', 0)
            })
        
            # Add processing history to metrics
            metrics['processing_history'] = data_context.get_processing_history()
            
            # Generate execution ID for tracking
            execution_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}_{timeframe}_target_backtest"
            metrics['execution_id'] = execution_id
            
            # Add final step to data context
            data_context.add_processing_step("target_backtest_complete", {
                "execution_id": execution_id,
                "total_trades": metrics['total_trades'],
                "win_rate": metrics['win_rate'],
                "final_return": metrics['final_return']
            })
            
            # DIAGNOSTIC: Log target distribution vs trade counts
            self.logger.info(f"Target distribution in data: {target_counts}")
            self.logger.info(f"Trade counts in backtest: long={metrics.get('long_trades', 0)}, " + 
                            f"short={metrics.get('short_trades', 0)}, no_trade=N/A")
            
            # Save and plot results
            self.result_logger.save_results(backtest_results, metrics, strategy_type="target_backtest")
            self.result_logger.plot_results(backtest_results, metrics, strategy_type="target_backtest")
            
            # Log some sample losing trades for inspection
            losing_trades = backtest_results[(backtest_results['trade_return'] < 0) & (~backtest_results['trade_return'].isna())]
            
            if not losing_trades.empty:
                # Sample a few losing trades for inspection
                sample_losing = losing_trades.head(5)
                
                self.logger.info("Sample of losing trades:")
                for idx, row in sample_losing.iterrows():
                    target = df.loc[idx, 'target']
                    pos = row['position']
                    entry = row['entry_price']
                    exit_p = row['exit_price']
                    ret = row['trade_return']
                    
                    self.logger.info(f"Target: {target}, Position: {pos}, Entry: {entry}, Exit: {exit_p}, Return: {ret:.4f}")
            else:
                self.logger.info("No losing trades found in target backtest - perfect performance")
            
            return backtest_results, metrics
            
        except Exception as e:
            self.logger.error(f"Error during target backtest: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(error_trace)
            
            # If we have a data context, record the error
            if 'data_context' in locals():
                data_context.add_processing_step("target_backtest_error", {
                    "error": str(e),
                    "traceback": error_trace
                })
            
            return False