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

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, config):
        """
        Initialize the backtest engine with configuration
        
        Args:
            config (dict): Configuration dictionary containing parameters for backtesting,
                            such as fee rates, thresholds, and other settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Extract fee rate from config or use default value
        self.fee_rate = self.config.get('backtesting', {}).get('fee_rate', 0.0006)
        
    def run_test(self, exchange='binance', symbol='BTC/USDT', timeframe='1m'):
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
        
        Args:
            exchange (str): Exchange name (e.g., 'binance')
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Data timeframe (e.g., '1h')
            
        Returns:
            tuple or bool: (backtest_results, stats) if successful, False otherwise
        """
        self.logger.info(f"Running backtest for {exchange} {symbol} {timeframe}")
        
        try:
            # Check for test data first
            symbol_safe = symbol.replace('/', '_')
            test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            # If test data exists, use it (preferred for proper evaluation)
            if test_file.exists():
                self.logger.info(f"Using held-out test data from {test_file}")
                df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                data_source = "test_set"  # Track the data source
            else:
                # Fall back to processed data with a warning
                self.logger.warning(f"No test set found. Using processed data instead. "
                        f"This may lead to overoptimistic results due to possible data leakage.")
                
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                
                if not input_file.exists():
                    self.logger.error(f"No processed data file found at {input_file}")
                    return False
                    
                df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                data_source = "full_data"  # Track the data source
            
            # Load model through the model factory
            model_factory = ModelFactory(self.config)
            model = model_factory.create_model()
            
            if not model.load_model(exchange, symbol, timeframe):
                self.logger.error("Failed to load model for backtesting")
                return False
            
            # Get predictions from the model
            probabilities = model.predict_probabilities(df)
            
            if probabilities is None:
                self.logger.error("Failed to get predictions for backtesting")
                return False
            
            # Run backtest with quantum-inspired approach
            backtest_results, stats = self._run_quantum_backtest(df, probabilities)
            
            # Add data source information for transparent reporting
            is_test_data = test_file.exists()
            
            # Save results with data source indicator for transparency
            self._save_backtest_results(backtest_results, stats, exchange, symbol, timeframe, data_source)
            
            # Plot results with data source indicator
            self._plot_backtest_results(backtest_results, exchange, symbol, timeframe, data_source)
            
            return backtest_results, stats
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_quantum_backtest(self, df, probabilities, threshold=0.5, hedge_threshold=0.3):
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
            threshold (float): Minimum probability to enter a position (default: 0.5)
            hedge_threshold (float): Threshold for considering hedging (default: 0.3)
            
        Returns:
            tuple: (df_backtest, stats) - DataFrame with backtest results and performance statistics
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
                if long_prob < 0.3 or short_prob > threshold:
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
                if short_prob < 0.3 or long_prob > threshold:
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
        
        stats = {
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
        
        return df_backtest, stats
    
    def _save_backtest_results(self, results, stats, exchange, symbol, timeframe, data_source="full_data"):
        """Save backtest results to CSV and stats to JSON"""
        try:
            # Create output directory
            symbol_safe = symbol.replace('/', '_')
            output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results with data source indicator
            results_file = output_dir / f"{timeframe}_{data_source}_{timestamp}.csv"
            results.to_csv(results_file)
            
            # Save stats with data source indicator
            stats_file = output_dir / f"{timeframe}_{data_source}_{timestamp}_stats.csv"
            stats_df = pd.DataFrame([stats])
            
            # Add data source information to stats
            stats_df['data_source'] = data_source
            stats_df.to_csv(stats_file, index=False)
            
            self.logger.info(f"Backtest results saved to {results_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")
            return False
    
    def _plot_backtest_results(self, results, exchange, symbol, timeframe, data_source="full_data"):
        """Plot backtest results"""
        try:
            symbol_safe = symbol.replace('/', '_')
            output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Add data source indicator to title
            title_prefix = f"TEST SET - " if data_source == "test_set" else ""
            
            # Upper plot: Price with positions
            axes[0].plot(results.index, results['close'], label=f'{symbol}', alpha=0.7)
            
            # Mark positions
            long_entries = results[results['position'].diff() == 1]
            short_entries = results[results['position'].diff() == -1]
            hedged_entries = results[results['position'].diff() == 2]
            exits = results[~results['exit_price'].isna()]
            
            # Plot position markers
            axes[0].scatter(long_entries.index, long_entries['close'], marker='^', color='green', 
                    s=100, label='Long Entry')
            axes[0].scatter(short_entries.index, short_entries['close'], marker='v', color='red', 
                    s=100, label='Short Entry')
            axes[0].scatter(hedged_entries.index, hedged_entries['close'], marker='s', color='purple', 
                    s=100, label='Hedged Position')
            axes[0].scatter(exits.index, exits['close'], marker='x', color='black', 
                    s=80, label='Exit')
            
            # Add probability shading for clarity (every 20th point to avoid clutter)
            for i in range(0, len(results), 20):
                if results['long_prob'].iloc[i] > 0.3:
                    axes[0].axvspan(results.index[i], results.index[min(i+1, len(results)-1)], 
                                  alpha=results['long_prob'].iloc[i] * 0.3, color='green', lw=0)
                if results['short_prob'].iloc[i] > 0.3:
                    axes[0].axvspan(results.index[i], results.index[min(i+1, len(results)-1)], 
                                  alpha=results['short_prob'].iloc[i] * 0.3, color='red', lw=0)
            
            axes[0].set_title(f'{title_prefix}{symbol} {timeframe} Price with Trading Signals')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Lower plot: Cumulative returns
            axes[1].plot(results.index, results['cumulative_returns'], label='Buy & Hold', color='blue')
            axes[1].plot(results.index, results['strategy_cumulative'], label='Quantum Strategy', color='purple')
            axes[1].set_title(f'{title_prefix}Strategy Performance')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Cumulative Returns')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plot_file = output_dir / f"{timeframe}_{data_source}_{timestamp}_plot.png"
            plt.savefig(plot_file)
            plt.close(fig)
            
            self.logger.info(f"Backtest plot saved to {plot_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")
            return False
        
    def run_multi_test(self, symbols, timeframes, exchange='binance'):
        """Run backtest across multiple symbols and timeframes"""
        self.logger.info(f"Running multi-symbol backtest for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Load the multi-symbol model
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model()
        
        # Generate model name
        model_name = model._generate_multi_model_name(symbols, timeframes)
        
        # Try to load the model
        model_dir = Path("models")
        scaler_path = model_dir / f"{exchange}_{model_name}_scaler.pkl"
        trace_path = model_dir / f"{exchange}_{model_name}_trace.netcdf"
        
        if not scaler_path.exists() or not trace_path.exists():
            self.logger.error(f"Multi-symbol model files not found. Please train the model first.")
            return False
        
        # Load the model
        with open(scaler_path, 'rb') as f:
            model.scaler = pickle.load(f)
        
        model.trace = az.from_netcdf(trace_path)
        
        # Define a dictionary to store results
        all_results = {}
        
        # Run backtest for each symbol/timeframe
        for symbol in symbols:
            symbol_results = {}
            for timeframe in timeframes:
                try:
                    # Check for test data first
                    symbol_safe = symbol.replace('/', '_')
                    test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
                    
                    # If test data exists, use it (preferred for proper evaluation)
                    if test_file.exists():
                        self.logger.info(f"Using held-out test data for {symbol} {timeframe}")
                        df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                        data_source = "test_set"
                    else:
                        # Fall back to processed data with a warning
                        self.logger.warning(f"No test set found for {symbol} {timeframe}. Using processed data instead.")
                        
                        input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                        
                        if not input_file.exists():
                            self.logger.warning(f"No processed data file found at {input_file}")
                            continue
                            
                        df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                        data_source = "full_data"

                    # Get predictions
                    probabilities = model.predict_probabilities(df)
                    
                    if probabilities is None:
                        self.logger.error(f"Failed to get predictions for {symbol} {timeframe}")
                        continue
                    
                    # Run backtest
                    backtest_results, stats = self._run_quantum_backtest(df, probabilities)
                    
                    # Add to results
                    symbol_results[timeframe] = {
                        'results': backtest_results,
                        'stats': stats,
                        'data_source': data_source
                    }
                    
                    # Save individual backtest
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Add model name and data source to output for identification
                    output_file = output_dir / f"{timeframe}_{model_name}_{data_source}_{timestamp}.csv"
                    backtest_results.to_csv(output_file)
                    
                    # Save stats
                    stats_file = output_dir / f"{timeframe}_{model_name}_{data_source}_{timestamp}_stats.csv"
                    stats_df = pd.DataFrame([stats])
                    stats_df['data_source'] = data_source
                    stats_df.to_csv(stats_file, index=False)
                    
                    self.logger.info(f"Saved backtest results for {symbol} {timeframe} to {output_file}")
                    
                    # Plot results
                    self._plot_backtest_results(backtest_results, exchange, symbol, 
                            f"{timeframe}_{model_name}", data_source)
                    
                except Exception as e:
                    self.logger.error(f"Error during backtest of {symbol} {timeframe}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
            
            if symbol_results:
                all_results[symbol] = symbol_results
        
        # Create a summary report
        self._create_multi_summary(all_results, symbols, timeframes, exchange, model_name)
        
        return all_results

    def _create_multi_summary(self, all_results, symbols, timeframes, exchange, model_name):
        """Create a summary report for multi-symbol backtest"""
        try:
            # Create a DataFrame for summary metrics
            metrics = ['win_rate', 'profit_factor', 'final_return', 'total_trades', 'data_source']
            summary = []
            
            for symbol in symbols:
                if symbol not in all_results:
                    continue
                    
                for timeframe in timeframes:
                    if timeframe not in all_results[symbol]:
                        continue
                    
                    result_info = all_results[symbol][timeframe]
                    stats = result_info['stats']
                    data_source = result_info['data_source']
                    
                    summary.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'win_rate': stats.get('win_rate', 0),
                        'profit_factor': stats.get('profit_factor', 0),
                        'final_return': stats.get('final_return', 0),
                        'total_trades': stats.get('total_trades', 0),
                        'data_source': data_source
                    })
            
            if not summary:
                self.logger.warning("No summary data available")
                return False
                
            summary_df = pd.DataFrame(summary)
            
            # Save summary
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f"data/backtest_results/{exchange}/summary")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_file = output_dir / f"multi_summary_{model_name}_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            
            # Create summary visualizations
            self._plot_multi_summary(summary_df, output_dir, model_name, timestamp)
            
            self.logger.info(f"Saved summary report to {summary_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {str(e)}")
            return False
    
    def _plot_multi_summary(self, summary_df, output_dir, model_name, timestamp):
        """Create summary visualization for multi-symbol results"""
        try:
            # Plot 1: Win rates by symbol and timeframe
            plt.figure(figsize=(12, 6))
            
            # Pivot data for plotting
            pivot_win_rate = summary_df.pivot(index='symbol', columns='timeframe', values='win_rate')
            
            # Create heatmap
            cmap = cm.get_cmap('RdYlGn')
            ax = plt.pcolor(pivot_win_rate, cmap=cmap, vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(ax)
            cbar.set_label('Win Rate')
            
            # Add labels
            plt.title(f'Win Rates by Symbol and Timeframe')
            plt.xticks(np.arange(len(pivot_win_rate.columns)) + 0.5, pivot_win_rate.columns)
            plt.yticks(np.arange(len(pivot_win_rate.index)) + 0.5, pivot_win_rate.index)
            
            # Add text annotations
            for i in range(len(pivot_win_rate.index)):
                for j in range(len(pivot_win_rate.columns)):
                    value = pivot_win_rate.iloc[i, j]
                    if not np.isnan(value):
                        plt.text(j + 0.5, i + 0.5, f'{value:.2f}',
                                 ha='center', va='center',
                                 color='white' if value < 0.5 else 'black')
            
            plt.tight_layout()
            win_rate_file = output_dir / f"win_rate_heatmap_{model_name}_{timestamp}.png"
            plt.savefig(win_rate_file)
            plt.close()
            
            # Plot 2: Returns by symbol and timeframe
            plt.figure(figsize=(10, 6))
            
            # Group by symbol and calculate mean return
            symbol_returns = summary_df.groupby('symbol')['final_return'].mean().sort_values(ascending=False)
            
            # Create bar chart
            plt.bar(symbol_returns.index, symbol_returns.values)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.title('Average Return by Symbol')
            plt.xlabel('Symbol')
            plt.ylabel('Return')
            plt.xticks(rotation=45)
            
            # Add text annotations
            for i, v in enumerate(symbol_returns):
                plt.text(i, v + 0.01, f'{v:.2%}',
                        ha='center', va='bottom',
                        color='green' if v > 0 else 'red')
            
            plt.tight_layout()
            returns_file = output_dir / f"returns_by_symbol_{model_name}_{timestamp}.png"
            plt.savefig(returns_file)
            plt.close()
            
            # Plot 3: Test vs. Full data comparison
            if 'data_source' in summary_df.columns and summary_df['data_source'].nunique() > 1:
                plt.figure(figsize=(12, 6))
                
                # Group by data source and calculate metrics
                source_metrics = summary_df.groupby('data_source').agg({
                    'win_rate': 'mean',
                    'final_return': 'mean',
                    'profit_factor': lambda x: x.replace([np.inf, -np.inf], np.nan).mean()
                })
                
                # Create grouped bar chart
                metrics_to_plot = ['win_rate', 'final_return']
                x = np.arange(len(metrics_to_plot))
                width = 0.35
                
                plt.bar(x - width/2, 
                      [source_metrics.loc['test_set', 'win_rate'], source_metrics.loc['test_set', 'final_return']], 
                      width, label='Test Set')
                plt.bar(x + width/2, 
                      [source_metrics.loc['full_data', 'win_rate'], source_metrics.loc['full_data', 'final_return']], 
                      width, label='Full Data')
                
                plt.title('Performance Comparison: Test Set vs Full Data')
                plt.xticks(x, ['Win Rate', 'Return'])
                plt.ylabel('Value')
                plt.legend()
                
                plt.tight_layout()
                compare_file = output_dir / f"test_vs_full_comparison_{model_name}_{timestamp}.png"
                plt.savefig(compare_file)
                plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating multi-summary plots: {str(e)}")
            # Continue execution even if plotting fails
            return False

    def run_test_on_test_set(self, exchange='binance', symbol='BTC/USDT', timeframe='1h'):
        """
        Run backtest strictly on test set data to ensure unbiased evaluation
        """
        self.logger.info(f"Running backtest on test set for {exchange} {symbol} {timeframe}")
        
        try:
            # 1. Load the test data specifically
            symbol_safe = symbol.replace('/', '_')
            test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            if not test_file.exists():
                self.logger.error(f"Test data file not found at {test_file}. Please run training first.")
                return False
                
            test_df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
            self.logger.info(f"Loaded test set with {len(test_df)} samples")
            
            # 2. Load model
            model_factory = ModelFactory(self.config)
            model = model_factory.create_model()
            
            if not model.load_model(exchange, symbol, timeframe):
                self.logger.error("Failed to load model for backtesting")
                return False
            
            # 3. Run predictions using the trained model
            probabilities = model.predict_probabilities(test_df)
            
            if probabilities is None:
                self.logger.error("Failed to get predictions for backtesting")
                return False
            
            # 4. Run backtest with quantum-inspired approach on test data only
            backtest_results, stats = self._run_quantum_backtest(test_df, probabilities)
            
            # 5. Save results with clear indication that these are test set results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._save_backtest_results(backtest_results, stats, exchange, symbol, timeframe, "test_set")
            
            # 6. Plot results 
            self._plot_backtest_results(backtest_results, exchange, symbol, timeframe, "test_set")
            
            # 7. Return results
            return backtest_results, stats
            
        except Exception as e:
            self.logger.error(f"Error during test set backtesting: {str(e)}")
            return False
        
    def run_position_sizing_test(self, exchange='binance', symbol='BTC/USDT', timeframe='1h', 
                                    no_trade_threshold=0.96, min_position_change=0.05):
        """
        Run a backtest with quantum position sizing instead of traditional binary signals.
        
        This method:
        1. Loads the appropriate test set or falls back to processed data
        2. Gets probability predictions from the model
        3. Applies quantum-inspired position sizing 
        4. Evaluates and visualizes performance
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            no_trade_threshold (float): Threshold for no_trade probability to ignore signals
            min_position_change (float): Minimum position change to avoid fee churn
                
        Returns:
            tuple or bool: (backtest_results, metrics, fig) if successful, False otherwise
        """
        self.logger.info(f"Running position sizing test for {exchange} {symbol} {timeframe}")
        
        try:
            # Check for test data first
            symbol_safe = symbol.replace('/', '_')
            test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            # If test data exists, use it (preferred for proper evaluation)
            if test_file.exists():
                self.logger.info(f"Using held-out test data from {test_file}")
                df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                data_source = "test_set"  # Track the data source
            else:
                # Fall back to processed data with a warning
                self.logger.warning(f"No test set found. Using processed data instead. "
                        f"This may lead to overoptimistic results due to possible data leakage.")
                
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                
                if not input_file.exists():
                    self.logger.error(f"No processed data file found at {input_file}")
                    return False
                    
                df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                data_source = "full_data"  # Track the data source
            
            # Load model through the model factory
            from ..models.model_factory import ModelFactory
            model_factory = ModelFactory(self.config)
            model = model_factory.create_model()
            
            if not model.load_model(exchange, symbol, timeframe):
                self.logger.error("Failed to load model for backtesting")
                return False
            
            # Run backtest with position sizing (delegates to the model's method)
            # Check if the model class has the position sizing method
            if hasattr(model, 'run_backtest_with_position_sizing'):
                results, metrics, fig = model.run_backtest_with_position_sizing(
                    df, 
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    no_trade_threshold=no_trade_threshold,
                    min_position_change=min_position_change
                )
                
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
            self._create_position_sizing_summary(all_results, symbols, timeframes, exchange)
            self.logger.info(f"Multi-symbol position sizing tests complete for {len(all_results)} symbols")
        else:
            self.logger.warning("No valid results were generated in multi-symbol position sizing tests")
        
        return all_results


    def _create_position_sizing_summary(self, all_results, symbols, timeframes, exchange):
        """
        Create summary report for multi-symbol position sizing tests
        
        Args:
            all_results (dict): Results dictionary
            symbols (list): List of symbols
            timeframes (list): List of timeframes
            exchange (str): Exchange name
                
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
            output_dir = Path(f"data/backtest_results/{exchange}/summary")
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