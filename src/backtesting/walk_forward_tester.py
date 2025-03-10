#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Walk-Forward Testing implementation for cryptocurrency trading models.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from ..models.model_factory import ModelFactory

class WalkForwardTester:
    """
    Implements Walk-Forward Testing for trading strategies
    
    Walk-Forward Testing is an enhanced backtesting technique that:
    1. Uses expanding/rolling windows to mimic real-world model retraining
    2. Completely separates training and testing data
    3. Validates model robustness across different market conditions
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fee_rate = self.config.get('backtesting', {}).get('fee_rate', 0.0006)
        
        # Configuration for walk-forward testing
        self.wf_config = self.config.get('walk_forward', {})
        self.train_window = self.wf_config.get('train_window', 180)  # Training window in days
        self.test_window = self.wf_config.get('test_window', 30)     # Testing window in days
        self.step_size = self.wf_config.get('step_size', 30)         # Step size for rolling window in days
        self.min_train_samples = self.wf_config.get('min_train_samples', 1000)  # Minimum samples for training
    
    def run_test(self, exchange='binance', symbol='BTC/USDT', timeframe='1h'):
        """
        Run Walk-Forward Testing for a given symbol and timeframe
        
        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            Combined results dataframe and performance metrics
        """
        self.logger.info(f"Running Walk-Forward Test for {exchange} {symbol} {timeframe}")
        
        try:
            # 1. Load data
            symbol_safe = symbol.replace('/', '_')
            input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
            
            if not input_file.exists():
                self.logger.error(f"No processed data file found at {input_file}")
                return False
                
            df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
            
            # Create output directory
            output_dir = Path(f"data/walk_forward_results/{exchange}/{symbol_safe}/{timeframe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. Determine time windows
            total_days = (df.index[-1] - df.index[0]).days
            self.logger.info(f"Total data span: {total_days} days from {df.index[0]} to {df.index[-1]}")
            
            if total_days < (self.train_window + self.test_window):
                self.logger.error(f"Insufficient data for Walk-Forward Testing. Need at least {self.train_window + self.test_window} days.")
                return False
            
            # 3. Calculate walk-forward windows
            start_date = df.index[0]
            end_date = df.index[-1]
            
            # Initialize as datetime objects
            current_train_start = start_date
            
            windows = []
            
            while True:
                current_train_end = current_train_start + pd.Timedelta(days=self.train_window)
                current_test_start = current_train_end
                current_test_end = current_test_start + pd.Timedelta(days=self.test_window)
                
                # Break if we've reached the end of data
                if current_test_end > end_date:
                    break
                
                windows.append({
                    'train_start': current_train_start,
                    'train_end': current_train_end,
                    'test_start': current_test_start,
                    'test_end': current_test_end
                })
                
                # Move to next window
                current_train_start = current_train_start + pd.Timedelta(days=self.step_size)
            
            self.logger.info(f"Created {len(windows)} walk-forward windows")
            
            # 4. Run walk-forward analysis for each window
            results_all = []
            performance_metrics = []
            
            for i, window in enumerate(windows):
                window_results = self._run_window(df, window, exchange, symbol, timeframe, window_id=i+1)
                
                if window_results:
                    results, metrics = window_results
                    results_all.append(results)
                    
                    # Add window info to metrics and append
                    metrics['window_id'] = i + 1
                    metrics['train_start'] = window['train_start']
                    metrics['train_end'] = window['train_end']
                    metrics['test_start'] = window['test_start']
                    metrics['test_end'] = window['test_end']
                    performance_metrics.append(metrics)
            
            # 5. Combine results
            if not results_all:
                self.logger.error("No valid results from any window")
                return False
                
            # Combine results dataframes
            combined_results = pd.concat(results_all)
            
            # Create performance metrics dataframe
            performance_df = pd.DataFrame(performance_metrics)
            
            # 6. Save combined results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = output_dir / f"combined_results_{timestamp}.csv"
            combined_results.to_csv(results_file)
            
            metrics_file = output_dir / f"performance_metrics_{timestamp}.csv"
            performance_df.to_csv(metrics_file)
            
            # 7. Plot combined performance
            self._plot_walk_forward_performance(combined_results, performance_df, output_dir, symbol, timeframe)
            
            return combined_results, performance_df
                
        except Exception as e:
            self.logger.error(f"Error during Walk-Forward Testing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_window(self, df, window, exchange, symbol, timeframe, window_id=1):
        """Process a single walk-forward window"""
        self.logger.info(f"Processing window {window_id}: "
                f"Train: {window['train_start']} to {window['train_end']}, "
                f"Test: {window['test_start']} to {window['test_end']}")
        
        try:
            # 1. Split data into train and test sets
            train_mask = (df.index >= window['train_start']) & (df.index < window['train_end'])
            test_mask = (df.index >= window['test_start']) & (df.index < window['test_end'])
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            if len(train_df) < self.min_train_samples:
                self.logger.warning(f"Window {window_id}: Insufficient training samples ({len(train_df)} < {self.min_train_samples})")
                return None
                
            self.logger.info(f"Window {window_id}: Train set size = {len(train_df)}, Test set size = {len(test_df)}")
            
            # 2. Create model
            model_factory = ModelFactory(self.config)
            model = model_factory.create_model()
            
            # 3. Create target variable for training
            train_df['target'] = model.create_target(train_df)
            
            # 4. Train model on training set
            feature_cols = model.feature_cols
            X_train = train_df[feature_cols].values
            y_train = train_df['target'].values
            
            # Scale features
            X_train_scaled = model.scaler.fit_transform(X_train)
            
            # Build and train model
            self.logger.info(f"Window {window_id}: Training model...")
            model.build_model(X_train_scaled, y_train)
            
            # 5. Make predictions on test set
            self.logger.info(f"Window {window_id}: Making predictions...")
            probabilities = model.predict_probabilities(test_df)
            
            if probabilities is None:
                self.logger.error(f"Window {window_id}: Failed to get predictions")
                return None
                
            # 6. Run backtest on test data
            self.logger.info(f"Window {window_id}: Running backtest...")
            
            test_results, stats = self._run_backtest(test_df, probabilities)
            
            # 7. Add window information to results
            test_results['window_id'] = window_id
            test_results['train_period'] = f"{window['train_start']} to {window['train_end']}"
            test_results['test_period'] = f"{window['test_start']} to {window['test_end']}"
            
            # 8. Save window results
            output_dir = Path(f"data/walk_forward_results/{exchange}/{symbol.replace('/', '_')}/{timeframe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = output_dir / f"window_{window_id}_results.csv"
            test_results.to_csv(result_file)
            
            plot_file = output_dir / f"window_{window_id}_plot.png"
            self._plot_window_results(test_results, plot_file, symbol, timeframe, window_id)
            
            self.logger.info(f"Window {window_id}: Complete. Win rate: {stats['win_rate']:.2%}, Return: {stats['final_return']:.2%}")
            
            return test_results, stats
            
        except Exception as e:
            self.logger.error(f"Error processing window {window_id}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _run_backtest(self, df, probabilities, threshold=0.5, hedge_threshold=0.3):
        """Run backtest with quantum-inspired approach"""
        df_backtest = df.copy()
        
        # Initialize columns for tracking positions and returns
        df_backtest['short_prob'] = probabilities[:, 0]
        df_backtest['no_trade_prob'] = probabilities[:, 1]
        df_backtest['long_prob'] = probabilities[:, 2]
        df_backtest['position'] = 0  # -1=short, 0=flat, 1=long, 2=hedged
        df_backtest['entry_price'] = np.nan
        df_backtest['exit_price'] = np.nan
        df_backtest['trade_return'] = np.nan
        df_backtest['trade_duration'] = np.nan
        
        # Implement trading logic similar to your original backtest engine
        # This is a simplified version - you would include your full logic here
        position = 0
        entry_idx = 0
        entry_price = 0
        
        for i in range(1, len(df_backtest) - 1):
            current_price = df_backtest['close'].iloc[i]
            
            # Get current probabilities
            short_prob = probabilities[i, 0]
            no_trade_prob = probabilities[i, 1]
            long_prob = probabilities[i, 2]
            
            # Update position based on current state and probabilities
            # (Insert your trading logic here - this is a placeholder)
            if position == 0:  # Currently flat
                if long_prob > threshold:
                    position = 1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                elif short_prob > threshold:
                    position = -1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
            elif position == 1:  # Long position
                if long_prob < 0.3 or short_prob > threshold:
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Calculate return (accounting for fees)
                    trade_return = (current_price / entry_price) - 1 - (self.fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    position = 0
            elif position == -1:  # Short position
                if short_prob < 0.3 or long_prob > threshold:
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Calculate return (accounting for fees)
                    trade_return = 1 - (current_price / entry_price) - (self.fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    position = 0
        
        # Close any open position at the end
        if position != 0:
            i = len(df_backtest) - 1
            current_price = df_backtest['close'].iloc[i]
            df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
            
            if position == 1:  # Long position
                trade_return = (current_price / entry_price) - 1 - (self.fee_rate * 2)
            else:  # Short position
                trade_return = 1 - (current_price / entry_price) - (self.fee_rate * 2)
            
            df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
            df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
        
        # Calculate strategy returns
        df_backtest['returns'] = df_backtest['close'].pct_change()
        
        # For simplicity, we'll just use position values (-1, 0, 1) directly
        df_backtest['strategy_returns'] = 0.0
        
        long_mask = df_backtest['position'].shift(1) == 1
        short_mask = df_backtest['position'].shift(1) == -1
        
        df_backtest.loc[long_mask, 'strategy_returns'] = df_backtest.loc[long_mask, 'returns']
        df_backtest.loc[short_mask, 'strategy_returns'] = -df_backtest.loc[short_mask, 'returns']
        
        # Calculate cumulative returns
        df_backtest['cumulative_returns'] = (1 + df_backtest['returns']).cumprod() - 1
        df_backtest['strategy_cumulative'] = (1 + df_backtest['strategy_returns']).cumprod() - 1
        
        # Calculate trading statistics
        trades = df_backtest[~df_backtest['trade_return'].isna()]
        
        stats = {
            'total_trades': len(trades),
            'long_trades': (trades['position'] == 1).sum(),
            'short_trades': (trades['position'] == -1).sum(),
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
                            if 'strategy_cumulative' in df_backtest.columns else 0
        }
        
        return df_backtest, stats
    
    def _plot_window_results(self, results, output_path, symbol, timeframe, window_id):
        """Plot backtest results for a single window"""
        try:
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Upper plot: Price with positions
            axes[0].plot(results.index, results['close'], label=f'{symbol}', alpha=0.7)
            
            # Mark positions
            long_entries = results[results['position'].diff() == 1]
            short_entries = results[results['position'].diff() == -1]
            exits = results[~results['exit_price'].isna()]
            
            # Plot position markers
            axes[0].scatter(long_entries.index, long_entries['close'], marker='^', color='green', 
                    s=100, label='Long Entry')
            axes[0].scatter(short_entries.index, short_entries['close'], marker='v', color='red', 
                    s=100, label='Short Entry')
            axes[0].scatter(exits.index, exits['close'], marker='x', color='black', 
                    s=80, label='Exit')
            
            # Add title
            axes[0].set_title(f'{symbol} {timeframe} - Window {window_id} Results')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Lower plot: Cumulative returns
            if 'cumulative_returns' in results.columns and 'strategy_cumulative' in results.columns:
                axes[1].plot(results.index, results['cumulative_returns'], label='Buy & Hold', color='blue')
                axes[1].plot(results.index, results['strategy_cumulative'], label='Strategy', color='purple')
                axes[1].set_title('Strategy Performance')
                axes[1].set_xlabel('Date')
                axes[1].set_ylabel('Cumulative Returns')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path)
            plt.close(fig)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting window results: {str(e)}")
            return False
    
    def _plot_walk_forward_performance(self, combined_results, performance_df, output_dir, symbol, timeframe):
        """Create summary plots for walk-forward performance"""
        try:
            # Plot 1: Combined equity curve
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Group by window and plot each window's equity curve
            for window_id, window_data in combined_results.groupby('window_id'):
                if 'strategy_cumulative' in window_data.columns:
                    ax.plot(window_data.index, window_data['strategy_cumulative'], 
                            label=f'Window {window_id}')
            
            ax.set_title(f'{symbol} {timeframe} - Walk-Forward Equity Curves by Window')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            equity_plot_path = output_dir / "walk_forward_equity_curves.png"
            plt.savefig(equity_plot_path)
            plt.close(fig)
            
            # Plot 2: Performance metrics across windows
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            # Win rate across windows
            axes[0].bar(performance_df['window_id'], performance_df['win_rate'])
            axes[0].set_title('Win Rate by Window')
            axes[0].set_xlabel('Window ID')
            axes[0].set_ylabel('Win Rate')
            axes[0].set_ylim(0, 1)
            axes[0].grid(True, alpha=0.3)
            
            # Return across windows
            axes[1].bar(performance_df['window_id'], performance_df['final_return'])
            axes[1].set_title('Return by Window')
            axes[1].set_xlabel('Window ID')
            axes[1].set_ylabel('Return')
            axes[1].grid(True, alpha=0.3)
            
            # Profit factor across windows
            axes[2].bar(performance_df['window_id'], performance_df['profit_factor'].clip(upper=5))
            axes[2].set_title('Profit Factor by Window (capped at 5)')
            axes[2].set_xlabel('Window ID')
            axes[2].set_ylabel('Profit Factor')
            axes[2].grid(True, alpha=0.3)
            
            # Number of trades across windows
            axes[3].bar(performance_df['window_id'], performance_df['total_trades'])
            axes[3].set_title('Number of Trades by Window')
            axes[3].set_xlabel('Window ID')
            axes[3].set_ylabel('Number of Trades')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            metrics_plot_path = output_dir / "walk_forward_performance_metrics.png"
            plt.savefig(metrics_plot_path)
            plt.close(fig)
            
            # Plot 3: Combined equity curve with buy & hold comparison
            if 'cumulative_returns' in combined_results.columns and 'strategy_cumulative' in combined_results.columns:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Create composite equity curves
                # Group by date and calculate average cumulative return
                combined_results_grouped = combined_results.groupby(level=0).mean()
                
                ax.plot(combined_results_grouped.index, combined_results_grouped['cumulative_returns'], 
                        label='Buy & Hold', color='blue')
                ax.plot(combined_results_grouped.index, combined_results_grouped['strategy_cumulative'], 
                        label='Strategy', color='purple')
                
                ax.set_title(f'{symbol} {timeframe} - Walk-Forward Performance vs Buy & Hold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Return')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                comparison_plot_path = output_dir / "walk_forward_vs_buy_hold.png"
                plt.savefig(comparison_plot_path)
                plt.close(fig)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting walk-forward performance: {str(e)}")
            return False
    
    def run_multi_test(self, symbols, timeframes, exchange='binance'):
        """
        Run walk-forward tests across multiple symbols and timeframes
        
        Args:
            symbols (list): List of trading pair symbols
            timeframes (list): List of timeframes to test
            exchange (str): Exchange name
            
        Returns:
            dict: Dictionary of results for all symbols and timeframes
        """
        self.logger.info(f"Running multi-symbol walk-forward test for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Dictionary to store all results
        all_results = {}
        
        # Process each symbol
        for symbol in symbols:
            symbol_results = {}
            
            # Process each timeframe for this symbol
            for timeframe in timeframes:
                self.logger.info(f"Running walk-forward test for {symbol} {timeframe}")
                
                try:
                    # Run walk-forward test for this symbol/timeframe
                    results, performance = self.run_test(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    
                    # Store results if successful
                    if results is not False:
                        symbol_results[timeframe] = {
                            'results': results,
                            'performance': performance
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error in walk-forward test for {symbol} {timeframe}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
            
            # Store all timeframe results for this symbol
            if symbol_results:
                all_results[symbol] = symbol_results
        
        # Create a summary report comparing performance across symbols and timeframes
        if all_results:
            self._create_multi_summary(all_results, symbols, timeframes, exchange)
            self.logger.info(f"Multi-symbol walk-forward test complete for {len(all_results)} symbols")
        else:
            self.logger.warning("No valid results were generated in multi-symbol walk-forward test")
        
        return all_results
    
    def _create_multi_summary(self, all_results, symbols, timeframes, exchange):
        """
        Create summary report for multi-symbol walk-forward tests
        
        Args:
            all_results (dict): Results dictionary
            symbols (list): List of symbols
            timeframes (list): List of timeframes
            exchange (str): Exchange name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Collect summary metrics from all tests
            summary_rows = []
            
            for symbol in symbols:
                if symbol not in all_results:
                    continue
                    
                for timeframe in timeframes:
                    if timeframe not in all_results[symbol]:
                        continue
                    
                    # Get performance metrics
                    performance = all_results[symbol][timeframe]['performance']
                    
                    # For each window in the walk-forward test
                    for _, row in performance.iterrows():
                        summary_rows.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'window_id': row['window_id'],
                            'train_start': row['train_start'],
                            'train_end': row['train_end'],
                            'test_start': row['test_start'],
                            'test_end': row['test_end'],
                            'win_rate': row['win_rate'],
                            'profit_factor': row['profit_factor'],
                            'final_return': row['final_return'],
                            'total_trades': row['total_trades']
                        })
            
            if not summary_rows:
                self.logger.warning("No summary data available")
                return False
                
            # Convert to DataFrame
            summary_df = pd.DataFrame(summary_rows)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f"data/walk_forward_results/{exchange}/summary")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_file = output_dir / f"multi_symbol_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            
            # Create visualizations
            self._plot_multi_symbol_summary(summary_df, output_dir, timestamp)
            
            self.logger.info(f"Multi-symbol summary saved to {summary_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating multi-symbol summary: {str(e)}")
            return False