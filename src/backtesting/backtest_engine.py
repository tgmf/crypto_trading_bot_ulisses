#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting engine for trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from models.model_factory import ModelFactory

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fee_rate = self.config.get('backtesting', {}).get('fee_rate', 0.0006)
        
    def run_backtest(self, exchange='binance', symbol='BTC/USD', timeframe='1h'):
        """Run backtest for a given model and dataset"""
        self.logger.info(f"Running backtest for {exchange} {symbol} {timeframe}")
        
        try:
            # Load processed data
            symbol_safe = symbol.replace('/', '_')
            input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
            
            if not input_file.exists():
                self.logger.error(f"No processed data file found at {input_file}")
                return False
                
            df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
            
            # Load model
            model_factory = ModelFactory(self.config)
            model = model_factory.create_model()
            
            if not model.load_model(exchange, symbol, timeframe):
                self.logger.error("Failed to load model for backtesting")
                return False
            
            # Get predictions
            probabilities = model.predict_probabilities(df)
            
            if probabilities is None:
                self.logger.error("Failed to get predictions for backtesting")
                return False
            
            # Run backtest with quantum-inspired approach
            backtest_results, stats = self._run_quantum_backtest(df, probabilities)
            
            # Save results
            self._save_backtest_results(backtest_results, stats, exchange, symbol, timeframe)
            
            # Plot results
            self._plot_backtest_results(backtest_results, exchange, symbol, timeframe)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            return False
    
    def _run_quantum_backtest(self, df, probabilities, threshold=0.5, hedge_threshold=0.3):
        """
        Run backtest with quantum-inspired trading approach
        
        Args:
            df: Price data DataFrame
            probabilities: Probability array [P(short), P(no_trade), P(long)]
            threshold: Minimum probability to enter a position
            hedge_threshold: Threshold for considering hedging
        """
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
        
        # Simulate trading
        position = 0  # Start with no position
        entry_idx = 0
        entry_price = 0
        
        for i in range(1, len(df_backtest) - 1):
            current_price = df_backtest['close'].iloc[i]
            
            # Get current probabilities
            short_prob = probabilities[i, 0]
            no_trade_prob = probabilities[i, 1]
            long_prob = probabilities[i, 2]
            
            # Update position based on current state and probabilities
            if position == 0:  # Currently flat
                # Check for new position signals
                if long_prob > threshold:
                    # Enter long position
                    position = 1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                
                elif short_prob > threshold:
                    # Enter short position
                    position = -1
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                
                # Handle quantum edge case: both probabilities high
                elif long_prob > hedge_threshold and short_prob > hedge_threshold:
                    # Enter hedged position if both signals are strong
                    position = 2  # Hedged
                    entry_idx = i
                    entry_price = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                    df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
            
            elif position == 1:  # Currently long
                # Check for exit or hedge signals
                if long_prob < 0.3 or short_prob > threshold:
                    # Exit long position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Calculate return (accounting for fees)
                    trade_return = (current_price / entry_price) - 1 - (self.fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Check for immediate reversal to short
                    if short_prob > threshold:
                        position = -1
                        entry_idx = i
                        entry_price = current_price
                        df_backtest.loc[df_backtest.index[i], 'position'] = -1
                        df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    else:
                        position = 0
                
                elif short_prob > hedge_threshold:
                    # Add hedge to long position
                    position = 2  # Hedged
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                
                else:
                    # Stay in long position
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
            
            elif position == -1:  # Currently short
                # Check for exit or hedge signals
                if short_prob < 0.3 or long_prob > threshold:
                    # Exit short position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Calculate return (accounting for fees)
                    trade_return = 1 - (current_price / entry_price) - (self.fee_rate * 2)
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    # Check for immediate reversal to long
                    if long_prob > threshold:
                        position = 1
                        entry_idx = i
                        entry_price = current_price
                        df_backtest.loc[df_backtest.index[i], 'position'] = 1
                        df_backtest.loc[df_backtest.index[i], 'entry_price'] = current_price
                    else:
                        position = 0
                
                elif long_prob > hedge_threshold:
                    # Add hedge to short position
                    position = 2  # Hedged
                    df_backtest.loc[df_backtest.index[i], 'position'] = 2
                
                else:
                    # Stay in short position
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
            
            elif position == 2:  # Currently hedged
                # Check for removing hedge
                if long_prob < hedge_threshold and short_prob < hedge_threshold:
                    # Exit hedged position
                    df_backtest.loc[df_backtest.index[i], 'exit_price'] = current_price
                    df_backtest.loc[df_backtest.index[i], 'position'] = 0
                    
                    # Hedged positions typically have near-zero returns plus double fees
                    trade_return = -(self.fee_rate * 4)  # Approximate cost of hedging
                    df_backtest.loc[df_backtest.index[i], 'trade_return'] = trade_return
                    df_backtest.loc[df_backtest.index[i], 'trade_duration'] = i - entry_idx
                    
                    position = 0
                
                elif long_prob > threshold and short_prob < hedge_threshold:
                    # Convert hedge to pure long
                    position = 1
                    df_backtest.loc[df_backtest.index[i], 'position'] = 1
                
                elif short_prob > threshold and long_prob < hedge_threshold:
                    # Convert hedge to pure short
                    position = -1
                    df_backtest.loc[df_backtest.index[i], 'position'] = -1
                
                else:
                    # Stay hedged
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
    
    def _save_backtest_results(self, results, stats, exchange, symbol, timeframe):
        """Save backtest results to CSV and stats to JSON"""
        try:
            # Create output directory
            symbol_safe = symbol.replace('/', '_')
            output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results
            results_file = output_dir / f"{timeframe}_{timestamp}.csv"
            results.to_csv(results_file)
            
            # Save stats
            stats_file = output_dir / f"{timeframe}_{timestamp}_stats.csv"
            pd.DataFrame([stats]).to_csv(stats_file, index=False)
            
            self.logger.info(f"Backtest results saved to {results_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")
            return False
    
    def _plot_backtest_results(self, results, exchange, symbol, timeframe):
        """Plot backtest results"""
        try:
            symbol_safe = symbol.replace('/', '_')
            output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
            
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
            
            axes[0].set_title(f'{symbol} {timeframe} Price with Trading Signals')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Lower plot: Cumulative returns
            axes[1].plot(results.index, results['cumulative_returns'], label='Buy & Hold', color='blue')
            axes[1].plot(results.index, results['strategy_cumulative'], 
                       label='Quantum Strategy', color='purple')
            axes[1].set_title('Strategy Performance')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Cumulative Returns')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plot_file = output_dir / f"{timeframe}_{timestamp}_plot.png"
            plt.savefig(plot_file)
            plt.close(fig)
            
            self.logger.info(f"Backtest plot saved to {plot_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")
            return False