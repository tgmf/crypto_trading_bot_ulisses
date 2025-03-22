#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization tools for trading strategy analysis.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require Qt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

class VisualizationTool:
    """Tools for visualizing trading strategy performance"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def plot_backtest_results(self, results, output_path=None, continuous_position=False):
        """Plot backtest results with support for continuous position sizing"""
        try:
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(14, 16), 
                                        gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Get symbol and timeframe info
            symbol = results.get('symbol', 'Unknown')
            timeframe = results.get('timeframe', 'Unknown')
            
            # Extract data
            df = results.get('data')
            if df is None:
                self.logger.error("No data found in results")
                return False
            
            # Upper plot: Price with positions
            axes[0].plot(df.index, df['close'], label=f'{symbol}', alpha=0.7)
        
            # Check if this is a traditional backtest (with exits) or position sizing
            is_position_sizing = ('position' in df.columns and 
                                    'exit_price' not in df.columns and
                                    df['position'].dtype == 'float64')
        
            if is_position_sizing:
                # For position sizing, shade the background based on position
                for i in range(1, len(df)):
                    pos = df['position'].iloc[i]
                    if pos > 0:
                        axes[0].axvspan(df.index[i-1], df.index[i], 
                                    alpha=min(0.3, abs(pos)*0.3), color='green', lw=0)
                    elif pos < 0:
                        axes[0].axvspan(df.index[i-1], df.index[i], 
                                    alpha=min(0.3, abs(pos)*0.3), color='red', lw=0)
            
            else:
            # Traditional backtesting with discrete positions
            # Mark positions
                if 'position' in df.columns:
                    long_entries = df[df['position'].diff() == 1]
                    short_entries = df[df['position'].diff() == -1]
                    hedged_entries = df[df['position'].diff() == 2]
                    
                    # Only try to use exit_price if it exists
                    if 'exit_price' in df.columns:
                        exits = df[~df['exit_price'].isna()]
                        axes[0].scatter(exits.index, exits['close'], marker='x', color='black', 
                                    s=80, label='Exit')
                    
                    # Plot position markers
                    axes[0].scatter(long_entries.index, long_entries['close'], marker='^', color='green', 
                                s=100, label='Long Entry')
                    axes[0].scatter(short_entries.index, short_entries['close'], marker='v', color='red', 
                                s=100, label='Short Entry')
                    axes[0].scatter(hedged_entries.index, hedged_entries['close'], marker='s', color='purple', 
                                s=100, label='Hedged Position')
            
            # Add probability shading if available
            if 'long_prob' in df.columns and 'short_prob' in df.columns:
                for i in range(0, len(df), 20):  # Every 20th point to avoid clutter
                    if df['long_prob'].iloc[i] > 0.3:
                        axes[0].axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                                    alpha=df['long_prob'].iloc[i] * 0.3, color='green', lw=0)
                    if df['short_prob'].iloc[i] > 0.3:
                        axes[0].axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                                    alpha=df['short_prob'].iloc[i] * 0.3, color='red', lw=0)
            
            axes[0].set_title(f'{symbol} {timeframe} Price with Trading Signals')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Middle plot: Cumulative returns
            if 'cumulative_returns' in df.columns and 'strategy_cumulative' in df.columns:
                axes[1].plot(df.index, df['cumulative_returns'], label='Buy & Hold', color='blue')
                axes[1].plot(df.index, df['strategy_cumulative'], 
                            label='Strategy', color='purple')
                axes[1].set_title('Strategy Performance')
                axes[1].set_ylabel('Cumulative Returns')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            # Bottom plot: Position over time
            if 'position' in df.columns:
                # Fix the conditional statement - it was comparing with a string
                if is_position_sizing:
                    # For continuous positions (-1.0 to 1.0)
                    axes[2].plot(df.index, df['position'], label='Position', color='black')
                    axes[2].fill_between(df.index, 0, df['position'], 
                                    where=df['position'] > 0, color='green', alpha=0.3)
                    axes[2].fill_between(df.index, 0, df['position'], 
                                    where=df['position'] < 0, color='red', alpha=0.3)
                    axes[2].set_title('Position Size Over Time')
                    axes[2].set_yticks([-1, -0.5, 0, 0.5, 1])
                    axes[2].axhline(y=0, color='gray', linestyle='--')
                else:
                    # For discrete positions (-1, 0, 1, 2)
                    axes[2].plot(df.index, df['position'], label='Position', color='black')
                    axes[2].set_title('Position Over Time')
                    axes[2].set_yticks([-1, 0, 1, 2])
                    axes[2].set_yticklabels(['Short', 'Flat', 'Long', 'Hedged'])
                
                axes[2].set_xlabel('Date')
                axes[2].set_ylabel('Position')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure if path provided
            if output_path:
                plt.savefig(output_path)
                self.logger.info(f"Backtest plot saved to {output_path}")
                plt.close(fig)
                return True
            else:
                return fig
                
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def plot_performance_metrics(self, results, output_path=None):
        """Plot performance metrics"""
        try:
            # Extract stats
            stats = results.get('stats')
            if stats is None:
                self.logger.error("No stats found in results")
                return False
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            # Plot trade outcomes (win/loss)
            if 'win_rate' in stats:
                win_rate = stats['win_rate']
                lose_rate = 1 - win_rate
                axes[0].pie([win_rate, lose_rate], 
                         labels=['Win', 'Loss'], 
                         colors=['green', 'red'],
                         autopct='%1.1f%%',
                         startangle=90)
                axes[0].set_title('Trade Outcomes')
            
            # Plot trade types
            if all(k in stats for k in ['long_trades', 'short_trades', 'hedged_trades']):
                trade_types = [stats['long_trades'], stats['short_trades'], stats['hedged_trades']]
                axes[1].pie(trade_types, 
                         labels=['Long', 'Short', 'Hedged'],
                         colors=['green', 'red', 'purple'],
                         autopct='%1.1f%%',
                         startangle=90)
                axes[1].set_title('Trade Types')
            
            # Plot average gain/loss
            if all(k in stats for k in ['avg_win', 'avg_loss']):
                avg_win = stats['avg_win']
                avg_loss = abs(stats['avg_loss'])
                axes[2].bar(['Win', 'Loss'], [avg_win, avg_loss], 
                          color=['green', 'red'])
                axes[2].set_title('Average Gain/Loss')
                axes[2].set_ylabel('Return (%)')
                
                # Add text labels
                axes[2].text(0, avg_win/2, f'{avg_win:.2%}', 
                          ha='center', va='center', color='white', fontweight='bold')
                axes[2].text(1, avg_loss/2, f'{avg_loss:.2%}', 
                          ha='center', va='center', color='white', fontweight='bold')
            
            # Plot key metrics
            if all(k in stats for k in ['win_rate', 'profit_factor', 'final_return']):
                metrics = {
                    'Win Rate': stats['win_rate'],
                    'Profit Factor': min(stats['profit_factor'], 10),  # Cap at 10 for visualization
                    'Total Return': stats['final_return']
                }
                axes[3].bar(metrics.keys(), metrics.values(), color='blue')
                axes[3].set_title('Key Performance Metrics')
                
                # Add text labels
                for i, (k, v) in enumerate(metrics.items()):
                    if k == 'Win Rate' or k == 'Total Return':
                        label = f'{v:.2%}'
                    else:
                        label = f'{v:.2f}'
                    axes[3].text(i, v/2, label, 
                              ha='center', va='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure if path provided
            if output_path:
                plt.savefig(output_path)
                self.logger.info(f"Performance metrics plot saved to {output_path}")
                plt.close(fig)
                return True
            else:
                return fig
                
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}")
            return False
    
    def plot_probability_distribution(self, results, output_path=None):
        """Plot probability distributions"""
        try:
            # Extract data
            df = results.get('data')
            if df is None or not all(col in df.columns for col in ['long_prob', 'short_prob', 'no_trade_prob']):
                self.logger.error("No probability data found in results")
                return False
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot probability distributions
            sns.histplot(df['long_prob'], kde=True, color='green', ax=axes[0, 0])
            axes[0, 0].set_title('Long Probability Distribution')
            axes[0, 0].set_xlabel('Probability')
            axes[0, 0].set_ylabel('Frequency')
            
            sns.histplot(df['short_prob'], kde=True, color='red', ax=axes[0, 1])
            axes[0, 1].set_title('Short Probability Distribution')
            axes[0, 1].set_xlabel('Probability')
            axes[0, 1].set_ylabel('Frequency')
            
            sns.histplot(df['no_trade_prob'], kde=True, color='blue', ax=axes[1, 0])
            axes[1, 0].set_title('No-Trade Probability Distribution')
            axes[1, 0].set_xlabel('Probability')
            axes[1, 0].set_ylabel('Frequency')
            
            # Plot probability over time
            df[['long_prob', 'short_prob', 'no_trade_prob']].plot(
                ax=axes[1, 1], alpha=0.7)
            axes[1, 1].set_title('Probability Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Probability')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure if path provided
            if output_path:
                plt.savefig(output_path)
                self.logger.info(f"Probability distribution plot saved to {output_path}")
                plt.close(fig)
                return True
            else:
                return fig
                
        except Exception as e:
            self.logger.error(f"Error plotting probability distributions: {str(e)}")
            return False