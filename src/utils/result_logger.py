#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified results logging utility for consistent reporting across all backtesting methods.

This module provides standardized functions for:
- Saving backtest results to CSV
- Generating performance metrics
- Creating visualizations
- Comparing strategies
- Producing summary reports

By using this utility across all backtesting methods, you'll get consistent 
output formats that make strategy comparison easier.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import json
import traceback

logger = logging.getLogger(__name__)

class ResultLogger:
    """Unified results logging for all trading strategies"""
    
    def __init__(self, config=None):
        """
        Initialize the result logger
        
        Args:
            config (dict): Configuration dictionary with logging and visualization settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Extract defaults from config
        self.fee_rate = self.config.get('backtesting', {}).get('fee_rate', 0.0006)
        
        # Configure visualization settings
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['figure.dpi'] = 100
    
    def save_results(self, results, metrics, exchange, symbol, timeframe, 
                    strategy_type, data_source="test_set"):
        """
        Save backtest results to standardized file formats
        
        Args:
            results (DataFrame): DataFrame containing backtest results 
            metrics (dict): Dictionary of performance metrics
            exchange (str): Exchange name
            symbol (str): Trading symbol
            timeframe (str): Timeframe of the data
            strategy_type (str): Strategy type identifier (e.g., 'quantum', 'position_sizing')
            data_source (str): Source of data ('test_set', 'full_data', etc.)
            
        Returns:
            dict: Dictionary with file paths of saved results
        """
        try:
            # Create output directory
            symbol_safe = symbol.replace('/', '_')
            output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results CSV
            results_file = output_dir / f"{timeframe}_{strategy_type}_{data_source}_{timestamp}.csv"
            results.to_csv(results_file)
            
            # Save metrics to JSON
            metrics['data_source'] = data_source
            metrics['strategy_type'] = strategy_type
            metrics['symbol'] = symbol
            metrics['timeframe'] = timeframe
            metrics['exchange'] = exchange
            metrics['timestamp'] = timestamp
            
            metrics_file = output_dir / f"{timeframe}_{strategy_type}_{data_source}_{timestamp}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4, default=str)
            
            # Save metrics to CSV for easy reading
            metrics_csv = output_dir / f"{timeframe}_{strategy_type}_{data_source}_{timestamp}_metrics.csv"
            pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
            
            # Return file paths
            files = {
                'results': str(results_file),
                'metrics_json': str(metrics_file),
                'metrics_csv': str(metrics_csv)
            }
            
            self.logger.info(f"Saved backtest results to {results_file}")
            return files
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def plot_results(self, results, metrics, exchange, symbol, timeframe, 
                    strategy_type, data_source="test_set", output_dir=None):
        """
        Create standardized visualizations for backtest results
        
        Args:
            results (DataFrame): DataFrame containing backtest results
            metrics (dict): Dictionary of performance metrics
            exchange (str): Exchange name
            symbol (str): Trading symbol
            timeframe (str): Timeframe of the data
            strategy_type (str): Strategy type identifier
            data_source (str): Source of data ('test_set', 'full_data', etc.)
            output_dir (Path, optional): Custom output directory
            
        Returns:
            dict: Dictionary with file paths of saved visualizations
        """
        try:
            # Setup output directory
            if output_dir is None:
                symbol_safe = symbol.replace('/', '_')
                output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Identify strategy type for proper visualization
            is_position_sizing = strategy_type == 'position_sizing'
            is_quantum = strategy_type == 'quantum'
            
            # Create visualization based on strategy type
            if is_position_sizing:
                viz_files = self._plot_position_sizing_results(
                    results, metrics, exchange, symbol, timeframe, 
                    strategy_type, data_source, output_dir, timestamp
                )
            else:
                viz_files = self._plot_standard_results(
                    results, metrics, exchange, symbol, timeframe, 
                    strategy_type, data_source, output_dir, timestamp
                )
            
            # Also create performance metrics visualization
            metrics_plot = self._plot_performance_metrics(
                metrics, exchange, symbol, timeframe, 
                strategy_type, data_source, output_dir, timestamp
            )
            
            # Add metrics plot to results
            if metrics_plot:
                viz_files['metrics_plot'] = metrics_plot
            
            self.logger.info(f"Created {len(viz_files)} visualizations for {symbol} {timeframe} {strategy_type}")
            return viz_files
            
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def _plot_standard_results(self, results, metrics, exchange, symbol, timeframe, 
                            strategy_type, data_source, output_dir, timestamp):
        """
        Plot standard backtest results (for quantum or traditional strategies)
        
        Args:
            results (DataFrame): DataFrame containing backtest results
            metrics (dict): Dictionary of performance metrics
            exchange (str): Exchange name
            symbol (str): Trading symbol
            timeframe (str): Timeframe of the data
            strategy_type (str): Strategy type identifier
            data_source (str): Source of data
            output_dir (Path): Output directory
            timestamp (str): Timestamp string
            
        Returns:
            dict: Dictionary with file paths of saved visualizations
        """
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Add data source indicator to title
        title_prefix = f"TEST SET - " if data_source == "test_set" else ""
        strategy_name = "Quantum" if strategy_type == "quantum" else "Strategy"
        
        # Upper plot: Price with positions
        axes[0].plot(results.index, results['close'], label=f'{symbol}', alpha=0.7)
        
        # Detect if data has discrete positions or continuous positions
        has_discrete_positions = 'position' in results.columns and results['position'].dtype == 'int64'
        has_continuous_positions = 'position' in results.columns and results['position'].dtype == 'float64'
        
        # Mark positions based on data type
        if has_discrete_positions:
            long_entries = results[results['position'].diff() == 1]
            short_entries = results[results['position'].diff() == -1]
            hedged_entries = results[results['position'].diff() == 2]
            exits = results[~results['exit_price'].isna()] if 'exit_price' in results.columns else pd.DataFrame()
            
            # Plot position markers
            axes[0].scatter(long_entries.index, long_entries['close'], marker='^', color='green', 
                        s=100, label='Long Entry')
            axes[0].scatter(short_entries.index, short_entries['close'], marker='v', color='red', 
                        s=100, label='Short Entry')
            
            if not hedged_entries.empty:
                axes[0].scatter(hedged_entries.index, hedged_entries['close'], marker='s', color='purple', 
                            s=100, label='Hedged Position')
                            
            if not exits.empty:
                axes[0].scatter(exits.index, exits['close'], marker='x', color='black', 
                            s=80, label='Exit')
        
        elif has_continuous_positions:
            # For continuous positions, shade the background based on position value
            for i in range(1, len(results)):
                pos = results['position'].iloc[i]
                if pos > 0:
                    axes[0].axvspan(results.index[i-1], results.index[i], 
                                    alpha=min(0.3, abs(pos)*0.3), color='green', lw=0)
                elif pos < 0:
                    axes[0].axvspan(results.index[i-1], results.index[i], 
                                    alpha=min(0.3, abs(pos)*0.3), color='red', lw=0)
        
        # Add probability shading if available
        if all(col in results.columns for col in ['long_prob', 'short_prob']):
            for i in range(0, len(results), 20):  # Every 20th point to avoid clutter
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
        
        # Middle plot: Cumulative returns
        buy_hold_col = 'cumulative_returns' if 'cumulative_returns' in results.columns else 'price_cumulative'
        strategy_col = 'strategy_cumulative'
        
        if buy_hold_col in results.columns and strategy_col in results.columns:
            axes[1].plot(results.index, results[buy_hold_col], label='Buy & Hold', color='blue')
            axes[1].plot(results.index, results[strategy_col], 
                        label=f'{strategy_name}', color='purple')
            axes[1].set_title(f'{title_prefix}Strategy Performance')
            axes[1].set_ylabel('Cumulative Returns')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Bottom plot: Position over time
        if 'position' in results.columns:
            if has_continuous_positions:
                # For continuous positions (-1.0 to 1.0)
                axes[2].plot(results.index, results['position'], label='Position', color='black')
                axes[2].fill_between(results.index, 0, results['position'], 
                                    where=results['position'] > 0, color='green', alpha=0.3)
                axes[2].fill_between(results.index, 0, results['position'], 
                                    where=results['position'] < 0, color='red', alpha=0.3)
                axes[2].set_title('Position Size Over Time')
                axes[2].set_yticks([-1, -0.5, 0, 0.5, 1])
            else:
                # For discrete positions (-1, 0, 1, 2)
                axes[2].plot(results.index, results['position'], label='Position', color='black')
                axes[2].set_title('Position Over Time')
                axes[2].set_yticks([-1, 0, 1, 2])
                axes[2].set_yticklabels(['Short', 'Flat', 'Long', 'Hedged'])
            
            axes[2].axhline(y=0, color='gray', linestyle='--')
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Position')
            axes[2].grid(True, alpha=0.3)
            
            # Format x-axis date labels
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                if len(results) < 20:  # For shorter timeframes, add day labels
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                elif len(results) < 180:  # For medium timeframes, show weekly labels
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                else:  # For longer timeframes, show monthly labels
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
        # Below the existing bottom plot, add a plot for compound factor if it exists
        if 'compound_factor' in results.columns:
            # Add a fourth subplot for compounding
            fig, axes = plt.subplots(4, 1, figsize=(14, 20), 
                                gridspec_kw={'height_ratios': [2, 1, 1, 1]})
            
            # Plot the compounding factor in the fourth subplot
            axes[3].plot(results.index, results['compound_factor'], 
                    label='Compound Factor', color='purple')
            axes[3].set_title('Account Growth Factor')
            axes[3].set_ylabel('Multiplier')
            axes[3].axhline(y=1.0, color='gray', linestyle='--')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout(pad=1.5)
        
        # Save figure
        plot_file = output_dir / f"{timeframe}_{strategy_type}_{data_source}_{timestamp}_plot.png"
        plt.savefig(plot_file, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # Create trade summary plot if trades are available
        summary_plot = None
        if 'trade_return' in results.columns:
            summary_plot = self._plot_trade_summary(
                results, metrics, symbol, timeframe, strategy_type, output_dir, timestamp
            )
        
        return {
            'main_plot': str(plot_file),
            'trade_summary': str(summary_plot) if summary_plot else None
        }
    
    def _plot_position_sizing_results(self, results, metrics, exchange, symbol, timeframe, 
                                    strategy_type, data_source, output_dir, timestamp):
        """Plot position sizing backtest results"""
        # Use the standard plotting function with position_sizing=True
        return self._plot_standard_results(
            results, metrics, exchange, symbol, timeframe, 
            strategy_type, data_source, output_dir, timestamp
        )
    
    def _plot_performance_metrics(self, metrics, exchange, symbol, timeframe, 
                                strategy_type, data_source, output_dir, timestamp):
        """Create a visualization of key performance metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Flatten axes for easier access
            axes = axes.flatten()
            
            # Top left: Return metrics
            return_metrics = [
                ('Total Return', metrics.get('final_return', metrics.get('total_return', 0)) * 100),
                ('Buy & Hold', metrics.get('buy_hold_return', 0) * 100),
                ('Alpha', metrics.get('alpha', 0) * 100),
            ]
            
            colors = ['green' if x[1] > 0 else 'red' for x in return_metrics]
            axes[0].bar([x[0] for x in return_metrics], [x[1] for x in return_metrics], color=colors)
            axes[0].set_title('Return Metrics (%)')
            axes[0].set_ylabel('Percent (%)')
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels to bars
            for i, v in enumerate([x[1] for x in return_metrics]):
                axes[0].text(i, v + (1 if v >= 0 else -1), 
                           f'{v:.2f}%', ha='center', va='bottom' if v >= 0 else 'top', 
                           color='black')
            
            # Top right: Risk metrics
            risk_metrics = [
                ('Sharpe', metrics.get('sharpe_ratio', 0)),
                ('Win Rate', metrics.get('win_rate', 0) * 100),
                ('Max DD', metrics.get('max_drawdown', 0) * 100),
            ]
            
            axes[1].bar([x[0] for x in risk_metrics], [x[1] for x in risk_metrics], 
                      color=['blue', 'blue', 'red'])
            axes[1].set_title('Risk Metrics')
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels to bars
            for i, v in enumerate([x[1] for x in risk_metrics]):
                axes[1].text(i, v + (0.1 if v >= 0 else -0.1), 
                           f'{v:.2f}' + ("%" if i > 0 else ""), 
                           ha='center', va='bottom' if v >= 0 else 'top', 
                           color='black')
            
            # Bottom left: Trade metrics
            trade_metrics = [
                ('# Trades', metrics.get('total_trades', metrics.get('num_position_changes', 0))),
                ('Avg Win', metrics.get('avg_win', 0) * 100),
                ('Avg Loss', abs(metrics.get('avg_loss', 0)) * 100),
            ]
            
            axes[2].bar([x[0] for x in trade_metrics], [x[1] for x in trade_metrics], 
                      color=['blue', 'green', 'red'])
            axes[2].set_title('Trade Metrics')
            axes[2].grid(True, alpha=0.3)
            
            # Add value labels to bars
            for i, v in enumerate([x[1] for x in trade_metrics]):
                if i == 0:  # Number of trades (no percentage)
                    axes[2].text(i, v + 1, f'{int(v)}', ha='center', va='bottom', color='black')
                else:  # Win/loss percentages
                    axes[2].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', color='black')
            
            # Bottom right: Fee impact
            if 'fee_drag' in metrics:
                fee_metrics = [
                    ('Fee Drag', metrics.get('fee_drag', 0) * 100),
                    ('Gross Return', (metrics.get('final_return', metrics.get('total_return', 0)) + 
                                       metrics.get('fee_drag', 0)) * 100),
                    ('Net Return', metrics.get('final_return', metrics.get('total_return', 0)) * 100)
                ]
                
                # Calculate colors - red for fee drag, comparison for returns
                fee_colors = ['red', 
                              'green' if fee_metrics[1][1] > 0 else 'red',
                              'green' if fee_metrics[2][1] > 0 else 'red']
                
                axes[3].bar([x[0] for x in fee_metrics], [x[1] for x in fee_metrics], color=fee_colors)
                axes[3].set_title('Fee Impact (%)')
                axes[3].grid(True, alpha=0.3)
                
                # Add value labels to bars
                for i, v in enumerate([x[1] for x in fee_metrics]):
                    axes[3].text(i, v + (0.5 if v >= 0 else -0.5), 
                               f'{v:.2f}%', ha='center', va='bottom' if v >= 0 else 'top', 
                               color='black')
            else:
                # Alternative: profit factor if fee drag not available
                profit_factor = metrics.get('profit_factor', 0)
                if isinstance(profit_factor, (int, float)) and profit_factor != float('inf'):
                    axes[3].bar(['Profit Factor'], [profit_factor], color='purple')
                    axes[3].set_title('Profit Factor (Win ÷ Loss)')
                    axes[3].grid(True, alpha=0.3)
                    axes[3].text(0, profit_factor + 0.1, f'{profit_factor:.2f}', 
                               ha='center', va='bottom', color='black')
                else:
                    axes[3].text(0.5, 0.5, 'No profit factor data available', 
                               ha='center', va='center', transform=axes[3].transAxes)
            
            # Add overall title and adjust layout
            plt.suptitle(f'{symbol} {timeframe} - {strategy_type.replace("_", " ").title()} Performance Metrics', 
                         fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)
            
            # Save figure
            metrics_file = output_dir / f"{timeframe}_{strategy_type}_{data_source}_{timestamp}_metrics_plot.png"
            plt.savefig(metrics_file, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            return str(metrics_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _plot_trade_summary(self, results, metrics, symbol, timeframe, strategy_type, output_dir, timestamp):
        """Create a trade summary visualization"""
        try:
            # Extract trades
            trades = results[~results['trade_return'].isna()] if 'trade_return' in results.columns else None
            
            if trades is None or len(trades) == 0:
                return None
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Top left: Trade returns distribution
            trades['trade_return_pct'] = trades['trade_return'] * 100
            axes[0, 0].hist(trades['trade_return_pct'], bins=20, alpha=0.7, color='purple')
            axes[0, 0].axvline(x=0, color='red', linestyle='--')
            axes[0, 0].set_title('Trade Returns Distribution')
            axes[0, 0].set_xlabel('Return (%)')
            axes[0, 0].set_ylabel('Number of Trades')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Top right: Win/Loss pie chart
            win_count = (trades['trade_return'] > 0).sum()
            loss_count = (trades['trade_return'] < 0).sum()
            even_count = ((trades['trade_return'] == 0)).sum()
            
            if win_count + loss_count + even_count > 0:
                axes[0, 1].pie(
                    [win_count, loss_count, even_count],
                    labels=['Wins', 'Losses', 'Breakeven'],
                    autopct='%1.1f%%',
                    colors=['green', 'red', 'gray'],
                    startangle=90
                )
                axes[0, 1].set_title('Win/Loss Ratio')
            else:
                axes[0, 1].text(0.5, 0.5, 'No trade data available', 
                             ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # Bottom left: Trade duration distribution
            if 'trade_duration' in trades.columns:
                axes[1, 0].hist(trades['trade_duration'], bins=20, alpha=0.7, color='blue')
                axes[1, 0].set_title('Trade Duration Distribution')
                axes[1, 0].set_xlabel('Duration (bars)')
                axes[1, 0].set_ylabel('Number of Trades')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No duration data available', 
                             ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Bottom right: Consecutive wins/losses
            if len(trades) > 0:
                trade_results = trades['trade_return'] > 0
                current_streak = 1
                streaks = []
                streak_types = []
                
                for i in range(1, len(trade_results)):
                    if trade_results.iloc[i] == trade_results.iloc[i-1]:
                        current_streak += 1
                    else:
                        streaks.append(current_streak)
                        streak_types.append('Win' if trade_results.iloc[i-1] else 'Loss')
                        current_streak = 1
                
                # Add the last streak
                if len(trade_results) > 0:
                    streaks.append(current_streak)
                    streak_types.append('Win' if trade_results.iloc[-1] else 'Loss')
                
                # Plot streak info
                max_win_streak = max([s for i, s in enumerate(streaks) if streak_types[i] == 'Win'], default=0)
                max_loss_streak = max([s for i, s in enumerate(streaks) if streak_types[i] == 'Loss'], default=0)
                
                axes[1, 1].bar(['Max Win Streak', 'Max Loss Streak'], [max_win_streak, max_loss_streak],
                             color=['green', 'red'])
                axes[1, 1].set_title('Maximum Consecutive Wins/Losses')
                axes[1, 1].set_ylabel('Number of Trades')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add value labels
                axes[1, 1].text(0, max_win_streak + 0.1, str(max_win_streak), ha='center', va='bottom')
                axes[1, 1].text(1, max_loss_streak + 0.1, str(max_loss_streak), ha='center', va='bottom')
            else:
                axes[1, 1].text(0.5, 0.5, 'No trade data available', 
                             ha='center', va='center', transform=axes[1, 1].transAxes)
            
            # Add title and adjust layout
            plt.suptitle(f'{symbol} {timeframe} - Trade Analysis ({len(trades)} Trades)', 
                         fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)
            
            # Save figure
            summary_file = output_dir / f"{timeframe}_{strategy_type}_trade_summary_{timestamp}.png"
            plt.savefig(summary_file, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            return str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting trade summary: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def create_multi_strategy_comparison(self, results_dict, output_dir=None):
        """
        Create a comparison between multiple strategies or configurations
        
        Args:
            results_dict (dict): Dictionary of results:
                {
                    'strategy_name': {
                        'results': DataFrame, 
                        'metrics': dict
                    }
                }
            output_dir (Path, optional): Custom output directory
            
        Returns:
            str: Path to comparison visualization
        """
        try:
            # Setup output directory
            if output_dir is None:
                output_dir = Path(f"data/backtest_results/comparisons")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Extract strategy names and metrics
            strategy_names = list(results_dict.keys())
            metrics_list = [results_dict[name]['metrics'] for name in strategy_names]
            
            # Collect key metrics for comparison
            comparison_metrics = {
                'Strategy': strategy_names,
                'Total Return (%)': [m.get('final_return', m.get('total_return', 0)) * 100 
                                     for m in metrics_list],
                'Sharpe Ratio': [m.get('sharpe_ratio', 0) for m in metrics_list],
                'Win Rate (%)': [m.get('win_rate', 0) * 100 for m in metrics_list],
                'Max Drawdown (%)': [m.get('max_drawdown', 0) * 100 for m in metrics_list],
                'Trades': [m.get('total_trades', m.get('num_position_changes', 0)) 
                          for m in metrics_list],
                'Fee Drag (%)': [m.get('fee_drag', 0) * 100 for m in metrics_list]
            }
            
            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_metrics)
            
            # Save comparison to CSV
            csv_file = output_dir / f"strategy_comparison_{timestamp}.csv"
            comparison_df.to_csv(csv_file, index=False)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Top left: Total returns comparison
            axes[0, 0].bar(comparison_df['Strategy'], comparison_df['Total Return (%)'], 
                         color=['green' if x > 0 else 'red' for x in comparison_df['Total Return (%)']])
            axes[0, 0].set_title('Total Return (%)')
            axes[0, 0].set_ylabel('Percent (%)')
            axes[0, 0].set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(comparison_df['Total Return (%)']):
                axes[0, 0].text(i, v + (1 if v >= 0 else -1), 
                              f'{v:.2f}%', ha='center', va='bottom' if v >= 0 else 'top')
            
            # Top right: Sharpe ratio comparison
            axes[0, 1].bar(comparison_df['Strategy'], comparison_df['Sharpe Ratio'],
                         color=['green' if x > 1 else 'orange' if x > 0 else 'red' 
                                for x in comparison_df['Sharpe Ratio']])
            axes[0, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Sharpe Ratio')
            axes[0, 1].set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(comparison_df['Sharpe Ratio']):
                axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
            
            # Bottom left: Win rate vs Max drawdown
            x = np.arange(len(comparison_df['Strategy']))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, comparison_df['Win Rate (%)'], width, 
                         label='Win Rate (%)', color='green')
            axes[1, 0].bar(x + width/2, -comparison_df['Max Drawdown (%)'], width, 
                         label='Max Drawdown (%)', color='red')
            
            axes[1, 0].set_title('Win Rate vs Max Drawdown')
            axes[1, 0].set_ylabel('Percent (%)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(comparison_df['Win Rate (%)']):
                axes[1, 0].text(i - width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom')
            
            for i, v in enumerate(comparison_df['Max Drawdown (%)']):
                axes[1, 0].text(i + width/2, -v - 1, f'{v:.1f}%', ha='center', va='top')
            
            # Bottom right: Number of trades vs Fee drag
            axes[1, 1].bar(comparison_df['Strategy'], comparison_df['Trades'], 
                         color='blue', alpha=0.7, label='# Trades')
            
            # Create a second y-axis for fee drag
            ax2 = axes[1, 1].twinx()
            ax2.plot(comparison_df['Strategy'], comparison_df['Fee Drag (%)'], 
                    'ro-', linewidth=2, markersize=8, label='Fee Drag (%)')
            
            axes[1, 1].set_title('Trades vs Fee Drag')
            axes[1, 1].set_ylabel('Number of Trades')
            ax2.set_ylabel('Fee Drag (%)', color='r')
            axes[1, 1].set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
            
            # Combine legends from both axes
            lines, labels = axes[1, 1].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')
            
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels for trades
            for i, v in enumerate(comparison_df['Trades']):
                axes[1, 1].text(i, v + 1, f'{int(v)}', ha='center', va='bottom')
            
            # Add title and adjust layout
            plt.suptitle('Strategy Comparison', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)
            
            # Save figure
            plot_file = output_dir / f"strategy_comparison_{timestamp}.png"
            plt.savefig(plot_file, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            # Create equity curve comparison
            self._plot_equity_curve_comparison(results_dict, output_dir, timestamp)
            
            return str(plot_file)
            
        except Exception as e:
            self.logger.error(f"Error creating strategy comparison: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _plot_equity_curve_comparison(self, results_dict, output_dir, timestamp):
        """Create a comparison of equity curves between strategies"""
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve for each strategy
            for strategy_name, data in results_dict.items():
                results = data['results']
                
                # Identify the equity curve column
                curve_col = None
                for col_name in ['strategy_cumulative', 'cumulative_strategy', 'equity_curve']:
                    if col_name in results.columns:
                        curve_col = col_name
                        break
                
                if curve_col is not None:
                    plt.plot(results.index, results[curve_col], 
                           label=f'{strategy_name} ({results[curve_col].iloc[-1]*100:.1f}%)', 
                           linewidth=2)
            
            # Add buy & hold for reference if it exists in any of the results
            for data in results_dict.values():
                results = data['results']
                
                # Look for buy & hold curve
                bh_col = None
                for col_name in ['cumulative_returns', 'price_cumulative', 'buy_hold_curve']:
                    if col_name in results.columns:
                        bh_col = col_name
                        break
                
                if bh_col is not None:
                    plt.plot(results.index, results[bh_col], 
                           label=f'Buy & Hold ({results[bh_col].iloc[-1]*100:.1f}%)', 
                           color='blue', linestyle='--', alpha=0.7)
                    break
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.title('Equity Curve Comparison')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis date labels
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Save figure
            curves_file = output_dir / f"equity_curves_comparison_{timestamp}.png"
            plt.savefig(curves_file, dpi=120, bbox_inches='tight')
            plt.close()
            
            return str(curves_file)
            
        except Exception as e:
            self.logger.error(f"Error creating equity curve comparison: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None