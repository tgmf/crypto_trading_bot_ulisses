#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis tools for backtest results, handling large datasets efficiently.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
import argparse
import re
from datetime import datetime

class BacktestAnalyzer:
    """Analyzes backtest results and generates summary statistics"""
    
    def __init__(self, config=None):
        """Initialize with optional configuration"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def find_latest_backtest(self, exchange, symbol, timeframe=None):
        """Find the latest backtest result for the given symbol and timeframe"""
        symbol_safe = symbol.replace('/', '_')
        base_path = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
        
        if not base_path.exists():
            self.logger.error(f"No backtest results found at {base_path}")
            return None
            
        # Pattern to match backtest result files
        pattern = f"{timeframe}_*_*.csv" if timeframe else "*.csv"
        
        # Exclude stats files
        files = [f for f in base_path.glob(pattern) if not f.name.endswith('_stats.csv')]
        
        if not files:
            self.logger.error(f"No matching backtest files found for {symbol} {timeframe}")
            return None
            
        # Sort by modification time (most recent first)
        latest_file = max(files, key=os.path.getmtime)
        
        self.logger.info(f"Found latest backtest file: {latest_file}")
        return latest_file
    
    def analyze_file(self, file_path, chunk_size=100000):
        """
        Analyze a backtest results file, processing in chunks to handle large files
        
        Args:
            file_path: Path to the backtest results CSV
            chunk_size: Number of rows to process at once
            
        Returns:
            dict: Summary statistics
        """
        self.logger.info(f"Analyzing backtest file: {file_path}")
        
        # Extract metadata from filename
        filename = os.path.basename(file_path)
        match = re.match(r'([^_]+)_([^_]+)_(\d+)_(\d+)\.csv', filename)
        
        if match:
            timeframe = match.group(1)
            data_source = match.group(2)
            timestamp = f"{match.group(3)}_{match.group(4)}"
        else:
            timeframe = "unknown"
            data_source = "unknown"
            timestamp = "unknown"
        
        # Initialize counters and accumulators
        total_rows = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_win_return = 0.0
        total_loss_return = 0.0
        trade_durations = []
        
        # Position distribution
        position_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
        
        # Track final values
        final_values = {
            'close': None,
            'cumulative_returns': None,
            'strategy_cumulative': None
        }
        
        # Time period
        start_date = None
        end_date = None
        
        # Determine the trade return column name - different methods use different naming
        trade_return_col = None
        position_size_col = None
        
        # Process file in chunks
        reader = pd.read_csv(file_path, chunksize=chunk_size)
        
        # Examine first chunk to determine column structure
        first_chunk = next(reader)
        
        # Check available columns
        if 'trade_return' in first_chunk.columns:
            trade_return_col = 'trade_return'
        elif 'position_return' in first_chunk.columns:  # Position sizing may use this name
            trade_return_col = 'position_return'
        elif 'trade_pnl' in first_chunk.columns:  # Another potential name
            trade_return_col = 'trade_pnl'
        
        # Check for position size column
        if 'position_size' in first_chunk.columns:
            position_size_col = 'position_size'
        
        # Reset the reader to include first chunk
        reader = pd.read_csv(file_path, chunksize=chunk_size)
        
        for i, chunk in enumerate(reader):
            # Track row count
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            
            # Process each chunk
            if i == 0:
                # First chunk - get start date
                if 'timestamp' in chunk.columns:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                    start_date = chunk['timestamp'].min()
            
            # Track last values
            for key in final_values.keys():
                if key in chunk.columns:
                    final_values[key] = chunk[key].iloc[-1]
            
            # Update end date
            if 'timestamp' in chunk.columns:
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                end_date = chunk['timestamp'].max()
            
            # Find trades in this chunk if we have a trade return column
            if trade_return_col is not None and trade_return_col in chunk.columns:
                trades = chunk[chunk[trade_return_col].notna()]
                
                chunk_total_trades = len(trades)
                total_trades += chunk_total_trades
                
                if chunk_total_trades > 0:
                    # Count winners and losers
                    winners = trades[trades[trade_return_col] > 0]
                    losers = trades[trades[trade_return_col] < 0]
                    
                    winning_trades += len(winners)
                    losing_trades += len(losers)
                    
                    # Accumulate returns
                    total_win_return += winners[trade_return_col].sum()
                    total_loss_return += losers[trade_return_col].sum()
                    
                    # Track durations
                    if 'trade_duration' in trades.columns:
                        trade_durations.extend(trades['trade_duration'].dropna().tolist())
            
            # Track positions
            if 'position' in chunk.columns:
                for pos in [-1, 0, 1, 2]:
                    position_counts[pos] += (chunk['position'] == pos).sum()
                    
            # Alternative: track position sizes if available
            elif position_size_col is not None:
                non_zero_positions = chunk[chunk[position_size_col] != 0]
                pos_sizes = non_zero_positions[position_size_col]
                
                # Count positive and negative positions
                position_counts[1] += (pos_sizes > 0).sum()
                position_counts[-1] += (pos_sizes < 0).sum()
                position_counts[0] += (chunk[position_size_col] == 0).sum()
            
            self.logger.info(f"Processed chunk {i+1} with {chunk_rows} rows, {chunk_total_trades if trade_return_col else 0} trades")
        
        # Calculate statistics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        average_win = total_win_return / winning_trades if winning_trades > 0 else 0
        average_loss = total_loss_return / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(total_win_return / total_loss_return) if total_loss_return < 0 and total_loss_return != 0 else float('inf')
        
        average_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Calculate position distribution
        position_distribution = {}
        total_positions = sum(position_counts.values())
        for pos, count in position_counts.items():
            position_distribution[pos] = count / total_positions if total_positions > 0 else 0
        
        # Handle missing values
        if final_values['strategy_cumulative'] is None and 'strategy_return' in first_chunk.columns:
            # Try to calculate from strategy returns
            strategy_returns = pd.read_csv(file_path, usecols=['strategy_return'])
            if not strategy_returns.empty:
                final_values['strategy_cumulative'] = (1 + strategy_returns['strategy_return']).cumprod().iloc[-1] - 1
        
        # Summarize results
        summary = {
            'file_path': str(file_path),
            'timeframe': timeframe,
            'data_source': data_source,
            'timestamp': timestamp,
            'start_date': start_date,
            'end_date': end_date,
            'total_rows': total_rows,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'average_win': average_win,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'average_duration': average_duration,
            'final_price': final_values['close'],
            'buy_hold_return': final_values['cumulative_returns'],
            'strategy_return': final_values['strategy_cumulative'],
            'position_distribution': position_distribution,
            'position_sizing_used': position_size_col is not None
        }
        
        return summary
    
    def print_summary(self, summary):
        """Print formatted summary statistics"""
        print("\n" + "="*80)
        print(f"BACKTEST SUMMARY: {summary['timeframe']} {summary['data_source']}")
        print("="*80)
        
        # Time period
        if summary['start_date'] and summary['end_date']:
            print(f"\nPeriod: {summary['start_date']} to {summary['end_date']}")
            days = (summary['end_date'] - summary['start_date']).days
            print(f"Duration: {days} days")
        
        # Performance
        print("\nPERFORMANCE METRICS:")
        print(f"Strategy Return: {summary['strategy_return']:.2%}")
        print(f"Buy & Hold Return: {summary['buy_hold_return']:.2%}")
        if summary['buy_hold_return'] != 0:
            alpha = summary['strategy_return'] - summary['buy_hold_return']
            print(f"Alpha: {alpha:.2%}")
        
        # Trade statistics
        print("\nTRADE STATISTICS:")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.2%}")
        print(f"Avg Win: {summary['average_win']:.4%}")
        print(f"Avg Loss: {summary['average_loss']:.4%}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print(f"Avg Trade Duration: {summary['average_duration']:.1f} bars")
        
        # Position distribution
        print("\nPOSITION DISTRIBUTION:")
        for pos, pct in summary['position_distribution'].items():
            pos_label = {-1: "Short", 0: "Flat", 1: "Long", 2: "Hedged"}.get(pos, str(pos))
            print(f"{pos_label}: {pct:.2%}")
        
        # Dataset info
        print("\nDATASET INFO:")
        print(f"Total Rows: {summary['total_rows']:,}")
        print(f"File: {summary['file_path']}")
        print(f"Generated: {summary['timestamp']}")
        
        print("\n" + "="*80)
        
    def save_summary(self, summary, output_path=None):
        """Save summary to JSON file"""
        if output_path is None:
            # Create default output path
            base_dir = Path("data/backtest_results/summary")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = base_dir / f"backtest_summary_{summary['timeframe']}_{timestamp}.json"
        
        # Convert summary to saveable format
        saveable_summary = summary.copy()
        
        # Convert datetime objects to strings
        if saveable_summary['start_date']:
            saveable_summary['start_date'] = saveable_summary['start_date'].strftime('%Y-%m-%d %H:%M:%S')
        if saveable_summary['end_date']:
            saveable_summary['end_date'] = saveable_summary['end_date'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(saveable_summary, f, indent=2)
            
        self.logger.info(f"Summary saved to {output_path}")
        return output_path
    
    def plot_summary(self, summary, output_path=None):
        """Create visualization of summary results"""
        # Create default output path if needed
        if output_path is None:
            base_dir = Path("data/backtest_results/summary")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = base_dir / f"backtest_summary_{summary['timeframe']}_{timestamp}.png"
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Returns comparison
        returns = [summary['strategy_return'], summary['buy_hold_return']]
        # Check if returns are valid numbers
        if all(not np.isnan(r) for r in returns):
            axes[0, 0].bar(['Strategy', 'Buy & Hold'], returns, color=['purple', 'blue'])
            axes[0, 0].set_title('Returns Comparison')
            axes[0, 0].set_ylabel('Return (%)')
            
            # Add text labels with percentages
            for i, v in enumerate(returns):
                axes[0, 0].text(i, v + 0.01 if v > 0 else v - 0.01, f'{v:.2%}', ha='center')
        else:
            axes[0, 0].text(0.5, 0.5, 'No return data available', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Returns Comparison (No Data)')
        
        # Plot 2: Win/Loss pie chart - only if we have valid trade data
        if summary['winning_trades'] > 0 or summary['losing_trades'] > 0:
            win_loss_data = [summary['winning_trades'], summary['losing_trades']]
            axes[0, 1].pie(
                win_loss_data, 
                labels=['Wins', 'Losses'],
                colors=['green', 'red'],
                autopct='%1.1f%%'
            )
            axes[0, 1].set_title(f'Win/Loss Distribution (Total: {sum(win_loss_data)})')
        else:
            axes[0, 1].text(0.5, 0.5, 'No trade data available', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Win/Loss Distribution (No Data)')
        
        # Plot 3: Position distribution
        positions = []
        pos_counts = []
        pos_colors = {'Short': 'red', 'Flat': 'gray', 'Long': 'green', 'Hedged': 'purple'}
        
        valid_distribution = False
        for pos, pct in summary['position_distribution'].items():
            if not np.isnan(pct) and pct > 0:
                valid_distribution = True
                pos_label = {-1: "Short", 0: "Flat", 1: "Long", 2: "Hedged"}.get(pos, str(pos))
                positions.append(pos_label)
                pos_counts.append(pct)
        
        if valid_distribution:
            axes[1, 0].bar(positions, pos_counts, color=[pos_colors.get(p, 'blue') for p in positions])
            axes[1, 0].set_title('Position Distribution')
            axes[1, 0].set_ylabel('Percentage')
            
            # Add text labels with percentages
            for i, v in enumerate(pos_counts):
                axes[1, 0].text(i, v + 0.01, f'{v:.2%}', ha='center')
        else:
            axes[1, 0].text(0.5, 0.5, 'No position data available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Position Distribution (No Data)')
        
        # Plot 4: Key metrics - only if we have valid metrics
        have_valid_metrics = (
            not np.isnan(summary['win_rate']) and 
            not np.isnan(summary['profit_factor']) and
            not np.isnan(summary['average_win']) and
            not np.isnan(summary['average_loss'])
        )
        
        if have_valid_metrics and summary['total_trades'] > 0:
            metrics = ['Win Rate', 'Profit Factor', 'Avg Win', 'Avg Loss']
            values = [
                summary['win_rate'], 
                min(summary['profit_factor'], 5) / 5,  # Scale profit factor to 0-1 range, cap at 5
                summary['average_win'],
                abs(summary['average_loss'])
            ]
            
            # Ensure no NaN values
            if not any(np.isnan(v) for v in values):
                axes[1, 1].bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
                axes[1, 1].set_title('Key Performance Metrics')
                
                # Add custom y-axis labels for the metrics
                ax2 = axes[1, 1].twinx()
                ax2.set_ylim(axes[1, 1].get_ylim())
                ax2.set_yticks([v for v in values])
                ax2.set_yticklabels([
                    f'{summary["win_rate"]:.2%}',
                    f'{min(summary["profit_factor"], 5):.2f}',
                    f'{summary["average_win"]:.2%}',
                    f'{abs(summary["average_loss"]):.2%}'
                ])
            else:
                axes[1, 1].text(0.5, 0.5, 'Some metrics contain NaN values', 
                            ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Key Performance Metrics (Invalid Data)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No trade metrics available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Key Performance Metrics (No Data)')
        
        # Set title and adjust layout
        fig.suptitle(f'Backtest Summary: {summary["timeframe"]} ({summary["data_source"]})', fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save figure
        try:
            plt.savefig(output_path)
            self.logger.info(f"Summary plot saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving plot: {str(e)}")
        finally:
            plt.close(fig)
        
        return output_path

def main():
    """Command-line interface for backtest analysis"""
    parser = argparse.ArgumentParser(description='Analyze backtest results')
    parser.add_argument('--file', type=str, help='Path to backtest results file')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--symbols', type=str, default='BTC/USDT', help='Trading pair symbol')
    parser.add_argument('--timeframes', type=str, help='Timeframe to analyze')
    parser.add_argument('--output', type=str, help='Output path for summary')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()
    
    analyzer = BacktestAnalyzer()
    
    # Determine file to analyze
    if args.file:
        file_path = Path(args.file)
    else:
        file_path = analyzer.find_latest_backtest(args.exchange, args.symbols, args.timeframes)
    
    if not file_path or not file_path.exists():
        print(f"Error: Could not find backtest file to analyze")
        return 1
    
    # Analyze the file
    try:
        summary = analyzer.analyze_file(file_path)
        
        # Print summary
        analyzer.print_summary(summary)
        
        # Save summary
        if args.output:
            analyzer.save_summary(summary, args.output)
        else:
            analyzer.save_summary(summary)
        
        # Create plot
        if not args.no_plot:
            try:
                analyzer.plot_summary(summary)
            except Exception as e:
                print(f"Warning: Could not create summary plot: {str(e)}")
                logging.error(f"Error creating plot: {str(e)}")
    except Exception as e:
        print(f"Error analyzing backtest file: {str(e)}")
        logging.error(f"Error analyzing backtest file: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())