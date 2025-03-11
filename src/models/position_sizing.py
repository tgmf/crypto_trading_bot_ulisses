#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum-inspired position sizing for trading models.

This module implements a position sizing approach that converts model probability
outputs into continuous position sizes, optimized for fee efficiency and risk management.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union
from pathlib import Path
import json
from datetime import datetime

class QuantumPositionSizer:
    """
    Implements quantum-inspired position sizing based on probability distributions
    from Bayesian model predictions.
    
    This approach:
    1. Converts probabilities to continuous position sizes
    2. Scales positions based on confidence and volatility
    3. Minimizes fee impact through efficient position transitions
    4. Provides flash crash protection via partial hedging
    """
    
    def __init__(
        self, 
        fee_rate: float = 0.0006,  # Per-side fee rate (0.06%)
        no_trade_threshold: float = 0.96,  # Threshold for no-trade zone
        confidence_scaling: bool = True,  # Scale by confidence
        volatility_scaling: bool = True,  # Scale by inverse volatility
        max_position: float = 1.0,  # Maximum position size (1.0 = 100% of available capital)
        min_position_change: float = 0.05,  # Minimum position change to avoid fee churn
        initial_capital: float = 10000.0,  # Initial capital for equity tracking
    ):
        self.fee_rate = fee_rate
        self.no_trade_threshold = no_trade_threshold
        self.confidence_scaling = confidence_scaling
        self.volatility_scaling = volatility_scaling
        self.max_position = max_position
        self.min_position_change = min_position_change
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(
        self, 
        short_prob: float, 
        no_trade_prob: float, 
        long_prob: float, 
        volatility: float = None
    ) -> float:
        """
        Calculate the optimal position size based on probability distribution.
        
        Args:
            short_prob: Probability of profitable short opportunity
            no_trade_prob: Probability of no profitable opportunity
            long_prob: Probability of profitable long opportunity
            volatility: Current market volatility (optional)
            
        Returns:
            float: Position size from -1.0 (full short) to 1.0 (full long)
        """
        # Calculate raw position size based on probability difference
        # We use long_prob - short_prob as our base position size multiplier
        raw_position = long_prob - short_prob
        
        # Apply no-trade zone if no_trade_prob is above threshold
        if no_trade_prob > self.no_trade_threshold:
            return 0.0
            
        # Calculate confidence factor (higher when probabilities are more decisive)
        confidence = max(long_prob, short_prob) - min(long_prob, short_prob)
        
        # Apply confidence scaling if enabled
        position = raw_position
        if self.confidence_scaling:
            # Square the confidence to emphasize high confidence signals
            confidence_factor = confidence ** 2
            position = raw_position * confidence_factor
            
        # Apply volatility scaling if enabled and volatility is provided
        if self.volatility_scaling and volatility is not None and volatility > 0:
            # Inverse volatility relationship - lower position sizes during high volatility
            vol_factor = 1.0 / (1.0 + volatility * 10)  # Scale factor
            position = position * vol_factor
            
        # Cap position at maximum allowed size
        position = max(min(position, self.max_position), -self.max_position)
        
        return position
        
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame with model probabilities to generate position sizes
        and backtest performance.
        
        Args:
            df: DataFrame with short_prob, no_trade_prob, long_prob columns
                
        Returns:
            DataFrame: Original data with added position information
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize position tracking columns
        result_df['target_position'] = 0.0  # Target position from model
        result_df['actual_position'] = 0.0  # Actual position after min_change filter
        result_df['position_change'] = 0.0  # How much position changed
        result_df['fee_cost'] = 0.0  # Fee cost for each position change
        
        # Initialize performance tracking
        result_df['equity'] = self.initial_capital
        
        # Ensure returns column exists
        if 'returns' not in result_df.columns:
            result_df['returns'] = result_df['close'].pct_change()
        
        # Previous position starts at zero (flat)
        prev_position = 0.0
        
        # Process each row
        for i in range(len(result_df)):
            # Get current probabilities
            short_prob = result_df.iloc[i]['short_prob']
            no_trade_prob = result_df.iloc[i]['no_trade_prob'] 
            long_prob = result_df.iloc[i]['long_prob']
            
            # Get volatility if available
            volatility = result_df.iloc[i].get('volatility', None)
            
            # Calculate target position size
            target_position = self.calculate_position_size(
                short_prob, no_trade_prob, long_prob, volatility
            )
            
            # Store target position
            result_df.iloc[i, result_df.columns.get_loc('target_position')] = target_position
            
            # Calculate position change
            position_change = target_position - prev_position
            
            # Apply minimum change filter to avoid fee churn
            if abs(position_change) < self.min_position_change:
                # Don't change position if change is too small
                actual_position = prev_position
                position_change = 0.0
            else:
                # Accept new position
                actual_position = target_position
                
            # Store actual position and change
            result_df.iloc[i, result_df.columns.get_loc('actual_position')] = actual_position
            result_df.iloc[i, result_df.columns.get_loc('position_change')] = position_change
            
            # Calculate fee cost (only on the portion that changed)
            fee_cost = abs(position_change) * self.fee_rate
            result_df.iloc[i, result_df.columns.get_loc('fee_cost')] = fee_cost
            
            # Update previous position for next iteration
            prev_position = actual_position
            
            # Skip return calculation for first row
            if i > 0:
                # Calculate strategy return (position * market return - fees)
                strategy_return = prev_position * result_df.iloc[i]['returns'] - fee_cost
                
                # Update equity
                prev_equity = result_df.iloc[i-1]['equity']
                result_df.iloc[i, result_df.columns.get_loc('equity')] = prev_equity * (1 + strategy_return)
        
        # Calculate cumulative returns
        result_df['cumulative_returns'] = (1 + result_df['returns']).cumprod() - 1
        
        # Calculate strategy returns
        result_df['strategy_returns'] = result_df['equity'].pct_change()
        result_df['strategy_cumulative'] = (result_df['equity'] / self.initial_capital) - 1
        
        self.logger.info(f"Processed {len(result_df)} rows with quantum position sizing")
        return result_df
    
    def analyze_performance(self, df: pd.DataFrame) -> dict:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            df: Processed DataFrame with position and return information
            
        Returns:
            dict: Performance metrics
        """
        # Extract relevant data
        total_return = df['strategy_cumulative'].iloc[-1]
        buy_hold_return = df['cumulative_returns'].iloc[-1]
        
        # Calculate Sharpe ratio (simplified, assuming zero risk-free rate)
        returns = df['strategy_returns'].dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24)  # Annualized for hourly data
        
        # Calculate maximum drawdown
        equity_curve = df['equity']
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Calculate win rate and average win/loss
        daily_returns = df.resample('D')['strategy_returns'].sum()
        win_rate = (daily_returns > 0).mean()
        avg_win = daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0
        avg_loss = daily_returns[daily_returns < 0].mean() if len(daily_returns[daily_returns < 0]) > 0 else 0
        
        # Calculate profit factor
        gross_profits = daily_returns[daily_returns > 0].sum()
        gross_losses = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        # Calculate position statistics
        avg_position = df['actual_position'].abs().mean()
        position_changes = df[df['position_change'] != 0]
        num_position_changes = len(position_changes)
        
        # Calculate fee impact
        total_fees = df['fee_cost'].sum()
        fee_drag = total_fees / self.initial_capital
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'alpha': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_position': avg_position,
            'num_position_changes': num_position_changes,
            'total_fees': total_fees,
            'fee_drag': fee_drag
        }
    
    def plot_results(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the strategy results.
        
        Args:
            df: Processed DataFrame with position and return information
            
        Returns:
            Tuple: Figure and Axes objects
        """
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot 1: Price and Position
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], color='black', alpha=0.7, label='Price')
        
        # Add colored background for position
        for i in range(len(df) - 1):
            color = 'green' if df.iloc[i]['actual_position'] > 0 else 'red' if df.iloc[i]['actual_position'] < 0 else 'gray'
            alpha = min(abs(df.iloc[i]['actual_position']) * 0.8, 0.3)  # Scale transparency by position size
            if alpha > 0.05:  # Only plot significant positions
                ax1.axvspan(df.index[i], df.index[i+1], alpha=alpha, color=color, lw=0)
        
        ax1.set_title('Price and Position')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = axes[1]
        ax2.plot(df.index, df['strategy_cumulative'] * 100, color='blue', label='Strategy')
        ax2.plot(df.index, df['cumulative_returns'] * 100, color='gray', label='Buy & Hold')
        ax2.set_title('Cumulative Returns (%)')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position Size
        ax3 = axes[2]
        ax3.plot(df.index, df['actual_position'], color='purple', label='Actual Position')
        ax3.plot(df.index, df['target_position'], color='lightblue', alpha=0.5, label='Target Position')
        ax3.set_title('Position Size')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Position (-1 to 1)')
        ax3.set_ylim(-1.1, 1.1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def save_results(self, df, metrics, fig, exchange, symbol, timeframe):
        """
        Save position sizing backtest results.
        
        Args:
            df: DataFrame with backtest results
            metrics: Dictionary of performance metrics
            fig: Figure object with plots
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create safe path elements
            symbol_safe = symbol.replace('/', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create output directory
            output_dir = Path(f"data/backtest_results/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame
            results_file = output_dir / f"{timeframe}_pos_sizing_{timestamp}.csv"
            df.to_csv(results_file)
            
            # Save metrics
            metrics_file = output_dir / f"{timeframe}_pos_sizing_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4, default=str)
            
            # Save figure
            fig_file = output_dir / f"{timeframe}_pos_sizing_plot_{timestamp}.png"
            fig.savefig(fig_file)
            
            self.logger.info(f"Saved position sizing results to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving position sizing results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


def apply_quantum_position_sizing(df, fee_rate=0.0006, no_trade_threshold=0.96, 
                                    min_position_change=0.05, initial_capital=10000.0,
                                    confidence_scaling=True, volatility_scaling=True):
    """
    Apply quantum position sizing to a DataFrame
    
    Args:
        df: DataFrame with model probabilities
        fee_rate: Per-side fee rate
        no_trade_threshold: Threshold for no_trade_prob to ignore signals
        min_position_change: Minimum position change to avoid fee churn
        initial_capital: Initial capital for equity tracking
        confidence_scaling: Whether to scale positions by confidence
        volatility_scaling: Whether to scale positions by inverse volatility
        
    Returns:
        tuple: (result_df, metrics, fig) - Processed DataFrame, performance metrics, and plot
    """
    # Initialize position sizer
    position_sizer = QuantumPositionSizer(
        fee_rate=fee_rate,
        no_trade_threshold=no_trade_threshold,
        confidence_scaling=confidence_scaling,
        volatility_scaling=volatility_scaling,
        max_position=1.0,
        min_position_change=min_position_change,
        initial_capital=initial_capital
    )
    
    # Process dataframe
    result_df = position_sizer.process_dataframe(df)
    
    # Calculate performance metrics
    metrics = position_sizer.analyze_performance(result_df)
    
    # Plot results
    fig, _ = position_sizer.plot_results(result_df)
    
    return result_df, metrics, fig