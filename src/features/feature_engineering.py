#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering for cryptocurrency trading models.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pandas_ta as ta
import traceback
from ..core.param_manager import ParamManager

class FeatureEngineer:
    """Creates and transforms features for ML models"""
    
    def __init__(self, params):
        """
        Initialize with parameter manager
        
        Args:
            params: ParamManager instance
        """
        self.params = params
        self.logger = logging.getLogger(__name__)
        self.logger.debug("FeatureEngineer initialized")
        
    def process_data(self):
        """
        Process raw data and generate features
        
        Reads raw data from data/raw/{exchange}/{symbol}/{timeframe}.csv
        Calculates technical indicators and derived features
        Saves processed data to data/processed/{exchange}/{symbol}/{timeframe}.csv
        
        Returns:
            bool: True if successful, False otherwise
        """
        symbols = self.params.get('data', 'symbols', default=[])
        timeframes = self.params.get('data', 'timeframes', default=[])
        exchanges = self.params.get('data', 'exchanges', default=[])
        
        success_count = 0
        error_count = 0
        
        for exchange in exchanges:
            for symbol in symbols:
                for timeframe in timeframes:
                    self.logger.info(f"Processing features for {symbol} {timeframe}")
                    
                    try:
                        # Load raw data
                        symbol_safe = symbol.replace('/', '_')
                        input_file = Path(f"data/raw/{exchange}/{symbol_safe}/{timeframe}.csv")
                        
                        if not input_file.exists():
                            self.logger.warning(f"No data file found at {input_file}")
                            continue
                            
                        df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                        
                        # Check if we have enough data
                        min_rows = self.params.get('data', 'min_rows', default=100)
                        if len(df) < min_rows:
                            self.logger.warning(f"Not enough data for {symbol} {timeframe} ({len(df)} rows < {min_rows}) at {exchange}")
                            continue
                        
                        # Generate features
                        df_featured = self._generate_features(df)
                        
                        # Quality check
                        if df_featured is None or len(df_featured) == 0:
                            self.logger.error(f"Feature generation returned empty DataFrame for {symbol} {timeframe} at {exchange}")
                            error_count += 1
                            continue
                        
                        # Save processed data
                        output_dir = Path(f"data/processed/{exchange}/{symbol_safe}")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_file = output_dir / f"{timeframe}.csv"
                        df_featured.to_csv(output_file)
                        
                        self.logger.info(f"Saved processed data to {output_file} ({len(df_featured)} rows)")
                        success_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        error_count += 1
                        continue
        
        self.logger.info(f"Feature processing complete: {success_count} successful, {error_count} errors")
        return success_count > 0
    
    def _generate_features(self, df):
        """
        Generate technical indicators and features
        
        Args:
            df (DataFrame): Raw OHLCV data
            
        Returns:
            DataFrame: Data with generated features
        """
        try:
            df = df.copy()
            
            # Get feature configuration
            include_returns = self.params.get('model', 'features', 'include_returns', default=True)
            include_volatility = self.params.get('model', 'features', 'include_volatility', default=True)
            include_bb = self.params.get('model', 'features', 'include_bb', default=True)
            include_rsi = self.params.get('model', 'features', 'include_rsi', default=True)
            include_macd = self.params.get('model', 'features', 'include_macd', default=True)
            include_trend = self.params.get('model', 'features', 'include_trend', default=True)
            include_volume = self.params.get('model', 'features', 'include_volume', default=True)
            
            # Price and returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']).diff()
            
            # Volatility measures
            if include_volatility:
                window = self.params.get('model', 'features', 'volatility_window', default=20)
                df['volatility'] = df['returns'].rolling(window=window).std()
                df['range'] = (df['high'] - df['low']) / df['close']
                
                # Calculate ATR
                atr_length = self.params.get('model', 'features', 'atr_length', default=14)
                atr = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
                df = df.join(atr)
            
            # Bollinger Bands
            if include_bb:
                bb_length = self.params.get('model', 'features', 'bb_length', default=20)
                bb_std = self.params.get('model', 'features', 'bb_std', default=2.0)
                bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
                df = df.join(bb)
                df['bb_pos'] = (df['close'] - df[f'BBL_{bb_length}_{bb_std}']) / (df[f'BBU_{bb_length}_{bb_std}'] - df[f'BBL_{bb_length}_{bb_std}'])
                df['bb_squeeze'] = (df[f'BBU_{bb_length}_{bb_std}'] - df[f'BBL_{bb_length}_{bb_std}']) / df[f'BBM_{bb_length}_{bb_std}']
            
            # RSI
            if include_rsi:
                rsi_length = self.params.get('model', 'features', 'rsi_length', default=14)
                rsi = ta.rsi(df['close'], length=rsi_length)
                df = df.join(rsi)
                df['rsi_diff'] = df[f'RSI_{rsi_length}'].diff()
            
            # MACD
            if include_macd:
                macd_fast = self.params.get('model', 'features', 'macd_fast', default=12)
                macd_slow = self.params.get('model', 'features', 'macd_slow', default=26)
                macd_signal = self.params.get('model', 'features', 'macd_signal', default=9)
                macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                df = df.join(macd)
                df['macd_hist_diff'] = df[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'].diff()
            
            # Moving Averages and Trend
            if include_trend:
                ema_short = self.params.get('model', 'features', 'ema_short', default=9)
                ema_long = self.params.get('model', 'features', 'ema_long', default=50)
                df[f'ema{ema_short}'] = ta.ema(df['close'], length=ema_short)
                df[f'ema{ema_long}'] = ta.ema(df['close'], length=ema_long)
                df['trend_strength'] = (df[f'ema{ema_short}'] - df[f'ema{ema_long}']) / df[f'ema{ema_long}']
            
            # Volume indicators
            if include_volume:
                volume_window = self.params.get('model', 'features', 'volume_window', default=20)
                df['volume_ma'] = df['volume'].rolling(window=volume_window).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Optional advanced features
            if self.params.get('model', 'features', 'include_advanced', default=False):
                self._add_advanced_features(df)
            
            # Generate feature metadata if requested
            if self.params.get('model', 'features', 'generate_metadata', default=False):
                self._generate_feature_metadata(df)
                
            # Support/Resistance features
            if self.params.get('model', 'features', 'include_sr', default=False):
                # Use single or multiple lookback periods based on params
                lookbacks = self.params.get('model', 'features', 'sr_lookbacks', default=[50])
                if not isinstance(lookbacks, list):
                    lookbacks = [lookbacks]  # Convert single value to list
                
                # Add minimal set of S/R features
                for lb in lookbacks:
                    # Calculate resistance and support levels
                    df[f'resistance_{lb}'] = df['high'].rolling(lb).max()
                    df[f'support_{lb}'] = df['low'].rolling(lb).min()
                    
                    # Distance to levels (normalized by ATR)
                    # If ATR already calculated, use it, otherwise calculate it
                    atr_col = 'ATR_14' if 'ATR_14' in df.columns else None
                    if atr_col is None:
                        # Calculate ATR if not already present
                        atr_length = self.params.get('model', 'features', 'atr_length', default=14)
                        if f'ATR_{atr_length}' not in df.columns:
                            atr = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
                            if isinstance(atr, pd.Series):
                                atr.name = f'ATR_{atr_length}'
                                df[f'ATR_{atr_length}'] = atr
                            else:
                                # If it returns a DataFrame, merge it carefully
                                for col in atr.columns:
                                    new_col_name = col.replace('ATRr_', 'ATR_')
                                    df[new_col_name] = atr[col]
                        atr_col = f'ATR_{atr_length}'
                    
                    # Calculate normalized distances
                    df[f'dist_to_resistance_{lb}'] = (df[f'resistance_{lb}'] - df['close']) / df[atr_col]
                    df[f'dist_to_support_{lb}'] = (df['close'] - df[f'support_{lb}']) / df[atr_col]
                    
                    # Optional breakout signals
                    if self.params.get('model', 'features', 'include_sr_breakouts', default=True):
                        df[f'resistance_break_{lb}'] = (df['close'] > df[f'resistance_{lb}'].shift(1)).astype(float)
                        df[f'support_break_{lb}'] = (df['close'] < df[f'support_{lb}'].shift(1)).astype(float)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Log feature statistics
            self.logger.debug(f"Generated {len(df.columns)} features, data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
        
    def _add_advanced_features(self, df):
        """
        Add advanced features to the dataframe
        
        Args:
            df (DataFrame): Dataframe with basic features
        """
        try:
            # Ichimoku Cloud
            if self.params.get('features', 'include_ichimoku', default=False):
                ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
                df = df.join(ichimoku[0])  # Only join the indicators, not the signals
            
            # Donchian Channels
            if self.params.get('features', 'include_donchian', default=False):
                donchian_length = self.params.get('features', 'donchian_length', default=20)
                donchian = ta.donchian(df['high'], df['low'], lower_length=donchian_length, upper_length=donchian_length)
                df = df.join(donchian)
                df['donchian_pos'] = (df['close'] - df[f'DCL_{donchian_length}_{donchian_length}']) / (df[f'DCU_{donchian_length}_{donchian_length}'] - df[f'DCL_{donchian_length}_{donchian_length}'])
            
            # Stochastic Oscillator
            if self.params.get('features', 'include_stoch', default=False):
                stoch_k = self.params.get('features', 'stoch_k', default=14)
                stoch_d = self.params.get('features', 'stoch_d', default=3)
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=stoch_k, d=stoch_d)
                df = df.join(stoch)
            
            # Price momentum
            if self.params.get('features', 'include_momentum', default=False):
                momentum_period = self.params.get('features', 'momentum_period', default=10)
                df['momentum'] = df['close'].diff(momentum_period)
                
        except Exception as e:
            self.logger.warning(f"Error adding advanced features: {str(e)}")
            # Continue without advanced features
    
    def _generate_feature_metadata(self, df):
        """
        Generate metadata about features
        
        This collects statistics about each feature and saves them
        for later use in model interpretation and feature selection.
        
        Args:
            df (DataFrame): DataFrame with features
        """
        try:
            # Skip non-feature columns
            skip_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df.columns if col not in skip_cols]
            
            # Calculate feature statistics
            stats = {}
            for col in feature_cols:
                # Calculate basic statistics
                mean = df[col].mean()
                std = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Calculate correlation with price and returns
                if 'returns' in df.columns:
                    corr_returns = df[col].corr(df['returns'])
                else:
                    corr_returns = None
                
                corr_close = df[col].corr(df['close'])
                
                # Store statistics
                stats[col] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'corr_returns': corr_returns,
                    'corr_close': corr_close
                }
            
            # Get symbols, timeframes for metadata storage path
            symbol = self.params.get('data', 'symbols', 0)
            timeframe = self.params.get('data', 'timeframes', 0)
            exchange = self.params.get('data', 'exchanges', 0)
            
            # Create metadata directory
            symbol_safe = symbol.replace('/', '_')
            metadata_dir = Path(f"data/metadata/{exchange}/{symbol_safe}")
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metadata to JSON
            import json
            metadata_file = metadata_dir / f"{timeframe}_feature_metadata.json"
            
            # Convert non-serializable numpy types to Python types
            serializable_stats = {}
            for col, col_stats in stats.items():
                serializable_stats[col] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in col_stats.items()
                }
            
            with open(metadata_file, 'w') as f:
                json.dump(serializable_stats, f, indent=4)
                
            self.logger.info(f"Saved feature metadata to {metadata_file}")
            
        except Exception as e:
            self.logger.warning(f"Error generating feature metadata: {str(e)}")
            # Continue without metadata