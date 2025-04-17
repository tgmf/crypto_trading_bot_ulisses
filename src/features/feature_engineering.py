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
from ..data.data_context import DataContext

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
                        
                        # Load raw data into pandas DataFrame    
                        df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                    
                        # Create DataContext from raw data
                        data_context = DataContext(self.params, df, exchange, symbol, timeframe, source="raw")
                        
                        # Check if we have enough data
                        min_rows = self.params.get('data', 'min_rows', default=100)
                        if not data_context.validate(required_columns=['open', 'high', 'low', 'close', 'volume'], 
                                                        min_rows=min_rows):
                            continue
                        
                        # Generate features with DataContext tracking
                        self._generate_features(data_context)
                        
                        # Final data validation after feature creation
                        feature_columns = self.params.get('model', 'feature_cols', default=[])
                        required_columns = ['open', 'high', 'low', 'close', 'volume'] + feature_columns
                        
                        if not data_context.validate(required_columns=required_columns, min_rows=min_rows):
                            self.logger.error(f"Feature generation did not produce all required columns for {symbol} {timeframe}")
                            error_count += 1
                            continue
                        
                        # Save processed data
                        output_dir = Path(f"data/processed/{exchange}/{symbol_safe}")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_file = output_dir / f"{timeframe}.csv"
                        data_context.df.to_csv(output_file)
                        data_context.add_processing_step("save_processed", {"path": str(output_file)})
                        
                        self.logger.info(f"Saved processed data to {output_file} ({len(data_context.df)} rows)")
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
        Generate technical indicators and features using modular approach
        
        Args:
            df (DataFrame): DataFrame with raw price data
            
        Returns:
            DataFrame: Data with generated features
        """
        try:
            df = df.copy()
            features_generated = []
            
            # Always generate base returns (required by many other features)
            df = self._generate_base_returns(df)
            features_generated.append('returns')
            
            # Dynamically check and generate enabled features from configuration
            for feature_group in self.params.get('model', 'features', default={}).keys():
                # Skip non-feature settings
                if feature_group in ['metadata']:
                    continue
                    
                # Check if feature group is enabled
                if self.params.get('model', 'features', feature_group, 'enabled', default=False):
                    # Look for corresponding method
                    method_name = f"_generate_{feature_group.lower()}"
                    
                    # Try to call method if it exists
                    if hasattr(self, method_name):
                        generator_method = getattr(self, method_name)
                        self.logger.debug(f"Generating built-in feature group: {feature_group}")
                        df = generator_method(df)
                        features_generated.append(feature_group)
                    else:
                        # Try to import dynamically from custom features directory first
                        custom_feature_found = False
                        
                        try:
                            # First check for a custom feature in the external /features directory
                            import sys
                            import importlib.util
                        
                            # Path to the custom features directory (top-level /features)
                            features_dir = Path(__file__).parents[2] / 'features'
                        
                            # Try to import the module
                            module_name = feature_group.lower()
                            module_path = features_dir / f"{module_name}.py"
                            
                            if module_path.exists():
                                self.logger.debug(f"Found custom feature module: {module_path}")
                                
                                # Add to path if needed
                                if str(features_dir) not in sys.path:
                                    sys.path.append(str(features_dir))
                                    
                                # Import the module
                                spec = importlib.util.spec_from_file_location(module_name, module_path)
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                
                                # Try different naming conventions for the generator class
                                class_names = [
                                    f"{feature_group.capitalize()}Generator",  # CustomGenerator
                                    f"{module_name.capitalize()}Generator",    # CustomgeneratorGenerator (fallback)
                                    f"{feature_group}Generator",               # customGenerator
                                    f"{feature_group.upper()}Generator"        # CUSTOMGenerator
                                ]
                                
                                generator_class = None
                                for class_name in class_names:
                                    if hasattr(module, class_name):
                                        generator_class = getattr(module, class_name)
                                        break
                                
                                if generator_class is not None:
                                    # Instantiate and generate features
                                    generator = generator_class(self.params)
                                    df = generator.generate(df)
                                    features_generated.append(f"custom:{feature_group}")
                                    custom_feature_found = True
                                    self.logger.info(f"Successfully used custom feature: {feature_group} from {module_path}")
                                else:
                                    self.logger.warning(f"Could not find generator class in {module_path}, tried: {', '.join(class_names)}")
                    
                        except Exception as e:
                            self.logger.debug(f"Error using custom feature {feature_group}: {str(e)}")
                        
                        # If no custom feature was found, try internal features as fallback
                        if not custom_feature_found:
                            try:
                                # Import from internal module
                                module_name = f"..features.{feature_group.lower()}"
                                module = __import__(module_name, fromlist=[''])
                                
                                # Get generator class and instantiate
                                generator_class = getattr(module, f"{feature_group.capitalize()}Generator")
                                generator = generator_class(self.params)
                                
                                # Generate features
                                df = generator.generate(df)
                                features_generated.append(feature_group)
                                
                            except (ImportError, AttributeError) as e:
                                self.logger.warning(f"Feature generator for '{feature_group}' not found: {str(e)}")
            
            # Generate metadata if enabled
            if self.params.get('model', 'features', 'metadata', 'enabled', default=False):
                self._generate_feature_metadata(df)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Log feature statistics
            self.logger.debug(f"Generated {len(features_generated)} feature groups, data shape: {df.shape}")
            
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
    
    # Basic feature generation methods
    def _generate_base_returns(self, df):
        """Base returns calculation (required for many features)"""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        return df

    def _generate_volatility(self, df):
        """Generate volatility-based features"""
        window = self.params.get('model', 'features', 'volatility', 'window', default=20)
        df['volatility'] = df['returns'].rolling(window=window).std()
        
        if self.params.get('model', 'features', 'volatility', 'range', default=True):
            df['range'] = (df['high'] - df['low']) / df['close']
        
        # Calculate ATR if enabled
        if self.params.get('model', 'features', 'volatility', 'atr', 'enabled', default=True):
            atr_length = self.params.get('model', 'features', 'volatility', 'atr', 'length', default=14)
            atr = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
            df = df.join(atr)
        
        return df

    def _generate_bollinger(self, df):
        """Generate Bollinger Bands features"""
        bb_length = self.params.get('model', 'features', 'bollinger', 'length', default=20)
        bb_std = self.params.get('model', 'features', 'bollinger', 'std', default=2.0)
        bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
        df = df.join(bb)
        
        if self.params.get('model', 'features', 'bollinger', 'include_position', default=True):
            df['bb_pos'] = (df['close'] - df[f'BBL_{bb_length}_{bb_std}']) / (df[f'BBU_{bb_length}_{bb_std}'] - df[f'BBL_{bb_length}_{bb_std}'])
        
        if self.params.get('model', 'features', 'bollinger', 'include_squeeze', default=True):
            df['bb_squeeze'] = (df[f'BBU_{bb_length}_{bb_std}'] - df[f'BBL_{bb_length}_{bb_std}']) / df[f'BBM_{bb_length}_{bb_std}']
        
        return df

    def _generate_rsi(self, df):
        """Generate RSI features"""
        rsi_length = self.params.get('model', 'features', 'rsi', 'length', default=14)
        rsi = ta.rsi(df['close'], length=rsi_length)
        df = df.join(rsi)
        
        if self.params.get('model', 'features', 'rsi', 'include_diff', default=False):
            df['rsi_diff'] = df[f'RSI_{rsi_length}'].diff()
        
        return df

    def _generate_macd(self, df):
        """Generate MACD features"""
        macd_fast = self.params.get('model', 'features', 'macd', 'fast', default=12)
        macd_slow = self.params.get('model', 'features', 'macd', 'slow', default=26)
        macd_signal = self.params.get('model', 'features', 'macd', 'signal', default=9)
        macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        df = df.join(macd)
        
        if self.params.get('model', 'features', 'macd', 'include_hist_diff', default=False):
            df['macd_hist_diff'] = df[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'].diff()
        
        return df

    def _generate_trend(self, df):
        """Generate trend-based features"""
        ema_short = self.params.get('model', 'features', 'trend', 'ema_short', default=9)
        ema_long = self.params.get('model', 'features', 'trend', 'ema_long', default=50)
        df[f'ema{ema_short}'] = ta.ema(df['close'], length=ema_short)
        df[f'ema{ema_long}'] = ta.ema(df['close'], length=ema_long)
        
        if self.params.get('model', 'features', 'trend', 'include_strength', default=True):
            df['trend_strength'] = (df[f'ema{ema_short}'] - df[f'ema{ema_long}']) / df[f'ema{ema_long}']
        
        return df

    def _generate_volume(self, df):
        """Generate volume-based features"""
        volume_window = self.params.get('model', 'features', 'volume', 'window', default=20)
        df['volume_ma'] = df['volume'].rolling(window=volume_window).mean()
        
        if self.params.get('model', 'features', 'volume', 'include_ratio', default=True):
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df

    def _generate_support_resistance(self, df):
        """Generate support and resistance features"""
        # Use single or multiple lookback periods based on params
        lookbacks = self.params.get('model', 'features', 'support_resistance', 'lookbacks', default=[50])
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
                if self.params.get('model', 'features', 'volatility', 'atr', 'enabled', default=True):
                    atr_length = self.params.get('model', 'features', 'volatility', 'atr', 'length', default=14)
                else:
                    atr_length = 14
                    
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
            if self.params.get('model', 'features', 'support_resistance', 'include_breakouts', default=True):
                df[f'resistance_break_{lb}'] = (df['close'] > df[f'resistance_{lb}'].shift(1)).astype(float)
                df[f'support_break_{lb}'] = (df['close'] < df[f'support_{lb}'].shift(1)).astype(float)
        
        return df

    def _generate_advanced(self, df):
        """Generate advanced features"""
        if self.params.get('model', 'features', 'advanced', 'include_ichimoku', default=False):
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            df = df.join(ichimoku[0])  # Only join the indicators, not the signals
        
        if self.params.get('model', 'features', 'advanced', 'include_donchian', default=False):
            donchian_length = self.params.get('model', 'features', 'advanced', 'donchian_length', default=20)
            donchian = ta.donchian(df['high'], df['low'], lower_length=donchian_length, upper_length=donchian_length)
            df = df.join(donchian)
            df['donchian_pos'] = (df['close'] - df[f'DCL_{donchian_length}_{donchian_length}']) / (df[f'DCU_{donchian_length}_{donchian_length}'] - df[f'DCL_{donchian_length}_{donchian_length}'])
        
        if self.params.get('model', 'features', 'advanced', 'include_stoch', default=False):
            stoch_k = self.params.get('model', 'features', 'advanced', 'stoch_k', default=14)
            stoch_d = self.params.get('model', 'features', 'advanced', 'stoch_d', default=3)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=stoch_k, d=stoch_d)
            df = df.join(stoch)
        
        if self.params.get('model', 'features', 'advanced', 'include_momentum', default=False):
            momentum_period = self.params.get('model', 'features', 'advanced', 'momentum_period', default=10)
            df['momentum'] = df['close'].diff(momentum_period)
        
        return df