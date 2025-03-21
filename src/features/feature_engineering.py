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

class FeatureEngineer:
    """Creates and transforms features for ML models"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.debug("FeatureEngineer initialized")
        
    def process_data(self):
        """Process raw data and generate features"""
        symbols = self.config.get('data', {}).get('symbols', [])
        timeframes = self.config.get('data', {}).get('timeframes', [])
        exchanges = self.config.get('data', {}).get('exchanges', [])
        
        for exchange in exchanges:
            for symbol in symbols:
                for timeframe in timeframes:
                    self.logger.info(f"Processing features for {exchange} {symbol} {timeframe}")
                    
                    try:
                        # Load raw data
                        symbol_safe = symbol.replace('/', '_')
                        input_file = Path(f"data/raw/{exchange}/{symbol_safe}/{timeframe}.csv")
                        
                        if not input_file.exists():
                            self.logger.warning(f"No data file found at {input_file}")
                            continue
                            
                        df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                        
                        # Generate features
                        df_featured = self._generate_features(df)
                        
                        # Save processed data
                        output_dir = Path(f"data/processed/{exchange}/{symbol_safe}")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_file = output_dir / f"{timeframe}.csv"
                        df_featured.to_csv(output_file)
                        
                        self.logger.info(f"Saved processed data to {output_file}")
                        
                    except Exception as e:
                        self.logger.error(f"FELOG. Error processing {symbol} {timeframe}: {str(e)}")
                        continue
    
    def _generate_features(self, df):
        """Generate technical indicators and features"""
        df = df.copy()
        
        # Price and returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # Volatility measures
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['range'] = (df['high'] - df['low']) / df['close']
        
        # Calculate ATR
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df = df.join(atr)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        df = df.join(bb)
        df['bb_pos'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        df['bb_squeeze'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
        
        # RSI
        rsi = ta.rsi(df['close'], length=14)
        df = df.join(rsi)
        df['rsi_diff'] = df['RSI_14'].diff()
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = df.join(macd)
        df['macd_hist_diff'] = df['MACDh_12_26_9'].diff()
        
        # Moving Averages
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['trend_strength'] = (df['ema9'] - df['ema50']) / df['ema50']
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df