#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data collection module for cryptocurrency markets.
"""

import logging  # For logging messages
import os  # For interacting with the operating system
import time  # For time-related functions
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import ccxt  # For interacting with cryptocurrency exchanges
import requests  # For making HTTP requests
import yfinance as yf  # For fetching financial data from Yahoo Finance
from datetime import datetime, timedelta, date  # For date and time manipulation
from pathlib import Path  # For handling file system paths

class DataCollector:
    """Collects historical and real-time data from cryptocurrency exchanges and alternative sources"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config  # Store the configuration
        self.logger = logging.getLogger(__name__)  # Create a logger for this class
        self.exchanges = {}  # Dictionary to store exchange connections
        self._initialize_exchanges()  # Initialize exchange connections
        
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        exchange_configs = self.config.get('data', {}).get('exchanges', [])  # Get exchange configurations from the config
        
        for exchange_name in exchange_configs:
            try:
                # Initialize without authentication for historical data
                exchange_class = getattr(ccxt, exchange_name)  # Get the exchange class from ccxt
                self.exchanges[exchange_name] = exchange_class({
                    'enableRateLimit': True,  # Enable rate limiting
                })
                self.logger.info(f"Initialized {exchange_name} connection")  # Log successful initialization
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {str(e)}")  # Log any errors during initialization
    
    def collect_data(self):
        """Collect historical data for configured symbols and timeframes"""
        symbols = self.config.get('data', {}).get('symbols', [])  # Get symbols from the config
        timeframes = self.config.get('data', {}).get('timeframes', [])  # Get timeframes from the config
        start_date = self.config.get('data', {}).get('start_date', '2020-01-01')  # Get start date from the config
        alt_sources = self.config.get('data', {}).get('alternative_sources', [])  # Get alternative data sources from the config
        
        # Collect from exchanges
        self._collect_from_exchanges(symbols, timeframes, start_date)
        
        # Collect from alternative sources if configured
        if 'yfinance' in alt_sources:
            # Filter out 4h timeframe which isn't supported by yfinance
            yf_timeframes = [tf for tf in timeframes if tf != '4h']
            self._collect_from_yfinance(symbols, yf_timeframes, start_date)
            
        if 'cryptocompare' in alt_sources:
            self._collect_from_cryptocompare(symbols, timeframes, start_date)
    
    def _collect_from_exchanges(self, symbols, timeframes, start_date):
        """Collect data from cryptocurrency exchanges"""
        # Convert start_date to timestamp, handling both string and date object
        if isinstance(start_date, str):
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        elif isinstance(start_date, date):
            # If it's already a date object, convert directly to timestamp
            since = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        else:
            # Handle other unexpected types
            self.logger.error(f"Unexpected type for start_date: {type(start_date)}")
            return
        
        for exchange_name, exchange in self.exchanges.items():
            for symbol in symbols:
                for timeframe in timeframes:
                    self.logger.info(f"Collecting {symbol} {timeframe} data from {exchange_name}")
                    
                    try:
                        # Create directory structure
                        output_dir = Path(f"data/raw/{exchange_name}/{symbol.replace('/', '_')}")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Fetch OHLCV data
                        ohlcv = []
                        current_since = since
                        
                        # Paginate requests to get complete history
                        while True:
                            self.logger.info(f"Fetching data from {datetime.fromtimestamp(current_since/1000)}")
                            
                            # Fetch data
                            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                            
                            if not candles or len(candles) == 0:
                                break
                                
                            ohlcv.extend(candles)  # Append fetched data to the list
                            
                            # Update timestamp for next batch
                            current_since = candles[-1][0] + 1
                            
                            # Basic rate limiting
                            time.sleep(exchange.rateLimit / 1000)
                            
                            # If we've reached current time, stop
                            if current_since > int(time.time() * 1000):
                                break
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Save to CSV
                        output_file = output_dir / f"{timeframe}.csv"
                        df.to_csv(output_file)
                        self.logger.info(f"Saved {len(df)} records to {output_file}")
                        
                    except Exception as e:
                        self.logger.error(f"Error collecting {symbol} {timeframe} from {exchange_name}: {str(e)}")
                        continue
    
    def _collect_from_yfinance(self, symbols, timeframes, start_date):
        """Collect data from Yahoo Finance with appropriate timeframe restrictions"""
        self.logger.info("Collecting data from yfinance")
        
        # Map timeframes to yfinance intervals and their maximum lookback periods
        timeframe_map = {
            '1m': {'interval': '1m', 'days_back': 7},
            '5m': {'interval': '5m', 'days_back': 60},
            '15m': {'interval': '15m', 'days_back': 60},
            '1h': {'interval': '1h', 'days_back': 730},
            '1d': {'interval': '1d', 'days_back': 9999}  # Effectively no limit
        }
        
        # Map crypto symbols to yfinance format
        symbol_map = {
            'BTC/USD': 'BTC-USD',
            'ETH/USD': 'ETH-USD',
            'BTC/USDT': 'BTC-USDT',
            'ETH/USDT': 'ETH-USDT',
            # Add more mappings as needed
        }
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Skip unsupported timeframes
                    if timeframe not in timeframe_map:
                        self.logger.warning(f"Timeframe {timeframe} not supported by yfinance, skipping")
                        continue
                    
                    # Get symbol in yfinance format
                    yf_symbol = symbol_map.get(symbol, symbol.replace('/', '-'))
                    tf_config = timeframe_map[timeframe]
                    yf_interval = tf_config['interval']
                    
                    # Calculate appropriate start date based on timeframe limitations
                    days_back = tf_config['days_back']
                    adjusted_start = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                    self.logger.info(f"Using start date {adjusted_start} for {timeframe} data (limit: {days_back} days)")
                    
                    # Fetch data
                    self.logger.info(f"Fetching {yf_symbol} {yf_interval} data from yfinance")
                    df = yf.download(yf_symbol, start=adjusted_start, interval=yf_interval)
                    
                    if df.empty:
                        self.logger.warning(f"No data returned for {symbol} {timeframe}")
                        continue
                    
                    # Rename columns to match our expected format
                    df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)
                    
                    # Save to file
                    output_dir = Path(f"data/raw/yfinance/{symbol.replace('/', '_')}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / f"{timeframe}.csv"
                    df.to_csv(output_file)
                    self.logger.info(f"Saved {len(df)} records to {output_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error collecting {symbol} {timeframe} from yfinance: {str(e)}")
    
    def _collect_from_cryptocompare(self, symbols, timeframes, start_date):
        """Collect data from CryptoCompare API"""
        self.logger.info("Collecting data from CryptoCompare")
        
        # Map timeframes to CryptoCompare format
        timeframe_map = {
            '1m': 'minute',
            '5m': 'minute',
            '15m': 'minute',
            '1h': 'hour',
            '4h': 'hour',
            '1d': 'day'
        }
        
        # Map multiples for non-standard timeframes
        multiple_map = {
            '5m': 5,
            '15m': 15,
            '4h': 4
        }
        
        base_url = "https://min-api.cryptocompare.com/data/v2/histo"
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Skip unsupported timeframes
                    if timeframe not in timeframe_map:
                        self.logger.warning(f"Timeframe {timeframe} not supported by CryptoCompare")
                        continue
                    
                    # Parse symbol
                    base, quote = symbol.split('/')
                    
                    # Get timeframe parameters
                    histo_type = timeframe_map[timeframe]
                    aggregate = multiple_map.get(timeframe, 1)
                    
                    # Calculate limit and number of API calls needed
                    limit = 2000  # Max allowed by API

                    # Convert start_date to timestamp
                    if isinstance(start_date, str):
                        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
                    elif isinstance(start_date, date):
                        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
                    else:
                        self.logger.error(f"Unexpected type for start_date: {type(start_date)}")
                        return

                    end_ts = int(datetime.now().timestamp())
                    
                    # Determine time delta in seconds based on timeframe
                    if histo_type == 'minute':
                        delta = 60 * aggregate
                    elif histo_type == 'hour':
                        delta = 3600 * aggregate
                    else:  # day
                        delta = 86400
                    
                    # Calculate number of data points needed
                    total_points = (end_ts - start_ts) // delta
                    
                    # Initialize dataframe to store all data
                    all_data = []
                    
                    # Make API calls in batches
                    for i in range(0, total_points, limit):
                        to_ts = end_ts - (i * delta)
                        
                        # Construct URL
                        url = f"{base_url}{histo_type}?fsym={base}&tsym={quote}&limit={limit}&toTs={to_ts}&aggregate={aggregate}"
                        
                        self.logger.info(f"Fetching data from {url}")
                        response = requests.get(url)
                        
                        if response.status_code != 200:
                            self.logger.error(f"API error: {response.text}")
                            break
                        
                        data = response.json()
                        
                        if data['Response'] != 'Success':
                            self.logger.error(f"API error: {data['Message']}")
                            break
                        
                        # No more data
                        if not data['Data']['Data']:
                            break
                        
                        all_data.extend(data['Data']['Data'])
                        
                        # If we got less than the limit, we've reached the end
                        if len(data['Data']['Data']) < limit:
                            break
                    
                    if not all_data:
                        self.logger.warning(f"No data returned for {symbol} {timeframe}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    
                    # Rename columns to match our expected format
                    df.rename(columns={
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volumefrom': 'volume'
                    }, inplace=True)
                    
                    # Select only the columns we need
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    # Save to file
                    output_dir = Path(f"data/raw/cryptocompare/{symbol.replace('/', '_')}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / f"{timeframe}.csv"
                    df.to_csv(output_file)
                    self.logger.info(f"Saved {len(df)} records to {output_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error collecting {symbol} {timeframe} from CryptoCompare: {str(e)}")