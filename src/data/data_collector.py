#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data collection module for cryptocurrency markets.
"""

import logging
import os
import time
import pandas as pd
import ccxt
from datetime import datetime
from pathlib import Path

class DataCollector:
    """Collects historical and real-time data from cryptocurrency exchanges"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchanges = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        exchange_configs = self.config.get('data', {}).get('exchanges', [])
        
        for exchange_name in exchange_configs:
            try:
                # Initialize without authentication for historical data
                exchange_class = getattr(ccxt, exchange_name)
                self.exchanges[exchange_name] = exchange_class({
                    'enableRateLimit': True,
                })
                self.logger.info(f"Initialized {exchange_name} connection")
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {str(e)}")
    
    def collect_data(self):
        """Collect historical data for configured symbols and timeframes"""
        symbols = self.config.get('data', {}).get('symbols', [])
        timeframes = self.config.get('data', {}).get('timeframes', [])
        start_date = self.config.get('data', {}).get('start_date', '2020-01-01')
        
        # Convert start_date to timestamp
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        
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
                                
                            ohlcv.extend(candles)
                            
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