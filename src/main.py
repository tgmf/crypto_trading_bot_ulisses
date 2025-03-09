#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the trading bot application.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

from src.data.data_collector import DataCollector
from src.features.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
from src.backtesting.backtest_engine import BacktestEngine

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/trading_bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """Main function to run the trading bot"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, 
                        choices=['collect', 'train', 'backtest', 'paper', 'live'],
                        default='collect', help='Operation mode')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Specific symbols to process (e.g., BTC/USD ETH/USD)')
    parser.add_argument('--timeframes', type=str, nargs='+',
                        help='Specific timeframes to process (e.g., 1h 4h 1d)')
    parser.add_argument('--exchange', type=str, default=None,
                        help='Specific exchange to use (default: first exchange in config)')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting trading bot in {args.mode} mode")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override config with command-line arguments if provided
    if args.symbols:
        config['data']['symbols'] = args.symbols
        logger.info(f"Using symbols from command line: {args.symbols}")
    
    if args.timeframes:
        config['data']['timeframes'] = args.timeframes
        logger.info(f"Using timeframes from command line: {args.timeframes}")
    
    if args.exchange:
        config['data']['exchanges'] = [args.exchange] + [e for e in config['data']['exchanges'] if e != args.exchange]
        logger.info(f"Using exchange from command line: {args.exchange}")
    
    # Get symbols, timeframes and exchange for operations
    symbols = config.get('data', {}).get('symbols', [])
    timeframes = config.get('data', {}).get('timeframes', [])
    exchange = config.get('data', {}).get('exchanges', ['binance'])[0]
    
    # Execute based on mode
    if args.mode == 'collect':
        collector = DataCollector(config)
        collector.collect_data()
    elif args.mode == 'train':
        feature_engineer = FeatureEngineer(config)
        feature_engineer.process_data()
        
        model_factory = ModelFactory(config)
        model = model_factory.create_model()
        
        # Check if training on multiple symbols/timeframes
        if len(symbols) > 1 or len(timeframes) > 1:
            logger.info(f"Training on multiple symbols/timeframes: {symbols} {timeframes}")
            model.train_multi(symbols, timeframes, exchange)
        else:
            model.train(exchange, symbols[0], timeframes[0])
    elif args.mode == 'backtest':
        backtest_engine = BacktestEngine(config)
        
        # Check if backtesting on multiple symbols/timeframes
        if len(symbols) > 1 or len(timeframes) > 1:
            logger.info(f"Backtesting on multiple symbols/timeframes: {symbols} {timeframes}")
            backtest_engine.run_multi_backtest(symbols, timeframes, exchange)
        else:
            backtest_engine.run_backtest(exchange, symbols[0], timeframes[0])
    elif args.mode == 'paper':
        logger.info("Paper trading mode not yet implemented")
    elif args.mode == 'live':
        logger.info("Live trading mode not yet implemented")
    
    logger.info("Trading bot finished execution")

if __name__ == "__main__":
    main()