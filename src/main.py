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

from data.data_collector import DataCollector
from features.feature_engineering import FeatureEngineer
from models.model_factory import ModelFactory
from backtesting.backtest_engine import BacktestEngine

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
    parser.add_argument('--mode', type=str, choices=['collect', 'train', 'backtest', 'paper', 'live'],
                        default='collect', help='Operation mode')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting trading bot in {args.mode} mode")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Execute based on mode
    if args.mode == 'collect':
        collector = DataCollector(config)
        collector.collect_data()
    elif args.mode == 'train':
        feature_engineer = FeatureEngineer(config)
        feature_engineer.process_data()
        
        model_factory = ModelFactory(config)
        model = model_factory.create_model()
        model.train()
    elif args.mode == 'backtest':
        backtest_engine = BacktestEngine(config)
        backtest_engine.run_backtest()
    elif args.mode == 'paper':
        logger.info("Paper trading mode not yet implemented")
    elif args.mode == 'live':
        logger.info("Live trading mode not yet implemented")
    
    logger.info("Trading bot finished execution")

if __name__ == "__main__":
    main()