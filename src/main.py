#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the trading bot application.
"""

import argparse  # For parsing command-line arguments
import logging  # For logging messages
import sys  # For system-specific parameters and functions
import yaml  # For parsing YAML configuration files
from pathlib import Path  # For handling file system paths

# Importing necessary modules from the project
from src.data.data_collector import DataCollector
from src.features.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.walk_forward_tester import WalkForwardTester

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO, # Set the logging level to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Define the log message format
        handlers=[
            logging.FileHandler("logs/trading_bot.log"),  # Log messages to a file
            logging.StreamHandler(sys.stdout)  # Also log messages to the console
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file) # Parse the YAML file and return the configuration as a dictionary

def main():
    """Main function to run the trading bot"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')  # Argument for specifying the config file path
    parser.add_argument('--mode', type=str, 
                        choices=['collect', 'train', 'backtest', 'paper', 'live'],
                        default='collect', help='Operation mode')  # Argument for specifying the operation mode
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Specific symbols to process (e.g., BTC/USD ETH/USDT)')  # Argument for specifying symbols
    parser.add_argument('--timeframes', type=str, nargs='+',
                        help='Specific timeframes to process (e.g., 1h 4h 1d)')  # Argument for specifying timeframes
    parser.add_argument('--exchange', type=str, default=None,
                        help='Specific exchange to use (default: first exchange in config)')  # Argument for specifying exchange
    parser.add_argument('--template', type=str,
                        help='Use a predefined template (e.g., crypto_majors, altcoins)')  # Argument for specifying a template
    parser.add_argument('--walk-forward', action='store_true',
                        help='Use walk-forward testing instead of standard backtesting') # Argument for enabling walk-forward testing
    parser.add_argument('--reverse', action='store_true',
                        help='Train on test set and evaluate on training set (for model consistency testing)')  # Argument for enabling reverse testing
    args = parser.parse_args()  # Parse the command-line arguments
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)  # Create a logger for this module
    logger.info(f"Starting trading bot in {args.mode} mode")  # Log the starting mode
    
    # Load configuration
    config = load_config(args.config)  # Load the configuration from the specified file
    logger.info(f"Loaded configuration from {args.config}")  # Log the loaded configuration file
    
    # Override config with command-line arguments if provided
    if args.symbols:
        config['data']['symbols'] = args.symbols  # Override symbols in the config
        logger.info(f"Using symbols from command line: {args.symbols}")  # Log the overridden symbols
    
    if args.timeframes:
        config['data']['timeframes'] = args.timeframes  # Override timeframes in the config
        logger.info(f"Using timeframes from command line: {args.timeframes}")  # Log the overridden timeframes
    
    if args.exchange:
        config['data']['exchanges'] = [args.exchange] + [e for e in config['data']['exchanges'] if e != args.exchange]  # Override exchange in the config
        logger.info(f"Using exchange from command line: {args.exchange}")  # Log the overridden exchange
        
    if args.template:
        template_path = f"config/templates/{args.template}.yaml"  # Construct the template file path
        try:
            with open(template_path, 'r') as file:
                template_config = yaml.safe_load(file)  # Load the template configuration
                # Update only the data section
                if 'data' in template_config:
                    config['data'] = template_config['data']  # Override the data section in the config
                logger.info(f"Loaded template from {template_path}")  # Log the loaded template
        except Exception as e:
            logger.error(f"Error loading template {args.template}: {str(e)}")  # Log any errors in loading the template
    
    # Get symbols, timeframes and exchange for operations
    symbols = config.get('data', {}).get('symbols', [])  # Get symbols from the config
    timeframes = config.get('data', {}).get('timeframes', [])  # Get timeframes from the config
    exchange = config.get('data', {}).get('exchanges', ['binance'])[0]  # Get the first exchange from the config
    
    # Execute based on mode
    if args.mode == 'collect':
        collector = DataCollector(config)  # Initialize the data collector
        collector.collect_data()  # Collect data
    elif args.mode == 'train':
        feature_engineer = FeatureEngineer(config)  # Initialize the feature engineer
        feature_engineer.process_data()  # Process data
        
        model_factory = ModelFactory(config)  # Initialize the model factory
        model = model_factory.create_model()  # Create a model
        
        # Check for multi-symbol/timeframe training
        if len(symbols) > 1 or len(timeframes) > 1:
            logger.info(f"Training on multiple symbols/timeframes: {symbols} {timeframes}")  # Log the training details
            # Multi-symbol training currently doesn't support reversal
            if args.reverse:
                logger.warning("Reverse training not supported for multi-symbol mode. Falling back to standard training.")
            if args.cv:
                logger.warning("Cross-validation not supported for multi-symbol mode. Falling back to standard training.")
            model.train_multi(symbols, timeframes, exchange)  # Train on multiple symbols/timeframes
        else:
            if args.reverse:
                # Use reversed dataset training
                logger.info(f"Training with reversed datasets for {symbols[0]} {timeframes[0]} (for consistency testing)")
                model.train_with_reversed_datasets(exchange, symbols[0], timeframes[0])  # Train on a single symbol/timeframe with reversed datasets
            elif args.cv:
                # Use time-series cross-validation
                logger.info(f"Training with time-series cross-validation for {symbols[0]} {timeframes[0]}")
                model.train_with_cv(exchange, symbols[0], timeframes[0])  # Train on a single symbol/timeframe with time-series cross-validation
            else:
                # Standard training with train-test split
                logger.info(f"Training with standard train-test split for {symbols[0]} {timeframes[0]}")
                model.train(exchange, symbols[0], timeframes[0])  # Train on a single symbol/timeframe
    elif args.mode == 'backtest':
        # Create the appropriate tester based on command-line arguments
        if args.walk_forward:
            tester = WalkForwardTester(config)
        else:
            tester = BacktestEngine(config)
        
        # Check if backtesting on multiple symbols/timeframes
        if len(symbols) > 1 or len(timeframes) > 1:
            logger.info(f"Testing on multiple symbols/timeframes: {symbols} {timeframes}")  # Log the backtesting details
            tester.run_multi_test(symbols, timeframes, exchange) # Run backtest on multiple symbols/timeframes
        else:
            logger.info(f"Testing on {symbols[0]} {timeframes[0]}")
            tester.run_test(exchange, symbols[0], timeframes[0])# Run backtest on a single symbol/timeframe
    elif args.mode == 'paper':
        logger.info("Paper trading mode not yet implemented")  # Log that paper trading is not implemented
    elif args.mode == 'live':
        logger.info("Live trading mode not yet implemented")  # Log that live trading is not implemented
    
    logger.info("Trading bot finished execution")  # Log the end of execution

if __name__ == "__main__":
    main()  # Run the main function if this script is executed directly