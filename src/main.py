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
    parser.add_argument('--model', type=str,
                    choices=['bayesian', 'tf_bayesian', 'enhanced_bayesian', 'quantum'],
                    help='Specify model type to use')  # Argument for specifying the model type
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
    parser.add_argument('--position-sizing', action='store_true',
                        help='Use quantum-inspired position sizing for backtesting') # Argument for enabling quantum-inspired position sizing
    parser.add_argument('--no-trade-threshold', type=float, default=0.96,
                        help='Threshold for no-trade probability (default: 0.96)') # Argument for specifying the no-trade threshold
    parser.add_argument('--min-position-change', type=float, default=0.05,
                        help='Minimum position change to avoid fee churn (default: 0.05)') # Argument for specifying the minimum position change
    parser.add_argument('--cv', action='store_true',
                    help='Use time-series cross-validation for training')  # Argument for enabling time-series cross-validation
    parser.add_argument('--reverse', action='store_true',
                        help='Train on test set and evaluate on training set (for model consistency testing)')  # Argument for enabling reverse testing
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Proportion of data to use for testing (0.0-1.0)')  # Argument for specifying test size
    args = parser.parse_args()  # Parse the command-line arguments
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)  # Create a logger for this module
    logger.info(f"Starting trading bot in {args.mode} mode")  # Log the starting mode
    
    # Load configuration
    config = load_config(args.config)  # Load the configuration from the specified file
    logger.info(f"Loaded configuration from {args.config}")  # Log the loaded configuration file
    
    # Override config with command-line arguments if provided
    if args.model:
        if 'model' not in config:
            config['model'] = {}
        config['model']['type'] = args.model
        logger.info(f"Using model type from command line: {args.model}")
        
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
            model.train_multi(symbols, timeframes, exchange, test_size=args.test_size)  # Train on multiple symbols/timeframes
        else:
            if args.reverse:
                # Use reversed dataset training
                logger.info(f"Training with reversed datasets for {symbols[0]} {timeframes[0]} (for consistency testing)")
                model.train_with_reversed_datasets(exchange, symbols[0], timeframes[0], test_size=args.test_size)  # Train on a single symbol/timeframe with reversed datasets
            elif args.cv:
                # Use time-series cross-validation
                logger.info(f"Training with time-series cross-validation for {symbols[0]} {timeframes[0]}")
                model.train_with_cv(exchange, symbols[0], timeframes[0], test_size=args.test_size)  # Train on a single symbol/timeframe with time-series cross-validation
            else:
                # Standard training with train-test split
                logger.info(f"Training with standard train-test split for {symbols[0]} {timeframes[0]}")
                model.train(exchange, symbols[0], timeframes[0], test_size=args.test_size)  # Train on a single symbol/timeframe
    elif args.mode == 'continue-train':
        # Load existing model
        model_factory = ModelFactory(config)
        model = model_factory.create_model()
        
        # Get symbols and timeframes from arguments
        if len(symbols) != 1 or len(timeframes) != 1:
            logger.error("Continue training mode requires exactly one symbol and one timeframe")
            sys.exit(1)
        
        symbol = symbols[0]
        timeframe = timeframes[0]
        
        # Try to load existing model
        if not model.load_model(exchange, symbol, timeframe):
            logger.error(f"No existing model found for {exchange} {symbol} {timeframe}")
            sys.exit(1)
        
        logger.info(f"Successfully loaded existing model for {exchange} {symbol} {timeframe}")
        
        # Load new data
        symbol_safe = symbol.replace('/', '_')
        data_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
        
        if not data_file.exists():
            logger.error(f"No processed data found at {data_file}")
            sys.exit(1)
        
        # Load the data
        df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from {data_file}")
        
        # Check for test set to avoid training on it
        test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
        if test_file.exists():
            test_df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
            logger.info(f"Found test set with {len(test_df)} rows")
            
            # Remove test data from training data
            df = df[~df.index.isin(test_df.index)]
            logger.info(f"Removed test data, {len(df)} rows remaining for training")
        
        # Apply data sampling if dataset is too large
        max_samples = 100000  # Adjust as needed
        if len(df) > max_samples:
            logger.info(f"Dataset too large, sampling {max_samples} rows")
            df = df.sample(max_samples, random_state=42)
        
        # Continue training the model
        logger.info(f"Continuing training for {exchange} {symbol} {timeframe} with {len(df)} samples")
        model.continue_training(df, exchange, symbol, timeframe)
        
        logger.info("Model training continued successfully")
    elif args.mode == 'backtest':
        # Create the appropriate tester based on command-line arguments
        if args.walk_forward:
            tester = WalkForwardTester(config)
        else:
            tester = BacktestEngine(config)
        
        if args.position_sizing:
            logger.info(f"Using quantum-inspired position sizing with no_trade_threshold={args.no_trade_threshold}")  # Log the testing details
            
            # Check if backtesting on multiple symbols/timeframes
            if len(symbols) > 1 or len(timeframes) > 1:
                logger.info(f"Testing on multiple symbols/timeframes: {symbols} {timeframes}")
                tester.run_multi_position_sizing_test(
                    symbols, 
                    timeframes, 
                    exchange,
                    no_trade_threshold=args.no_trade_threshold,
                    min_position_change=args.min_position_change
                ) # Run position sizing test on multiple symbols/timeframes
            else:
                logger.info(f"Testing on {symbols[0]} {timeframes[0]}")
                tester.run_position_sizing_test(
                    exchange, 
                    symbols[0], 
                    timeframes[0],
                    no_trade_threshold=args.no_trade_threshold,
                    min_position_change=args.min_position_change
                ) # Run position sizing test on a single symbol/timeframe
        else:
            # Original backtesting without position sizing
            if len(symbols) > 1 or len(timeframes) > 1: # Check if backtesting on multiple symbols/timeframes
                logger.info(f"Testing on multiple symbols/timeframes: {symbols} {timeframes}")
                tester.run_multi_test(symbols, timeframes, exchange) # Run test on multiple symbols/timeframes
            else:
                logger.info(f"Testing on {symbols[0]} {timeframes[0]}")
                tester.run_test(exchange, symbols[0], timeframes[0]) # Run test on a single symbol/timeframe
    elif args.mode == 'paper':
        logger.info("Paper trading mode not yet implemented")  # Log that paper trading is not implemented
    elif args.mode == 'live':
        logger.info("Live trading mode not yet implemented")  # Log that live trading is not implemented
    
    logger.info("Trading bot finished execution")  # Log the end of execution

if __name__ == "__main__":
    main()  # Run the main function if this script is executed directly