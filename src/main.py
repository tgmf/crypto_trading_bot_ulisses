#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Ulisses trading bot.
"""

import logging  # For logging messages
import sys  # For system-specific parameters and functions
from pathlib import Path  # For handling file system paths
import argparse  # For parsing command-line arguments
import pandas as pd # For data manipulation


def setup_logging():
    """Configure logging settings"""
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Reset the root logger
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Define the log message format
        handlers=[
            logging.FileHandler("logs/trading_bot.log"),  # Log messages to a file
            logging.StreamHandler(sys.stdout)  # Also log messages to the console
        ]
    )
    return logging.getLogger(__name__)

# Configure logging before any other imports
logger = setup_logging()

# Import ParamManager after logging is configured
from src.core.param_manager import ParamManager
from src.data.data_collector import DataCollector
from src.features.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.walk_forward_tester import WalkForwardTester
from src.training.incremental_training import train_incrementally
from src.utils.jax_config import get_jax_config, configure_jax

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    
    # Command to execute
    parser.add_argument('command', type=str, 
                        choices=['train', 'backtest', 'collect-data', 'incremental', 
                                'continue-train', 'live', 'paper'],
                        help='Command to execute')
    
    # Common parameters
    parser.add_argument('--config', type=str, default='config/base.yaml',
                        help='Path to configuration file')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Specific symbols to process (e.g., BTC/USD ETH/USDT)')
    parser.add_argument('--timeframes', type=str, nargs='+',
                        help='Specific timeframes to process (e.g., 1h 4h 1d)')
    parser.add_argument('--exchanges', type=str,
                        help='Specific exchanges to use')
    parser.add_argument('--model', type=str,
                        choices=['bayesian', 'tf_bayesian', 'enhanced_bayesian', 'quantum'],
                        help='Specify model type to use')
    parser.add_argument('--template', type=str,
                        help='Use a predefined template (e.g., crypto_majors, altcoins)')
    
    # Training parameters
    parser.add_argument('--cv', action='store_true',
                        help='Use time-series cross-validation for training')
    parser.add_argument('--reverse', action='store_true',
                        help='Train on test set and evaluate on training set')
    parser.add_argument('--test-size', type=float,
                        help='Proportion of data to use for testing (0.0-1.0)')
    parser.add_argument('--comment', type=str,
                        help='Comment to add to training results')
    
    # Incremental training parameters
    parser.add_argument('--chunk-size', type=int,
                        help='Number of samples per chunk for incremental training')
    parser.add_argument('--overlap', type=int,
                        help='Overlap between chunks for incremental training')
    parser.add_argument('--max-memory', type=int,
                        help='Maximum memory usage in MB for incremental training')
    parser.add_argument('--checkpoint-freq', type=int,
                        help='Save checkpoint frequency for incremental training')
    parser.add_argument('--resume-from', type=int,
                        help='Resume incremental training from chunk index')
    
    # Backtesting parameters
    parser.add_argument('--walk-forward', action='store_true',
                        help='Use walk-forward testing for backtesting')
    parser.add_argument('--position-sizing', action='store_true',
                        help='Use quantum-inspired position sizing for backtesting')
    parser.add_argument('--no-trade-threshold', type=float,
                        help='Threshold for no-trade probability')
    parser.add_argument('--min-position-change', type=float,
                        help='Minimum position change to avoid fee churn')
    
    return parser.parse_args()

def main():
    """Main function to run the Ulisses trading bot"""    
    args = parse_args()
    command = args.command
    
    logger.info(f"Starting trading bot with command: {command}")  # Log the starting mode
    
    # Initialize parameter manager with all sources
    params = ParamManager.get_instance(
        base_config_path=args.config,
        cli_args=args,
        env_vars=True
    )
    
    # Configure JAX once at startup and store in params
    jax_config = get_jax_config()
    if jax_config is None:
        logger.info("JAX not yet configured, configuring now...")
        jax_config = configure_jax()
    else:
        logger.info("Using previously configured JAX settings")
    
    if jax_config:
        params.set(jax_config['jax_available'], 'model', 'jax_available')
        params.set(jax_config['acceleration_type'], 'model', 'acceleration_type')
        params.set(jax_config['performance_score'], 'model', 'jax_performance')
        params.set(jax_config['jax_available'], 'model', 'use_jax')  # Default to using JAX if available
    else:
        logger.warning("JAX acceleration not available")
        params.set(False, 'model', 'jax_available')
        params.set('CPU-Only', 'model', 'acceleration_type')
    
    # Load template if specified
    if args.template:
        template_path = f"config/templates/{args.template}.yaml"
        try:
            params.load_from_file(template_path, section='data')
            logger.info(f"Loaded template from {template_path}")
        except Exception as e:
            logger.error(f"Error loading template {args.template}: {str(e)}")
    
    # Get common parameters
    symbols = params.get('data', 'symbols') # Get symbols from config
    timeframes = params.get('data', 'timeframes') # Get timeframes from config
    exchanges = params.get('data', 'exchanges')  # Ger exchanges from config TODO: Support multiple exchanges
    
    # Add after initializing params in main.py
    if args.model:
        logger.info(f"CLI argument model={args.model}")
        # Log how it's stored in ParamManager
        params.set(args.model, 'model', 'type')  # Override the config value
        
    # Execute based on command
    if command == 'collect-data':
        collector = DataCollector(params)
        collector.collect_data()
        
    elif command == 'train':
        feature_engineer = FeatureEngineer(params)  # Initialize the feature engineer
        feature_engineer.process_data()  # Process data
        
        model_factory = ModelFactory(params)  # Initialize the model factory
        model = model_factory.create_model()  # Create a model
        
        # Check for multi-symbol/timeframe training
        if len(symbols) > 1 or len(timeframes) > 1:
            logger.info(f"Training on multiple symbols/timeframes: {symbols} {timeframes}")  # Log the training details
            # Check if incompatible options are selected
            if params.get('training', 'reverse_split', default=False):
                logger.warning("Reverse training not supported for multi-symbol mode. Falling back to standard training.")
            if params.get('training', 'use_cv', default=False):
                logger.warning("Cross-validation not supported for multi-symbol mode. Falling back to standard training.")
                
            model.train_multi()
        else:
            # Single symbol/timeframe training with options
            symbol = symbols[0]
            timeframe = timeframes[0]
            
            if params.get('training', 'reverse_split', default=False):
                # Use reversed dataset training
                logger.info(f"Training with reversed datasets for {symbol} {timeframe}")
                model.train_with_reversed_datasets()  # Train on a single symbol/timeframe with reversed datasets
            elif args.cv:
                # Use time-series cross-validation
                logger.info(f"Training with time-series cross-validation for {symbols[0]} {timeframes[0]}")
                model.train_with_cv()  # Train on a single symbol/timeframe with time-series cross-validation
            else:
                # Standard training with train-test split
                logger.info(f"Training with standard train-test split for {symbols[0]} {timeframes[0]}")
                model.train()  # Train on a single symbol/timeframe
                
    elif command == 'continue-train':
        # Load existing model
        model_factory = ModelFactory(params)
        model = model_factory.create_model()
        
        # Get symbols and timeframes from arguments
        if len(symbols) != 1 or len(timeframes) != 1:
            logger.error("Continue training mode requires exactly one symbol and one timeframe")
            sys.exit(1)
        
        symbol = symbols[0]
        timeframe = timeframes[0]
        
        # Try to load existing model
        if not model.load_model():
            logger.error(f"No existing model found for {symbol} {timeframe}")
            sys.exit(1)
        
        logger.info(f"Successfully loaded existing model for {symbol} {timeframe}")
        
        # Continue training the model
        logger.info(f"Continuing training for {symbol} {timeframe}")
        model.continue_training()
        
        logger.info("Model training continued successfully")
        
    elif command == 'incremental':
        train_incrementally(params)
        
    elif command == 'backtest':
        # Create the appropriate tester based on command-line arguments
        use_walk_forward = params.get('backtesting', 'use_walk_forward', default=False)
        use_position_sizing = params.get('backtesting', 'use_position_sizing', default=False)
        
        if use_walk_forward:
            tester = WalkForwardTester(params)
            logger.info("Using walk-forward testing")
        else:
            tester = BacktestEngine(params)
        
        if use_position_sizing:
            no_trade_threshold = params.get('backtesting', 'no_trade_threshold')
            min_position_change = params.get('backtesting', 'min_position_change')
            
            logger.info(f"Using position sizing with threshold={no_trade_threshold}, min_change={min_position_change}")
            
            # Run with position sizing
            if len(symbols) > 1 or len(timeframes) > 1:
                tester.run_multi_position_sizing_test()
            else:
                tester.run_position_sizing_test()
        else:
            # Standard backtesting
            if len(symbols) > 1 or len(timeframes) > 1:
                tester.run_multi_test()
            else:
                tester.run_test() # Run the backtest on a single symbol/timeframe
                
    elif command == 'paper':
        logger.info("Paper trading mode not yet implemented")  # Log that paper trading is not implemented
    elif command == 'live':
        logger.info("Live trading mode not yet implemented")  # Log that live trading is not implemented
    
    logger.info("Trading bot finished execution")  # Log the end of execution

if __name__ == "__main__":
    main()  # Run the main function if this script is executed directly