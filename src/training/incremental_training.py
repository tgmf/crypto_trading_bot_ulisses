"""
Incremental Training Script

This script trains a model incrementally on large historical datasets
by processing data in chronological chunks.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import os

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

def load_config(config_file='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_incrementally(symbol, timeframe, exchange, model_type, chunk_size=50000, test_size=0.001, overlap=0):
    """
    Train a model incrementally on historical data
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Data timeframe (e.g., '1m')
        exchange: Exchange name
        model_type: Type of model to use
        chunk_size: Number of samples per training chunk
        test_size: Proportion of data to reserve for final testing
        overlap: Number of samples to overlap between chunks for continuity
    """
    # Load configuration
    config = load_config()
    
    # Temporarily override the model type in the config
    original_model_type = config.get('model', {}).get('type', 'auto')
    if 'model' not in config:
        config['model'] = {}
    config['model']['type'] = model_type
    
    # Initialize model factory
    model_factory = ModelFactory(config)
    
    # Create model - no need to pass model_type as it's in the config now
    model = model_factory.create_model()
    
    # Restore original model type in config
    config['model']['type'] = original_model_type
    
    # Prepare paths
    symbol_safe = symbol.replace('/', '_')
    data_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return False
    
    # Rest of the implementation remains unchanged
    
    # Get file info
    file_size_gb = data_file.stat().st_size / (1024**3)
    logger.info(f"Processing file {data_file} ({file_size_gb:.2f} GB)")
    
    # Count rows without loading entire file
    with open(data_file, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    
    logger.info(f"Total rows: {total_rows:,}")
    
    # Calculate number of chunks
    effective_chunk_size = chunk_size - overlap
    num_chunks = (total_rows - overlap) // effective_chunk_size + 1
    logger.info(f"Processing in {num_chunks} chunks of {chunk_size} rows (with {overlap} overlap)")
    
    # First, create a small test set from the most recent data
    test_size_rows = int(total_rows * test_size)
    if test_size_rows > 0:
        logger.info(f"Creating test set with {test_size_rows:,} rows from most recent data")
        
        # Skip to the appropriate position from the end
        skiprows_test = list(range(1, total_rows - test_size_rows + 1))
        test_df = pd.read_csv(data_file, skiprows=skiprows_test, index_col='timestamp', parse_dates=True)
        logger.info(f"Test set created with {len(test_df):,} rows")
        
        # Save test set
        test_dir = Path(f"data/test_sets/{exchange}/{symbol_safe}")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / f"{timeframe}_incremental_test.csv"
        test_df.to_csv(test_file)
        logger.info(f"Test set saved to {test_file}")
        
        # Adjust total rows for training
        total_rows -= test_size_rows
    else:
        logger.info("No separate test set created (test_size too small)")
        
    # Process data in chronological chunks
    is_first_chunk = True
    last_chunk = None
    
    for chunk_idx in range(num_chunks):
        start_row = chunk_idx * effective_chunk_size + 1  # +1 to skip header
        end_row = min(start_row + chunk_size - 1, total_rows)
        
        logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks} (rows {start_row:,}-{end_row:,})")
        
        # Skip appropriate rows to get this chunk
        skiprows = list(range(1, start_row)) + list(range(end_row + 1, total_rows + 1))
        
        try:
            chunk_df = pd.read_csv(data_file, skiprows=skiprows, index_col='timestamp', parse_dates=True)
            logger.info(f"Loaded chunk with {len(chunk_df):,} rows")
            
            # Sort by timestamp to ensure chronological order
            chunk_df = chunk_df.sort_index()
            
            # Add previous overlap if available
            if last_chunk is not None and overlap > 0:
                # Combine with overlap from previous chunk
                combined_df = pd.concat([last_chunk.tail(overlap), chunk_df])
                combined_df = combined_df.drop_duplicates()
                logger.info(f"Combined with previous overlap: {len(combined_df):,} rows")
                chunk_df = combined_df
            
            # Save overlap for next iteration
            if overlap > 0:
                last_chunk = chunk_df.copy()
                
            # Process first chunk or continue training
            if is_first_chunk:
                logger.info("Initial training on first chunk")
                # Initial training
                model.train(exchange=exchange, symbol=symbol, timeframe=timeframe, 
                            test_size=0, custom_df=chunk_df)  # No test split for chunks
                is_first_chunk = False
            else:
                logger.info("Continuing training on next chunk")
                # Continue training with this chunk
                model.continue_training(new_data_df=chunk_df, exchange=exchange, 
                                        symbol=symbol, timeframe=timeframe)
            
            # Save checkpoint after each chunk
            logger.info(f"Saving checkpoint after chunk {chunk_idx+1}")
            model.save_model(exchange, symbol, timeframe, suffix=f"_chunk{chunk_idx+1}")
            
        except Exception as e:
            logger.error(f"ITLOG. Error processing chunk {chunk_idx+1}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue with next chunk
            continue
    
    # Final save with standard name
    model.save_model(exchange, symbol, timeframe)
    logger.info("Incremental training complete")
    
    return True

def main():
    """Main function for incremental training script"""
    parser = argparse.ArgumentParser(description='Incremental Training Script')
    parser.add_argument('--symbols', type=str, required=True, help='Trading pair symbol')
    parser.add_argument('--timeframes', type=str, required=True, help='Data timeframe')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['bayesian', 'tf_bayesian', 'enhanced_bayesian', 'quantum'], 
                        help='Model type')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Samples per chunk')
    parser.add_argument('--test-size', type=float, default=0.001, help='Test set proportion')
    parser.add_argument('--overlap', type=int, default=5000, help='Overlap between chunks')
    
    args = parser.parse_args()
    
    # Handle multiple symbols and timeframes
    symbols = args.symbols.split()
    timeframes = args.timeframes.split()
    
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Starting incremental training for {symbol} {timeframe}")
            train_incrementally(
                symbol=symbol,
                timeframe=timeframe,
                exchange=args.exchange,
                model_type=args.model,
                chunk_size=args.chunk_size,
                test_size=args.test_size,
                overlap=args.overlap
            )

if __name__ == "__main__":
    main()