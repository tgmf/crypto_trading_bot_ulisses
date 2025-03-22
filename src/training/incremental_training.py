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
import gc
import time
import psutil
from datetime import datetime

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

def load_config(config_file='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def clear_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    
    # Try to explicitly clear PyTensor/PyMC cache if available
    try:
        import pytensor
        pytensor.config.gcc__cxxflags = "-fno-inline"  # Reduce memory usage in compilation
        if hasattr(pytensor, 'sandbox') and hasattr(pytensor.sandbox, 'cuda'):
            pytensor.sandbox.cuda.cuda_ndarray.cuda_ndarray.CudaNdarray.sync()
    except:
        pass
    
    # Wait a bit to ensure memory is released
    time.sleep(1)


def train_incrementally(symbol="BTC/USDT", timeframe="15m", exchange="binance", model_type="enhanced_bayesian", 
                        chunk_size=50000, test_size=0.001, overlap=0, max_memory_mb=None, 
                        checkpoint_frequency=1, resume_from=None):
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
    
    # Add memory management settings to config
    if 'training' not in config:
        config['training'] = {}
    config['training']['memory_efficient'] = True
    
    # Temporarily override the model type in the config
    original_model_type = config.get('model', {}).get('type', 'auto')
    if 'model' not in config:
        config['model'] = {}
    config['model']['type'] = model_type
    
    # Prepare paths
    symbol_safe = symbol.replace('/', '_')
    data_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return False
    
    # Create logs directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/incremental_training/{exchange}/{symbol_safe}/{timeframe}/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
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
    
    # Create status file to track progress
    status_file = log_dir / "status.log"
    with open(status_file, 'w') as f:
        f.write(f"Started: {timestamp}\n")
        f.write(f"Symbol: {symbol}, Timeframe: {timeframe}\n")
        f.write(f"Total chunks: {num_chunks}\n")
        f.write(f"Chunk size: {chunk_size}\n")
        f.write(f"Overlap: {overlap}\n")
        f.write(f"Initial memory: {initial_memory:.2f} MB\n")
        f.write("------------------------------\n")
    
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
        
        # Free memory
        del test_df
        clear_memory()
        
        # Adjust total rows for training
        total_rows -= test_size_rows
    else:
        logger.info("No separate test set created (test_size too small)")
    
    # Initialize model when needed to avoid memory usage
    model = None
    is_first_chunk = True
    last_chunk = None
    
    # Determine starting chunk if resuming
    start_chunk = 0
    if resume_from is not None and resume_from > 0:
        start_chunk = resume_from
        is_first_chunk = False
        logger.info(f"Resuming from chunk {start_chunk}")
        
        # Check if model exists from previous run
        model_path = Path(f"models/{exchange}/{symbol_safe}/{timeframe}/{model_type}_chunk{start_chunk-1}.pkl")
        if model_path.exists():
            # Initialize model factory
            model_factory = ModelFactory(config)
            model = model_factory.create_model()
            
            # Load the model state
            logger.info(f"Loading model state from {model_path}")
            model.load_model(exchange, symbol, timeframe, suffix=f"_chunk{start_chunk-1}")
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"No model found at {model_path}, starting fresh")
            is_first_chunk = True
    
    # Process data in chronological chunks
    for chunk_idx in range(start_chunk, num_chunks):
        # Update status file
        with open(status_file, 'a') as f:
            f.write(f"Starting chunk {chunk_idx+1}/{num_chunks} at {datetime.now().strftime('%H:%M:%S')}\n")
            
        start_row = chunk_idx * effective_chunk_size + 1  # +1 to skip header
        end_row = min(start_row + chunk_size - 1, total_rows)
        
        logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks} (rows {start_row:,}-{end_row:,})")
        
        # Check memory before loading chunk
        mem_before = get_memory_usage()
        logger.info(f"Memory before loading chunk: {mem_before:.2f} MB")
        
        # Skip appropriate rows to get this chunk
        skiprows = list(range(1, start_row)) + list(range(end_row + 1, total_rows + 1))
        
        try:
            # Load chunk with specific dtypes to reduce memory
            dtype_dict = {
                'open': np.float32,
                'high': np.float32,
                'low': np.float32,
                'close': np.float32,
                'volume': np.float32
            }
            
            chunk_df = pd.read_csv(data_file, skiprows=skiprows, index_col='timestamp', 
                                 parse_dates=True, dtype=dtype_dict)
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
                
                # Clear previous overlap to free memory
                del last_chunk
                clear_memory()
            
            # Save overlap for next iteration if needed
            if overlap > 0:
                last_chunk = chunk_df.tail(overlap).copy()
            
            # Check if we need to initialize the model
            if model is None:
                logger.info("Initializing model...")
                # Initialize model factory
                model_factory = ModelFactory(config)
                model = model_factory.create_model()
                logger.info(f"Model {model_type} initialized")
            
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
            
            # Free dataframe memory after training
            del chunk_df
            
            # Save checkpoint if needed
            if (chunk_idx + 1) % checkpoint_frequency == 0 or chunk_idx == num_chunks - 1:
                logger.info(f"Saving checkpoint after chunk {chunk_idx+1}")
                model.save_model(exchange, symbol, timeframe, suffix=f"_chunk{chunk_idx+1}")
                
                # Update status file
                with open(status_file, 'a') as f:
                    f.write(f"Saved checkpoint for chunk {chunk_idx+1} at {datetime.now().strftime('%H:%M:%S')}\n")
                    current_mem = get_memory_usage()
                    f.write(f"Memory usage: {current_mem:.2f} MB\n")
            
            # Check memory after processing
            mem_after = get_memory_usage()
            logger.info(f"Memory after processing chunk: {mem_after:.2f} MB (delta: {mem_after-mem_before:.2f} MB)")
            
            # Check if we're approaching memory limit
            if max_memory_mb is not None and mem_after > max_memory_mb * 0.9:
                logger.warning(f"Memory usage ({mem_after:.2f} MB) approaching limit ({max_memory_mb} MB)")
                logger.info("Performing emergency memory cleanup...")
                
                # Save model state
                emergency_suffix = f"_emergency_chunk{chunk_idx+1}"
                model.save_model(exchange, symbol, timeframe, suffix=emergency_suffix)
                
                # Clear model from memory
                del model
                clear_memory()
                
                # Reload model (creates fresh instance)
                model_factory = ModelFactory(config)
                model = model_factory.create_model()
                model.load_model(exchange, symbol, timeframe, suffix=emergency_suffix)
                
                logger.info(f"Model reloaded after emergency cleanup. New memory: {get_memory_usage():.2f} MB")
            
            # Force garbage collection
            clear_memory()
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx+1}: {str(e)}")
            # Log detailed traceback
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            
            # Write error to status file
            with open(status_file, 'a') as f:
                f.write(f"ERROR in chunk {chunk_idx+1}: {str(e)}\n")
                f.write(tb + "\n")
            
            # Try to save model state if available
            if model is not None:
                try:
                    error_suffix = f"_error_chunk{chunk_idx+1}"
                    model.save_model(exchange, symbol, timeframe, suffix=error_suffix)
                    logger.info(f"Saved model state before error at {error_suffix}")
                except Exception as save_error:
                    logger.error(f"Could not save model state: {save_error}")
            
            # Clear memory before next attempt
            clear_memory()
            
            # Continue with next chunk
            continue
    
    # Final save with standard name
    if model is not None:
        model.save_model(exchange, symbol, timeframe)
        logger.info("Final model saved")
    
    # Update status file
    with open(status_file, 'a') as f:
        f.write(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Final memory usage: {get_memory_usage():.2f} MB\n")
        f.write("------------------------------\n")
    
    logger.info("Incremental training complete")
    return True

def main():
    """Main function for incremental training script"""
    parser = argparse.ArgumentParser(description='Incremental Training Script')
    parser.add_argument('--symbols', type=str, required=True, help='Trading pair symbol')
    parser.add_argument('--timeframes', type=str, required=True, help='Data timeframe')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['bayesian', 'tf_bayesian', 'enhanced_bayesian', 'jax_bayesian', 'quantum'], 
                        help='Model type')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Samples per chunk')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--overlap', type=int, default=5000, help='Overlap between chunks')
    parser.add_argument('--max-memory', type=int, default=None, help='Maximum memory usage in MB')
    parser.add_argument('--checkpoint-freq', type=int, default=1, help='Save checkpoint every N chunks')
    parser.add_argument('--resume-from', type=int, default=None, help='Resume from chunk index')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/incremental_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
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
                overlap=args.overlap,
                max_memory_mb=args.max_memory,
                checkpoint_frequency=args.checkpoint_freq,
                resume_from=args.resume_from
            )

if __name__ == "__main__":
    main()