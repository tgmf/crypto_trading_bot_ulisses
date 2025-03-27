"""
Incremental Training Module

This module handles training models incrementally on large historical datasets
by processing data in chronological chunks.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import gc
import time
import traceback
from datetime import datetime

# Try to import psutil for memory monitoring, but don't fail if unavailable
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
from ..models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    return 0  # Return 0 if psutil is not available

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

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


def cleanup_old_checkpoints(symbol, timeframe, model_type, current_chunk, max_to_keep=2):
    """
    Remove old checkpoints to save disk space
    
    Args:
        symbol (str): Trading pair symbol
        timeframe (str): Data timeframe
        model_type (str): Type of model
        current_chunk (int): Current chunk number
        max_to_keep (int): Maximum number of old checkpoints to keep
    """
    symbol_safe = symbol.replace('/', '_')
    model_dir = Path(f"models/{symbol_safe}/{timeframe}")
    
    if not model_dir.exists():
        return
        
    # Find all checkpoint files for this model with chunk in their name
    checkpoint_files = []
    for suffix in ['_scaler.pkl', '_trace.netcdf', '_feature_cols.pkl', '_metrics.json']:
        checkpoint_files.extend(list(model_dir.glob(f"*chunk*{suffix}")))
    
    # Group by chunk number
    chunk_groups = {}
    for file in checkpoint_files:
        # Extract chunk number from filename
        for part in file.stem.split('_'):
            if part.startswith('chunk'):
                chunk_num = int(part[5:])  # Extract number after "chunk"
                if chunk_num != current_chunk:  # Don't delete current chunk
                    if chunk_num not in chunk_groups:
                        chunk_groups[chunk_num] = []
                    chunk_groups[chunk_num].append(file)
                break
    
    # Get list of chunks sorted by number (oldest first)
    chunks_to_delete = sorted(chunk_groups.keys())
    
    # Keep only the N most recent chunks
    if len(chunks_to_delete) > max_to_keep:
        # Delete the oldest chunks beyond our keep limit
        for chunk_num in chunks_to_delete[:-max_to_keep]:
            for file in chunk_groups[chunk_num]:
                logger.info(f"Removing old checkpoint file: {file}")
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {file}: {e}")


def train_incrementally(params):
    """
    Train a model incrementally on large historical datasets
    
    This function:
    1. Loads large datasets in manageable chunks
    2. Trains on each chunk with overlap to maintain continuity
    3. Creates checkpoints to allow for resuming after interruptions
    4. Monitors memory usage to avoid OOM errors
    
    Args:
        params: ParamManager instance with all configuration
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Set all parameters with defaults from config
    try:
        # Get parameters from ParamManager
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        exchange = params.get('data', 'exchanges', 0)
        model_type = params.get('model', 'type', default='bayesian')
        
        # Get incremental training parameters
        chunk_size = params.get('training', 'incremental_training', 'chunk_size', default=50000)
        test_size = params.get('training', 'test_size', default=0.001)
        overlap = params.get('training', 'incremental_training', 'overlap', default=2500)
        max_memory_mb = params.get('training', 'max_memory_mb', default=8000)
        checkpoint_freq = params.get('training', 'incremental_training', 'checkpoint_frequency', default=1)
        resume_from = params.get('training', 'incremental_training', 'resume_from', default=None)
    
        # Prepare paths
        symbol_safe = symbol.replace('/', '_')
        data_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return False
        
        # Create logs directory for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"logs/incremental_training/{symbol_safe}/{timeframe}/{timestamp}")
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
            test_size_rows = 0  # Ensure test_size_rows is defined even when no test set is created
        
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
            model_factory = ModelFactory(params)
            model = model_factory.create_model()
            
            # Set suffix for loading checkpoint
            params.set(f"chunk{start_chunk-1}", 'model', 'suffix')
            
            # Load the model state
            if model.load_model():
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"No checkpoint model found, starting fresh")
                is_first_chunk = True
                model = None
    
        # Process data in chronological chunks
        for chunk_idx in range(start_chunk, num_chunks):
            # Update status file
            with open(status_file, 'a') as f:
                f.write(f"Starting chunk {chunk_idx+1}/{num_chunks} at {datetime.now().strftime('%H:%M:%S')}\n")
            
            # Calculate chunk boundaries    
            start_row = chunk_idx * effective_chunk_size + 1  # +1 to skip header
            end_row = min(start_row + chunk_size - 1, total_rows)
            
            logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks} (rows {start_row:,}-{end_row:,})")
            
            # Check memory before loading chunk
            mem_before = get_memory_usage()
            logger.info(f"Memory before loading chunk: {mem_before:.2f} MB")
            
            # Skip appropriate rows to get this chunk
            skiprows = list(range(1, start_row)) + list(range(end_row + 1, total_rows + 1))
            
            try:
                try:
                    # Get column names and types from a small sample
                    sample_df = pd.read_csv(data_file, nrows=5)
                    all_columns = sample_df.columns.tolist()
                    
                    # Create dtype dictionary for memory optimization
                    dtype_dict = {}
                    for col in all_columns:
                        # Skip the timestamp column as it will be the index
                        if col == 'timestamp':
                            continue
                            
                        # Detect column type from sample data
                        if sample_df[col].dtype == np.float64:
                            # Downcast float64 to float32
                            dtype_dict[col] = np.float32
                        elif sample_df[col].dtype == np.int64:
                            # Downcast int64 to int32 or int16 if possible
                            max_val = sample_df[col].max()
                            if max_val < 32767:
                                dtype_dict[col] = np.int16
                            else:
                                dtype_dict[col] = np.int32
                        elif sample_df[col].nunique() < 10 and sample_df[col].dtype == object:
                            # Use category for low-cardinality string columns
                            dtype_dict[col] = 'category'
                    
                    logger.info(f"Optimizing memory usage for {len(dtype_dict)} columns")
                    
                except Exception as e:
                    logger.warning(f"Could not analyze column types: {e}")
                    logger.warning("Using default dtypes")
                    dtype_dict = {}
                
                # Load the chunk with all columns preserved
                chunk_df = pd.read_csv(data_file, skiprows=skiprows, nrows=chunk_size,
                                    index_col='timestamp', parse_dates=True, 
                                    dtype=dtype_dict)
                
                logger.info(f"Loaded chunk with {len(chunk_df):,} rows and {len(chunk_df.columns)} columns")
                
                # Verify if any feature columns might be needed later
                feature_col_count = sum(1 for col in chunk_df.columns if col not in ['open', 'high', 'low', 'close', 'volume'])
                logger.info(f"Chunk contains {feature_col_count} feature columns")
                
                chunk_df = optimize_df_memory(chunk_df)
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
                    model_factory = ModelFactory(params)
                    model = model_factory.create_model()
                    logger.info(f"Model {model_type} initialized")
                
                # Process first chunk or continue training
                if is_first_chunk:
                    logger.info("Initial training on first chunk")
                    # Set custom dataframe for training
                    params.set(chunk_df, 'custom_df')
                    # Initial training - no test split for chunks
                    params.set(0.0, 'training', 'test_size')
                    model.train()
                    is_first_chunk = False
                else:
                    logger.info("Continuing training on next chunk")
                    # Set new data for continue_training
                    params.set(chunk_df, 'new_data_df')
                    # Continue training with this chunk
                    model.continue_training()
            
                # Free dataframe memory after training
                del chunk_df
                clear_memory()
            
                # Save checkpoint if needed
                if (chunk_idx + 1) % checkpoint_freq == 0 or chunk_idx == num_chunks - 1:
                    logger.info(f"Saving checkpoint after chunk {chunk_idx+1}")
                    # Set checkpoint suffix
                    params.set(f"chunk{chunk_idx+1}", 'model', 'suffix')
                    model.save_model()
                    
                    # Clean up old checkpoints
                    cleanup_old_checkpoints(symbol, timeframe, model_type, 
                                            current_chunk=chunk_idx+1, max_to_keep=2)
                    
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
                    
                    # Save model state with latest emergency tag
                    emergency_suffix = f"emergency_chunk{chunk_idx+1}"
                    params.set(emergency_suffix, 'model', 'suffix')
                    model.save_model()
                
                    # Clear model from memory
                    del model
                    clear_memory()
                    
                    # Reload model (creates fresh instance)
                    model_factory = ModelFactory(params)
                    model = model_factory.create_model()
                    # Keep the emergency suffix to load the right model
                    model.load_model()
                    
                    # Reset suffix after loading
                    params.set(None, 'model', 'suffix')
                
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
                        error_suffix = f"error_chunk{chunk_idx+1}"
                        params.set(error_suffix, 'model', 'suffix')
                        model.save_model()
                        logger.info(f"Saved model state before error at {error_suffix}")
                    except Exception as save_error:
                        logger.error(f"Could not save model state: {save_error}")
                
                # Clear memory before next attempt
                clear_memory()
                
                # Continue with next chunk
                continue
    
        # Final save with standard name
        if model is not None:
            # Clear suffix for final save
            params.set(None, 'model', 'suffix')
            model.save_model()
            logger.info("Final model saved")
    
        # Update status file
        with open(status_file, 'a') as f:
            f.write(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Final memory usage: {get_memory_usage():.2f} MB\n")
            f.write("------------------------------\n")
        
        logger.info("Incremental training complete")
        return True
        
    except Exception as e:
        logger.error(f"Error in incremental training: {str(e)}")
        logger.error(traceback.format_exc())
        return False
                    
# Apply additional memory optimization after loading
def optimize_df_memory(df):
    """Optimize dataframe memory usage without losing data"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Optimize categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < df.shape[0] * 0.5:  # If fewer than 50% unique values
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem * 100
    
    logger.info(f"Memory optimization: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    return df

def main():
    """Main function for incremental training script"""
    import argparse
    
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
    
    # Import ParamManager
    # This is imported here to avoid circular imports when called from main.py
    try:
        from ..utils.param_manager import ParamManager
    except ImportError:
        # When running standalone, use direct import
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from src.utils.param_manager import ParamManager
    
    # Handle multiple symbols and timeframes
    symbols = args.symbols.split()
    timeframes = args.timeframes.split()
    
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Starting incremental training for {symbol} {timeframe}")
            
            # Initialize ParamManager with CLI args converted to params
            # Use a fresh instance for each symbol/timeframe combination
            params = ParamManager.get_instance(reset=True)
            
            # Set data parameters
            params.set([symbol], 'data', 'symbols')
            params.set([timeframe], 'data', 'timeframes')
            params.set(args.exchange, 'data', 'exchanges', 0)
            
            # Set model parameters
            params.set(args.model, 'model', 'type')
            
            # Set training parameters
            params.set(args.chunk_size, 'training', 'incremental_training', 'chunk_size')
            params.set(args.test_size, 'training', 'test_size')
            params.set(args.overlap, 'training', 'incremental_training', 'overlap')
            if args.max_memory is not None:
                params.set(args.max_memory, 'training', 'max_memory_mb')
            params.set(args.checkpoint_freq, 'training', 'incremental_training', 'checkpoint_frequency')
            params.set(args.resume_from, 'training', 'incremental_training', 'resume_from')
            
            # Run incremental training with ParamManager
            train_incrementally(params)

if __name__ == "__main__":
    main()