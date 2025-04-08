"""Configure JAX for accelerated computation in WSL2."""

import os
import logging
import multiprocessing
import time
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

_JAX_CONFIG_CACHE = None

def is_wsl():
    """Check if running in WSL."""
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    return False

def configure_jax():
    """
    Configure JAX for optimal performance.
    
    - Sets multiprocessing method to 'spawn' to avoid fork issues
    - Configures precision settings
    - Optimizes for WSL if detected
    - Returns configuration information
    
    Returns:
        dict: Configuration information and status
    """
    
    global _JAX_CONFIG_CACHE
    
    # If already configured, return cached config
    if _JAX_CONFIG_CACHE is not None:
        logger.info("Using cached JAX configuration")
        return _JAX_CONFIG_CACHE
    
    config_info = {
        'jax_available': False,
        'acceleration_type': 'CPU-Only',
        'performance_score': float('inf'),
        'multiprocessing_method': None,
        'devices': [],
        'pytensor_config': {}  # Store PyTensor settings here
    }
    
    # Add debug info
    logger.info("Starting JAX configuration")
    
    # Set multiprocessing start method to 'spawn' to avoid JAX fork issues
    if hasattr(multiprocessing, 'get_start_method'):
        current_method = multiprocessing.get_start_method(allow_none=True)
        config_info['multiprocessing_method'] = current_method
        
        if current_method != 'spawn':
            try:
                multiprocessing.set_start_method('spawn', force=True)
                config_info['multiprocessing_method'] = 'spawn'
                logger.info("Set multiprocessing start method to 'spawn' for JAX compatibility")
            except RuntimeError as e:
                logger.warning(f"Could not set multiprocessing start method: {e}")
    
    try:
        import jax
        import jax.numpy as jnp
        from jax import config
        
        config_info['jax_available'] = True
    
        # Log JAX backend details
        logger.info(f"JAX backend: {config.x64_enabled}")
        
        # Set JAX to use 32-bit precision by default
        os.environ['JAX_ENABLE_X64'] = '0'
        config.update("jax_enable_x64", False)
        
        # Configure proper paths for WSL2
        if is_wsl():
            logger.info("WSL environment detected, optimizing JAX configuration")
            
            # Point to CUDA installation
            os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
            
            # Enable auto hardware selection 
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging noise
            
            # For Intel GPU optimization
            os.environ['DNNL_PRIMITIVE_CACHE_CAPACITY'] = '1024'  # Larger cache for Intel MKL-DNN
            os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())  # Use all CPU cores
        try:
            # Run quick performance benchmark with more monitoring
            logger.info("Starting JAX performance benchmark...")
            # Run quick performance benchmark
            size = 2000  # Smaller size for quick test
            key = jax.random.PRNGKey(42)
            
            # Generate a random normal distribution matrix
            x = jax.random.normal(key, (size, size), dtype=jnp.float32)
        
            # Warmup run to ensure JAX is ready
            _ = jnp.dot(x, x.T).block_until_ready()
            
            # Timed run
            start = time.time()
            result = jnp.dot(x, x.T)
            result.block_until_ready()
            elapsed = time.time() - start
            logger.info(f"Benchmark completed in {elapsed:.3f}s")
            
            config_info['performance_score'] = elapsed
        
            # Determine acceleration type
            if elapsed < 0.05:
                config_info['acceleration_type'] = 'JAX-GPU'
                logger.info(f"JAX GPU acceleration detected (benchmark: {elapsed:.3f}s)")
            elif elapsed < 0.2:
                config_info['acceleration_type'] = 'JAX-Accelerated'
                logger.info(f"JAX acceleration detected (benchmark: {elapsed:.3f}s)")
            else:
                config_info['acceleration_type'] = 'JAX-CPU'
                logger.info(f"JAX CPU mode detected (benchmark: {elapsed:.3f}s)")
                
        except Exception as bench_e:
            logger.error(f"JAX benchmark failed: {bench_e}", exc_info=True)
            # Still mark JAX as available but with no acceleration
            config_info['acceleration_type'] = 'JAX-CPU-NoBench'
            config_info['performance_score'] = float('inf')
        
        # Log device information
        devices = jax.devices()
        config_info['devices'] = [str(d) for d in devices]
        logger.info(f"JAX is using {len(devices)} device(s): {config_info['devices']}")
        
        # Configure PyTensor settings (store but don't apply yet)
        config_info['pytensor_config'] = {
            'compute_test_value': "off",
            'floatX': 'float32',  # Default, can be overridden by params
            'gcc__cxxflags': None,  # Will be set to "-fno-inline" if memory_efficient
        }
        
        # Store JAX configuration globally for access by models
        _JAX_CONFIG_CACHE = config_info
        
        return config_info
    
    except ImportError:
        logger.warning("JAX not available")
        config_info['jax_available'] = False
        config_info['acceleration_type'] = 'CPU-Only'
        config_info['performance_score'] = float('inf')
        return config_info
    except Exception as e:
        logger.warning(f"Error configuring JAX: {e}")
        config_info['jax_available'] = False
        config_info['acceleration_type'] = 'CPU-Only'
        config_info['performance_score'] = float('inf') 
        return config_info

def get_jax_config():
    """Get the current JAX configuration."""
    global _JAX_CONFIG_CACHE
    return _JAX_CONFIG_CACHE

def setup_model_pytensor(params):
    """
    Apply PyTensor configuration for a model based on params.
    Call this when initializing a model that needs PyTensor.
    """
    global _JAX_CONFIG
    
    # If JAX isn't available or PyTensor config not set, return False
    if not _JAX_CONFIG or not _JAX_CONFIG['jax_available']:
        return False
    
    try:
        import pytensor
        
        # Apply the pre-configured settings
        pytensor.config.compute_test_value = _JAX_CONFIG['pytensor_config']['compute_test_value']
        
        # Get precision from params or use default
        jax_precision = params.get('model', 'jax_precision', default='float32')
        pytensor.config.floatX = jax_precision
        
        # Apply memory efficiency settings if requested
        if params.get('model', 'jax_memory_efficient', default=False):
            pytensor.config.gcc__cxxflags = "-fno-inline"
        
        # Apply XLA debug flags if requested
        if params.get('model', 'jax_xla_debug', default=False):
            os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dumps'
        
        logger.debug(f"PyTensor configured for JAX with precision {jax_precision}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to configure PyTensor for JAX: {e}")
        return False