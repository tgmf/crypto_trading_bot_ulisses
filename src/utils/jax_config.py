"""Configure JAX for accelerated computation in WSL2."""

import os
import logging
import subprocess
import sys

# Setup logging
logger = logging.getLogger(__name__)

def is_wsl():
    """Check if running in WSL."""
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    return False

def configure_jax():
    """Configure JAX for optimal performance."""
    # Set JAX to use 32-bit precision by default
    os.environ['JAX_ENABLE_X64'] = '0'
    
    # Configure proper paths for WSL2
    if is_wsl():
        logger.info("WSL environment detected, optimizing JAX configuration")
        
        # Point to CUDA installation
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
        
        # Enable auto hardware selection 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging noise
    
    # Run timing test if requested
    if __name__ == "__main__":
        benchmark_jax()

def benchmark_jax():
    """Run a simple benchmark to verify JAX performance."""
    try:
        import jax
        import jax.numpy as jnp
        import time
        
        logger.info(f"JAX version: {jax.__version__}")
        logger.info(f"JAX devices: {jax.devices()}")
        
        # Run matrix multiplication benchmark
        size = 5000
        logger.info(f"Running {size}x{size} matrix multiplication benchmark...")
        
        # Create random matrices
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (size, size), dtype=jnp.float32)
        
        # Warmup run
        _ = jnp.dot(x, x.T).block_until_ready()
        
        # Timed run
        start = time.time()
        result = jnp.dot(x, x.T)
        result.block_until_ready()  # Ensure computation completes
        end = time.time()
        
        elapsed = end - start
        logger.info(f"JAX benchmark completed in {elapsed:.4f} seconds")
        
        # Evaluate performance
        if elapsed < 1.0:
            logger.info("✅ Excellent performance - hardware acceleration is working")
        elif elapsed < 3.0:
            logger.info("✅ Good performance - likely using acceleration")
        else:
            logger.info("⚠️ Slow performance - may be using unaccelerated CPU only")
            
        return elapsed
        
    except ImportError:
        logger.error("JAX not installed. Please install JAX with 'pip install jax'")
        return None
    except Exception as e:
        logger.error(f"Error in JAX benchmark: {e}")
        return None

if __name__ == "__main__":
    configure_jax()