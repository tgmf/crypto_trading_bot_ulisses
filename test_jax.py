import logging
import time
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jax_test")

def test_jax():
    """Test JAX functionality step by step"""
    try:
        logger.info("Testing JAX installation...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Try importing JAX
        logger.info("Importing JAX...")
        import jax
        logger.info(f"JAX version: {jax.__version__}")
        
        # Try importing JAX numpy
        logger.info("Importing JAX numpy...")
        import jax.numpy as jnp
        
        # Get device info
        logger.info("Checking available devices...")
        devices = jax.devices()
        logger.info(f"Available devices: {devices}")
        
        # Test basic array operations
        logger.info("Testing basic array operations...")
        x = jnp.array([1, 2, 3])
        logger.info(f"Array sum: {jnp.sum(x)}")
        
        # Test matrix operations with progressively larger sizes
        sizes = [100, 500, 1000, 2000]
        key = jax.random.PRNGKey(42)
        
        for size in sizes:
            try:
                logger.info(f"Testing {size}x{size} matrix...")
                x = jax.random.normal(key, (size, size), dtype=jnp.float32)
                
                # Warmup
                logger.info("Running warmup...")
                _ = jnp.dot(x, x.T).block_until_ready()
                
                # Timed run
                logger.info("Running timed test...")
                start = time.time()
                result = jnp.dot(x, x.T)
                result.block_until_ready()
                elapsed = time.time() - start
                logger.info(f"Matrix size {size}x{size} completed in {elapsed:.3f}s")
                
            except Exception as e:
                logger.error(f"Failed at size {size}: {e}", exc_info=True)
                break
        
        logger.info("JAX test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"JAX test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_jax()