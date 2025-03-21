import os
import sys
import logging
import subprocess
import platform

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_wsl():
    """Check if running in Windows Subsystem for Linux."""
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            if "microsoft" in f.read().lower():
                return True
    return False

def configure_gpu_for_wsl():
    """Configure PyTensor to use GPU in WSL2 environment with JAX backend."""
    # Check if we're in WSL
    if not is_wsl():
        logger.warning("Not running in WSL, using standard configuration")
        return configure_gpu_standard()
    
    logger.info("WSL environment detected, using WSL-specific GPU configuration")
    
    # Clean any existing configuration
    if 'PYTENSOR_FLAGS' in os.environ:
        del os.environ['PYTENSOR_FLAGS']
    if 'THEANO_FLAGS' in os.environ:
        del os.environ['THEANO_FLAGS']
    
    # Set thread limits to avoid conflicts
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Check for NVIDIA GPU via nvidia-smi (should work through WSL)
    try:
        has_nvidia = subprocess.call("nvidia-smi", shell=True, 
                                        stdout=subprocess.DEVNULL, 
                                        stderr=subprocess.DEVNULL) == 0
    except:
        has_nvidia = False
    
    if not has_nvidia:
        logger.warning("NVIDIA GPU not detected in WSL")
        return False
    
    logger.info("NVIDIA GPU detected in WSL")
    
    # For PyTensor with JAX backend on WSL2
    os.environ['PYTENSOR_FLAGS'] = 'device=cpu'  # Start with CPU to avoid old GPU backend
    
    # Test if JAX with CUDA is available
    try:
        import jax
        jax_devices = jax.devices()
        logger.info(f"JAX devices: {jax_devices}")
        
        # Check if JAX detects GPU
        has_jax_gpu = any('gpu' in str(d).lower() for d in jax_devices)
        logger.info(f"JAX GPU available: {has_jax_gpu}")
        
        if has_jax_gpu:
            # Configure PyMC to use JAX
            os.environ['PYMC_TESTVAL_START'] = '0.9'  # Helps with JAX compilation
            return True
    except ImportError:
        logger.warning("JAX not installed, cannot use JAX backend")
    except Exception as e:
        logger.warning(f"Error testing JAX: {e}")
    
    return False

def configure_gpu_standard():
    """Standard GPU configuration for non-WSL environments."""
    # Check for NVIDIA GPU
    try:
        has_nvidia = subprocess.call("nvidia-smi", shell=True, 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL) == 0
    except:
        has_nvidia = False
    
    if has_nvidia:
        logger.info("NVIDIA GPU detected, configuring PyTensor")
        os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float32'
        return True
    else:
        logger.info("No NVIDIA GPU detected, using CPU")
        return False

def configure_gpu():
    """Main configuration function that detects environment and configures accordingly."""
    if is_wsl():
        return configure_gpu_for_wsl()
    else:
        return configure_gpu_standard()

def check_gpu_availability():
    """Check if PyTensor is successfully using GPU after configuration."""
    try:
        import pytensor
        
        logger.info(f"PyTensor version: {pytensor.__version__}")
        logger.info(f"PyTensor device: {pytensor.config.device}")
        logger.info(f"PyTensor floatX: {pytensor.config.floatX}")
        
        is_gpu = pytensor.config.device == 'cuda'
        
        if is_gpu:
            logger.info("GPU acceleration is enabled")
        else:
            logger.info("GPU acceleration is NOT enabled, using CPU")
        
        return is_gpu
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return False

if __name__ == "__main__":
    # Log system info
    logger.info(f"System: {platform.system()}")
    logger.info(f"Release: {platform.release()}")
    
    # WSL detection
    wsl_detected = is_wsl()
    logger.info(f"WSL detected: {wsl_detected}")
    
    # Configure GPU
    gpu_configured = configure_gpu()
    logger.info(f"GPU configuration {'successful' if gpu_configured else 'failed'}")
    
    # Check if configuration was successful
    is_gpu = check_gpu_availability()
    
    # Print final status
    print(f"\nFinal Status:")
    print(f"WSL Environment: {'Yes' if wsl_detected else 'No'}")
    print(f"GPU Acceleration: {'Enabled' if is_gpu else 'Disabled'}")