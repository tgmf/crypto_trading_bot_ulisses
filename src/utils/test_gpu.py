"""Test GPU availability for PyTensor/PyMC"""
import os
import sys
import subprocess

# Clear any previous configuration
if 'PYTENSOR_FLAGS' in os.environ:
    del os.environ['PYTENSOR_FLAGS']
if 'THEANO_FLAGS' in os.environ:
    del os.environ['THEANO_FLAGS']

# Basic thread configuration
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("Importing PyTensor...")
import pytensor

# Check PyTensor version
print(f"PyTensor version: {pytensor.__version__}")

# Print current configuration
print("\nCurrent PyTensor configuration:")
print(f"device: {pytensor.config.device}")
print(f"floatX: {pytensor.config.floatX}")

# Check available devices
print("\nChecking available computation backends...")
try:
    has_nvidia = subprocess.call("nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    print(f"NVIDIA GPU detected: {has_nvidia}")
    
    if has_nvidia:
        # Try to set device to cuda if GPU is available
        try:
            # Check if CUDA backend is available
            import pytensor.tensor.blas
            pytensor.config.blas__ldflags = ""  # Minimize blas dependency for testing
            print("PyTensor BLAS module imported successfully")
        except ImportError:
            print("PyTensor BLAS import failed - might affect GPU usage")
            
        print("\nTrying PyTensor operations...")
        import pytensor.tensor as tt
        x = tt.matrix('x')
        y = tt.matrix('y')
        z = tt.dot(x, y)
        f = pytensor.function([x, y], z, mode='FAST_COMPILE')
        
        # Create test matrices
        import numpy as np
        a = np.random.rand(100, 100).astype('float32')
        b = np.random.rand(100, 100).astype('float32')
        
        # Try computation
        c = f(a, b)
        print("Matrix multiplication completed successfully")
        
        # Check which device was actually used
        print(f"Device after operation: {pytensor.config.device}")
except Exception as e:
    print(f"Error in GPU test: {e}")

# Check for PyMC availability
print("\nChecking for PyMC...")
try:
    import pymc as pm
    print(f"PyMC version: {pm.__version__}")
    
    # Try a minimal PyMC model
    print("\nTesting PyMC with a simple model...")
    with pm.Model() as simple_model:
        x = pm.Normal('x', mu=0, sigma=1)
        # Draw just a few samples to test
        trace = pm.sample(draws=10, tune=10, chains=1)
    
    print("PyMC sampling completed successfully")
    
except ImportError as e:
    print(f"PyMC import error: {e}")
except Exception as e:
    print(f"PyMC test error: {e}")

print("\nTest complete. Check messages above for GPU usage information.")