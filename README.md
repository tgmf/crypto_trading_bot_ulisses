# Ulisses the Trading Bot 🔄

A cryptocurrency trading system using Bayesian and quantum-inspired probabilistic decision making.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyMC](https://img.shields.io/badge/PyMC-5.x-orange.svg)](https://www.pymc.io/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/tgmf/crypto_trading_bot_ulisses/issues)

## 🌟 Features

- **Quantum-Inspired Decision Making**: Treats positions as quantum states with superposition properties
- **Fee-Aware Training**: Incorporates transaction costs directly into model objectives
- **Bayesian Probabilistic Modeling**: Quantifies uncertainty in trading decisions
- **Advanced Backtesting Framework**: Multiple validation approaches to ensure robust performance
- **GPU Acceleration**: CUDA support for faster model training and inference
- **Comprehensive Risk Management**: Dynamic position sizing and market regime detection

## 📊 Trading Model Architecture

The system uses a three-state quantum-inspired approach:

- **Short (-1)**: Betting on price decrease
- **Neutral (0)**: No clear edge detected
- **Long (1)**: Betting on price increase

Unlike traditional binary classification models, our approach:
- Outputs continuous probability distributions across all states
- Supports partial position sizing based on prediction confidence
- Allows for hedged positions when uncertainties are high
- Directly incorporates fees into the objective function

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional but recommended)
- Conda package manager

### Local Installation

```bash
# Clone the repository
git clone https://github.com/tgmf/crypto_trading_bot_ulisses.git
cd quantumtrader

# Create conda environment
conda create -n trading_env python=3.10 -y
conda activate trading_env

# Install BLAS and core dependencies properly
conda install -c conda-forge pymc numpy scipy mkl-service openblas numba

# Install GPU acceleration (optional)
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install remaining dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/test_sets data/walk_forward_results models/comparisons logs
```

### Docker Installation

```bash
# Build Docker container
docker-compose build

# Start services
docker-compose up
```

## 💻 Quick Start

The project includes a convenience script for common operations:

```bash
# Show available commands
./launch.sh help

# Collect historical data from exchanges
./launch.sh collect-data --symbols "BTC/USDT" --timeframes "1h"

# Train a model on the collected data
./launch.sh train --symbols "BTC/USDT" --timeframes "1h"

# Run a backtest on the trained model
./launch.sh backtest --symbols "BTC/USDT" --timeframes "1h"

# Continue training an existing model with new data
./launch.sh continue-train --symbols "BTC/USDT" --timeframes "1h"
```

## 📝 Configuration

Key settings in `config/base.yaml`:

The system uses a centralized parameter management approach with ParamManager, which provides:

1. Clear Parameter Precedence:

- Command-line arguments (highest priority)
- Environment variables
- Configuration file values
- Default values (lowest priority)

2. Type-Safe Parameter Access:

- Nested parameter paths with dot notation
- Default value fallbacks
- Automatic type conversion
- Required parameter validation

3. Base Configuration File
/config/base.yaml:

```yaml

# Data Collection Settings
data:
  timeframes: [1m, 5m, 15m, 1h, 4h]
  symbols: [BTC/USD, ETH/USD, BTC/USDT, ETH/USDT]
  exchanges: [binance, kraken, kucoin]

# Backtesting Parameters
backtesting:
  min_profit_target: 0.008  # Minimum profit threshold

# Model Selection
model:
  type: "bayesian"  # Options: "bayesian", "enhanced_bayesian"

# Exchange Configuration
exchange:
  fee_rate: 0.0006  # Per-side fee rate
```

4. Modular Configuration Files 
/config/model/bayesian.yaml:
```yaml
model:
  type: 'bayesian'
  max_samples_per_batch: 100000

  features:
    volatility:
      enabled: true
      window: 20
```

5. Environment Variables
You can also set configuration with environment variables in a .env file:

```bash
# Exchange Configuration
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret_key
BINANCE_TESTNET=true

# Hardware Configuration
MAX_MEMORY=8000
CHUNK_SIZE=50000
```

6. Templates with common parameters
You can use custom templates that will overwrite variables:
```bash
./launch train --template crypto_bluechips
```

## 🔍 Testing Methodologies

The system supports multiple evaluation approaches:

### Standard Backtesting
```bash
./launch.sh backtest --symbols "BTC/USDT" --timeframes "1h"
```
Best for quick strategy evaluation with automatic test set detection.

### Walk-Forward Analysis
```bash
./launch.sh backtest --symbols "BTC/USDT" --timeframes "1h" --walk-forward
```
Simulates real-world model retraining for more realistic performance evaluation.

### Reversed Dataset Testing
```bash
./launch.sh train --symbols "BTC/USDT" --timeframes "1h" --reverse
```
Validates model consistency by swapping train and test sets, detecting overfitting.

### Multi-Symbol Validation
```bash
./launch.sh backtest --symbols "BTC/USDT ETH/USDT" --timeframes "1h 4h"
```
Tests model robustness across different assets and timeframes.


## Custom Feature Generators

/features/ directory contains custom feature generators that can be used by the trading bot, without modifying the core code in `/src`.

Read dedicated README in the directory for instructions.
 
## 📁 Project Structure

```
crypto_trading_bot_ulisses/
├── config/               # Configuration files
│   ├── templates/        # Template files
│   ├── model/            # Model-specific configuration
│   ├── strategy/         # Strategy-specific configuration
│   ├── timeframe/        # Timeframe-specific configuration
│   ├── symbol/           # Symbol-specific configuration
│   └── environment/      # Environment-specific configuration
├── data/                 # Data storage
│   ├── raw/              # Original data from exchanges
│   ├── processed/        # Feature-engineered data
│   ├── test_sets/        # Reserved test data
│   └── backtest_results/ # Backtest outputs
├── src/                  # Source code
│   ├── data/             # Data collection modules
│   ├── features/         # Feature engineering
│   ├── models/           # Trading models
│   ├── backtesting/      # Backtesting framework
│   ├── utils/            # Utility functions
│   └── visualization/    # Visualization tools
├── notebooks/            # Jupyter notebooks
├── models/               # Saved model artifacts
├── features/             # Custom features for training models
└── logs/                 # Application logs
```

## 📈 Performance Optimization

### GPU Acceleration

The system supports GPU acceleration for model training:

```python
# Uses JAX for GPU-accelerated sampling when available
trace = sample_numpyro_nuts(
    draws=1000,
    tune=1000, 
    chains=2,
    target_accept=0.85
)
```

### Memory Management

For handling large datasets, smart sampling is implemented:

```bash
# Set maximum samples for training
./launch.sh train --symbols "BTC/USDT" --timeframes "1m" --max-samples 100000
```

## 🔬 Advanced Usage

### Model Selection

Choose between different model implementations:

```bash
# Set model type via command line
./launch.sh train --symbols "BTC/USDT" --timeframes "1h" --model enhanced_bayesian

# Set in config.yaml, your_template.yaml or environment variable
export MODEL_TYPE=enhanced_bayesian
./launch.sh train --symbols "BTC/USDT" --timeframes "1h"
```

### Time-Series Cross-Validation

Ensure robust model performance with proper validation:

```bash
./launch.sh train --symbols "BTC/USDT" --timeframes "1h" --cv
```

### Incremental Training
Train large datasets in manageable chunks:
```bash
./launch.sh incremental --symbols "BTC/USDT" --timeframes "1h" --chunk-size 50000 --overlap 5000
```

Or continue training an existing model with new data:

```bash
./launch.sh continue-train --symbols "BTC/USDT" --timeframes "1h"
```

Or add custom data in params in your code:

```python
new_data_df = pd.read_csv('some_custom_data.csv')
params.set(new_data_df, 'new_data_df')

# Then call continue_training
model.continue_training()
```

### Programmatic Parameter Access
Within your Python code, access parameters consistently:
```python
from src.utils.param_manager import ParamManager

# Get global parameter manager
params = ParamManager.get_instance()

# Access parameters with defaults
fee_rate = params.get('exchange', 'fee_rate', default=0.0006)
symbols = params.get('data', 'symbols')

# Access with type conversion
test_size = params.get('training', 'test_size', default=0.3, _type=float)

# Required parameters that raise errors if missing
api_key = params.get('exchanges', 'binance', 'api_key', required=True)
```

## 📊 Visualization

The system generates trading performance visualizations:

- Equity curves comparing strategy vs buy-and-hold
- Trade distribution analysis
- Position duration statistics
- Probability distribution evolution
- Model consistency evaluation

# TensorFlow Environment Setup

## Separate Environment for TensorFlow-based Models

Due to dependency conflicts between PyMC/Arviz and TensorFlow, we maintain separate conda environments for different model types. This approach ensures maximum compatibility and stability without sacrificing functionality.

### Creating the TensorFlow Environment

```bash
# Create a new environment specifically for TensorFlow
conda create -n tf_trading_env python=3.10
conda activate tf_trading_env

# Install TensorFlow with GPU support
pip install tensorflow==2.16.1
pip install tensorflow-probability==0.23.0

# Install core dependencies
conda install -c conda-forge numpy=1.24.3 pandas matplotlib
pip install scikit-learn ccxt pandas-ta

# Install other project dependencies
pip install -r requirements-minimal.txt
```

### Switching Between Environments

#### For Bayesian Model Training (PyMC-based)
```bash
conda activate trading_env
./launch.sh train --symbols 'BTC/USDT' --timeframes '1h' --model bayesian
```

#### For TensorFlow Model Training (GPU-accelerated)
```bash
conda activate tf_trading_env
./launch.sh train --symbols 'BTC/USDT' --timeframes '1h' --model tf_bayesian
```

### Creating a requirements-minimal.txt File

To facilitate TensorFlow environment setup, create a `requirements-minimal.txt` file with minimal dependencies:

```
# Core dependencies
pandas>=1.5.0
numpy>=1.24.0,<1.25.0
matplotlib>=3.7.0
scikit-learn>=1.0.0

# Trading-specific
ccxt>=4.0.0
pandas-ta>=0.3.14b0

# Data handling
pyarrow>=12.0.0
pyyaml>=6.0
python-dotenv>=1.0.0

# Logging and utilities
tqdm>=4.65.0
schedule>=1.2.0
psutil>=5.9.0
```

### Sharing Models Between Environments

Models trained in either environment can be shared via the saved model files. Since models are saved to disk and loaded based on file paths, a model trained in the TensorFlow environment can be used for prediction in the PyMC environment, and vice versa, through the respective loading functions.

### GPU Monitoring

While using the TensorFlow environment, you can monitor GPU utilization with:

```bash
# For NVIDIA GPUs
watch -n0.5 nvidia-smi

# Detailed GPU stats
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1
```

### Common Issues

1. **OOM (Out of Memory) Errors**: If you encounter OOM errors, try:
   ```python
   # Limit GPU memory growth
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       tf.config.set_logical_device_configuration(
           gpus[0], 
           [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Limit to 4GB
       )
   ```

2. **CUDA Version Conflicts**: If you see CUDA-related errors, ensure your NVIDIA drivers are compatible with TensorFlow:
   ```bash
   # Check CUDA version
   nvcc --version
   
   # TensorFlow 2.16+ requires CUDA 11.8 or newer
   ```

3. **Mixed Precision Issues**: If you encounter numerical instability, try disabling mixed precision:
   ```python
   # Use full precision instead of mixed
   tf.keras.mixed_precision.set_global_policy('float32')
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- PyMC community for Bayesian modeling support
- Contributors to the CCXT library for exchange connectivity

## 👷🏾 ToDo

- More models, better models!
- ✅ Centralized parameter management
- Data Context
- Check Multi-symbol/timeframe training and backtesting
- Base Model class to handle basic methods across all the models
- Agile feature engineering infrastructure
- Better plotting utils
- Proper pathing from config everywhere