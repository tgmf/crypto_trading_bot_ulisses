# QuantumTrader 🔄

A cryptocurrency trading system using Bayesian and quantum-inspired probabilistic decision making.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyMC](https://img.shields.io/badge/PyMC-5.x-orange.svg)](https://www.pymc.io/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/yourusername/quantumtrader/issues)

![Trading System Architecture](https://via.placeholder.com/800x300?text=Trading+System+Architecture)

## 🌟 Features

- **Quantum-Inspired Decision Making**: Treats positions as quantum states with superposition properties
- **Fee-Aware Training**: Incorporates transaction costs directly into model objectives
- **Bayesian Probabilistic Modeling**: Quantifies uncertainty in trading decisions
- **Advanced Backtesting Framework**: Multiple validation approaches to ensure robust performance
- **GPU Acceleration**: CUDA support for faster model training and inference
- **Comprehensive Risk Management**: Dynamic position sizing and market regime detection

## 📊 Trading Model Architecture

The system uses a unique three-state quantum-inspired approach:

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
git clone https://github.com/yourusername/quantumtrader.git
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
./launch.sh collect --symbols "BTC/USDT" --timeframes "1h"

# Train a model on the collected data
./launch.sh train --symbols "BTC/USDT" --timeframes "1h"

# Run a backtest on the trained model
./launch.sh backtest --symbols "BTC/USDT" --timeframes "1h"

# Continue training an existing model with new data
./launch.sh continue-train --symbols "BTC/USDT" --timeframes "1h"
```

## 📝 Configuration

Key settings in `config/config.yaml`:

```yaml
# Data Collection Settings
data:
  timeframes: [1m, 5m, 15m, 1h, 4h]
  symbols: [BTC/USD, ETH/USD, BTC/USDT, ETH/USDT]
  exchanges: [binance, kraken, kucoin]

# Backtesting Parameters
backtesting:
  fee_rate: 0.0006  # Per-side fee rate
  min_profit_target: 0.008  # Minimum profit threshold

# Model Selection
model:
  type: "bayesian"  # Options: "bayesian", "enhanced_bayesian", "quantum"
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

## 📁 Project Structure

```
quantumtrader/
├── config/               # Configuration files
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
│   └── visualization/    # Visualization tools
├── notebooks/            # Jupyter notebooks
├── models/               # Saved model artifacts
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
# Set in config.yaml or environment variable
export MODEL_TYPE=enhanced_bayesian
./launch.sh train --symbols "BTC/USDT" --timeframes "1h"
```

### Time-Series Cross-Validation

Ensure robust model performance with proper validation:

```bash
./launch.sh train --symbols "BTC/USDT" --timeframes "1h" --cv
```

### Incremental Training

Continue training an existing model with new data:

```bash
./launch.sh continue-train --symbols "BTC/USDT" --timeframes "1h"
```

## 📊 Visualization

The system generates trading performance visualizations:

- Equity curves comparing strategy vs buy-and-hold
- Trade distribution analysis
- Position duration statistics
- Probability distribution evolution
- Model consistency evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- PyMC community for Bayesian modeling support
- The quantum computing community for inspiration
- Contributors to the CCXT library for exchange connectivity