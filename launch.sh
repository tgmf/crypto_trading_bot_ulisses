#!/bin/bash

# Trading Bot Launch Script

# Function to print help message
show_help() {
  echo "Cryptocurrency Trading Bot"
  echo ""
  echo "Usage: ./launch.sh [command] [options]"
  echo ""
  echo "Commands:"
  echo "  analyze       - Analyze backtest results"
  echo "  backtest      - Backtest trained model"
  echo "  collect-data  - Collect historical data from exchanges"
  echo "  continue-train - Continue training existing model"
  echo "  docker        - Run in Docker container"
  echo "  docker-dev    - Run development environment in Docker"
  echo "  incremental   - Train incrementally on large datasets"
  echo "  notebook      - Launch Jupyter notebook for analysis"
  echo "  train         - Train trading model on collected data"
  echo "  help          - Show this help message"
  echo ""
  echo "Common Options:"
  echo "  --comment                     - Optional comment to identify model purpose"
  echo "  --config path/to/config.yaml  - Specify custom config file"
  echo "  --exchange binance            - Specify exchange to use"
  echo "  --model bayesian|tf_bayesian|enhanced_bayesian  - Specify model type to use"
  echo "  --symbols 'BTC/USD ETH/USD'   - Specify symbols to process"
  exho "  --template 'template_name'    - Specify template to use over config file"
  echo "  --timeframes '1h 4h'          - Specify timeframes to process"
  echo ""
  echo "Training Options:"
  echo "  --cv                          - Use time-series cross-validation for training"
  echo "  --chunk-size 50000             - Chunk size for incremental training (default: 10000)"
  echo "  --reverse                     - Train on test data and test on training data"
  echo "  --test-size 0.3               - Proportion of data to use for testing (default: 0.3)"
  echo ""
  echo "Backtesting Options:"
  echo "  --file models/BTC_USDT/15m/   - Specify path to trained model for backtesting"
  echo "  --min-position-change 0.015   - Minimum position change to avoid fee churn (default: 0.005)"
  echo "  --no-trade-threshold 0.96     - Threshold for no-trade probability (default: 0.96)"
  echo "  --position-sizing             - Use quantum-inspired position sizing strategy"
  echo "  --walk-forward                - Use walk-forward testing for backtesting"
  echo ""
  echo "Examples:"
  echo "  ./launch.sh train --timeframes '1h' --model bayesian"
  echo "  ./launch.sh train --symbols 'ETH/USDT' --timeframes '1m' --cv"
  echo "  ./launch.sh train --symbols 'ETH/USDT' --timeframes '1m' --cv --comment 'rocket fuel'"
  echo "  ./launch.sh train --symbols 'BTC/USD ETH/USDT' --timeframes '1d' --reverse"
  echo "  ./launch.sh incremental --symbol 'BTC/USDT' --timeframe '1m' --model enhanced_bayesian --chunk-size 50000"
  echo "  ./launch.sh backtest --symbols 'BTC/USD' --timeframes '1h' --walk-forward"
  echo "  ./launch.sh backtest --file 'data/backtest_results/binance/BTC_USDT/1m_test_set_20250311_135759.csv'"
  echo "  ./launch.sh backtest --symbols 'BTC/USDT' --timeframes '1h' --position-sizing --no-trade-threshold 0.97"
  echo ""
}

# Set matplotlib to use Agg backend
export MPLBACKEND=Agg
# Prevent Qt from looking for Wayland
export QT_QPA_PLATFORM=offscreen

# Make sure we're in the project root directory
cd "$(dirname "$0")"

# Load environment variables if .env exists
if [ -f .env ]; then
  source .env
fi

if [ "$DEBUG" = "true" ]; then
  echo "Environment variables loaded:"
  env | grep -E 'BINANCE|KRAKEN|KUCOIN|SYMBOLS|TIMEFRAMES|MODEL'
fi

# Check if command is specified
if [ $# -eq 0 ]; then
  show_help
  exit 1
fi

# Parse command
COMMAND=$1
shift 1

# Define valid commands
MAIN_COMMANDS=("train" "backtest" "collect-data" "continue-train" "incremental" "live" "paper")
SPECIAL_COMMANDS=("notebook" "analyze" "help" "docker" "docker-dev")

# Check if the command is valid for main.py
if [[ " ${MAIN_COMMANDS[*]} " =~ " ${COMMAND} " ]]; then
  # For valid commands, pass to main.py
  echo "Executing command: $COMMAND"
  python -m src.main "$COMMAND" "$@"
  exit 0
else
  # Handle special commands that don't use main.py
  case "$COMMAND" in
    analyze)
      echo "Analyzing backtest results..."
      python -m src.analysis.backtest_analyzer "$@"
      exit 0
      ;;
    docker)
      echo "Running in Docker container..."
      docker run -it --rm -v $(pwd):/app trading-bot "$@"
      exit 0
      ;;
    docker-dev)
      echo "Running development environment in Docker..."
      docker-compose up -d
      exit 0
      ;;
    notebook)
      echo "Launching Jupyter notebook..."
      jupyter notebook
      exit 0
      ;;
    help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: Unknown command '$COMMAND'"
      show_help
      exit 1
      ;;
  esac
fi