#!/bin/bash

# Trading Bot Launch Script

# Function to print help message
show_help() {
  echo "Cryptocurrency Trading Bot"
  echo ""
  echo "Usage: ./launch.sh [command] [options]"
  echo ""
  echo "Commands:"
  echo "  analyze     - Analyze backtest results"
  echo "  backtest    - Backtest trained model"
  echo "  collect     - Collect historical data from exchanges"
  echo "  docker      - Run in Docker container"
  echo "  docker-dev  - Run development environment in Docker"
  echo "  incremental    - Train incrementally on large datasets"
  echo "  notebook    - Launch Jupyter notebook for analysis"
  echo "  train       - Train trading model on collected data"
  echo "  help        - Show this help message"
  echo ""
  echo "Options:"
  echo "  --comment        Optional comment to identify model purpose"
  echo "  --cv                          - Use time-series cross-validation for training"
  echo "  --exchange binance            - Specify exchange to use"
  echo "  --file 'path/to/backtest.csv' - Specify file for analysis"
  echo "  --min-position-change 0.015    - Minimum position change to avoid fee churn (default: 0.005)"
  echo "  --model bayesian|tf_bayesian|enhanced_bayesian  - Specify model type to use"
  echo "  --no-trade-threshold 0.96     - Threshold for no-trade probability (default: 0.96)"
  echo "  --position-sizing             - Use quantum-inspired position sizing"
  echo "  --reverse                     - Train on test data and test on training data"
  echo "  --symbols 'BTC/USD ETH/USD'   - Specify symbols to process"
  echo "  --template crypto_majors      - Use a predefined template"
  echo "  --test-size 0.3               - Proportion of data to use for testing (default: 0.3)"
  echo "  --timeframes '1h 4h'          - Specify timeframes to process"
  echo "  --walk-forward                - Use walk-forward testing for backtesting"
  echo ""
  echo "Examples:"
  echo "  ./launch.sh train --timeframes '1h' --model bayesian"
  echo "  ./launch.sh train --symbols 'ETH/USDT' --timeframes '1m' --cv"
  echo "  ./launch.sh train --symbols 'ETH/USDT' --timeframes '1m' --cv --comment 'rocket fuel'"
  echo "  ./launch.sh train --symbols 'BTC/USD ETH/USDT' --timeframes '1d' --reverse"
  echo "  ./launch.sh incremental --symbol 'BTC/USDT' --timeframe '1m' --model enhanced_bayesian --chunk-size 50000"
  echo "  ./launch.sh backtest --symbols 'BTC/USD ETH/USD' --timeframes '1h 4h'"
  echo "  ./launch.sh backtest --symbols 'BTC/USD' --timeframes '1h' --walk-forward"
  echo "  ./launch.sh backtest --template crypto_majors"
  echo "  ./launch.sh analyze --symbols 'BTC/USDT' --timeframes '1m'"
  echo "  ./launch.sh analyze --file 'data/backtest_results/binance/BTC_USDT/1m_test_set_20250311_135759.csv'"
  echo "  ./launch.sh backtest --symbols 'BTC/USDT' --timeframes '1h' --position-sizing"
  echo "  ./launch.sh backtest --symbols 'BTC/USDT' --timeframes '1h' --position-sizing --no-trade-threshold 0.97"
  echo ""
}



# Set matplotlib to use Agg backend
export MPLBACKEND=Agg
# Prevent Qt from looking for Wayland
export QT_QPA_PLATFORM=offscreen

# Make sure we're in the project root directory
cd "$(dirname "$0")"

# Parse command
COMMAND=$1
shift 1
# Parse command line arguments
COMMENT=""
# Extract options
OPTIONS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols|--timeframes|--exchange|--template|--model|--no-trade-threshold|--min-position-change|--test-size|--chunk-size|--overlap)
      OPTIONS="$OPTIONS $1 $2"
      shift 2
      ;;
    --file)
      FILE="$2"
      shift 2
      ;;
    --cv|--reverse|--walk-forward|--position-sizing)
      OPTIONS="$OPTIONS $1"
      shift 1
      ;;
    --comment)
      COMMENT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Handle the comment option
if [ ! -z "$COMMENT" ] && [[ "$COMMAND" == "train" || "$COMMAND" == "incremental" ]]; then
  OPTIONS="$OPTIONS --comment \"$COMMENT\""
fi

case "$COMMAND" in
  analyze)
    echo "Analyzing backtest results..."
    python -m src.analysis.backtest_analyzer --file "$FILE" $OPTIONS
    ;;
  backtest)
    echo "Running backtest..."
    python -m src.main --mode backtest $OPTIONS
    ;;
  collect)
    echo "Collecting historical data..."
    python -m src.main --mode collect $OPTIONS
    ;;
  continue-train)
    echo "Continuing training on new data..."
    python -m src.main --mode continue-train $OPTIONS
    ;;
  docker)
    echo "Running in Docker container..."
    docker-compose up
    ;;
  docker-dev)
    echo "Running development environment in Docker..."
    docker-compose run --rm trading_bot bash
    ;;
  incremental)
    echo "Training incrementally on large dataset..."
    python -m src.main --mode incremental $OPTIONS
    ;;
  notebook)
    echo "Launching Jupyter notebook..."
    jupyter notebook
    ;;
  train)
    echo "Training model..."
    python -m src.main --mode train $OPTIONS
    ;;
  help|*)
    show_help
    ;;
esac