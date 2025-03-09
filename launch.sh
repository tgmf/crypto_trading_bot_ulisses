#!/bin/bash

# Trading Bot Launch Script

# Function to print help message
show_help() {
  echo "Cryptocurrency Trading Bot"
  echo ""
  echo "Usage: ./launch.sh [command] [options]"
  echo ""
  echo "Commands:"
  echo "  collect     - Collect historical data from exchanges"
  echo "  train       - Train trading model on collected data"
  echo "  backtest    - Backtest trained model"
  echo "  notebook    - Launch Jupyter notebook for analysis"
  echo "  docker      - Run in Docker container"
  echo "  docker-dev  - Run development environment in Docker"
  echo "  help        - Show this help message"
  echo ""
  echo "Options:"
  echo "  --symbols 'BTC/USD ETH/USD'   - Specify symbols to process"
  echo "  --timeframes '1h 4h'          - Specify timeframes to process"
  echo "  --exchange binance            - Specify exchange to use"
  echo "  --template crypto_majors      - Use a predefined template"
  echo ""
  echo "Examples:"
  echo "  ./launch.sh train --symbols 'BTC/USD' --timeframes '1h'"
  echo "  ./launch.sh backtest --symbols 'BTC/USD ETH/USD' --timeframes '1h 4h'"
  echo "  ./launch.sh backtest --template crypto_majors"
  echo ""
}

# Make sure we're in the project root directory
cd "$(dirname "$0")"

# Parse command
COMMAND=$1
shift 1

# Extract options
OPTIONS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols|--timeframes|--exchange|--template)
      OPTIONS="$OPTIONS $1 $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

case "$COMMAND" in
  collect)
    echo "Collecting historical data..."
    python -m src.main --mode collect $OPTIONS
    ;;
  train)
    echo "Training model..."
    python -m src.main --mode train $OPTIONS
    ;;
  backtest)
    echo "Running backtest..."
    python -m src.main --mode backtest $OPTIONS
    ;;
  notebook)
    echo "Launching Jupyter notebook..."
    jupyter notebook
    ;;
  docker)
    echo "Running in Docker container..."
    docker-compose up
    ;;
  docker-dev)
    echo "Running development environment in Docker..."
    docker-compose run --rm trading_bot bash
    ;;
  help|*)
    show_help
    ;;
esac