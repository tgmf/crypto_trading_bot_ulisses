#!/bin/bash

# Trading Bot Launch Script

# Function to print help message
show_help() {
  echo "Cryptocurrency Trading Bot"
  echo ""
  echo "Usage: ./launch.sh [command]"
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
}

# Make sure we're in the project root directory
cd "$(dirname "$0")"

# Parse command
case "$1" in
  collect)
    echo "Collecting historical data..."
    python src/main.py --mode collect
    ;;
  train)
    echo "Training model..."
    python src/main.py --mode train
    ;;
  backtest)
    echo "Running backtest..."
    python src/main.py --mode backtest
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