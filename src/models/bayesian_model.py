#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bayesian model implementation for cryptocurrency trading.
"""

import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import json
import matplotlib.pyplot as plt

class BayesianModel:
    """Bayesian model for trading signals"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        
        # Model parameters
        self.fee_rate = self.config.get('backtesting', {}).get('fee_rate', 0.0006)
        self.min_profit = self.config.get('backtesting', {}).get('min_profit_target', 0.008)
        
        # Feature columns to use
        self.feature_cols = [
            'bb_pos', 'RSI_14', 'MACDh_12_26_9', 'trend_strength', 
            'volatility', 'volume_ratio', 'range', 'macd_hist_diff', 
            'rsi_diff', 'bb_squeeze'
        ]
    
    def create_target(self, df, forward_window=20):
        """
        Create target variable for training:
        -1 = Short opportunity
        0 = No trade
        1 = Long opportunity
        """
        n_samples = len(df)
        targets = np.zeros(n_samples)
        
        # Calculate total fee cost for round trip
        fee_cost = self.fee_rate * 2
        
        for i in range(n_samples - forward_window):
            entry_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+forward_window+1].values
            
            # Calculate potential returns for long positions
            long_returns = future_prices / entry_price - 1 - fee_cost
            max_long_return = np.max(long_returns) if len(long_returns) > 0 else -np.inf
            
            # Calculate potential returns for short positions
            short_returns = 1 - (future_prices / entry_price) - fee_cost
            max_short_return = np.max(short_returns) if len(short_returns) > 0 else -np.inf
            
            # Determine the trade direction with the highest potential return
            if max_long_return >= self.min_profit and max_long_return > max_short_return:
                targets[i] = 1  # Long opportunity
            elif max_short_return >= self.min_profit and max_short_return > max_long_return:
                targets[i] = -1  # Short opportunity
            else:
                targets[i] = 0  # No trade opportunity
        
        return targets
    
    def build_model(self, X_train, y_train):
        """Build Bayesian model for three-state classification"""
        # Adjust y_train to be 0, 1, 2 instead of -1, 0, 1
        y_train_adj = y_train + 1
        
        unique_classes = np.unique(y_train_adj)
        n_classes = len(unique_classes)
        
        with pm.Model() as model:
            # Priors for unknown model parameters
            # For ordered logistic, we need n_classes-1 cutpoints
            alpha = pm.Normal("alpha", mu=0, sigma=10, shape=n_classes-1)
            
            # Coefficients for each feature
            betas = pm.Normal("betas", mu=0, sigma=2, shape=X_train.shape[1])
            
            # Expected values of latent variable
            eta = pm.math.dot(X_train, betas)
            
            # Ordered logistic regression
            # NOTE: No need to add alpha to eta, as cutpoints handles that
            p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_train_adj)
            
            # Sample from the posterior
            trace = pm.sample(1000, tune=1000, chains=2, cores=1, return_inferencedata=True)
        
        self.model = model
        self.trace = trace
        return model, trace
    
    def train(self, exchange='binance', symbol='BTC/USDT', timeframe='1m'):
        """Train the model on processed data"""
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe}")
        
        try:
            # Load data
            symbol_safe = symbol.replace('/', '_')
            input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
            df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
            
            # Create target
            df['target'] = self.create_target(df)
            
            # Chronological split - use 70% for training, 30% for testing
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # Save test data for later evaluation
            test_output_dir = Path(f"data/test_sets/{exchange}/{symbol_safe}")
            test_output_dir.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(test_output_dir / f"{timeframe}_test.csv")
            
            # Prepare training data
            X_train = train_df[self.feature_cols].values
            y_train = train_df['target'].values
            
            # Scale features (only fit on training data)
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Build and train model
            self.build_model(X_train_scaled, y_train)
            self.save_model(exchange, symbol, timeframe)
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False
    
    def save_model(self, exchange, symbol, timeframe):
        """Save the trained model"""
        try:
            # Create directory
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Save scaler
            symbol_safe = symbol.replace('/', '_')
            scaler_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save trace (posterior samples)
            trace_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_trace.netcdf"
            az.to_netcdf(self.trace, trace_path)
            
            self.logger.info(f"Model saved to {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, exchange, symbol, timeframe):
        """Load a trained model"""
        try:
            # Create paths
            model_dir = Path("models")
            symbol_safe = symbol.replace('/', '_')
            
            scaler_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_scaler.pkl"
            trace_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_trace.netcdf"
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load trace
            self.trace = az.from_netcdf(trace_path)
            
            self.logger.info(f"Model loaded from {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_probabilities(self, df):
        """
        Predict probabilities for each state:
        P(short), P(no_trade), P(long)
        
        Returns a Nx3 array of probabilities
        """
        # Ensure we have the trace loaded
        if self.trace is None:
            self.logger.error("No model trace available. Load or train a model first.")
            return None
        
        # Prepare features
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get parameter posteriors
        alpha_post = self.trace.posterior["alpha"].mean(("chain", "draw")).values
        betas_post = self.trace.posterior["betas"].mean(("chain", "draw")).values
        
        # Calculate linear predictor
        eta = np.dot(X_scaled, betas_post)
        
        # Calculate ordered logit probabilities
        probs = np.zeros((len(X), 3))
        
        # Use sigmoid to convert to probabilities
        p0 = 1 / (1 + np.exp(-(alpha_post[0] - eta)))  # P(y <= 0)
        p1 = 1 / (1 + np.exp(-(alpha_post[1] - eta)))  # P(y <= 1)
        
        probs[:, 0] = p0  # P(y=0) = P(y<=0)
        probs[:, 1] = p1 - p0  # P(y=1) = P(y<=1) - P(y<=0)
        probs[:, 2] = 1 - p1  # P(y=2) = 1 - P(y<=1)
        
        return probs
    
    # Prototyping the multi-symbol model (not ready for backtesting)
    def train_multi(self, symbols, timeframes, exchange='binance'):
        """Train the model on multiple symbols and timeframes"""
        self.logger.info(f"Training model on multiple symbols and timeframes")
        
        # Lists to store all training data
        all_X = []
        all_y = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                self.logger.info(f"Processing {symbol} {timeframe}")
                
                try:
                    # Load processed data
                    symbol_safe = symbol.replace('/', '_')
                    input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                    
                    if not input_file.exists():
                        self.logger.warning(f"No processed data file found at {input_file}")
                        continue
                        
                    df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                    
                    # Create target
                    df['target'] = self.create_target(df)
                    
                    # Prepare training data
                    X = df[self.feature_cols].values
                    y = df['target'].values
                    
                    # Append to combined datasets
                    all_X.append(X)
                    all_y.append(y)
                    self.logger.info(f"Added {len(X)} rows from {symbol} {timeframe}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")
                    continue
        
        if not all_X:
            self.logger.error("No data available for training")
            return False
        
        # Combine all datasets
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        # Scale features
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        
        # Build and train model
        self.logger.info(f"Building Bayesian model with {len(X_combined)} rows")
        self.build_model(X_combined_scaled, y_combined)
        
        # Create a model name based on symbols and timeframes
        model_name = self._generate_multi_model_name(symbols, timeframes)
        
        # Save model
        self.save_model_multi(exchange, model_name)
        
        return True

    def _generate_multi_model_name(self, symbols, timeframes):
        """Generate a consistent name for multi-symbol models"""
        symbols_str = "_".join([s.replace('/', '_') for s in symbols])
        timeframes_str = "_".join(timeframes)
        
        # Truncate if too long
        if len(symbols_str) > 40:
            symbols_str = symbols_str[:37] + "..."
            
        return f"multi_{symbols_str}_{timeframes_str}"

    def save_model_multi(self, exchange, model_name):
        """Save the multi-symbol model"""
        try:
            # Create directory
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Save scaler
            scaler_path = model_dir / f"{exchange}_{model_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save trace (posterior samples)
            trace_path = model_dir / f"{exchange}_{model_name}_trace.netcdf"
            az.to_netcdf(self.trace, trace_path)
            
            self.logger.info(f"Multi-symbol model saved to {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving multi-symbol model: {str(e)}")
            return False
        
    def train_with_cv(self, exchange='binance', symbol='BTC/USDT', timeframe='1h', 
                test_size=0.2, n_splits=5, gap=0):
        """
        Train with time-series cross-validation and proper test set separation
        
        This advanced implementation:
        1. Loads data and creates target variables
        2. Reserves a final hold-out test set (completely untouched)
        3. Performs time-series cross-validation on the remaining data
        4. Trains the final model on all training+validation data
        5. Saves the model and test set for later evaluation
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            test_size (float): Proportion of data to reserve for final testing
            n_splits (int): Number of CV folds
            gap (int): Gap between train and validation sets
            
        Returns:
            tuple: (train_val_df, test_df, cv_results) if successful, False otherwise
        """
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe} with CV")
        
        try:
            # Load processed data
            symbol_safe = symbol.replace('/', '_')
            input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
            
            if not input_file.exists():
                self.logger.error(f"No processed data file found at {input_file}")
                return False
                
            df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
            
            # Create target
            self.logger.info("Creating target variables")
            df['target'] = self.create_target(df)
            
            # Drop rows with NaN targets (likely at the end due to forward window)
            df = df.dropna(subset=['target'])
            
            # 1. First split into train+validation and test sets (chronologically)
            train_val_size = int(len(df) * (1 - test_size))
            train_val_df = df.iloc[:train_val_size]
            test_df = df.iloc[train_val_size:]
            
            self.logger.info(f"Reserved {len(test_df)} samples for test set")
            
            # 2. Save test data for later evaluation
            test_output_dir = Path(f"data/test_sets/{exchange}/{symbol_safe}")
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            test_output_file = test_output_dir / f"{timeframe}_test.csv"
            test_df.to_csv(test_output_file)
            self.logger.info(f"Saved test set to {test_output_file}")
            
            # 3. Set up time series cross-validation on train_val set
            X = train_val_df[self.feature_cols].values
            y = train_val_df['target'].values
            
            # Store CV results
            cv_results = {
                'fold': [],
                'train_accuracy': [],
                'val_accuracy': [],
                'train_indices': [],
                'val_indices': []
            }
            
            # Time Series Split for cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
            
            # 4. Perform cross-validation
            self.logger.info(f"Performing {n_splits}-fold time-series cross-validation")
            
            for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                fold = i + 1
                self.logger.info(f"Processing fold {fold}/{n_splits}")
                
                # Split data for this fold
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Create a new scaler for this fold
                fold_scaler = StandardScaler()
                
                # Scale features (fit only on training data)
                X_train_fold_scaled = fold_scaler.fit_transform(X_train_fold)
                X_val_fold_scaled = fold_scaler.transform(X_val_fold)
                
                # Build and train model for this fold
                with pm.Model() as model:
                    # Create priors for model parameters
                    # For ordered logistic, we need n_classes-1 cutpoints
                    alpha = pm.Normal("alpha", mu=0, sigma=10, shape=2)
                    
                    # Coefficients for each feature
                    betas = pm.Normal("betas", mu=0, sigma=2, shape=X_train_fold_scaled.shape[1])
                    
                    # Expected values of latent variable
                    eta = pm.math.dot(X_train_fold_scaled, betas)
                    
                    # Ordered logistic regression
                    # Add 1 to y_train_fold to convert from (-1,0,1) to (0,1,2)
                    y_train_adj = y_train_fold + 1
                    
                    p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_train_adj)
                    
                    # Sample from the posterior
                    trace = pm.sample(1000, tune=1000, chains=2, cores=1, return_inferencedata=True)
                
                # Evaluate on training data
                train_preds = self._predict_class(X_train_fold_scaled, trace)
                train_accuracy = np.mean(train_preds == (y_train_fold + 1))
                
                # Evaluate on validation data
                val_preds = self._predict_class(X_val_fold_scaled, trace)
                val_accuracy = np.mean(val_preds == (y_val_fold + 1))
                
                # Store results for this fold
                cv_results['fold'].append(fold)
                cv_results['train_indices'].append(train_idx)
                cv_results['val_indices'].append(val_idx)
                cv_results['train_accuracy'].append(train_accuracy)
                cv_results['val_accuracy'].append(val_accuracy)
                
                self.logger.info(f"Fold {fold} results: Train acc={train_accuracy:.4f}, Val acc={val_accuracy:.4f}")
            
            # Calculate average performance
            avg_train_acc = np.mean(cv_results['train_accuracy'])
            avg_val_acc = np.mean(cv_results['val_accuracy'])
            
            self.logger.info(f"CV complete. Avg train accuracy: {avg_train_acc:.4f}, Avg val accuracy: {avg_val_acc:.4f}")
            
            # 5. Final model training on all train_val data
            self.logger.info(f"Training final model on all {len(train_val_df)} training+validation samples")
            
            # Prepare all training data
            X_all = train_val_df[self.feature_cols].values
            y_all = train_val_df['target'].values
            
            # Scale features using all training data
            X_all_scaled = self.scaler.fit_transform(X_all)
            
            # Build and train final model
            self.build_model(X_all_scaled, y_all)
            
            # 6. Save model
            self.save_model(exchange, symbol, timeframe)
            
            # Plot CV performance
            self._plot_cv_results(cv_results, exchange, symbol, timeframe)
            
            return train_val_df, test_df, cv_results
            
        except Exception as e:
            self.logger.error(f"Error training model with CV: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def _predict_class(self, X, trace):
        """
        Predict class from features using trace
        
        Args:
            X (ndarray): Feature matrix
            trace: PyMC trace with posterior samples
            
        Returns:
            ndarray: Class predictions (0, 1, or 2)
        """
        # Get parameter posteriors
        alpha_post = trace.posterior["alpha"].mean(("chain", "draw")).values
        betas_post = trace.posterior["betas"].mean(("chain", "draw")).values
        
        # Calculate linear predictor
        eta = np.dot(X, betas_post)
        
        # Calculate ordered logit probabilities
        probs = np.zeros((len(X), 3))
        
        # Use sigmoid to convert to probabilities
        p0 = 1 / (1 + np.exp(-(alpha_post[0] - eta)))  # P(y <= 0)
        p1 = 1 / (1 + np.exp(-(alpha_post[1] - eta)))  # P(y <= 1)
        
        probs[:, 0] = p0  # P(y=0) = P(y<=0)
        probs[:, 1] = p1 - p0  # P(y=1) = P(y<=1) - P(y<=0)
        probs[:, 2] = 1 - p1  # P(y=2) = 1 - P(y<=1)
        
        # Return most likely class
        return np.argmax(probs, axis=1)
        
    def _plot_cv_results(self, cv_results, exchange, symbol, timeframe):
        """
        Plot cross-validation results
        
        Args:
            cv_results (dict): Dictionary with cross-validation results
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
        """
        try:
            # Create directory for plots if it doesn't exist
            symbol_safe = symbol.replace('/', '_')
            output_dir = Path(f"models/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot train and validation accuracies
            folds = cv_results['fold']
            train_accs = cv_results['train_accuracy']
            val_accs = cv_results['val_accuracy']
            
            plt.plot(folds, train_accs, 'o-', label='Training Accuracy')
            plt.plot(folds, val_accs, 'o-', label='Validation Accuracy')
            
            # Add mean lines
            avg_train = np.mean(train_accs)
            avg_val = np.mean(val_accs)
            
            plt.axhline(avg_train, linestyle='--', color='blue', alpha=0.7, 
                        label=f'Mean Train: {avg_train:.4f}')
            plt.axhline(avg_val, linestyle='--', color='orange', alpha=0.7,
                        label=f'Mean Val: {avg_val:.4f}')
            
            # Add labels and legends
            plt.title(f'Time Series Cross-Validation - {symbol} {timeframe}')
            plt.xlabel('Fold')
            plt.ylabel('Accuracy')
            plt.xticks(folds)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            plot_file = output_dir / f"cv_results_{timeframe}.png"
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Cross-validation plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting CV results: {str(e)}")
            
    def train_with_reversed_datasets(self, exchange='binance', symbol='BTC/USDT', timeframe='1h', test_size=0.3):
        """
        Train model with reversed datasets to test consistency and robustness
        
        This method:
        1. Loads the original train-test split (if available)
        2. If not, creates a new split and first runs normal training
        3. Reverses the datasets (trains on test, tests on train)
        4. Saves the model with a 'reversed' indicator
        5. Evaluates and compares performance on both approaches
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            test_size (float): Proportion of data to use for testing (0.0-1.0)
            
        Returns:
            tuple: (train_df, test_df, comparison_metrics) if successful, False otherwise
        """
        self.logger.info(f"Training model with reversed datasets for {exchange} {symbol} {timeframe}")
        
        try:
            # First, check if we already have a test set
            symbol_safe = symbol.replace('/', '_')
            test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            if test_file.exists():
                # Load existing test set
                self.logger.info(f"Loading existing test set from {test_file}")
                test_df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                
                # Load original data to get the training set
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                full_df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                
                # Identify training data (all data not in test set)
                # We match by index (timestamp) rather than trying to guess split ratio
                train_df = full_df[~full_df.index.isin(test_df.index)].copy()
                
                self.logger.info(f"Identified original training set with {len(train_df)} samples")
            else:
                # No existing test set, create a new split and run normal training first
                self.logger.info("No existing test set found. Creating new train-test split")
                
                # Load data
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                full_df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                
                # Create target
                full_df['target'] = self.create_target(full_df)
                
                # Drop rows with NaN targets
                full_df = full_df.dropna(subset=['target'])
                
                # Chronological train-test split
                train_size = int(len(full_df) * (1 - test_size))
                train_df = full_df.iloc[:train_size].copy()
                test_df = full_df.iloc[train_size:].copy()
                
                self.logger.info(f"Created new train-test split: train={len(train_df)}, test={len(test_df)} samples")
                
                # Save test set for future use
                test_output_dir = Path(f"data/test_sets/{exchange}/{symbol_safe}")
                test_output_dir.mkdir(parents=True, exist_ok=True)
                test_output_file = test_output_dir / f"{timeframe}_test.csv"
                test_df.to_csv(test_output_file)
                self.logger.info(f"Saved test set to {test_output_file}")
                
                # Run normal training first
                self.logger.info("Running normal training first")
                X_train = train_df[self.feature_cols].values
                y_train = train_df['target'].values
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                
                # Build and train model
                self.build_model(X_train_scaled, y_train)
                
                # Save original model
                self.save_model(exchange, symbol, timeframe)
                self.logger.info("Original model trained and saved")
            
            # Now for the reversed training
            self.logger.info("Starting reversed training (train on test, test on train)")
            
            # Ensure both datasets have target variable
            if 'target' not in train_df.columns:
                train_df['target'] = self.create_target(train_df)
            if 'target' not in test_df.columns:
                test_df['target'] = self.create_target(test_df)
            
            # Prepare reversed training
            # Original test set becomes training data
            X_train_reversed = test_df[self.feature_cols].values
            y_train_reversed = test_df['target'].values
            
            # Create a new scaler for the reversed model
            self.scaler = StandardScaler()
            
            # Scale features
            X_train_reversed_scaled = self.scaler.fit_transform(X_train_reversed)
            
            # Build and train reversed model
            self.logger.info(f"Training reversed model on {len(test_df)} samples")
            self.build_model(X_train_reversed_scaled, y_train_reversed)
            
            # Save reversed model with indicator
            self.save_model(exchange, symbol, f"{timeframe}_reversed")
            
            # Evaluate original and reversed models
            self.logger.info("Evaluating both models for comparison")
            
            # Load the original model for comparison
            original_model = self.__class__(self.config)
            original_model.load_model(exchange, symbol, timeframe)
            
            # Test original model on original test set
            X_orig_test = test_df[self.feature_cols].values
            X_orig_test_scaled = original_model.scaler.transform(X_orig_test)
            orig_probs = original_model._predict_probs(X_orig_test_scaled)
            orig_preds = np.argmax(orig_probs, axis=1) - 1  # Convert back to -1, 0, 1
            orig_acc = np.mean(orig_preds == test_df['target'].values)
            
            # Test reversed model on original training set
            X_rev_test = train_df[self.feature_cols].values
            X_rev_test_scaled = self.scaler.transform(X_rev_test)
            rev_probs = self._predict_probs(X_rev_test_scaled)
            rev_preds = np.argmax(rev_probs, axis=1) - 1  # Convert back to -1, 0, 1
            rev_acc = np.mean(rev_preds == train_df['target'].values)
            
            # Compare feature importances
            orig_importances = np.abs(original_model.trace.posterior["betas"].mean(("chain", "draw")).values)
            rev_importances = np.abs(self.trace.posterior["betas"].mean(("chain", "draw")).values)
            
            orig_ranks = np.argsort(-orig_importances)
            rev_ranks = np.argsort(-rev_importances)
            
            # Calculate rank correlation
            rank_correlation = np.corrcoef(orig_ranks, rev_ranks)[0, 1]
            
            # Assemble comparison metrics
            comparison_metrics = {
                'original_accuracy': orig_acc,
                'reversed_accuracy': rev_acc,
                'accuracy_difference': abs(orig_acc - rev_acc),
                'feature_importance_correlation': rank_correlation,
                'original_feature_importance': dict(zip(self.feature_cols, orig_importances)),
                'reversed_feature_importance': dict(zip(self.feature_cols, rev_importances))
            }
            
            # Save comparison metrics
            metrics_dir = Path(f"models/comparisons/{exchange}/{symbol_safe}")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / f"{timeframe}_consistency_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(comparison_metrics, f, indent=4)
            
            # Create visualization of model consistency
            self._plot_model_consistency(comparison_metrics, exchange, symbol, timeframe)
            
            # Log summary
            self.logger.info(f"Model consistency evaluation complete:")
            self.logger.info(f"Original accuracy: {orig_acc:.4f}")
            self.logger.info(f"Reversed accuracy: {rev_acc:.4f}")
            self.logger.info(f"Accuracy difference: {abs(orig_acc - rev_acc):.4f}")
            self.logger.info(f"Feature importance correlation: {rank_correlation:.4f}")
            
            # Provide interpretation
            if abs(orig_acc - rev_acc) < 0.1 and rank_correlation > 0.7:
                self.logger.info("CONCLUSION: Model shows good consistency across datasets")
            elif abs(orig_acc - rev_acc) < 0.2 and rank_correlation > 0.5:
                self.logger.info("CONCLUSION: Model shows moderate consistency across datasets")
            else:
                self.logger.warning("CONCLUSION: Model shows poor consistency, may be overfit or unstable")
            
            return train_df, test_df, comparison_metrics
            
        except Exception as e:
            self.logger.error(f"Error training with reversed datasets: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def _predict_probs(self, X):
        """
        Predict probabilities using the current model
        
        Args:
            X (ndarray): Feature matrix (scaled)
            
        Returns:
            ndarray: Probability array [P(short), P(no_trade), P(long)]
        """
        # Get parameter posteriors
        alpha_post = self.trace.posterior["alpha"].mean(("chain", "draw")).values
        betas_post = self.trace.posterior["betas"].mean(("chain", "draw")).values
        
        # Calculate linear predictor
        eta = np.dot(X, betas_post)
        
        # Calculate ordered logit probabilities
        probs = np.zeros((len(X), 3))
        
        # Use sigmoid to convert to probabilities
        p0 = 1 / (1 + np.exp(-(alpha_post[0] - eta)))  # P(y <= 0)
        p1 = 1 / (1 + np.exp(-(alpha_post[1] - eta)))  # P(y <= 1)
        
        probs[:, 0] = p0  # P(y=0) = P(y<=0)
        probs[:, 1] = p1 - p0  # P(y=1) = P(y<=1) - P(y<=0)
        probs[:, 2] = 1 - p1  # P(y=2) = 1 - P(y<=1)
        
        return probs

    def _plot_model_consistency(self, metrics, exchange, symbol, timeframe):
        """
        Create visualizations of model consistency metrics
        
        Args:
            metrics (dict): Dictionary with comparison metrics
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
        """
        try:
            # Create output directory
            symbol_safe = symbol.replace('/', '_')
            output_dir = Path(f"models/comparisons/{exchange}/{symbol_safe}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Accuracy comparison
            accuracies = [metrics['original_accuracy'], metrics['reversed_accuracy']]
            axes[0].bar(['Original', 'Reversed'], accuracies, color=['blue', 'orange'])
            axes[0].set_ylim(0, 1)
            axes[0].set_title('Accuracy Comparison')
            axes[0].set_ylabel('Accuracy')
            
            # Add text annotations
            for i, v in enumerate(accuracies):
                axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center')
            
            # Add difference annotation
            diff = metrics['accuracy_difference']
            axes[0].text(0.5, 0.5, f'Difference: {diff:.4f}', 
                    ha='center', va='center', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                    transform=axes[0].transAxes)
            
            # Plot 2: Feature importance comparison
            orig_imp = metrics['original_feature_importance']
            rev_imp = metrics['reversed_feature_importance']
            
            # Sort features by original importance
            sorted_features = sorted(orig_imp.keys(), key=lambda x: orig_imp[x], reverse=True)
            
            # Prepare data for bar chart
            orig_values = [orig_imp[f] for f in sorted_features]
            rev_values = [rev_imp[f] for f in sorted_features]
            
            # Get x positions
            x = np.arange(len(sorted_features))
            width = 0.35
            
            # Create grouped bar chart
            axes[1].bar(x - width/2, orig_values, width, label='Original')
            axes[1].bar(x + width/2, rev_values, width, label='Reversed')
            
            axes[1].set_title('Feature Importance Comparison')
            axes[1].set_ylabel('Absolute Importance')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(sorted_features, rotation=45, ha='right')
            axes[1].legend()
            
            # Add correlation annotation
            corr = metrics['feature_importance_correlation']
            axes[1].text(0.5, 0.9, f'Rank Correlation: {corr:.4f}', 
                    ha='center', va='center', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                    transform=axes[1].transAxes)
            
            # Set overall title
            plt.suptitle(f'Model Consistency Evaluation - {symbol} {timeframe}', fontsize=16)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save figure
            plot_file = output_dir / f"{timeframe}_consistency_plot.png"
            plt.savefig(plot_file)
            plt.close(fig)
            
            self.logger.info(f"Model consistency plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting model consistency: {str(e)}")