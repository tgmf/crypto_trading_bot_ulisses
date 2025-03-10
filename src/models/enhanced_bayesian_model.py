#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Bayesian model with proper train/test split for cryptocurrency trading.
"""

import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import matplotlib.pyplot as plt

class EnhancedBayesianModel:
    """Bayesian model for trading signals with proper train/test separation"""
    
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
        
        # Feature columns to use - can be configured from config file
        self.feature_cols = self.config.get('model', {}).get('feature_cols', [
            'bb_pos', 'RSI_14', 'MACDh_12_26_9', 'trend_strength', 
            'volatility', 'volume_ratio', 'range', 'macd_hist_diff', 
            'rsi_diff', 'bb_squeeze'
        ])
    
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
            p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_train_adj)
            
            # Sample from the posterior
            trace = pm.sample(1000, tune=1000, chains=2, cores=1, return_inferencedata=True)
        
        self.model = model
        self.trace = trace
        return model, trace
    
    def train(self, exchange='binance', symbol='BTC/USDT', timeframe='1m', 
                test_size=0.3, n_splits=5):
        """
        Train model with proper time-series cross-validation
        
        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Data timeframe
            test_size: Proportion of data to reserve for final test set
            n_splits: Number of splits for time-series cross-validation
        """
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe} with time-series CV")
        
        try:
            # Load processed data
            symbol_safe = symbol.replace('/', '_')
            input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
            
            if not input_file.exists():
                self.logger.error(f"No processed data file found at {input_file}")
                return False
                
            df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
            
            # Create target
            df['target'] = self.create_target(df)
            
            # Drop rows with NaN targets (likely at the end due to forward window)
            df = df.dropna(subset=['target'])
            
            # 1. First split into train+validation and test sets
            train_val_size = int(len(df) * (1 - test_size))
            train_val_df = df.iloc[:train_val_size]
            test_df = df.iloc[train_val_size:]
            
            # 2. Save test data for later evaluation
            test_output_dir = Path(f"data/test_sets/{exchange}/{symbol_safe}")
            test_output_dir.mkdir(parents=True, exist_ok=True)
            test_output_file = test_output_dir / f"{timeframe}_test.csv"
            test_df.to_csv(test_output_file)
            self.logger.info(f"Reserved {len(test_df)} samples for test set, saved to {test_output_file}")
            
            # 3. Use TimeSeriesSplit for cross-validation on train_val set
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Prepare data for cross-validation
            X = train_val_df[self.feature_cols].values
            y = train_val_df['target'].values
            
            # Dictionary to store cross-validation results
            cv_results = {
                'train_indices': [],
                'val_indices': [],
                'train_scores': [],
                'val_scores': []
            }
            
            # Perform cross-validation
            self.logger.info(f"Performing {n_splits}-fold time-series cross-validation")
            
            for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                self.logger.info(f"Fold {i+1}/{n_splits}")
                
                # Get train and validation data for this fold
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_fold_scaled = scaler.fit_transform(X_train_fold)
                X_val_fold_scaled = scaler.transform(X_val_fold)
                
                # Build and train model on this fold
                self.logger.info(f"Fold {i+1}: Training on {len(X_train_fold)} samples, validating on {len(X_val_fold)} samples")
                model, trace = self.build_model(X_train_fold_scaled, y_train_fold)
                
                # Score on validation set
                val_predictions = self._predict_with_model(X_val_fold_scaled, trace)
                val_accuracy = self._calculate_accuracy(val_predictions, y_val_fold + 1)  # +1 to match model output
                
                # Score on training set
                train_predictions = self._predict_with_model(X_train_fold_scaled, trace)
                train_accuracy = self._calculate_accuracy(train_predictions, y_train_fold + 1)
                
                # Store results
                cv_results['train_indices'].append(train_idx)
                cv_results['val_indices'].append(val_idx)
                cv_results['train_scores'].append(train_accuracy)
                cv_results['val_scores'].append(val_accuracy)
                
                self.logger.info(f"Fold {i+1} results: Train accuracy = {train_accuracy:.4f}, Validation accuracy = {val_accuracy:.4f}")
            
            # 4. Train final model on all train_val data
            self.logger.info(f"Training final model on all {len(train_val_df)} train+validation samples")
            
            # Scale features for full training set
            X_scaled = self.scaler.fit_transform(X)
            
            # Build and train final model
            self.build_model(X_scaled, y)
            
            # 5. Save model and cross-validation results
            self.save_model(exchange, symbol, timeframe)
            self._save_cv_results(cv_results, exchange, symbol, timeframe)
            
            # 6. Plot cross-validation results
            self._plot_cv_results(cv_results, exchange, symbol, timeframe)
            
            return train_val_df, test_df, cv_results
            
        except Exception as e:
            self.logger.error(f"Error training model with time-series CV: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _predict_with_model(self, X, trace):
        """Make predictions using the given trace"""
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
    
    def _calculate_accuracy(self, predictions, targets):
        """Calculate accuracy of predictions"""
        return np.mean(predictions == targets)
    
    def _save_cv_results(self, cv_results, exchange, symbol, timeframe):
        """Save cross-validation results"""
        try:
            # Create directory
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Save CV results
            symbol_safe = symbol.replace('/', '_')
            cv_results_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_cv_results.pkl"
            with open(cv_results_path, 'wb') as f:
                pickle.dump(cv_results, f)
            
            self.logger.info(f"CV results saved to {cv_results_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving CV results: {str(e)}")
            return False
    
    def _plot_cv_results(self, cv_results, exchange, symbol, timeframe):
        """Plot cross-validation results"""
        try:
            # Create directory
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot train and validation scores
            x = np.arange(len(cv_results['train_scores']))
            ax.plot(x, cv_results['train_scores'], 'o-', label='Training Accuracy')
            ax.plot(x, cv_results['val_scores'], 'o-', label='Validation Accuracy')
            
            # Add mean scores
            mean_train = np.mean(cv_results['train_scores'])
            mean_val = np.mean(cv_results['val_scores'])
            ax.axhline(mean_train, linestyle='--', color='blue', alpha=0.7, 
                        label=f'Mean Train: {mean_train:.4f}')
            ax.axhline(mean_val, linestyle='--', color='orange', alpha=0.7,
                        label=f'Mean Val: {mean_val:.4f}')
            
            # Set labels and title
            ax.set_xlabel('Fold')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Time-Series Cross-Validation Results for {symbol} {timeframe}')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Fold {i+1}' for i in x])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save figure
            symbol_safe = symbol.replace('/', '_')
            plot_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_cv_plot.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
            
            self.logger.info(f"CV plot saved to {plot_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting CV results: {str(e)}")
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
            
            # Save feature columns list
            feat_cols_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_feature_cols.pkl"
            with open(feat_cols_path, 'wb') as f:
                pickle.dump(self.feature_cols, f)
            
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
            feat_cols_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_feature_cols.pkl"
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load trace
            self.trace = az.from_netcdf(trace_path)
            
            # Load feature columns if available
            if feat_cols_path.exists():
                with open(feat_cols_path, 'rb') as f:
                    self.feature_cols = pickle.load(f)
            
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
        
        # Prepare features - ensure all feature columns are present
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing feature columns in data: {missing_cols}")
            return None
        
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