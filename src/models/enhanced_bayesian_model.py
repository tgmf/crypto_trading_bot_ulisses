#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Bayesian model with proper train-test split for cryptocurrency trading.

This module extends the base Bayesian model with advanced features:
- Time series cross-validation
- Proper train-test separation
- Dataset reversal testing for model consistency
- More sophisticated backtesting integration

The enhanced model places a stronger emphasis on proper evaluation and
performance consistency across different market regimes.
"""

import logging
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as tt
import arviz as az
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json
import time
import matplotlib
from datetime import datetime
matplotlib.use('Agg')
plt = matplotlib.pyplot

from .bayesian_model import BayesianModel

class EnhancedBayesianModel(BayesianModel):
    """
    Enhanced Bayesian model for trading signals with robust evaluation
    
    This model extends the core Bayesian approach with more advanced methods
    for evaluation and validation, preventing overfitting and ensuring robustness
    across different market conditions. It predicts the probability of profitable
    trading opportunities in three states: short (-1), neutral (0), and long (1).
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        # Flag for monitoring performance
        self.performance_metrics = {}
    
    def build_model(self, X_train, y_train):
        """
        Build Enhanced Bayesian ordered logistic regression model with GPU acceleration when available
        
        This method creates a Bayesian model with ordered logistic regression,
        which is appropriate for categorical outcomes with a natural ordering
        (short < neutral < long).
        
        Args:
            X_train (ndarray): Feature matrix
            y_train (ndarray): Target array with values -1, 0, 1
            
        Returns:
            tuple: (model, trace) - PyMC model and sampling trace
        """
        # Note: This method might be replaced by the factory if JAX acceleration is available

        # Adjust y_train to be 0, 1, 2 instead of -1, 0, 1 for ordered logistic
        y_train_adj = y_train + 1
        
        # Get unique classes for cutpoint creation
        unique_classes = np.unique(y_train_adj)
        n_classes = len(unique_classes)
        
        # Check current PyTensor configuration
        try:
            import pytensor
            self.logger.info(f"PyTensor device: {pytensor.config.device}")
            self.logger.info(f"PyTensor floatX: {pytensor.config.floatX}")
        except:
            self.logger.info("Could not check PyTensor configuration")
        
        # Try to build the model with robust error handling
        try:
            with pm.Model() as model:
                # Priors for model parameters
                alpha = pm.Normal("alpha", mu=0, sigma=3, shape=n_classes-1)
                betas = pm.Normal("betas", mu=0, sigma=1, shape=X_train.shape[1])
                
                # Linear predictor
                eta = pm.math.dot(X_train, betas)
                
                # Use tt.sort for ordering cutpoints
                ordered_alpha = pm.Deterministic('ordered_alpha', tt.sort(alpha))
                
                # Ordered logistic regression likelihood
                pm.OrderedLogistic("p", eta=eta, cutpoints=ordered_alpha, observed=y_train_adj)
                
                # Modern MCMC sampling with PyMC
                self.logger.info("Starting MCMC sampling...")
                start_time = time.time()
                
                trace = pm.sample(
                    draws=800,
                    tune=1000,
                    chains=4,
                    target_accept=0.9,
                    compute_convergence_checks=False
                )
                
                end_time = time.time()
                sampling_time = end_time - start_time
                self.logger.info(f"Sampling completed in {sampling_time:.2f} seconds")
                
                # Store performance metrics
                self.performance_metrics['sampling_time'] = sampling_time
            
            self.model = model
            self.trace = trace
            return model, trace
            
        except Exception as e:
            self.logger.error(f"Initial model fitting failed: {str(e)}")
            self.logger.info("Attempting fallback model with simpler priors...")
            
            # Fallback to a simpler model
            with pm.Model() as fallback_model:
                # More conservative priors
                alpha = pm.Normal("alpha", mu=0, sigma=1, shape=n_classes-1)
                betas = pm.Normal("betas", mu=0, sigma=0.5, shape=X_train.shape[1])
                
                # Linear predictor
                eta = pm.math.dot(X_train, betas)
                
                # Ordered logistic regression likelihood - no sorting to avoid errors
                pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_train_adj)
                
                # Very conservative sampling settings
                self.logger.info("Using conservative fallback sampling settings")
                start_time = time.time()
                
                fallback_trace = pm.sample(
                    draws=600,
                    tune=1000,
                    chains=1,
                    cores=1,
                    init="adapt_diag",
                    target_accept=0.95,
                    return_inferencedata=True,
                    discard_tuned_samples=True,
                    compute_convergence_checks=False
                )
                
                end_time = time.time()
                fallback_sampling_time = end_time - start_time
                self.logger.info(f"Fallback sampling completed in {fallback_sampling_time:.2f} seconds")
                
                # Store performance metrics
                self.performance_metrics['sampling_time'] = fallback_sampling_time
                self.performance_metrics['used_fallback'] = True
            
            self.model = fallback_model
            self.trace = fallback_trace
            return fallback_model, fallback_trace
    
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
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe} with time-series CV")

        # Before starting, log JAX status
        if self.using_jax_acceleration:
            self.logger.info("Using JAX acceleration for cross-validation training")
        
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
                    trace = pm.sample(1000, tune=1000, chains=4, cores=1, return_inferencedata=True)
                
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
            self.save_model(symbol, timeframe)
            
            # Plot CV performance
            self._plot_cv_results(cv_results, symbol, timeframe)
            
            return train_val_df, test_df, cv_results
            
        except Exception as e:
            self.logger.error(f"Error training model with CV: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
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
                self.save_model(symbol, timeframe)
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
            self.save_model(symbol, timeframe, suffix="_reversed")
            
            # Evaluate original and reversed models
            self.logger.info("Evaluating both models for comparison")
            
            # Load the original model for comparison
            original_model = self.__class__(self.config)
            original_model.load_model(symbol, timeframe)
            
            # Test original model on original test set
            X_orig_test = test_df[self.feature_cols].values
            X_orig_test_scaled = original_model.scaler.transform(X_orig_test)
            orig_probs = original_model.predict_probabilities(X_orig_test_scaled)
            orig_preds = np.argmax(orig_probs, axis=1) - 1  # Convert back to -1, 0, 1
            orig_acc = np.mean(orig_preds == test_df['target'].values)
            
            # Test reversed model on original training set
            X_rev_test = train_df[self.feature_cols].values
            X_rev_test_scaled = self.scaler.transform(X_rev_test)
            rev_probs = self.predict_probabilities(X_rev_test_scaled)
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
                'original_feature_importance': dict(zip(self.feature_cols, orig_importances.tolist())),
                'reversed_feature_importance': dict(zip(self.feature_cols, rev_importances.tolist()))
            }
            
            # Save comparison metrics
            metrics_dir = Path(f"models/comparisons/{symbol_safe}/{timeframe}")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / f"{timeframe}_consistency_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(comparison_metrics, f, indent=4)
            
            # Create visualization of model consistency
            self._plot_model_consistency(comparison_metrics, symbol, timeframe)
            
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
    
    def save_model(self, symbol, timeframe, comment=None, suffix=None):
        """
        Save the trained model with enhanced metadata
        
        Saves all components needed for prediction, plus additional metrics:
        - Scaler for feature normalization
        - Trace (posterior samples) from PyMC
        - Feature column list
        - Performance metrics
        - JAX acceleration status
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            comment (str): Optional comment for the model
            suffix (str): Optional suffix for the model files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First call the parent class method
            result, filename_base = super().save_model(symbol, timeframe, comment, suffix)
            if not result:
                return False
                
            # Add enhanced model specific data
            symbol_safe = symbol.replace('/', '_')
            model_type = self.__class__.__name__.lower()
        
            # Get latest metrics file to update
            model_dir = Path(f"models/{symbol_safe}/{timeframe}/{model_type}")
        
            # Use the filename_base for consistent naming
            metrics_path = model_dir / f"{filename_base}_metrics.json" 

            # Combine performance metrics with acceleration info
            combined_metrics = self.performance_metrics.copy() if hasattr(self, 'performance_metrics') else {}
            combined_metrics['model_type'] = 'enhanced_bayesian'
            combined_metrics['comment'] = comment
            
            with open(metrics_path, 'w') as f:
                json.dump(combined_metrics, f, indent=2)
        
            # Create 'latest' symlink for convenience
            latest_metrics = model_dir / "latest_metrics.json"
            if latest_metrics.exists():
                latest_metrics.unlink()
            latest_metrics.symlink_to(metrics_path.name)
            
            self.logger.info(f"Enhanced model metrics saved to {metrics_path}")
            self.logger.info(f"Also created symlink at {latest_metrics}")
            return True, filename_base
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced model: {str(e)}")
            return False, None
    
    def load_model(self, symbol, timeframe, timestamp=None, 
                    comment=None, suffix=None, version='latest'):
        """
        Load a trained model using parent implementation
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            timestamp (str, optional): Timestamp to filter by
            comment (str, optional): Comment to filter by
            suffix (str, optional): Optional suffix for the model files
            version (str): 'latest' or 'specific'
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Initialize empty performance_metrics
        self.performance_metrics = {}
        
        # Just use the parent class implementation
        return super().load_model(symbol, timeframe, timestamp, 
                                comment, suffix, version)
    
    def predict_probabilities(self, df_or_X):
        """
        Predict probabilities for each state
        
        This unified method can handle either DataFrames or pre-scaled feature arrays,
        making it flexible for different use cases. It returns a probability array:
        P(short), P(no_trade), P(long)
        
        Args:
            df_or_X: Either a DataFrame with feature columns or a pre-scaled feature array
            
        Returns:
            ndarray: Probability array [P(short), P(no_trade), P(long)]
        """
        # Ensure we have the trace loaded
        if self.trace is None:
            self.logger.error("No model trace available. Load or train a model first.")
            return None
        
        # Handle different input types
        if isinstance(df_or_X, pd.DataFrame):
            # Check for missing feature columns
            missing_cols = [col for col in self.feature_cols if col not in df_or_X.columns]
            if missing_cols:
                self.logger.error(f"Missing feature columns in data: {missing_cols}")
                return None
            
            # Extract features and scale
            X = df_or_X[self.feature_cols].values
            X_scaled = self.scaler.transform(X)
        else:
            # Assume input is already a properly scaled feature array
            X_scaled = df_or_X
        
        # Get parameter posteriors from the Bayesian model
        alpha_post = self.trace.posterior["alpha"].mean(("chain", "draw")).values
        betas_post = self.trace.posterior["betas"].mean(("chain", "draw")).values
        
        # Calculate linear predictor (dot product of features and coefficients)
        eta = np.dot(X_scaled, betas_post)
        
        # Calculate ordered logit probabilities
        probs = np.zeros((len(X_scaled), 3))
        
        # Use sigmoid to convert to probabilities
        p0 = 1 / (1 + np.exp(-(alpha_post[0] - eta)))  # P(y <= 0)
        p1 = 1 / (1 + np.exp(-(alpha_post[1] - eta)))  # P(y <= 1)
        
        # Calculate probabilities for each class
        probs[:, 0] = p0                # P(y=0) = P(y<=0)
        probs[:, 1] = p1 - p0           # P(y=1) = P(y<=1) - P(y<=0)
        probs[:, 2] = 1 - p1            # P(y=2) = 1 - P(y<=1)
        
        return probs
    
    def _predict_class(self, X, trace):
        """
        Predict class from features using trace
        
        Args:
            X (ndarray): Feature matrix (scaled)
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
    
    def _plot_cv_results(self, cv_results, symbol, timeframe):
        """
        Plot cross-validation results
        
        Creates a visualization showing training and validation accuracy
        across different cross-validation folds, helping to diagnose
        overfitting and model stability.
        
        Args:
            cv_results (dict): Dictionary with cross-validation results
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory for plots if it doesn't exist
            symbol_safe = symbol.replace('/', '_')
            model_type = self.__class__.__name__.lower()
            output_dir = Path(f"models/{symbol_safe}/{timeframe}/{model_type}")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = output_dir / f"cv_results_{timestamp}.png"
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Cross-validation plot saved to {plot_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting CV results: {str(e)}")
            return False

    def _plot_model_consistency(self, metrics, symbol, timeframe):
        """
        Create visualizations of model consistency metrics
        
        Creates plots comparing model performance and feature importance
        between original and reversed training configurations, helping to
        diagnose model robustness and potential overfitting.
        
        Args:
            metrics (dict): Dictionary with comparison metrics
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory
            symbol_safe = symbol.replace('/', '_')
            model_type = self.__class__.__name__.lower()
            output_dir = Path(f"models/comparisons/{symbol_safe}/{timeframe}/{model_type}")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = output_dir / f"consistency_plot_{timestamp}.png"
            plt.savefig(plot_file)
            plt.close(fig)
            
            self.logger.info(f"Model consistency plot saved to {plot_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting model consistency: {str(e)}")
            return False
    
    def train_multi(self, symbols, timeframes, exchange='binance', max_samples=500000):
        """
        Train the model on multiple symbols and timeframes with enhanced features
        
        This method trains a universal model that can be used across multiple
        trading pairs and timeframes. This is useful for finding patterns that
        generalize across different assets and time horizons.
        
        Args:
            symbols (list): List of trading pair symbols
            timeframes (list): List of timeframe strings
            exchange (str): Exchange name
            max_samples (int): Maximum number of samples to use for training
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Training enhanced model on multiple symbols and timeframes")
        # Log JAX status at the beginning
        if self.using_jax_acceleration:
            self.logger.info("Using JAX acceleration for multi-symbol training")
        
        # Lists to store all training data
        all_X = []
        all_y = []
        
        # Calculate maximum samples per source to stay within memory limits
        max_rows_per_source = max_samples // (len(symbols) * len(timeframes))
        
        # Keep track of data distribution
        class_distribution = {-1: 0, 0: 0, 1: 0}
        total_rows = 0
        
        # Process each symbol and timeframe
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
                    
                    # Check if test set exists and exclude it from training
                    test_file = Path(f"data/test_sets/{exchange}/{symbol_safe}/{timeframe}_test.csv")
                    if test_file.exists():
                        self.logger.info(f"Found test set for {symbol} {timeframe}, excluding from training")
                        test_df = pd.read_csv(test_file, index_col='timestamp', parse_dates=True)
                        # Exclude test data to avoid data leakage
                        df = df[~df.index.isin(test_df.index)]
                        self.logger.info(f"After removing test data, {len(df)} rows remain")
                    
                    # Create target
                    df['target'] = self.create_target(df)
                    
                    # Drop rows with NaN targets
                    df = df.dropna(subset=['target'])
                    
                    # Update class distribution
                    for cls in [-1, 0, 1]:
                        class_count = (df['target'] == cls).sum()
                        class_distribution[cls] += class_count
                        total_rows += class_count
                    
                    # Sample data if it exceeds the per-source limit
                    if len(df) > max_rows_per_source:
                        self.logger.info(f"Dataset too large, sampling {max_rows_per_source} rows from {symbol} {timeframe}")
                        
                        # Stratified sampling to maintain class distribution
                        y_values = df['target'].values
                        unique_classes, counts = np.unique(y_values, return_counts=True)
                        
                        # Calculate samples per class to maintain balance
                        # Aim for equal distribution by default
                        target_counts = {}
                        for cls in unique_classes:
                            target_counts[cls] = max_rows_per_source // len(unique_classes)
                        
                        # Adjust for class imbalance if needed
                        if np.min(counts) < target_counts[unique_classes[0]]:
                            # If we have fewer samples than target for any class, adjust ratios
                            min_cls = unique_classes[np.argmin(counts)]
                            min_count = np.min(counts)
                            
                            # Keep all samples of the minority class
                            target_counts[min_cls] = min_count
                            
                            # Adjust other classes proportionally
                            remaining_samples = max_rows_per_source - min_count
                            remaining_classes = [c for c in unique_classes if c != min_cls]
                            
                            # Distribute remaining samples proportionally
                            remaining_counts = [counts[np.where(unique_classes == c)[0][0]] for c in remaining_classes]
                            total_remaining = sum(remaining_counts)
                            
                            for i, cls in enumerate(remaining_classes):
                                if total_remaining > 0:
                                    target_counts[cls] = int(remaining_samples * (remaining_counts[i] / total_remaining))
                                else:
                                    target_counts[cls] = max_rows_per_source // len(unique_classes)
                        
                        # Sample from each class according to target counts
                        sampled_indices = []
                        for cls in unique_classes:
                            cls_indices = df.index[df['target'] == cls]
                            if len(cls_indices) > target_counts[cls]:
                                sampled_cls_indices = np.random.choice(
                                    cls_indices, 
                                    size=target_counts[cls], 
                                    replace=False
                                )
                                sampled_indices.extend(sampled_cls_indices)
                            else:
                                # If we don't have enough samples of this class, take all of them
                                sampled_indices.extend(cls_indices)
                        
                        # Sample the dataframe
                        df = df.loc[sampled_indices]
                    
                    # Extract features and target
                    X = df[self.feature_cols].values
                    y = df['target'].values
                    
                    # Append to combined datasets
                    all_X.append(X)
                    all_y.append(y)
                    self.logger.info(f"Added {len(X)} rows from {symbol} {timeframe}")
                    
                except Exception as e:
                    self.logger.error(f"EBMLOG. Error processing {symbol} {timeframe}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
            
            # Periodically check memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
                self.logger.info(f"Current memory usage: {memory_gb:.2f} GB")
            except ImportError:
                pass
        
        # Check if we have enough data
        if not all_X:
            self.logger.error("No data available for training")
            return False
        
        # Log class distribution
        if total_rows > 0:
            self.logger.info(f"Class distribution before final sampling:")
            for cls in [-1, 0, 1]:
                percentage = (class_distribution[cls] / total_rows) * 100
                self.logger.info(f"Class {cls}: {class_distribution[cls]} samples ({percentage:.2f}%)")
        
        # Combine all datasets
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        self.logger.info(f"Combined dataset has {len(X_combined)} samples")
        
        # Apply final sampling if needed to ensure we stay within memory limits
        if len(X_combined) > max_samples:
            self.logger.info(f"Combined dataset too large, sampling {max_samples} rows")
            
            # Stratified sampling for the final dataset
            unique_classes, counts = np.unique(y_combined, return_counts=True)
            
            # Calculate samples per class
            target_counts = {}
            for i, cls in enumerate(unique_classes):
                # Distribute samples proportionally to maintain distribution
                target_counts[cls] = int(max_samples * (counts[i] / len(y_combined)))
            
            # Ensure we use exactly max_samples by adjusting the largest class
            total_allocated = sum(target_counts.values())
            if total_allocated < max_samples:
                # Find the largest class
                largest_class = unique_classes[np.argmax(counts)]
                # Add remaining samples to it
                target_counts[largest_class] += (max_samples - total_allocated)
            
            # Sample from each class
            final_indices = []
            for cls in unique_classes:
                cls_indices = np.where(y_combined == cls)[0]
                if len(cls_indices) > target_counts[cls]:
                    sampled_cls_indices = np.random.choice(
                        cls_indices, 
                        size=target_counts[cls], 
                        replace=False
                    )
                    final_indices.extend(sampled_cls_indices)
                else:
                    # If we don't have enough samples of this class, take all of them
                    final_indices.extend(cls_indices)
            
            # Sample the combined dataset
            X_combined = X_combined[final_indices]
            y_combined = y_combined[final_indices]
            self.logger.info(f"Final dataset has {len(X_combined)} samples after sampling")
        
        # Scale features
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        
        # Log final class distribution
        unique_classes, counts = np.unique(y_combined, return_counts=True)
        self.logger.info(f"Final class distribution:")
        for i, cls in enumerate(unique_classes):
            percentage = (counts[i] / len(y_combined)) * 100
            self.logger.info(f"Class {cls}: {counts[i]} samples ({percentage:.2f}%)")
        
        # Build and train model
        self.logger.info(f"Building Enhanced Bayesian model with {len(X_combined)} rows")
        
        try:
            self.build_model(X_combined_scaled, y_combined)
            
            # Create a model name based on symbols and timeframes
            model_name = self._generate_multi_model_name(symbols, timeframes)
            
            # Save model
            self.save_model_multi(model_name)
            
            # Create visualization of the training process
            self._plot_training_distribution(y_combined, symbols, timeframes)
            
            self.logger.info(f"Multi-symbol enhanced model trained and saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _plot_training_distribution(self, y_combined, symbols, timeframes):
        """Create visualization of training data distribution"""
        try:
            import matplotlib.pyplot as plt
            
            # Count classes
            classes, counts = np.unique(y_combined, return_counts=True)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot class distribution
            plt.bar(['Short (-1)', 'Neutral (0)', 'Long (1)'], counts)
            
            # Add percentages
            total = len(y_combined)
            for i, count in enumerate(counts):
                percentage = (count / total) * 100
                plt.text(i, count + (max(counts) * 0.01), f"{percentage:.1f}%", 
                        ha='center', va='bottom')
            
            # Add labels
            plt.title(f'Training Data Distribution\n{", ".join(symbols)} - {", ".join(timeframes)}')
            plt.ylabel('Number of Samples')
            plt.xlabel('Classes')
            plt.grid(axis='y', alpha=0.3)
            
            # Save figure
            model_name = self._generate_multi_model_name(symbols, timeframes)
            output_dir = Path(f"models/{symbols[0]}/{timeframes[0]}/enhanced_bayesian")
            output_dir.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(output_dir / f"{model_name}_class_distribution.png")
            plt.close()
            
            self.logger.info(f"Saved class distribution visualization")
            return True
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {str(e)}")
            return False

    def _generate_multi_model_name(self, symbols, timeframes):
        """
        Generate a consistent name for multi-symbol models
        
        Creates a standardized name for models trained on multiple symbols
        and timeframes, which is used for file naming and identification.
        
        Args:
            symbols (list): List of trading pair symbols
            timeframes (list): List of timeframe strings
            
        Returns:
            str: Generated model name
        """
        symbols_str = "_".join([s.replace('/', '_') for s in symbols])
        timeframes_str = "_".join(timeframes)
        
        # Truncate if too long
        if len(symbols_str) > 40:
            symbols_str = symbols_str[:37] + "..."
            
        return f"multi_{symbols_str}_{timeframes_str}"

    def save_model_multi(self, model_name):
        """
        Save the multi-symbol model
        
        Saves a model trained on multiple symbols and timeframes with
        a special naming convention for identification.
        
        Args:
            model_name (str): Generated model name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory for multi-symbol models
            model_dir = Path("models/multi_symbol/")
            model_dir.mkdir(exist_ok=True)
            
            # Save scaler
            scaler_path = model_dir / f"{model_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save trace (posterior samples)
            trace_path = model_dir / f"{model_name}_trace.netcdf"
            az.to_netcdf(self.trace, trace_path)
            
            # Save feature columns list
            feat_cols_path = model_dir / f"{model_name}_feature_cols.pkl"
            with open(feat_cols_path, 'wb') as f:
                pickle.dump(self.feature_cols, f)
            
            self.logger.info(f"Multi-symbol model saved to {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving multi-symbol model: {str(e)}")
            return False
    
    def _calculate_accuracy(self, predictions, targets):
        """Calculate accuracy of predictions"""
        return np.mean(predictions == targets)
        
    def continue_training(self, new_data_df, symbol='BTC/USDT', timeframe='1h'):
        """
        Continue training the model with new data
        
        This method uses the current posterior distributions as priors for the new model,
        effectively performing Bayesian updating.
        
        Args:
            new_data_df (DataFrame): New data to train on
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            bool: Success or failure
        """
        # Check if a model exists
        if self.trace is None:
            self.logger.error("No existing model found for continued training")
            return False
        
        try:
            # Process new data
            self.logger.info(f"Continuing training with {len(new_data_df)} new samples")
            
            # Create target for new data if not present
            if 'target' not in new_data_df.columns:
                self.logger.info("Creating target variables for new data")
                new_data_df['target'] = self.create_target(new_data_df)
            
            new_data_df = new_data_df.dropna(subset=['target'])
            
            # Extract feature columns
            X_new = new_data_df[self.feature_cols].values
            y_new = new_data_df['target'].values
            
            # Scale new data using existing scaler
            X_new_scaled = self.scaler.transform(X_new)
            
            # Get current posterior means for parameters
            alpha_mean = self.trace.posterior["alpha"].mean(("chain", "draw")).values
            betas_mean = self.trace.posterior["betas"].mean(("chain", "draw")).values
            
            # Get standard deviations for uncertainty
            alpha_std = self.trace.posterior["alpha"].std(("chain", "draw")).values
            betas_std = self.trace.posterior["betas"].std(("chain", "draw")).values
            
            # Adjust y for ordered logistic
            y_new_adj = y_new + 1
            
            try:
                # Create new model with informed priors
                with pm.Model() as new_model:
                    # Use previous posterior as new prior
                    alpha = pm.Normal("alpha", mu=alpha_mean, sigma=alpha_std, shape=len(alpha_mean))
                    betas = pm.Normal("betas", mu=betas_mean, sigma=betas_std, shape=len(betas_mean))
                    
                    # Linear predictor
                    eta = pm.math.dot(X_new_scaled, betas)
                    
                    # Ordered logistic regression
                    p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_new_adj)
                    
                    # Sample with more conservative settings for continued training
                    self.logger.info("Sampling posterior for continued training")
                    new_trace = pm.sample(
                        draws=800, 
                        tune=1000, 
                        chains=4, 
                        cores=1, 
                        return_inferencedata=True,
                        target_accept=0.9
                    )
                
                # Update the model
                self.model = new_model
                self.trace = new_trace
                
                # Save updated model
                self.save_model(symbol, timeframe)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error in standard continue_training: {str(e)}")
                self.logger.info("Attempting fallback continued training with simpler model")
                
                # Fallback to simpler model
                with pm.Model() as fallback_model:
                    # More conservative priors and initialization
                    alpha = pm.Normal("alpha", mu=alpha_mean, sigma=alpha_std * 0.5, shape=len(alpha_mean),
                                    initval=alpha_mean)
                    betas = pm.Normal("betas", mu=betas_mean, sigma=betas_std * 0.5, shape=len(betas_mean),
                                    initval=betas_mean)
                    
                    eta = pm.math.dot(X_new_scaled, betas)
                    p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_new_adj)
                    
                    # Very conservative sampling
                    fallback_trace = pm.sample(
                        draws=600, 
                        tune=800, 
                        chains=1, 
                        cores=1, 
                        return_inferencedata=True,
                        target_accept=0.95,
                        init="adapt_diag"
                    )
                
                # Update model with fallback
                self.model = fallback_model
                self.trace = fallback_trace
                
                # Save updated model
                self.save_model(symbol, timeframe)
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error continuing training: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
    def train_with_reservoir(self, exchange='binance', symbol='BTC/USDT', timeframe='1m', max_samples=10000, new_data_df=None):
        """
        Train model using reservoir sampling to maintain a representative dataset
        
        Reservoir sampling allows the model to maintain a fixed-size representative
        dataset that combines historical and new data without biasing toward either.
        This is useful for incremental learning in non-stationary environments like
        financial markets.
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            max_samples (int): Maximum size of the reservoir
            new_data_df (DataFrame, optional): New data to add to the reservoir
            
        Returns:
            tuple: (train_df, test_df) if successful, False otherwise
        """
        self.logger.info(f"Training with reservoir sampling for {exchange} {symbol} {timeframe}")# Log JAX status at the beginning
        if self.using_jax_acceleration:
            self.logger.info("Using JAX acceleration for training with reservoir")
        
        try:
            # Check if we already have a reservoir
            symbol_safe = symbol.replace('/', '_')
            reservoir_path = Path(f"data/reservoir/{exchange}/{symbol_safe}/{timeframe}.csv")
            
            if reservoir_path.exists():
                # Load existing reservoir
                reservoir_df = pd.read_csv(reservoir_path, index_col='timestamp', parse_dates=True)
                self.logger.info(f"Loaded existing reservoir with {len(reservoir_df)} samples")
            else:
                # Create new reservoir
                reservoir_df = pd.DataFrame()
                reservoir_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If new data provided, add it using reservoir sampling
            if new_data_df is not None:
                if len(reservoir_df) < max_samples:
                    # Reservoir not full yet, just append
                    reservoir_df = pd.concat([reservoir_df, new_data_df])
                    if len(reservoir_df) > max_samples:
                        # If we exceeded max_samples, randomly sample
                        reservoir_df = reservoir_df.sample(max_samples)
                else:
                    # Reservoir full, randomly replace elements
                    for i, row in new_data_df.iterrows():
                        if np.random.random() < len(new_data_df) / (len(reservoir_df) + i):
                            # Replace a random element
                            replace_idx = np.random.randint(0, len(reservoir_df))
                            reservoir_df.iloc[replace_idx] = row
                
                # Save updated reservoir
                reservoir_df.to_csv(reservoir_path)
                self.logger.info(f"Updated reservoir with {len(reservoir_df)} samples")
            elif len(reservoir_df) == 0:
                # No reservoir and no new data provided, load historical data
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                
                if not input_file.exists():
                    self.logger.error(f"No processed data file found at {input_file}")
                    return False
                    
                historical_df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                self.logger.info(f"Loaded {len(historical_df)} historical samples for initial reservoir")
                
                # Initialize reservoir with random sample from historical data
                sample_size = min(len(historical_df), max_samples)
                reservoir_df = historical_df.sample(sample_size)
                
                # Save initial reservoir
                reservoir_df.to_csv(reservoir_path)
                self.logger.info(f"Created initial reservoir with {len(reservoir_df)} samples")
            
            # Train on the reservoir
            self.logger.info(f"Training on reservoir dataset with {len(reservoir_df)} samples")
            return self.train(exchange=exchange, symbol=symbol, timeframe=timeframe, 
                            test_size=0.2, custom_df=reservoir_df)
            
        except Exception as e:
            self.logger.error(f"Error training with reservoir: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def train_with_time_weighting(self, exchange='binance', symbol='BTC/USDT', timeframe='1h', recency_weight=2.0, all_data_df=None):
        """
        Train with time-weighted sampling (newer data gets higher probability)
        
        This method implements time-weighted sampling where more recent data points
        have a higher probability of being selected for training. This helps the
        model adapt to recent market conditions while still maintaining some
        knowledge of historical patterns.
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            recency_weight (float): Weight factor for recency (higher = more focus on recent data)
            all_data_df (DataFrame, optional): Data to sample from, if None loads from file
            
        Returns:
            tuple: (train_df, test_df) if successful, False otherwise
        """
        self.logger.info(f"Training with time-weighted sampling for {exchange} {symbol} {timeframe}")# Log JAX status at the beginning
        if self.using_jax_acceleration:
            self.logger.info("Using JAX acceleration for training with time weighting")
        
        try:
            # If no data provided, load from file
            if all_data_df is None:
                symbol_safe = symbol.replace('/', '_')
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                
                if not input_file.exists():
                    self.logger.error(f"No processed data file found at {input_file}")
                    return False
                    
                all_data_df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
                
            # Sort by time
            all_data_df = all_data_df.sort_index()
            
            # Create time-based weights (newer data gets higher weight)
            time_indices = np.arange(len(all_data_df))
            weights = np.exp(recency_weight * time_indices / len(all_data_df))
            weights = weights / weights.sum()  # Normalize
            
            # Sample with weights
            sample_size = min(len(all_data_df), 50000)  # Adjust as needed
            sampled_indices = np.random.choice(
                len(all_data_df), 
                size=sample_size, 
                replace=False, 
                p=weights
            )
            
            sampled_df = all_data_df.iloc[sampled_indices]
            self.logger.info(f"Created time-weighted sample with {len(sampled_df)} points")
            
            # Train on weighted sample
            return self.train(exchange=exchange, symbol=symbol, timeframe=timeframe, 
                            test_size=0.2, custom_df=sampled_df)
            
        except Exception as e:
            self.logger.error(f"Error training with time weighting: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
    def run_backtest_with_position_sizing(self, df, exchange='binance', symbol='BTC/USDT', 
                                            timeframe='15m', no_trade_threshold=0.96, 
                                            min_position_change=0.025, compound_returns=True, exaggerate=True):
        """
        Run backtest with continuous position sizing based on probability distributions
    
        This implementation:
        1. Treats long and short positions separately (can have both simultaneously)
        2. Position sizes directly correspond to probability values 
        3. Only updates positions when probability changes exceed threshold
        4. Optionally compounds returns to grow/shrink position sizes
        5. Optionally exaggerates position sizes to increase contrast between signals
        
        Args:
            df (DataFrame): Price data with OHLCV columns
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            no_trade_threshold (float): Threshold for no_trade probability to ignore signals
            min_position_change (float): Minimum position change to avoid fee churn
            compound_returns (bool): Whether to compound returns for position sizing
            exaggerate (bool): Whether to exaggerate position sizes based on probability differences

        Returns:
            tuple: (results_df, metrics, fig) with backtest results, performance metrics, and visualization
        """
        self.logger.info(f"Running position sizing backtest for {symbol} {timeframe}")
        
        try:
            # 1. Get probabilistic predictions
            probabilities = self.predict_probabilities(df)
            
            if probabilities is None:
                self.logger.error("Failed to get probability predictions")
                return False
                
            # 2. Create a copy of the dataframe for backtesting
            results = df.copy()
            
            # 3. Add probability columns
            results['short_prob'] = probabilities[:, 0]
            results['no_trade_prob'] = probabilities[:, 1]
            results['long_prob'] = probabilities[:, 2]
        
            # Store original probabilities for reference
            results['original_long_prob'] = results['long_prob'].copy()
            results['original_short_prob'] = results['short_prob'].copy()
            
            # 4. Apply exaggeration if enabled
            if exaggerate:
            
                # Calculate the redistributable probability (what's below no_trade_threshold)
                results['redistributable_prob'] = np.maximum(0, no_trade_threshold - results['no_trade_prob'])
                
                # Apply exaggeration to each row
                for i in range(len(results)):
                    if results.iloc[i]['no_trade_prob'] < no_trade_threshold:
                        # We have redistributable probability
                        redist_prob = results.iloc[i]['redistributable_prob']
                        long_prob = results.iloc[i]['long_prob']
                        short_prob = results.iloc[i]['short_prob']
                        
                        # Calculate the proportion of long vs short
                        total_directional = long_prob + short_prob
                        if total_directional > 0:
                            # Create a power function to exaggerate the difference
                            # Higher power = more exaggeration
                            exaggeration_power = 2.0  # Adjust this parameter for more/less exaggeration
                            
                            # Calculate normalized proportions
                            long_ratio = long_prob / total_directional
                            short_ratio = short_prob / total_directional
                            
                            # Apply power function to exaggerate differences
                            long_exaggerated = long_ratio ** exaggeration_power
                            short_exaggerated = short_ratio ** exaggeration_power
                            
                            # Renormalize to ensure they sum to 1
                            total_exaggerated = long_exaggerated + short_exaggerated
                            if total_exaggerated > 0:  # Avoid division by zero
                                long_exaggerated = long_exaggerated / total_exaggerated
                                short_exaggerated = short_exaggerated / total_exaggerated
                                
                                # Distribute redistributable probability according to exaggerated ratios
                                long_boost = redist_prob * long_exaggerated
                                short_boost = redist_prob * short_exaggerated
                                
                                # Update probabilities
                                results.iloc[i, results.columns.get_loc('long_prob')] = long_prob + long_boost
                                results.iloc[i, results.columns.get_loc('short_prob')] = short_prob + short_boost
            
            # 5. Calculate raw position sizes directly from probabilities
            # Ignore probabilities if no_trade is high
            results['raw_long_position'] = np.where(
                results['no_trade_prob'] >= no_trade_threshold,
                0,  # No position when no_trade probability is high
                results['long_prob']  # Otherwise use long probability directly
            )
            
            results['raw_short_position'] = np.where(
                results['no_trade_prob'] >= no_trade_threshold,
                0,  # No position when no_trade probability is high
                results['short_prob']  # Otherwise use short probability directly
            )
        
            # 6. Initialize compounding factor (starting with 1.0 = 100% of base size)
            if compound_returns:
                results['compound_factor'] = 1.0
            
            # 7. Apply minimum change threshold to avoid fee churn
            # Handle each position type separately
            results['long_position'] = 0.0
            results['short_position'] = 0.0
            prev_long = 0.0
            prev_short = 0.0
            
            # 8. Calculate price return
            results['price_return'] = results['close'].pct_change()
            
            # 9. Loop through data and calculate positions with optional compounding
            compound_factor = 1.0  # Start with base sizing
            realized_pnl = 0.0     # Track realized P&L for compounding
            prev_long = 0.0
            prev_short = 0.0
            
            for i in range(len(results)):
                # Get raw position signals based on probabilities
                raw_long = results.iloc[i]['raw_long_position']
                raw_short = results.iloc[i]['raw_short_position']
                
                if compound_returns:
                    # Apply current compound factor to raw positions
                    raw_long_sized = raw_long * compound_factor
                    raw_short_sized = raw_short * compound_factor
                else:
                    raw_long_sized = raw_long
                    raw_short_sized = raw_short
    
                    # Calculate potential position changes
                    long_change = abs(raw_long_sized - prev_long)
                    short_change = abs(raw_short_sized - prev_short)
                    
                    # Apply position changes only if they exceed minimum threshold
                    if long_change >= min_position_change:
                        # Position is changing - ONLY NOW do we realize P&L and compound
                        if i > 0 and prev_long > 0:
                            # Calculate P&L from previous position
                            price_return = results.iloc[i-1]['price_return'] if i > 1 else 0
                            if not np.isnan(price_return):
                                # Calculate P&L on the CLOSED portion of the position
                                closed_size = abs(raw_long_sized - prev_long)
                                direction = 1 if raw_long_sized < prev_long else -1  # Reducing or increasing
                                pnl_from_price = prev_long * price_return
                            
                                # We realize P&L proportional to position adjustment
                                realized_pnl_long = pnl_from_price * (closed_size / prev_long) * direction
                                
                                # Apply fee for the position change
                                fee_impact = closed_size * 0.0006  # 0.06% fee
                                
                                # Update compound factor ONLY on position changes
                                realized_pnl += realized_pnl_long - fee_impact
                        
                        # Update long position
                        new_long = raw_long_sized
                    else:
                        # No change to long position
                        new_long = prev_long
                            
                            
                    # Same logic for short positions
                    if short_change >= min_position_change:
                        # Position is changing - ONLY NOW do we realize P&L and compound
                        if i > 0 and prev_short > 0:
                            # Calculate P&L from previous position (shorts gain when price falls)
                            price_return = results.iloc[i-1]['price_return'] if i > 1 else 0
                            if not np.isnan(price_return):
                                # Calculate P&L on the CLOSED portion of the position
                                closed_size = abs(raw_short_sized - prev_short)
                                direction = 1 if raw_short_sized < prev_short else -1  # Reducing or increasing
                                pnl_from_price = -prev_short * price_return  # Negative for shorts
                                
                                # We realize P&L proportional to position adjustment
                                realized_pnl_short = pnl_from_price * (closed_size / prev_short) * direction
                                
                                # Apply fee for the position change
                                fee_impact = closed_size * 0.0006  # 0.06% fee
                                
                                # Update compound factor ONLY on position changes
                                realized_pnl += realized_pnl_short - fee_impact
                                
                        # Update short position
                        new_short = raw_short_sized
                    else:
                        # No change to short position
                        new_short = prev_short
                    
                    # Update compound factor based on REALIZED pnl only
                    if compound_returns and i > 0 and (long_change >= min_position_change or short_change >= min_position_change):
                        # Only update compound factor when positions change
                        compound_factor *= (1 + realized_pnl)
                        realized_pnl = 0.0  # Reset realized P&L after compounding
                    
                    # Store positions and compound factor
                    results.iloc[i, results.columns.get_loc('long_position')] = new_long
                    results.iloc[i, results.columns.get_loc('short_position')] = new_short
                    results.iloc[i, results.columns.get_loc('compound_factor')] = compound_factor
                    
                    # Update previous positions for next iteration
                    prev_long = new_long
                    prev_short = new_short
            
            # 10. Calculate net position (for compatibility)
            results['position'] = results['long_position'] - results['short_position']
            
            # 11. Calculate separate returns for long and short positions
            results['long_return'] = results['long_position'].shift(1) * results['price_return']
            results['short_return'] = -1 * results['short_position'].shift(1) * results['price_return']  # Negative for shorts
            results['strategy_return'] = results['long_return'] + results['short_return']
            
            # 12. Add fee impact - fees are proportional to position change
            fee_rate = 0.0006  # Typical fee of 0.06%
            results['long_change'] = results['long_position'].diff().abs()
            results['short_change'] = results['short_position'].diff().abs()
            results['fee_impact'] = (results['long_change'] + results['short_change']) * fee_rate
            
            # 13. Net returns after fees
            results['net_return'] = results['strategy_return'] - results['fee_impact']
            
            # 14. Calculate cumulative returns
            results['price_cumulative'] = (1 + results['price_return']).cumprod() - 1
            results['strategy_cumulative'] = (1 + results['net_return']).cumprod() - 1
            
            # 15. Calculate separate cumulative returns for long and short
            results['long_cumulative'] = (1 + results['long_return']).cumprod() - 1
            results['short_cumulative'] = (1 + results['short_return']).cumprod() - 1
            
            # 16. Calculate performance metrics
            metrics = self._calculate_position_sizing_metrics(results, min_position_change)
            
            # Add compounding information to metrics
            if compound_returns:
                metrics['compound_returns'] = True
                metrics['final_compound_factor'] = results['compound_factor'].iloc[-1]
                metrics['max_compound_factor'] = results['compound_factor'].max()
                metrics['min_compound_factor'] = results['compound_factor'].min()
                metrics['max_position_size'] = max(
                    results['long_position'].max(),
                    results['short_position'].max()
                )
            else:
                metrics['compound_returns'] = False
            
            # Add exaggeration information to metrics
            metrics['exaggerate_positions'] = exaggerate
            if exaggerate:
                metrics['avg_long_boost'] = (results['long_prob'] - results['original_long_prob']).mean()
                metrics['avg_short_boost'] = (results['short_prob'] - results['original_short_prob']).mean()
                metrics['max_long_boost'] = (results['long_prob'] - results['original_long_prob']).max()
                metrics['max_short_boost'] = (results['short_prob'] - results['original_short_prob']).max()
            
            # 17. Create visualization using result logger
            from ..utils.result_logger import ResultLogger
            result_logger = ResultLogger(getattr(self, 'config', {}))

            # Strategy type suffix based on exaggeration
            strategy_type = 'position_sizing_exaggerated' if exaggerate else 'position_sizing'

            # Save results to standardized formats
            files = result_logger.save_results(
                results, metrics, exchange, symbol, timeframe, 
                strategy_type=strategy_type, data_source='test_set'
            )

            # Create visualizations
            viz_files = result_logger.plot_results(
                results, metrics, exchange, symbol, timeframe, 
                strategy_type=strategy_type, data_source='test_set'
            )
            
            # Add visualization paths to metrics
            metrics['visualization_files'] = viz_files
            
            # Return results
            fig = None  # The ResultLogger already saved the figures
            return results, metrics, fig
            
        except Exception as e:
            self.logger.error(f"Error in position sizing backtest: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, False, False

    def _calculate_position_sizing_metrics(self, results, min_position_change):
        """
        Calculate performance metrics for position sizing backtest
        
        Args:
            results (DataFrame): Backtest results dataframe
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Filter out rows with NaN returns
        valid_results = results.dropna(subset=['net_return'])
        
        if len(valid_results) < 2:
            return {'error': 'Not enough valid data points'}
        
        # Get final returns
        final_price_return = valid_results['price_cumulative'].iloc[-1]
        final_strategy_return = valid_results['strategy_cumulative'].iloc[-1]
        final_long_return = valid_results['long_cumulative'].iloc[-1] if 'long_cumulative' in valid_results.columns else 0
        final_short_return = valid_results['short_cumulative'].iloc[-1] if 'short_cumulative' in valid_results.columns else 0
        
        # Calculate Sharpe ratio (annualized)
        # Assuming daily data, adjust for other timeframes
        timeframe_multiplier = {
            '1m': 365 * 24 * 60,
            '5m': 365 * 24 * 12,
            '15m': 365 * 24 * 4,
            '30m': 365 * 24 * 2,
            '1h': 365 * 24,
            '4h': 365 * 6,
            '1d': 365,
            '1w': 52
        }
        
        # Default to daily if timeframe not recognized
        multiplier = timeframe_multiplier.get('1d', 365)
        
        returns_mean = valid_results['net_return'].mean()
        returns_std = valid_results['net_return'].std()
        sharpe = (returns_mean / returns_std) * np.sqrt(multiplier) if returns_std > 0 else 0
        
        # Calculate max drawdown
        cum_returns = valid_results['strategy_cumulative']
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        daily_returns = valid_results.resample('D')['net_return'].sum().dropna()
        win_rate = (daily_returns > 0).mean()
        
        # Calculate profit factor
        winning_days = daily_returns[daily_returns > 0].sum()
        losing_days = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = winning_days / losing_days if losing_days != 0 else float('inf')
        
        # Calculate alpha (excess return over buy & hold)
        alpha = final_strategy_return - final_price_return
        
        # Calculate fee impact
        total_fees = valid_results['fee_impact'].sum()
        total_return_before_fees = (1 + valid_results['strategy_return']).cumprod().iloc[-1] - 1
        total_return_after_fees = (1 + valid_results['net_return']).cumprod().iloc[-1] - 1
        fee_drag = total_return_before_fees - total_return_after_fees
        
        # Count number of position changes
        long_changes = (valid_results['long_change'] > min_position_change).sum() if 'long_change' in valid_results.columns else 0
        short_changes = (valid_results['short_change'] > min_position_change).sum() if 'short_change' in valid_results.columns else 0
        
        # Position exposure metrics
        long_exposure = valid_results['long_position'].mean()
        short_exposure = valid_results['short_position'].mean()
        net_exposure = long_exposure - short_exposure
        
        # Calculate long-only metrics
        if 'long_return' in valid_results.columns:
            long_returns = valid_results['long_return']
            long_win_rate = (long_returns > 0).mean() if (valid_results['long_position'] > 0).any() else 0
            long_sharpe = (long_returns.mean() / long_returns.std()) * np.sqrt(multiplier) if long_returns.std() > 0 else 0
        else:
            long_win_rate = 0
            long_sharpe = 0
        
        # Calculate short-only metrics
        if 'short_return' in valid_results.columns:
            short_returns = valid_results['short_return']
            short_win_rate = (short_returns > 0).mean() if (valid_results['short_position'] > 0).any() else 0
            short_sharpe = (short_returns.mean() / short_returns.std()) * np.sqrt(multiplier) if short_returns.std() > 0 else 0
        else:
            short_win_rate = 0
            short_sharpe = 0
        
        # Create metrics dictionary
        metrics = {
            'total_return': final_strategy_return,
            'buy_hold_return': final_price_return,
            'long_only_return': final_long_return,
            'short_only_return': final_short_return,
            'alpha': alpha,
            'sharpe_ratio': sharpe,
            'long_sharpe': long_sharpe,
            'short_sharpe': short_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'profit_factor': profit_factor,
            'long_changes': long_changes,
            'short_changes': short_changes,
            'total_position_changes': long_changes + short_changes,
            'fee_drag': fee_drag,
            'total_trading_days': len(daily_returns),
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'avg_long_size': valid_results[valid_results['long_position'] > 0]['long_position'].mean() if (valid_results['long_position'] > 0).any() else 0,
            'avg_short_size': valid_results[valid_results['short_position'] > 0]['short_position'].mean() if (valid_results['short_position'] > 0).any() else 0
        }
        
        return metrics

    # def _plot_position_sizing_results(self, results, symbol, timeframe):
    #     """
    #     Create visualization of position sizing backtest results
        
    #     Args:
    #         results (DataFrame): Backtest results dataframe
    #         symbol (str): Trading pair symbol
    #         timeframe (str): Data timeframe
            
    #     Returns:
    #         Figure: Matplotlib figure with backtest visualization
    #     """
    #     try:
    #         # Import the visualization tool
    #         from src.visualization.visualization import VisualizationTool
            
    #         # Create a visualization tool instance
    #         # Create a minimal config if self.config is not available
    #         config = getattr(self, 'config', {'logging': {'level': 'INFO'}})
    #         viz_tool = VisualizationTool(config)
            
    #         # Format results for the visualization tool
    #         formatted_results = {
    #             'data': results,
    #             'symbol': symbol,
    #             'timeframe': timeframe,
    #             'stats': {
    #                 'win_rate': (results['net_return'] > 0).mean() if 'net_return' in results.columns else 0,
    #                 'final_return': results['strategy_cumulative'].iloc[-1] if 'strategy_cumulative' in results.columns else 0,
    #                 'long_trades': (results['position'] > 0).sum(),
    #                 'short_trades': (results['position'] < 0).sum(),
    #                 'hedged_trades': 0,  # Position sizing doesn't use hedging
    #                 'avg_win': results.loc[results['net_return'] > 0, 'net_return'].mean() 
    #                         if (results['net_return'] > 0).any() else 0,
    #                 'avg_loss': results.loc[results['net_return'] < 0, 'net_return'].mean() 
    #                         if (results['net_return'] < 0).any() else 0,
    #                 'profit_factor': abs(results.loc[results['net_return'] > 0, 'net_return'].sum() / 
    #                                 results.loc[results['net_return'] < 0, 'net_return'].sum())
    #                                 if (results['net_return'] < 0).any() and 
    #                                 (results['net_return'] > 0).any() else float('inf')
    #             }
    #         }
            
    #         # Create output directory
    #         symbol_safe = symbol.replace('/', '_')
    #         output_dir = Path(f"data/backtest_results/position_sizing/{symbol_safe}")
    #         output_dir.mkdir(parents=True, exist_ok=True)
            
    #         # Plot and save the results
    #         output_file = output_dir / f"{timeframe}_position_sizing.png"
    #         fig = viz_tool.plot_backtest_results(formatted_results, output_file, continuous_position=True)
            
    #         # Also create a metrics chart
    #         metrics_file = output_dir / f"{timeframe}_position_sizing_metrics.png"
    #         viz_tool.plot_performance_metrics(formatted_results, metrics_file)
            
    #         # Create probability distribution chart if probabilities are available
    #         if all(col in results.columns for col in ['long_prob', 'short_prob', 'no_trade_prob']):
    #             prob_file = output_dir / f"{timeframe}_position_sizing_probabilities.png"
    #             viz_tool.plot_probability_distribution(formatted_results, prob_file)
            
    #         # Log success
    #         self.logger.info(f"Position sizing visualizations saved to {output_dir}")
            
    #         return fig
            
    #     except Exception as e:
    #         self.logger.error(f"Error plotting position sizing results: {str(e)}")
    #         import traceback
    #         self.logger.error(traceback.format_exc())
            
    #         # Return a minimal figure as fallback
    #         fig, ax = plt.subplots(figsize=(8, 5))
    #         ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
    #                 ha='center', va='center', transform=ax.transAxes)
    #         return fig