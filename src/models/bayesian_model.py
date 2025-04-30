#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bayesian model implementation for cryptocurrency trading.

This module implements a Bayesian ordered logistic regression model for
predicting trading signals. Key features include:
- Three-state classification (short, neutral, long)
- Fee-aware target creation
- Proper train-test separation
- Dataset reversal testing for model consistency
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
import json
import matplotlib.pyplot as plt
from datetime import datetime
from ..core.result_logger import ResultLogger
from ..core.param_manager import ParamManager
from ..data.data_context import DataContext

class BayesianModel:
    """
    Bayesian model for trading signals using ordered logistic regression
    
    This model predicts the probability of profitable trading opportunities in
    three states: short (-1), neutral (0), and long (1). It uses Bayesian inference
    to quantify uncertainty and incorporates trading fees directly into the target
    creation process.
    """
    
    def __init__(self, params):
        """
        Initialize with parameter manager
    
        Args:
            params: ParamManager instance
        """
        
        self.params = params
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.data_context_metadata = None 
        
        # Feature columns to use for prediction
        self.feature_cols = self.params.get('model', 'feature_cols', default=[
            'bb_pos', 'RSI_14', 'MACDh_12_26_9', 'trend_strength', 
            'volatility', 'volume_ratio', 'range', 'macd_hist_diff', 
            'rsi_diff', 'bb_squeeze'
        ])
        
        # Define support/resistance feature columns for easy selection
        self.sr_feature_cols = []
    
        # Check if S/R features are enabled
        if self.params.get('model', 'features', 'include_sr', default=False):
            lookbacks = self.params.get('model', 'features', 'sr_lookbacks', default=[50])
            include_breakouts = self.params.get('model', 'features', 'include_sr_breakouts', default=True)
            
            # Add base S/R features
            for lb in lookbacks:
                self.sr_feature_cols.extend([
                    f'dist_to_resistance_{lb}',
                    f'dist_to_support_{lb}'
                ])
                
                # Add breakout features if enabled
                if include_breakouts:
                    self.sr_feature_cols.extend([
                        f'resistance_break_{lb}',
                        f'support_break_{lb}'
                    ])

            self.feature_cols.extend(self.sr_feature_cols)
    
    # create_target() v 0.2
    def create_target(self, df, forward_window=40):
        """
        Create target variable for training
        
        This method looks forward in time to identify profitable trading opportunities,
        accounting for transaction fees. The target values are:
        -1 = Short opportunity (profitable short trade)
        0 = No trade opportunity (neither long nor short is profitable)
        1 = Long opportunity (profitable long trade)
        
        Args:
            df (DataFrame): Price data with OHLCV columns
            forward_window (int): Number of periods to look ahead for profit opportunities
            
        Returns:
            ndarray: Array of target values (-1, 0, or 1)
        """
        n_samples = len(df)
        targets = np.zeros(n_samples)
        
        # Extract model parameters from params
        fee_rate = self.params.get('exchange', 'fee_rate', default=0.0006)
        min_profit = self.params.get('training', 'min_profit_target', default=fee_rate*2)
        forward_window = self.params.get('training', 'forward_window', default=forward_window)
        
        # Increase no_trade buffer to create clearer separation between classes
        no_trade_buffer = min_profit
        min_directional_strength = min_profit * 0.8  # Clear signal required
    
        # Add volatility detection for sideways markets
        # Calculate rolling volatility over a shorter window than forward_window
        volatility_window = min(30, forward_window // 4)  # Use smaller window for volatility
        df['rolling_volatility'] = df['close'].pct_change().rolling(volatility_window).std()
    
        # Define low volatility threshold as a fraction of min_profit
        low_vol_threshold = min_profit * 0.4  # 40% of min_profit
        
        # For each point in time, look forward to find profitable trades
        for i in range(n_samples - forward_window):
            entry_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+forward_window+1].values
            
            # Get current position (target from previous step)
            current_position = 0
            if i > 0:
                current_position = targets[i-1]
            
            # Calculate potential returns based on current position
            if current_position == 1:  # Already long
                long_fee_cost = fee_rate  # Only need to pay closing fee
                short_fee_cost = fee_rate * 3  # Need to close long (1) + open short (1) + close short later (1)
            elif current_position == 0:  # No position
                long_fee_cost = fee_rate * 2  # Need to open (1) and close (1)
                short_fee_cost = fee_rate * 2  # Need to open (1) and close (1)
            else:  # current_position == -1, Already short
                long_fee_cost = fee_rate * 3  # Need to close short (1) + open long (1) + close long later (1)
                short_fee_cost = fee_rate  # Only need to pay closing fee
            
            # For long positions
            long_returns = future_prices / entry_price - 1 - long_fee_cost
            max_long_return = np.max(long_returns) if len(long_returns) > 0 else -np.inf
            
            # For short positions
            short_returns = 1 - (future_prices / entry_price) - short_fee_cost
            max_short_return = np.max(short_returns) if len(short_returns) > 0 else -np.inf
            
            # Get current volatility
            current_volatility = df['rolling_volatility'].iloc[i]
        
            # Check for low volatility sideways market
            is_low_volatility = not np.isnan(current_volatility) and current_volatility < low_vol_threshold
            
            # Check if both long and short returns are weak
            weak_returns = max(max_long_return, max_short_return) < min_profit
            
            # Check if returns are too similar (ambiguous direction)
            similar_returns = abs(max_long_return - max_short_return) < no_trade_buffer * 0.5
            
            # Add extreme volatility check
            volatility_z_score = (current_volatility - df['rolling_volatility'].mean()) / df['rolling_volatility'].std() if df['rolling_volatility'].std() > 0 else 0
            extreme_volatility = abs(volatility_z_score) > 2.0  # Volatility is 2 std devs from mean
            
            # Determine the optimal trade direction
            # if indicator_divergence:
            #     # Force no trade in low volatility or when signals are weak or ambiguous
            #     targets[i] = 0  # No trade opportunity
            if (max_long_return >= min_profit and 
                max_long_return > max_short_return + no_trade_buffer and
                max_long_return > min_directional_strength):
                targets[i] = 1  # Long opportunity
            elif (max_short_return >= min_profit and 
                    max_short_return > max_long_return + no_trade_buffer and
                    max_short_return > min_directional_strength):
                targets[i] = -1  # Short opportunity
            else:
                targets[i] = 0  # No trade opportunity
                
        # Clean up the added column to prevent contaminating the DataFrame
        if 'rolling_volatility' in df.columns:
            df.drop('rolling_volatility', axis=1, inplace=True)
            
        # Print distribution to verify better balance
        unique, counts = np.unique(targets, return_counts=True)
        print(dict(zip(unique, counts)))
        
        return targets
    
    def build_model(self, X_train, y_train):
        """
        Build Bayesian ordered logistic regression model
        
        This method creates a Bayesian model with ordered logistic regression,
        which is appropriate for categorical outcomes with a natural ordering
        (short < neutral < long).
        
        Args:
            X_train (ndarray): Feature matrix
            y_train (ndarray): Target array with values -1, 0, 1
            
        Returns:
            tuple: (model, trace) - PyMC model and sampling trace
        """
        import time
    
        # Adjust y_train to be 0, 1, 2 instead of -1, 0, 1 for ordered logistic
        y_train_adj = y_train + 1
    
        # Log start time
        start_time = time.time()
        
        # Create PyMC model
        with pm.Model() as model:
            # Priors for unknown model parameters
            alpha = pm.Normal("alpha", mu=0, sigma=10, shape=2)
            betas = pm.Normal("betas", mu=0, sigma=2, shape=X_train.shape[1])
            
            # Linear predictor
            eta = pm.math.dot(X_train, betas)
            
            # Ordered logistic regression likelihood
            p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_train_adj)
            
            if self.using_jax_acceleration:
                # JAX-optimized sampling parameters
                cores = self.params.get('model', 'jax_cores', default=1) # JAX works better with single-core sampling
                draws = self.params.get('model', 'jax_draws', 
                                        default=min(800, self.params.get('model', 'mcmc_draws', default=1000)))
                tune = self.params.get('model', 'jax_tune', default=800)
                target_accept = self.params.get('model', 'jax_target_accept', default=0.9)
                self.logger.info(f"Starting MCMC sampling with JAX acceleration (cores={cores}, draws={draws})")
            else:
                # Standard sampling parameters 
                cores = self.params.get('model', 'mcmc_cores', default=4)
                draws = self.params.get('model', 'mcmc_draws', default=1000)
                tune = self.params.get('model', 'mcmc_tune', default=1000)
                target_accept = self.params.get('model', 'mcmc_target_accept', default=0.9)
            
            self.logger.info("Starting MCMC sampling with standard PyMC backend")           
            # Sample from the posterior distribution
            model_build_time = time.time() - start_time
            self.logger.info(f"Model building took {model_build_time:.2f} seconds")
        
            sampling_start = time.time()
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=self.params.get('model', 'mcmc_chains', default=4),
                cores=cores,
                target_accept=target_accept,
                return_inferencedata=True,
                compute_convergence_checks=False
            )
        
            sampling_time = time.time() - sampling_start
            total_time = time.time() - start_time
            
            # Log timing information
            self.logger.info(f"MCMC sampling completed in {sampling_time:.2f} seconds")
            self.logger.info(f"Total model build and sampling time: {total_time:.2f} seconds")
            
            # Store timing information
            self.metrics = {
                'model_build_time': model_build_time,
                'sampling_time': sampling_time,
                'total_time': total_time,
                'using_jax': self.using_jax_acceleration,
                'draws': draws,
                'cores': cores
            }
        
        self.model = model
        self.trace = trace
        return model, trace
    
    def train(self):
        """
        Train the model on processed data with proper train-test split
        
        This implementation:
        1. Loads the data using DataContext
        2. Creates chronological train-test split (last 30% for testing by default)
        3. Creates target variables
        4. Saves the test set separately
        5. Trains the model on training data only
        6. Saves the model
            
        Returns:
            tuple: (train_df, test_df) if successful, False otherwise
        """
        
        # Get parameters from ParamManager
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        test_size = params.get('training', 'test_size', default=0.3)
        
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe} with train-test split")
        
        try:
            # Load data using DataContext
            data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
    
            if data_context is None:
                self.logger.error(f"Failed to load data for {symbol} {timeframe}")
                return False
        
            # Validate the data has required columns
            if not data_context.validate(required_columns=self.feature_cols + ['open', 'high', 'low', 'close', 'volume']):
                return False
            
            # Create target if not already present - must be done before splitting to avoid data leakage
            if 'target' not in data_context.df.columns:
                self.logger.info("Creating target variables")
                data_context.df['target'] = self.create_target(data_context.df)
                data_context.add_processing_step("create_target", {"forward_window": 60})
            
            # Drop rows with NaN targets (occurs at the end due to forward window)
            data_context.df = data_context.df.dropna(subset=['target'])
            data_context.add_processing_step("dropna", {"subset": ['target']})

            # Create train-test split chronologically
            splits = data_context.create_time_split(test_size=test_size)
            train_df = splits['train']
            test_df = splits['test']
            
            self.logger.info(f"Split data into training ({len(train_df)} samples) and test ({len(test_df)} samples)")
            
            # Save test set for later evaluation if test_size > 0
            if test_size > 0 and len(test_df) > 0:
                # Create a new DataContext just for the test set
                test_context = DataContext(self.params, test_df, exchange, symbol, timeframe, source="test_set")
                
                # Create test set directory
                symbol_safe = symbol.replace('/', '_')
                test_sets_path = params.get('data', 'test_set', 'path', default="data/test_sets")
                test_output_path = Path(f"{test_sets_path}/{exchange}/{symbol_safe}")
                test_output_path.mkdir(parents=True, exist_ok=True)
            
                test_output_file = test_output_path / f"{timeframe}_test.csv"
                test_df.to_csv(test_output_file)
                test_context.add_processing_step("save_test_set", {"path": str(test_output_file)})
                self.logger.info(f"Saved test set to {test_output_file}")

            # Get max samples per batch from params, default to 100000
            max_samples_per_batch = self.params.get('training', 'max_samples_per_batch', default=100000)
            
            # Check if we need to batch process due to dataset size
            if len(train_df) > max_samples_per_batch:
                self.logger.warning(f"Dataset has {len(train_df)} samples, which exceeds max_samples_per_batch ({max_samples_per_batch})")
                self.logger.info(f"Using stratified sampling to reduce dataset size")
                
                # Perform stratified sampling to maintain class distribution
                train_df_sampled = pd.DataFrame()
                for target_value in train_df['target'].unique():
                    target_df = train_df[train_df['target'] == target_value]
                    # Calculate how many samples to take from this class
                    n_samples = min(len(target_df), int(max_samples_per_batch * len(target_df) / len(train_df)))
                    sampled = target_df.sample(n_samples)
                    train_df_sampled = pd.concat([train_df_sampled, sampled])
        
                data_context.add_processing_step("stratified_sampling", {
                    "original_size": len(train_df),
                    "sampled_size": len(train_df_sampled),
                    "sampling_method": "stratified",
                    "target_distribution": {str(v): int((train_df['target'] == v).sum()) for v in train_df['target'].unique()}
                })
                
                self.logger.info(f"Reduced dataset to {len(train_df_sampled)} samples (maintaining class balance)")
                train_df = train_df_sampled
            
            # Prepare training data
            X_train = train_df[self.feature_cols].values
            y_train = train_df['target'].values
            
            # Scale features (fit only on training data to avoid data leakage)
            X_train_scaled = self.scaler.fit_transform(X_train)
    
            data_context.add_processing_step("scale_features", {
                "scaler": "StandardScaler",
                "feature_cols": self.feature_cols,
                "fitted_on": "train"
            })
            
            # Build and train model on training data only
            self.logger.info("Building Bayesian model...")
            self.build_model(X_train_scaled, y_train)
    
            # Store DataContext metadata for model reproducibility
            self.data_context_metadata = {
                "processing_history": data_context.get_processing_history(),
                "feature_columns": self.feature_cols,
                "training_samples": len(train_df),
                "test_samples": len(test_df) if test_df is not None else 0,
                "target_distribution": {str(v): int((train_df['target'] == v).sum()) for v in train_df['target'].unique()}
            }
            
            # Save model
            self.save_model()
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
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
        
        # Ensure probabilities are non-negative
        probs = np.maximum(0, probs)

        # Normalize rows to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / row_sums
        
        return probs

    def train_with_cv(self):
        """
        Train with time-series cross-validation and proper test set separation
        
        This advanced implementation:
        1. Loads data and creates target variables
        2. Reserves a final hold-out test set (completely untouched)
        3. Performs time-series cross-validation on the remaining data
        4. Trains the final model on all training+validation data
        5. Saves the model and test set for later evaluation
            
        Returns:
            tuple: (train_val_df, test_df, cv_results) if successful, False otherwise
        """
            
        # Get parameters from ParamManager
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        test_size = params.get('training', 'test_size', default=0.2)
        n_splits = params.get('training', 'cv_splits', default=5)
        gap = params.get('training', 'cv_gap', default=0)
        
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe} with cross-validation")
        
        try:
            # Load data using DataContext
            data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
            
            if data_context is None:
                self.logger.error(f"Failed to load data for {symbol} {timeframe}")
                return False
                
            # Validate the data has required columns
            if not data_context.validate(required_columns=self.feature_cols + ['open', 'high', 'low', 'close', 'volume']):
                return False
            
            # Create target if not already present
            if 'target' not in data_context.df.columns:
                self.logger.info("Creating target variables")
                data_context.df['target'] = self.create_target(data_context.df)
                data_context.add_processing_step("create_target", {"forward_window": 60})
            
            # Drop rows with NaN targets (occurs at the end due to forward window)
            data_context.df = data_context.df.dropna(subset=['target'])
            data_context.add_processing_step("dropna", {"subset": ['target']})
            
            # First split into train+validation and test sets (chronologically)
            splits = data_context.create_time_split(test_size=test_size)
            train_val_df = splits['train']
            test_df = splits['test']
            
            self.logger.info(f"Reserved {len(test_df)} samples for test set")
            
            # Save test data for later evaluation
            if len(test_df) > 0:
                # Create a new DataContext just for the test set
                test_context = DataContext(self.params, test_df, exchange, symbol, timeframe, source="test_set")
                
                # Save the test set
                symbol_safe = symbol.replace('/', '_')
                test_sets_path = params.get('data', 'test_set', 'path', default="data/test_sets")
                test_output_path = Path(f"{test_sets_path}/{exchange}/{symbol_safe}")
                test_output_path.mkdir(parents=True, exist_ok=True)
                
                test_output_file = test_output_path / f"{timeframe}_test.csv"
                test_df.to_csv(test_output_file)
                test_context.add_processing_step("save_test_set", {"path": str(test_output_file)})
                self.logger.info(f"Saved test set to {test_output_file}")
            
            # Set up time series cross-validation on train_val set
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
            
            # Perform cross-validation
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
            
            # Final model training on all train_val data
            self.logger.info(f"Training final model on all {len(train_val_df)} training+validation samples")
            
            # Prepare all training data
            X_all = train_val_df[self.feature_cols].values
            y_all = train_val_df['target'].values
            
            # Scale features using all training data
            X_all_scaled = self.scaler.fit_transform(X_all)
            
            # Build and train final model
            self.build_model(X_all_scaled, y_all)
            
            # Store DataContext metadata for model reproducibility
            self.data_context_metadata = {
                "processing_history": data_context.get_processing_history(),
                "feature_columns": self.feature_cols,
                "training_samples": len(train_val_df),
                "test_samples": len(test_df),
                "cv_folds": n_splits,
                "cv_gap": gap,
                "target_distribution": {str(v): int((train_val_df['target'] == v).sum()) for v in train_val_df['target'].unique()}
            }
            
            # Save model
            self.save_model()
            
            # Plot CV performance
            self._plot_cv_results(cv_results)
            
            return train_val_df, test_df, cv_results
            
        except Exception as e:
            self.logger.error(f"Error training model with CV: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
    def train_with_reversed_datasets(self):
        """
        Train model with reversed datasets to test consistency and robustness
        
        This method:
        1. Loads the original train-test split usinf DataContext
        3. Reverses the datasets (trains on test, tests on train)
        4. Saves the model with a 'reversed' indicator
        5. Evaluates and compares performance on both approaches
            
        Returns:
            tuple: (train_df, test_df, comparison_metrics) if successful, False otherwise
        """
        # Get parameters from ParamManager
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        test_size = params.get('training', 'test_size', default=0.3)
        
        self.logger.info(f"Training model with reversed datasets for {exchange} {symbol} {timeframe}")
        
        try:
            # Load data using DataContext
            data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
            
            if data_context is None:
                self.logger.error(f"Failed to load data for {symbol} {timeframe}")
                return False
                
            # Validate the data has required columns
            if not data_context.validate(required_columns=self.feature_cols + ['open', 'high', 'low', 'close', 'volume']):
                return False
            
            # Create target if not already present
            if 'target' not in data_context.df.columns:
                self.logger.info("Creating target variables")
                data_context.df['target'] = self.create_target(data_context.df)
                data_context.add_processing_step("create_target", {"forward_window": 60})
            
            # Drop rows with NaN targets
            data_context.df = data_context.df.dropna(subset=['target'])
            data_context.add_processing_step("dropna", {"subset": ['target']})
            
            # First, check if we already have a test set
            symbol_safe = symbol.replace('/', '_')
            test_sets_path = params.get('data', 'test_set', 'path', default="data/test_sets")
            test_file_path = Path(f"{test_sets_path}/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
            if test_file_path.exists():
                # Load existing test set
                self.logger.info(f"Loading existing test set from {test_file_path}")
                test_df = pd.read_csv(test_file_path, index_col='timestamp', parse_dates=True)
                
                # Create test DataContext
                test_context = DataContext(self.params, test_df, exchange, symbol, timeframe, source="test_set")
                
                # Identify training data (all data not in test set)
                train_df = data_context.df[~data_context.df.index.isin(test_df.index)].copy()
            
                self.logger.info(f"Identified original training set with {len(train_df)} samples")
            else:
                # No existing test set, create a new split and run normal training first
                self.logger.info("No existing test set found. Creating new train-test split")
                
                # Create train-test split
                splits = data_context.create_time_split(test_size=test_size)
                train_df = splits['train']
                test_df = splits['test']
                
                self.logger.info(f"Created new train-test split: train={len(train_df)}, test={len(test_df)} samples")
                
                # Save test set for future use
                test_context = DataContext(self.params, test_df, exchange, symbol, timeframe, source="test_set")
                test_output_path = Path(f"{test_sets_path}/{exchange}/{symbol_safe}")
                test_output_path.mkdir(parents=True, exist_ok=True)
                test_output_file = test_output_path / f"{timeframe}_test.csv"
                test_df.to_csv(test_output_file)
                test_context.add_processing_step("save_test_set", {"path": str(test_output_file)})
                self.logger.info(f"Saved test set to {test_output_file}")
                
                # Run normal training first
                self.logger.info("Running normal training first")
                X_train = train_df[self.feature_cols].values
                y_train = train_df['target'].values
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                
                # Build and train model
                self.build_model(X_train_scaled, y_train)
            
                # Store DataContext metadata for model reproducibility
                self.data_context_metadata = {
                    "processing_history": data_context.get_processing_history(),
                    "feature_columns": self.feature_cols,
                    "training_samples": len(train_df),
                    "test_samples": len(test_df),
                    "target_distribution": {str(v): int((train_df['target'] == v).sum()) for v in train_df['target'].unique()}
                }
                
                # Save original model
                self.save_model()
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
            params.set("reversed", 'model', 'suffix')
            self.save_model()
            
            # Evaluate original and reversed models
            self.logger.info("Evaluating both models for comparison")
            
            # Load the original model for comparison
            original_model = self.__class__(self.params)
            original_model.load_model()
            
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
            self._plot_model_consistency(comparison_metrics)
            
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
            
            # Store DataContext metadata for reversed model
            self.data_context_metadata = {
                "processing_history": test_context.get_processing_history(),
                "reversed_training": True,
                "feature_columns": self.feature_cols,
                "training_samples": len(test_df),  # Now training on test data
                "test_samples": len(train_df),     # Now testing on train data
                "target_distribution": {str(v): int((test_df['target'] == v).sum()) for v in test_df['target'].unique()}
            }
            
            # Save reversed model with indicator
            params.set("reversed", 'model', 'suffix')
            self.save_model()
            
            return train_df, test_df, comparison_metrics
            
        except Exception as e:
            self.logger.error(f"Error training with reversed datasets: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def save_model(self, save_path=None, suffix=None):
        """
        Save the trained model to disk
        
        Saves all components needed for prediction using a structured path format:
        models/{symbol_safe}/{timeframe}/{model_type}/{datetime}_{comment}_{suffix}.{ext}
        
        Args:
            save_path (str, optional): Optional path to save the model to.
            suffix (str, optional): Additional identifier for special versions
            
        Returns:
            tuple: (bool, filename_base) — Success flag and base filename
        """
        try:
            # Get parameters from ParamManager
            params = self.params
            symbol = params.get('data', 'symbols', 0)
            timeframe = params.get('data', 'timeframes', 0)
            comment = params.get('model', 'comment', default=None)
            version = params.get('model', 'version', default='latest')
    
            # Check for suffix in params (used only if argument is None)
            param_suffix = params.get('model', 'suffix', default=None)
        
            # Use explicit argument if provided, otherwise use param_suffix
            suffix = suffix if suffix is not None else param_suffix
            
            # Create safe path elements
            models_path = params.get('model', 'path', default="models")
            symbol_safe = symbol.replace('/', '_')
            model_type = self.__class__.__name__.lower()
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create base directory structure
            model_dir = Path(f"{models_path}/{symbol_safe}/{timeframe}/{model_type}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create base filename with optional elements
            filename_parts = [timestamp]
            if comment:
                # Sanitize comment for safe filename use
                sanitized_comment = comment.replace(' ', '_')
                sanitized_comment = ''.join(c for c in sanitized_comment if c.isalnum() or c in '_-')
                if sanitized_comment:  # Only add if we have something after sanitizing
                    filename_parts.append(sanitized_comment)
            if suffix:
                # Sanitize suffix as well just to be safe
                sanitized_suffix = suffix.lstrip('_')
                sanitized_suffix = sanitized_suffix.replace(' ', '_')
                sanitized_suffix = ''.join(c for c in sanitized_suffix if c.isalnum() or c in '_-')
                if sanitized_suffix:
                    filename_parts.append(sanitized_suffix)
            
            filename_base = "_".join(filename_parts)
            
            # Paths for components
            scaler_path = model_dir / f"{filename_base}_scaler.pkl"
            trace_path = model_dir / f"{filename_base}_trace.netcdf"
            feat_cols_path = model_dir / f"{filename_base}_feature_cols.pkl"
            metrics_path = model_dir / f"{filename_base}_metrics.json"
            data_context_path = model_dir / f"{filename_base}_data_context.json"
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save trace (posterior samples) if available
            if hasattr(self, 'trace') and self.trace is not None:
                az.to_netcdf(self.trace, trace_path)
            
            # Save feature columns list
            with open(feat_cols_path, 'wb') as f:
                pickle.dump(self.feature_cols, f)
            
            # Save additional metrics if available
            metrics = {}
            if hasattr(self, 'metrics'):
                metrics = self.metrics
            
            # Add metadata to metrics
            metrics['saved_at'] = datetime.now().isoformat()
            metrics['model_type'] = model_type
            metrics['symbol'] = symbol
            metrics['timeframe'] = timeframe
            metrics['comment'] = comment
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save data context metadata if available
            if hasattr(self, 'data_context_metadata') and self.data_context_metadata:
                with open(data_context_path, 'w') as f:
                    json.dump(self.data_context_metadata, f, indent=2)
            
            # Create a symlink with a version or to the latest model
            version_scaler = model_dir / f"{version}_scaler.pkl"
            version_trace = model_dir / f"{version}_trace.netcdf"
            version_feat_cols = model_dir / f"{version}_feature_cols.pkl"
            version_metrics = model_dir / f"{version}_metrics.json"
            version_data_context = model_dir / f"{version}_data_context.json"  # Add this line
    
            # Remove existing symlinks if they exist
            for path in [version_scaler, version_trace, version_feat_cols, version_metrics, version_data_context]:
                if path.exists() or path.is_symlink():
                    path.unlink()
            
            # Create new symlinks
            version_scaler.symlink_to(scaler_path.name)
            version_trace.symlink_to(trace_path.name)
            version_feat_cols.symlink_to(feat_cols_path.name)
            version_metrics.symlink_to(metrics_path.name)
            version_data_context.symlink_to(data_context_path.name)  # Add this line

            # Verify symlinks
            if not version_scaler.exists() or version_scaler.stat().st_size == 0:
                self.logger.warning(f"Symlink verification failed for {version_scaler}. Attempting absolute path.")
                if version_scaler.exists() or version_scaler.is_symlink():
                    version_scaler.unlink()
                version_scaler.symlink_to(scaler_path.absolute())
                
            if not version_trace.exists() or version_trace.stat().st_size == 0:
                self.logger.warning(f"Symlink verification failed for {version_trace}. Attempting absolute path.")
                if version_trace.exists() or version_trace.is_symlink():
                    version_trace.unlink()
                version_trace.symlink_to(trace_path.absolute())
                
            if not version_feat_cols.exists() or version_feat_cols.stat().st_size == 0:
                self.logger.warning(f"Symlink verification failed for {version_feat_cols}. Attempting absolute path.")
                if version_feat_cols.exists() or version_feat_cols.is_symlink():
                    version_feat_cols.unlink()
                version_feat_cols.symlink_to(feat_cols_path.absolute())
                
            if not version_metrics.exists() or version_metrics.stat().st_size == 0:
                self.logger.warning(f"Symlink verification failed for {version_metrics}. Attempting absolute path.")
                if version_metrics.exists() or version_metrics.is_symlink():
                    version_metrics.unlink()
                version_metrics.symlink_to(metrics_path.absolute())
                
            # Final verification
            symlink_success = all([
                version_scaler.exists(), 
                version_trace.exists(), 
                version_feat_cols.exists(), 
                version_metrics.exists()
            ])
            
            if not symlink_success:
                self.logger.warning("Some symlinks could not be created properly.")
            
            self.logger.info(f"Model saved to {model_dir}/{filename_base}")
            return True, filename_base
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, None

    def load_model(self):
        """
        Load a trained model with flexible path resolution
        
        This method can load models from either:
        1. The new structured directory format
        2. The legacy flat format for backward compatibility
        3. Latest version using 'latest' symlinks
        4. Specific version by providing timestamp/comment/suffix
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get parameters from ParamManager
            params = self.params
            symbol = params.get('data', 'symbols', 0)
            timeframe = params.get('data', 'timeframes', 0)
            timestamp = params.get('model', 'timestamp', default=None)
            comment = params.get('model', 'comment', default=None)
            suffix = params.get('model', 'suffix', default=None)
            version = params.get('model', 'version', default='latest')
            
            # Create safe symbol name for paths
            symbol_safe = symbol.replace('/', '_')
            model_type = self.__class__.__name__.lower()
            
            # Try the new directory structure first
            model_dir = Path(f"models/{symbol_safe}/{timeframe}/{model_type}")
        
            if not model_dir.exists():
                self.logger.error(f"Model directory not found: {model_dir}")
                return False
            
            found_model = False
            model_files = {}
            
            # If version is 'latest', try to load the latest symlinks
            if version:
                scaler_path = model_dir / f"{version}_scaler.pkl"
                trace_path = model_dir / f"{version}_trace.netcdf" 
                feat_cols_path = model_dir / f"{version}_feature_cols.pkl"
                
                if scaler_path.exists() and trace_path.exists():
                    model_files = {
                        'scaler': scaler_path,
                        'trace': trace_path,
                        'feat_cols': feat_cols_path
                    }
                    found_model = True
                    self.logger.info(f"Loading {version} model from {model_dir}")
            
            # Otherwise look for a specific version using filters
            elif version == 'specific' and any([timestamp, comment, suffix]):
                # Build a pattern to match against
                pattern_parts = []
                if timestamp:
                    pattern_parts.append(timestamp)
                if comment:
                    pattern_parts.append(comment.replace(' ', '_'))
                if suffix:
                    pattern_parts.append(suffix)
                
                # Create pattern - we'll only search for scaler files and then derive the others
                file_pattern = "_".join(pattern_parts) if pattern_parts else ""
                
                # Find all model files that match the pattern
                all_scalers = list(model_dir.glob(f"*{file_pattern}*_scaler.pkl"))
                
                if all_scalers:
                    # Choose the most recent one by name (which includes timestamp)
                    scaler_path = sorted(all_scalers)[-1]
                    base_name = scaler_path.name.replace("_scaler.pkl", "")
                    
                    trace_path = model_dir / f"{base_name}_trace.netcdf"
                    feat_cols_path = model_dir / f"{base_name}_feature_cols.pkl"
                    
                    model_files = {
                        'scaler': scaler_path,
                        'trace': trace_path,
                        'feat_cols': feat_cols_path
                    }
                    found_model = True
                    self.logger.info(f"Found specific model: {base_name}")
            
            # Load the model if found
            if found_model:
                # Load scaler
                with open(model_files['scaler'], 'rb') as f:
                    self.scaler = pickle.load(f)
                
                # Load trace
                self.trace = az.from_netcdf(model_files['trace'])
                
                # Load feature columns if available
                if model_files['feat_cols'].exists():
                    with open(model_files['feat_cols'], 'rb') as f:
                        self.feature_cols = pickle.load(f)
                
                # Load data context metadata if available
                data_context_path = model_dir / f"{version}_data_context.json"
                if data_context_path.exists():
                    with open(data_context_path, 'r') as f:
                        self.data_context_metadata = json.load(f)
                        self.logger.info("Loaded data context metadata")
                else:
                    self.logger.warning("No data context metadata found")
                
                self.logger.info(f"Model loaded successfully")
                return True
            else:
                self.logger.error(f"No model found for {symbol} {timeframe} with the specified criteria")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

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
    
    def _plot_cv_results(self, cv_results):
        """
        Plot cross-validation results
        
        Creates a visualization showing training and validation accuracy
        across different cross-validation folds, helping to diagnose
        overfitting and model stability.
        
        Args:
            cv_results (dict): Dictionary with cross-validation results
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get parameters from ParamManager
            params = self.params
            symbol = params.get('data', 'symbols', 0)
            timeframe = params.get('data', 'timeframes', 0)
            
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
    
    def _plot_model_consistency(self, metrics):
        """
        Create visualizations of model consistency metrics
        
        Creates plots comparing model performance and feature importance
        between original and reversed training configurations, helping to
        diagnose model robustness and potential overfitting.
        
        Args:
            metrics (dict): Dictionary with comparison metrics
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get parameters from ParamManager
            params = self.params
            symbol = params.get('data', 'symbols', 0)
            timeframe = params.get('data', 'timeframes', 0)
            
            # Create output directory
            symbol_safe = symbol.replace('/', '_')
            model_type = self.__class__.__name__.lower()
            output_dir = Path(f"data/backtest_results/position_sizing/{symbol_safe}/{timeframe}/{model_type}")
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
    
    def train_multi(self):
        """
        Train the model on multiple symbols and timeframes
        
        This method trains a universal model that can be used across multiple
        trading pairs and timeframes. This is useful for finding patterns that
        generalize across different assets and time horizons.
                
        Returns:
            bool: True if successful, False otherwise
        """
        
        # Get parameters from ParamManager
        params = self.params
        symbols = params.get('data', 'symbols')
        timeframes = params.get('data', 'timeframes')
        exchange = params.get('data', 'exchanges', 0)
        max_samples = params.get('training', 'max_samples_per_batch', default=50000)
        
        self.logger.info(f"Training model on multiple symbols and timeframes")
        
        # Lists to store all training data
        all_X = []
        all_y = []
        all_contexts = []
        
        # Calculate maximum samples per source to stay within memory limits
        max_rows_per_source = max_samples // (len(symbols) * len(timeframes))
        
        # Process each symbol and timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                self.logger.info(f"Processing {symbol} {timeframe}")
                
                try:
                    # Load processed data using DataContext
                    data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
                    
                    if data_context is None:
                        self.logger.warning(f"Failed to load data for {symbol} {timeframe}")
                        continue
                        
                    # Validate the data has required columns
                    if not data_context.validate(required_columns=self.feature_cols + ['open', 'high', 'low', 'close', 'volume']):
                        continue
                    
                    # Create target if not already present
                    if 'target' not in data_context.df.columns:
                        self.logger.info(f"Creating target variables for {symbol} {timeframe}")
                        data_context.df['target'] = self.create_target(data_context.df)
                        data_context.add_processing_step("create_target", {"forward_window": 60})
                    
                    # Drop rows with NaN targets
                    data_context.df = data_context.df.dropna(subset=['target'])
                    data_context.add_processing_step("dropna", {"subset": ['target']})
                    
                    # Sample data if it exceeds the per-source limit
                    if len(df) > max_rows_per_source:
                        self.logger.info(f"Dataset too large, sampling {max_rows_per_source} rows from {symbol} {timeframe}")
                        
                        # Use stratified sampling to maintain class distribution
                        sampled_df = pd.DataFrame()
                        for target_value in data_context.df['target'].unique():
                            target_df = data_context.df[data_context.df['target'] == target_value]
                            # Calculate how many samples to take from this class
                            n_samples = min(len(target_df), int(max_rows_per_source * len(target_df) / len(data_context.df)))
                            sampled = target_df.sample(n_samples)
                            sampled_df = pd.concat([sampled_df, sampled])
                        
                        data_context.df = sampled_df
                        data_context.add_processing_step("stratified_sampling", {
                            "original_size": len(data_context.df),
                            "sampled_size": len(sampled_df),
                            "max_rows_per_source": max_rows_per_source
                        })
                    
                    # Extract features and target
                    X = data_context.df[self.feature_cols].values
                    y = data_context.df['target'].values
                    
                    # Append to combined datasets
                    all_X.append(X)
                    all_y.append(y)
                    all_contexts.append(data_context)
                    self.logger.info(f"Added {len(X)} rows from {symbol} {timeframe}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")
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
        
        # Combine all datasets
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        self.logger.info(f"Combined dataset has {len(X_combined)} samples")
        
        # Apply final sampling if needed to ensure we stay within memory limits
        if len(X_combined) > max_samples:
            self.logger.info(f"Combined dataset too large, sampling {max_samples} rows")
            # Random sampling for the final dataset
            indices = np.random.choice(len(X_combined), size=max_samples, replace=False)
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            self.logger.info(f"Final dataset has {len(X_combined)} samples after sampling")
        
        # Scale features
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        
        # Build and train model
        self.logger.info(f"Building Bayesian model with {len(X_combined)} rows")
        
        try:
            self.build_model(X_combined_scaled, y_combined)
            
            # Store combined DataContext metadata
            self.data_context_metadata = {
                "multi_symbol_model": True,
                "symbols": symbols,
                "timeframes": timeframes,
                "exchange": exchange,
                "sample_counts": {f"{ctx.symbol}_{ctx.timeframe}": len(ctx.df) for ctx in all_contexts},
                "processing_histories": {f"{ctx.symbol}_{ctx.timeframe}": ctx.get_processing_history() for ctx in all_contexts},
                "feature_columns": self.feature_cols,
                "total_samples": len(X_combined),
                "target_distribution": {str(v): int((y_combined == v).sum()) for v in np.unique(y_combined)}
            }
            
            # Save model
            self.save_model_multi(self._generate_multi_model_name())
            
            self.logger.info(f"Multi-symbol model trained and saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _generate_multi_model_name(self):
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
        # Get parameters from ParamManager
        params = self.params
        symbols = params.get('data', 'symbols')
        timeframes = params.get('data', 'timeframes')
        
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
            # Get parameters from ParamManager
            params = self.params
            
            # Check if a suffix is specified in ParamManager
            suffix = params.get('model', 'suffix', default=None)
        
            # Append suffix to model_name if provided
            if suffix:
                # Sanitize suffix
                sanitized_suffix = suffix.lstrip('_')
                sanitized_suffix = sanitized_suffix.replace(' ', '_')
                sanitized_suffix = ''.join(c for c in sanitized_suffix if c.isalnum() or c in '_-')
                if sanitized_suffix:
                    model_name = f"{model_name}_{sanitized_suffix}"
            
            # Create directory
            model_dir = Path("models")
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
        
    def continue_training(self):
        """
        Continue training an existing model with new data
        
        This method can either:
        1. Load new data from the standard location (automatically)
        2. Use a custom dataframe provided via params.get('new_data_df')
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Get parameters from ParamManager
        params = self.params
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        exchange = params.get('data', 'exchanges', 0)
        
        self.logger.info(f"Continuing training for {symbol} {timeframe}")
        
        # Check if a model exists
        if self.trace is None:
            self.logger.error("No existing model found. Please train a model first.")
            return False
        
        try:
            # Check if custom data is provided via params
            new_data_df = params.get('new_data_df', default=None)
            
            if new_data_df is None:
                # Load new data using DataContext
                data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
                
                if data_context is None:
                    self.logger.error(f"Failed to load data for {symbol} {timeframe}")
                    return False
                    
                # Validate the data has required columns
                if not data_context.validate(required_columns=self.feature_cols + ['open', 'high', 'low', 'close', 'volume']):
                    self.logger.error(f"Data validation failed for {symbol} {timeframe}")
                    return False
                
                # Check for test set to avoid training on it
                symbol_safe = symbol.replace('/', '_')
                test_sets_path = params.get('data', 'test_set', 'path', default="data/test_sets")
                test_file_path = Path(f"{test_sets_path}/{exchange}/{symbol_safe}/{timeframe}_test.csv")
            
                if test_file_path.exists():
                    test_df = pd.read_csv(test_file_path, index_col='timestamp', parse_dates=True)
                    self.logger.info(f"Found test set with {len(test_df)} rows")
                    
                    # Remove test data from training data
                    data_context.df = data_context.df[~data_context.df.index.isin(test_df.index)]
                    data_context.add_processing_step("remove_test_data", {"removed_rows": len(test_df)})
                    self.logger.info(f"Removed test data, {len(data_context.df)} rows remaining for training")
                    
                # Apply data sampling if dataset is too large
                max_samples = params.get('training', 'max_samples_per_batch', default=100000)
                if len(data_context.df) > max_samples:
                    self.logger.info(f"Dataset too large, sampling {max_samples} rows")
                    
                    # Use stratified sampling
                    sampled_df = pd.DataFrame()
                    for target_value in data_context.df['target'].unique():
                        target_df = data_context.df[data_context.df['target'] == target_value]
                        n_samples = min(len(target_df), int(max_samples * len(target_df) / len(data_context.df)))
                        sampled = target_df.sample(n_samples)
                        sampled_df = pd.concat([sampled_df, sampled])
                    
                    data_context.df = sampled_df
                    data_context.add_processing_step("stratified_sampling", {
                        "original_size": len(data_context.df),
                        "sampled_size": len(sampled_df),
                        "sampling_method": "stratified"
                    })
                
                new_data_df = data_context.df
            else:
                # Create a DataContext for the custom data
                data_context = DataContext(self.params, new_data_df, exchange, symbol, timeframe, source="custom_data")
                self.logger.info(f"Using provided custom dataframe with {len(new_data_df)} samples")
                
            # Process data
            if 'target' not in data_context.df.columns:
                self.logger.info("Creating target variables for new data")
                data_context.df['target'] = self.create_target(data_context.df)
                data_context.add_processing_step("create_target", {"forward_window": 60})
            
            # Drop rows with NaN targets
            data_context.df = data_context.df.dropna(subset=['target'])
            data_context.add_processing_step("dropna", {"subset": ['target']})
            
            # Extract features and target
            X_new = data_context.df[self.feature_cols].values
            y_new = data_context.df['target'].values
            
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
            
            # Create new model with informed priors
            with pm.Model() as new_model:
                # Use previous posterior as new prior
                alpha = pm.Normal("alpha", mu=alpha_mean, sigma=alpha_std, shape=2)
                betas = pm.Normal("betas", mu=betas_mean, sigma=betas_std, shape=X_new_scaled.shape[1])
                
                # Linear predictor
                eta = pm.math.dot(X_new_scaled, betas)
                
                # Ordered logistic regression
                p = pm.OrderedLogistic("p", eta=eta, cutpoints=alpha, observed=y_new_adj)
                
                # Sample
                new_trace = pm.sample(1000, tune=1000, chains=4, cores=1, return_inferencedata=True)
            
            # Update the model
            self.model = new_model
            self.trace = new_trace
            
            # Update the data_context_metadata
            if hasattr(self, 'data_context_metadata') and self.data_context_metadata:
                # Update existing metadata
                self.data_context_metadata["continued_training"] = True
                self.data_context_metadata["additional_samples"] = len(data_context.df)
                self.data_context_metadata["continuation_history"] = data_context.get_processing_history()
            else:
                # Create new metadata
                self.data_context_metadata = {
                    "continued_training": True,
                    "feature_columns": self.feature_cols,
                    "training_samples": len(data_context.df),
                    "processing_history": data_context.get_processing_history(),
                    "target_distribution": {str(v): int((data_context.df['target'] == v).sum()) for v in data_context.df['target'].unique()}
                }
            
            # Save updated model
            self.save_model()
            
            self.logger.info(f"Model continued training successfully with {len(data_context.df)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error continuing training: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
    def train_with_reservoir(self):
        """
        Train model using reservoir sampling to maintain a representative dataset
        using DataContext for tracking processing steps
        """
        # Get parameters from ParamManager
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        max_samples = params.get('model', 'reservoir_sample_size', default=50000)
    
        # Get new data from params
        new_data_df = params.get('new_data_df', default=None)
        if new_data_df is None:
            self.logger.error("No new data provided via params.set('new_data_df', df)")
            return False
    
        # Create DataContext for new data
        new_data_context = DataContext(self.params, new_data_df, exchange, symbol, timeframe, source="new_data")
        
        
        # Check if we already have a reservoir
        symbol_safe = symbol.replace('/', '_')
        reservoir_path = params.get('data', 'reservoir', 'path', default="data/reservoir")
        reservoir_dir = Path(f"{reservoir_path}/{exchange}/{symbol_safe}/{timeframe}.csv")
        
        if reservoir_dir.exists():
            # Load existing reservoir
            reservoir_df = pd.read_csv(reservoir_dir, index_col='timestamp', parse_dates=True)
            reservoir_context = DataContext(self.params, reservoir_df, exchange, symbol, timeframe, source="reservoir")
            self.logger.info(f"Loaded existing reservoir with {len(reservoir_df)} samples")
        else:
            # Create new reservoir
            reservoir_df = pd.DataFrame()
            reservoir_context = DataContext(self.params, reservoir_df, exchange, symbol, timeframe, source="new_reservoir")
            reservoir_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Add new data using reservoir sampling
        if len(reservoir_df) < max_samples:
            # Reservoir not full yet, just append
            reservoir_df = pd.concat([reservoir_df, new_data_df])
            reservoir_context.add_processing_step("append_data", {"added_samples": len(new_data_df)})
            if len(reservoir_df) > max_samples:
                # If we exceeded max_samples, randomly sample
                reservoir_df = reservoir_df.sample(max_samples)
                reservoir_context.add_processing_step("random_sample", {"target_size": max_samples})
        else:
            # Reservoir full, randomly replace elements
            for i, row in new_data_df.iterrows():
                if np.random.random() < len(new_data_df) / (len(reservoir_df) + i):
                    # Replace a random element
                    replace_idx = np.random.randint(0, len(reservoir_df))
                    reservoir_df.iloc[replace_idx] = row
        
            reservoir_context.add_processing_step("reservoir_sample", {
                "new_samples_processed": len(new_data_df),
                "reservoir_size": max_samples
            })
        
        # Save updated reservoir
        reservoir_df.to_csv(reservoir_dir)
        reservoir_context.df = reservoir_df  # Update the DataFrame in the context
        self.logger.info(f"Updated reservoir with {len(reservoir_df)} samples")
        
        # Set the prepared DataFrame as input
        self.params.set('model', 'input_dataframe', reservoir_df)
        
        # Call standard train method with reservoir metadata
        result = self.train()
        
        # Update metadata specific to reservoir sampling
        if hasattr(self, 'data_context_metadata') and self.data_context_metadata:
            self.data_context_metadata["reservoir_training"] = True
            self.data_context_metadata["reservoir_size"] = len(reservoir_df)
        
        return result
    
    def train_with_time_weighting(self):
        """
        Train with time-weighted sampling (newer data gets higher probability)
        using DataContext to track processing steps
        """
        # Get parameters from ParamManager
        params = self.params
        exchange = params.get('data', 'exchanges', 0)
        symbol = params.get('data', 'symbols', 0)
        timeframe = params.get('data', 'timeframes', 0)
        recency_weight = params.get('training', 'recency_weight', default=2.0)
    
        # Get all data using DataContext
        data_context = DataContext.from_processed_data(self.params, exchange, symbol, timeframe)
        
        if data_context is None:
            self.logger.error(f"Failed to load data for {symbol} {timeframe}")
            return False
            
        # Validate the data has required columns
        if not data_context.validate(required_columns=self.feature_cols + ['open', 'high', 'low', 'close', 'volume']):
            return False
        
        # Create target if not already present
        if 'target' not in data_context.df.columns:
            self.logger.info("Creating target variables")
            data_context.df['target'] = self.create_target(data_context.df)
            data_context.add_processing_step("create_target", {"forward_window": 60})
        
        # Drop rows with NaN targets
        data_context.df = data_context.df.dropna(subset=['target'])
        data_context.add_processing_step("dropna", {"subset": ['target']})
        
        # Sort by time
        data_context.df = data_context.df.sort_index()
        
        # Create time-based weights (newer data gets higher weight)
        time_indices = np.arange(len(data_context.df))
        weights = np.exp(recency_weight * time_indices / len(data_context.df))
        weights = weights / weights.sum()  # Normalize
        
        # Sample with weights
        sample_size = min(len(data_context.df), params.get('model', 'max_samples_per_batch', default=50000))
        sampled_indices = np.random.choice(
            len(data_context.df), 
            size=sample_size, 
            replace=False, 
            p=weights
        )
        
        sampled_df = data_context.df.iloc[sampled_indices]
        
        # Update DataContext with sampled data
        data_context.df = sampled_df
        data_context.add_processing_step("time_weighted_sampling", {
            "original_size": len(data_context.df),
            "sampled_size": sample_size,
            "recency_weight": recency_weight
        })
        
        # Train on weighted sample
        X_train = sampled_df[self.feature_cols].values
        y_train = sampled_df['target'].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build and train model
        self.logger.info(f"Training model on time-weighted sample with {len(sampled_df)} samples")
        self.build_model(X_train_scaled, y_train)
        
        # Store DataContext metadata
        self.data_context_metadata = {
            "time_weighted_training": True,
            "recency_weight": recency_weight,
            "sample_size": sample_size,
            "feature_columns": self.feature_cols,
            "processing_history": data_context.get_processing_history(),
            "target_distribution": {str(v): int((sampled_df['target'] == v).sum()) for v in sampled_df['target'].unique()}
        }
        
        # Save model
        self.save_model()
        
        return True
        
    def run_backtest_with_position_sizing(self, data_context, probabilities=None, no_trade_threshold=0.00096, 
                                            min_position_change=0.25, compound_returns=True, exaggerate=True):
        """
        Run backtest with continuous position sizing based on probability distributions
    
        
        Args:
            data_context (DataContext): Optional DataContext containing price data
            probabilities (array): Probabilities for long, short, and no_trade
            no_trade_threshold (float): Threshold for no_trade probability to ignore signals
            min_position_change (float): Minimum position change to avoid fee churn
            compound_returns (bool): Whether to compound returns for position sizing
            exaggerate (bool): Whether to exaggerate position sizes based on probability differences

        Returns:
            tuple: (results_df, metrics, fig) with backtest results, performance metrics, and visualization
        """
        exchange = self.params.get('data', 'exchanges', 0)
        symbol = self.params.get('data', 'symbols', 0)
        timeframe = self.params.get('data', 'timeframes', 0)
        no_trade_threshold = self.params.get('backtesting', 'enter_threshold', default=no_trade_threshold)
        exaggerate = self.params.get('backtesting', 'exaggerate', default=exaggerate)
        min_position_change = self.params.get('backtesting', 'min_position_change', default=min_position_change)
        
        try:
            # Record the start of position sizing backtest
            data_context.add_processing_step("position_sizing_backtest_start", {
                "no_trade_threshold": no_trade_threshold,
                "min_position_change": min_position_change,
                "compound_returns": compound_returns,
                "exaggerate": exaggerate,
                "data_source": data_context.source
            })
        
            # Get DataFrame from DataContext
            df = data_context.df
                    
            # Generate probabilities if not provided
            if probabilities is None:
                self.logger.info(f"Generating probability predictions for {len(df)} rows")
                probabilities = self.predict_probabilities(df)
                data_context.add_processing_step("generate_probabilities", {
                    "shape": probabilities.shape if probabilities is not None else None
                })
            
            if probabilities is None:
                self.logger.error("Failed to get probability predictions")
                return False, False, False
                
            # Create a copy of the dataframe for backtesting
            results = df.copy()
            
            # Add probability columns
            results['short_prob'] = probabilities[:, 0]
            results['no_trade_prob'] = probabilities[:, 1]
            results['long_prob'] = probabilities[:, 2]
        
            # Store original probabilities for reference
            results['original_long_prob'] = results['long_prob'].copy()
            results['original_short_prob'] = results['short_prob'].copy()
            
            # Calculate raw position sizes directly from probabilities
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
            
            # Apply exaggeration if enabled
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
        
            # Initialize compounding factor (starting with 1.0 = 100% of base size)
            if compound_returns:
                results['compound_factor'] = 1.0
            
            # Apply minimum change threshold to avoid fee churn
            # Handle each position type separately
            results['long_position'] = 0.0
            results['short_position'] = 0.0
            
            # Calculate price return
            results['price_return'] = results['close'].pct_change()
            
            # Loop through data and calculate positions with optional compounding
            compound_factor = 1.0  # Start with base sizing
            realized_pnl = 0.0     # Track realized P&L for compounding
            prev_long = 0.0
            prev_short = 0.0
            trade_count = 0
            
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
            
                # Track if a position change occurred in this iteration
                position_changed = False
                
                # Apply position changes only if they exceed minimum threshold
                if long_change >= min_position_change:
                    position_changed = True
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
                    position_changed = True
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
            
                # Update trade count if position changed
                if position_changed:
                    trade_count += 1
                
                # Update compound factor based on REALIZED pnl only
                if compound_returns and i > 0 and (long_change >= min_position_change or short_change >= min_position_change):
                    # Only update compound factor when positions change
                    compound_factor *= (1 + realized_pnl)
                    realized_pnl = 0.0  # Reset realized P&L after compounding
                
                # Store positions and compound factor
                results.iloc[i, results.columns.get_loc('long_position')] = new_long
                results.iloc[i, results.columns.get_loc('short_position')] = new_short
                if compound_returns:
                    results.iloc[i, results.columns.get_loc('compound_factor')] = compound_factor
            
                # Update previous positions for next iteration
                prev_long = new_long
                prev_short = new_short
            
            # Calculate net position (for compatibility)
            results['position'] = results['long_position'] - results['short_position']
            
            # Calculate separate returns for long and short positions
            results['long_return'] = results['long_position'].shift(1) * results['price_return']
            results['short_return'] = -1 * results['short_position'].shift(1) * results['price_return']  # Negative for shorts
            results['strategy_return'] = results['long_return'] + results['short_return']
            
            # Add fee impact - fees are proportional to position change
            fee_rate = self.params.get('exchange', 'fee_rate', default=0.0006)
            results['long_change'] = results['long_position'].diff().abs()
            results['short_change'] = results['short_position'].diff().abs()
            results['fee_impact'] = (results['long_change'] + results['short_change']) * fee_rate
            
            # Net returns after fees
            results['net_return'] = results['strategy_return'] - results['fee_impact']
        
            # Calculate cumulative returns
            results['price_cumulative'] = (1 + results['price_return'].fillna(0)).cumprod() - 1
            results['strategy_cumulative'] = (1 + results['net_return'].fillna(0)).cumprod() - 1
            
            # Calculate separate cumulative returns for long and short
            results['long_cumulative'] = (1 + results['long_return'].fillna(0)).cumprod() - 1
            results['short_cumulative'] = (1 + results['short_return'].fillna(0)).cumprod() - 1
            
            # Calculate performance metrics
            metrics = self._calculate_position_sizing_metrics(results, min_position_change)
            
            # Information about the backtest
            metrics['min_position_change'] = min_position_change
            metrics['no_trade_threshold'] = no_trade_threshold
            metrics['compound_returns'] = compound_returns
            metrics['exaggerate_positions'] = exaggerate
            metrics['data_source'] = data_context.source
            metrics['total_position_changes'] = trade_count
            
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
            
            # Add exaggeration information to metrics
            if exaggerate:
                metrics['avg_long_boost'] = (results['long_prob'] - results['original_long_prob']).mean()
                metrics['avg_short_boost'] = (results['short_prob'] - results['original_short_prob']).mean()
                metrics['max_long_boost'] = (results['long_prob'] - results['original_long_prob']).max()
                metrics['max_short_boost'] = (results['short_prob'] - results['original_short_prob']).max()
            
            # Update DataContext with results
            data_context.df = results
        
            # Add backtest results to processing history
            data_context.add_processing_step("position_sizing_backtest_complete", {
                "final_return": metrics.get('final_return', 0),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "max_drawdown": metrics.get('max_drawdown', 0),
                "win_rate": metrics.get('win_rate', 0),
                "total_trades": trade_count
            })
        
            # Add processing history to metrics
            metrics['processing_history'] = data_context.get_processing_history()
        
            # Create visualization using result logger
            from ..core.result_logger import ResultLogger
            result_logger = ResultLogger(self.params)

            # Strategy type suffix based on exaggeration
            strategy_type = 'position_sizing_exaggerated' if exaggerate else 'position_sizing'

            # Save results to standardized formats with context metadata included
            saved_files = result_logger.save_results(results, metrics, strategy_type=strategy_type)
        
            # Create visualizations
            viz_files = result_logger.plot_results(results, metrics, strategy_type=strategy_type)
            
            # Add visualization paths to metrics
            metrics['visualization_files'] = viz_files
            metrics['data_files'] = saved_files
            
            # Return results
            return results, metrics, None
            
        except Exception as e:
            self.logger.error(f"Error in position sizing backtest: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Add error information to DataContext
            if 'data_context' in locals():
                data_context.add_processing_step("position_sizing_error", {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
            
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
        
        # Get timeframe from params
        timeframe = self.params.get('data', 'timeframes', 0)
    
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
        multiplier = timeframe_multiplier.get(timeframe, 365)
        
        returns_mean = valid_results['net_return'].mean()
        returns_std = valid_results['net_return'].std()
        sharpe = (returns_mean / returns_std) * np.sqrt(multiplier) if returns_std > 0 else 0
        
        # Calculate max drawdown
        cumulative = valid_results['strategy_cumulative'].values
        max_dd, max_dd_duration = self._calculate_max_drawdown(cumulative)
    
        # Calculate position metrics
        position_changes = np.sum(
            (valid_results['long_position'].diff().abs() + valid_results['short_position'].diff().abs()) > min_position_change
        )
    
        # Calculate trade metrics - a trade happens when position changes
        long_changes = valid_results['long_position'].diff().abs() > min_position_change
        short_changes = valid_results['short_position'].diff().abs() > min_position_change
        
        # Track returns on position changes
        trade_returns = []
        for i in range(1, len(valid_results)):
            if long_changes.iloc[i] or short_changes.iloc[i]:
                # Calculate return since last change
                # This is a simplified approach - for precise returns we'd need entry/exit prices
                # But this gives us a way to estimate trade-level metrics
                trade_returns.append(valid_results['net_return'].iloc[i])
            
        # Calculate win rate and average win/loss
        if trade_returns:
            win_count = sum(ret > 0 for ret in trade_returns)
            loss_count = sum(ret < 0 for ret in trade_returns)
            win_rate = win_count / len(trade_returns) if len(trade_returns) > 0 else 0
            
            # Average win and loss
            win_returns = [ret for ret in trade_returns if ret > 0]
            loss_returns = [ret for ret in trade_returns if ret < 0]
            
            avg_win = sum(win_returns) / len(win_returns) if win_returns else 0
            avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0
            
            # Profit factor
            gross_profit = sum(win_returns)
            gross_loss = abs(sum(loss_returns))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
    
        # Calculate fee impact
        total_fees = valid_results['fee_impact'].sum() if 'fee_impact' in valid_results.columns else 0
        fee_drag = total_fees
    
        # Calculate other metrics
        alpha = final_strategy_return - final_price_return  # Excess return over buy & hold
        
        # Create metrics dictionary with ALL required keys
        metrics = {
            'final_return': final_strategy_return,
            'buy_hold_return': final_price_return,
            'long_only_return': final_long_return,
            'short_only_return': final_short_return,
            'alpha': alpha,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_dd_duration': max_dd_duration,
            'total_trades': len(trade_returns),
            'position_changes': position_changes,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'fee_drag': fee_drag,
            'min_position_change': min_position_change
        }
        
        return metrics

    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown and duration from cumulative returns array"""
        # Check for valid input
        if len(cumulative_returns) < 2:
            return 0, 0
            
        # Starting with 1 + returns, not percentage form
        cum_returns = 1 + np.array(cumulative_returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown in percentage terms
        drawdown = (cum_returns - running_max) / running_max
        
        # Find the maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Calculate drawdown duration
        drawdown_start = np.argmax(cum_returns[:np.argmin(drawdown)])
        drawdown_end = np.argmin(drawdown)
        drawdown_duration = drawdown_end - drawdown_start
        
        return abs(max_drawdown), drawdown_duration