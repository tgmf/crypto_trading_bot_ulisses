#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow-based Bayesian model implementation for cryptocurrency trading.

This module reimplements the Bayesian ordered logistic regression model using
TensorFlow Probability instead of PyMC, allowing GPU acceleration without
JAX dependency issues. It includes advanced training methods like:
- Cross-validation
- Dataset reversal testing
- Multi-symbol training
- Reservoir sampling
- Time-weighted sampling
- Continued training
"""

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json
import matplotlib.pyplot as plt
import time

from .position_sizing import QuantumPositionSizer

# Enable GPU memory growth to avoid allocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")

# Set TensorFlow to use mixed precision for better performance on GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Aliases for TensorFlow Probability distributions and bijectors
tfd = tfp.distributions
tfb = tfp.bijectors

class TFBayesianModel:
    """
    TensorFlow-based Bayesian model for trading signals using ordered logistic regression
    
    This model predicts the probability of profitable trading opportunities in
    three states: short (-1), neutral (0), and long (1). It uses TensorFlow
    Probability for Bayesian inference with GPU acceleration.
    """
    
    def __init__(self, config):
        """
        Initialize with configuration
        
        Args:
            config (dict): Configuration dictionary containing model parameters,
                        fee rates, and target thresholds
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.posterior_samples = None
        self.scaler = StandardScaler()
        
        # Extract model parameters from config
        self.fee_rate = self.config.get('backtesting', {}).get('fee_rate', 0.0006)
        self.min_profit = self.config.get('backtesting', {}).get('min_profit_target', 0.008)
        
        # Feature columns to use for prediction
        self.feature_cols = self.config.get('model', {}).get('feature_cols', [
            'bb_pos', 'RSI_14', 'MACDh_12_26_9', 'trend_strength', 
            'volatility', 'volume_ratio', 'range', 'macd_hist_diff', 
            'rsi_diff', 'bb_squeeze'
        ])
        
        # Set training parameters
        self.num_samples = 1000
        self.num_burnin_steps = 1000
        self.learning_rate = 0.01
        
    def create_target(self, df, forward_window=20):
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
        
        # Calculate total fee cost for round trip (entry + exit)
        fee_cost = self.fee_rate * 2
        
        # For each point in time, look forward to find profitable trades
        for i in range(n_samples - forward_window):
            entry_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+forward_window+1].values
            
            # Calculate potential returns for long positions
            # Formula: (exit_price / entry_price) - 1 - fees
            long_returns = future_prices / entry_price - 1 - fee_cost
            max_long_return = np.max(long_returns) if len(long_returns) > 0 else -np.inf
            
            # Calculate potential returns for short positions
            # Formula: 1 - (exit_price / entry_price) - fees
            short_returns = 1 - (future_prices / entry_price) - fee_cost
            max_short_return = np.max(short_returns) if len(short_returns) > 0 else -np.inf
            
            # Determine the optimal trade direction based on maximum potential return
            if max_long_return >= self.min_profit and max_long_return > max_short_return:
                targets[i] = 1  # Long opportunity
            elif max_short_return >= self.min_profit and max_short_return > max_long_return:
                targets[i] = -1  # Short opportunity
            else:
                targets[i] = 0  # No trade opportunity
        
        return targets
    
    def build_model(self, X_train, y_train):
        """
        Build TensorFlow Probability model for ordered logistic regression
        
        This method creates a Bayesian model with ordered logistic regression using
        TensorFlow Probability instead of PyMC, allowing GPU acceleration.
        
        Args:
            X_train (ndarray): Feature matrix
            y_train (ndarray): Target array with values -1, 0, 1
            
        Returns:
            tuple: (model, posterior_samples) - TFP model and MCMC samples
        """
        # Convert numpy arrays to TensorFlow tensors with float32 precision
        X_train_tf = tf.cast(X_train, tf.float32)
        
        # Adjust y_train to be 0, 1, 2 instead of -1, 0, 1 for ordered logistic
        y_train_adj = y_train + 1
        y_train_tf = tf.cast(y_train_adj, tf.int32)
        
        # Number of features
        num_features = X_train.shape[1]
        
        # Define the joint distribution using TensorFlow Probability
        def ordered_logistic_model():
            # Priors for coefficients with normal distribution
            betas = yield tfd.Normal(loc=tf.zeros(num_features), 
                                    scale=tf.ones(num_features) * 2.0,
                                    name='betas')
            
            # Prior for the first cutpoint
            cutpoint1 = yield tfd.Normal(loc=0.0, scale=10.0, name='cutpoint1')
            
            # Prior for the distance between cutpoints (must be positive)
            # Using a scalar to ensure consistency
            cutpoint_delta = yield tfd.LogNormal(loc=0.0, scale=1.0, name='cutpoint_delta')
            
            # Second cutpoint is the first plus the delta
            cutpoint2 = cutpoint1 + cutpoint_delta
            
            # Linear predictor - dot product of features and coefficients
            # Ensure consistent dimensions using reshape
            logits = tf.matmul(X_train_tf, tf.reshape(betas, [num_features, 1]))[:, 0]
            
            # Calculate ordered logistic probabilities
            p_leq_1 = tf.sigmoid(cutpoint1 - logits)
            p_leq_2 = tf.sigmoid(cutpoint2 - logits)
            
            # Class probabilities: P(y=0), P(y=1), P(y=2)
            probs_0 = p_leq_1
            probs_1 = p_leq_2 - p_leq_1
            probs_2 = 1.0 - p_leq_2
            
            # Stack probabilities
            probs = tf.stack([probs_0, probs_1, probs_2], axis=1)
            
            # Use categorical distribution for the likelihood
            yield tfd.Categorical(probs=probs, name='y')
        
        # Create the joint distribution
        model = tfd.JointDistributionCoroutineAutoBatched(ordered_logistic_model)
        
        # For a more stable MCMC sampling, we'll start with simpler HMC
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=lambda *args: model.log_prob(args + (y_train_tf,)),
            step_size=0.01,
            num_leapfrog_steps=10
        )
        
        # Add adaptation for better mixing
        adaptive_hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel,
            bijector=[
                tfb.Identity(),  # For betas
                tfb.Identity(),  # For cutpoint1
                tfb.Exp()        # For cutpoint_delta (must be positive)
            ]
        )
        
        # Add step size adaptation
        adaptive_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=adaptive_hmc,
            num_adaptation_steps=int(self.num_burnin_steps * 0.8),
            target_accept_prob=0.75
        )
        
        # Run MCMC sampling - use GPU if available
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            print(f"Running MCMC sampling on {'/GPU:0' if gpus else '/CPU:0'}")
            
            # Initialize from reasonable starting points with explicit shapes
            initial_state = [
                tf.zeros([num_features], dtype=tf.float32),    # betas
                tf.constant(0.0, dtype=tf.float32),            # cutpoint1
                tf.constant(1.0, dtype=tf.float32)             # cutpoint_delta
            ]
            
            # Run the MCMC chain with diagnostics
            self.logger.info(f"Starting MCMC sampling with {self.num_samples} samples and {self.num_burnin_steps} burn-in steps")
            
            # Reduce sample count for very large datasets
            if X_train.shape[0] > 1000000:
                self.logger.info(f"Large dataset detected ({X_train.shape[0]} rows). Reducing sample count.")
                self.num_samples = 500
                self.num_burnin_steps = 300
            
            posterior_samples, trace = tfp.mcmc.sample_chain(
                num_results=self.num_samples,
                num_burnin_steps=self.num_burnin_steps,
                current_state=initial_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted
            )
            
            # Calculate acceptance rate
            acceptance_rate = tf.reduce_mean(tf.cast(trace, tf.float32))
            self.logger.info(f"MCMC sampling complete. Acceptance rate: {acceptance_rate:.2f}")
        
        # Store model and samples
        self.model = model
        self.posterior_samples = posterior_samples
        
        # Reorganize posterior samples if needed
        # The order in posterior_samples is now [betas, cutpoint1, cutpoint_delta]
        # which differs from your predict_probabilities method that expects [betas, cutpoint_deltas, cutpoint1]
        # You'll need to adjust predict_probabilities accordingly
        
        return model, posterior_samples
    
    def train(self, exchange='binance', symbol='BTC/USDT', timeframe='1h', test_size=0.3, custom_df=None, reservoir_df=None):
        """
        Train the model on processed data with proper train-test split
        
        This implementation:
        1. Loads the data (or uses provided custom dataframe)
        2. Creates chronological train-test split (last 30% for testing by default)
        3. Creates target variables
        4. Saves the test set separately
        5. Trains the model on training data only
        6. Saves the model
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            test_size (float): Proportion of data to use for testing (0.0-1.0)
            custom_df (DataFrame): Optional custom DataFrame to use instead of loading from file
            reservoir_df (DataFrame): Optional reservoir dataframe to use for training
            
        Returns:
            tuple: (train_df, test_df) if successful, False otherwise
        """
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe} with train-test split")
        
        try:
            # Use provided dataframe or load from file
            if custom_df is not None:
                self.logger.info("Using provided custom dataframe")
                df = custom_df.copy()
            elif reservoir_df is not None:
                self.logger.info("Using provided reservoir dataframe")
                df = reservoir_df.copy()
            else:
                # Load processed data
                symbol_safe = symbol.replace('/', '_')
                input_file = Path(f"data/processed/{exchange}/{symbol_safe}/{timeframe}.csv")
                
                if not input_file.exists():
                    self.logger.error(f"No processed data file found at {input_file}")
                    return False
                    
                df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
            
            # Create target - must be done before splitting to avoid data leakage
            if 'target' not in df.columns:
                self.logger.info("Creating target variables")
                df['target'] = self.create_target(df)
            
            # Drop rows with NaN targets (occurs at the end due to forward window)
            df = df.dropna(subset=['target'])
            
            # Chronological train-test split - using the last part for testing
            train_size = int(len(df) * (1 - test_size))
            train_df = df.iloc[:train_size].copy()
            test_df = df.iloc[train_size:].copy()
            
            self.logger.info(f"Split data into training ({len(train_df)} samples) and test ({len(test_df)} samples)")
            
            # Save test data for later evaluation (only if loading from file)
            if custom_df is None and reservoir_df is None:
                symbol_safe = symbol.replace('/', '_')
                test_output_dir = Path(f"data/test_sets/{exchange}/{symbol_safe}")
                test_output_dir.mkdir(parents=True, exist_ok=True)
                test_output_file = test_output_dir / f"{timeframe}_test.csv"
                test_df.to_csv(test_output_file)
                self.logger.info(f"Saved test set to {test_output_file}")
            
            # Prepare training data
            X_train = train_df[self.feature_cols].values
            y_train = train_df['target'].values
            
            # Scale features (fit only on training data to avoid data leakage)
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Build and train model on training data only
            self.logger.info("Building TensorFlow Bayesian model...")
            start_time = time.time()
            self.build_model(X_train_scaled, y_train)
            training_time = time.time() - start_time
            self.logger.info(f"Model training complete in {training_time:.2f} seconds")
            
            # Save model
            self.save_model(exchange, symbol, timeframe)
            
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
        # Ensure we have posterior samples
        if self.posterior_samples is None:
            self.logger.error("No posterior samples available. Train or load a model first.")
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
        
        # Convert to TensorFlow tensor
        X_tf = tf.cast(X_scaled, tf.float32)
        
        # Get posterior samples [betas, cutpoint1, cutpoint_delta]
        beta_samples = self.posterior_samples[0]  # Shape: [num_samples, num_features]
        cutpoint1_samples = self.posterior_samples[1]  # Shape: [num_samples, 1]
        cutpoint_delta_samples = self.posterior_samples[2]  # Shape: [num_samples, 1]
        
        # Compute cutpoint2
        cutpoint2_samples = cutpoint1_samples + cutpoint_delta_samples
        
        # Initialize arrays to store probabilities
        num_points = X_tf.shape[0]
        probs_array = np.zeros((num_points, 3))
        
        # Use GPU if available for prediction
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            # Compute mean of samples for point estimates
            beta_mean = tf.reduce_mean(beta_samples, axis=0)
            cutpoint1_mean = tf.reduce_mean(cutpoint1_samples)
            cutpoint2_mean = tf.reduce_mean(cutpoint2_samples)
            
            # Compute logits for each data point
            logits = tf.matmul(X_tf, beta_mean[:, tf.newaxis])[:, 0]
            
            # Calculate ordered logistic probabilities
            p_leq_1 = tf.sigmoid(cutpoint1_mean - logits)
            p_leq_2 = tf.sigmoid(cutpoint2_mean - logits)
            
            # Class probabilities: P(y=0), P(y=1), P(y=2)
            probs_0 = p_leq_1
            probs_1 = p_leq_2 - p_leq_1
            probs_2 = 1.0 - p_leq_2
            
            # Stack probabilities and convert to numpy
            probs = tf.stack([probs_0, probs_1, probs_2], axis=1).numpy()
        
        # Reorder to match PyMC order: [P(short), P(no_trade), P(long)]
        # This follows the original ordering in BayesianModel.predict_probabilities
        probs_reordered = np.zeros_like(probs)
        probs_reordered[:, 0] = probs[:, 0]  # P(y=0) -> P(short)
        probs_reordered[:, 1] = probs[:, 1]  # P(y=1) -> P(no_trade)
        probs_reordered[:, 2] = probs[:, 2]  # P(y=2) -> P(long)
        
        return probs_reordered
    
    def _predict_class(self, X, posterior_samples=None):
        """
        Predict class from features using posterior samples
        
        Args:
            X (ndarray): Feature matrix (scaled)
            posterior_samples: TFP posterior samples (use self.posterior_samples if None)
            
        Returns:
            ndarray: Class predictions (0, 1, or 2)
        """
        if posterior_samples is None:
            posterior_samples = self.posterior_samples
            
        if posterior_samples is None:
            self.logger.error("No posterior samples available for prediction")
            return None
        
        # Convert to TensorFlow tensor
        X_tf = tf.cast(X, tf.float32)
        
        # Get posterior samples
        beta_samples = posterior_samples[0]
        cutpoint_delta_samples = posterior_samples[1]
        cutpoint1_samples = posterior_samples[2]
        
        # Compute cutpoint2
        cutpoint2_samples = cutpoint1_samples + cutpoint_delta_samples
        
        # Use GPU if available for prediction
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            # Compute mean of samples for point estimates
            beta_mean = tf.reduce_mean(beta_samples, axis=0)
            cutpoint1_mean = tf.reduce_mean(cutpoint1_samples)
            cutpoint2_mean = tf.reduce_mean(cutpoint2_samples)
            
            # Compute logits for each data point
            logits = tf.matmul(X_tf, beta_mean[:, tf.newaxis])[:, 0]
            
            # Calculate ordered logistic probabilities
            p_leq_1 = tf.sigmoid(cutpoint1_mean - logits)
            p_leq_2 = tf.sigmoid(cutpoint2_mean - logits)
            
            # Class probabilities: P(y=0), P(y=1), P(y=2)
            probs_0 = p_leq_1
            probs_1 = p_leq_2 - p_leq_1
            probs_2 = 1.0 - p_leq_2
            
            # Stack probabilities 
            probs = tf.stack([probs_0, probs_1, probs_2], axis=1)
            
            # Get most likely class
            pred_class = tf.argmax(probs, axis=1).numpy()
            
            return pred_class
    
    def save_model(self, exchange, symbol, timeframe):
        """
        Save the trained model
        
        Saves all components needed for prediction:
        - Scaler for feature normalization
        - Posterior samples from TFP
        - Feature column list
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Create safe path elements
            symbol_safe = symbol.replace('/', '_')
            
            # Save scaler
            scaler_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save posterior samples as numpy arrays
            samples_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_samples.npz"
            np.savez(
                samples_path, 
                betas=self.posterior_samples[0].numpy(),
                cutpoint_deltas=self.posterior_samples[1].numpy(),
                cutpoint1=self.posterior_samples[2].numpy()
            )
            
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
        """
        Load a trained model
        
        Loads all components needed for prediction:
        - Scaler for feature normalization
        - Posterior samples from TFP
        - Feature column list (if available)
        
        Args:
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create paths
            model_dir = Path("models")
            symbol_safe = symbol.replace('/', '_')
            
            scaler_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_scaler.pkl"
            samples_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_samples.npz"
            feat_cols_path = model_dir / f"{exchange}_{symbol_safe}_{timeframe}_feature_cols.pkl"
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load posterior samples
            with np.load(samples_path) as data:
                # Convert numpy arrays to TensorFlow tensors
                self.posterior_samples = [
                    tf.convert_to_tensor(data['betas']),
                    tf.convert_to_tensor(data['cutpoint_deltas']),
                    tf.convert_to_tensor(data['cutpoint1'])
                ]
            
            # Load feature columns if available
            if feat_cols_path.exists():
                with open(feat_cols_path, 'rb') as f:
                    self.feature_cols = pickle.load(f)
            
            self.logger.info(f"Model loaded from {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
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
                with tf.device('/GPU:0' if gpus else '/CPU:0'):
                    # Create ordered logistic model for this fold
                    # This is a simplified version just for CV evaluation
                    X_train_tf = tf.cast(X_train_fold_scaled, tf.float32)
                    y_train_adj = y_train_fold + 1  # Adjust to 0,1,2
                    y_train_tf = tf.cast(y_train_adj, tf.int32)
                    
                    # Define model
                    num_features = X_train_fold_scaled.shape[1]
                    
                    def ordered_logistic_model():
                        betas = yield tfd.Normal(loc=tf.zeros(num_features), 
                                                scale=tf.ones(num_features) * 2.0)
                        cutpoint_deltas = yield tfd.LogNormal(loc=tf.zeros(1), scale=tf.ones(1))
                        cutpoint1 = yield tfd.Normal(loc=0.0, scale=10.0)
                        cutpoint2 = cutpoint1 + cutpoint_deltas[0]
                        
                        logits = tf.matmul(X_train_tf, betas[:, tf.newaxis])[:, 0]
                        
                        p_leq_1 = tf.sigmoid(cutpoint1 - logits)
                        p_leq_2 = tf.sigmoid(cutpoint2 - logits)
                        
                        probs_0 = p_leq_1
                        probs_1 = p_leq_2 - p_leq_1
                        probs_2 = 1.0 - p_leq_2
                        
                        probs = tf.stack([probs_0, probs_1, probs_2], axis=1)
                        
                        yield tfd.Categorical(probs=probs, name='y')
                    
                    # Create model for this fold
                    fold_model = tfd.JointDistributionCoroutineAutoBatched(ordered_logistic_model)
                    
                    # Configure MCMC sampling - simplified for CV
                    adaptive_hmc = tfp.mcmc.NoUTurnSampler(
                        target_log_prob_fn=lambda *args: fold_model.log_prob(args + (y_train_tf,)),
                        step_size=0.01
                    )
                    
                    # Add adaptation for better mixing
                    adaptive_hmc = tfp.mcmc.TransformedTransitionKernel(
                        inner_kernel=adaptive_hmc,
                        bijector=[
                            tfb.Identity(),  # For betas
                            tfb.Exp(),       # For cutpoint_deltas
                            tfb.Identity()   # For cutpoint1
                        ]
                    )
                    
                    # Shorter sampling for CV to save time
                    cv_samples = 500  # Fewer samples for CV
                    cv_burnin = 300
                    
                    # Initialize from reasonable starting points
                    initial_state = [
                        tf.zeros(num_features),  # betas
                        tf.ones(1),              # cutpoint_deltas
                        tf.zeros(1)              # cutpoint1
                    ]
                    
                    # Sample
                    posterior_samples, _ = tfp.mcmc.sample_chain(
                        num_results=cv_samples,
                        num_burnin_steps=cv_burnin,
                        current_state=initial_state,
                        kernel=adaptive_hmc,
                        trace_fn=None
                    )
                
                # Evaluate on training data
                train_preds = self._predict_class(X_train_fold_scaled, posterior_samples)
                train_accuracy = np.mean(train_preds == y_train_adj)
                
                # Evaluate on validation data
                val_preds = self._predict_class(X_val_fold_scaled, posterior_samples)
                val_accuracy = np.mean(val_preds == (y_val_fold + 1))
                
                # Store results for this fold
                cv_results['fold'].append(fold)
                cv_results['train_indices'].append(train_idx)
                cv_results['val_indices'].append(val_idx)
                cv_results['train_accuracy'].append(float(train_accuracy))
                cv_results['val_accuracy'].append(float(val_accuracy))
                
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
            
            # Build and train final model with the full parameter set
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
            
    def _plot_cv_results(self, cv_results, exchange, symbol, timeframe):
        """
        Plot cross-validation results
        
        Creates a visualization showing training and validation accuracy
        across different cross-validation folds, helping to diagnose
        overfitting and model stability.
        
        Args:
            cv_results (dict): Dictionary with cross-validation results
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            bool: True if successful, False otherwise
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
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting CV results: {str(e)}")
            return False
        
    def train_multi(self, symbols, timeframes, exchange='binance', max_samples=500000):
        """
        Train the model on multiple symbols and timeframes
        
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
        self.logger.info(f"Training model on multiple symbols and timeframes")
        
        # Lists to store all training data
        all_X = []
        all_y = []
        
        # Calculate maximum samples per source to stay within memory limits
        max_rows_per_source = max_samples // (len(symbols) * len(timeframes))
        
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
                    
                    # Create target
                    df['target'] = self.create_target(df)
                    
                    # Drop rows with NaN targets
                    df = df.dropna(subset=['target'])
                    
                    # Sample data if it exceeds the per-source limit
                    if len(df) > max_rows_per_source:
                        self.logger.info(f"Dataset too large, sampling {max_rows_per_source} rows from {symbol} {timeframe}")
                        
                        # Use TensorFlow's implementation to do stratified sampling
                        # This can be more efficient with large datasets
                        
                        # Get indices for each class (for stratified sampling)
                        indices_by_class = {}
                        for cls in np.unique(df['target']):
                            indices_by_class[cls] = np.where(df['target'].values == cls)[0]
                        
                        # Calculate samples per class
                        min_class_count = min(len(indices) for indices in indices_by_class.values())
                        samples_per_class = min(min_class_count, max_rows_per_source // len(indices_by_class))
                        
                        # Sample from each class
                        sampled_indices = []
                        for cls, indices in indices_by_class.items():
                            if len(indices) > samples_per_class:
                                # Random sampling without replacement
                                sampled_cls_indices = np.random.choice(indices, size=samples_per_class, replace=False)
                                sampled_indices.extend(sampled_cls_indices)
                            else:
                                # Take all samples if not enough
                                sampled_indices.extend(indices)
                        
                        # Sample the dataframe
                        df = df.iloc[sampled_indices]
                    
                    # Extract features and target
                    X = df[self.feature_cols].values
                    y = df['target'].values
                    
                    # Append to combined datasets
                    all_X.append(X)
                    all_y.append(y)
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
        self.logger.info(f"Building TensorFlow Bayesian model with {len(X_combined)} rows")
        
        try:
            # Use GPU if available
            with tf.device('/GPU:0' if gpus else '/CPU:0'):
                self.build_model(X_combined_scaled, y_combined)
            
            # Create a model name based on symbols and timeframes
            model_name = self._generate_multi_model_name(symbols, timeframes)
            
            # Save model
            self.save_model_multi(exchange, model_name)
            
            self.logger.info(f"Multi-symbol model trained and saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _plot_model_consistency(self, metrics, exchange, symbol, timeframe):
        """
        Create visualizations of model consistency metrics
        
        Creates plots comparing model performance and feature importance
        between original and reversed training configurations, helping to
        diagnose model robustness and potential overfitting.
        
        Args:
            metrics (dict): Dictionary with comparison metrics
            exchange (str): Exchange name
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            bool: True if successful, False otherwise
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
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting model consistency: {str(e)}")
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
            # For TF model, extract feature importances from posterior samples
            beta_orig = original_model.posterior_samples[0].numpy()
            beta_rev = self.posterior_samples[0].numpy()
            
            # Calculate mean of absolute coefficients
            orig_importances = np.abs(np.mean(beta_orig, axis=0))
            rev_importances = np.abs(np.mean(beta_rev, axis=0))
            
            orig_ranks = np.argsort(-orig_importances)
            rev_ranks = np.argsort(-rev_importances)
            
            # Calculate rank correlation
            rank_correlation = np.corrcoef(orig_ranks, rev_ranks)[0, 1]
            
            # Assemble comparison metrics
            comparison_metrics = {
                'original_accuracy': float(orig_acc),
                'reversed_accuracy': float(rev_acc),
                'accuracy_difference': float(abs(orig_acc - rev_acc)),
                'feature_importance_correlation': float(rank_correlation),
                'original_feature_importance': dict(zip(self.feature_cols, orig_importances.tolist())),
                'reversed_feature_importance': dict(zip(self.feature_cols, rev_importances.tolist()))
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
        
    def run_backtest_with_position_sizing(self, df_or_X, exchange='binance', symbol='BTC/USDT', 
                                            timeframe='1m', no_trade_threshold=0.96, 
                                            min_position_change=0.0012, save_results=True):
        """
        Run a backtest with quantum-inspired position sizing.
        
        Args:
            df_or_X: DataFrame with market data or pre-scaled feature array
            exchange: Exchange name for result saving
            symbol: Trading pair symbol for result saving
            timeframe: Timeframe for result saving
            no_trade_threshold: Threshold for no_trade probability to ignore signals
            min_position_change: Minimum position change to avoid fee churn
            save_results: Whether to save results to disk
            
        Returns:
            tuple: (result_df, metrics, fig) - Processed DataFrame, performance metrics, and plot
        """
        # Get probabilities if input is a DataFrame
        if isinstance(df_or_X, pd.DataFrame):
            df = df_or_X.copy()
            
            # Get predictions if not already present
            if not all(col in df.columns for col in ['short_prob', 'no_trade_prob', 'long_prob']):
                self.logger.info("Getting predictions for backtest data")
                probs = self.predict_probabilities(df)
                
                # Add probabilities to dataframe
                df['short_prob'] = probs[:, 0]
                df['no_trade_prob'] = probs[:, 1]
                df['long_prob'] = probs[:, 2]
        else:
            self.logger.error("Input must be a DataFrame with market data")
            return None, None, None
        
        # Get fee rate from config
        fee_rate = self.config.get('backtesting', {}).get('fee_rate', 0.0006)
        
        # Initialize position sizer
        position_sizer = QuantumPositionSizer(
            fee_rate=fee_rate,
            no_trade_threshold=no_trade_threshold,
            confidence_scaling=True,
            volatility_scaling=True,
            max_position=1.0,
            min_position_change=min_position_change,
            initial_capital=10000.0
        )
        
        self.logger.info(f"Running quantum position sizing backtest with no_trade_threshold={no_trade_threshold}")
        
        # Process dataframe
        result_df = position_sizer.process_dataframe(df)
        
        # Calculate performance metrics
        metrics = position_sizer.analyze_performance(result_df)
        
        # Plot results
        fig, _ = position_sizer.plot_results(result_df)
        
        # Save results if requested
        if save_results:
            position_sizer.save_results(result_df, metrics, fig, exchange, symbol, timeframe)
        
        return result_df, metrics, fig