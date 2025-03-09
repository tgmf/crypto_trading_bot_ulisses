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
    
    def train(self, exchange='binance', symbol='BTC/USDT', timeframe='1h'):
        """Train the model on processed data"""
        self.logger.info(f"Training model for {exchange} {symbol} {timeframe}")
        
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
            
            # Use a subset of data for initial development (faster iterations)
            train_size = min(len(df), 5000)  # Limit to 5000 samples for initial development
            train_df = df.iloc[:train_size]
            
            # Prepare training data
            X_train = train_df[self.feature_cols].values
            y_train = train_df['target'].values
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Build and train model
            self.logger.info("Building Bayesian model...")
            self.build_model(X_train_scaled, y_train)
            
            # Save model
            self.save_model(exchange, symbol, timeframe)
            
            return True
            
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