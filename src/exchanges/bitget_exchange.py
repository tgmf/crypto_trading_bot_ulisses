#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bitget Exchange API Client

This module implements the Bitget-specific client for the exchange API
interface, focusing on limit order placement for maker fee optimization.
"""

import logging
import time
import json
import hmac
import base64
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
from urllib.parse import urlencode
import pandas as pd
from dotenv import load_dotenv
import os

from base_exchange import ExchangeClientBase
from ..core.param_manager import ParamManager


class BitgetClient(ExchangeClientBase):
    """
    Client for interacting with the Bitget API with a focus on limit orders.
    
    This client handles authentication, rate limiting, and provides methods for
    order placement, monitoring, and management.
    """
    
    def __init__(self, api_key=None, api_secret=None, passphrase=None, use_testnet=True):
        """
        Initialize the Bitget API client.
        
        Args:
            api_key: Bitget API key. If None, loads from environment variable BITGET_API_KEY.
            api_secret: Bitget API secret. If None, loads from environment variable BITGET_API_SECRET.
            passphrase: Bitget API passphrase. If None, loads from environment variable BITGET_API_PASSPHRASE.
            use_testnet: Whether to use the testnet (default True).
        """
        super().__init__(api_key, api_secret, passphrase, use_testnet)
        
        # Load environment variables
        load_dotenv()
        
        # Set API credentials (prioritize direct args, fall back to env vars)
        self.api_key = api_key or os.getenv('BITGET_API_KEY')
        self.api_secret = api_secret or os.getenv('BITGET_API_SECRET')
        self.passphrase = passphrase or os.getenv('BITGET_API_PASSPHRASE')
        
        # Validate credentials
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise ValueError("API key, secret, and passphrase are required")
        
        # Set up API URLs
        self.use_testnet = use_testnet
        if use_testnet:
            self.base_url = "https://api.bitget-uat.com"
            self.logger.info("Using Bitget testnet")
        else:
            self.base_url = "https://api.bitget.com"
            self.logger.info("Using Bitget production API")
        
        # Set up rate limiting
        self.request_interval = 0.2  # 200ms between requests to avoid rate limits
        self.last_request_time = 0
        
        # Initialize session for connection pooling
        self.session = requests.Session()
        
        # Store instruments data (will be populated on first use)
        self._instruments = {}
        
    def _get_timestamp(self):
        """Get ISO 8601 timestamp for request signing."""
        return str(int(time.time() * 1000))
    
    def _generate_signature(self, timestamp, method, request_path, body=None):
        """
        Generate a signature for API authentication.
        
        Args:
            timestamp: Request timestamp
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body for POST/PUT requests
            
        Returns:
            str: Base64-encoded HMAC-SHA256 signature
        """
        # Create message string
        if body is None or body == "":
            message = timestamp + method + request_path
        else:
            message = timestamp + method + request_path + body
            
        # Create HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        
        # Return Base64-encoded signature
        return base64.b64encode(signature).decode('utf-8')
    
    def _request(self, method, endpoint, params=None, data=None):
        """
        Send a request to the Bitget API with authentication.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: URL parameters for GET requests
            data: JSON body for POST/PUT requests
            
        Returns:
            dict: API response
        """
        # Apply rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Convert data to JSON if provided
        json_data = None
        if data:
            json_data = json.dumps(data)
        
        # Create URL with parameters for GET requests
        if params and method.upper() == 'GET':
            url += '?' + urlencode(params)
        
        # Create authentication headers
        timestamp = self._get_timestamp()
        signature = self._generate_signature(
            timestamp, 
            method.upper(), 
            endpoint + ('?' + urlencode(params) if params and method.upper() == 'GET' else ''),
            json_data
        )
        
        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        # Log request details (debug level)
        self.logger.debug(f"Request: {method} {url}")
        if data:
            self.logger.debug(f"Request body: {json_data}")
        
        # Send request
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params if method.upper() != 'GET' else None,
                data=json_data,
                timeout=10  # 10 second timeout
            )
            
            # Handle HTTP errors
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Check for API errors
            if result.get('code') != '00000':
                self.logger.error(f"API error: {result.get('msg', 'Unknown error')}")
                raise ValueError(f"API error: {result.get('msg', 'Unknown error')}")
            
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
    
    def get_account_info(self):
        """
        Get account information including balance.
        
        Returns:
            dict: Account information
        """
        return self._request("GET", "/api/v2/spot/account/info")
    
    def get_ticker(self, symbol):
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Ticker information
        """
        return self._request("GET", "/api/v2/spot/market/ticker", {"symbol": symbol})
    
    def get_order_book(self, symbol, limit=50):
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of order book entries (default 50)
            
        Returns:
            dict: Order book information
        """
        bitget_response = self._request("GET", "/api/v2/spot/market/orderbook", {
            "symbol": symbol,
            "limit": limit
        })
        
        # Transform to standard format if successful
        if bitget_response.get('code') == '00000' and 'data' in bitget_response:
            return {
                'exchange': 'bitget',
                'symbol': symbol,
                'bids': bitget_response['data'].get('bids', []),
                'asks': bitget_response['data'].get('asks', []),
                'timestamp': int(time.time() * 1000),
                'raw_response': bitget_response
            }
        
        return bitget_response
    
    def place_limit_order(self, symbol, side, price, size, client_order_id=None,
                         time_in_force="post_only"):
        """
        Place a limit order (maker order).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            price: Order price
            size: Order quantity
            client_order_id: Client-defined order ID
            time_in_force: Time in force: 'normal', 'post_only', 'fok', 'ioc'
                           'post_only' ensures the order is a maker order
            
        Returns:
            dict: Order response
        """
        # Normalize inputs
        side = side.lower()
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        time_in_force = time_in_force.lower()
        if time_in_force not in ['normal', 'post_only', 'fok', 'ioc']:
            raise ValueError("Time in force must be 'normal', 'post_only', 'fok', or 'ioc'")
        
        # Ensure post_only for maker orders
        if time_in_force != 'post_only':
            self.logger.warning("Using time_in_force='post_only' to ensure maker orders")
            time_in_force = 'post_only'
        
        # Build order data
        order_data = {
            "symbol": symbol,
            "side": side,
            "orderType": "limit",
            "price": price,
            "quantity": size,
            "force": time_in_force,
        }
        
        # Add client order ID if provided
        if client_order_id:
            order_data["clientOrderId"] = client_order_id
        
        # Place order
        self.logger.info(f"Placing limit order: {order_data}")
        response = self._request("POST", "/api/v2/spot/trade/place-order", None, order_data)
        self.logger.info(f"Order placed: {response}")
        
        return response
    
    def place_market_order(self, symbol, side, size, client_order_id=None):
        """
        Place a market order (taker order).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            size: Order quantity
            client_order_id: Client-defined order ID
            
        Returns:
            dict: Order response
        """
        # Normalize inputs
        side = side.lower()
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        # Build order data
        order_data = {
            "symbol": symbol,
            "side": side,
            "orderType": "market",
            "quantity": size,
        }
        
        # Add client order ID if provided
        if client_order_id:
            order_data["clientOrderId"] = client_order_id
        
        # Place order
        self.logger.warning("Placing market order (taker order, higher fees)")
        return self._request("POST", "/api/v2/spot/trade/place-order", None, order_data)
    
    def get_order(self, symbol, order_id=None, client_order_id=None):
        """
        Get order information.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            order_id: Exchange order ID
            client_order_id: Client-defined order ID
            
        Returns:
            dict: Order information
        """
        # Check that at least one ID is provided
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")
        
        params = {"symbol": symbol}
        
        # Add appropriate ID
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["clientOrderId"] = client_order_id
        
        return self._request("GET", "/api/v2/spot/trade/orderInfo", params)
    
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            order_id: Exchange order ID
            client_order_id: Client-defined order ID
            
        Returns:
            dict: Cancellation response
        """
        # Check that at least one ID is provided
        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")
        
        # Build cancellation data
        cancel_data = {"symbol": symbol}
        
        # Add appropriate ID
        if order_id:
            cancel_data["orderId"] = order_id
        elif client_order_id:
            cancel_data["clientOrderId"] = client_order_id
        
        self.logger.info(f"Cancelling order: {cancel_data}")
        return self._request("POST", "/api/v2/spot/trade/cancel-order", None, cancel_data)
    
    def get_open_orders(self, symbol=None):
        """
        Get all open orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Open orders information
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        return self._request("GET", "/api/v2/spot/trade/open-orders", params)
    
    def get_instruments(self, force_refresh=False):
        """
        Get all trading instruments and their specifications.
        
        Args:
            force_refresh: Whether to force a refresh from the API
            
        Returns:
            dict: Instruments information indexed by symbol
        """
        # Return cached data if available and not forcing refresh
        if self._instruments and not force_refresh:
            return self._instruments
        
        response = self._request("GET", "/api/v2/spot/public/instruments")
        
        # Index by symbol for easier lookup
        instruments = {}
        for instrument in response.get('data', []):
            symbol = instrument.get('symbol')
            if symbol:
                instruments[symbol] = instrument
        
        # Cache instruments data
        self._instruments = instruments
        
        return instruments
    
    def get_instrument(self, symbol):
        """
        Get information for a specific instrument.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Instrument information
        """
        # Load instruments if not already loaded
        if not self._instruments:
            self.get_instruments()
        
        # Return specific instrument
        if symbol in self._instruments:
            return self._instruments[symbol]
        
        # Try to get it by refreshing instruments
        self.get_instruments(force_refresh=True)
        
        if symbol in self._instruments:
            return self._instruments[symbol]
        
        raise ValueError(f"Instrument {symbol} not found")
    
    def get_min_order_size(self, symbol):
        """
        Get minimum order size for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Minimum order size
        """
        instrument = self.get_instrument(symbol)
        return float(instrument.get('minTradeAmount', 0))
    
    def get_price_precision(self, symbol):
        """
        Get price precision for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            int: Price precision (decimal places)
        """
        instrument = self.get_instrument(symbol)
        return int(instrument.get('pricePrecision', 0))
    
    def get_size_precision(self, symbol):
        """
        Get size precision for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            int: Size precision (decimal places)
        """
        instrument = self.get_instrument(symbol)
        return int(instrument.get('quantityPrecision', 0))
    
    def format_price(self, symbol, price):
        """
        Format price according to symbol precision.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            price: Price to format
            
        Returns:
            str: Formatted price
        """
        precision = self.get_price_precision(symbol)
        return f"{float(price):.{precision}f}"
    
    def format_size(self, symbol, size):
        """
        Format size according to symbol precision.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            size: Size to format
            
        Returns:
            str: Formatted size
        """
        precision = self.get_size_precision(symbol)
        return f"{float(size):.{precision}f}"
    
    def calculate_maker_price(self, symbol, side, offset_percentage=0.0012):
        """
        Calculate a price for a maker order based on the current book.
        
        For buy orders, this will be slightly below the current bid.
        For sell orders, this will be slightly above the current ask.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            offset_percentage: Percentage to offset from best bid/ask (default 0.12%)
            
        Returns:
            str: Formatted price for a maker order
        """
        orderbook = self.get_order_book(symbol, limit=5)
        
        if 'bids' not in orderbook or 'asks' not in orderbook:
            raise ValueError(f"Could not get orderbook for {symbol}")
        
        if side.lower() == 'buy':
            # For buy orders, use a price slightly below the highest bid
            bids = orderbook['bids']
            if not bids:
                raise ValueError(f"No bids found for {symbol}")
            
            best_bid_price = float(bids[0][0])
            maker_price = best_bid_price * (1 - offset_percentage)
        else:
            # For sell orders, use a price slightly above the lowest ask
            asks = orderbook['asks']
            if not asks:
                raise ValueError(f"No asks found for {symbol}")
            
            best_ask_price = float(asks[0][0])
            maker_price = best_ask_price * (1 + offset_percentage)
        
        return self.format_price(symbol, maker_price)
    
    def place_optimal_maker_order(self, symbol, side, size, 
                                offset_percentage=0.0012, 
                                client_order_id=None):
        """
        Place a limit order with an optimal price to ensure it's a maker order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            size: Order quantity
            offset_percentage: Percentage to offset from best bid/ask (default 0.12%)
            client_order_id: Client-defined order ID
            
        Returns:
            dict: Order response
        """
        # Format size
        formatted_size = self.format_size(symbol, size)
        
        # Calculate maker price
        maker_price = self.calculate_maker_price(symbol, side, offset_percentage)
        
        # Place limit order with post_only flag
        return self.place_limit_order(
            symbol=symbol,
            side=side,
            price=maker_price,
            size=formatted_size,
            client_order_id=client_order_id,
            time_in_force="post_only"
        )
    
    def place_smart_limit_order(self, symbol, side, total_amount, min_order_count=1, 
                               max_order_count=5, spread_range=0.03, client_order_id_prefix=None):
        """
        Place multiple limit orders across a price range for better fills.
        
        This method splits a large order into multiple smaller orders at different
        price levels to increase the chances of getting filled as a maker.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            total_amount: Total order quantity
            min_order_count: Minimum number of orders to place
            max_order_count: Maximum number of orders to place
            spread_range: Price range as a percentage (default 3%)
            client_order_id_prefix: Prefix for client order IDs
            
        Returns:
            list: List of order responses
        """
        # Validate inputs
        if min_order_count < 1:
            min_order_count = 1
        if max_order_count < min_order_count:
            max_order_count = min_order_count
        
        # Get instrument details
        min_order_size = self.get_min_order_size(symbol)
        
        # Calculate number of orders based on total size
        # More orders for larger sizes, but respecting min/max
        order_count = min(max(min_order_count, int(total_amount / min_order_size / 5)), max_order_count)
        
        # Adjust if we can't have enough orders
        if total_amount / order_count < min_order_size:
            order_count = max(1, int(total_amount / min_order_size))
        
        # Calculate size per order (equal distribution)
        size_per_order = total_amount / order_count
        
        # Get orderbook
        orderbook = self.get_order_book(symbol, limit=20)
        
        if 'bids' not in orderbook or 'asks' not in orderbook:
            raise ValueError(f"Could not get orderbook for {symbol}")
        
        # Determine base price
        if side.lower() == 'buy':
            bids = orderbook['bids']
            if not bids:
                raise ValueError(f"No bids found for {symbol}")
            base_price = float(bids[0][0])
        else:
            asks = orderbook['asks']
            if not asks:
                raise ValueError(f"No asks found for {symbol}")
            base_price = float(asks[0][0])
        
        # Calculate price range
        if side.lower() == 'buy':
            # For buys, set prices below the best bid
            min_price = base_price * (1 - spread_range)
            max_price = base_price * 0.9995  # Just below best bid
        else:
            # For sells, set prices above the best ask
            min_price = base_price * 1.0005  # Just above best ask
            max_price = base_price * (1 + spread_range)
        
        # Calculate price step
        price_step = (max_price - min_price) / (order_count - 1) if order_count > 1 else 0
        
        # Place orders
        orders = []
        for i in range(order_count):
            # Calculate price for this order
            if order_count == 1:
                # If only one order, use the base price with offset
                if side.lower() == 'buy':
                    price = base_price * 0.9995
                else:
                    price = base_price * 1.0005
            else:
                # Otherwise, distribute prices across range
                if side.lower() == 'buy':
                    # For buys, start from higher price to lower
                    price = max_price - (i * price_step)
                else:
                    # For sells, start from lower price to higher
                    price = min_price + (i * price_step)
            
            # Format price and size
            formatted_price = self.format_price(symbol, price)
            formatted_size = self.format_size(symbol, size_per_order)
            
            # Create client order ID if prefix provided
            client_order_id = None
            if client_order_id_prefix:
                client_order_id = f"{client_order_id_prefix}_{i+1}"
            
            # Place order
            try:
                order = self.place_limit_order(
                    symbol=symbol,
                    side=side,
                    price=formatted_price,
                    size=formatted_size,
                    client_order_id=client_order_id,
                    time_in_force="post_only"
                )
                orders.append(order)
                
                # Brief delay between orders
                time.sleep(0.05)
                
            except Exception as e:
                self.logger.error(f"Error placing order {i+1}/{order_count}: {e}")
        
        return orders
    
    def cancel_all_orders(self, symbol=None):
        """
        Cancel all open orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Cancellation response
        """
        # Build cancellation data
        cancel_data = {}
        if symbol:
            cancel_data["symbol"] = symbol
        
        return self._request("POST", "/api/v2/spot/trade/cancel-all-orders", None, cancel_data)
    
    def get_maker_fee(self, symbol=None):
        """
        Get the maker fee rate for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Maker fee rate (e.g., 0.0006 for 0.06%)
        """
        # Bitget standard maker fee is 0.06%
        return 0.0006
    
    def get_taker_fee(self, symbol=None):
        """
        Get the taker fee rate for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Taker fee rate (e.g., 0.001 for 0.1%)
        """
        # Bitget standard taker fee is 0.1%
        return 0.001