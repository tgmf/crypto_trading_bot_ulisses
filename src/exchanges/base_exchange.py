#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Exchange Client Interface

This module defines the abstract base class that all exchange-specific
clients must implement, ensuring a consistent interface across exchanges.
"""

import logging 
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple

class ExchangeClientBase(ABC):
    """
    Abstract base class for cryptocurrency exchange API clients.
    
    All exchange-specific implementations should inherit from this class
    and implement the required methods to provide a consistent interface
    across different exchanges.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                api_secret: Optional[str] = None, 
                passphrase: Optional[str] = None,
                use_testnet: bool = True):
        """
        Initialize the exchange client.
        
        Args:
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            passphrase: API passphrase (if required by the exchange)
            use_testnet: Whether to use the exchange's testnet
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.use_testnet = use_testnet
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balances.
        
        Returns:
            dict: Account information
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Ticker information
        """
        pass
    
    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of order book entries
            
        Returns:
            dict: Standardized order book information
        """
        pass
    
    @abstractmethod
    def place_limit_order(self, symbol: str, side: str, price: str, size: str, 
                        client_order_id: Optional[str] = None,
                        time_in_force: str = "post_only") -> Dict[str, Any]:
        """
        Place a limit order (maker order).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            price: Order price
            size: Order quantity
            client_order_id: Client-defined order ID
            time_in_force: Time in force instruction
            
        Returns:
            dict: Order response
        """
        pass
    
    @abstractmethod
    def place_market_order(self, symbol: str, side: str, size: str,
                          client_order_id: Optional[str] = None) -> Dict[str, Any]:
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
        pass
    
    @abstractmethod
    def get_order(self, symbol: str, order_id: Optional[str] = None, 
                client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get order information.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            order_id: Exchange order ID
            client_order_id: Client-defined order ID
            
        Returns:
            dict: Order information
        """
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: Optional[str] = None,
                    client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            order_id: Exchange order ID
            client_order_id: Client-defined order ID
            
        Returns:
            dict: Cancellation response
        """
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all open orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Open orders information
        """
        pass
    
    @abstractmethod
    def calculate_maker_price(self, symbol: str, side: str, 
                             offset_percentage: float = 0.0015) -> str:
        """
        Calculate a price for a maker order based on the current book.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            offset_percentage: Percentage to offset from best bid/ask
            
        Returns:
            str: Formatted price for a maker order
        """
        pass
    
    @abstractmethod
    def place_optimal_maker_order(self, symbol: str, side: str, size: float,
                                offset_percentage: float = 0.0015,
                                client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Place a limit order with an optimal price to ensure it's a maker order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            size: Order quantity
            offset_percentage: Percentage to offset from best bid/ask
            client_order_id: Client-defined order ID
            
        Returns:
            dict: Order response
        """
        pass
    
    @abstractmethod
    def place_smart_limit_order(self, symbol: str, side: str, total_amount: float,
                              min_order_count: int = 1, max_order_count: int = 5,
                              spread_range: float = 0.03, 
                              client_order_id_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Place multiple limit orders across a price range for better fills.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            total_amount: Total order quantity
            min_order_count: Minimum number of orders to place
            max_order_count: Maximum number of orders to place
            spread_range: Price range as a percentage
            client_order_id_prefix: Prefix for client order IDs
            
        Returns:
            list: List of order responses
        """
        pass
    
    @abstractmethod
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel all open orders.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Cancellation response
        """
        pass
    
    @abstractmethod
    def get_min_order_size(self, symbol: str) -> float:
        """
        Get minimum order size for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Minimum order size
        """
        pass
    
    @abstractmethod
    def get_price_precision(self, symbol: str) -> int:
        """
        Get price precision for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            int: Price precision (decimal places)
        """
        pass
    
    @abstractmethod
    def get_size_precision(self, symbol: str) -> int:
        """
        Get size precision for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            int: Size precision (decimal places)
        """
        pass
    
    @abstractmethod
    def format_price(self, symbol: str, price: float) -> str:
        """
        Format price according to symbol precision.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            price: Price to format
            
        Returns:
            str: Formatted price
        """
        pass
    
    @abstractmethod
    def format_size(self, symbol: str, size: float) -> str:
        """
        Format size according to symbol precision.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            size: Size to format
            
        Returns:
            str: Formatted size
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of the exchange.
        
        Returns:
            str: Exchange name
        """
        return self.__class__.__name__.replace('Client', '')
    
    def get_maker_fee(self, symbol: Optional[str] = None) -> float:
        """
        Get the maker fee rate for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Maker fee rate (e.g., 0.0006 for 0.06%)
        """
        # Default implementation - override if exchange provides fee info
        return 0.0002  # Default 0.02%
    
    def get_taker_fee(self, symbol: Optional[str] = None) -> float:
        """
        Get the taker fee rate for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Taker fee rate (e.g., 0.001 for 0.1%)
        """
        # Default implementation - override if exchange provides fee info
        return 0.0006  # Default 0.06%