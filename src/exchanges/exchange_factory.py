#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exchange Client Factory

This module provides a factory for creating exchange-specific API clients.
It handles dynamic loading of exchange implementations and centralized configuration.
"""

import logging
import importlib
import os
from typing import Dict, List, Optional, Union, Type, Any

# Import the base client interface
from base_exchange import ExchangeClientBase
from src.core.param_manager import ParamManager

# Explicitly import supported exchange clients
# This ensures they're available even if not directly imported elsewhere
from bitget_exchange import BitgetClient


class ExchangeClientFactory:
    """
    Factory for creating exchange API clients.
    
    This factory handles the instantiation of exchange-specific clients
    and ensures they all follow the same interface.
    """
    
    _instance = None
    _client_registry = {}
    _client_instances = {}
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the factory."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the factory and register known client classes."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load parameter manager
        self.params = ParamManager.get_instance()
        
        # Register known client classes
        self._register_client_class('bitget', BitgetClient)
        
        # Try to discover additional exchange clients
        self._discover_clients()
    
    def _register_client_class(self, exchange_id: str, client_class: Type[ExchangeClientBase]):
        """
        Register an exchange client class.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'bitget')
            client_class: Exchange client class
        """
        exchange_id = exchange_id.lower()
        self._client_registry[exchange_id] = client_class
        self.logger.debug(f"Registered exchange client: {exchange_id}")
    
    def _discover_clients(self):
        """
        Discover exchange client implementations from the api directory.
        """
        try:
            # Get the module directory
            module_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Look for Python files in the directory
            for filename in os.listdir(module_dir):
                if filename.endswith('_client.py') and filename != 'base_client.py':
                    # Extract exchange name from filename
                    exchange_id = filename.replace('_client.py', '').lower()
                    
                    # Skip already registered clients
                    if exchange_id in self._client_registry:
                        continue
                    
                    try:
                        # Import the module
                        module_name = f"src.api.{filename[:-3]}"
                        module = importlib.import_module(module_name)
                        
                        # Look for a class that implements ExchangeClientBase
                        for attr_name in dir(module):
                            if attr_name.endswith('Client') and attr_name != 'ExchangeClientBase':
                                client_class = getattr(module, attr_name)
                                if (isinstance(client_class, type) and 
                                    issubclass(client_class, ExchangeClientBase)):
                                    self._register_client_class(exchange_id, client_class)
                                    break
                    except (ImportError, AttributeError) as e:
                        self.logger.warning(f"Error loading exchange client {exchange_id}: {e}")
        except Exception as e:
            self.logger.error(f"Error discovering exchange clients: {e}")
    
    def get_available_exchanges(self) -> List[str]:
        """
        Get a list of available exchange clients.
        
        Returns:
            List of exchange identifiers
        """
        return list(self._client_registry.keys())
    
    def create_client(self, exchange_id: str, api_key: Optional[str] = None, 
                    api_secret: Optional[str] = None, passphrase: Optional[str] = None,
                    use_testnet: Optional[bool] = None) -> ExchangeClientBase:
        """
        Create an exchange client instance for the specified exchange.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'bitget')
            api_key: API key (optional, will load from config if not provided)
            api_secret: API secret (optional, will load from config if not provided)
            passphrase: API passphrase (optional, will load from config if not provided)
            use_testnet: Whether to use testnet (optional, will load from config if not provided)
            
        Returns:
            Exchange client instance
            
        Raises:
            ValueError: If the exchange is not supported
        """
        exchange_id = exchange_id.lower()
        
        # Check if client class is registered
        if exchange_id not in self._client_registry:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
        
        # Get default configuration from parameter manager
        if api_key is None:
            api_key = self.params.get('exchanges', exchange_id, 'api_key')
        
        if api_secret is None:
            api_secret = self.params.get('exchanges', exchange_id, 'api_secret')
        
        if passphrase is None:
            passphrase = self.params.get('exchanges', exchange_id, 'passphrase')
        
        if use_testnet is None:
            use_testnet = self.params.get('exchanges', exchange_id, 'testnet', default=True)
        
        # Create client instance
        client_class = self._client_registry[exchange_id]
        client = client_class(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            use_testnet=use_testnet
        )
        
        return client
    
    def get_client(self, exchange_id: str, force_new: bool = False) -> ExchangeClientBase:
        """
        Get a cached client instance or create a new one.
        
        This method caches client instances to avoid creating multiple connections
        to the same exchange.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'bitget')
            force_new: Whether to force creation of a new instance
            
        Returns:
            Exchange client instance
            
        Raises:
            ValueError: If the exchange is not supported
        """
        exchange_id = exchange_id.lower()
        
        # Return cached instance if available and not forcing new
        if not force_new and exchange_id in self._client_instances:
            return self._client_instances[exchange_id]
        
        # Create new client instance
        client = self.create_client(exchange_id)
        
        # Cache the instance
        self._client_instances[exchange_id] = client
        
        return client