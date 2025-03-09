#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the data collector module.
"""

import pytest
import os
import pandas as pd
from pathlib import Path
import yaml

# Import module to test
from src.data.data_collector import DataCollector

class TestDataCollector:
    """Tests for DataCollector class"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return {
            'data': {
                'exchanges': ['binance'],
                'symbols': ['BTC/USDT'],
                'timeframes': ['1h'],
                'start_date': '2023-01-01'
            }
        }
    
    def test_initialization(self, config):
        """Test initialization of data collector"""
        collector = DataCollector(config)
        assert collector is not None
        assert 'binance' in collector.exchanges