"""
Scheduler module for periodic data synchronization.

This module provides the infrastructure for scheduling and executing
periodic data synchronization tasks.
"""
from enum import Enum, auto
from typing import Optional, Dict, Any, List

from loguru import logger


class DataType(Enum):
    """Types of data that can be synchronized."""
    BALANCES = "balances" 
    POSITIONS = "positions"
    ORDERS = "orders"
    PRICES = "prices"
    ALL = "all"
    
    def __str__(self):
        return self.value


# This will be initialized by the exchange_data_synchronizer module
exchange_data_synchronizer = None

# Import at the end to avoid circular imports
from alpha_pulse.data_pipeline.scheduler.exchange_synchronizer import ExchangeDataSynchronizer

# Create the global instance of the synchronizer
exchange_data_synchronizer = ExchangeDataSynchronizer()