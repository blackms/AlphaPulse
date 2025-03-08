"""
Scheduler module for periodic data synchronization.

This module provides the infrastructure for scheduling and executing
periodic data synchronization tasks.
"""
from enum import Enum
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


# Import from the new module structure
from alpha_pulse.data_pipeline.scheduler.sync_module import ExchangeDataSynchronizer

# Get the singleton instance of the synchronizer
# Since ExchangeDataSynchronizer is a singleton, this will either create a new instance
# or return the existing one
logger.debug("Getting ExchangeDataSynchronizer singleton instance")
exchange_data_synchronizer = ExchangeDataSynchronizer()
logger.debug(f"ExchangeDataSynchronizer instance: {exchange_data_synchronizer}")

# Export classes and instances to be used in other modules
__all__ = [
    "DataType",
    "ExchangeDataSynchronizer",
    "exchange_data_synchronizer"
]