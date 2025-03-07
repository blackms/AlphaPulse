"""
Storage implementations for the AlphaPulse Monitoring System.
"""

from .interfaces import TimeSeriesStorage, MetricsStorageFactory
from .memory import InMemoryStorage
from .influxdb import InfluxDBStorage
from .timescaledb import TimescaleDBStorage
from ..config import load_config

__all__ = [
    'TimeSeriesStorage',
    'MetricsStorageFactory',
    'InMemoryStorage',
    'InfluxDBStorage',
    'TimescaleDBStorage',
    'get_storage_manager'
]


def get_storage_manager() -> TimeSeriesStorage:
    """
    Get a configured storage manager instance based on the monitoring configuration.
    
    Returns:
        TimeSeriesStorage: A configured storage manager instance
    """
    config = load_config()
    storage_config = config.storage.get_storage_config()
    return MetricsStorageFactory.create_storage(config.storage.type, storage_config)