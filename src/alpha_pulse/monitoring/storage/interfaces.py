"""
Interfaces for time series database storage.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta


class TimeSeriesStorage(ABC):
    """
    Abstract base class for time series database storage.
    Implementations should handle connection management and data operations.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the database.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the database connection.
        """
        pass

    @abstractmethod
    async def store_metrics(
        self, metric_type: str, timestamp: datetime, data: Dict[str, Any]
    ) -> None:
        """
        Store metrics in the time series database.

        Args:
            metric_type: Type of metrics (e.g., 'performance', 'risk', 'trade')
            timestamp: Timestamp for the metrics
            data: Dictionary of metric values
        """
        pass

    @abstractmethod
    async def query_metrics(
        self,
        metric_type: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics from the time series database.

        Args:
            metric_type: Type of metrics to query
            start_time: Start time for the query range
            end_time: End time for the query range
            aggregation: Optional aggregation interval (e.g., '1m', '1h', '1d')

        Returns:
            List of metric data points
        """
        pass

    @abstractmethod
    async def get_latest_metrics(
        self, metric_type: str, limit: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get the latest metrics of a specific type.

        Args:
            metric_type: Type of metrics to query
            limit: Maximum number of data points to return

        Returns:
            List of the latest metric data points
        """
        pass

    @abstractmethod
    async def delete_metrics(
        self, metric_type: str, older_than: Optional[datetime] = None
    ) -> int:
        """
        Delete metrics from the database.

        Args:
            metric_type: Type of metrics to delete
            older_than: Optional timestamp, delete metrics older than this

        Returns:
            Number of deleted data points
        """
        pass

    @abstractmethod
    async def create_retention_policy(
        self, name: str, duration: timedelta, default: bool = False
    ) -> None:
        """
        Create a retention policy for automatic data cleanup.

        Args:
            name: Name of the retention policy
            duration: How long to keep the data
            default: Whether this is the default policy
        """
        pass


class MetricsStorageFactory:
    """
    Factory for creating time series storage instances.
    """

    @staticmethod
    def create_storage(storage_type: str, config: Dict[str, Any]) -> TimeSeriesStorage:
        """
        Create a storage instance based on the specified type.

        Args:
            storage_type: Type of storage ('influxdb', 'timescaledb', etc.)
            config: Configuration parameters for the storage

        Returns:
            TimeSeriesStorage implementation
        """
        if storage_type == "influxdb":
            from .influxdb import InfluxDBStorage
            return InfluxDBStorage(config)
        elif storage_type == "timescaledb":
            from .timescaledb import TimescaleDBStorage
            return TimescaleDBStorage(config)
        elif storage_type == "memory":
            from .memory import InMemoryStorage
            return InMemoryStorage(config)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")