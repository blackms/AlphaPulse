"""
In-memory implementation of time series storage for development and testing.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import bisect
import copy
from collections import defaultdict

from .interfaces import TimeSeriesStorage


class InMemoryStorage(TimeSeriesStorage):
    """
    In-memory implementation of time series storage.
    Useful for development, testing, and environments where a database is not available.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize in-memory storage.

        Args:
            config: Configuration parameters
                - max_points: Maximum number of data points per metric type (default: 10000)
                - retention_duration: Default retention duration (default: 7 days)
        """
        self.max_points = config.get("max_points", 10000)
        self.retention_duration = config.get("retention_duration", timedelta(days=7))
        self.retention_policies = {}
        
        # Main storage structure: Dict[metric_type, List[Tuple[timestamp, data]]]
        # We'll keep this sorted by timestamp for efficient querying
        self._storage = defaultdict(list)
        self._timestamps = defaultdict(list)  # For binary search
        self._connected = False

    async def connect(self) -> None:
        """
        Establish connection (no-op for in-memory storage).
        """
        self._connected = True

    async def disconnect(self) -> None:
        """
        Close the connection (no-op for in-memory storage).
        """
        self._connected = False

    async def store_metrics(
        self, metric_type: str, timestamp: datetime, data: Dict[str, Any]
    ) -> None:
        """
        Store metrics in memory.

        Args:
            metric_type: Type of metrics
            timestamp: Timestamp for the metrics
            data: Dictionary of metric values
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Find the insertion point to maintain sorted order
        timestamps = self._timestamps[metric_type]
        storage = self._storage[metric_type]
        
        # Use binary search to find insertion point
        idx = bisect.bisect_left(timestamps, timestamp)
        
        # If timestamp already exists, update the data
        if idx < len(timestamps) and timestamps[idx] == timestamp:
            storage[idx] = (timestamp, copy.deepcopy(data))
        else:
            # Insert new data point
            timestamps.insert(idx, timestamp)
            storage.insert(idx, (timestamp, copy.deepcopy(data)))
        
        # Enforce max points limit
        if len(storage) > self.max_points:
            self._storage[metric_type] = storage[-self.max_points:]
            self._timestamps[metric_type] = timestamps[-self.max_points:]
        
        # Apply retention policy
        await self._apply_retention_policy(metric_type)

    async def query_metrics(
        self,
        metric_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        aggregation: Optional[str] = None,
        metric_names: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics from memory.

        Args:
            metric_type: Type of metrics to query
            start_time: Start time for the query range
            end_time: End time for the query range
            aggregation: Optional aggregation interval (e.g., '1m', '1h', '1d')
            metric_names: List of specific metric names to query (overrides metric_type)

        Returns:
            List of metric data points
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Use metric_names if provided, otherwise use metric_type
        metric_types_to_query = []
        if metric_names:
            metric_types_to_query = metric_names
        elif metric_type and metric_type != "all":
            metric_types_to_query = [metric_type]
        else:
            # Query all available metrics
            metric_types_to_query = list(self._storage.keys())

        # Default time range if not specified
        if start_time is None:
            start_time = datetime.min
        if end_time is None:
            end_time = datetime.max

        result = []
        for mt in metric_types_to_query:
            if mt not in self._storage:
                continue

            # Find indices for the time range using binary search
            timestamps = self._timestamps[mt]
            start_idx = bisect.bisect_left(timestamps, start_time)
            end_idx = bisect.bisect_right(timestamps, end_time)
            
            # Extract data points in the range
            data_points = self._storage[mt][start_idx:end_idx]
            
            # If no aggregation requested, return raw data
            if not aggregation:
                result.extend([{"timestamp": ts, "name": mt, **data} for ts, data in data_points])
            else:
                # Perform aggregation
                aggregated = self._aggregate_data(data_points, aggregation)
                for point in aggregated:
                    point["name"] = mt
                result.extend(aggregated)
        
        return result

    async def get_latest_metrics(
        self, metric_type: str = None, limit: int = 1, metric_names: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the latest metrics of a specific type.

        Args:
            metric_type: Type of metrics to query
            limit: Maximum number of data points to return
            metric_names: List of specific metric names to query (overrides metric_type)

        Returns:
            List of the latest metric data points
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Use metric_names if provided, otherwise use metric_type
        metric_types_to_query = []
        if metric_names:
            metric_types_to_query = metric_names
        elif metric_type and metric_type != "all":
            metric_types_to_query = [metric_type]
        else:
            # Query all available metrics
            metric_types_to_query = list(self._storage.keys())

        result = []
        for mt in metric_types_to_query:
            if mt not in self._storage:
                continue

            # Get the latest data points
            latest_points = self._storage[mt][-limit:]
            
            # Format the response
            result.extend([{"timestamp": ts, "name": mt, **data} for ts, data in latest_points])
        
        return result

    async def delete_metrics(
        self, metric_type: str, older_than: Optional[datetime] = None
    ) -> int:
        """
        Delete metrics from memory.

        Args:
            metric_type: Type of metrics to delete
            older_than: Optional timestamp, delete metrics older than this

        Returns:
            Number of deleted data points
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        if metric_type not in self._storage:
            return 0

        if older_than is None:
            # Delete all metrics of this type
            count = len(self._storage[metric_type])
            self._storage[metric_type] = []
            self._timestamps[metric_type] = []
            return count
        
        # Find the cutoff index
        timestamps = self._timestamps[metric_type]
        cutoff_idx = bisect.bisect_left(timestamps, older_than)
        
        if cutoff_idx == 0:
            # No data points to delete
            return 0
        
        # Count deleted points
        deleted_count = cutoff_idx
        
        # Remove old data points
        self._storage[metric_type] = self._storage[metric_type][cutoff_idx:]
        self._timestamps[metric_type] = timestamps[cutoff_idx:]
        
        return deleted_count

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
        self.retention_policies[name] = {
            "duration": duration,
            "default": default
        }
        
        if default:
            # Set as default retention duration
            self.retention_duration = duration
            
            # Apply to all existing data
            for metric_type in self._storage.keys():
                await self._apply_retention_policy(metric_type)

    async def _apply_retention_policy(self, metric_type: str) -> None:
        """
        Apply the retention policy to a specific metric type.

        Args:
            metric_type: Type of metrics to apply the policy to
        """
        # Calculate the cutoff time based on retention duration
        cutoff_time = datetime.now() - self.retention_duration
        
        # Delete metrics older than the cutoff time
        await self.delete_metrics(metric_type, older_than=cutoff_time)

    def _aggregate_data(
        self, data_points: List[tuple], aggregation: str
    ) -> List[Dict[str, Any]]:
        """
        Aggregate data points based on the specified interval.

        Args:
            data_points: List of (timestamp, data) tuples
            aggregation: Aggregation interval (e.g., '1m', '1h', '1d')

        Returns:
            List of aggregated data points
        """
        if not data_points:
            return []
            
        # Parse aggregation interval
        interval = self._parse_aggregation_interval(aggregation)
        if not interval:
            # Invalid aggregation interval, return raw data
            return [{"timestamp": ts, **data} for ts, data in data_points]
        
        # Group data points by time bucket
        buckets = defaultdict(list)
        for ts, data in data_points:
            # Calculate bucket start time
            bucket_start = self._get_bucket_start(ts, interval)
            buckets[bucket_start].append(data)
        
        # Aggregate data in each bucket
        result = []
        for bucket_start, bucket_data in sorted(buckets.items()):
            # Calculate averages for numeric values
            aggregated = {}
            for key in bucket_data[0].keys():
                values = [d.get(key) for d in bucket_data if key in d]
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    # Numeric values - calculate average
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        aggregated[key] = sum(valid_values) / len(valid_values)
                    else:
                        aggregated[key] = None
                else:
                    # Non-numeric or mixed values - use the most recent
                    for d in reversed(bucket_data):
                        if key in d:
                            aggregated[key] = d[key]
                            break
            
            result.append({"timestamp": bucket_start, **aggregated})
        
        return result

    def _parse_aggregation_interval(self, aggregation: str) -> Optional[timedelta]:
        """
        Parse aggregation interval string into a timedelta.

        Args:
            aggregation: Aggregation interval (e.g., '1m', '1h', '1d')

        Returns:
            Timedelta representing the interval, or None if invalid
        """
        if not aggregation:
            return None
            
        try:
            value = int(aggregation[:-1])
            unit = aggregation[-1].lower()
            
            if unit == 's':
                return timedelta(seconds=value)
            elif unit == 'm':
                return timedelta(minutes=value)
            elif unit == 'h':
                return timedelta(hours=value)
            elif unit == 'd':
                return timedelta(days=value)
            else:
                return None
        except (ValueError, IndexError):
            return None

    def _get_bucket_start(self, timestamp: datetime, interval: timedelta) -> datetime:
        """
        Calculate the start time of a bucket for a given timestamp.

        Args:
            timestamp: The timestamp to bucket
            interval: The bucket interval

        Returns:
            Start time of the bucket
        """
        # Calculate seconds since epoch
        epoch = datetime(1970, 1, 1)
        seconds = (timestamp - epoch).total_seconds()
        
        # Calculate bucket start in seconds
        interval_seconds = interval.total_seconds()
        bucket_seconds = (seconds // interval_seconds) * interval_seconds
        
        # Convert back to datetime
        return epoch + timedelta(seconds=bucket_seconds)