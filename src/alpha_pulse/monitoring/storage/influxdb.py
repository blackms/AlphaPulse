"""
InfluxDB implementation of time series storage.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
import json
from urllib.parse import urlencode

import aiohttp
from aiohttp.client_exceptions import ClientError

from .interfaces import TimeSeriesStorage


class InfluxDBStorage(TimeSeriesStorage):
    """
    InfluxDB implementation of time series storage.
    Uses the InfluxDB HTTP API for data operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize InfluxDB storage.

        Args:
            config: Configuration parameters
                - url: InfluxDB server URL
                - token: API token for authentication
                - org: Organization name
                - bucket: Default bucket name
                - batch_size: Number of points to write in a batch (default: 5000)
                - timeout: Request timeout in seconds (default: 10)
        """
        self.url = config["url"]
        self.token = config["token"]
        self.org = config["org"]
        self.bucket = config["bucket"]
        self.batch_size = config.get("batch_size", 5000)
        self.timeout = config.get("timeout", 10)
        
        self.session = None
        self.logger = logging.getLogger("influxdb_storage")
        self._connected = False

    async def connect(self) -> None:
        """
        Establish connection to InfluxDB.
        """
        if self._connected:
            return
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # Test connection
        try:
            await self._ping()
            self._connected = True
            self.logger.info(f"Connected to InfluxDB at {self.url}")
        except Exception as e:
            await self.session.close()
            self.session = None
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close the connection to InfluxDB.
        """
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False

    async def store_metrics(
        self, metric_type: str, timestamp: datetime, data: Dict[str, Any]
    ) -> None:
        """
        Store metrics in InfluxDB.

        Args:
            metric_type: Type of metrics
            timestamp: Timestamp for the metrics
            data: Dictionary of metric values
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Convert data to InfluxDB line protocol
        line = self._to_line_protocol(metric_type, timestamp, data)
        
        # Write to InfluxDB
        await self._write_points([line])

    async def query_metrics(
        self,
        metric_type: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics from InfluxDB.

        Args:
            metric_type: Type of metrics to query
            start_time: Start time for the query range
            end_time: End time for the query range
            aggregation: Optional aggregation interval (e.g., '1m', '1h', '1d')

        Returns:
            List of metric data points
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Build Flux query
        query = self._build_query(metric_type, start_time, end_time, aggregation)
        
        # Execute query
        result = await self._execute_query(query)
        
        # Parse result
        return self._parse_query_result(result)

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
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Build Flux query for latest points
        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: -30d)
            |> filter(fn: (r) => r._measurement == "{metric_type}")
            |> group()
            |> sort(columns: ["_time"], desc: true)
            |> limit(n: {limit})
        """
        
        # Execute query
        result = await self._execute_query(query)
        
        # Parse result
        return self._parse_query_result(result)

    async def delete_metrics(
        self, metric_type: str, older_than: Optional[datetime] = None
    ) -> int:
        """
        Delete metrics from InfluxDB.

        Args:
            metric_type: Type of metrics to delete
            older_than: Optional timestamp, delete metrics older than this

        Returns:
            Number of deleted data points (always 0 for InfluxDB as it doesn't return count)
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Build predicate for delete
        predicate = f'_measurement="{metric_type}"'
        
        # Add time predicate if specified
        start = ""
        stop = ""
        
        if older_than:
            # Format timestamp for InfluxDB
            stop = older_than.strftime("%Y-%m-%dT%H:%M:%SZ")
            start = "1970-01-01T00:00:00Z"  # Beginning of time
        
        # Execute delete
        await self._delete_points(predicate, start, stop)
        
        # InfluxDB doesn't return count of deleted points
        return 0

    async def create_retention_policy(
        self, name: str, duration: timedelta, default: bool = False
    ) -> None:
        """
        Create a retention policy for automatic data cleanup.
        
        Note: In InfluxDB 2.x, retention policies are defined at the bucket level.
        This method creates a new bucket with the specified retention period.

        Args:
            name: Name of the retention policy (bucket)
            duration: How long to keep the data
            default: Whether this is the default policy (ignored in InfluxDB 2.x)
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Convert duration to string format expected by InfluxDB
        duration_str = self._format_duration(duration)
        
        # Create bucket with retention policy
        url = f"{self.url}/api/v2/buckets"
        
        payload = {
            "name": name,
            "orgID": await self._get_org_id(),
            "retentionRules": [
                {
                    "type": "expire",
                    "everySeconds": int(duration.total_seconds())
                }
            ]
        }
        
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    self.logger.error(f"Failed to create bucket: {error_text}")
                    raise RuntimeError(f"Failed to create bucket: {response.status} {error_text}")
                
                self.logger.info(f"Created bucket {name} with retention {duration_str}")
        except ClientError as e:
            self.logger.error(f"Network error creating bucket: {e}")
            raise

    async def _ping(self) -> None:
        """
        Ping the InfluxDB server to check connectivity.
        """
        url = f"{self.url}/ping"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 204:
                    text = await response.text()
                    raise RuntimeError(f"Failed to ping InfluxDB: {response.status} {text}")
        except ClientError as e:
            raise RuntimeError(f"Network error pinging InfluxDB: {e}")

    async def _write_points(self, lines: List[str]) -> None:
        """
        Write data points to InfluxDB.

        Args:
            lines: List of line protocol strings
        """
        if not lines:
            return
            
        url = f"{self.url}/api/v2/write"
        params = {
            "org": self.org,
            "bucket": self.bucket,
            "precision": "ns"
        }
        
        query_string = urlencode(params)
        full_url = f"{url}?{query_string}"
        
        # Join lines with newlines
        data = "\n".join(lines)
        
        headers = self._get_headers()
        headers["Content-Type"] = "text/plain; charset=utf-8"
        
        try:
            async with self.session.post(full_url, data=data, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    self.logger.error(f"Failed to write points: {error_text}")
                    raise RuntimeError(f"Failed to write points: {response.status} {error_text}")
        except ClientError as e:
            self.logger.error(f"Network error writing points: {e}")
            raise

    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a Flux query against InfluxDB.

        Args:
            query: Flux query string

        Returns:
            Query result as a dictionary
        """
        url = f"{self.url}/api/v2/query"
        params = {"org": self.org}
        
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        
        payload = {
            "query": query,
            "type": "flux"
        }
        
        try:
            async with self.session.post(
                url, json=payload, params=params, headers=headers
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    self.logger.error(f"Query failed: {error_text}")
                    raise RuntimeError(f"Query failed: {response.status} {error_text}")
                
                return await response.json()
        except ClientError as e:
            self.logger.error(f"Network error executing query: {e}")
            raise

    async def _delete_points(
        self, predicate: str, start: str = "", stop: str = ""
    ) -> None:
        """
        Delete data points from InfluxDB.

        Args:
            predicate: Delete predicate (e.g., '_measurement="cpu"')
            start: Start time in RFC3339 format
            stop: Stop time in RFC3339 format
        """
        url = f"{self.url}/api/v2/delete"
        
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        
        payload = {
            "orgID": await self._get_org_id(),
            "bucketID": await self._get_bucket_id(),
            "predicate": predicate
        }
        
        if start:
            payload["start"] = start
        if stop:
            payload["stop"] = stop
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    self.logger.error(f"Delete failed: {error_text}")
                    raise RuntimeError(f"Delete failed: {response.status} {error_text}")
        except ClientError as e:
            self.logger.error(f"Network error deleting points: {e}")
            raise

    async def _get_org_id(self) -> str:
        """
        Get the organization ID from the organization name.

        Returns:
            Organization ID
        """
        url = f"{self.url}/api/v2/orgs"
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to get orgs: {response.status} {error_text}")
                
                data = await response.json()
                for org in data.get("orgs", []):
                    if org.get("name") == self.org:
                        return org.get("id")
                
                raise ValueError(f"Organization not found: {self.org}")
        except ClientError as e:
            raise RuntimeError(f"Network error getting org ID: {e}")

    async def _get_bucket_id(self) -> str:
        """
        Get the bucket ID from the bucket name.

        Returns:
            Bucket ID
        """
        url = f"{self.url}/api/v2/buckets"
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to get buckets: {response.status} {error_text}")
                
                data = await response.json()
                for bucket in data.get("buckets", []):
                    if bucket.get("name") == self.bucket:
                        return bucket.get("id")
                
                raise ValueError(f"Bucket not found: {self.bucket}")
        except ClientError as e:
            raise RuntimeError(f"Network error getting bucket ID: {e}")

    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for InfluxDB API requests.

        Returns:
            Dictionary of headers
        """
        return {
            "Authorization": f"Token {self.token}"
        }

    def _to_line_protocol(
        self, measurement: str, timestamp: datetime, fields: Dict[str, Any], tags: Dict[str, str] = None
    ) -> str:
        """
        Convert data to InfluxDB line protocol.

        Args:
            measurement: Measurement name
            timestamp: Timestamp
            fields: Field values
            tags: Optional tag values

        Returns:
            Line protocol string
        """
        # Escape special characters in measurement
        escaped_measurement = measurement.replace(",", "\\,").replace(" ", "\\ ")
        
        # Build tags string
        tags_str = ""
        if tags:
            tags_list = []
            for k, v in tags.items():
                # Escape special characters in tag keys and values
                k_esc = k.replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")
                v_esc = str(v).replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")
                tags_list.append(f"{k_esc}={v_esc}")
            
            if tags_list:
                tags_str = "," + ",".join(tags_list)
        
        # Build fields string
        fields_list = []
        for k, v in fields.items():
            # Escape special characters in field keys
            k_esc = k.replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")
            
            # Format field value based on type
            if v is None:
                continue  # Skip null values
            elif isinstance(v, bool):
                v_fmt = str(v).lower()
            elif isinstance(v, (int, float)):
                if isinstance(v, int):
                    v_fmt = f"{v}i"
                else:
                    v_fmt = str(v)
            else:
                # String value - escape and quote
                v_str = str(v).replace('"', '\\"')
                v_fmt = f'"{v_str}"'
            
            fields_list.append(f"{k_esc}={v_fmt}")
        
        if not fields_list:
            raise ValueError("No valid fields to write")
        
        fields_str = ",".join(fields_list)
        
        # Convert timestamp to nanoseconds
        timestamp_ns = int(timestamp.timestamp() * 1e9)
        
        # Combine all parts
        return f"{escaped_measurement}{tags_str} {fields_str} {timestamp_ns}"

    def _build_query(
        self,
        metric_type: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: Optional[str] = None,
    ) -> str:
        """
        Build a Flux query for retrieving metrics.

        Args:
            metric_type: Type of metrics to query
            start_time: Start time for the query range
            end_time: End time for the query range
            aggregation: Optional aggregation interval (e.g., '1m', '1h', '1d')

        Returns:
            Flux query string
        """
        # Format timestamps for Flux
        start_rfc = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        stop_rfc = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Base query
        query = f"""
        from(bucket: "{self.bucket}")
            |> range(start: {start_rfc}, stop: {stop_rfc})
            |> filter(fn: (r) => r._measurement == "{metric_type}")
        """
        
        # Add aggregation if specified
        if aggregation:
            # Parse aggregation interval
            window = self._parse_aggregation_window(aggregation)
            if window:
                query += f"""
            |> aggregateWindow(
                every: {window},
                fn: mean,
                createEmpty: false
            )
                """
        
        # Add final sorting
        query += """
            |> sort(columns: ["_time"])
        """
        
        return query

    def _parse_aggregation_window(self, aggregation: str) -> Optional[str]:
        """
        Parse aggregation interval string into a Flux window duration.

        Args:
            aggregation: Aggregation interval (e.g., '1m', '1h', '1d')

        Returns:
            Flux window duration string, or None if invalid
        """
        if not aggregation:
            return None
            
        try:
            value = int(aggregation[:-1])
            unit = aggregation[-1].lower()
            
            if unit == 's':
                return f"{value}s"
            elif unit == 'm':
                return f"{value}m"
            elif unit == 'h':
                return f"{value}h"
            elif unit == 'd':
                return f"{value}d"
            else:
                return None
        except (ValueError, IndexError):
            return None

    def _parse_query_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse InfluxDB query result into a list of data points.

        Args:
            result: Query result from InfluxDB

        Returns:
            List of data points
        """
        if not result or "tables" not in result:
            return []
            
        # Extract table schema and data
        tables = result.get("tables", [])
        if not tables:
            return []
            
        # Process all tables
        data_points = []
        
        for table in tables:
            # Get column names from table schema
            columns = []
            for col in table.get("columns", []):
                columns.append(col.get("name"))
                
            # Process each row
            for row in table.get("data", {}).get("values", []):
                if len(row) != len(columns):
                    continue
                    
                # Create data point
                point = {}
                for i, value in enumerate(row):
                    col_name = columns[i]
                    
                    # Handle special columns
                    if col_name == "_time":
                        point["timestamp"] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    elif col_name not in ("_measurement", "_field", "_value", "result", "table"):
                        point[col_name] = value
                        
                # Add the value if present
                value_idx = columns.index("_value") if "_value" in columns else -1
                field_idx = columns.index("_field") if "_field" in columns else -1
                
                if value_idx >= 0 and field_idx >= 0:
                    field_name = row[field_idx]
                    field_value = row[value_idx]
                    point[field_name] = field_value
                
                data_points.append(point)
        
        return data_points

    def _format_duration(self, duration: timedelta) -> str:
        """
        Format a timedelta as a duration string for InfluxDB.

        Args:
            duration: Duration to format

        Returns:
            Duration string (e.g., "7d", "24h", "30m")
        """
        total_seconds = int(duration.total_seconds())
        
        # Convert to appropriate unit
        if total_seconds % 86400 == 0:
            # Days
            return f"{total_seconds // 86400}d"
        elif total_seconds % 3600 == 0:
            # Hours
            return f"{total_seconds // 3600}h"
        elif total_seconds % 60 == 0:
            # Minutes
            return f"{total_seconds // 60}m"
        else:
            # Seconds
            return f"{total_seconds}s"