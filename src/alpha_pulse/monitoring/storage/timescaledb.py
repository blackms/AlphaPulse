"""
TimescaleDB implementation of time series storage.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import json
import asyncpg
from asyncpg.exceptions import PostgresError

from .interfaces import TimeSeriesStorage


class TimescaleDBStorage(TimeSeriesStorage):
    """
    TimescaleDB implementation of time series storage.
    Uses asyncpg for async PostgreSQL access.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TimescaleDB storage.

        Args:
            config: Configuration parameters
                - host: Database host
                - port: Database port
                - user: Database user
                - password: Database password
                - database: Database name
                - schema: Schema name (default: public)
                - ssl: Whether to use SSL (default: True)
                - pool_min_size: Minimum connection pool size (default: 1)
                - pool_max_size: Maximum connection pool size (default: 10)
        """
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.schema = config.get("schema", "public")
        self.ssl = config.get("ssl", True)
        self.pool_min_size = config.get("pool_min_size", 1)
        self.pool_max_size = config.get("pool_max_size", 10)
        
        self.pool = None
        self.logger = logging.getLogger("timescaledb_storage")
        self._connected = False
        self._tables_created = False

    async def connect(self) -> None:
        """
        Establish connection to TimescaleDB.
        """
        if self._connected:
            return
            
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                ssl=self.ssl
            )
            
            # Ensure tables exist
            await self._ensure_tables()
            
            self._connected = True
            self.logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}")
        except Exception as e:
            if self.pool:
                await self.pool.close()
                self.pool = None
            self.logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close the connection to TimescaleDB.
        """
        if self.pool:
            await self.pool.close()
            self.pool = None
        self._connected = False

    async def store_metrics(
        self, metric_type: str, timestamp: datetime, data: Dict[str, Any]
    ) -> None:
        """
        Store metrics in TimescaleDB.

        Args:
            metric_type: Type of metrics
            timestamp: Timestamp for the metrics
            data: Dictionary of metric values
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Ensure table exists for this metric type
        table_name = self._get_table_name(metric_type)
        if not await self._table_exists(table_name):
            await self._create_metrics_table(table_name)
        
        # Extract field names and values
        fields = list(data.keys())
        values = [data[field] for field in fields]
        
        # Build SQL query
        placeholders = [f"${i+3}" for i in range(len(fields))]
        fields_sql = ", ".join([f'"{field}"' for field in fields])
        values_sql = ", ".join(placeholders)
        
        query = f"""
        INSERT INTO {self.schema}.{table_name} (
            time, 
            data,
            {fields_sql}
        ) VALUES (
            $1, 
            $2,
            {values_sql}
        )
        """
        
        # Execute query
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query, 
                    timestamp, 
                    json.dumps(data),
                    *values
                )
        except PostgresError as e:
            self.logger.error(f"Failed to store metrics: {e}")
            raise

    async def query_metrics(
        self,
        metric_type: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics from TimescaleDB.

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

        # Get table name
        table_name = self._get_table_name(metric_type)
        
        # Check if table exists
        if not await self._table_exists(table_name):
            return []
        
        # Build query
        if aggregation:
            # With aggregation
            interval = self._parse_aggregation_interval(aggregation)
            if not interval:
                self.logger.warning(f"Invalid aggregation interval: {aggregation}")
                # Fall back to non-aggregated query
                return await self._query_raw(table_name, start_time, end_time)
            
            return await self._query_aggregated(table_name, start_time, end_time, interval)
        else:
            # Without aggregation
            return await self._query_raw(table_name, start_time, end_time)

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

        # Get table name
        table_name = self._get_table_name(metric_type)
        
        # Check if table exists
        if not await self._table_exists(table_name):
            return []
        
        # Build query
        query = f"""
        SELECT time, data
        FROM {self.schema}.{table_name}
        ORDER BY time DESC
        LIMIT $1
        """
        
        # Execute query
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, limit)
                
                # Parse results
                result = []
                for row in rows:
                    data = json.loads(row["data"])
                    data["timestamp"] = row["time"]
                    result.append(data)
                
                return result
        except PostgresError as e:
            self.logger.error(f"Failed to get latest metrics: {e}")
            raise

    async def delete_metrics(
        self, metric_type: str, older_than: Optional[datetime] = None
    ) -> int:
        """
        Delete metrics from TimescaleDB.

        Args:
            metric_type: Type of metrics to delete
            older_than: Optional timestamp, delete metrics older than this

        Returns:
            Number of deleted data points
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Get table name
        table_name = self._get_table_name(metric_type)
        
        # Check if table exists
        if not await self._table_exists(table_name):
            return 0
        
        # Build query
        if older_than:
            query = f"""
            DELETE FROM {self.schema}.{table_name}
            WHERE time < $1
            """
            params = [older_than]
        else:
            query = f"""
            DELETE FROM {self.schema}.{table_name}
            """
            params = []
        
        # Execute query
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(query, *params)
                
                # Parse result to get count
                # Result format is "DELETE count"
                count_str = result.split(" ")[1]
                return int(count_str)
        except PostgresError as e:
            self.logger.error(f"Failed to delete metrics: {e}")
            raise

    async def create_retention_policy(
        self, name: str, duration: timedelta, default: bool = False
    ) -> None:
        """
        Create a retention policy for automatic data cleanup.
        
        In TimescaleDB, this is implemented using continuous aggregates and retention policies.

        Args:
            name: Name of the retention policy
            duration: How long to keep the data
            default: Whether this is the default policy
        """
        if not self._connected:
            raise RuntimeError("Storage is not connected")

        # Convert duration to interval string
        interval = self._format_interval(duration)
        
        try:
            async with self.pool.acquire() as conn:
                # Check if TimescaleDB extension is installed
                is_timescaledb = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
                )
                
                if not is_timescaledb:
                    self.logger.error("TimescaleDB extension is not installed")
                    raise RuntimeError("TimescaleDB extension is not installed")
                
                # Get all metric tables
                tables = await self._get_metric_tables()
                
                for table in tables:
                    # Add retention policy to each table
                    policy_name = f"{table}_{name}_policy"
                    
                    # Check if policy exists
                    policy_exists = await conn.fetchval(
                        """
                        SELECT EXISTS(
                            SELECT 1 
                            FROM timescaledb_information.jobs j
                            JOIN timescaledb_information.job_stats s ON j.job_id = s.job_id
                            WHERE j.proc_name = 'policy_retention' AND s.hypertable_name = $1
                        )
                        """,
                        table
                    )
                    
                    if policy_exists:
                        # Drop existing policy
                        await conn.execute(
                            f"SELECT remove_retention_policy('{self.schema}.{table}')"
                        )
                    
                    # Create new policy
                    await conn.execute(
                        f"SELECT add_retention_policy('{self.schema}.{table}', INTERVAL '{interval}')"
                    )
                    
                    self.logger.info(f"Created retention policy for {table} with duration {interval}")
        except PostgresError as e:
            self.logger.error(f"Failed to create retention policy: {e}")
            raise

    async def _ensure_tables(self) -> None:
        """
        Ensure that required tables and extensions exist.
        """
        if self._tables_created:
            return
            
        try:
            async with self.pool.acquire() as conn:
                # Check if TimescaleDB extension is installed
                is_timescaledb = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
                )
                
                if not is_timescaledb:
                    # Create TimescaleDB extension
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                
                # Create schema if it doesn't exist
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                
                # Create metrics registry table
                await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.metrics_registry (
                    metric_type TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    fields JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """)
                
                self._tables_created = True
        except PostgresError as e:
            self.logger.error(f"Failed to ensure tables: {e}")
            raise

    async def _table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Table name to check

        Returns:
            True if the table exists, False otherwise
        """
        try:
            async with self.pool.acquire() as conn:
                exists = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 
                        FROM information_schema.tables 
                        WHERE table_schema = $1 AND table_name = $2
                    )
                    """,
                    self.schema, table_name
                )
                return exists
        except PostgresError as e:
            self.logger.error(f"Failed to check if table exists: {e}")
            raise

    async def _create_metrics_table(self, table_name: str) -> None:
        """
        Create a metrics table for a specific metric type.

        Args:
            table_name: Table name to create
        """
        try:
            async with self.pool.acquire() as conn:
                # Create table
                await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.{table_name} (
                    time TIMESTAMPTZ NOT NULL,
                    data JSONB NOT NULL
                )
                """)
                
                # Convert to hypertable
                await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.schema}.{table_name}', 
                    'time', 
                    if_not_exists => TRUE
                )
                """)
                
                # Register metric type
                metric_type = table_name.replace("metrics_", "")
                await conn.execute(
                    f"""
                    INSERT INTO {self.schema}.metrics_registry (metric_type, fields)
                    VALUES ($1, $2)
                    ON CONFLICT (metric_type) DO UPDATE
                    SET updated_at = NOW()
                    """,
                    metric_type, json.dumps({})
                )
                
                self.logger.info(f"Created metrics table: {table_name}")
        except PostgresError as e:
            self.logger.error(f"Failed to create metrics table: {e}")
            raise

    async def _get_metric_tables(self) -> List[str]:
        """
        Get all metric tables.

        Returns:
            List of table names
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = $1 AND table_name LIKE 'metrics_%'
                    """,
                    self.schema
                )
                return [row["table_name"] for row in rows]
        except PostgresError as e:
            self.logger.error(f"Failed to get metric tables: {e}")
            raise

    async def _query_raw(
        self, table_name: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Query raw metrics without aggregation.

        Args:
            table_name: Table name to query
            start_time: Start time for the query range
            end_time: End time for the query range

        Returns:
            List of metric data points
        """
        query = f"""
        SELECT time, data
        FROM {self.schema}.{table_name}
        WHERE time >= $1 AND time <= $2
        ORDER BY time
        """
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, start_time, end_time)
                
                # Parse results
                result = []
                for row in rows:
                    data = json.loads(row["data"])
                    data["timestamp"] = row["time"]
                    result.append(data)
                
                return result
        except PostgresError as e:
            self.logger.error(f"Failed to query raw metrics: {e}")
            raise

    async def _query_aggregated(
        self, table_name: str, start_time: datetime, end_time: datetime, interval: str
    ) -> List[Dict[str, Any]]:
        """
        Query aggregated metrics.

        Args:
            table_name: Table name to query
            start_time: Start time for the query range
            end_time: End time for the query range
            interval: Aggregation interval

        Returns:
            List of aggregated metric data points
        """
        # Get column names from a sample row
        columns = await self._get_table_columns(table_name)
        if not columns:
            return []
        
        # Build aggregation expressions
        agg_exprs = []
        for col in columns:
            if col not in ("time", "data"):
                agg_exprs.append(f'AVG("{col}") as "{col}"')
        
        # If no columns to aggregate, use data from JSONB
        if not agg_exprs:
            # Use time_bucket with jsonb_to_record
            query = f"""
            WITH expanded AS (
                SELECT 
                    time,
                    data,
                    jsonb_each_text(data) AS kv
            )
            SELECT 
                time_bucket($3, time) AS bucket_time,
                jsonb_object_agg(kv.key, AVG((kv.value)::float)) AS data
            FROM expanded
            WHERE time >= $1 AND time <= $2
                AND (kv.value ~ '^-?[0-9]+(\\.[0-9]+)?$')  -- Only aggregate numeric values
            GROUP BY bucket_time
            ORDER BY bucket_time
            """
        else:
            # Use time_bucket with explicit columns
            agg_cols = ", ".join(agg_exprs)
            query = f"""
            SELECT 
                time_bucket($3, time) AS bucket_time,
                {agg_cols}
            FROM {self.schema}.{table_name}
            WHERE time >= $1 AND time <= $2
            GROUP BY bucket_time
            ORDER BY bucket_time
            """
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, start_time, end_time, interval)
                
                # Parse results
                result = []
                for row in rows:
                    if "data" in row and row["data"]:
                        # Data from jsonb_object_agg
                        data = dict(row["data"])
                    else:
                        # Data from explicit columns
                        data = {k: v for k, v in dict(row).items() if k != "bucket_time"}
                    
                    data["timestamp"] = row["bucket_time"]
                    result.append(data)
                
                return result
        except PostgresError as e:
            self.logger.error(f"Failed to query aggregated metrics: {e}")
            raise

    async def _get_table_columns(self, table_name: str) -> List[str]:
        """
        Get column names for a table.

        Args:
            table_name: Table name

        Returns:
            List of column names
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = $1 AND table_name = $2
                    """,
                    self.schema, table_name
                )
                return [row["column_name"] for row in rows]
        except PostgresError as e:
            self.logger.error(f"Failed to get table columns: {e}")
            raise

    def _get_table_name(self, metric_type: str) -> str:
        """
        Get table name for a metric type.

        Args:
            metric_type: Metric type

        Returns:
            Table name
        """
        # Sanitize metric type for use in table name
        sanitized = metric_type.lower().replace("-", "_")
        return f"metrics_{sanitized}"

    def _parse_aggregation_interval(self, aggregation: str) -> Optional[str]:
        """
        Parse aggregation interval string into a PostgreSQL interval.

        Args:
            aggregation: Aggregation interval (e.g., '1m', '1h', '1d')

        Returns:
            PostgreSQL interval string, or None if invalid
        """
        if not aggregation:
            return None
            
        try:
            value = int(aggregation[:-1])
            unit = aggregation[-1].lower()
            
            if unit == 's':
                return f"{value} seconds"
            elif unit == 'm':
                return f"{value} minutes"
            elif unit == 'h':
                return f"{value} hours"
            elif unit == 'd':
                return f"{value} days"
            else:
                return None
        except (ValueError, IndexError):
            return None

    def _format_interval(self, duration: timedelta) -> str:
        """
        Format a timedelta as a PostgreSQL interval string.

        Args:
            duration: Duration to format

        Returns:
            PostgreSQL interval string
        """
        total_seconds = int(duration.total_seconds())
        
        # Convert to appropriate unit
        if total_seconds % 86400 == 0:
            # Days
            return f"{total_seconds // 86400} days"
        elif total_seconds % 3600 == 0:
            # Hours
            return f"{total_seconds // 3600} hours"
        elif total_seconds % 60 == 0:
            # Minutes
            return f"{total_seconds // 60} minutes"
        else:
            # Seconds
            return f"{total_seconds} seconds"