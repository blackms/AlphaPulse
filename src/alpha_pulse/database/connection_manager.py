"""Database connection manager with health checks and validation."""

import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..config.database_config import DatabaseConfig, DatabaseNode
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class ConnectionState(Enum):
    """Connection health states."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ConnectionHealth:
    """Tracks connection health metrics."""
    
    def __init__(self, node_id: str):
        """Initialize health tracker."""
        self.node_id = node_id
        self.state = ConnectionState.UNKNOWN
        self.last_check = None
        self.consecutive_failures = 0
        self.response_times: List[float] = []
        self.error_count = 0
        self.last_error = None
        
    def record_success(self, response_time: float):
        """Record successful health check."""
        self.state = ConnectionState.HEALTHY
        self.last_check = datetime.utcnow()
        self.consecutive_failures = 0
        self.response_times.append(response_time)
        
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_failure(self, error: Exception):
        """Record failed health check."""
        self.last_check = datetime.utcnow()
        self.consecutive_failures += 1
        self.error_count += 1
        self.last_error = str(error)
        
        # Update state based on consecutive failures
        if self.consecutive_failures >= 3:
            self.state = ConnectionState.UNHEALTHY
        else:
            self.state = ConnectionState.DEGRADED
    
    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self.state == ConnectionState.HEALTHY


class ConnectionValidator:
    """Validates database connections."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize connection validator."""
        self.config = config
        self.metrics = metrics_collector
        self.health_trackers: Dict[str, ConnectionHealth] = {}
        self._validation_queries = {
            "basic": "SELECT 1",
            "performance": "SELECT 1, NOW(), version()",
            "tables": """
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """,
            "connections": """
                SELECT COUNT(*) 
                FROM pg_stat_activity 
                WHERE state = 'active'
            """
        }
    
    async def validate_connection(
        self,
        session: AsyncSession,
        validation_level: str = "basic"
    ) -> bool:
        """Validate a database connection."""
        try:
            query = self._validation_queries.get(validation_level, "SELECT 1")
            start_time = time.time()
            
            # Execute validation query
            result = await session.execute(text(query))
            await result.fetchone()
            
            response_time = time.time() - start_time
            
            if self.metrics:
                self.metrics.histogram(
                    "db.validation.response_time",
                    response_time,
                    {"level": validation_level}
                )
            
            return response_time < 1.0  # Consider valid if under 1 second
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            if self.metrics:
                self.metrics.increment("db.validation.failures")
            return False
    
    async def health_check(
        self,
        node: DatabaseNode,
        session: Optional[AsyncSession] = None
    ) -> ConnectionHealth:
        """Perform health check on database node."""
        node_id = f"{node.host}:{node.port}"
        
        # Get or create health tracker
        if node_id not in self.health_trackers:
            self.health_trackers[node_id] = ConnectionHealth(node_id)
        
        health = self.health_trackers[node_id]
        
        try:
            start_time = time.time()
            
            if session:
                # Use provided session
                is_valid = await self.validate_connection(session, "performance")
            else:
                # Create temporary connection for health check
                conn = await asyncpg.connect(
                    host=node.host,
                    port=node.port,
                    database=node.database,
                    user=node.username,
                    password=node.password,
                    timeout=node.health_check_timeout
                )
                
                try:
                    # Run health check query
                    await conn.fetchval("SELECT 1")
                    is_valid = True
                finally:
                    await conn.close()
            
            response_time = time.time() - start_time
            
            if is_valid:
                health.record_success(response_time)
                logger.debug(f"Health check passed for {node_id} in {response_time:.3f}s")
            else:
                health.record_failure(Exception("Validation failed"))
                logger.warning(f"Health check failed for {node_id}")
            
            # Record metrics
            if self.metrics:
                self.metrics.gauge(
                    "db.health.status",
                    1 if health.is_healthy else 0,
                    {"node": node_id}
                )
                self.metrics.histogram(
                    "db.health.response_time",
                    response_time,
                    {"node": node_id}
                )
            
        except Exception as e:
            health.record_failure(e)
            logger.error(f"Health check error for {node_id}: {e}")
            
            if self.metrics:
                self.metrics.increment(
                    "db.health.errors",
                    {"node": node_id, "error": type(e).__name__}
                )
        
        return health
    
    async def check_replication_lag(
        self,
        master_session: AsyncSession,
        replica_session: AsyncSession
    ) -> Optional[float]:
        """Check replication lag between master and replica."""
        try:
            # Get current WAL position on master
            master_result = await master_session.execute(
                text("SELECT pg_current_wal_lsn() as wal_lsn")
            )
            master_lsn = (await master_result.fetchone()).wal_lsn
            
            # Get last received WAL position on replica
            replica_result = await replica_session.execute(
                text("SELECT pg_last_wal_receive_lsn() as wal_lsn")
            )
            replica_lsn = (await replica_result.fetchone()).wal_lsn
            
            # Calculate lag in bytes
            lag_result = await master_session.execute(
                text(
                    "SELECT pg_wal_lsn_diff(:master_lsn, :replica_lsn) as lag_bytes"
                ),
                {"master_lsn": master_lsn, "replica_lsn": replica_lsn}
            )
            lag_bytes = (await lag_result.fetchone()).lag_bytes
            
            # Convert to approximate seconds (rough estimate)
            lag_seconds = lag_bytes / (16 * 1024 * 1024)  # Assume 16MB/s
            
            if self.metrics:
                self.metrics.gauge("db.replication.lag_bytes", lag_bytes)
                self.metrics.gauge("db.replication.lag_seconds", lag_seconds)
            
            return lag_seconds
            
        except Exception as e:
            logger.error(f"Failed to check replication lag: {e}")
            return None
    
    def get_health_summary(self) -> Dict[str, Dict]:
        """Get health summary for all nodes."""
        summary = {}
        
        for node_id, health in self.health_trackers.items():
            summary[node_id] = {
                "state": health.state.value,
                "is_healthy": health.is_healthy,
                "last_check": health.last_check.isoformat() if health.last_check else None,
                "consecutive_failures": health.consecutive_failures,
                "error_count": health.error_count,
                "average_response_time": health.average_response_time,
                "last_error": health.last_error
            }
        
        return summary


class ConnectionMonitor:
    """Monitors database connections and performance."""
    
    def __init__(
        self,
        validator: ConnectionValidator,
        check_interval: int = 30
    ):
        """Initialize connection monitor."""
        self.validator = validator
        self.check_interval = check_interval
        self._monitoring_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable] = []
    
    def add_health_callback(self, callback: Callable):
        """Add callback for health state changes."""
        self._callbacks.append(callback)
    
    async def start_monitoring(self, connection_pool):
        """Start monitoring connections."""
        self._monitoring_task = asyncio.create_task(
            self._monitor_loop(connection_pool)
        )
        logger.info("Started connection health monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring connections."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped connection health monitoring")
    
    async def _monitor_loop(self, connection_pool):
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Check master health
                config = connection_pool.config
                master_health = await self.validator.health_check(config.master_node)
                
                # Check replica health
                for i, replica in enumerate(config.read_replicas):
                    replica_health = await self.validator.health_check(replica)
                    
                    # Notify callbacks if state changed
                    old_state = self.validator.health_trackers.get(
                        f"{replica.host}:{replica.port}"
                    )
                    if old_state and old_state.state != replica_health.state:
                        for callback in self._callbacks:
                            await callback(replica, replica_health)
                
                # Check replication lag if replicas exist
                if config.read_replicas and config.enable_read_write_split:
                    async with connection_pool.get_master_session() as master:
                        for i in range(len(config.read_replicas)):
                            async with connection_pool.get_replica_session() as replica:
                                lag = await self.validator.check_replication_lag(
                                    master, replica
                                )
                                if lag and lag > 10:  # More than 10 seconds
                                    logger.warning(
                                        f"High replication lag detected: {lag:.2f}s"
                                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)