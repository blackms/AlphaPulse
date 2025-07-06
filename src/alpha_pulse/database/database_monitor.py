"""Database performance monitoring."""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector
from ..monitoring.alert_manager import AlertManager, AlertSeverity

logger = get_logger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for query performance."""
    
    query_pattern: str
    execution_count: int = 0
    total_time: float = 0
    min_time: float = float('inf')
    max_time: float = 0
    error_count: int = 0
    rows_returned: int = 0
    
    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.execution_count if self.execution_count > 0 else 0


@dataclass
class ConnectionMetrics:
    """Metrics for database connections."""
    
    timestamp: datetime
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_connections: int
    max_connections: int
    connection_utilization: float


@dataclass
class TableMetrics:
    """Metrics for database tables."""
    
    table_name: str
    row_count: int
    dead_tuples: int
    table_size: int
    index_size: int
    last_vacuum: Optional[datetime]
    last_analyze: Optional[datetime]
    
    @property
    def total_size(self) -> int:
        """Total size including indexes."""
        return self.table_size + self.index_size
    
    @property
    def bloat_ratio(self) -> float:
        """Ratio of dead tuples to total."""
        total = self.row_count + self.dead_tuples
        return self.dead_tuples / total if total > 0 else 0


class DatabaseMonitor:
    """Monitors database performance and health."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: Optional[AlertManager] = None
    ):
        """Initialize database monitor."""
        self.metrics = metrics_collector
        self.alerts = alert_manager
        
        # Metrics storage
        self._query_metrics: Dict[str, QueryMetrics] = {}
        self._connection_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self._table_metrics: Dict[str, TableMetrics] = {}
        
        # Performance tracking
        self._slow_query_threshold = 1.0  # seconds
        self._connection_threshold = 0.8  # 80% utilization
        self._bloat_threshold = 0.2  # 20% dead tuples
        
        # Monitoring tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._collection_interval = 60  # seconds
        
    async def start_monitoring(self, session_factory):
        """Start monitoring tasks."""
        if self._monitor_task:
            logger.warning("Database monitoring already running")
            return
        
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(session_factory)
        )
        logger.info("Started database monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring tasks."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Stopped database monitoring")
    
    async def _monitor_loop(self, session_factory):
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._collection_interval)
                
                async with session_factory() as session:
                    # Collect various metrics
                    await self._collect_connection_metrics(session)
                    await self._collect_table_metrics(session)
                    await self._collect_performance_metrics(session)
                    await self._check_replication_status(session)
                    
                    # Check for issues
                    await self._check_thresholds()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self._collection_interval)
    
    async def _collect_connection_metrics(self, session: AsyncSession):
        """Collect connection pool metrics."""
        try:
            # Get connection statistics
            result = await session.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE state = 'active') as active,
                        COUNT(*) FILTER (WHERE state = 'idle') as idle,
                        COUNT(*) FILTER (WHERE wait_event_type = 'Client') as waiting,
                        current_setting('max_connections')::int as max_connections
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """)
            )
            
            row = await result.fetchone()
            
            metrics = ConnectionMetrics(
                timestamp=datetime.utcnow(),
                total_connections=row.total,
                active_connections=row.active,
                idle_connections=row.idle,
                waiting_connections=row.waiting,
                max_connections=row.max_connections,
                connection_utilization=row.total / row.max_connections
            )
            
            self._connection_history.append(metrics)
            
            # Export to Prometheus
            self.metrics.gauge("db.connections.total", metrics.total_connections)
            self.metrics.gauge("db.connections.active", metrics.active_connections)
            self.metrics.gauge("db.connections.idle", metrics.idle_connections)
            self.metrics.gauge("db.connections.waiting", metrics.waiting_connections)
            self.metrics.gauge("db.connections.utilization", metrics.connection_utilization)
            
            # Check for high utilization
            if metrics.connection_utilization > self._connection_threshold:
                await self._alert_high_connection_usage(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect connection metrics: {e}")
    
    async def _collect_table_metrics(self, session: AsyncSession):
        """Collect table-level metrics."""
        try:
            # Get table statistics
            result = await session.execute(
                text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples,
                        pg_table_size(schemaname||'.'||tablename) as table_size,
                        pg_indexes_size(schemaname||'.'||tablename) as index_size,
                        last_vacuum,
                        last_analyze
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_table_size(schemaname||'.'||tablename) DESC
                    LIMIT 50
                """)
            )
            
            for row in result:
                table_metrics = TableMetrics(
                    table_name=f"{row.schemaname}.{row.tablename}",
                    row_count=row.live_tuples,
                    dead_tuples=row.dead_tuples,
                    table_size=row.table_size,
                    index_size=row.index_size,
                    last_vacuum=row.last_vacuum,
                    last_analyze=row.last_analyze
                )
                
                self._table_metrics[table_metrics.table_name] = table_metrics
                
                # Export metrics
                labels = {"table": table_metrics.table_name}
                self.metrics.gauge("db.table.rows", table_metrics.row_count, labels)
                self.metrics.gauge("db.table.dead_tuples", table_metrics.dead_tuples, labels)
                self.metrics.gauge("db.table.size_bytes", table_metrics.table_size, labels)
                self.metrics.gauge("db.table.bloat_ratio", table_metrics.bloat_ratio, labels)
                
                # Check for bloat
                if table_metrics.bloat_ratio > self._bloat_threshold:
                    await self._alert_table_bloat(table_metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect table metrics: {e}")
    
    async def _collect_performance_metrics(self, session: AsyncSession):
        """Collect database performance metrics."""
        try:
            # Database-wide statistics
            result = await session.execute(
                text("""
                    SELECT 
                        xact_commit,
                        xact_rollback,
                        blks_read,
                        blks_hit,
                        tup_returned,
                        tup_fetched,
                        tup_inserted,
                        tup_updated,
                        tup_deleted,
                        conflicts,
                        deadlocks,
                        checksum_failures,
                        temp_files,
                        temp_bytes
                    FROM pg_stat_database
                    WHERE datname = current_database()
                """)
            )
            
            row = await result.fetchone()
            
            # Calculate rates (these would need previous values for real rates)
            cache_hit_ratio = row.blks_hit / (row.blks_hit + row.blks_read) if (row.blks_hit + row.blks_read) > 0 else 0
            
            # Export metrics
            self.metrics.gauge("db.transactions.committed", row.xact_commit)
            self.metrics.gauge("db.transactions.rolled_back", row.xact_rollback)
            self.metrics.gauge("db.cache.hit_ratio", cache_hit_ratio)
            self.metrics.gauge("db.rows.returned", row.tup_returned)
            self.metrics.gauge("db.rows.fetched", row.tup_fetched)
            self.metrics.gauge("db.conflicts", row.conflicts)
            self.metrics.gauge("db.deadlocks", row.deadlocks)
            
            # Check for issues
            if row.deadlocks > 0:
                await self._alert_deadlocks(row.deadlocks)
            
            if cache_hit_ratio < 0.9:  # Less than 90% cache hit
                await self._alert_low_cache_hit_ratio(cache_hit_ratio)
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    async def _check_replication_status(self, session: AsyncSession):
        """Check replication status and lag."""
        try:
            # Get replication statistics
            result = await session.execute(
                text("""
                    SELECT 
                        client_addr,
                        state,
                        sent_lsn,
                        write_lsn,
                        flush_lsn,
                        replay_lsn,
                        pg_wal_lsn_diff(sent_lsn, replay_lsn) as lag_bytes,
                        sync_state
                    FROM pg_stat_replication
                """)
            )
            
            replicas = []
            for row in result:
                lag_mb = row.lag_bytes / (1024 * 1024) if row.lag_bytes else 0
                
                replicas.append({
                    "client": str(row.client_addr),
                    "state": row.state,
                    "lag_mb": lag_mb,
                    "sync_state": row.sync_state
                })
                
                # Export metrics
                labels = {"replica": str(row.client_addr)}
                self.metrics.gauge("db.replication.lag_mb", lag_mb, labels)
                self.metrics.gauge("db.replication.connected", 1 if row.state == "streaming" else 0, labels)
                
                # Check for high lag
                if lag_mb > 100:  # More than 100MB lag
                    await self._alert_replication_lag(str(row.client_addr), lag_mb)
            
            # Alert if no replicas
            if not replicas:
                await self._alert_no_replicas()
            
        except Exception as e:
            logger.error(f"Failed to check replication status: {e}")
    
    async def _check_thresholds(self):
        """Check various thresholds and alert if needed."""
        # Check long-running queries
        await self._check_long_running_queries()
        
        # Check for table maintenance needs
        await self._check_maintenance_needs()
    
    async def _check_long_running_queries(self):
        """Check for long-running queries."""
        try:
            # This would be implemented with pg_stat_activity
            pass
        except Exception as e:
            logger.error(f"Failed to check long-running queries: {e}")
    
    async def _check_maintenance_needs(self):
        """Check if tables need maintenance."""
        for table_name, metrics in self._table_metrics.items():
            # Check if vacuum is needed
            if metrics.last_vacuum:
                days_since_vacuum = (datetime.utcnow() - metrics.last_vacuum).days
                if days_since_vacuum > 7:  # More than a week
                    await self._alert_vacuum_needed(table_name, days_since_vacuum)
            
            # Check if analyze is needed
            if metrics.last_analyze:
                days_since_analyze = (datetime.utcnow() - metrics.last_analyze).days
                if days_since_analyze > 3:  # More than 3 days
                    await self._alert_analyze_needed(table_name, days_since_analyze)
    
    async def _alert_high_connection_usage(self, metrics: ConnectionMetrics):
        """Alert on high connection usage."""
        if self.alerts:
            await self.alerts.send_alert(
                title="High Database Connection Usage",
                message=f"Connection utilization at {metrics.connection_utilization:.1%}",
                severity=AlertSeverity.WARNING,
                tags={"component": "database", "metric": "connections"}
            )
    
    async def _alert_table_bloat(self, metrics: TableMetrics):
        """Alert on table bloat."""
        if self.alerts:
            await self.alerts.send_alert(
                title="Table Bloat Detected",
                message=f"Table {metrics.table_name} has {metrics.bloat_ratio:.1%} bloat",
                severity=AlertSeverity.WARNING,
                tags={"component": "database", "table": metrics.table_name}
            )
    
    async def _alert_deadlocks(self, count: int):
        """Alert on deadlocks."""
        if self.alerts:
            await self.alerts.send_alert(
                title="Database Deadlocks Detected",
                message=f"{count} deadlocks detected",
                severity=AlertSeverity.ERROR,
                tags={"component": "database", "metric": "deadlocks"}
            )
    
    async def _alert_low_cache_hit_ratio(self, ratio: float):
        """Alert on low cache hit ratio."""
        if self.alerts:
            await self.alerts.send_alert(
                title="Low Database Cache Hit Ratio",
                message=f"Cache hit ratio at {ratio:.1%}",
                severity=AlertSeverity.WARNING,
                tags={"component": "database", "metric": "cache"}
            )
    
    async def _alert_replication_lag(self, replica: str, lag_mb: float):
        """Alert on replication lag."""
        if self.alerts:
            await self.alerts.send_alert(
                title="High Replication Lag",
                message=f"Replica {replica} has {lag_mb:.1f}MB lag",
                severity=AlertSeverity.WARNING,
                tags={"component": "database", "replica": replica}
            )
    
    async def _alert_no_replicas(self):
        """Alert when no replicas are connected."""
        if self.alerts:
            await self.alerts.send_alert(
                title="No Database Replicas Connected",
                message="No streaming replicas are connected to master",
                severity=AlertSeverity.ERROR,
                tags={"component": "database", "metric": "replication"}
            )
    
    async def _alert_vacuum_needed(self, table: str, days: int):
        """Alert when vacuum is needed."""
        if self.alerts:
            await self.alerts.send_alert(
                title="Table Vacuum Needed",
                message=f"Table {table} hasn't been vacuumed in {days} days",
                severity=AlertSeverity.INFO,
                tags={"component": "database", "table": table}
            )
    
    async def _alert_analyze_needed(self, table: str, days: int):
        """Alert when analyze is needed."""
        if self.alerts:
            await self.alerts.send_alert(
                title="Table Analyze Needed",
                message=f"Table {table} hasn't been analyzed in {days} days",
                severity=AlertSeverity.INFO,
                tags={"component": "database", "table": table}
            )
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get connection usage summary."""
        if not self._connection_history:
            return {"message": "No connection data available"}
        
        recent = list(self._connection_history)[-60:]  # Last hour
        
        return {
            "current": {
                "total": recent[-1].total_connections if recent else 0,
                "active": recent[-1].active_connections if recent else 0,
                "utilization": recent[-1].connection_utilization if recent else 0
            },
            "hourly_avg": {
                "total": sum(m.total_connections for m in recent) / len(recent),
                "active": sum(m.active_connections for m in recent) / len(recent),
                "utilization": sum(m.connection_utilization for m in recent) / len(recent)
            },
            "hourly_max": {
                "total": max(m.total_connections for m in recent) if recent else 0,
                "active": max(m.active_connections for m in recent) if recent else 0,
                "utilization": max(m.connection_utilization for m in recent) if recent else 0
            }
        }
    
    def get_table_summary(self) -> List[Dict[str, Any]]:
        """Get table metrics summary."""
        tables = []
        
        for name, metrics in sorted(
            self._table_metrics.items(),
            key=lambda x: x[1].total_size,
            reverse=True
        )[:20]:  # Top 20 tables by size
            tables.append({
                "name": name,
                "rows": metrics.row_count,
                "size_mb": metrics.total_size / (1024 * 1024),
                "bloat_ratio": metrics.bloat_ratio,
                "last_vacuum": metrics.last_vacuum.isoformat() if metrics.last_vacuum else None,
                "last_analyze": metrics.last_analyze.isoformat() if metrics.last_analyze else None
            })
        
        return tables