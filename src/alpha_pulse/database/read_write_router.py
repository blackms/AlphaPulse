"""Database read/write splitting and routing."""

import asyncio
import random
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from ..config.database_config import DatabaseConfig, DatabaseNode
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector
from .connection_pool import ConnectionPool
from .connection_manager import ConnectionValidator, ConnectionHealth

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Intent of a database query."""
    
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    UNKNOWN = "unknown"


class ReplicaLagPolicy(Enum):
    """Policy for handling replica lag."""
    
    STRICT = "strict"  # Fail if lag exceeds threshold
    FALLBACK = "fallback"  # Fall back to master
    BEST_EFFORT = "best_effort"  # Use replica anyway
    ADAPTIVE = "adaptive"  # Dynamically adjust based on lag


class ReadWriteRouter:
    """Routes database queries to appropriate nodes."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        connection_pool: ConnectionPool,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize read/write router."""
        self.config = config
        self.pool = connection_pool
        self.metrics = metrics_collector
        
        # Replica tracking
        self._replica_health: Dict[str, ConnectionHealth] = {}
        self._replica_lag: Dict[str, float] = {}
        self._blacklisted_replicas: Set[str] = set()
        
        # Routing statistics
        self._routing_stats = {
            "reads_to_master": 0,
            "reads_to_replica": 0,
            "writes_to_master": 0,
            "fallbacks": 0
        }
        
        # Configuration
        self.max_replica_lag = 10.0  # seconds
        self.lag_check_interval = 30  # seconds
        self.lag_policy = ReplicaLagPolicy.FALLBACK
        
        # Monitoring
        self._lag_monitor_task: Optional[asyncio.Task] = None
        
    @asynccontextmanager
    async def get_session(
        self,
        intent: QueryIntent = QueryIntent.READ,
        consistency_required: bool = False
    ):
        """Get a database session based on query intent."""
        # Always use master for writes or when consistency is required
        if intent == QueryIntent.WRITE or consistency_required:
            self._routing_stats["writes_to_master"] += 1
            async with self.pool.get_master_session() as session:
                yield session
            return
        
        # For reads, try to use replica if available and healthy
        if intent == QueryIntent.READ and self.config.enable_read_write_split:
            replica_id = await self._select_healthy_replica()
            
            if replica_id:
                try:
                    self._routing_stats["reads_to_replica"] += 1
                    async with self._get_replica_session(replica_id) as session:
                        yield session
                    return
                except Exception as e:
                    logger.warning(f"Failed to use replica {replica_id}: {e}")
                    self._routing_stats["fallbacks"] += 1
        
        # Fall back to master
        self._routing_stats["reads_to_master"] += 1
        async with self.pool.get_master_session() as session:
            yield session
    
    async def _select_healthy_replica(self) -> Optional[str]:
        """Select a healthy replica for read operations."""
        if not self.config.read_replicas:
            return None
        
        available_replicas = []
        
        for i, replica in enumerate(self.config.read_replicas):
            replica_id = f"replica_{i}"
            
            # Skip blacklisted replicas
            if replica_id in self._blacklisted_replicas:
                continue
            
            # Check replica health
            health = self._replica_health.get(replica_id)
            if health and not health.is_healthy:
                continue
            
            # Check replica lag
            lag = self._replica_lag.get(replica_id, 0)
            
            if self.lag_policy == ReplicaLagPolicy.STRICT:
                if lag > self.max_replica_lag:
                    continue
            elif self.lag_policy == ReplicaLagPolicy.ADAPTIVE:
                # Adjust selection probability based on lag
                if lag > self.max_replica_lag * 2:
                    continue
            
            available_replicas.append((replica_id, replica.weight, lag))
        
        if not available_replicas:
            return None
        
        # Select based on weights and lag
        if self.config.connection_pool.load_balancing == "weighted":
            # Weighted random selection
            total_weight = sum(r[1] for r in available_replicas)
            rand = random.uniform(0, total_weight)
            
            cumulative = 0
            for replica_id, weight, lag in available_replicas:
                cumulative += weight
                if rand <= cumulative:
                    return replica_id
        
        elif self.config.connection_pool.load_balancing == "least_connections":
            # Select replica with least connections
            return min(
                available_replicas,
                key=lambda r: self.pool._active_connections.get(r[0], 0)
            )[0]
        
        else:  # round_robin or random
            return random.choice(available_replicas)[0]
    
    @asynccontextmanager
    async def _get_replica_session(self, replica_id: str):
        """Get session for specific replica."""
        # This is a simplified version - would need proper replica routing
        async with self.pool.get_replica_session() as session:
            yield session
    
    def analyze_query_intent(self, query: str) -> QueryIntent:
        """Analyze SQL query to determine its intent."""
        query_upper = query.strip().upper()
        
        # Check for write operations
        write_keywords = ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE"]
        for keyword in write_keywords:
            if query_upper.startswith(keyword):
                return QueryIntent.WRITE
        
        # Check for read operations
        if query_upper.startswith("SELECT"):
            # Check for locks or write intents
            if "FOR UPDATE" in query_upper or "FOR SHARE" in query_upper:
                return QueryIntent.WRITE
            return QueryIntent.READ
        
        # Check for transactions
        if query_upper.startswith(("BEGIN", "COMMIT", "ROLLBACK")):
            return QueryIntent.READ_WRITE
        
        return QueryIntent.UNKNOWN
    
    async def check_replica_lag(self, session: AsyncSession, replica_id: str) -> float:
        """Check replication lag for a specific replica."""
        try:
            # Get current WAL position on master
            async with self.pool.get_master_session() as master_session:
                master_result = await master_session.execute(
                    text("SELECT pg_current_wal_lsn() as wal_lsn")
                )
                master_lsn = (await master_result.fetchone()).wal_lsn
            
            # Get replica WAL position
            replica_result = await session.execute(
                text("SELECT pg_last_wal_receive_lsn() as wal_lsn")
            )
            replica_lsn = (await replica_result.fetchone()).wal_lsn
            
            # Calculate lag
            lag_result = await master_session.execute(
                text(
                    "SELECT pg_wal_lsn_diff(:master_lsn, :replica_lsn) as lag_bytes"
                ),
                {"master_lsn": master_lsn, "replica_lsn": replica_lsn}
            )
            lag_bytes = (await lag_result.fetchone()).lag_bytes
            
            # Estimate lag in seconds (rough approximation)
            lag_seconds = lag_bytes / (16 * 1024 * 1024)  # Assume 16MB/s
            
            self._replica_lag[replica_id] = lag_seconds
            
            # Record metrics
            if self.metrics:
                self.metrics.gauge(
                    "db.replica.lag_seconds",
                    lag_seconds,
                    {"replica": replica_id}
                )
            
            return lag_seconds
            
        except Exception as e:
            logger.error(f"Failed to check lag for replica {replica_id}: {e}")
            return float('inf')  # Assume high lag on error
    
    async def blacklist_replica(
        self,
        replica_id: str,
        duration: Optional[timedelta] = None
    ):
        """Temporarily blacklist a replica."""
        self._blacklisted_replicas.add(replica_id)
        logger.warning(f"Blacklisted replica {replica_id}")
        
        if self.metrics:
            self.metrics.increment("db.replica.blacklisted", {"replica": replica_id})
        
        # Schedule removal from blacklist
        if duration:
            asyncio.create_task(
                self._remove_from_blacklist_after(replica_id, duration)
            )
    
    async def _remove_from_blacklist_after(
        self,
        replica_id: str,
        duration: timedelta
    ):
        """Remove replica from blacklist after duration."""
        await asyncio.sleep(duration.total_seconds())
        self._blacklisted_replicas.discard(replica_id)
        logger.info(f"Removed replica {replica_id} from blacklist")
    
    async def start_lag_monitoring(self):
        """Start monitoring replica lag."""
        if self._lag_monitor_task:
            return
        
        self._lag_monitor_task = asyncio.create_task(self._lag_monitor_loop())
        logger.info("Started replica lag monitoring")
    
    async def stop_lag_monitoring(self):
        """Stop monitoring replica lag."""
        if self._lag_monitor_task:
            self._lag_monitor_task.cancel()
            try:
                await self._lag_monitor_task
            except asyncio.CancelledError:
                pass
            self._lag_monitor_task = None
        logger.info("Stopped replica lag monitoring")
    
    async def _lag_monitor_loop(self):
        """Monitor replica lag in background."""
        validator = ConnectionValidator(self.config, self.metrics)
        
        while True:
            try:
                await asyncio.sleep(self.lag_check_interval)
                
                # Check each replica
                for i, replica in enumerate(self.config.read_replicas):
                    replica_id = f"replica_{i}"
                    
                    try:
                        # Check health
                        health = await validator.health_check(replica)
                        self._replica_health[replica_id] = health
                        
                        if not health.is_healthy:
                            # Blacklist unhealthy replicas
                            await self.blacklist_replica(
                                replica_id,
                                duration=timedelta(minutes=5)
                            )
                            continue
                        
                        # Check lag
                        # This is simplified - would need actual replica session
                        async with self.pool.get_replica_session() as session:
                            lag = await self.check_replica_lag(session, replica_id)
                            
                            if lag > self.max_replica_lag * 3:
                                # Very high lag - blacklist temporarily
                                await self.blacklist_replica(
                                    replica_id,
                                    duration=timedelta(minutes=10)
                                )
                    
                    except Exception as e:
                        logger.error(f"Error monitoring replica {replica_id}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in lag monitor loop: {e}")
                await asyncio.sleep(self.lag_check_interval)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_reads = (
            self._routing_stats["reads_to_master"] + 
            self._routing_stats["reads_to_replica"]
        )
        
        return {
            "total_reads": total_reads,
            "reads_to_master": self._routing_stats["reads_to_master"],
            "reads_to_replica": self._routing_stats["reads_to_replica"],
            "writes_to_master": self._routing_stats["writes_to_master"],
            "fallbacks": self._routing_stats["fallbacks"],
            "read_replica_percentage": (
                (self._routing_stats["reads_to_replica"] / total_reads * 100)
                if total_reads > 0 else 0
            ),
            "blacklisted_replicas": list(self._blacklisted_replicas),
            "replica_lag": dict(self._replica_lag),
            "config": {
                "max_lag_seconds": self.max_replica_lag,
                "lag_policy": self.lag_policy.value,
                "replicas": len(self.config.read_replicas)
            }
        }
    
    def reset_stats(self):
        """Reset routing statistics."""
        self._routing_stats = {
            "reads_to_master": 0,
            "reads_to_replica": 0,
            "writes_to_master": 0,
            "fallbacks": 0
        }


class TransactionRouter:
    """Routes transactions based on read/write requirements."""
    
    def __init__(self, router: ReadWriteRouter):
        """Initialize transaction router."""
        self.router = router
        self._transaction_intents: Dict[str, QueryIntent] = {}
    
    @asynccontextmanager
    async def transaction(
        self,
        read_only: bool = False,
        consistency_required: bool = False
    ):
        """Start a routed transaction."""
        transaction_id = str(id(asyncio.current_task()))
        
        # Determine initial intent
        if read_only:
            intent = QueryIntent.READ
        else:
            intent = QueryIntent.READ_WRITE
        
        self._transaction_intents[transaction_id] = intent
        
        try:
            # Get session based on intent
            async with self.router.get_session(
                intent=intent,
                consistency_required=consistency_required
            ) as session:
                async with session.begin():
                    yield session
        finally:
            self._transaction_intents.pop(transaction_id, None)
    
    def upgrade_to_write(self, transaction_id: Optional[str] = None):
        """Upgrade transaction to write mode."""
        if not transaction_id:
            transaction_id = str(id(asyncio.current_task()))
        
        current_intent = self._transaction_intents.get(transaction_id)
        if current_intent == QueryIntent.READ:
            logger.warning(
                "Transaction upgraded from READ to WRITE - "
                "consider starting with WRITE intent"
            )
            self._transaction_intents[transaction_id] = QueryIntent.WRITE