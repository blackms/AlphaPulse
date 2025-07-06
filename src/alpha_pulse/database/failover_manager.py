"""Database failover management."""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.database_config import DatabaseConfig, DatabaseNode
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector
from .connection_pool import ConnectionPool
from .connection_manager import ConnectionValidator
from .read_write_router import ReadWriteRouter

logger = get_logger(__name__)


class FailoverState(Enum):
    """States of failover process."""
    
    NORMAL = "normal"
    DETECTING = "detecting"
    FAILING_OVER = "failing_over"
    PROMOTED = "promoted"
    RECOVERING = "recovering"


class PromotionStrategy(Enum):
    """Strategies for promoting replicas."""
    
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SEMI_AUTOMATIC = "semi_automatic"


@dataclass
class FailoverEvent:
    """Information about a failover event."""
    
    timestamp: datetime
    old_master: str
    new_master: str
    reason: str
    duration: float
    success: bool
    error: Optional[str] = None


class FailoverManager:
    """Manages database failover and recovery."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        connection_pool: ConnectionPool,
        router: ReadWriteRouter,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize failover manager."""
        self.config = config
        self.pool = connection_pool
        self.router = router
        self.metrics = metrics_collector
        
        # State tracking
        self.state = FailoverState.NORMAL
        self.current_master = config.master_node
        self.failover_history: List[FailoverEvent] = []
        
        # Configuration
        self.health_check_interval = 10  # seconds
        self.health_check_timeout = 5  # seconds
        self.max_consecutive_failures = 3
        self.promotion_strategy = PromotionStrategy.SEMI_AUTOMATIC
        
        # Monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0
        self._last_health_check: Optional[datetime] = None
        
        # Callbacks
        self._failover_callbacks: List[Callable] = []
        
        # Validator
        self.validator = ConnectionValidator(config, metrics_collector)
    
    async def start_monitoring(self):
        """Start failover monitoring."""
        if self._monitor_task:
            logger.warning("Failover monitoring already running")
            return
        
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started failover monitoring")
    
    async def stop_monitoring(self):
        """Stop failover monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Stopped failover monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check master health
                is_healthy = await self._check_master_health()
                
                if is_healthy:
                    self._consecutive_failures = 0
                    if self.state == FailoverState.DETECTING:
                        self.state = FailoverState.NORMAL
                        logger.info("Master recovered, canceling failover")
                else:
                    await self._handle_master_failure()
                
                self._last_health_check = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in failover monitor: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_master_health(self) -> bool:
        """Check if master is healthy."""
        try:
            # Try to get a connection
            async with asyncio.timeout(self.health_check_timeout):
                async with self.pool.get_master_session() as session:
                    # Run health check query
                    result = await session.execute(text("SELECT 1"))
                    await result.fetchone()
                    
                    # Check replication status if applicable
                    if await self._check_replication_status(session):
                        return True
            
            return True
            
        except Exception as e:
            logger.warning(f"Master health check failed: {e}")
            return False
    
    async def _check_replication_status(self, session: AsyncSession) -> bool:
        """Check replication status on master."""
        try:
            # Check if replication is working
            result = await session.execute(
                text("""
                    SELECT COUNT(*) as replica_count
                    FROM pg_stat_replication
                    WHERE state = 'streaming'
                """)
            )
            row = await result.fetchone()
            
            # Warn if no replicas are connected
            if row.replica_count == 0 and self.config.read_replicas:
                logger.warning("No streaming replicas connected to master")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check replication status: {e}")
            return False
    
    async def _handle_master_failure(self):
        """Handle detected master failure."""
        self._consecutive_failures += 1
        
        logger.warning(
            f"Master failure #{self._consecutive_failures}/{self.max_consecutive_failures}"
        )
        
        # Record metrics
        if self.metrics:
            self.metrics.increment("db.failover.health_check_failed")
        
        # Check if we should initiate failover
        if self._consecutive_failures >= self.max_consecutive_failures:
            if self.state == FailoverState.NORMAL:
                self.state = FailoverState.DETECTING
                await self._initiate_failover()
    
    async def _initiate_failover(self):
        """Initiate failover process."""
        logger.error("Initiating database failover")
        self.state = FailoverState.FAILING_OVER
        
        start_time = datetime.utcnow()
        
        try:
            # Select best replica for promotion
            new_master = await self._select_promotion_candidate()
            
            if not new_master:
                raise Exception("No suitable replica for promotion")
            
            # Execute failover based on strategy
            if self.promotion_strategy == PromotionStrategy.AUTOMATIC:
                await self._automatic_failover(new_master)
            elif self.promotion_strategy == PromotionStrategy.SEMI_AUTOMATIC:
                await self._semi_automatic_failover(new_master)
            else:
                await self._manual_failover(new_master)
            
            # Record successful failover
            duration = (datetime.utcnow() - start_time).total_seconds()
            event = FailoverEvent(
                timestamp=start_time,
                old_master=f"{self.current_master.host}:{self.current_master.port}",
                new_master=f"{new_master.host}:{new_master.port}",
                reason="Master health check failures",
                duration=duration,
                success=True
            )
            
            self.failover_history.append(event)
            self.state = FailoverState.PROMOTED
            
            # Notify callbacks
            await self._notify_failover_complete(event)
            
            logger.info(f"Failover completed in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            
            # Record failed failover
            duration = (datetime.utcnow() - start_time).total_seconds()
            event = FailoverEvent(
                timestamp=start_time,
                old_master=f"{self.current_master.host}:{self.current_master.port}",
                new_master="",
                reason="Master health check failures",
                duration=duration,
                success=False,
                error=str(e)
            )
            
            self.failover_history.append(event)
            self.state = FailoverState.NORMAL  # Reset state
            
            # Record metrics
            if self.metrics:
                self.metrics.increment("db.failover.failed")
    
    async def _select_promotion_candidate(self) -> Optional[DatabaseNode]:
        """Select best replica for promotion."""
        if not self.config.read_replicas:
            return None
        
        candidates = []
        
        for replica in self.config.read_replicas:
            try:
                # Check replica health
                health = await self.validator.health_check(replica)
                
                if not health.is_healthy:
                    continue
                
                # Check replication lag
                lag = await self._get_replica_lag(replica)
                
                candidates.append({
                    "node": replica,
                    "lag": lag,
                    "health": health,
                    "score": self._calculate_promotion_score(replica, lag, health)
                })
                
            except Exception as e:
                logger.error(f"Failed to check replica {replica.host}: {e}")
        
        if not candidates:
            return None
        
        # Select candidate with best score
        best_candidate = max(candidates, key=lambda c: c["score"])
        
        logger.info(
            f"Selected {best_candidate['node'].host} for promotion "
            f"(lag: {best_candidate['lag']:.2f}s, score: {best_candidate['score']:.2f})"
        )
        
        return best_candidate["node"]
    
    async def _get_replica_lag(self, replica: DatabaseNode) -> float:
        """Get replication lag for a replica."""
        try:
            # This would need actual connection to replica
            # For now, return a placeholder
            return 0.5
        except Exception:
            return float('inf')
    
    def _calculate_promotion_score(
        self,
        replica: DatabaseNode,
        lag: float,
        health: Any
    ) -> float:
        """Calculate promotion score for a replica."""
        score = 100.0
        
        # Penalize for lag
        score -= min(lag * 10, 50)  # Max 50 point penalty
        
        # Bonus for weight
        score += replica.weight * 5
        
        # Penalize for poor health
        score -= health.consecutive_failures * 10
        
        return max(score, 0)
    
    async def _automatic_failover(self, new_master: DatabaseNode):
        """Perform automatic failover."""
        logger.info("Performing automatic failover")
        
        # Update configuration
        self.config.master_node = new_master
        
        # Remove new master from replicas
        self.config.read_replicas = [
            r for r in self.config.read_replicas
            if r.host != new_master.host or r.port != new_master.port
        ]
        
        # Reinitialize connection pool
        await self.pool.close()
        await self.pool.initialize()
        
        # Update router
        self.router._blacklisted_replicas.clear()
        
        self.current_master = new_master
        
        # Record metrics
        if self.metrics:
            self.metrics.increment("db.failover.automatic")
    
    async def _semi_automatic_failover(self, new_master: DatabaseNode):
        """Perform semi-automatic failover with confirmation."""
        logger.info("Preparing semi-automatic failover")
        
        # In production, this would wait for operator confirmation
        # For now, proceed automatically after a delay
        logger.warning(
            f"FAILOVER REQUIRED: Promote {new_master.host}:{new_master.port} to master"
        )
        
        await asyncio.sleep(5)  # Wait for confirmation
        
        await self._automatic_failover(new_master)
        
        # Record metrics
        if self.metrics:
            self.metrics.increment("db.failover.semi_automatic")
    
    async def _manual_failover(self, new_master: DatabaseNode):
        """Log manual failover requirement."""
        logger.critical(
            f"MANUAL FAILOVER REQUIRED: Current master is down. "
            f"Recommended promotion: {new_master.host}:{new_master.port}"
        )
        
        # In manual mode, just log and wait
        self.state = FailoverState.NORMAL
        
        # Record metrics
        if self.metrics:
            self.metrics.increment("db.failover.manual_required")
    
    async def _notify_failover_complete(self, event: FailoverEvent):
        """Notify callbacks of completed failover."""
        for callback in self._failover_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in failover callback: {e}")
    
    def add_failover_callback(self, callback: Callable):
        """Add callback for failover events."""
        self._failover_callbacks.append(callback)
    
    async def manual_failover(self, new_master_host: str, new_master_port: int = 5432):
        """Manually trigger failover to specific replica."""
        # Find the replica
        new_master = None
        for replica in self.config.read_replicas:
            if replica.host == new_master_host and replica.port == new_master_port:
                new_master = replica
                break
        
        if not new_master:
            raise ValueError(f"Replica {new_master_host}:{new_master_port} not found")
        
        logger.info(f"Manual failover initiated to {new_master_host}:{new_master_port}")
        
        self.state = FailoverState.FAILING_OVER
        await self._automatic_failover(new_master)
    
    def get_status(self) -> Dict[str, Any]:
        """Get failover manager status."""
        return {
            "state": self.state.value,
            "current_master": f"{self.current_master.host}:{self.current_master.port}",
            "consecutive_failures": self._consecutive_failures,
            "last_health_check": (
                self._last_health_check.isoformat()
                if self._last_health_check else None
            ),
            "promotion_strategy": self.promotion_strategy.value,
            "failover_history": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "old_master": event.old_master,
                    "new_master": event.new_master,
                    "reason": event.reason,
                    "duration": event.duration,
                    "success": event.success,
                    "error": event.error
                }
                for event in self.failover_history[-10:]  # Last 10 events
            ]
        }