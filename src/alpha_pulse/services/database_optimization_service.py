"""Database optimization service."""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker

from ..config.database_config import DatabaseConfig
from ..database.connection_pool import ConnectionPool
from ..database.connection_manager import ConnectionValidator, ConnectionMonitor
from ..database.query_analyzer import QueryAnalyzer
from ..database.slow_query_detector import SlowQueryDetector
from ..database.query_optimizer import QueryOptimizer
from ..database.index_advisor import IndexAdvisor
from ..database.index_manager import IndexManager
from ..database.partition_manager import PartitionManager
from ..database.read_write_router import ReadWriteRouter, TransactionRouter, QueryIntent
from ..database.load_balancer import AdaptiveLoadBalancer
from ..database.failover_manager import FailoverManager
from ..database.database_monitor import DatabaseMonitor
from ..monitoring.metrics import MetricsCollector
from ..monitoring.alert_manager import AlertManager
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseOptimizationService:
    """Comprehensive database optimization service."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        metrics_collector: MetricsCollector,
        alert_manager: Optional[AlertManager] = None
    ):
        """Initialize database optimization service."""
        self.config = config
        self.metrics = metrics_collector
        self.alerts = alert_manager
        
        # Core components
        self.connection_pool = ConnectionPool(config, metrics_collector)
        self.validator = ConnectionValidator(config, metrics_collector)
        
        # Query optimization
        self.query_analyzer = QueryAnalyzer(metrics_collector)
        self.query_optimizer = QueryOptimizer(config.query_optimization, self.query_analyzer)
        self.slow_query_detector = SlowQueryDetector(
            config.query_optimization,
            self.query_analyzer,
            metrics_collector
        )
        
        # Index management
        self.index_advisor = IndexAdvisor(metrics_collector)
        self.index_manager = IndexManager(self.index_advisor, metrics_collector)
        
        # Partitioning
        self.partition_manager = PartitionManager(metrics_collector)
        
        # Read/write splitting
        self.router = ReadWriteRouter(config, self.connection_pool, metrics_collector)
        self.transaction_router = TransactionRouter(self.router)
        self.load_balancer = AdaptiveLoadBalancer(metrics_collector)
        
        # Failover
        self.failover_manager = FailoverManager(
            config, self.connection_pool, self.router, metrics_collector
        )
        
        # Monitoring
        self.connection_monitor = ConnectionMonitor(self.validator)
        self.database_monitor = DatabaseMonitor(metrics_collector, alert_manager)
        
        # Service state
        self._is_initialized = False
        self._maintenance_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize all database optimization components."""
        try:
            logger.info("Initializing database optimization service...")
            
            # Initialize connection pool
            await self.connection_pool.initialize()
            
            # Attach slow query detector to engines
            if self.connection_pool._master_engine:
                self.slow_query_detector.attach_to_engine(
                    self.connection_pool._master_engine
                )
            
            # Set up load balancer nodes
            for i, replica in enumerate(self.config.read_replicas):
                self.load_balancer.add_node(replica, f"replica_{i}")
            
            # Start monitoring services
            await self.router.start_lag_monitoring()
            await self.failover_manager.start_monitoring()
            await self.connection_monitor.start_monitoring(self.connection_pool)
            await self.database_monitor.start_monitoring(
                self.connection_pool._master_session_factory
            )
            await self.slow_query_detector.start_monitoring()
            
            # Start maintenance tasks
            await self._start_maintenance()
            
            self._is_initialized = True
            logger.info("Database optimization service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database optimization service: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all database optimization components."""
        try:
            logger.info("Shutting down database optimization service...")
            
            # Stop maintenance
            await self._stop_maintenance()
            
            # Stop monitoring services
            await self.router.stop_lag_monitoring()
            await self.failover_manager.stop_monitoring()
            await self.connection_monitor.stop_monitoring()
            await self.database_monitor.stop_monitoring()
            await self.slow_query_detector.stop_monitoring()
            await self.index_manager.stop_maintenance()
            await self.partition_manager.stop_maintenance()
            
            # Close connection pool
            await self.connection_pool.close()
            
            self._is_initialized = False
            logger.info("Database optimization service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during database optimization shutdown: {e}")
    
    async def get_read_session(self, consistency_required: bool = False):
        """Get a database session for read operations."""
        return self.router.get_session(
            intent=QueryIntent.READ,
            consistency_required=consistency_required
        )
    
    async def get_write_session(self):
        """Get a database session for write operations."""
        return self.router.get_session(intent=QueryIntent.WRITE)
    
    async def analyze_query(
        self,
        query: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze a query and provide optimization suggestions."""
        async with self.connection_pool.get_master_session() as session:
            # Analyze query
            plan = await self.query_analyzer.analyze_query(session, query, params)
            
            # Get optimization suggestions
            optimized_query, suggestions = await self.query_optimizer.optimize_query(
                session, query, params
            )
            
            # Estimate cost
            cost = await self.query_optimizer.estimate_cost(session, query, params)
            
            return {
                "original_query": query,
                "optimized_query": optimized_query,
                "execution_plan": {
                    "type": plan.plan_type,
                    "cost": plan.total_cost,
                    "rows": plan.rows,
                    "is_efficient": plan.is_efficient
                },
                "cost_estimate": {
                    "cpu_cost": cost.cpu_cost,
                    "io_cost": cost.io_cost,
                    "total_cost": cost.total_cost,
                    "estimated_time_ms": cost.estimated_time
                },
                "suggestions": [
                    {
                        "type": s.optimization_type.value,
                        "description": s.description,
                        "priority": s.priority,
                        "estimated_improvement": s.estimated_improvement
                    }
                    for s in suggestions
                ]
            }
    
    async def analyze_database_health(self) -> Dict[str, Any]:
        """Get comprehensive database health analysis."""
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "connection_pool": await self.connection_pool.get_pool_stats(),
            "routing": self.router.get_routing_stats(),
            "load_balancing": self.load_balancer.get_node_status(),
            "failover": self.failover_manager.get_status(),
            "slow_queries": await self._get_slow_query_summary(),
            "index_recommendations": self.index_advisor.get_recommendations_summary(),
            "connections": self.database_monitor.get_connection_summary(),
            "tables": self.database_monitor.get_table_summary(),
            "health_checks": self.validator.get_health_summary()
        }
        
        return health_report
    
    async def _get_slow_query_summary(self) -> Dict[str, Any]:
        """Get slow query summary."""
        async with self.connection_pool.get_master_session() as session:
            return await self.slow_query_detector.analyze_slow_queries(session)
    
    async def optimize_indexes(
        self,
        tables: Optional[List[str]] = None,
        apply_recommendations: bool = False
    ) -> Dict[str, Any]:
        """Analyze and optionally optimize database indexes."""
        async with self.connection_pool.get_master_session() as session:
            # Analyze indexes
            recommendations = await self.index_advisor.analyze_indexes(session, tables)
            
            results = {
                "recommendations": [
                    {
                        "table": r.table_name,
                        "columns": r.columns,
                        "type": r.index_type,
                        "reason": r.reason,
                        "priority": r.priority,
                        "estimated_benefit": r.estimated_benefit
                    }
                    for r in recommendations
                ],
                "applied": {}
            }
            
            # Apply recommendations if requested
            if apply_recommendations and recommendations:
                applied = await self.index_manager.apply_recommendations(
                    session,
                    recommendations,
                    priority_threshold=3
                )
                results["applied"] = applied
            
            return results
    
    async def setup_partitioning(
        self,
        table_name: str,
        partition_column: str = "timestamp",
        interval: str = "daily",
        retention_days: int = 90
    ) -> bool:
        """Set up partitioning for a table."""
        from ..database.partition_manager import PartitionStrategy, PartitionType, PartitionInterval
        
        # Create partition strategy
        strategy = PartitionStrategy(
            table_name=table_name,
            partition_column=partition_column,
            partition_type=PartitionType.RANGE,
            interval=PartitionInterval(interval),
            retention_days=retention_days
        )
        
        async with self.connection_pool.get_master_session() as session:
            return await self.partition_manager.setup_partitioning(
                session, table_name, strategy
            )
    
    async def _start_maintenance(self):
        """Start background maintenance tasks."""
        if self._maintenance_task:
            return
        
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info("Started database maintenance tasks")
    
    async def _stop_maintenance(self):
        """Stop background maintenance tasks."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
        logger.info("Stopped database maintenance tasks")
    
    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while True:
            try:
                # Run every hour
                await asyncio.sleep(3600)
                
                logger.info("Running database maintenance")
                
                async with self.connection_pool.get_master_session() as session:
                    # Update table statistics
                    await self.query_optimizer.update_table_statistics(session)
                    
                    # Check for index bloat
                    bloated = await self.index_manager.check_index_bloat(session)
                    if bloated:
                        logger.warning(f"Found {len(bloated)} bloated indexes")
                    
                    # Analyze indexes
                    recommendations = await self.index_advisor.analyze_indexes(session)
                    if recommendations:
                        high_priority = [r for r in recommendations if r.priority >= 4]
                        if high_priority:
                            logger.info(
                                f"Found {len(high_priority)} high-priority "
                                "index recommendations"
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(3600)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "initialized": self._is_initialized,
            "components": {
                "connection_pool": self.connection_pool._is_initialized,
                "read_write_router": len(self.router._blacklisted_replicas) == 0,
                "failover_manager": self.failover_manager.state.value,
                "slow_query_detector": self.slow_query_detector.get_statistics(),
                "index_advisor": bool(self.index_advisor._last_analysis),
                "load_balancer": len(self.load_balancer._active_nodes)
            },
            "config": {
                "master": f"{self.config.master_node.host}:{self.config.master_node.port}",
                "replicas": len(self.config.read_replicas),
                "read_write_split": self.config.enable_read_write_split,
                "query_optimization": self.config.query_optimization.enable_query_cache
            }
        }