"""Slow query detection and monitoring."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from sqlalchemy.engine import Engine

from ..config.database_config import QueryOptimizationConfig
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector
from .query_analyzer import QueryAnalyzer, QueryPlan

logger = get_logger(__name__)


@dataclass
class SlowQueryInfo:
    """Information about a slow query."""
    
    query: str
    execution_time: float
    timestamp: datetime
    session_id: str
    user: Optional[str] = None
    database: Optional[str] = None
    application: Optional[str] = None
    parameters: Optional[Dict] = None
    query_plan: Optional[QueryPlan] = None
    stack_trace: Optional[List[str]] = None


@dataclass 
class QueryPattern:
    """Pattern for grouping similar queries."""
    
    pattern: str
    count: int = 0
    total_time: float = 0
    min_time: float = float('inf')
    max_time: float = 0
    queries: List[SlowQueryInfo] = field(default_factory=list)
    
    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.count if self.count > 0 else 0
    
    def add_query(self, query_info: SlowQueryInfo):
        """Add query to pattern statistics."""
        self.count += 1
        self.total_time += query_info.execution_time
        self.min_time = min(self.min_time, query_info.execution_time)
        self.max_time = max(self.max_time, query_info.execution_time)
        self.queries.append(query_info)
        
        # Keep only last 100 queries per pattern
        if len(self.queries) > 100:
            self.queries.pop(0)


class SlowQueryDetector:
    """Detects and monitors slow database queries."""
    
    def __init__(
        self,
        config: QueryOptimizationConfig,
        query_analyzer: QueryAnalyzer,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize slow query detector."""
        self.config = config
        self.analyzer = query_analyzer
        self.metrics = metrics_collector
        
        # Storage
        self._slow_queries: List[SlowQueryInfo] = []
        self._query_patterns: Dict[str, QueryPattern] = {}
        self._alert_callbacks: List[Callable] = []
        
        # Monitoring
        self._query_start_times: Dict[str, float] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._total_queries = 0
        self._slow_query_count = 0
        
    def attach_to_engine(self, engine: AsyncEngine):
        """Attach detector to SQLAlchemy engine."""
        # Listen for query execution events
        event.listen(engine.sync_engine, "before_cursor_execute", self._before_query)
        event.listen(engine.sync_engine, "after_cursor_execute", self._after_query)
        
        logger.info(f"Slow query detector attached to engine with threshold: {self.config.slow_query_threshold}s")
    
    def _before_query(self, conn, cursor, statement, parameters, context, executemany):
        """Called before query execution."""
        # Generate unique query ID
        query_id = id(context)
        self._query_start_times[query_id] = time.time()
        
        # Store query details in context
        context._query_info = {
            "statement": statement,
            "parameters": parameters,
            "start_time": time.time()
        }
    
    def _after_query(self, conn, cursor, statement, parameters, context, executemany):
        """Called after query execution."""
        query_id = id(context)
        
        if query_id not in self._query_start_times:
            return
        
        # Calculate execution time
        start_time = self._query_start_times.pop(query_id)
        execution_time = time.time() - start_time
        
        self._total_queries += 1
        
        # Check if query is slow
        if execution_time >= self.config.slow_query_threshold:
            self._handle_slow_query(
                statement, 
                parameters, 
                execution_time,
                context
            )
        
        # Record metrics
        if self.metrics:
            self.metrics.histogram(
                "db.query.execution_time",
                execution_time,
                {"slow": execution_time >= self.config.slow_query_threshold}
            )
    
    def _handle_slow_query(
        self, 
        statement: str, 
        parameters: Any,
        execution_time: float,
        context: Any
    ):
        """Handle detected slow query."""
        self._slow_query_count += 1
        
        # Create query info
        query_info = SlowQueryInfo(
            query=statement,
            execution_time=execution_time,
            timestamp=datetime.utcnow(),
            session_id=str(id(context)),
            parameters=parameters if isinstance(parameters, dict) else None
        )
        
        # Add connection info if available
        if hasattr(context, "connection"):
            conn_info = context.connection.info
            query_info.user = conn_info.get("user")
            query_info.database = conn_info.get("database")
            query_info.application = conn_info.get("application_name")
        
        # Store slow query
        self._slow_queries.append(query_info)
        
        # Keep only last 10000 slow queries
        if len(self._slow_queries) > 10000:
            self._slow_queries.pop(0)
        
        # Group by pattern
        pattern = self._get_query_pattern(statement)
        if pattern not in self._query_patterns:
            self._query_patterns[pattern] = QueryPattern(pattern)
        self._query_patterns[pattern].add_query(query_info)
        
        # Log if enabled
        if self.config.log_slow_queries:
            logger.warning(
                f"Slow query detected ({execution_time:.2f}s): "
                f"{statement[:200]}{'...' if len(statement) > 200 else ''}"
            )
        
        # Trigger alerts
        asyncio.create_task(self._trigger_alerts(query_info))
        
        # Record in analyzer
        asyncio.create_task(
            self.analyzer.log_slow_query(statement, execution_time)
        )
    
    def _get_query_pattern(self, query: str) -> str:
        """Extract pattern from query for grouping."""
        import re
        
        # Normalize whitespace
        pattern = " ".join(query.split())
        
        # Replace values with placeholders
        # Numbers
        pattern = re.sub(r'\b\d+\b', 'N', pattern)
        # Quoted strings
        pattern = re.sub(r"'[^']*'", 'S', pattern)
        pattern = re.sub(r'"[^"]*"', 'S', pattern)
        # Parameters
        pattern = re.sub(r'\$\d+', '$N', pattern)
        pattern = re.sub(r':\w+', ':param', pattern)
        
        # Limit length
        if len(pattern) > 500:
            pattern = pattern[:500] + "..."
        
        return pattern
    
    async def _trigger_alerts(self, query_info: SlowQueryInfo):
        """Trigger alert callbacks for slow query."""
        for callback in self._alert_callbacks:
            try:
                await callback(query_info)
            except Exception as e:
                logger.error(f"Error in slow query alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for slow query alerts."""
        self._alert_callbacks.append(callback)
    
    async def analyze_slow_queries(self, session: AsyncSession) -> Dict[str, Any]:
        """Analyze collected slow queries."""
        if not self._slow_queries:
            return {"message": "No slow queries detected"}
        
        # Get query patterns sorted by total time
        patterns = sorted(
            self._query_patterns.values(),
            key=lambda p: p.total_time,
            reverse=True
        )
        
        # Analyze top patterns
        analyzed_patterns = []
        
        for pattern in patterns[:10]:  # Top 10 patterns
            # Get a sample query from this pattern
            sample_query = pattern.queries[0].query if pattern.queries else None
            
            if sample_query and self.config.analyze_query_plans:
                # Analyze execution plan
                try:
                    plan = await self.analyzer.analyze_query(session, sample_query)
                    pattern_info = {
                        "pattern": pattern.pattern,
                        "count": pattern.count,
                        "total_time": pattern.total_time,
                        "avg_time": pattern.avg_time,
                        "min_time": pattern.min_time,
                        "max_time": pattern.max_time,
                        "query_plan": {
                            "cost": plan.total_cost,
                            "rows": plan.rows,
                            "efficient": plan.is_efficient,
                            "suggestions": plan.suggestions
                        }
                    }
                except Exception as e:
                    logger.error(f"Failed to analyze query pattern: {e}")
                    pattern_info = {
                        "pattern": pattern.pattern,
                        "count": pattern.count,
                        "total_time": pattern.total_time,
                        "avg_time": pattern.avg_time,
                        "min_time": pattern.min_time,
                        "max_time": pattern.max_time
                    }
            else:
                pattern_info = {
                    "pattern": pattern.pattern,
                    "count": pattern.count,
                    "total_time": pattern.total_time,
                    "avg_time": pattern.avg_time,
                    "min_time": pattern.min_time,
                    "max_time": pattern.max_time
                }
            
            analyzed_patterns.append(pattern_info)
        
        # Overall statistics
        total_slow_time = sum(q.execution_time for q in self._slow_queries)
        
        return {
            "total_queries": self._total_queries,
            "slow_queries": self._slow_query_count,
            "slow_query_percentage": (
                (self._slow_query_count / self._total_queries * 100)
                if self._total_queries > 0 else 0
            ),
            "total_slow_time": total_slow_time,
            "threshold": self.config.slow_query_threshold,
            "top_patterns": analyzed_patterns,
            "recent_queries": [
                {
                    "query": q.query[:200] + "..." if len(q.query) > 200 else q.query,
                    "execution_time": q.execution_time,
                    "timestamp": q.timestamp.isoformat()
                }
                for q in self._slow_queries[-10:]  # Last 10 queries
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "total_queries": self._total_queries,
            "slow_queries": self._slow_query_count,
            "slow_percentage": (
                (self._slow_query_count / self._total_queries * 100)
                if self._total_queries > 0 else 0
            ),
            "pattern_count": len(self._query_patterns),
            "threshold": self.config.slow_query_threshold
        }
    
    async def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring_task:
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started slow query monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("Stopped slow query monitoring")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean up old queries (keep last 24 hours)
                cutoff = datetime.utcnow() - timedelta(hours=24)
                self._slow_queries = [
                    q for q in self._slow_queries
                    if q.timestamp > cutoff
                ]
                
                # Log statistics
                stats = self.get_statistics()
                if stats["slow_queries"] > 0:
                    logger.info(
                        f"Slow query stats: {stats['slow_queries']} slow queries "
                        f"({stats['slow_percentage']:.1f}%) out of {stats['total_queries']} total"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in slow query monitoring: {e}")
                await asyncio.sleep(60)