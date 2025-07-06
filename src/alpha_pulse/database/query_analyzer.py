"""Query execution plan analyzer."""

import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of database queries."""
    
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"
    TRUNCATE = "truncate"


@dataclass
class QueryPlan:
    """Query execution plan details."""
    
    query: str
    plan_type: str
    total_cost: float
    rows: int
    width: int
    execution_time: Optional[float] = None
    node_details: List[Dict] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        """Initialize empty lists."""
        if self.node_details is None:
            self.node_details = []
        if self.suggestions is None:
            self.suggestions = []
    
    @property
    def is_efficient(self) -> bool:
        """Check if query plan is efficient."""
        # Consider query inefficient if:
        # - Uses sequential scan on large tables
        # - Has high cost relative to rows
        # - Contains nested loops with high iterations
        
        if self.total_cost > 10000:
            return False
        
        if self.rows > 0 and (self.total_cost / self.rows) > 100:
            return False
        
        # Check for sequential scans
        for node in self.node_details:
            if node.get("node_type") == "Seq Scan" and node.get("rows", 0) > 1000:
                return False
        
        return True


class QueryAnalyzer:
    """Analyzes query execution plans and performance."""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize query analyzer."""
        self.metrics = metrics_collector
        self._query_cache: Dict[str, QueryPlan] = {}
        self._slow_query_log: List[Dict] = []
        
    async def analyze_query(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict] = None
    ) -> QueryPlan:
        """Analyze query execution plan."""
        try:
            # Normalize query for caching
            normalized_query = self._normalize_query(query)
            
            # Check cache
            if normalized_query in self._query_cache:
                return self._query_cache[normalized_query]
            
            # Get query type
            query_type = self._get_query_type(query)
            
            # Get execution plan
            plan = await self._get_execution_plan(session, query, params)
            
            # Analyze plan
            analyzed_plan = self._analyze_plan(plan, query)
            
            # Add optimization suggestions
            analyzed_plan.suggestions = self._get_optimization_suggestions(
                analyzed_plan, query_type
            )
            
            # Cache result
            self._query_cache[normalized_query] = analyzed_plan
            
            # Record metrics
            if self.metrics:
                self.metrics.gauge(
                    "db.query.cost",
                    analyzed_plan.total_cost,
                    {"type": query_type.value}
                )
                self.metrics.increment(
                    "db.query.analyzed",
                    {"type": query_type.value, "efficient": analyzed_plan.is_efficient}
                )
            
            return analyzed_plan
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return QueryPlan(
                query=query,
                plan_type="unknown",
                total_cost=0,
                rows=0,
                width=0
            )
    
    async def _get_execution_plan(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict] = None
    ) -> List[Dict]:
        """Get query execution plan from database."""
        # Use EXPLAIN ANALYZE for actual execution stats
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
        
        try:
            result = await session.execute(text(explain_query), params or {})
            plan_data = result.scalar()
            
            if isinstance(plan_data, list) and plan_data:
                return plan_data[0].get("Plan", {})
            
            return {}
            
        except Exception as e:
            # Fallback to regular EXPLAIN if ANALYZE fails
            try:
                explain_query = f"EXPLAIN (FORMAT JSON) {query}"
                result = await session.execute(text(explain_query), params or {})
                plan_data = result.scalar()
                
                if isinstance(plan_data, list) and plan_data:
                    return plan_data[0].get("Plan", {})
                
            except Exception as fallback_error:
                logger.error(f"Failed to get execution plan: {fallback_error}")
                
            return {}
    
    def _analyze_plan(self, plan: Dict, query: str) -> QueryPlan:
        """Analyze execution plan details."""
        # Extract plan information
        plan_type = plan.get("Node Type", "Unknown")
        total_cost = plan.get("Total Cost", 0)
        rows = plan.get("Plan Rows", 0)
        width = plan.get("Plan Width", 0)
        execution_time = plan.get("Actual Total Time", None)
        
        # Recursively collect node details
        node_details = []
        self._collect_node_details(plan, node_details)
        
        return QueryPlan(
            query=query,
            plan_type=plan_type,
            total_cost=total_cost,
            rows=rows,
            width=width,
            execution_time=execution_time,
            node_details=node_details
        )
    
    def _collect_node_details(self, node: Dict, details: List[Dict], depth: int = 0):
        """Recursively collect details from plan nodes."""
        node_info = {
            "depth": depth,
            "node_type": node.get("Node Type"),
            "cost": node.get("Total Cost", 0),
            "rows": node.get("Plan Rows", 0),
            "width": node.get("Plan Width", 0),
            "actual_time": node.get("Actual Total Time"),
            "actual_rows": node.get("Actual Rows"),
            "relation": node.get("Relation Name"),
            "index": node.get("Index Name"),
            "filter": node.get("Filter"),
            "join_type": node.get("Join Type"),
        }
        
        # Remove None values
        node_info = {k: v for k, v in node_info.items() if v is not None}
        details.append(node_info)
        
        # Process child plans
        if "Plans" in node:
            for child in node["Plans"]:
                self._collect_node_details(child, details, depth + 1)
    
    def _get_optimization_suggestions(
        self,
        plan: QueryPlan,
        query_type: QueryType
    ) -> List[str]:
        """Generate query optimization suggestions."""
        suggestions = []
        
        # Check for sequential scans
        seq_scans = [
            node for node in plan.node_details
            if node.get("node_type") == "Seq Scan" and node.get("rows", 0) > 1000
        ]
        
        if seq_scans:
            for scan in seq_scans:
                if scan.get("relation"):
                    suggestions.append(
                        f"Consider adding index on {scan['relation']} "
                        f"(scanned {scan.get('rows', 0)} rows)"
                    )
        
        # Check for nested loops with high cost
        nested_loops = [
            node for node in plan.node_details
            if node.get("node_type") == "Nested Loop" and node.get("cost", 0) > 1000
        ]
        
        if nested_loops:
            suggestions.append(
                "High-cost nested loops detected. Consider using hash joins "
                "or ensuring proper indexes exist"
            )
        
        # Check for missing statistics
        if plan.total_cost > 10000 and plan.rows == 0:
            suggestions.append(
                "Query planner estimates 0 rows. Run ANALYZE to update statistics"
            )
        
        # Check for sorting operations
        sorts = [
            node for node in plan.node_details
            if node.get("node_type") in ["Sort", "Sort Key"]
        ]
        
        if sorts and plan.total_cost > 5000:
            suggestions.append(
                "Consider adding indexes to support sort operations"
            )
        
        # Query-type specific suggestions
        if query_type == QueryType.SELECT:
            if plan.total_cost > 10000:
                suggestions.append("Consider query result caching for expensive SELECT")
            
            # Check for SELECT *
            if "SELECT *" in plan.query.upper():
                suggestions.append("Avoid SELECT *, specify only needed columns")
        
        elif query_type == QueryType.UPDATE:
            # Check for missing WHERE clause
            if "WHERE" not in plan.query.upper():
                suggestions.append("WARNING: UPDATE without WHERE clause affects all rows")
        
        elif query_type == QueryType.DELETE:
            # Check for missing WHERE clause
            if "WHERE" not in plan.query.upper():
                suggestions.append("WARNING: DELETE without WHERE clause removes all rows")
        
        return suggestions
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for caching."""
        # Remove extra whitespace
        normalized = " ".join(query.split())
        
        # Remove comments
        normalized = re.sub(r'/\*.*?\*/', '', normalized)
        normalized = re.sub(r'--.*?$', '', normalized, flags=re.MULTILINE)
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove parameter values (basic approach)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        normalized = re.sub(r"'[^']*'", '?', normalized)
        
        return normalized.strip()
    
    def _get_query_type(self, query: str) -> QueryType:
        """Determine query type from SQL."""
        query_upper = query.strip().upper()
        
        for query_type in QueryType:
            if query_upper.startswith(query_type.value.upper()):
                return query_type
        
        return QueryType.SELECT  # Default
    
    async def log_slow_query(
        self,
        query: str,
        execution_time: float,
        plan: Optional[QueryPlan] = None
    ):
        """Log slow query for analysis."""
        slow_query = {
            "query": query,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "plan": plan
        }
        
        self._slow_query_log.append(slow_query)
        
        # Keep only last 1000 slow queries
        if len(self._slow_query_log) > 1000:
            self._slow_query_log.pop(0)
        
        logger.warning(
            f"Slow query detected ({execution_time:.2f}s): "
            f"{query[:100]}{'...' if len(query) > 100 else ''}"
        )
        
        if self.metrics:
            self.metrics.increment("db.query.slow")
            self.metrics.histogram(
                "db.query.slow_execution_time",
                execution_time
            )
    
    def get_slow_query_summary(self) -> Dict[str, Any]:
        """Get summary of slow queries."""
        if not self._slow_query_log:
            return {"count": 0, "queries": []}
        
        # Group by normalized query
        query_groups: Dict[str, List[Dict]] = {}
        
        for entry in self._slow_query_log:
            normalized = self._normalize_query(entry["query"])
            if normalized not in query_groups:
                query_groups[normalized] = []
            query_groups[normalized].append(entry)
        
        # Calculate statistics for each group
        summary = {
            "count": len(self._slow_query_log),
            "queries": []
        }
        
        for normalized, entries in query_groups.items():
            exec_times = [e["execution_time"] for e in entries]
            
            query_stats = {
                "query": entries[0]["query"],  # Sample query
                "count": len(entries),
                "avg_time": sum(exec_times) / len(exec_times),
                "max_time": max(exec_times),
                "min_time": min(exec_times),
                "last_seen": max(e["timestamp"] for e in entries)
            }
            
            summary["queries"].append(query_stats)
        
        # Sort by average execution time
        summary["queries"].sort(key=lambda x: x["avg_time"], reverse=True)
        
        return summary
    
    def clear_cache(self):
        """Clear query plan cache."""
        self._query_cache.clear()
        logger.info("Query plan cache cleared")