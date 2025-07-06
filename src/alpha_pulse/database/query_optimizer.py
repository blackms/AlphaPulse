"""Query optimization and cost estimation."""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.database_config import QueryOptimizationConfig
from ..utils.logging_utils import get_logger
from .query_analyzer import QueryAnalyzer, QueryType

logger = get_logger(__name__)


class OptimizationType(Enum):
    """Types of query optimizations."""
    
    INDEX_HINT = "index_hint"
    JOIN_ORDER = "join_order"
    SUBQUERY_REWRITE = "subquery_rewrite"
    PARTITION_PRUNING = "partition_pruning"
    MATERIALIZED_VIEW = "materialized_view"
    QUERY_REWRITE = "query_rewrite"


@dataclass
class QueryCost:
    """Estimated query cost."""
    
    cpu_cost: float
    io_cost: float
    network_cost: float
    memory_cost: float
    total_cost: float
    estimated_rows: int
    estimated_time: float  # milliseconds
    
    @classmethod
    def from_plan(cls, plan: Dict) -> "QueryCost":
        """Create cost estimate from execution plan."""
        total_cost = plan.get("Total Cost", 0)
        rows = plan.get("Plan Rows", 0)
        
        # Rough estimates based on plan
        return cls(
            cpu_cost=total_cost * 0.3,
            io_cost=total_cost * 0.5,
            network_cost=total_cost * 0.1,
            memory_cost=total_cost * 0.1,
            total_cost=total_cost,
            estimated_rows=rows,
            estimated_time=total_cost * 0.1  # Very rough estimate
        )


@dataclass
class OptimizationSuggestion:
    """Query optimization suggestion."""
    
    optimization_type: OptimizationType
    description: str
    estimated_improvement: float  # Percentage
    implementation: str
    priority: int  # 1-5, higher is more important


class QueryOptimizer:
    """Optimizes database queries."""
    
    def __init__(
        self,
        config: QueryOptimizationConfig,
        query_analyzer: QueryAnalyzer
    ):
        """Initialize query optimizer."""
        self.config = config
        self.analyzer = query_analyzer
        
        # Cache for query statistics
        self._query_stats: Dict[str, Dict] = {}
        self._table_stats: Dict[str, Dict] = {}
        
        # Prepared statement cache
        self._prepared_statements: Dict[str, str] = {}
        
    async def estimate_cost(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict] = None
    ) -> QueryCost:
        """Estimate query execution cost."""
        try:
            # Get execution plan
            plan = await self.analyzer.analyze_query(session, query, params)
            
            # Extract cost information
            cost = QueryCost.from_plan({
                "Total Cost": plan.total_cost,
                "Plan Rows": plan.rows
            })
            
            # Adjust based on query type
            query_type = self.analyzer._get_query_type(query)
            
            if query_type == QueryType.SELECT:
                # Check for joins
                join_count = query.upper().count(" JOIN ")
                cost.cpu_cost *= (1 + 0.2 * join_count)
                
                # Check for aggregations
                if any(func in query.upper() for func in ["COUNT", "SUM", "AVG", "GROUP BY"]):
                    cost.cpu_cost *= 1.5
                    cost.memory_cost *= 2
                
            elif query_type in [QueryType.INSERT, QueryType.UPDATE]:
                # Write operations have higher I/O cost
                cost.io_cost *= 2
                
                # Check for triggers/constraints
                cost.cpu_cost *= 1.3
            
            return cost
            
        except Exception as e:
            logger.error(f"Failed to estimate query cost: {e}")
            # Return default high cost on error
            return QueryCost(
                cpu_cost=1000,
                io_cost=1000,
                network_cost=100,
                memory_cost=500,
                total_cost=2600,
                estimated_rows=0,
                estimated_time=1000
            )
    
    async def optimize_query(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict] = None
    ) -> Tuple[str, List[OptimizationSuggestion]]:
        """Optimize query and return improved version with suggestions."""
        suggestions = []
        optimized_query = query
        
        # Analyze current query
        plan = await self.analyzer.analyze_query(session, query, params)
        
        # Apply various optimization techniques
        if self.config.enable_query_hints:
            optimized_query, hint_suggestions = await self._apply_index_hints(
                session, optimized_query, plan
            )
            suggestions.extend(hint_suggestions)
        
        if self.config.enable_join_optimization:
            optimized_query, join_suggestions = self._optimize_joins(
                optimized_query, plan
            )
            suggestions.extend(join_suggestions)
        
        if self.config.enable_subquery_optimization:
            optimized_query, subquery_suggestions = self._optimize_subqueries(
                optimized_query
            )
            suggestions.extend(subquery_suggestions)
        
        # Check for query rewrite opportunities
        rewrite_suggestions = self._suggest_query_rewrites(query, plan)
        suggestions.extend(rewrite_suggestions)
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        
        return optimized_query, suggestions
    
    async def _apply_index_hints(
        self,
        session: AsyncSession,
        query: str,
        plan: Any
    ) -> Tuple[str, List[OptimizationSuggestion]]:
        """Apply index hints to query."""
        suggestions = []
        
        # Check for tables with sequential scans
        seq_scan_tables = []
        for node in plan.node_details:
            if (node.get("node_type") == "Seq Scan" and 
                node.get("relation") and 
                node.get("rows", 0) > 1000):
                seq_scan_tables.append(node["relation"])
        
        if not seq_scan_tables:
            return query, suggestions
        
        # Get available indexes for these tables
        for table_name in seq_scan_tables:
            indexes = await self._get_table_indexes(session, table_name)
            
            if indexes:
                # Suggest using available indexes
                suggestion = OptimizationSuggestion(
                    optimization_type=OptimizationType.INDEX_HINT,
                    description=f"Use index on table {table_name}",
                    estimated_improvement=30,
                    implementation=f"/*+ INDEX({table_name} {indexes[0]['name']}) */",
                    priority=4
                )
                suggestions.append(suggestion)
        
        return query, suggestions
    
    def _optimize_joins(
        self,
        query: str,
        plan: Any
    ) -> Tuple[str, List[OptimizationSuggestion]]:
        """Optimize join order and methods."""
        suggestions = []
        
        # Check for nested loop joins with high cost
        nested_loops = [
            node for node in plan.node_details
            if node.get("node_type") == "Nested Loop" and 
            node.get("cost", 0) > 1000
        ]
        
        if nested_loops:
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.JOIN_ORDER,
                description="Reorder joins to reduce nested loops",
                estimated_improvement=40,
                implementation="Consider hash joins for large datasets",
                priority=5
            )
            suggestions.append(suggestion)
        
        # Check for cross joins
        if " CROSS JOIN " in query.upper():
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.JOIN_ORDER,
                description="Replace CROSS JOIN with proper join conditions",
                estimated_improvement=60,
                implementation="Add join conditions to avoid cartesian product",
                priority=5
            )
            suggestions.append(suggestion)
        
        return query, suggestions
    
    def _optimize_subqueries(self, query: str) -> Tuple[str, List[OptimizationSuggestion]]:
        """Optimize subqueries."""
        suggestions = []
        
        # Check for correlated subqueries
        if re.search(r'WHERE.*EXISTS\s*\(|WHERE.*IN\s*\(.*SELECT', query, re.IGNORECASE):
            # Check if it's likely correlated
            if query.count("SELECT") > 1:
                suggestion = OptimizationSuggestion(
                    optimization_type=OptimizationType.SUBQUERY_REWRITE,
                    description="Convert correlated subquery to JOIN",
                    estimated_improvement=50,
                    implementation="Use JOIN instead of EXISTS/IN subquery",
                    priority=4
                )
                suggestions.append(suggestion)
        
        # Check for scalar subqueries in SELECT
        if re.search(r'SELECT.*\(SELECT', query, re.IGNORECASE):
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.SUBQUERY_REWRITE,
                description="Move scalar subquery to JOIN",
                estimated_improvement=30,
                implementation="Use LEFT JOIN with aggregation",
                priority=3
            )
            suggestions.append(suggestion)
        
        return query, suggestions
    
    def _suggest_query_rewrites(self, query: str, plan: Any) -> List[OptimizationSuggestion]:
        """Suggest query rewrites."""
        suggestions = []
        
        # Check for SELECT *
        if "SELECT *" in query.upper():
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.QUERY_REWRITE,
                description="Replace SELECT * with specific columns",
                estimated_improvement=20,
                implementation="List only required columns",
                priority=2
            )
            suggestions.append(suggestion)
        
        # Check for DISTINCT on large result sets
        if "DISTINCT" in query.upper() and plan.rows > 10000:
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.QUERY_REWRITE,
                description="Consider GROUP BY instead of DISTINCT",
                estimated_improvement=15,
                implementation="Use GROUP BY for better performance",
                priority=2
            )
            suggestions.append(suggestion)
        
        # Check for OR conditions that could use UNION
        or_count = query.upper().count(" OR ")
        if or_count > 3:
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.QUERY_REWRITE,
                description="Convert multiple OR conditions to UNION",
                estimated_improvement=25,
                implementation="Split query with UNION for index usage",
                priority=3
            )
            suggestions.append(suggestion)
        
        # Check for functions on indexed columns
        if re.search(r'WHERE.*\w+\([^)]+\)\s*=', query, re.IGNORECASE):
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.QUERY_REWRITE,
                description="Avoid functions on indexed columns",
                estimated_improvement=35,
                implementation="Rewrite to allow index usage",
                priority=4
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _get_table_indexes(
        self,
        session: AsyncSession,
        table_name: str
    ) -> List[Dict]:
        """Get indexes for a table."""
        try:
            result = await session.execute(
                text("""
                    SELECT 
                        indexname as name,
                        indexdef as definition
                    FROM pg_indexes
                    WHERE tablename = :table_name
                    AND schemaname = 'public'
                """),
                {"table_name": table_name}
            )
            
            return [
                {"name": row.name, "definition": row.definition}
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Failed to get indexes for {table_name}: {e}")
            return []
    
    async def prepare_statement(
        self,
        session: AsyncSession,
        name: str,
        query: str
    ) -> bool:
        """Prepare a statement for repeated execution."""
        if not self.config.enable_prepared_statements:
            return False
        
        try:
            # PostgreSQL PREPARE syntax
            prepare_query = f"PREPARE {name} AS {query}"
            await session.execute(text(prepare_query))
            
            self._prepared_statements[name] = query
            logger.info(f"Prepared statement '{name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare statement '{name}': {e}")
            return False
    
    async def execute_prepared(
        self,
        session: AsyncSession,
        name: str,
        params: Optional[List] = None
    ) -> Any:
        """Execute a prepared statement."""
        if name not in self._prepared_statements:
            raise ValueError(f"Prepared statement '{name}' not found")
        
        try:
            # PostgreSQL EXECUTE syntax
            if params:
                param_placeholders = ", ".join([f"${i+1}" for i in range(len(params))])
                execute_query = f"EXECUTE {name}({param_placeholders})"
                result = await session.execute(text(execute_query), params)
            else:
                execute_query = f"EXECUTE {name}"
                result = await session.execute(text(execute_query))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute prepared statement '{name}': {e}")
            raise
    
    async def update_table_statistics(
        self,
        session: AsyncSession,
        table_name: Optional[str] = None
    ):
        """Update table statistics for query planner."""
        try:
            if table_name:
                await session.execute(text(f"ANALYZE {table_name}"))
                logger.info(f"Updated statistics for table {table_name}")
            else:
                await session.execute(text("ANALYZE"))
                logger.info("Updated statistics for all tables")
                
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activities."""
        return {
            "prepared_statements": len(self._prepared_statements),
            "cached_queries": len(self._query_stats),
            "analyzed_tables": len(self._table_stats),
            "config": {
                "hints_enabled": self.config.enable_query_hints,
                "join_optimization": self.config.enable_join_optimization,
                "subquery_optimization": self.config.enable_subquery_optimization
            }
        }