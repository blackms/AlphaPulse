"""Database index advisor and management."""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import text, MetaData
from sqlalchemy.ext.asyncio import AsyncSession

from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


@dataclass
class IndexInfo:
    """Information about a database index."""
    
    table_name: str
    index_name: str
    columns: List[str]
    is_unique: bool
    is_primary: bool
    index_type: str  # btree, hash, gin, gist
    size_bytes: int
    usage_count: int
    last_used: Optional[datetime]
    
    @property
    def size_mb(self) -> float:
        """Index size in megabytes."""
        return self.size_bytes / (1024 * 1024)


@dataclass
class IndexRecommendation:
    """Index recommendation from advisor."""
    
    table_name: str
    columns: List[str]
    index_type: str
    reason: str
    estimated_benefit: float  # 0-100 score
    estimated_size_mb: float
    affected_queries: List[str]
    priority: int  # 1-5, higher is more important
    
    @property
    def create_statement(self) -> str:
        """Generate CREATE INDEX statement."""
        columns_str = ", ".join(self.columns)
        index_name = f"idx_{self.table_name}_{'_'.join(self.columns)}"
        
        if self.index_type == "btree":
            return f"CREATE INDEX {index_name} ON {self.table_name} ({columns_str})"
        else:
            return f"CREATE INDEX {index_name} ON {self.table_name} USING {self.index_type} ({columns_str})"


@dataclass
class IndexUsageStats:
    """Index usage statistics."""
    
    index_name: str
    table_name: str
    index_scans: int
    index_size: int
    table_size: int
    last_scan: Optional[datetime]
    
    @property
    def usage_ratio(self) -> float:
        """Ratio of index scans to table size."""
        if self.table_size == 0:
            return 0
        return self.index_scans / self.table_size
    
    @property
    def size_ratio(self) -> float:
        """Ratio of index size to table size."""
        if self.table_size == 0:
            return 0
        return self.index_size / self.table_size


class IndexAdvisor:
    """Advises on database index creation and management."""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize index advisor."""
        self.metrics = metrics_collector
        
        # Cache
        self._existing_indexes: Dict[str, List[IndexInfo]] = {}
        self._index_usage: Dict[str, IndexUsageStats] = {}
        self._query_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Recommendations
        self._recommendations: List[IndexRecommendation] = []
        self._last_analysis: Optional[datetime] = None
        
    async def analyze_indexes(
        self,
        session: AsyncSession,
        tables: Optional[List[str]] = None
    ) -> List[IndexRecommendation]:
        """Analyze database and recommend indexes."""
        logger.info("Starting index analysis...")
        
        # Get existing indexes
        await self._load_existing_indexes(session, tables)
        
        # Get index usage statistics
        await self._load_index_usage(session, tables)
        
        # Analyze slow queries
        slow_queries = await self._get_slow_queries(session)
        
        # Generate recommendations
        recommendations = []
        
        # Check for missing indexes
        missing_indexes = await self._find_missing_indexes(session, slow_queries)
        recommendations.extend(missing_indexes)
        
        # Check for duplicate indexes
        duplicate_indexes = self._find_duplicate_indexes()
        recommendations.extend(duplicate_indexes)
        
        # Check for unused indexes
        unused_indexes = self._find_unused_indexes()
        recommendations.extend(unused_indexes)
        
        # Check for partial index opportunities
        partial_indexes = await self._find_partial_index_opportunities(session)
        recommendations.extend(partial_indexes)
        
        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        self._recommendations = recommendations
        self._last_analysis = datetime.utcnow()
        
        # Record metrics
        if self.metrics:
            self.metrics.gauge("db.index.recommendations", len(recommendations))
            self.metrics.gauge("db.index.existing", sum(len(idx) for idx in self._existing_indexes.values()))
        
        logger.info(f"Index analysis complete. Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    async def _load_existing_indexes(
        self,
        session: AsyncSession,
        tables: Optional[List[str]] = None
    ):
        """Load existing indexes from database."""
        try:
            query = """
                SELECT 
                    t.relname as table_name,
                    i.relname as index_name,
                    a.attname as column_name,
                    ix.indisunique as is_unique,
                    ix.indisprimary as is_primary,
                    am.amname as index_type,
                    pg_relation_size(i.oid) as size_bytes,
                    idx_scan as usage_count
                FROM pg_index ix
                JOIN pg_class t ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN pg_am am ON am.oid = i.relam
                LEFT JOIN pg_stat_user_indexes ui ON ui.indexrelid = i.oid
                LEFT JOIN pg_attribute a ON a.attrelid = t.oid 
                    AND a.attnum = ANY(ix.indkey)
                WHERE t.relkind = 'r'
                AND t.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """
            
            if tables:
                placeholders = ", ".join([f":table{i}" for i in range(len(tables))])
                query += f" AND t.relname IN ({placeholders})"
                params = {f"table{i}": table for i, table in enumerate(tables)}
            else:
                params = {}
            
            result = await session.execute(text(query), params)
            
            # Group by index
            index_data = defaultdict(lambda: {
                "columns": [],
                "table_name": None,
                "is_unique": False,
                "is_primary": False,
                "index_type": "btree",
                "size_bytes": 0,
                "usage_count": 0
            })
            
            for row in result:
                idx = index_data[row.index_name]
                idx["table_name"] = row.table_name
                idx["columns"].append(row.column_name)
                idx["is_unique"] = row.is_unique
                idx["is_primary"] = row.is_primary
                idx["index_type"] = row.index_type
                idx["size_bytes"] = row.size_bytes or 0
                idx["usage_count"] = row.usage_count or 0
            
            # Convert to IndexInfo objects
            self._existing_indexes.clear()
            
            for index_name, data in index_data.items():
                table_name = data["table_name"]
                if table_name not in self._existing_indexes:
                    self._existing_indexes[table_name] = []
                
                index_info = IndexInfo(
                    table_name=table_name,
                    index_name=index_name,
                    columns=data["columns"],
                    is_unique=data["is_unique"],
                    is_primary=data["is_primary"],
                    index_type=data["index_type"],
                    size_bytes=data["size_bytes"],
                    usage_count=data["usage_count"],
                    last_used=None  # Would need pg_stat_user_indexes timestamp
                )
                
                self._existing_indexes[table_name].append(index_info)
            
            logger.info(f"Loaded {sum(len(idx) for idx in self._existing_indexes.values())} existing indexes")
            
        except Exception as e:
            logger.error(f"Failed to load existing indexes: {e}")
    
    async def _load_index_usage(
        self,
        session: AsyncSession,
        tables: Optional[List[str]] = None
    ):
        """Load index usage statistics."""
        try:
            query = """
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    pg_relation_size(indexrelid) as index_size,
                    pg_relation_size(relid) as table_size
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
            """
            
            if tables:
                placeholders = ", ".join([f":table{i}" for i in range(len(tables))])
                query += f" AND tablename IN ({placeholders})"
                params = {f"table{i}": table for i, table in enumerate(tables)}
            else:
                params = {}
            
            result = await session.execute(text(query), params)
            
            self._index_usage.clear()
            
            for row in result:
                usage_stats = IndexUsageStats(
                    index_name=row.indexname,
                    table_name=row.tablename,
                    index_scans=row.idx_scan or 0,
                    index_size=row.index_size or 0,
                    table_size=row.table_size or 0,
                    last_scan=None
                )
                
                self._index_usage[row.indexname] = usage_stats
            
        except Exception as e:
            logger.error(f"Failed to load index usage: {e}")
    
    async def _get_slow_queries(self, session: AsyncSession) -> List[Dict]:
        """Get slow queries from pg_stat_statements."""
        try:
            # Check if pg_stat_statements is available
            check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                )
            """
            result = await session.execute(text(check_query))
            if not result.scalar():
                logger.warning("pg_stat_statements extension not available")
                return []
            
            # Get slow queries
            query = """
                SELECT
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows
                FROM pg_stat_statements
                WHERE mean_time > 100  -- Queries averaging > 100ms
                ORDER BY total_time DESC
                LIMIT 100
            """
            
            result = await session.execute(text(query))
            
            slow_queries = []
            for row in result:
                slow_queries.append({
                    "query": row.query,
                    "calls": row.calls,
                    "total_time": row.total_time,
                    "mean_time": row.mean_time,
                    "rows": row.rows
                })
            
            return slow_queries
            
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []
    
    async def _find_missing_indexes(
        self,
        session: AsyncSession,
        slow_queries: List[Dict]
    ) -> List[IndexRecommendation]:
        """Find missing indexes based on query patterns."""
        recommendations = []
        
        for query_info in slow_queries:
            query = query_info["query"]
            
            # Extract WHERE clause columns
            where_columns = self._extract_where_columns(query)
            
            # Extract JOIN columns
            join_columns = self._extract_join_columns(query)
            
            # Extract ORDER BY columns
            order_columns = self._extract_order_columns(query)
            
            # Check each table referenced in query
            tables = self._extract_table_names(query)
            
            for table in tables:
                existing = self._existing_indexes.get(table, [])
                existing_columns = set()
                for idx in existing:
                    existing_columns.update(idx.columns)
                
                # Check WHERE columns
                for col in where_columns.get(table, []):
                    if col not in existing_columns:
                        rec = IndexRecommendation(
                            table_name=table,
                            columns=[col],
                            index_type="btree",
                            reason=f"Column {col} used in WHERE clause of slow query",
                            estimated_benefit=70,
                            estimated_size_mb=10,  # Rough estimate
                            affected_queries=[query[:100]],
                            priority=4
                        )
                        recommendations.append(rec)
                
                # Check JOIN columns
                for col in join_columns.get(table, []):
                    if col not in existing_columns:
                        rec = IndexRecommendation(
                            table_name=table,
                            columns=[col],
                            index_type="btree",
                            reason=f"Column {col} used in JOIN of slow query",
                            estimated_benefit=80,
                            estimated_size_mb=10,
                            affected_queries=[query[:100]],
                            priority=5
                        )
                        recommendations.append(rec)
                
                # Check ORDER BY columns
                order_cols = order_columns.get(table, [])
                if order_cols and not any(
                    idx.columns[:len(order_cols)] == order_cols 
                    for idx in existing
                ):
                    rec = IndexRecommendation(
                        table_name=table,
                        columns=order_cols,
                        index_type="btree",
                        reason="Columns used in ORDER BY without matching index",
                        estimated_benefit=60,
                        estimated_size_mb=15,
                        affected_queries=[query[:100]],
                        priority=3
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _find_duplicate_indexes(self) -> List[IndexRecommendation]:
        """Find duplicate or redundant indexes."""
        recommendations = []
        
        for table, indexes in self._existing_indexes.items():
            # Check each pair of indexes
            for i in range(len(indexes)):
                for j in range(i + 1, len(indexes)):
                    idx1, idx2 = indexes[i], indexes[j]
                    
                    # Check if one index is prefix of another
                    if (idx1.columns[:len(idx2.columns)] == idx2.columns or
                        idx2.columns[:len(idx1.columns)] == idx1.columns):
                        
                        # Recommend dropping the smaller index
                        to_drop = idx1 if len(idx1.columns) < len(idx2.columns) else idx2
                        
                        rec = IndexRecommendation(
                            table_name=table,
                            columns=[],  # Empty for DROP recommendation
                            index_type="DROP",
                            reason=f"Index {to_drop.index_name} is redundant",
                            estimated_benefit=20,
                            estimated_size_mb=-to_drop.size_mb,  # Negative for space savings
                            affected_queries=[],
                            priority=2
                        )
                        recommendations.append(rec)
        
        return recommendations
    
    def _find_unused_indexes(self) -> List[IndexRecommendation]:
        """Find indexes that are never used."""
        recommendations = []
        
        for index_name, usage in self._index_usage.items():
            # Skip primary keys
            idx_info = None
            for indexes in self._existing_indexes.values():
                for idx in indexes:
                    if idx.index_name == index_name:
                        idx_info = idx
                        break
            
            if idx_info and idx_info.is_primary:
                continue
            
            # Check if index is unused
            if usage.index_scans == 0 and usage.index_size > 1024 * 1024:  # > 1MB
                rec = IndexRecommendation(
                    table_name=usage.table_name,
                    columns=[],  # Empty for DROP recommendation
                    index_type="DROP",
                    reason=f"Index {index_name} is never used",
                    estimated_benefit=10,
                    estimated_size_mb=-usage.index_size / (1024 * 1024),
                    affected_queries=[],
                    priority=1
                )
                recommendations.append(rec)
        
        return recommendations
    
    async def _find_partial_index_opportunities(
        self,
        session: AsyncSession
    ) -> List[IndexRecommendation]:
        """Find opportunities for partial indexes."""
        recommendations = []
        
        # This would require analyzing data distribution
        # For now, return empty list
        # In a full implementation, we would:
        # 1. Check for columns with low cardinality
        # 2. Analyze WHERE clauses for common filters
        # 3. Suggest partial indexes for frequently filtered values
        
        return recommendations
    
    def _extract_where_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in WHERE clause."""
        where_columns = defaultdict(list)
        
        # Simple regex patterns - would need more sophisticated parsing
        where_pattern = r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)'
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract column references (table.column or just column)
            col_pattern = r'(\w+)\.(\w+)\s*[=<>]|(\w+)\s*[=<>]'
            
            for match in re.finditer(col_pattern, where_clause):
                if match.group(1) and match.group(2):
                    # table.column format
                    where_columns[match.group(1)].append(match.group(2))
                elif match.group(3):
                    # Just column name - would need table inference
                    pass
        
        return dict(where_columns)
    
    def _extract_join_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in JOINs."""
        join_columns = defaultdict(list)
        
        # Pattern for JOIN conditions
        join_pattern = r'JOIN\s+(\w+)\s+(?:\w+\s+)?ON\s+(.*?)(?:JOIN|WHERE|GROUP|ORDER|$)'
        
        for match in re.finditer(join_pattern, query, re.IGNORECASE | re.DOTALL):
            table = match.group(1)
            condition = match.group(2)
            
            # Extract columns from join condition
            col_pattern = r'(\w+)\.(\w+)'
            for col_match in re.finditer(col_pattern, condition):
                join_columns[col_match.group(1)].append(col_match.group(2))
        
        return dict(join_columns)
    
    def _extract_order_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in ORDER BY."""
        order_columns = defaultdict(list)
        
        order_pattern = r'ORDER\s+BY\s+(.*?)(?:LIMIT|OFFSET|$)'
        order_match = re.search(order_pattern, query, re.IGNORECASE)
        
        if order_match:
            order_clause = order_match.group(1)
            
            # Extract columns
            col_pattern = r'(\w+)\.(\w+)|(\w+)(?:\s+(?:ASC|DESC))?'
            
            for match in re.finditer(col_pattern, order_clause):
                if match.group(1) and match.group(2):
                    # table.column format
                    order_columns[match.group(1)].append(match.group(2))
        
        return dict(order_columns)
    
    def _extract_table_names(self, query: str) -> Set[str]:
        """Extract table names from query."""
        tables = set()
        
        # FROM clause
        from_pattern = r'FROM\s+(\w+)'
        for match in re.finditer(from_pattern, query, re.IGNORECASE):
            tables.add(match.group(1))
        
        # JOIN clauses
        join_pattern = r'JOIN\s+(\w+)'
        for match in re.finditer(join_pattern, query, re.IGNORECASE):
            tables.add(match.group(1))
        
        return tables
    
    def get_recommendations_summary(self) -> Dict[str, Any]:
        """Get summary of recommendations."""
        if not self._recommendations:
            return {"message": "No analysis performed yet"}
        
        summary = {
            "last_analysis": self._last_analysis.isoformat() if self._last_analysis else None,
            "total_recommendations": len(self._recommendations),
            "by_priority": defaultdict(int),
            "by_type": defaultdict(int),
            "estimated_benefit": sum(r.estimated_benefit for r in self._recommendations),
            "estimated_size_change_mb": sum(r.estimated_size_mb for r in self._recommendations)
        }
        
        for rec in self._recommendations:
            summary["by_priority"][rec.priority] += 1
            summary["by_type"][rec.index_type] += 1
        
        return dict(summary)