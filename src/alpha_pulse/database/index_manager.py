"""Database index management and maintenance."""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector
from .index_advisor import IndexAdvisor, IndexRecommendation, IndexInfo

logger = get_logger(__name__)


class IndexOperation(Enum):
    """Types of index operations."""
    
    CREATE = "create"
    DROP = "drop"
    REBUILD = "rebuild"
    REINDEX = "reindex"
    ANALYZE = "analyze"


class IndexManager:
    """Manages database indexes."""
    
    def __init__(
        self,
        index_advisor: IndexAdvisor,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize index manager."""
        self.advisor = index_advisor
        self.metrics = metrics_collector
        
        # Operation tracking
        self._operations_in_progress: Set[str] = set()
        self._operation_history: List[Dict] = []
        
        # Maintenance scheduling
        self._maintenance_task: Optional[asyncio.Task] = None
        self._last_maintenance: Optional[datetime] = None
        
    async def create_index(
        self,
        session: AsyncSession,
        recommendation: IndexRecommendation,
        concurrent: bool = True
    ) -> bool:
        """Create an index based on recommendation."""
        index_key = f"{recommendation.table_name}_{','.join(recommendation.columns)}"
        
        if index_key in self._operations_in_progress:
            logger.warning(f"Index creation already in progress for {index_key}")
            return False
        
        self._operations_in_progress.add(index_key)
        
        try:
            # Generate index name
            index_name = f"idx_{recommendation.table_name}_{'_'.join(recommendation.columns)}"
            
            # Build CREATE INDEX statement
            columns_str = ", ".join(recommendation.columns)
            
            if concurrent:
                create_sql = f"""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
                    ON {recommendation.table_name} ({columns_str})
                """
            else:
                create_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {recommendation.table_name} ({columns_str})
                """
            
            # Add index type if not btree
            if recommendation.index_type != "btree":
                create_sql = create_sql.replace(
                    f"ON {recommendation.table_name}",
                    f"ON {recommendation.table_name} USING {recommendation.index_type}"
                )
            
            logger.info(f"Creating index: {index_name}")
            start_time = datetime.utcnow()
            
            # Execute index creation
            await session.execute(text(create_sql))
            await session.commit()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Record operation
            self._record_operation(
                IndexOperation.CREATE,
                index_name,
                recommendation.table_name,
                duration,
                True
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.increment("db.index.created")
                self.metrics.histogram("db.index.create_duration", duration)
            
            logger.info(f"Index {index_name} created successfully in {duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            self._record_operation(
                IndexOperation.CREATE,
                index_key,
                recommendation.table_name,
                0,
                False,
                str(e)
            )
            
            if self.metrics:
                self.metrics.increment("db.index.create_failed")
            
            return False
            
        finally:
            self._operations_in_progress.discard(index_key)
    
    async def drop_index(
        self,
        session: AsyncSession,
        index_name: str,
        concurrent: bool = True
    ) -> bool:
        """Drop an index."""
        if index_name in self._operations_in_progress:
            logger.warning(f"Operation already in progress for index {index_name}")
            return False
        
        self._operations_in_progress.add(index_name)
        
        try:
            if concurrent:
                drop_sql = f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}"
            else:
                drop_sql = f"DROP INDEX IF EXISTS {index_name}"
            
            logger.info(f"Dropping index: {index_name}")
            start_time = datetime.utcnow()
            
            await session.execute(text(drop_sql))
            await session.commit()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Record operation
            self._record_operation(
                IndexOperation.DROP,
                index_name,
                "",
                duration,
                True
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.increment("db.index.dropped")
                self.metrics.histogram("db.index.drop_duration", duration)
            
            logger.info(f"Index {index_name} dropped successfully in {duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            self._record_operation(
                IndexOperation.DROP,
                index_name,
                "",
                0,
                False,
                str(e)
            )
            
            if self.metrics:
                self.metrics.increment("db.index.drop_failed")
            
            return False
            
        finally:
            self._operations_in_progress.discard(index_name)
    
    async def rebuild_index(
        self,
        session: AsyncSession,
        index_name: str
    ) -> bool:
        """Rebuild an index to reduce bloat."""
        if index_name in self._operations_in_progress:
            logger.warning(f"Operation already in progress for index {index_name}")
            return False
        
        self._operations_in_progress.add(index_name)
        
        try:
            # PostgreSQL uses REINDEX
            reindex_sql = f"REINDEX INDEX CONCURRENTLY {index_name}"
            
            logger.info(f"Rebuilding index: {index_name}")
            start_time = datetime.utcnow()
            
            await session.execute(text(reindex_sql))
            await session.commit()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Record operation
            self._record_operation(
                IndexOperation.REBUILD,
                index_name,
                "",
                duration,
                True
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.increment("db.index.rebuilt")
                self.metrics.histogram("db.index.rebuild_duration", duration)
            
            logger.info(f"Index {index_name} rebuilt successfully in {duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index {index_name}: {e}")
            self._record_operation(
                IndexOperation.REBUILD,
                index_name,
                "",
                0,
                False,
                str(e)
            )
            
            if self.metrics:
                self.metrics.increment("db.index.rebuild_failed")
            
            return False
            
        finally:
            self._operations_in_progress.discard(index_name)
    
    async def analyze_table(
        self,
        session: AsyncSession,
        table_name: str
    ) -> bool:
        """Update table statistics for query planner."""
        try:
            analyze_sql = f"ANALYZE {table_name}"
            
            logger.info(f"Analyzing table: {table_name}")
            start_time = datetime.utcnow()
            
            await session.execute(text(analyze_sql))
            await session.commit()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Record operation
            self._record_operation(
                IndexOperation.ANALYZE,
                table_name,
                table_name,
                duration,
                True
            )
            
            logger.info(f"Table {table_name} analyzed successfully in {duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
            return False
    
    async def apply_recommendations(
        self,
        session: AsyncSession,
        recommendations: List[IndexRecommendation],
        max_concurrent: int = 2,
        priority_threshold: int = 3
    ) -> Dict[str, int]:
        """Apply index recommendations."""
        # Filter by priority
        to_apply = [r for r in recommendations if r.priority >= priority_threshold]
        
        logger.info(f"Applying {len(to_apply)} index recommendations")
        
        results = {
            "created": 0,
            "dropped": 0,
            "failed": 0
        }
        
        # Group by operation type
        creates = [r for r in to_apply if r.index_type != "DROP"]
        drops = [r for r in to_apply if r.index_type == "DROP"]
        
        # Apply drops first (to free space)
        for rec in drops:
            # Extract index name from recommendation
            # This is a simplified approach - would need proper parsing
            success = await self.drop_index(session, rec.reason.split()[1])
            if success:
                results["dropped"] += 1
            else:
                results["failed"] += 1
        
        # Apply creates with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def create_with_limit(rec):
            async with semaphore:
                success = await self.create_index(session, rec)
                if success:
                    results["created"] += 1
                else:
                    results["failed"] += 1
        
        # Create indexes concurrently
        await asyncio.gather(*[create_with_limit(rec) for rec in creates])
        
        return results
    
    async def get_index_statistics(
        self,
        session: AsyncSession,
        table_name: Optional[str] = None
    ) -> List[Dict]:
        """Get detailed index statistics."""
        try:
            query = """
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size,
                    pg_stat_get_numscans(indexrelid) as scans_since_reset
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
            """
            
            if table_name:
                query += " AND tablename = :table_name"
                params = {"table_name": table_name}
            else:
                params = {}
            
            result = await session.execute(text(query), params)
            
            stats = []
            for row in result:
                stats.append({
                    "schema": row.schemaname,
                    "table": row.tablename,
                    "index": row.indexname,
                    "scans": row.idx_scan,
                    "tuples_read": row.idx_tup_read,
                    "tuples_fetched": row.idx_tup_fetch,
                    "size": row.size,
                    "efficiency": (
                        row.idx_tup_fetch / row.idx_tup_read 
                        if row.idx_tup_read > 0 else 0
                    )
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            return []
    
    async def check_index_bloat(
        self,
        session: AsyncSession,
        bloat_threshold: float = 30.0
    ) -> List[Dict]:
        """Check for bloated indexes."""
        try:
            # Query to estimate index bloat
            query = """
                WITH btree_index_atts AS (
                    SELECT 
                        nspname,
                        indexclass.relname as index_name,
                        indexclass.reltuples,
                        indexclass.relpages,
                        tableclass.relname as tablename,
                        regexp_split_to_table(indkey::text, ' ')::smallint AS attnum,
                        indexrelid as index_oid
                    FROM pg_index
                    JOIN pg_class indexclass ON pg_index.indexrelid = indexclass.oid
                    JOIN pg_class tableclass ON pg_index.indrelid = tableclass.oid
                    JOIN pg_namespace ON pg_namespace.oid = indexclass.relnamespace
                    WHERE indexclass.relam = (SELECT oid FROM pg_am WHERE amname = 'btree')
                    AND nspname NOT IN ('pg_catalog', 'information_schema')
                )
                SELECT
                    index_name,
                    tablename,
                    pg_size_pretty(pg_relation_size(index_oid)) as index_size,
                    CASE WHEN relpages > 0
                        THEN round(100.0 * relpages / (reltuples * 0.1))::numeric
                        ELSE 0
                    END as bloat_ratio
                FROM btree_index_atts
                GROUP BY index_name, tablename, index_oid, relpages, reltuples
                HAVING CASE WHEN relpages > 0
                    THEN round(100.0 * relpages / (reltuples * 0.1))::numeric
                    ELSE 0
                END > :threshold
            """
            
            result = await session.execute(
                text(query),
                {"threshold": bloat_threshold}
            )
            
            bloated_indexes = []
            for row in result:
                bloated_indexes.append({
                    "index_name": row.index_name,
                    "table_name": row.tablename,
                    "size": row.index_size,
                    "bloat_ratio": float(row.bloat_ratio)
                })
            
            if bloated_indexes:
                logger.warning(f"Found {len(bloated_indexes)} bloated indexes")
            
            return bloated_indexes
            
        except Exception as e:
            logger.error(f"Failed to check index bloat: {e}")
            return []
    
    async def start_maintenance(
        self,
        session: AsyncSession,
        check_interval: int = 3600,  # 1 hour
        bloat_threshold: float = 30.0
    ):
        """Start automatic index maintenance."""
        if self._maintenance_task:
            logger.warning("Maintenance already running")
            return
        
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(session, check_interval, bloat_threshold)
        )
        logger.info("Started index maintenance")
    
    async def stop_maintenance(self):
        """Stop automatic index maintenance."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
        logger.info("Stopped index maintenance")
    
    async def _maintenance_loop(
        self,
        session: AsyncSession,
        check_interval: int,
        bloat_threshold: float
    ):
        """Background maintenance loop."""
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                logger.info("Running index maintenance check")
                
                # Check for bloated indexes
                bloated = await self.check_index_bloat(session, bloat_threshold)
                
                # Rebuild bloated indexes
                for idx in bloated:
                    if idx["bloat_ratio"] > bloat_threshold * 2:
                        # Very bloated - rebuild
                        await self.rebuild_index(session, idx["index_name"])
                
                # Run index analysis
                recommendations = await self.advisor.analyze_indexes(session)
                
                # Apply high-priority recommendations
                if recommendations:
                    high_priority = [r for r in recommendations if r.priority >= 4]
                    if high_priority:
                        await self.apply_recommendations(
                            session, 
                            high_priority,
                            max_concurrent=1
                        )
                
                self._last_maintenance = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(check_interval)
    
    def _record_operation(
        self,
        operation: IndexOperation,
        index_name: str,
        table_name: str,
        duration: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record index operation in history."""
        record = {
            "timestamp": datetime.utcnow(),
            "operation": operation.value,
            "index_name": index_name,
            "table_name": table_name,
            "duration": duration,
            "success": success,
            "error": error
        }
        
        self._operation_history.append(record)
        
        # Keep only last 1000 operations
        if len(self._operation_history) > 1000:
            self._operation_history.pop(0)
    
    def get_operation_history(self) -> List[Dict]:
        """Get recent operation history."""
        return [
            {
                "timestamp": op["timestamp"].isoformat(),
                "operation": op["operation"],
                "index_name": op["index_name"],
                "table_name": op["table_name"],
                "duration": op["duration"],
                "success": op["success"],
                "error": op["error"]
            }
            for op in self._operation_history[-100:]  # Last 100 operations
        ]