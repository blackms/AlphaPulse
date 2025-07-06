"""Database table partitioning strategies."""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class PartitionType(Enum):
    """Types of table partitioning."""
    
    RANGE = "range"
    LIST = "list"
    HASH = "hash"
    COMPOSITE = "composite"


class PartitionInterval(Enum):
    """Time intervals for range partitioning."""
    
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class PartitionInfo:
    """Information about a table partition."""
    
    parent_table: str
    partition_name: str
    partition_type: PartitionType
    constraint_def: str
    size_bytes: int
    row_count: int
    created_at: datetime
    
    @property
    def size_mb(self) -> float:
        """Partition size in megabytes."""
        return self.size_bytes / (1024 * 1024)


@dataclass
class PartitionStrategy:
    """Partitioning strategy for a table."""
    
    table_name: str
    partition_column: str
    partition_type: PartitionType
    interval: Optional[PartitionInterval] = None
    retention_days: Optional[int] = None
    pre_create_days: int = 7  # Pre-create partitions for future
    
    def get_partition_name(self, value: Any) -> str:
        """Generate partition name based on value."""
        if self.partition_type == PartitionType.RANGE and self.interval:
            if isinstance(value, datetime):
                if self.interval == PartitionInterval.DAILY:
                    return f"{self.table_name}_{value.strftime('%Y%m%d')}"
                elif self.interval == PartitionInterval.MONTHLY:
                    return f"{self.table_name}_{value.strftime('%Y%m')}"
                elif self.interval == PartitionInterval.YEARLY:
                    return f"{self.table_name}_{value.strftime('%Y')}"
        
        return f"{self.table_name}_{str(value).replace('-', '_')}"


class PartitionManager:
    """Manages database table partitioning."""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize partition manager."""
        self.metrics = metrics_collector
        
        # Partition tracking
        self._partitions: Dict[str, List[PartitionInfo]] = {}
        self._strategies: Dict[str, PartitionStrategy] = {}
        
        # Maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        
        # Default strategies for common tables
        self._default_strategies = {
            "market_data": PartitionStrategy(
                table_name="market_data",
                partition_column="timestamp",
                partition_type=PartitionType.RANGE,
                interval=PartitionInterval.DAILY,
                retention_days=90
            ),
            "trades": PartitionStrategy(
                table_name="trades",
                partition_column="created_at",
                partition_type=PartitionType.RANGE,
                interval=PartitionInterval.MONTHLY,
                retention_days=365
            ),
            "signals": PartitionStrategy(
                table_name="signals",
                partition_column="timestamp",
                partition_type=PartitionType.RANGE,
                interval=PartitionInterval.DAILY,
                retention_days=30
            ),
            "logs": PartitionStrategy(
                table_name="logs",
                partition_column="created_at",
                partition_type=PartitionType.RANGE,
                interval=PartitionInterval.DAILY,
                retention_days=7
            )
        }
    
    async def setup_partitioning(
        self,
        session: AsyncSession,
        table_name: str,
        strategy: Optional[PartitionStrategy] = None
    ) -> bool:
        """Set up partitioning for a table."""
        try:
            # Use provided strategy or default
            if strategy:
                self._strategies[table_name] = strategy
            elif table_name in self._default_strategies:
                strategy = self._default_strategies[table_name]
                self._strategies[table_name] = strategy
            else:
                logger.error(f"No partitioning strategy for table {table_name}")
                return False
            
            # Check if table exists and is not already partitioned
            check_query = """
                SELECT 
                    c.relname,
                    c.relkind,
                    p.partstrat
                FROM pg_class c
                LEFT JOIN pg_partitioned_table p ON c.oid = p.partrelid
                WHERE c.relname = :table_name
                AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """
            
            result = await session.execute(
                text(check_query),
                {"table_name": table_name}
            )
            row = result.first()
            
            if not row:
                logger.error(f"Table {table_name} does not exist")
                return False
            
            if row.partstrat:
                logger.info(f"Table {table_name} is already partitioned")
                return True
            
            # Convert existing table to partitioned table
            # This is complex and requires careful planning
            logger.info(f"Converting {table_name} to partitioned table")
            
            # For new tables, we would create them as partitioned
            # For existing tables, we need to:
            # 1. Create new partitioned table
            # 2. Copy data
            # 3. Swap tables
            # 4. Drop old table
            
            # This is a simplified version for new tables
            if strategy.partition_type == PartitionType.RANGE:
                await self._create_range_partitioned_table(
                    session, table_name, strategy
                )
            
            # Create initial partitions
            await self._create_future_partitions(session, table_name, strategy)
            
            logger.info(f"Partitioning setup complete for {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup partitioning for {table_name}: {e}")
            return False
    
    async def _create_range_partitioned_table(
        self,
        session: AsyncSession,
        table_name: str,
        strategy: PartitionStrategy
    ):
        """Create a range-partitioned table."""
        # This would need the actual table schema
        # For demonstration, using a simple structure
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name}_partitioned (
                LIKE {table_name} INCLUDING ALL
            ) PARTITION BY RANGE ({strategy.partition_column})
        """
        
        await session.execute(text(create_sql))
        await session.commit()
    
    async def create_partition(
        self,
        session: AsyncSession,
        table_name: str,
        start_value: Any,
        end_value: Any
    ) -> bool:
        """Create a single partition."""
        try:
            strategy = self._strategies.get(table_name)
            if not strategy:
                logger.error(f"No strategy found for table {table_name}")
                return False
            
            partition_name = strategy.get_partition_name(start_value)
            
            # Check if partition already exists
            check_query = """
                SELECT 1 FROM pg_class 
                WHERE relname = :partition_name
                AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """
            
            result = await session.execute(
                text(check_query),
                {"partition_name": partition_name}
            )
            
            if result.first():
                logger.debug(f"Partition {partition_name} already exists")
                return True
            
            # Create partition
            if strategy.partition_type == PartitionType.RANGE:
                create_sql = f"""
                    CREATE TABLE {partition_name} 
                    PARTITION OF {table_name}
                    FOR VALUES FROM ('{start_value}') TO ('{end_value}')
                """
            else:
                # Other partition types would have different syntax
                logger.error(f"Unsupported partition type: {strategy.partition_type}")
                return False
            
            logger.info(f"Creating partition: {partition_name}")
            await session.execute(text(create_sql))
            await session.commit()
            
            # Create indexes on partition
            await self._create_partition_indexes(session, table_name, partition_name)
            
            # Record metrics
            if self.metrics:
                self.metrics.increment("db.partition.created")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create partition: {e}")
            return False
    
    async def _create_partition_indexes(
        self,
        session: AsyncSession,
        parent_table: str,
        partition_name: str
    ):
        """Create indexes on a partition matching parent table."""
        try:
            # Get indexes from parent table
            index_query = """
                SELECT 
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = :parent_table
                AND schemaname = 'public'
            """
            
            result = await session.execute(
                text(index_query),
                {"parent_table": parent_table}
            )
            
            for row in result:
                # Modify index definition for partition
                index_def = row.indexdef.replace(
                    f"ON {parent_table}",
                    f"ON {partition_name}"
                ).replace(
                    f"INDEX {row.indexname}",
                    f"INDEX {row.indexname}_{partition_name}"
                )
                
                await session.execute(text(index_def))
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Failed to create partition indexes: {e}")
    
    async def drop_partition(
        self,
        session: AsyncSession,
        partition_name: str
    ) -> bool:
        """Drop a partition."""
        try:
            drop_sql = f"DROP TABLE IF EXISTS {partition_name}"
            
            logger.info(f"Dropping partition: {partition_name}")
            await session.execute(text(drop_sql))
            await session.commit()
            
            # Record metrics
            if self.metrics:
                self.metrics.increment("db.partition.dropped")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop partition {partition_name}: {e}")
            return False
    
    async def _create_future_partitions(
        self,
        session: AsyncSession,
        table_name: str,
        strategy: PartitionStrategy
    ):
        """Create partitions for future dates."""
        if strategy.partition_type != PartitionType.RANGE or not strategy.interval:
            return
        
        now = datetime.utcnow()
        
        if strategy.interval == PartitionInterval.DAILY:
            for i in range(strategy.pre_create_days):
                date = now + timedelta(days=i)
                start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=1)
                
                await self.create_partition(
                    session, table_name, start_date, end_date
                )
        
        elif strategy.interval == PartitionInterval.MONTHLY:
            # Create partitions for next few months
            for i in range(3):  # Next 3 months
                if now.month + i > 12:
                    year = now.year + 1
                    month = (now.month + i) % 12
                else:
                    year = now.year
                    month = now.month + i
                
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1)
                else:
                    end_date = datetime(year, month + 1, 1)
                
                await self.create_partition(
                    session, table_name, start_date, end_date
                )
    
    async def _cleanup_old_partitions(
        self,
        session: AsyncSession,
        table_name: str,
        strategy: PartitionStrategy
    ):
        """Remove old partitions based on retention policy."""
        if not strategy.retention_days:
            return
        
        cutoff_date = datetime.utcnow() - timedelta(days=strategy.retention_days)
        
        # Get partitions older than cutoff
        query = """
            SELECT 
                c.relname as partition_name,
                pg_get_expr(c.relpartbound, c.oid) as constraint_def
            FROM pg_class c
            JOIN pg_inherits i ON c.oid = i.inhrelid
            JOIN pg_class p ON i.inhparent = p.oid
            WHERE p.relname = :table_name
            AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        """
        
        result = await session.execute(
            text(query),
            {"table_name": table_name}
        )
        
        for row in result:
            # Parse constraint to get date range
            # This is simplified - would need proper parsing
            if str(cutoff_date.date()) in row.constraint_def:
                await self.drop_partition(session, row.partition_name)
    
    async def get_partition_info(
        self,
        session: AsyncSession,
        table_name: str
    ) -> List[PartitionInfo]:
        """Get information about table partitions."""
        try:
            query = """
                SELECT 
                    p.relname as parent_table,
                    c.relname as partition_name,
                    pg_get_expr(c.relpartbound, c.oid) as constraint_def,
                    pg_relation_size(c.oid) as size_bytes,
                    c.reltuples as row_count,
                    s.n_tup_ins as inserts,
                    s.n_tup_upd as updates,
                    s.n_tup_del as deletes
                FROM pg_class c
                JOIN pg_inherits i ON c.oid = i.inhrelid
                JOIN pg_class p ON i.inhparent = p.oid
                LEFT JOIN pg_stat_user_tables s ON c.oid = s.relid
                WHERE p.relname = :table_name
                AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
                ORDER BY c.relname
            """
            
            result = await session.execute(
                text(query),
                {"table_name": table_name}
            )
            
            partitions = []
            for row in result:
                partition = PartitionInfo(
                    parent_table=row.parent_table,
                    partition_name=row.partition_name,
                    partition_type=PartitionType.RANGE,  # Would need to determine actual type
                    constraint_def=row.constraint_def,
                    size_bytes=row.size_bytes or 0,
                    row_count=int(row.row_count or 0),
                    created_at=datetime.utcnow()  # Would need actual creation time
                )
                partitions.append(partition)
            
            self._partitions[table_name] = partitions
            return partitions
            
        except Exception as e:
            logger.error(f"Failed to get partition info for {table_name}: {e}")
            return []
    
    async def analyze_partition_usage(
        self,
        session: AsyncSession,
        table_name: str
    ) -> Dict[str, Any]:
        """Analyze partition usage patterns."""
        partitions = await self.get_partition_info(session, table_name)
        
        if not partitions:
            return {"message": "No partitions found"}
        
        # Calculate statistics
        total_size = sum(p.size_bytes for p in partitions)
        total_rows = sum(p.row_count for p in partitions)
        
        # Find hot and cold partitions
        sorted_by_size = sorted(partitions, key=lambda p: p.size_bytes, reverse=True)
        
        return {
            "table_name": table_name,
            "partition_count": len(partitions),
            "total_size_mb": total_size / (1024 * 1024),
            "total_rows": total_rows,
            "average_partition_size_mb": (total_size / len(partitions)) / (1024 * 1024),
            "largest_partitions": [
                {
                    "name": p.partition_name,
                    "size_mb": p.size_mb,
                    "rows": p.row_count
                }
                for p in sorted_by_size[:5]
            ],
            "strategy": self._strategies.get(table_name)
        }
    
    async def start_maintenance(
        self,
        session: AsyncSession,
        check_interval: int = 3600  # 1 hour
    ):
        """Start partition maintenance tasks."""
        if self._maintenance_task:
            logger.warning("Partition maintenance already running")
            return
        
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(session, check_interval)
        )
        logger.info("Started partition maintenance")
    
    async def stop_maintenance(self):
        """Stop partition maintenance tasks."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
        logger.info("Stopped partition maintenance")
    
    async def _maintenance_loop(
        self,
        session: AsyncSession,
        check_interval: int
    ):
        """Background maintenance loop."""
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                logger.info("Running partition maintenance")
                
                for table_name, strategy in self._strategies.items():
                    # Create future partitions
                    await self._create_future_partitions(session, table_name, strategy)
                    
                    # Cleanup old partitions
                    await self._cleanup_old_partitions(session, table_name, strategy)
                    
                    # Analyze partition usage
                    usage = await self.analyze_partition_usage(session, table_name)
                    
                    if self.metrics:
                        self.metrics.gauge(
                            "db.partition.count",
                            usage.get("partition_count", 0),
                            {"table": table_name}
                        )
                        self.metrics.gauge(
                            "db.partition.total_size_mb",
                            usage.get("total_size_mb", 0),
                            {"table": table_name}
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in partition maintenance: {e}")
                await asyncio.sleep(check_interval)