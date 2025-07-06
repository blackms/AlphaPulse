"""Database connection pool implementation."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

from ..config.database_config import DatabaseConfig, DatabaseNode, ConnectionPoolConfig
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class ConnectionPool:
    """Manages database connection pooling."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize connection pool."""
        self.config = config
        self.metrics = metrics_collector
        
        # SQLAlchemy engines
        self._master_engine: Optional[AsyncEngine] = None
        self._replica_engines: Dict[str, AsyncEngine] = {}
        
        # Session factories
        self._master_session_factory: Optional[sessionmaker] = None
        self._replica_session_factories: Dict[str, sessionmaker] = {}
        
        # Connection tracking
        self._active_connections: Dict[str, int] = {}
        self._connection_wait_times: List[float] = []
        
        # Round-robin index for load balancing
        self._replica_index = 0
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize connection pools."""
        try:
            logger.info("Initializing database connection pools...")
            
            # Create master connection pool
            await self._create_master_pool()
            
            # Create replica connection pools
            if self.config.read_replicas:
                await self._create_replica_pools()
            
            self._is_initialized = True
            logger.info("Database connection pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise
    
    async def _create_master_pool(self) -> None:
        """Create connection pool for master database."""
        pool_config = self.config.connection_pool
        
        # Configure SQLAlchemy pool
        pool_class = self._get_pool_class(pool_config)
        pool_args = self._get_pool_args(pool_config)
        
        # Create engine with connection pool
        self._master_engine = create_async_engine(
            self.config.master_node.connection_string,
            poolclass=pool_class,
            **pool_args,
            echo=pool_config.echo_pool,
            connect_args={
                "server_settings": {
                    "application_name": "alphapulse_master",
                    "jit": "off"
                },
                "timeout": pool_config.connect_timeout,
            }
        )
        
        # Create session factory
        self._master_session_factory = sessionmaker(
            self._master_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Initialize connection tracking
        self._active_connections["master"] = 0
        
        logger.info(f"Created master connection pool: {self.config.master_node.host}")
    
    async def _create_replica_pools(self) -> None:
        """Create connection pools for read replicas."""
        pool_config = self.config.connection_pool
        pool_class = self._get_pool_class(pool_config)
        pool_args = self._get_pool_args(pool_config)
        
        for i, replica in enumerate(self.config.read_replicas):
            replica_id = f"replica_{i}"
            
            # Create engine for replica
            engine = create_async_engine(
                replica.connection_string,
                poolclass=pool_class,
                **pool_args,
                echo=pool_config.echo_pool,
                connect_args={
                    "server_settings": {
                        "application_name": f"alphapulse_{replica_id}",
                        "jit": "off"
                    },
                    "timeout": pool_config.connect_timeout,
                }
            )
            
            # Create session factory
            session_factory = sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._replica_engines[replica_id] = engine
            self._replica_session_factories[replica_id] = session_factory
            self._active_connections[replica_id] = 0
            
            logger.info(f"Created replica connection pool: {replica.host}")
    
    def _get_pool_class(self, config: ConnectionPoolConfig):
        """Get appropriate pool class based on configuration."""
        if config.max_size == 0:
            return NullPool
        elif config.min_size == config.max_size:
            return StaticPool
        else:
            return QueuePool
    
    def _get_pool_args(self, config: ConnectionPoolConfig) -> Dict[str, Any]:
        """Get pool arguments from configuration."""
        return {
            "pool_size": config.min_size,
            "max_overflow": config.overflow,
            "pool_timeout": config.pool_timeout,
            "pool_pre_ping": config.pool_pre_ping,
            "pool_recycle": config.pool_recycle,
        }
    
    @asynccontextmanager
    async def get_master_session(self):
        """Get a session for write operations."""
        if not self._is_initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        
        async with self._lock:
            self._active_connections["master"] += 1
        
        try:
            async with self._master_session_factory() as session:
                # Track connection acquisition time
                wait_time = (datetime.utcnow() - start_time).total_seconds()
                self._connection_wait_times.append(wait_time)
                
                if self.metrics:
                    self.metrics.gauge(
                        "db.pool.active_connections",
                        self._active_connections["master"],
                        {"pool": "master"}
                    )
                    self.metrics.histogram(
                        "db.pool.wait_time",
                        wait_time,
                        {"pool": "master"}
                    )
                
                yield session
                
        finally:
            async with self._lock:
                self._active_connections["master"] -= 1
    
    @asynccontextmanager
    async def get_replica_session(self):
        """Get a session for read operations."""
        if not self._is_initialized:
            await self.initialize()
        
        # Use master if no replicas configured
        if not self._replica_session_factories:
            async with self.get_master_session() as session:
                yield session
            return
        
        # Select replica based on load balancing strategy
        replica_id = await self._select_replica()
        
        start_time = datetime.utcnow()
        
        async with self._lock:
            self._active_connections[replica_id] += 1
        
        try:
            session_factory = self._replica_session_factories[replica_id]
            async with session_factory() as session:
                # Track connection acquisition time
                wait_time = (datetime.utcnow() - start_time).total_seconds()
                self._connection_wait_times.append(wait_time)
                
                if self.metrics:
                    self.metrics.gauge(
                        "db.pool.active_connections",
                        self._active_connections[replica_id],
                        {"pool": replica_id}
                    )
                    self.metrics.histogram(
                        "db.pool.wait_time",
                        wait_time,
                        {"pool": replica_id}
                    )
                
                yield session
                
        finally:
            async with self._lock:
                self._active_connections[replica_id] -= 1
    
    async def _select_replica(self) -> str:
        """Select a replica based on load balancing strategy."""
        strategy = self.config.connection_pool.load_balancing
        
        if strategy == "round_robin":
            # Simple round-robin selection
            replica_ids = list(self._replica_session_factories.keys())
            replica_id = replica_ids[self._replica_index % len(replica_ids)]
            self._replica_index += 1
            return replica_id
            
        elif strategy == "least_connections":
            # Select replica with least active connections
            min_connections = float('inf')
            selected_replica = None
            
            for replica_id in self._replica_session_factories:
                connections = self._active_connections.get(replica_id, 0)
                if connections < min_connections:
                    min_connections = connections
                    selected_replica = replica_id
            
            return selected_replica or list(self._replica_session_factories.keys())[0]
            
        elif strategy == "weighted":
            # Weighted selection based on replica weights
            # Simple implementation - can be improved
            replica_ids = list(self._replica_session_factories.keys())
            return replica_ids[self._replica_index % len(replica_ids)]
            
        else:  # random
            import random
            replica_ids = list(self._replica_session_factories.keys())
            return random.choice(replica_ids)
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = {
            "pools": {},
            "total_active_connections": sum(self._active_connections.values()),
            "average_wait_time": (
                sum(self._connection_wait_times) / len(self._connection_wait_times)
                if self._connection_wait_times else 0
            )
        }
        
        # Master pool stats
        if self._master_engine:
            pool = self._master_engine.pool
            stats["pools"]["master"] = {
                "size": pool.size() if hasattr(pool, 'size') else 0,
                "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else 0,
                "overflow": pool.overflow() if hasattr(pool, 'overflow') else 0,
                "active": self._active_connections.get("master", 0)
            }
        
        # Replica pool stats
        for replica_id, engine in self._replica_engines.items():
            pool = engine.pool
            stats["pools"][replica_id] = {
                "size": pool.size() if hasattr(pool, 'size') else 0,
                "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else 0,
                "overflow": pool.overflow() if hasattr(pool, 'overflow') else 0,
                "active": self._active_connections.get(replica_id, 0)
            }
        
        return stats
    
    async def close(self) -> None:
        """Close all connection pools."""
        try:
            # Close master engine
            if self._master_engine:
                await self._master_engine.dispose()
            
            # Close replica engines
            for engine in self._replica_engines.values():
                await engine.dispose()
            
            self._replica_engines.clear()
            self._replica_session_factories.clear()
            self._active_connections.clear()
            self._connection_wait_times.clear()
            
            self._is_initialized = False
            
            logger.info("Database connection pools closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing connection pools: {e}")
            raise