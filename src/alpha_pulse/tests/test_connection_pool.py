"""Unit tests for database connection pool."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from alpha_pulse.database.connection_pool import ConnectionPool
from alpha_pulse.database.connection_manager import (
    ConnectionValidator, ConnectionHealth, ConnectionState
)
from alpha_pulse.config.database_config import (
    DatabaseConfig, DatabaseNode, ConnectionPoolConfig, LoadBalancingStrategy
)


class TestConnectionPool:
    """Test database connection pool functionality."""
    
    @pytest.fixture
    def db_config(self):
        """Create test database configuration."""
        config = DatabaseConfig()
        config.master_node = DatabaseNode(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        # Add test replicas
        config.read_replicas = [
            DatabaseNode(
                host="replica1",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass",
                is_master=False
            ),
            DatabaseNode(
                host="replica2",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass",
                is_master=False,
                weight=2
            )
        ]
        
        config.connection_pool = ConnectionPoolConfig(
            min_size=2,
            max_size=10,
            pool_timeout=5.0,
            load_balancing=LoadBalancingStrategy.ROUND_ROBIN
        )
        
        return config
    
    @pytest.fixture
    def metrics_mock(self):
        """Create mock metrics collector."""
        metrics = Mock()
        metrics.gauge = Mock()
        metrics.histogram = Mock()
        metrics.increment = Mock()
        return metrics
    
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, db_config, metrics_mock):
        """Test connection pool initialization."""
        with patch('alpha_pulse.database.connection_pool.create_async_engine') as mock_engine:
            # Mock engine creation
            mock_engine.return_value = Mock()
            
            pool = ConnectionPool(db_config, metrics_mock)
            assert pool._is_initialized is False
            
            await pool.initialize()
            
            assert pool._is_initialized is True
            assert pool._master_engine is not None
            assert len(pool._replica_engines) == 2
            assert "master" in pool._active_connections
            assert pool._active_connections["master"] == 0
    
    @pytest.mark.asyncio
    async def test_master_session_acquisition(self, db_config, metrics_mock):
        """Test acquiring master database session."""
        with patch('alpha_pulse.database.connection_pool.create_async_engine'):
            pool = ConnectionPool(db_config, metrics_mock)
            
            # Mock session factory
            mock_session = AsyncMock()
            mock_factory = AsyncMock(return_value=mock_session)
            pool._master_session_factory = mock_factory
            pool._is_initialized = True
            
            async with pool.get_master_session() as session:
                assert session == mock_session
                assert pool._active_connections["master"] == 0  # Should be decremented after context
            
            # Verify metrics were recorded
            metrics_mock.gauge.assert_called()
            metrics_mock.histogram.assert_called()
    
    @pytest.mark.asyncio
    async def test_replica_session_round_robin(self, db_config, metrics_mock):
        """Test round-robin replica selection."""
        with patch('alpha_pulse.database.connection_pool.create_async_engine'):
            pool = ConnectionPool(db_config, metrics_mock)
            
            # Mock replica sessions
            mock_sessions = [AsyncMock() for _ in range(2)]
            mock_factories = [
                AsyncMock(return_value=mock_sessions[i]) 
                for i in range(2)
            ]
            
            pool._replica_session_factories = {
                "replica_0": mock_factories[0],
                "replica_1": mock_factories[1]
            }
            pool._is_initialized = True
            
            # First call should use replica_0
            pool._replica_index = 0
            async with pool.get_replica_session() as session:
                assert session == mock_sessions[0]
            
            # Second call should use replica_1
            async with pool.get_replica_session() as session:
                assert session == mock_sessions[1]
            
            # Third call should wrap back to replica_0
            async with pool.get_replica_session() as session:
                assert session == mock_sessions[0]
    
    @pytest.mark.asyncio
    async def test_least_connections_strategy(self, db_config, metrics_mock):
        """Test least connections load balancing."""
        db_config.connection_pool.load_balancing = LoadBalancingStrategy.LEAST_CONNECTIONS
        
        with patch('alpha_pulse.database.connection_pool.create_async_engine'):
            pool = ConnectionPool(db_config, metrics_mock)
            pool._is_initialized = True
            
            # Set up active connections
            pool._active_connections = {
                "replica_0": 5,
                "replica_1": 2
            }
            
            pool._replica_session_factories = {
                "replica_0": AsyncMock(),
                "replica_1": AsyncMock()
            }
            
            # Should select replica_1 (fewer connections)
            replica_id = await pool._select_replica()
            assert replica_id == "replica_1"
    
    @pytest.mark.asyncio
    async def test_connection_timeout(self, db_config, metrics_mock):
        """Test connection acquisition timeout."""
        db_config.connection_pool.pool_timeout = 0.1  # 100ms timeout
        
        with patch('alpha_pulse.database.connection_pool.create_async_engine'):
            pool = ConnectionPool(db_config, metrics_mock)
            pool._is_initialized = True
            
            # Mock factory that takes too long
            async def slow_factory():
                await asyncio.sleep(0.5)  # Longer than timeout
                return AsyncMock()
            
            pool._master_session_factory = slow_factory
            
            with pytest.raises(TimeoutError):
                async with pool.get_master_session():
                    pass
            
            metrics_mock.increment.assert_called_with("db.pool.timeout", {"pool": "master"})
    
    @pytest.mark.asyncio
    async def test_pool_statistics(self, db_config, metrics_mock):
        """Test connection pool statistics."""
        with patch('alpha_pulse.database.connection_pool.create_async_engine') as mock_engine:
            # Mock pool methods
            mock_pool = Mock()
            mock_pool.size.return_value = 5
            mock_pool.checkedin.return_value = 3
            mock_pool.overflow.return_value = 0
            
            mock_engine.return_value.pool = mock_pool
            
            pool = ConnectionPool(db_config, metrics_mock)
            await pool.initialize()
            
            stats = await pool.get_pool_stats()
            
            assert "pools" in stats
            assert "master" in stats["pools"]
            assert stats["pools"]["master"]["size"] == 5
            assert stats["pools"]["master"]["checked_in"] == 3
            assert stats["total_active_connections"] == 0


class TestConnectionValidator:
    """Test connection validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create test validator."""
        config = DatabaseConfig()
        return ConnectionValidator(config)
    
    @pytest.mark.asyncio
    async def test_validate_connection_success(self, validator):
        """Test successful connection validation."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone = AsyncMock(return_value=(1,))
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        is_valid = await validator.validate_connection(mock_session)
        assert is_valid is True
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_connection_failure(self, validator):
        """Test failed connection validation."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("Connection error"))
        
        is_valid = await validator.validate_connection(mock_session)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_health_check(self, validator):
        """Test database node health check."""
        node = DatabaseNode(host="test_host", port=5432)
        
        with patch('asyncpg.connect') as mock_connect:
            mock_conn = AsyncMock()
            mock_conn.fetchval = AsyncMock(return_value=1)
            mock_connect.return_value = mock_conn
            
            health = await validator.health_check(node)
            
            assert health.is_healthy is True
            assert health.consecutive_failures == 0
            assert len(health.response_times) == 1
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, validator):
        """Test failed health check."""
        node = DatabaseNode(host="test_host", port=5432)
        
        with patch('asyncpg.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            health = await validator.health_check(node)
            
            assert health.is_healthy is False
            assert health.consecutive_failures == 1
            assert health.error_count == 1
    
    def test_connection_health_state_transitions(self):
        """Test connection health state transitions."""
        health = ConnectionHealth("test_node")
        
        # Initial state
        assert health.state == ConnectionState.UNKNOWN
        
        # Success makes it healthy
        health.record_success(0.1)
        assert health.state == ConnectionState.HEALTHY
        assert health.consecutive_failures == 0
        
        # First failure degrades
        health.record_failure(Exception("Error"))
        assert health.state == ConnectionState.DEGRADED
        assert health.consecutive_failures == 1
        
        # Third failure makes unhealthy
        health.record_failure(Exception("Error"))
        health.record_failure(Exception("Error"))
        assert health.state == ConnectionState.UNHEALTHY
        assert health.consecutive_failures == 3
        
        # Success resets
        health.record_success(0.1)
        assert health.state == ConnectionState.HEALTHY
        assert health.consecutive_failures == 0