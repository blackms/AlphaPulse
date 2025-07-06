"""Tests for database optimization features."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from alpha_pulse.config.database_config import (
    DatabaseConfig, DatabaseNode, ConnectionPoolConfig,
    QueryOptimizationConfig, LoadBalancingStrategy
)
from alpha_pulse.database.query_analyzer import QueryAnalyzer, QueryPlan, QueryType
from alpha_pulse.database.slow_query_detector import SlowQueryDetector, SlowQueryInfo
from alpha_pulse.database.index_advisor import IndexAdvisor, IndexRecommendation
from alpha_pulse.database.partition_manager import (
    PartitionManager, PartitionStrategy, PartitionType, PartitionInterval
)
from alpha_pulse.database.read_write_router import ReadWriteRouter, QueryIntent
from alpha_pulse.database.load_balancer import LoadBalancer, NodeStatus
from alpha_pulse.database.failover_manager import FailoverManager, FailoverState
from alpha_pulse.services.database_optimization_service import DatabaseOptimizationService


@pytest.fixture
def db_config():
    """Create test database configuration."""
    config = DatabaseConfig()
    config.master_node = DatabaseNode(
        host="master.db",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_pass"
    )
    
    config.read_replicas = [
        DatabaseNode(
            host="replica1.db",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            is_master=False,
            weight=1
        ),
        DatabaseNode(
            host="replica2.db",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            is_master=False,
            weight=2
        )
    ]
    
    config.connection_pool = ConnectionPoolConfig(
        min_size=5,
        max_size=20,
        pool_timeout=30.0,
        load_balancing=LoadBalancingStrategy.WEIGHTED
    )
    
    config.query_optimization = QueryOptimizationConfig(
        enable_query_cache=True,
        slow_query_threshold=1.0,
        analyze_query_plans=True
    )
    
    return config


@pytest.fixture
def metrics_mock():
    """Create mock metrics collector."""
    metrics = Mock()
    metrics.gauge = Mock()
    metrics.histogram = Mock()
    metrics.increment = Mock()
    return metrics


class TestQueryAnalyzer:
    """Test query analyzer functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_query(self, metrics_mock):
        """Test query analysis."""
        analyzer = QueryAnalyzer(metrics_mock)
        
        # Mock session
        session = AsyncMock()
        session.execute = AsyncMock(return_value=AsyncMock(
            scalar=AsyncMock(return_value=[{
                "Plan": {
                    "Node Type": "Seq Scan",
                    "Total Cost": 1000,
                    "Plan Rows": 100,
                    "Plan Width": 50
                }
            }])
        ))
        
        # Analyze query
        plan = await analyzer.analyze_query(
            session,
            "SELECT * FROM users WHERE created_at > '2024-01-01'"
        )
        
        assert plan.plan_type == "Seq Scan"
        assert plan.total_cost == 1000
        assert plan.rows == 100
        assert not plan.is_efficient  # Sequential scan is not efficient
        assert len(plan.suggestions) > 0
    
    def test_get_query_type(self):
        """Test query type detection."""
        analyzer = QueryAnalyzer()
        
        assert analyzer._get_query_type("SELECT * FROM users") == QueryType.SELECT
        assert analyzer._get_query_type("INSERT INTO users VALUES (1)") == QueryType.INSERT
        assert analyzer._get_query_type("UPDATE users SET name = 'test'") == QueryType.UPDATE
        assert analyzer._get_query_type("DELETE FROM users") == QueryType.DELETE
    
    def test_normalize_query(self):
        """Test query normalization."""
        analyzer = QueryAnalyzer()
        
        query = "SELECT * FROM users WHERE id = 123 AND name = 'John'"
        normalized = analyzer._normalize_query(query)
        
        assert "?" in normalized  # Numbers replaced
        assert "123" not in normalized
        assert "'john'" not in normalized  # Lowercase


class TestSlowQueryDetector:
    """Test slow query detection."""
    
    @pytest.mark.asyncio
    async def test_slow_query_detection(self, metrics_mock):
        """Test detection of slow queries."""
        config = QueryOptimizationConfig(slow_query_threshold=0.5)
        analyzer = QueryAnalyzer()
        detector = SlowQueryDetector(config, analyzer, metrics_mock)
        
        # Simulate slow query
        context = Mock()
        context._query_info = {
            "statement": "SELECT * FROM large_table",
            "parameters": None,
            "start_time": 0
        }
        
        # Before query
        detector._before_query(None, None, "SELECT * FROM large_table", None, context, None)
        
        # Simulate delay
        detector._query_start_times[id(context)] = 0  # Mock start time
        
        # After query (with 1 second execution time)
        with patch('time.time', return_value=1.0):
            detector._after_query(None, None, "SELECT * FROM large_table", None, context, None)
        
        assert detector._slow_query_count == 1
        assert len(detector._slow_queries) == 1
        assert detector._slow_queries[0].execution_time == 1.0
    
    def test_query_pattern_extraction(self):
        """Test query pattern extraction."""
        config = QueryOptimizationConfig()
        analyzer = QueryAnalyzer()
        detector = SlowQueryDetector(config, analyzer)
        
        query = "SELECT * FROM users WHERE id = 123 AND status = 'active'"
        pattern = detector._get_query_pattern(query)
        
        assert "N" in pattern  # Number replaced
        assert "S" in pattern  # String replaced
        assert "123" not in pattern
        assert "'active'" not in pattern


class TestIndexAdvisor:
    """Test index advisor functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_indexes(self, metrics_mock):
        """Test index analysis."""
        advisor = IndexAdvisor(metrics_mock)
        
        # Mock session
        session = AsyncMock()
        
        # Mock existing indexes query
        session.execute = AsyncMock(side_effect=[
            # Existing indexes
            AsyncMock(return_value=[]),
            # Index usage
            AsyncMock(return_value=[]),
            # Slow queries
            AsyncMock(scalar=AsyncMock(return_value=False))
        ])
        
        recommendations = await advisor.analyze_indexes(session)
        
        assert isinstance(recommendations, list)
        metrics_mock.gauge.assert_called()
    
    def test_extract_where_columns(self):
        """Test WHERE clause column extraction."""
        advisor = IndexAdvisor()
        
        query = """
            SELECT * FROM users u
            WHERE u.status = 'active' 
            AND u.created_at > '2024-01-01'
            AND orders.user_id = u.id
        """
        
        columns = advisor._extract_where_columns(query)
        
        assert "users" in columns
        assert "status" in columns["users"]
        assert "created_at" in columns["users"]
    
    def test_create_index_recommendation(self):
        """Test index recommendation creation."""
        rec = IndexRecommendation(
            table_name="users",
            columns=["status", "created_at"],
            index_type="btree",
            reason="Columns used in WHERE clause",
            estimated_benefit=75,
            estimated_size_mb=10,
            affected_queries=["SELECT * FROM users WHERE status = ?"],
            priority=4
        )
        
        assert "CREATE INDEX" in rec.create_statement
        assert "users" in rec.create_statement
        assert "status" in rec.create_statement


class TestPartitionManager:
    """Test partition management."""
    
    @pytest.mark.asyncio
    async def test_create_partition(self, metrics_mock):
        """Test partition creation."""
        manager = PartitionManager(metrics_mock)
        
        # Create strategy
        strategy = PartitionStrategy(
            table_name="market_data",
            partition_column="timestamp",
            partition_type=PartitionType.RANGE,
            interval=PartitionInterval.DAILY,
            retention_days=30
        )
        
        manager._strategies["market_data"] = strategy
        
        # Mock session
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        
        # Create partition
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        success = await manager.create_partition(
            session, "market_data", start_date, end_date
        )
        
        assert success
        session.execute.assert_called()
        metrics_mock.increment.assert_called_with("db.partition.created")
    
    def test_partition_naming(self):
        """Test partition naming strategy."""
        strategy = PartitionStrategy(
            table_name="trades",
            partition_column="created_at",
            partition_type=PartitionType.RANGE,
            interval=PartitionInterval.MONTHLY
        )
        
        date = datetime(2024, 3, 15)
        name = strategy.get_partition_name(date)
        
        assert name == "trades_202403"


class TestReadWriteRouter:
    """Test read/write routing."""
    
    @pytest.mark.asyncio
    async def test_read_routing(self, db_config, metrics_mock):
        """Test routing read queries to replicas."""
        pool = Mock()
        pool.get_master_session = AsyncMock()
        pool.get_replica_session = AsyncMock()
        pool._active_connections = {"replica_0": 5, "replica_1": 3}
        
        router = ReadWriteRouter(db_config, pool, metrics_mock)
        
        # Test read routing
        async with router.get_session(QueryIntent.READ):
            pass
        
        # Should route to replica
        assert router._routing_stats["reads_to_replica"] > 0
    
    @pytest.mark.asyncio
    async def test_write_routing(self, db_config, metrics_mock):
        """Test routing write queries to master."""
        pool = Mock()
        pool.get_master_session = AsyncMock()
        
        router = ReadWriteRouter(db_config, pool, metrics_mock)
        
        # Test write routing
        async with router.get_session(QueryIntent.WRITE):
            pass
        
        # Should route to master
        pool.get_master_session.assert_called()
        assert router._routing_stats["writes_to_master"] == 1
    
    def test_query_intent_analysis(self, db_config, metrics_mock):
        """Test query intent analysis."""
        pool = Mock()
        router = ReadWriteRouter(db_config, pool, metrics_mock)
        
        assert router.analyze_query_intent("SELECT * FROM users") == QueryIntent.READ
        assert router.analyze_query_intent("INSERT INTO users") == QueryIntent.WRITE
        assert router.analyze_query_intent("UPDATE users SET") == QueryIntent.WRITE
        assert router.analyze_query_intent("SELECT * FOR UPDATE") == QueryIntent.WRITE


class TestLoadBalancer:
    """Test load balancing."""
    
    def test_round_robin_selection(self, metrics_mock):
        """Test round-robin load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN, metrics_mock)
        
        # Add nodes
        node1 = DatabaseNode(host="replica1", port=5432)
        node2 = DatabaseNode(host="replica2", port=5432)
        
        balancer.add_node(node1, "node1")
        balancer.add_node(node2, "node2")
        
        # Test round-robin
        selections = [balancer.select_node() for _ in range(4)]
        
        assert selections[0] == "node1"
        assert selections[1] == "node2"
        assert selections[2] == "node1"
        assert selections[3] == "node2"
    
    def test_least_connections_selection(self, metrics_mock):
        """Test least connections load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS, metrics_mock)
        
        # Add nodes
        node1 = DatabaseNode(host="replica1", port=5432)
        node2 = DatabaseNode(host="replica2", port=5432)
        
        balancer.add_node(node1, "node1")
        balancer.add_node(node2, "node2")
        
        # Set connection counts
        balancer._nodes["node1"].metrics.active_connections = 10
        balancer._nodes["node2"].metrics.active_connections = 5
        
        # Should select node2 (fewer connections)
        assert balancer.select_node() == "node2"
    
    def test_circuit_breaker(self, metrics_mock):
        """Test circuit breaker functionality."""
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN, metrics_mock)
        
        # Add node
        node = DatabaseNode(host="replica1", port=5432)
        balancer.add_node(node, "node1")
        
        # Report failures
        for _ in range(5):
            balancer.report_failure("node1", Exception("Connection failed"))
        
        # Circuit breaker should be open
        assert balancer._nodes["node1"].circuit_breaker_open
        assert "node1" not in balancer._active_nodes


class TestFailoverManager:
    """Test failover management."""
    
    @pytest.mark.asyncio
    async def test_failover_detection(self, db_config, metrics_mock):
        """Test master failure detection."""
        pool = Mock()
        pool.get_master_session = AsyncMock(side_effect=Exception("Connection failed"))
        
        router = Mock()
        
        manager = FailoverManager(db_config, pool, router, metrics_mock)
        manager.max_consecutive_failures = 2
        
        # Simulate failures
        for _ in range(2):
            is_healthy = await manager._check_master_health()
            assert not is_healthy
            await manager._handle_master_failure()
        
        assert manager._consecutive_failures == 2
        assert manager.state == FailoverState.DETECTING
    
    def test_promotion_candidate_selection(self, db_config, metrics_mock):
        """Test replica promotion candidate selection."""
        pool = Mock()
        router = Mock()
        
        manager = FailoverManager(db_config, pool, router, metrics_mock)
        
        # Test scoring
        replica = db_config.read_replicas[0]
        score = manager._calculate_promotion_score(replica, lag=0.5, health=Mock(consecutive_failures=0))
        
        assert score > 0
        assert score <= 100


class TestDatabaseOptimizationService:
    """Test database optimization service."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, db_config, metrics_mock):
        """Test service initialization."""
        with patch('alpha_pulse.services.database_optimization_service.ConnectionPool'):
            service = DatabaseOptimizationService(db_config, metrics_mock)
            
            # Mock initialization
            service.connection_pool.initialize = AsyncMock()
            service.router.start_lag_monitoring = AsyncMock()
            service.failover_manager.start_monitoring = AsyncMock()
            service.connection_monitor.start_monitoring = AsyncMock()
            service.database_monitor.start_monitoring = AsyncMock()
            service.slow_query_detector.start_monitoring = AsyncMock()
            
            await service.initialize()
            
            assert service._is_initialized
            service.connection_pool.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_analysis(self, db_config, metrics_mock):
        """Test query analysis functionality."""
        service = DatabaseOptimizationService(db_config, metrics_mock)
        
        # Mock components
        service.connection_pool.get_master_session = AsyncMock()
        service.query_analyzer.analyze_query = AsyncMock(return_value=QueryPlan(
            query="SELECT * FROM users",
            plan_type="Seq Scan",
            total_cost=100,
            rows=50,
            width=100
        ))
        service.query_optimizer.optimize_query = AsyncMock(return_value=(
            "SELECT id, name FROM users",
            []
        ))
        service.query_optimizer.estimate_cost = AsyncMock(return_value=Mock(
            cpu_cost=50,
            io_cost=40,
            total_cost=90,
            estimated_time=10
        ))
        
        result = await service.analyze_query("SELECT * FROM users")
        
        assert "original_query" in result
        assert "optimized_query" in result
        assert "execution_plan" in result
        assert "cost_estimate" in result
    
    def test_service_status(self, db_config, metrics_mock):
        """Test service status reporting."""
        service = DatabaseOptimizationService(db_config, metrics_mock)
        
        status = service.get_status()
        
        assert "initialized" in status
        assert "components" in status
        assert "config" in status
        assert status["config"]["replicas"] == 2