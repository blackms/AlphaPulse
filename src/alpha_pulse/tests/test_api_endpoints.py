"""
Tests for API endpoints to improve coverage.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json


@pytest.fixture
def mock_dependencies():
    """Mock all API dependencies."""
    with patch('alpha_pulse.api.main.get_db') as mock_db, \
         patch('alpha_pulse.api.main.verify_jwt_token') as mock_jwt, \
         patch('alpha_pulse.api.main.RateLimiter') as mock_limiter:
        
        # Mock database session
        mock_db.return_value = Mock()
        
        # Mock JWT verification
        mock_jwt.return_value = {"user_id": "test_user", "role": "admin"}
        
        # Mock rate limiter
        mock_limiter.return_value.check_rate_limit = AsyncMock(return_value=True)
        
        yield {
            "db": mock_db,
            "jwt": mock_jwt,
            "limiter": mock_limiter
        }


@pytest.fixture
def test_client(mock_dependencies):
    """Create test client with mocked dependencies."""
    from alpha_pulse.api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, test_client):
        """Test basic health check."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_readiness_check(self, test_client):
        """Test readiness check."""
        response = test_client.get("/ready")
        assert response.status_code == 200


class TestPortfolioEndpoints:
    """Test portfolio endpoints."""
    
    @patch('alpha_pulse.api.routers.portfolio.get_portfolio_manager')
    def test_get_portfolio_status(self, mock_manager, test_client):
        """Test GET /api/v1/portfolio/status."""
        mock_manager.return_value.get_current_state = AsyncMock(
            return_value={
                "total_value": 100000,
                "cash": 50000,
                "positions": {"BTC": {"quantity": 1.0, "value": 50000}}
            }
        )
        
        response = test_client.get(
            "/api/v1/portfolio/status",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_value"] == 100000
        assert "positions" in data
    
    @patch('alpha_pulse.api.routers.portfolio.get_portfolio_manager')
    def test_get_portfolio_metrics(self, mock_manager, test_client):
        """Test GET /api/v1/portfolio/metrics."""
        mock_manager.return_value.calculate_metrics = AsyncMock(
            return_value={
                "returns": 0.15,
                "volatility": 0.20,
                "sharpe_ratio": 0.75,
                "max_drawdown": 0.10
            }
        )
        
        response = test_client.get(
            "/api/v1/portfolio/metrics",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sharpe_ratio"] == 0.75


class TestTradingEndpoints:
    """Test trading endpoints."""
    
    @patch('alpha_pulse.api.routers.trading.get_trade_executor')
    def test_create_order(self, mock_executor, test_client):
        """Test POST /api/v1/trading/orders."""
        mock_executor.return_value.create_order = AsyncMock(
            return_value={
                "order_id": "TEST123",
                "status": "pending",
                "symbol": "BTC",
                "quantity": 0.1
            }
        )
        
        order_data = {
            "symbol": "BTC",
            "side": "buy",
            "quantity": 0.1,
            "order_type": "market"
        }
        
        response = test_client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["order_id"] == "TEST123"
    
    @patch('alpha_pulse.api.routers.trading.get_trade_executor')
    def test_get_orders(self, mock_executor, test_client):
        """Test GET /api/v1/trading/orders."""
        mock_executor.return_value.get_orders = AsyncMock(
            return_value=[
                {
                    "order_id": "TEST123",
                    "symbol": "BTC",
                    "status": "filled",
                    "quantity": 0.1
                }
            ]
        )
        
        response = test_client.get(
            "/api/v1/trading/orders",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["order_id"] == "TEST123"


class TestRiskEndpoints:
    """Test risk management endpoints."""
    
    @patch('alpha_pulse.api.routers.risk.get_risk_manager')
    def test_get_risk_metrics(self, mock_manager, test_client):
        """Test GET /api/v1/risk/metrics."""
        mock_manager.return_value.get_current_metrics = AsyncMock(
            return_value={
                "portfolio_var": 0.05,
                "portfolio_cvar": 0.08,
                "max_drawdown": 0.15,
                "correlation_risk": 0.6
            }
        )
        
        response = test_client.get(
            "/api/v1/risk/metrics",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_var"] == 0.05
    
    @patch('alpha_pulse.api.routers.risk.get_risk_manager')
    def test_update_risk_limits(self, mock_manager, test_client):
        """Test PUT /api/v1/risk/limits."""
        mock_manager.return_value.update_limits = AsyncMock(
            return_value={"status": "updated"}
        )
        
        limits_data = {
            "max_position_size": 10000,
            "max_leverage": 2.0,
            "stop_loss_pct": 0.05
        }
        
        response = test_client.put(
            "/api/v1/risk/limits",
            json=limits_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"


class TestMarketDataEndpoints:
    """Test market data endpoints."""
    
    @patch('alpha_pulse.api.routers.market_data.get_data_provider')
    def test_get_market_data(self, mock_provider, test_client):
        """Test GET /api/v1/market/data/{symbol}."""
        mock_provider.return_value.get_latest_data = AsyncMock(
            return_value={
                "symbol": "BTC",
                "price": 50000,
                "volume": 1000000,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        response = test_client.get(
            "/api/v1/market/data/BTC",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC"
        assert data["price"] == 50000


class TestBacktestingEndpoints:
    """Test backtesting endpoints."""
    
    @patch('alpha_pulse.api.routers.backtesting.BacktestEngine')
    def test_run_backtest(self, mock_engine, test_client):
        """Test POST /api/v1/backtesting/run."""
        mock_engine.return_value.run = AsyncMock(
            return_value={
                "total_return": 0.25,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.15,
                "win_rate": 0.65
            }
        )
        
        backtest_config = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "initial_capital": 100000,
            "strategy": "momentum"
        }
        
        response = test_client.post(
            "/api/v1/backtesting/run",
            json=backtest_config,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_return"] == 0.25
        assert data["sharpe_ratio"] == 1.2


class TestWebSocketEndpoints:
    """Test WebSocket endpoints."""
    
    @patch('alpha_pulse.api.routers.websocket.ConnectionManager')
    def test_websocket_connection(self, mock_manager, test_client):
        """Test WebSocket connection."""
        with test_client.websocket_connect("/ws") as websocket:
            # Send test message
            websocket.send_json({"type": "subscribe", "channel": "portfolio"})
            
            # Mock receiving data
            mock_manager.return_value.broadcast = AsyncMock()
            
            # Connection should be established
            assert websocket is not None


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    @patch('alpha_pulse.api.auth.authenticate_user')
    @patch('alpha_pulse.api.auth.create_access_token')
    def test_login(self, mock_create_token, mock_auth, test_client):
        """Test POST /api/v1/auth/login."""
        mock_auth.return_value = {"user_id": "test_user", "role": "trader"}
        mock_create_token.return_value = "test_jwt_token"
        
        login_data = {
            "username": "test_user",
            "password": "test_password"
        }
        
        response = test_client.post(
            "/api/v1/auth/login",
            json=login_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"


class TestMonitoringEndpoints:
    """Test monitoring endpoints."""
    
    @patch('alpha_pulse.api.routers.monitoring.get_metrics_collector')
    def test_get_system_metrics(self, mock_collector, test_client):
        """Test GET /api/v1/monitoring/metrics."""
        mock_collector.return_value.get_all_metrics = AsyncMock(
            return_value={
                "cpu_usage": 45.5,
                "memory_usage": 60.2,
                "disk_usage": 30.0,
                "active_connections": 25
            }
        )
        
        response = test_client.get(
            "/api/v1/monitoring/metrics",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["cpu_usage"] == 45.5
        assert data["active_connections"] == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])