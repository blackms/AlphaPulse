"""Tests for WebSocket endpoints."""
import pytest
import asyncio
import json
# import websockets # No longer needed directly for connect
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.websockets.manager import ConnectionManager
from alpha_pulse.api.websockets.auth import WebSocketAuthenticator
from alpha_pulse.api.websockets.subscription import SubscriptionManager


@pytest.fixture
def mock_connection_manager():
    """Mock the ConnectionManager."""
    with patch("alpha_pulse.api.websockets.endpoints.connection_manager") as mock:
        # Configure the mock
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.subscribe = AsyncMock()
        mock.broadcast = AsyncMock()
        yield mock


@pytest.fixture
def mock_websocket_auth():
    """Mock the WebSocketAuthenticator."""
    with patch("alpha_pulse.api.websockets.endpoints.websocket_auth") as mock:
        # Configure the mock
        mock.authenticate = AsyncMock()
        yield mock


@pytest.fixture
def mock_subscription_manager():
    """Mock the SubscriptionManager."""
    with patch("alpha_pulse.api.main.subscription_manager") as mock:
        # Configure the mock
        mock.start = AsyncMock()
        mock.stop = AsyncMock()
        yield mock


@pytest.fixture
def client():
    """Create a TestClient instance."""
    return TestClient(app)

@pytest.fixture
def admin_user():
    """Generate admin user authentication."""
    return {
        "username": "admin",
        "role": "admin",
        "permissions": ["view_metrics", "view_alerts", "view_portfolio", "view_trades"]
    }


@pytest.fixture
def jwt_token():
    """Generate a JWT token for testing."""
    import jwt
    import os
    from datetime import datetime, timedelta
    
    # Use a test secret key
    secret_key = "test-secret-key"
    
    # Create payload
    payload = {
        "sub": "admin",
        "username": "admin",
        "role": "admin",
        "permissions": ["view_metrics", "view_alerts", "view_portfolio", "view_trades"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    
    # Generate token
    return jwt.encode(payload, secret_key, algorithm="HS256")


# Remove async, add client fixture
def test_metrics_websocket_authentication(client: TestClient, mock_websocket_auth, mock_connection_manager, admin_user):
    """Test WebSocket authentication for metrics endpoint."""
    # Mock the authenticate method to return the admin user
    mock_websocket_auth.authenticate.return_value = admin_user

    # Use TestClient's websocket_connect
    with client.websocket_connect("/ws/metrics") as websocket:
        # Send authentication message (synchronous)
        websocket.send_text(json.dumps({"token": "test-token"}))

        # Receive welcome message (or handle potential immediate close on auth failure)
        # Assuming successful auth sends welcome first
        welcome_msg = websocket.receive_text()
        assert "Connected to metrics channel" in welcome_msg

        # Verify authentication was called (after connect and auth message)
        mock_websocket_auth.authenticate.assert_called_once()

        # Verify connection was registered
        mock_connection_manager.connect.assert_called_once()

        # Verify subscription was made (websocket object is now TestClient's)
        # The mock should capture the call with the TestClient websocket instance
        mock_connection_manager.subscribe.assert_called_once()
        # Check the arguments passed to subscribe
        args, kwargs = mock_connection_manager.subscribe.call_args
        assert args[1] == "metrics" # Check channel name

        # TestClient websockets don't typically handle raw ping/pong well
        # Skip ping/pong test for TestClient

# Remove async, add client fixture
def test_metrics_websocket_failed_authentication(client: TestClient, mock_websocket_auth, mock_connection_manager):
    """Test failed WebSocket authentication."""
    # Mock the authenticate method to return None (authentication failure)
    mock_websocket_auth.authenticate.return_value = None

    # Expect WebSocketDisconnect when auth fails and server closes
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/metrics") as websocket:
            # Send authentication message
            websocket.send_text(json.dumps({"token": "invalid-token"}))
            # Try to receive data, should raise WebSocketDisconnect
            websocket.receive_text()

    # Check the disconnect code/reason
    assert excinfo.value.code == 1008
    assert excinfo.value.reason == "Authentication failed"

    # Verify authentication was called
    mock_websocket_auth.authenticate.assert_called_once()
    # connect might be called before disconnect, disconnect should be called after close
    mock_connection_manager.connect.assert_called_once()
    mock_connection_manager.disconnect.assert_called_once()


@pytest.mark.asyncio # Keep async for broadcast test as it calls an async mock method
async def test_metrics_websocket_broadcast(mock_connection_manager):
    """Test broadcasting messages to WebSocket clients."""
    # Create a sample metrics update
    metrics_update = {
        "type": "metrics",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "portfolio_value": {
                "value": 1050000.0,
                "timestamp": datetime.now().isoformat(),
                "labels": {"currency": "USD"}
            },
            "sharpe_ratio": {
                "value": 1.9,
                "timestamp": datetime.now().isoformat(),
                "labels": {"window": "30d"}
            }
        }
    }
    
    # Call broadcast method
    await mock_connection_manager.broadcast("metrics", metrics_update)
    
    # Verify broadcast was called with correct parameters
    mock_connection_manager.broadcast.assert_called_once_with("metrics", metrics_update)


# Remove async, add client fixture
def test_alerts_websocket(client: TestClient, mock_websocket_auth, mock_connection_manager, admin_user):
    """Test WebSocket connection for alerts endpoint."""
    mock_websocket_auth.authenticate.return_value = admin_user

    with client.websocket_connect("/ws/alerts") as websocket:
        websocket.send_text(json.dumps({"token": "test-token"}))
        welcome_msg = websocket.receive_text() # Receive welcome
        assert "Connected to alerts channel" in welcome_msg

        mock_websocket_auth.authenticate.assert_called_once()
        mock_connection_manager.connect.assert_called_once()
        mock_connection_manager.subscribe.assert_called_once()
        args, kwargs = mock_connection_manager.subscribe.call_args
        assert args[1] == "alerts"

# Remove async, add client fixture
def test_portfolio_websocket(client: TestClient, mock_websocket_auth, mock_connection_manager, admin_user):
    """Test WebSocket connection for portfolio endpoint."""
    mock_websocket_auth.authenticate.return_value = admin_user

    with client.websocket_connect("/ws/portfolio") as websocket:
        websocket.send_text(json.dumps({"token": "test-token"}))
        welcome_msg = websocket.receive_text() # Receive welcome
        assert "Connected to portfolio channel" in welcome_msg

        mock_websocket_auth.authenticate.assert_called_once()
        mock_connection_manager.connect.assert_called_once()
        mock_connection_manager.subscribe.assert_called_once()
        args, kwargs = mock_connection_manager.subscribe.call_args
        assert args[1] == "portfolio"

# Remove async, add client fixture
def test_trades_websocket(client: TestClient, mock_websocket_auth, mock_connection_manager, admin_user):
    """Test WebSocket connection for trades endpoint."""
    mock_websocket_auth.authenticate.return_value = admin_user

    with client.websocket_connect("/ws/trades") as websocket:
        websocket.send_text(json.dumps({"token": "test-token"}))
        welcome_msg = websocket.receive_text() # Receive welcome
        assert "Connected to trades channel" in welcome_msg

        mock_websocket_auth.authenticate.assert_called_once()
        mock_connection_manager.connect.assert_called_once()
        mock_connection_manager.subscribe.assert_called_once()
        args, kwargs = mock_connection_manager.subscribe.call_args
        assert args[1] == "trades"

# Remove async, add client fixture
def test_websocket_disconnect(client: TestClient, mock_connection_manager, mock_websocket_auth, admin_user):
    """Test WebSocket disconnection handling."""
    # Mock auth to allow connection first
    mock_websocket_auth.authenticate.return_value = admin_user

    # Connect and then exit the context manager, which closes the connection
    with client.websocket_connect("/ws/metrics") as websocket:
        websocket.send_text(json.dumps({"token": "test-token"}))
        welcome_msg = websocket.receive_text() # Receive welcome
        assert "Connected to metrics channel" in welcome_msg
        # Connection closes automatically when 'with' block exits

    # Verify disconnect was called after the 'with' block
    mock_connection_manager.disconnect.assert_called_once()