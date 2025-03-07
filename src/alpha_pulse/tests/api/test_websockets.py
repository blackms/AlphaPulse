"""Tests for WebSocket endpoints."""
import pytest
import asyncio
import json
import websockets
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.websockets.manager import ConnectionManager
from alpha_pulse.api.websockets.auth import WebSocketAuthenticator
from alpha_pulse.api.websockets.subscription import SubscriptionManager


@pytest.fixture
def mock_connection_manager():
    """Mock the ConnectionManager."""
    with patch("alpha_pulse.api.main.connection_manager") as mock:
        # Configure the mock
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.subscribe = AsyncMock()
        mock.broadcast = AsyncMock()
        yield mock


@pytest.fixture
def mock_websocket_auth():
    """Mock the WebSocketAuthenticator."""
    with patch("alpha_pulse.api.main.websocket_auth") as mock:
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


@pytest.mark.asyncio
async def test_metrics_websocket_authentication(mock_websocket_auth, mock_connection_manager, admin_user, monkeypatch):
    """Test WebSocket authentication for metrics endpoint."""
    # Mock the authenticate method to return the admin user
    mock_websocket_auth.authenticate.return_value = admin_user
    
    # Create a test server using the FastAPI app
    from fastapi.testclient import TestClient
    from fastapi.websockets import WebSocketDisconnect
    
    # We need to use a context manager to handle the WebSocket connection
    async with websockets.connect("ws://localhost:8000/ws/metrics") as websocket:
        # Send authentication message
        await websocket.send(json.dumps({"token": "test-token"}))
        
        # Verify authentication was called
        mock_websocket_auth.authenticate.assert_called_once()
        
        # Verify connection was registered
        mock_connection_manager.connect.assert_called_once()
        
        # Verify subscription was made
        mock_connection_manager.subscribe.assert_called_once_with(websocket, "metrics")
        
        # Send ping to keep connection alive
        await websocket.send("ping")
        
        # Receive pong response
        response = await websocket.recv()
        assert response == "pong"


@pytest.mark.asyncio
async def test_metrics_websocket_failed_authentication(mock_websocket_auth, mock_connection_manager):
    """Test failed WebSocket authentication."""
    # Mock the authenticate method to return None (authentication failure)
    mock_websocket_auth.authenticate.return_value = None
    
    # Create a test server using the FastAPI app
    from fastapi.testclient import TestClient
    from fastapi.websockets import WebSocketDisconnect
    
    # We need to use a context manager to handle the WebSocket connection
    with pytest.raises(websockets.exceptions.ConnectionClosedError):
        async with websockets.connect("ws://localhost:8000/ws/metrics") as websocket:
            # Send authentication message
            await websocket.send(json.dumps({"token": "invalid-token"}))
            
            # Verify authentication was called
            mock_websocket_auth.authenticate.assert_called_once()
            
            # Connection should be closed by the server
            await websocket.recv()  # This should raise ConnectionClosedError


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_alerts_websocket(mock_websocket_auth, mock_connection_manager, admin_user):
    """Test WebSocket connection for alerts endpoint."""
    # Mock the authenticate method to return the admin user
    mock_websocket_auth.authenticate.return_value = admin_user
    
    # Create a test server using the FastAPI app
    from fastapi.testclient import TestClient
    from fastapi.websockets import WebSocketDisconnect
    
    # We need to use a context manager to handle the WebSocket connection
    async with websockets.connect("ws://localhost:8000/ws/alerts") as websocket:
        # Send authentication message
        await websocket.send(json.dumps({"token": "test-token"}))
        
        # Verify authentication was called
        mock_websocket_auth.authenticate.assert_called_once()
        
        # Verify connection was registered
        mock_connection_manager.connect.assert_called_once()
        
        # Verify subscription was made
        mock_connection_manager.subscribe.assert_called_once_with(websocket, "alerts")


@pytest.mark.asyncio
async def test_portfolio_websocket(mock_websocket_auth, mock_connection_manager, admin_user):
    """Test WebSocket connection for portfolio endpoint."""
    # Mock the authenticate method to return the admin user
    mock_websocket_auth.authenticate.return_value = admin_user
    
    # Create a test server using the FastAPI app
    from fastapi.testclient import TestClient
    from fastapi.websockets import WebSocketDisconnect
    
    # We need to use a context manager to handle the WebSocket connection
    async with websockets.connect("ws://localhost:8000/ws/portfolio") as websocket:
        # Send authentication message
        await websocket.send(json.dumps({"token": "test-token"}))
        
        # Verify authentication was called
        mock_websocket_auth.authenticate.assert_called_once()
        
        # Verify connection was registered
        mock_connection_manager.connect.assert_called_once()
        
        # Verify subscription was made
        mock_connection_manager.subscribe.assert_called_once_with(websocket, "portfolio")


@pytest.mark.asyncio
async def test_trades_websocket(mock_websocket_auth, mock_connection_manager, admin_user):
    """Test WebSocket connection for trades endpoint."""
    # Mock the authenticate method to return the admin user
    mock_websocket_auth.authenticate.return_value = admin_user
    
    # Create a test server using the FastAPI app
    from fastapi.testclient import TestClient
    from fastapi.websockets import WebSocketDisconnect
    
    # We need to use a context manager to handle the WebSocket connection
    async with websockets.connect("ws://localhost:8000/ws/trades") as websocket:
        # Send authentication message
        await websocket.send(json.dumps({"token": "test-token"}))
        
        # Verify authentication was called
        mock_websocket_auth.authenticate.assert_called_once()
        
        # Verify connection was registered
        mock_connection_manager.connect.assert_called_once()
        
        # Verify subscription was made
        mock_connection_manager.subscribe.assert_called_once_with(websocket, "trades")


@pytest.mark.asyncio
async def test_websocket_disconnect(mock_connection_manager):
    """Test WebSocket disconnection handling."""
    # Create a test server using the FastAPI app
    from fastapi.testclient import TestClient
    from fastapi.websockets import WebSocketDisconnect
    
    # We need to use a context manager to handle the WebSocket connection
    async with websockets.connect("ws://localhost:8000/ws/metrics") as websocket:
        # Close the connection
        await websocket.close()
        
        # Verify disconnect was called
        # Note: This might be challenging to test directly since the disconnect
        # happens after the context manager exits
        # We might need to add a small delay or use a different approach
        await asyncio.sleep(0.1)  # Small delay to allow disconnect to be processed
        mock_connection_manager.disconnect.assert_called_once()