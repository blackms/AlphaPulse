"""Shared fixtures for API tests."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import contextmanager

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user, get_user


@pytest.fixture
def client():
    """Return a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def admin_user():
    """Generate admin user authentication."""
    return {
        "username": "admin",
        "role": "admin",
        "permissions": [
            "view_metrics", 
            "view_alerts", 
            "acknowledge_alerts",
            "view_portfolio",
            "view_trades",
            "view_system"
        ]
    }


@pytest.fixture
def operator_user():
    """Generate operator user authentication."""
    return {
        "username": "operator",
        "role": "operator",
        "permissions": [
            "view_metrics", 
            "view_alerts", 
            "acknowledge_alerts",
            "view_portfolio",
            "view_trades",
            "view_system"
        ]
    }


@pytest.fixture
def trader_user():
    """Generate trader user authentication."""
    return {
        "username": "trader",
        "role": "trader",
        "permissions": [
            "view_metrics", 
            "view_alerts",
            "view_portfolio",
            "view_trades",
            "execute_trades"
        ]
    }


@pytest.fixture
def viewer_user():
    """Generate viewer user authentication."""
    return {
        "username": "viewer",
        "role": "viewer",
        "permissions": [
            "view_metrics",
            "view_alerts",
            "view_portfolio"
        ]
    }


@pytest.fixture
def restricted_user():
    """Generate restricted user authentication."""
    return {
        "username": "restricted",
        "role": "restricted",
        "permissions": []
    }


@pytest.fixture
def auth_override():
    """Override the get_current_user and get_user dependencies."""
    @contextmanager
    def _override_dependency(user):
        app.dependency_overrides = {
            get_current_user: lambda: user,
            get_user: lambda: user # Simplified override
        }
        try:
            yield
        finally:
            app.dependency_overrides = {}
    return _override_dependency


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
        "permissions": [
            "view_metrics", 
            "view_alerts", 
            "acknowledge_alerts",
            "view_portfolio",
            "view_trades",
            "view_system"
        ],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    
    # Generate token
    return jwt.encode(payload, secret_key, algorithm="HS256")


@pytest.fixture
def sample_datetime():
    """Return a fixed datetime for consistent testing."""
    return datetime(2025, 3, 7, 12, 0, 0)