"""
Dependencies for the API.

This module provides dependencies for the FastAPI application.
"""
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader

from alpha_pulse.monitoring.alerting import AlertManager, load_alerting_config

from .data import (
    MetricsDataAccessor,
    AlertDataAccessor,
    PortfolioDataAccessor,
    TradeDataAccessor,
    SystemDataAccessor
)
from .auth import get_current_user, oauth2_scheme

logger = logging.getLogger(__name__)

# Authentication schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load alerting configuration
alerting_config = load_alerting_config()
alert_manager = AlertManager(alerting_config)

# Create data accessor instances
metric_accessor = MetricsDataAccessor()
alert_accessor = AlertDataAccessor(alert_manager)
portfolio_accessor = PortfolioDataAccessor()
trade_accessor = TradeDataAccessor()
system_accessor = SystemDataAccessor()


# This function is now imported from auth.py


async def get_api_key_user(
    api_key: Optional[str] = Depends(api_key_header)
) -> Dict[str, Any]:
    """
    Get the user from the API key.
    
    In a real implementation, this would validate the API key
    and return the user information.
    """
    if not api_key:
        return None
    
    # For testing purposes, return a mock admin user
    return {
        "username": "api_user",
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


async def get_user(
    request: Request,
    token_user: Optional[Dict[str, Any]] = Depends(get_current_user),
    api_key_user: Optional[Dict[str, Any]] = Depends(get_api_key_user)
) -> Dict[str, Any]:
    """
    Get the user from either token or API key.
    
    This dependency will try to authenticate using JWT token first,
    and if that fails, it will try API key authentication.
    """
    # Add permissions to the user object
    if token_user:
        token_user["permissions"] = [
            "view_metrics",
            "view_alerts",
            "acknowledge_alerts",
            "view_portfolio",
            "view_trades",
            "view_system"
        ]
        return token_user
        
    if api_key_user:
        return api_key_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
    )


def check_permission(permission: str):
    """
    Check if the user has the required permission.
    
    This is a dependency factory that creates a dependency
    to check if the user has a specific permission.
    """
    async def _check_permission(user: Dict[str, Any] = Depends(get_user)) -> Dict[str, Any]:
        # For testing purposes, allow all permissions
        return user
    return _check_permission


# Specific permission checks
require_view_metrics = check_permission("view_metrics")
require_view_alerts = check_permission("view_alerts")
require_acknowledge_alerts = check_permission("acknowledge_alerts")
require_view_portfolio = check_permission("view_portfolio")
require_view_trades = check_permission("view_trades")
require_view_system = check_permission("view_system")


# Data accessor dependencies
def get_metric_accessor():
    """Get the metric data accessor."""
    return metric_accessor


def get_alert_accessor():
    """Get the alert data accessor."""
    return alert_accessor


def get_portfolio_accessor():
    """Get the portfolio data accessor."""
    return portfolio_accessor


def get_trade_accessor():
    """Get the trade data accessor."""
    return trade_accessor


def get_system_accessor():
    """Get the system data accessor."""
    return system_accessor


def get_alert_manager():
    """Get the alert manager instance."""
    return alert_manager