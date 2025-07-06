"""
Dependencies for the API.

This module provides dependencies for the FastAPI application.
"""
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader

from alpha_pulse.monitoring.alerting import AlertManager, load_alerting_config
from alpha_pulse.risk_management import RiskManager, RiskConfig
from alpha_pulse.exchanges.exchange_adapters import PaperTradingAdapter

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
        # Check if the user has the required permission
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this resource",
            )
        return user  # Return the user if authorized
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


# Risk manager instance (singleton)
_risk_manager = None


def get_risk_manager(request: Request):
    """Get the risk manager instance."""
    global _risk_manager
    if _risk_manager is None:
        # Use paper trading adapter for API
        exchange = PaperTradingAdapter()
        risk_config = RiskConfig(
            max_position_size=0.2,
            max_portfolio_leverage=1.5,
            max_drawdown=0.25,
            stop_loss=0.1,
            var_confidence=0.95,
            risk_free_rate=0.0,
            target_volatility=0.15,
            rebalance_threshold=0.1,
            initial_portfolio_value=100000.0
        )
        
        # Get risk budgeting service if available
        risk_budgeting_service = None
        if hasattr(request.app.state, 'risk_budgeting_service'):
            risk_budgeting_service = request.app.state.risk_budgeting_service
        
        _risk_manager = RiskManager(
            exchange=exchange, 
            config=risk_config,
            risk_budgeting_service=risk_budgeting_service
        )
    return _risk_manager


def get_risk_budgeting_service(request: Request):
    """Get the risk budgeting service instance from app state."""
    if hasattr(request.app.state, 'risk_budgeting_service'):
        return request.app.state.risk_budgeting_service
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Risk budgeting service not available"
    )


def get_regime_detection_service(request: Request):
    """Get the regime detection service instance from app state."""
    if hasattr(request.app.state, 'regime_detection_service'):
        return request.app.state.regime_detection_service
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Regime detection service not available"
    )


def get_gpu_service(request: Request):
    """Get the GPU service instance from app state."""
    if hasattr(request.app.state, 'gpu_service') and request.app.state.gpu_service:
        return request.app.state.gpu_service
    return None  # GPU service is optional - return None if not available


# Agent manager instance (singleton)
_agent_manager = None


def get_agent_manager(request: Request):
    """Get the agent manager instance."""
    global _agent_manager
    if _agent_manager is None:
        from alpha_pulse.agents.manager import AgentManager
        
        # Get ensemble service if available
        ensemble_service = None
        if hasattr(request.app.state, 'ensemble_service'):
            ensemble_service = request.app.state.ensemble_service
        
        # Get GPU service if available
        gpu_service = None
        if hasattr(request.app.state, 'gpu_service'):
            gpu_service = request.app.state.gpu_service
        
        # Initialize agent manager with ensemble and GPU integration
        config = {
            "use_ensemble": True,
            "use_gpu_acceleration": True,
            "agent_weights": {
                "activist": 0.15,
                "value": 0.20,
                "fundamental": 0.20,
                "sentiment": 0.15,
                "technical": 0.15,
                "valuation": 0.15
            }
        }
        
        _agent_manager = AgentManager(
            config=config, 
            ensemble_service=ensemble_service,
            gpu_service=gpu_service
        )
    
    return _agent_manager