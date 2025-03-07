"""
Dependency injection and utilities for the AlphaPulse API.
"""
from datetime import datetime, timedelta
from typing import Annotated, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from loguru import logger

from .config import config
from alpha_pulse.exchanges.bybit import BybitExchange

# Authentication schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name="X-API-Key")


def create_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=config.auth.token_expiry)
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, config.auth.jwt_secret, algorithm="HS256")


async def authenticate_token(token: str) -> Optional[Dict]:
    """Authenticate JWT token."""
    try:
        payload = jwt.decode(token, config.auth.jwt_secret, algorithms=["HS256"])
        if payload.get("exp") < datetime.utcnow().timestamp():
            return None
        return payload
    except jwt.PyJWTError:
        return None


async def authenticate_api_key(api_key: str) -> Optional[Dict]:
    """Authenticate API key."""
    if not config.auth.api_keys_enabled:
        return None
        
    if api_key not in config.auth.api_keys:
        return None
        
    username = config.auth.api_keys[api_key]
    return {"username": username, "role": "api", "sub": username}


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token."""
    user = await authenticate_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_api_client(api_key: str = Depends(api_key_header)):
    """Get API client from API key."""
    client = await authenticate_api_key(api_key)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return client


def has_permission(user: Dict, permission: str) -> bool:
    """Check if user has permission."""
    # Simple role-based permission check
    role = user.get("role", "viewer")
    
    # Define role-based permissions
    role_permissions = {
        "admin": ["view_metrics", "view_portfolio", "view_alerts", "acknowledge_alerts", 
                 "view_trades", "execute_trades", "view_system"],
        "operator": ["view_metrics", "view_portfolio", "view_alerts", "acknowledge_alerts", 
                     "view_trades"],
        "viewer": ["view_metrics", "view_portfolio", "view_alerts"],
        "api": ["view_metrics", "view_portfolio", "view_alerts", "view_trades"]
    }
    
    return permission in role_permissions.get(role, [])


# Exchange client instance (initialized once)
_exchange: Optional[BybitExchange] = None

async def get_exchange_client() -> BybitExchange:
    """
    Get an initialized exchange client.
    This dependency can be used by endpoints that need to interact with exchanges.
    """
    global _exchange
    
    try:
        if _exchange is None:
            _exchange = BybitExchange(testnet=config.exchange.testnet)
            await _exchange.initialize()
        return _exchange
    except Exception as e:
        logger.error(f"Failed to initialize exchange client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exchange service unavailable"
        )


# Cleanup function to close exchange connection
async def cleanup_exchange():
    """Cleanup exchange connection on shutdown."""
    global _exchange
    if _exchange:
        await _exchange.close()
        _exchange = None