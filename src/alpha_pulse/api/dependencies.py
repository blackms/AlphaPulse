"""
Dependency injection and utilities for the AlphaPulse API.
"""
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from loguru import logger

from .config import config
from alpha_pulse.exchanges.bybit import BybitExchange

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: Annotated[str, Depends(api_key_header)]) -> str:
    """Verify the API key from the request header."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required"
        )
    
    if api_key not in config.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return config.api_keys[api_key]

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
            _exchange = BybitExchange(testnet=False)  # Credentials handled by credentials_manager
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