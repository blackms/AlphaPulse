"""
Dependency injection and utilities for the AlphaPulse API.
"""
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from loguru import logger

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: Annotated[str, Depends(api_key_header)]) -> str:
    """
    Verify the API key from the request header.
    TODO: Implement proper API key verification against a secure store.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required"
        )
    # TODO: Implement actual API key verification
    return api_key

async def get_exchange_client():
    """
    Get an initialized exchange client.
    This dependency can be used by endpoints that need to interact with exchanges.
    """
    try:
        from alpha_pulse.exchanges.bybit import BybitExchange
        exchange = BybitExchange(testnet=True)  # TODO: Configure based on environment
        await exchange.initialize()
        return exchange
    except Exception as e:
        logger.error(f"Failed to initialize exchange client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exchange service unavailable"
        )

# Common response models and utilities can be added here