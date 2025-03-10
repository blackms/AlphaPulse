"""
Portfolio router.

This module defines the API endpoints for portfolio data.
"""
import logging
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Query, Body

from ..dependencies import (
    require_view_portfolio,
    get_portfolio_accessor
)
from ..data import PortfolioDataAccessor
from ..exchange_sync_integration import trigger_exchange_sync

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/portfolio")
async def get_portfolio(
    include_history: bool = Query(False, description="Include portfolio history"),
    refresh: bool = Query(False, description="Force refresh from exchange"),
    _: Dict[str, Any] = Depends(require_view_portfolio),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    """
    Get portfolio data.
    
    Args:
        include_history: Whether to include historical data
        refresh: Whether to force a refresh from the exchange
    
    Returns:
        Portfolio data
    """
    try:
        # If refresh is requested, get data directly from exchange
        if refresh:
            return await portfolio_accessor._get_portfolio_from_exchange()
        
        return await portfolio_accessor.get_portfolio(include_history=include_history)
    except Exception as e:
        logger.error(f"Error getting portfolio data: {e}")
        return {"error": str(e), "total_value": 0, "cash": 0, "positions": []}


# Create a development endpoint for reloading data without authentication
@router.post("/debug/reload")
async def reload_exchange_data_debug(
    data_type: Optional[str] = Query(None, description="Type of data to reload (orders, balances, positions, prices, all)"),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    """
    Force reload of exchange data (DEBUG endpoint, no authentication required).
    
    This endpoint is intended for development and testing purposes only.
    It allows triggering an immediate data refresh without authentication.
    
    Args:
        data_type: Optional type of data to reload (orders, balances, positions, prices, all)
                  If not specified, all data types will be reloaded
    
    Returns:
        Status information about the reload operation
    """
    # Use the exchange_sync integration to trigger a sync
    result = await trigger_exchange_sync(portfolio_accessor._exchange_id)
    return result


@router.post("/portfolio/reload")
async def reload_exchange_data(
    _: Dict[str, Any] = Depends(require_view_portfolio),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    return await trigger_exchange_sync(portfolio_accessor._exchange_id)