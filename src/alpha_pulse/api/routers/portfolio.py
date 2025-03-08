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
from alpha_pulse.data_pipeline.scheduler import DataType

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/portfolio")
async def get_portfolio(
    include_history: bool = Query(False, description="Include portfolio history"),
    use_cache: bool = Query(True, description="Use cached data if available"),
    _: Dict[str, Any] = Depends(require_view_portfolio),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    """
    Get portfolio data.
    
    Args:
        include_history: Whether to include historical data
        use_cache: Whether to use cached data (if available) or force direct exchange fetch
    
    Returns:
        Portfolio data
    """
    return await portfolio_accessor.get_portfolio(include_history=include_history, use_cache=use_cache)


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
    # Convert string data_type to enum if provided
    enum_data_type = None
    if data_type:
        try:
            enum_data_type = DataType(data_type.lower())
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid data type: {data_type}. Valid types: {', '.join([t.value for t in DataType])}"
            }
    
    # Trigger the reload
    result = await portfolio_accessor.reload_data(enum_data_type)
    result["endpoint"] = "debug"
    return result


@router.post("/portfolio/reload")
async def reload_exchange_data(
    data_type: Optional[str] = Query(None, description="Type of data to reload (orders, balances, positions, prices, all)"),
    _: Dict[str, Any] = Depends(require_view_portfolio),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    """
    Force reload of exchange data.
    
    This endpoint triggers an immediate fetch of the latest data from the exchange and updates the cache.
    The operation runs asynchronously, so the API returns immediately while the data sync happens in the background.
    
    Args:
        data_type: Optional type of data to reload (orders, balances, positions, prices, all)
                  If not specified, all data types will be reloaded
    
    Returns:
        Status information about the reload operation
    """
    # Convert string data_type to enum if provided
    enum_data_type = None
    if data_type:
        try:
            enum_data_type = DataType(data_type.lower())
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid data type: {data_type}. Valid types: {', '.join([t.value for t in DataType])}"
            }
    
    # Trigger the reload
    return await portfolio_accessor.reload_data(enum_data_type)