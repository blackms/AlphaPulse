"""
Portfolio router.

This module defines the API endpoints for portfolio data.
"""
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Query

from ..dependencies import (
    require_view_portfolio,
    get_portfolio_accessor
)
from ..data import PortfolioDataAccessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/portfolio")
async def get_portfolio(
    include_history: bool = Query(False, description="Include portfolio history"),
    _: Dict[str, Any] = Depends(require_view_portfolio),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    """
    Get portfolio data.
    
    Args:
        include_history: Whether to include historical data
    
    Returns:
        Portfolio data
    """
    return await portfolio_accessor.get_portfolio(include_history=include_history)