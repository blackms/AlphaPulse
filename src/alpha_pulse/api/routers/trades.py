"""
Trades router.

This module defines the API endpoints for trade data.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query

from ..dependencies import (
    require_view_trades,
    get_trade_accessor
)
from ..data import TradeDataAccessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/trades")
async def get_trades(
    symbol: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    _: Dict[str, Any] = Depends(require_view_trades),
    trade_accessor: TradeDataAccessor = Depends(get_trade_accessor)
) -> List[Dict[str, Any]]:
    """
    Get trade data.
    
    Args:
        symbol: Filter by symbol
        start_time: Start time for the query
        end_time: End time for the query
    
    Returns:
        List of trades
    """
    return await trade_accessor.get_trades(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time
    )