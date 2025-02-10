"""
Router for position-related endpoints.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from loguru import logger

from alpha_pulse.exchanges.base import BaseExchange
from alpha_pulse.hedging.common.types import SpotPosition, FuturesPosition
from alpha_pulse.hedging.execution.position_fetcher import ExchangePositionFetcher
from ..dependencies import get_exchange_client, verify_api_key

router = APIRouter()

class PositionMetrics(BaseModel):
    """Position metrics response model."""
    symbol: str
    spot_value: float
    spot_qty: float
    futures_value: float
    futures_qty: float
    net_exposure: float
    hedge_ratio: float

@router.get(
    "/spot",
    response_model=List[SpotPosition],
    dependencies=[Depends(verify_api_key)]
)
async def get_spot_positions(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current spot positions."""
    try:
        position_fetcher = ExchangePositionFetcher(exchange)
        positions = await position_fetcher.get_spot_positions()
        return positions
    except Exception as e:
        logger.error(f"Error fetching spot positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch spot positions"
        )

@router.get(
    "/futures",
    response_model=List[FuturesPosition],
    dependencies=[Depends(verify_api_key)]
)
async def get_futures_positions(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current futures positions."""
    try:
        position_fetcher = ExchangePositionFetcher(exchange)
        positions = await position_fetcher.get_futures_positions()
        return positions
    except Exception as e:
        logger.error(f"Error fetching futures positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch futures positions"
        )

@router.get(
    "/metrics",
    response_model=List[PositionMetrics],
    dependencies=[Depends(verify_api_key)]
)
async def get_position_metrics(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get detailed metrics for all positions."""
    try:
        position_fetcher = ExchangePositionFetcher(exchange)
        spot_positions = await position_fetcher.get_spot_positions()
        futures_positions = await position_fetcher.get_futures_positions()
        
        from alpha_pulse.examples.demo_hedging import calculate_asset_metrics
        metrics_dict = calculate_asset_metrics(spot_positions, futures_positions)
        
        # Convert dictionary to list of PositionMetrics
        metrics = [
            PositionMetrics(
                symbol=symbol,
                spot_value=float(data["spot_value"]),
                spot_qty=float(data["spot_qty"]),
                futures_value=float(data["futures_value"]),
                futures_qty=float(data["futures_qty"]),
                net_exposure=float(data["net_exposure"]),
                hedge_ratio=float(data["hedge_ratio"])
            )
            for symbol, data in metrics_dict.items()
        ]
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating position metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate position metrics"
        )