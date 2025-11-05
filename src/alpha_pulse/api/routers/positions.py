"""
Router for position-related endpoints.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from loguru import logger

from alpha_pulse.exchanges.interfaces import BaseExchange
from alpha_pulse.hedging.common.types import SpotPosition, FuturesPosition
from alpha_pulse.hedging.execution.position_fetcher import ExchangePositionFetcher
from ..dependencies import get_exchange_client, get_api_client, get_current_user
from ..middleware.tenant_context import get_current_tenant_id

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
    response_model=List[SpotPosition]
)
async def get_spot_positions(
    tenant_id: str = Depends(get_current_tenant_id),
    user: dict = Depends(get_current_user),
    exchange: BaseExchange = Depends(get_exchange_client),
    _api_client = Depends(get_api_client)
):
    """Get current spot positions."""
    try:
        logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Getting spot positions")

        position_fetcher = ExchangePositionFetcher(exchange)
        positions = await position_fetcher.get_spot_positions()

        logger.info(f"[Tenant: {tenant_id}] Spot positions retrieved: count={len(positions)}")
        return positions
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error fetching spot positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch spot positions"
        )

@router.get(
    "/futures",
    response_model=List[FuturesPosition]
)
async def get_futures_positions(
    tenant_id: str = Depends(get_current_tenant_id),
    user: dict = Depends(get_current_user),
    exchange: BaseExchange = Depends(get_exchange_client),
    _api_client = Depends(get_api_client)
):
    """Get current futures positions."""
    try:
        logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Getting futures positions")

        position_fetcher = ExchangePositionFetcher(exchange)
        positions = await position_fetcher.get_futures_positions()

        logger.info(f"[Tenant: {tenant_id}] Futures positions retrieved: count={len(positions)}")
        return positions
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error fetching futures positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch futures positions"
        )

@router.get(
    "/metrics",
    response_model=List[PositionMetrics]
)
async def get_position_metrics(
    tenant_id: str = Depends(get_current_tenant_id),
    user: dict = Depends(get_current_user),
    exchange: BaseExchange = Depends(get_exchange_client),
    _api_client = Depends(get_api_client)
):
    """Get detailed metrics for all positions."""
    try:
        logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Getting position metrics")

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

        logger.info(f"[Tenant: {tenant_id}] Position metrics calculated: symbols={len(metrics)}")
        return metrics
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error calculating position metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate position metrics"
        )