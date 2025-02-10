"""
Router for risk management-related endpoints.
"""
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from decimal import Decimal
from loguru import logger

from alpha_pulse.exchanges.base import BaseExchange
from alpha_pulse.risk_management.manager import RiskManager
from alpha_pulse.risk_management.analysis import RiskAnalyzer
from alpha_pulse.risk_management.position_sizing import PositionSizer
from ..dependencies import get_exchange_client, verify_api_key

router = APIRouter()

class RiskMetricsResponse(BaseModel):
    """Risk metrics response model."""
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    correlation_matrix: Dict[str, Dict[str, float]]

class RiskLimitsResponse(BaseModel):
    """Risk limits response model."""
    position_limits: Dict[str, float]
    margin_limits: Dict[str, float]
    exposure_limits: Dict[str, float]
    drawdown_limits: Dict[str, float]

class PositionSizeResponse(BaseModel):
    """Position size recommendation response model."""
    asset: str
    recommended_size: float
    max_size: float
    risk_per_trade: float
    stop_loss: float
    take_profit: float

@router.get(
    "/exposure",
    response_model=Dict[str, float],
    dependencies=[Depends(verify_api_key)]
)
async def get_risk_exposure(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current risk exposure metrics."""
    try:
        risk_manager = RiskManager(exchange)
        exposure = await risk_manager.calculate_risk_exposure()
        
        return {
            asset: float(value)
            for asset, value in exposure.items()
        }
    except Exception as e:
        logger.error(f"Error calculating risk exposure: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate risk exposure"
        )

@router.get(
    "/metrics",
    response_model=RiskMetricsResponse,
    dependencies=[Depends(verify_api_key)]
)
async def get_risk_metrics(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get detailed risk metrics."""
    try:
        risk_analyzer = RiskAnalyzer(exchange)
        metrics = await risk_analyzer.calculate_risk_metrics()
        
        return RiskMetricsResponse(
            var_95=float(metrics.var_95),
            var_99=float(metrics.var_99),
            expected_shortfall=float(metrics.expected_shortfall),
            sharpe_ratio=float(metrics.sharpe_ratio),
            max_drawdown=float(metrics.max_drawdown),
            beta=float(metrics.beta),
            correlation_matrix={
                asset: {
                    corr_asset: float(corr)
                    for corr_asset, corr in correlations.items()
                }
                for asset, correlations in metrics.correlation_matrix.items()
            }
        )
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate risk metrics"
        )

@router.get(
    "/limits",
    response_model=RiskLimitsResponse,
    dependencies=[Depends(verify_api_key)]
)
async def get_risk_limits(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current risk limits."""
    try:
        risk_manager = RiskManager(exchange)
        limits = await risk_manager.get_risk_limits()
        
        return RiskLimitsResponse(
            position_limits={
                asset: float(limit)
                for asset, limit in limits.position_limits.items()
            },
            margin_limits={
                asset: float(limit)
                for asset, limit in limits.margin_limits.items()
            },
            exposure_limits={
                asset: float(limit)
                for asset, limit in limits.exposure_limits.items()
            },
            drawdown_limits={
                asset: float(limit)
                for asset, limit in limits.drawdown_limits.items()
            }
        )
    except Exception as e:
        logger.error(f"Error fetching risk limits: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch risk limits"
        )

@router.get(
    "/position-size/{asset}",
    response_model=PositionSizeResponse,
    dependencies=[Depends(verify_api_key)]
)
async def get_position_size_recommendation(
    asset: str,
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get position size recommendation for an asset."""
    try:
        position_sizer = PositionSizer(exchange)
        recommendation = await position_sizer.get_position_size_recommendation(asset)
        
        return PositionSizeResponse(
            asset=asset,
            recommended_size=float(recommendation.recommended_size),
            max_size=float(recommendation.max_size),
            risk_per_trade=float(recommendation.risk_per_trade),
            stop_loss=float(recommendation.stop_loss),
            take_profit=float(recommendation.take_profit)
        )
    except Exception as e:
        logger.error(f"Error calculating position size for {asset}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate position size for {asset}"
        )