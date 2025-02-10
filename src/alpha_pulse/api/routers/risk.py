"""
Router for risk management-related endpoints.
"""
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from decimal import Decimal
import pandas as pd
from loguru import logger

from alpha_pulse.exchanges.base import BaseExchange
from alpha_pulse.exchanges import OHLCV
from alpha_pulse.risk_management.manager import RiskManager, RiskConfig
from alpha_pulse.risk_management.analysis import RiskAnalyzer
from alpha_pulse.risk_management.position_sizing import PositionSizer
from ..dependencies import get_exchange_client, verify_api_key

router = APIRouter()

class RiskMetricsResponse(BaseModel):
    """Risk metrics response model."""
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

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
        risk_manager = RiskManager(
            exchange=exchange,
            config=RiskConfig(
                max_position_size=0.2,
                max_portfolio_leverage=1.5,
                max_drawdown=0.25,
                target_volatility=0.15
            )
        )
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
        # Get historical data for risk calculations
        candles = await exchange.fetch_ohlcv(
            symbol="BTC/USDT",  # Use BTC as market proxy
            timeframe="1d",
            limit=252  # One year of daily data
        )
        
        # Calculate returns
        prices = pd.Series([float(c.close) for c in candles])
        returns = prices.pct_change().dropna()
        
        # Calculate risk metrics
        metrics = risk_analyzer.calculate_metrics(returns)
        
        return RiskMetricsResponse(
            volatility=float(metrics.volatility),
            var_95=float(metrics.var_95),
            cvar_95=float(metrics.cvar_95),
            max_drawdown=float(metrics.max_drawdown),
            sharpe_ratio=float(metrics.sharpe_ratio),
            sortino_ratio=float(metrics.sortino_ratio),
            calmar_ratio=float(metrics.calmar_ratio)
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
        risk_manager = RiskManager(
            exchange=exchange,
            config=RiskConfig(
                max_position_size=0.2,
                max_portfolio_leverage=1.5,
                max_drawdown=0.25,
                target_volatility=0.15
            )
        )
        
        # Get portfolio value as Decimal
        portfolio_value = await exchange.get_portfolio_value()
        portfolio_value_float = float(portfolio_value)
        
        # Get position limits
        position_limits = risk_manager.get_position_limits(portfolio_value_float)
        
        # Get risk report for additional limits
        risk_report = risk_manager.get_risk_report()
        
        # Calculate limits using float values to avoid decimal/float operation issues
        return RiskLimitsResponse(
            position_limits={"default": float(position_limits["default"])},
            margin_limits={"total": float(portfolio_value) * risk_manager.config.max_portfolio_leverage},
            exposure_limits={"total": float(portfolio_value)},
            drawdown_limits={"max": float(portfolio_value) * risk_manager.config.max_drawdown}
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