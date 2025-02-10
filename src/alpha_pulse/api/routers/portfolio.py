"""
Router for portfolio-related endpoints.
"""
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from decimal import Decimal
from loguru import logger

from alpha_pulse.exchanges.base import BaseExchange
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.portfolio.data_models import PortfolioAnalysis, PortfolioMetrics
from ..dependencies import get_exchange_client, verify_api_key

router = APIRouter()

class PortfolioAnalysisResponse(BaseModel):
    """Portfolio analysis response model."""
    allocation: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    recommendations: List[str]

class PortfolioMetricsResponse(BaseModel):
    """Portfolio metrics response model."""
    total_value: float
    pnl_24h: float
    pnl_7d: float
    risk_level: str
    sharpe_ratio: float
    volatility: float
    max_drawdown: float

@router.get(
    "/analysis",
    response_model=PortfolioAnalysisResponse,
    dependencies=[Depends(verify_api_key)]
)
async def get_portfolio_analysis(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get detailed portfolio analysis."""
    try:
        portfolio_manager = PortfolioManager(exchange)
        analysis = await portfolio_manager.analyze_portfolio()
        
        return PortfolioAnalysisResponse(
            allocation=analysis.allocation,
            risk_metrics=analysis.risk_metrics,
            performance_metrics=analysis.performance_metrics,
            recommendations=analysis.recommendations
        )
    except Exception as e:
        logger.error(f"Error performing portfolio analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform portfolio analysis"
        )

@router.get(
    "/metrics",
    response_model=PortfolioMetricsResponse,
    dependencies=[Depends(verify_api_key)]
)
async def get_portfolio_metrics(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current portfolio metrics."""
    try:
        portfolio_manager = PortfolioManager(exchange)
        metrics = await portfolio_manager.get_portfolio_metrics()
        
        return PortfolioMetricsResponse(
            total_value=float(metrics.total_value),
            pnl_24h=float(metrics.pnl_24h),
            pnl_7d=float(metrics.pnl_7d),
            risk_level=metrics.risk_level,
            sharpe_ratio=float(metrics.sharpe_ratio),
            volatility=float(metrics.volatility),
            max_drawdown=float(metrics.max_drawdown)
        )
    except Exception as e:
        logger.error(f"Error fetching portfolio metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch portfolio metrics"
        )

@router.get(
    "/performance",
    response_model=Dict[str, float],
    dependencies=[Depends(verify_api_key)]
)
async def get_portfolio_performance(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get historical portfolio performance."""
    try:
        portfolio_manager = PortfolioManager(exchange)
        performance = await portfolio_manager.get_historical_performance()
        
        # Convert Decimal values to float for JSON serialization
        return {
            timestamp: float(value) 
            for timestamp, value in performance.items()
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch portfolio performance"
        )