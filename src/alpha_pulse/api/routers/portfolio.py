"""
Router for portfolio-related endpoints.
"""
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from decimal import Decimal
from loguru import logger
import pandas as pd

from alpha_pulse.exchanges.interfaces import BaseExchange
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from pathlib import Path
from alpha_pulse.portfolio.data_models import PortfolioMetrics
from ..dependencies import get_exchange_client, get_api_client

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
    dependencies=[Depends(get_api_client)]
)
async def get_portfolio_analysis(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get detailed portfolio analysis."""
    try:
        # Initialize portfolio manager with config
        config_path = Path(__file__).parent.parent.parent / "portfolio" / "portfolio_config.yaml"
        portfolio_manager = PortfolioManager(str(config_path))
        
        # Get current allocation
        current_allocation = await portfolio_manager.get_current_allocation(exchange)
        
        # Get historical data for analysis
        historical_data = await portfolio_manager._fetch_historical_data(exchange, list(current_allocation.keys()))
        
        # Get target allocation
        target = portfolio_manager.strategy.compute_target_allocation(
            current_allocation,
            historical_data,
            portfolio_manager.risk_constraints
        )
        
        # Get portfolio data for risk metrics
        portfolio_data = await portfolio_manager.get_portfolio_data(exchange)
        
        return PortfolioAnalysisResponse(
            allocation={k: float(v) for k, v in current_allocation.items()},
            risk_metrics=portfolio_data.risk_metrics,
            performance_metrics={
                "daily_returns": 0.0,  # TODO: Calculate from historical data
                "weekly_returns": 0.0,
                "monthly_returns": 0.0,
                "sharpe_ratio": float(portfolio_data.risk_metrics.get('sharpe_ratio', 0.0))
            },
            recommendations=[
                f"Target allocation differs from current for {asset}: {float(target.get(asset, 0)) - float(current_allocation.get(asset, 0)):.2%}"
                for asset in set(target) | set(current_allocation)
                if abs(float(target.get(asset, 0)) - float(current_allocation.get(asset, 0))) > 0.01
            ]
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
    dependencies=[Depends(get_api_client)]
)
async def get_portfolio_metrics(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current portfolio metrics."""
    try:
        # Initialize portfolio manager with config
        config_path = Path(__file__).parent.parent.parent / "portfolio" / "portfolio_config.yaml"
        portfolio_manager = PortfolioManager(str(config_path))
        
        # Get portfolio data
        portfolio_data = await portfolio_manager.get_portfolio_data(exchange)
        
        # Extract metrics from portfolio data
        return PortfolioMetricsResponse(
            total_value=float(portfolio_data.total_value),
            pnl_24h=0.0,  # TODO: Calculate from historical data
            pnl_7d=0.0,   # TODO: Calculate from historical data
            risk_level="MODERATE",  # TODO: Calculate from risk metrics
            sharpe_ratio=float(portfolio_data.risk_metrics.get('sharpe_ratio', 0.0)),
            volatility=float(portfolio_data.risk_metrics.get('volatility_target', 0.15)),
            max_drawdown=float(portfolio_data.risk_metrics.get('max_drawdown_limit', 0.25))
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
    dependencies=[Depends(get_api_client)]
)
async def get_portfolio_performance(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get historical portfolio performance."""
    try:
        # Initialize portfolio manager with config
        config_path = Path(__file__).parent.parent.parent / "portfolio" / "portfolio_config.yaml"
        portfolio_manager = PortfolioManager(str(config_path))
        
        # Get current portfolio data
        portfolio_data = await portfolio_manager.get_portfolio_data(exchange)
        
        # Get historical data for all assets
        historical_data = await portfolio_manager._fetch_historical_data(
            exchange,
            [pos.asset_id for pos in portfolio_data.positions]
        )
        
        try:
            # Calculate daily portfolio values
            performance = {}
            for date_str in historical_data.index:
                # Convert string date to datetime if needed
                date = date_str if isinstance(date_str, str) else date_str.strftime("%Y-%m-%d")
                daily_value = 0.0
                try:
                    for pos in portfolio_data.positions:
                        if pos.asset_id in historical_data.columns:
                            value = historical_data.loc[date_str, pos.asset_id]
                            if not pd.isna(value):
                                daily_value += float(pos.quantity) * value
                    
                    # Only include non-zero values
                    if daily_value > 0:
                        performance[date] = daily_value
                except Exception as e:
                    logger.warning(f"Error calculating daily value for {date}: {str(e)}")
                    continue
            
            return performance
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to calculate portfolio performance"
            )
    except Exception as e:
        logger.error(f"Error fetching portfolio performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch portfolio performance"
        )