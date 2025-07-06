"""
Liquidity risk management API endpoints.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime

from alpha_pulse.services.liquidity_risk_service import LiquidityRiskService
from ..dependencies import get_current_user


router = APIRouter()


class LiquidityAssessmentRequest(BaseModel):
    """Request for liquidity assessment."""
    symbol: str = Field(..., description="Trading symbol")
    position_size: float = Field(..., description="Position size to assess")
    current_price: Optional[float] = Field(None, description="Current market price")


class MarketImpactRequest(BaseModel):
    """Request for market impact estimation."""
    symbol: str = Field(..., description="Trading symbol")
    order_size: float = Field(..., description="Order size")
    side: str = Field(..., description="Order side: 'buy' or 'sell'")


class ExecutionStrategyRequest(BaseModel):
    """Request for optimal execution strategy."""
    symbol: str = Field(..., description="Trading symbol")
    total_quantity: float = Field(..., description="Total quantity to execute")
    urgency: str = Field("normal", description="Execution urgency: low, normal, high")
    max_impact: Optional[float] = Field(0.01, description="Maximum acceptable impact")


def get_liquidity_service() -> LiquidityRiskService:
    """Get liquidity risk service instance."""
    # In production, this would get the service from app state
    # For now, return a placeholder
    raise HTTPException(
        status_code=503,
        detail="Liquidity service not initialized. Please check service configuration."
    )


@router.post("/assess")
async def assess_liquidity(
    request: LiquidityAssessmentRequest,
    user=Depends(get_current_user),
    service: LiquidityRiskService = Depends(get_liquidity_service)
):
    """
    Assess liquidity risk for a position.
    
    Returns comprehensive liquidity metrics including:
    - Liquidity score (0-1)
    - Market impact estimates
    - Slippage predictions
    - Execution recommendations
    """
    try:
        assessment = await service.assess_position_liquidity(
            symbol=request.symbol,
            position_size=request.position_size,
            current_price=request.current_price
        )
        
        return {
            "symbol": request.symbol,
            "position_size": request.position_size,
            "assessment": assessment,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/impact")
async def estimate_market_impact(
    request: MarketImpactRequest,
    user=Depends(get_current_user),
    service: LiquidityRiskService = Depends(get_liquidity_service)
):
    """
    Estimate market impact for an order.
    
    Returns:
    - Temporary impact (immediate price movement)
    - Permanent impact (lasting price effect)
    - Total expected impact
    - Confidence intervals
    """
    try:
        impact = await service.estimate_market_impact(
            symbol=request.symbol,
            order_size=request.order_size,
            side=request.side
        )
        
        return {
            "symbol": request.symbol,
            "order_size": request.order_size,
            "side": request.side,
            "impact_analysis": impact,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execution-strategy")
async def get_optimal_execution(
    request: ExecutionStrategyRequest,
    user=Depends(get_current_user),
    service: LiquidityRiskService = Depends(get_liquidity_service)
):
    """
    Get optimal execution strategy for large orders.
    
    Returns recommended execution approach:
    - TWAP (Time-Weighted Average Price)
    - VWAP (Volume-Weighted Average Price)
    - Iceberg orders
    - Implementation shortfall
    """
    try:
        strategy = await service.plan_optimal_execution(
            symbol=request.symbol,
            total_quantity=request.total_quantity,
            urgency=request.urgency,
            max_impact=request.max_impact
        )
        
        return {
            "symbol": request.symbol,
            "total_quantity": request.total_quantity,
            "strategy": strategy,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
async def get_portfolio_liquidity(
    user=Depends(get_current_user),
    service: LiquidityRiskService = Depends(get_liquidity_service)
):
    """
    Get portfolio-level liquidity analysis.
    
    Returns:
    - Overall portfolio liquidity score
    - Position-level liquidity metrics
    - Concentration risks
    - Liquidation time estimates
    """
    try:
        analysis = await service.analyze_portfolio_liquidity()
        
        return {
            "portfolio_liquidity": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{symbol}")
async def get_liquidity_metrics(
    symbol: str,
    user=Depends(get_current_user),
    service: LiquidityRiskService = Depends(get_liquidity_service)
):
    """
    Get current liquidity metrics for a symbol.
    
    Returns:
    - Bid-ask spread
    - Market depth
    - Average daily volume
    - Amihud illiquidity ratio
    - Kyle's lambda
    """
    try:
        metrics = await service.get_liquidity_metrics(symbol)
        
        return {
            "symbol": symbol,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_liquidity_history(
    symbol: str,
    days: int = Query(30, description="Number of days of history"),
    user=Depends(get_current_user),
    service: LiquidityRiskService = Depends(get_liquidity_service)
):
    """
    Get historical liquidity metrics for a symbol.
    
    Returns time series of:
    - Liquidity scores
    - Spread history
    - Volume patterns
    - Impact estimates
    """
    try:
        history = await service.get_liquidity_history(
            symbol=symbol,
            days=days
        )
        
        return {
            "symbol": symbol,
            "days": days,
            "history": history,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stress-test")
async def liquidity_stress_test(
    scenario: str = Query("market_crash", description="Stress scenario type"),
    user=Depends(get_current_user),
    service: LiquidityRiskService = Depends(get_liquidity_service)
):
    """
    Run liquidity stress test on portfolio.
    
    Scenarios:
    - market_crash: Sudden market downturn
    - flash_crash: Extreme short-term volatility
    - liquidity_crisis: Market-wide liquidity dry-up
    """
    try:
        results = await service.stress_test_liquidity(scenario)
        
        return {
            "scenario": scenario,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))