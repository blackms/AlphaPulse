"""
Risk budget API endpoints.

Provides endpoints for dynamic risk budgeting including current allocations,
utilization monitoring, rebalancing recommendations, and regime-based adjustments.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel, Field
from loguru import logger

from alpha_pulse.api.dependencies import get_current_user, get_risk_budgeting_service
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
from alpha_pulse.models.risk_budget import (
    RiskBudget, RiskBudgetType, AllocationMethod,
    RiskAllocation, RiskBudgetRebalancing, RiskBudgetSnapshot
)
from alpha_pulse.models.market_regime import RegimeType

router = APIRouter()


class RiskBudgetResponse(BaseModel):
    """Response model for risk budget information."""
    budget_type: str
    allocation_method: str
    total_budget: float
    utilized_budget: float
    utilization_rate: float
    allocations: List[Dict[str, Any]]
    volatility_target: float
    leverage_limit: float
    last_updated: datetime
    

class RiskBudgetUtilizationResponse(BaseModel):
    """Response model for budget utilization metrics."""
    total_utilization: float
    by_strategy: Dict[str, float]
    by_asset_class: Dict[str, float]
    available_budget: float
    warnings: List[str]
    

class RebalancingRecommendation(BaseModel):
    """Model for rebalancing recommendations."""
    requires_rebalancing: bool
    reason: Optional[str]
    target_allocations: Dict[str, float]
    current_allocations: Dict[str, float]
    expected_improvement: Dict[str, float]
    priority: str
    

class RiskBudgetHistoryResponse(BaseModel):
    """Response model for historical risk budget data."""
    snapshots: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    regime_changes: List[Dict[str, Any]]


@router.get("/current", response_model=RiskBudgetResponse)
async def get_current_budget(
    budget_type: Optional[str] = Query(None, description="Specific budget type"),
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    risk_budgeting_service = Depends(get_risk_budgeting_service)
) -> RiskBudgetResponse:
    """
    Get current risk budget allocations.

    Returns the active risk budget including allocations by strategy,
    volatility targets, and leverage limits.
    """
    try:
        # Get current portfolio budget
        portfolio_budget = await risk_budgeting_service.get_portfolio_budget()

        if not portfolio_budget:
            logger.warning(f"[Tenant: {tenant_id}] No active risk budget found")
            raise HTTPException(
                status_code=404,
                detail="No active risk budget found"
            )

        # Calculate utilization
        utilized = sum(alloc.utilized_amount for alloc in portfolio_budget.allocations)
        utilization_rate = utilized / portfolio_budget.total_budget if portfolio_budget.total_budget > 0 else 0

        # Format allocations
        allocations = [
            {
                "entity": alloc.entity_id,
                "entity_type": alloc.entity_type,
                "allocated_amount": alloc.allocated_amount,
                "utilized_amount": alloc.utilized_amount,
                "utilization_rate": alloc.utilized_amount / alloc.allocated_amount if alloc.allocated_amount > 0 else 0,
                "constraints": alloc.constraints
            }
            for alloc in portfolio_budget.allocations
        ]

        logger.info(f"[Tenant: {tenant_id}] Retrieved current risk budget")

        return RiskBudgetResponse(
            budget_type=portfolio_budget.budget_type.value,
            allocation_method=portfolio_budget.allocation_method.value,
            total_budget=portfolio_budget.total_budget,
            utilized_budget=utilized,
            utilization_rate=utilization_rate,
            allocations=allocations,
            volatility_target=portfolio_budget.volatility_target.target_volatility,
            leverage_limit=portfolio_budget.leverage_limit,
            last_updated=portfolio_budget.timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting current budget: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/utilization", response_model=RiskBudgetUtilizationResponse)
async def get_budget_utilization(
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    risk_budgeting_service = Depends(get_risk_budgeting_service)
) -> RiskBudgetUtilizationResponse:
    """
    Get current risk budget utilization metrics.
    
    Provides detailed breakdown of how risk budgets are being used
    across strategies and asset classes.
    """
    try:
        # Get utilization metrics
        utilization = await risk_budgeting_service.calculate_utilization()
        
        # Identify warnings
        warnings = []
        if utilization.total_utilization > 0.9:
            warnings.append("Total utilization above 90% - consider reducing positions")
        
        for strategy, util in utilization.by_strategy.items():
            if util > 0.95:
                warnings.append(f"{strategy} utilization above 95%")

        logger.info(f"[Tenant: {tenant_id}] Retrieved budget utilization")

        return RiskBudgetUtilizationResponse(
            total_utilization=utilization.total_utilization,
            by_strategy=utilization.by_strategy,
            by_asset_class=utilization.by_asset_class,
            available_budget=utilization.available_budget,
            warnings=warnings
        )

    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting budget utilization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebalance", response_model=Dict[str, Any])
async def trigger_rebalancing(
    force: bool = Query(False, description="Force rebalancing even if not needed"),
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    risk_budgeting_service = Depends(get_risk_budgeting_service)
) -> Dict[str, Any]:
    """
    Manually trigger risk budget rebalancing.
    
    Analyzes current allocations and market regime to determine
    optimal risk budget distribution.
    """
    try:
        # Check if user has permission to rebalance
        if "manage_risk" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to trigger rebalancing"
            )
        
        # Trigger rebalancing
        result = await risk_budgeting_service.rebalance_budgets(force=force)

        logger.info(f"[Tenant: {tenant_id}] Triggered risk budget rebalancing (force={force})")

        return {
            "status": "completed" if result.executed else "skipped",
            "rebalanced": result.executed,
            "reason": result.reason,
            "changes": result.changes if result.executed else {},
            "new_allocations": result.new_allocations if result.executed else {},
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error triggering rebalancing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=RebalancingRecommendation)
async def get_rebalancing_recommendations(
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    risk_budgeting_service = Depends(get_risk_budgeting_service)
) -> RebalancingRecommendation:
    """
    Get pending rebalancing recommendations.
    
    Analyzes current portfolio against optimal risk budgets
    and provides actionable recommendations.
    """
    try:
        # Get recommendations
        recommendations = await risk_budgeting_service.get_recommendations()
        
        # Calculate expected improvements
        expected_improvement = {}
        if recommendations.expected_sharpe_improvement:
            expected_improvement["sharpe_ratio"] = recommendations.expected_sharpe_improvement
        if recommendations.expected_volatility_reduction:
            expected_improvement["volatility"] = -recommendations.expected_volatility_reduction
        if recommendations.expected_drawdown_reduction:
            expected_improvement["max_drawdown"] = -recommendations.expected_drawdown_reduction

        logger.info(f"[Tenant: {tenant_id}] Retrieved rebalancing recommendations")

        return RebalancingRecommendation(
            requires_rebalancing=recommendations.requires_rebalancing,
            reason=recommendations.reason,
            target_allocations=recommendations.target_allocation,
            current_allocations=recommendations.current_allocation,
            expected_improvement=expected_improvement,
            priority=recommendations.priority
        )

    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=RiskBudgetHistoryResponse)
async def get_budget_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    risk_budgeting_service = Depends(get_risk_budgeting_service)
) -> RiskBudgetHistoryResponse:
    """
    Get historical risk budget data and performance.
    
    Returns snapshots of risk budgets over time along with
    performance metrics and regime changes.
    """
    try:
        # Get historical snapshots
        start_date = datetime.now() - timedelta(days=days)
        snapshots = await risk_budgeting_service.get_budget_history(start_date)
        
        # Format snapshots
        formatted_snapshots = [
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "total_budget": snapshot.total_budget,
                "utilization": snapshot.utilization_rate,
                "volatility": snapshot.realized_volatility,
                "leverage": snapshot.leverage,
                "regime": snapshot.market_regime
            }
            for snapshot in snapshots
        ]
        
        # Calculate performance metrics
        if snapshots:
            performance_metrics = {
                "avg_utilization": sum(s.utilization_rate for s in snapshots) / len(snapshots),
                "max_utilization": max(s.utilization_rate for s in snapshots),
                "avg_volatility": sum(s.realized_volatility for s in snapshots) / len(snapshots),
                "regime_changes": len(set(s.market_regime for s in snapshots)) - 1
            }
        else:
            performance_metrics = {
                "avg_utilization": 0,
                "max_utilization": 0,
                "avg_volatility": 0,
                "regime_changes": 0
            }
        
        # Identify regime changes
        regime_changes = []
        if len(snapshots) > 1:
            current_regime = snapshots[0].market_regime
            for snapshot in snapshots[1:]:
                if snapshot.market_regime != current_regime:
                    regime_changes.append({
                        "date": snapshot.timestamp.isoformat(),
                        "from_regime": current_regime,
                        "to_regime": snapshot.market_regime
                    })
                    current_regime = snapshot.market_regime

        logger.info(f"[Tenant: {tenant_id}] Retrieved budget history ({days} days)")

        return RiskBudgetHistoryResponse(
            snapshots=formatted_snapshots,
            performance_metrics=performance_metrics,
            regime_changes=regime_changes
        )

    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting budget history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/{regime_type}")
async def get_regime_specific_budget(
    regime_type: str,
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    risk_budgeting_service = Depends(get_risk_budgeting_service)
) -> Dict[str, Any]:
    """
    Get risk budget configuration for a specific market regime.
    
    Returns the risk allocation strategy that would be applied
    in the specified market regime.
    """
    try:
        # Validate regime type
        valid_regimes = ["bull", "bear", "neutral", "crisis", "recovery"]
        if regime_type.lower() not in valid_regimes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regime type. Must be one of: {valid_regimes}"
            )
        
        # Get regime-specific budget
        regime_budget = await risk_budgeting_service.get_regime_budget(
            RegimeType[regime_type.upper()]
        )

        logger.info(f"[Tenant: {tenant_id}] Retrieved regime-specific budget for {regime_type}")

        return {
            "regime": regime_type,
            "volatility_target": regime_budget.volatility_target,
            "leverage_limit": regime_budget.leverage_limit,
            "asset_class_limits": regime_budget.asset_class_limits,
            "strategy_allocations": regime_budget.strategy_allocations,
            "risk_factors": regime_budget.risk_factors,
            "description": regime_budget.description
        }

    except HTTPException:
        raise
    except KeyError:
        logger.warning(f"[Tenant: {tenant_id}] Invalid regime type: {regime_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regime type: {regime_type}"
        )
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting regime budget: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))