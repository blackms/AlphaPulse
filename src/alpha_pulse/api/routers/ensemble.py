"""
Ensemble methods API endpoints.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel, Field
from datetime import datetime

from alpha_pulse.ml.ensemble.ensemble_manager import EnsembleConfig
from ..dependencies import get_current_user


router = APIRouter()


class CreateEnsembleRequest(BaseModel):
    """Request to create a new ensemble."""
    name: str = Field(..., description="Ensemble name")
    ensemble_type: str = Field("weighted_voting", description="Ensemble type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Ensemble parameters")
    description: Optional[str] = Field(None, description="Ensemble description")


class AgentRegistrationRequest(BaseModel):
    """Request to register an agent with ensemble."""
    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Agent type")
    initial_weight: float = Field(1.0, description="Initial weight")


class PredictionRequest(BaseModel):
    """Request for ensemble prediction."""
    signals: List[Dict[str, Any]] = Field(..., description="Agent signals")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


def get_ensemble_service(request: Request):
    """Get ensemble service instance."""
    # Get the service from app state
    if hasattr(request.app.state, 'ensemble_service'):
        return request.app.state.ensemble_service
    else:
        raise HTTPException(
            status_code=503,
            detail="Ensemble service not initialized. Please check service configuration."
        )


@router.post("/create")
async def create_ensemble(
    request: CreateEnsembleRequest,
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """
    Create a new ensemble configuration.
    
    Ensemble types:
    - weighted_voting: Simple weighted majority voting
    - stacking: Meta-learning with secondary model
    - boosting: Sequential learning with error focus
    - dynamic: Adaptive weight adjustment
    """
    try:
        
        config = EnsembleConfig(
            name=request.name,
            ensemble_type=request.ensemble_type,
            parameters=request.parameters,
            description=request.description
        )
        
        ensemble_id = await service.create_ensemble(config)
        
        return {
            "ensemble_id": ensemble_id,
            "name": request.name,
            "type": request.ensemble_type,
            "status": "created",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ensemble_id}/register-agent")
async def register_agent(
    ensemble_id: str,
    request: AgentRegistrationRequest,
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """Register an agent with the ensemble."""
    try:
        
        await service.register_agent(
            ensemble_id=ensemble_id,
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            initial_weight=request.initial_weight
        )
        
        return {
            "ensemble_id": ensemble_id,
            "agent_id": request.agent_id,
            "status": "registered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ensemble_id}/predict")
async def get_ensemble_prediction(
    ensemble_id: str,
    request: PredictionRequest,
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """Get ensemble prediction from agent signals."""
    try:
        
        prediction = await service.get_ensemble_prediction(
            ensemble_id=ensemble_id,
            agent_signals=request.signals,
            metadata=request.metadata
        )
        
        return {
            "ensemble_id": ensemble_id,
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ensemble_id}/performance")
async def get_ensemble_performance(
    ensemble_id: str,
    days: int = Query(30, description="Number of days of history"),
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """Get ensemble performance metrics."""
    try:
        
        performance = await service.get_ensemble_performance(
            ensemble_id=ensemble_id,
            lookback_days=days
        )
        
        return {
            "ensemble_id": ensemble_id,
            "performance": performance,
            "days": days,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ensemble_id}/weights")
async def get_agent_weights(
    ensemble_id: str,
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """Get current agent weights in ensemble."""
    try:
        
        weights = await service.get_agent_weights(ensemble_id)
        
        return {
            "ensemble_id": ensemble_id,
            "weights": weights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ensemble_id}/optimize-weights")
async def optimize_ensemble_weights(
    ensemble_id: str,
    metric: str = Query("sharpe_ratio", description="Optimization metric"),
    lookback_days: int = Query(30, description="Historical data period"),
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """
    Optimize agent weights based on historical performance.
    
    Metrics:
    - sharpe_ratio: Risk-adjusted returns
    - accuracy: Prediction accuracy
    - profit_factor: Win/loss ratio
    - calmar_ratio: Return/drawdown ratio
    """
    try:
        
        new_weights = await service.optimize_weights(
            ensemble_id=ensemble_id,
            metric=metric,
            lookback_days=lookback_days
        )
        
        return {
            "ensemble_id": ensemble_id,
            "optimized_weights": new_weights,
            "metric": metric,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_ensembles(
    active_only: bool = Query(True, description="Only show active ensembles"),
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """List all ensemble configurations."""
    try:
        
        ensembles = await service.list_ensembles(active_only=active_only)
        
        return {
            "ensembles": ensembles,
            "count": len(ensembles),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-rankings")
async def get_agent_rankings(
    metric: str = Query("sharpe_ratio", description="Ranking metric"),
    days: int = Query(30, description="Evaluation period"),
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """Get agent performance rankings across all ensembles."""
    try:
        
        rankings = await service.get_agent_rankings(
            metric=metric,
            lookback_days=days
        )
        
        return {
            "rankings": rankings,
            "metric": metric,
            "days": days,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{ensemble_id}")
async def delete_ensemble(
    ensemble_id: str,
    user=Depends(get_current_user),
    service=Depends(get_ensemble_service)
):
    """Delete an ensemble configuration."""
    try:
        
        await service.delete_ensemble(ensemble_id)
        
        return {
            "ensemble_id": ensemble_id,
            "status": "deleted",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))