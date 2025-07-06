"""
Online learning API endpoints.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel, Field
from datetime import datetime

from alpha_pulse.ml.online.online_model import (
    LearningSessionRequest,
    PredictionRequest,
    StreamingBatch
)
from ..dependencies import get_current_user


router = APIRouter()


class ModelUpdateRequest(BaseModel):
    """Request to update model with new data."""
    session_id: str = Field(..., description="Learning session ID")
    data_points: List[Dict[str, Any]] = Field(..., description="New data points")
    labels: Optional[List[Any]] = Field(None, description="Labels for supervised learning")


class SessionStatusResponse(BaseModel):
    """Response for session status query."""
    session_id: str
    status: str
    strategy: str
    start_time: datetime
    data_points_processed: int
    current_performance: Dict[str, float]
    drift_detected: bool
    last_update: datetime


def get_online_learning_service(request: Request):
    """Get online learning service instance."""
    if hasattr(request.app.state, 'online_learning_service'):
        return request.app.state.online_learning_service
    else:
        raise HTTPException(
            status_code=503,
            detail="Online learning service not initialized. Please check service configuration."
        )


@router.post("/sessions/create")
async def create_learning_session(
    request: LearningSessionRequest,
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """
    Create a new online learning session.
    
    Strategies:
    - incremental_sgd: Stochastic Gradient Descent
    - naive_bayes: Incremental Naive Bayes
    - passive_aggressive: Online passive-aggressive algorithms
    - hoeffding_tree: Incremental decision tree
    - adaptive_forest: Adaptive Random Forest
    - gradient_boosting: Online gradient boosting
    - ensemble: Ensemble of multiple learners
    """
    try:
        response = await service.start_session(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/update")
async def update_model(
    session_id: str,
    request: ModelUpdateRequest,
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Update model with new data points."""
    try:
        # Create streaming batch
        batch = StreamingBatch(
            batch_id=None,  # Will be auto-generated
            session_id=session_id,
            data_points=request.data_points,
            labels=request.labels,
            timestamp=datetime.utcnow()
        )
        
        metrics = await service.update_batch(session_id, batch)
        
        return {
            "session_id": session_id,
            "data_points_processed": len(request.data_points),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/predict")
async def get_prediction(
    session_id: str,
    request: PredictionRequest,
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Get prediction from online learning model."""
    try:
        response = await service.predict(session_id, request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: str,
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
) -> SessionStatusResponse:
    """Get current status of learning session."""
    try:
        status = await service.get_session_status(session_id)
        
        return SessionStatusResponse(
            session_id=session_id,
            status=status.get("status", "unknown"),
            strategy=status.get("strategy", ""),
            start_time=status.get("start_time"),
            data_points_processed=status.get("data_points_processed", 0),
            current_performance=status.get("performance", {}),
            drift_detected=status.get("drift_detected", False),
            last_update=status.get("last_update")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/metrics")
async def get_learning_metrics(
    session_id: str,
    window_size: int = Query(100, description="Number of recent points to analyze"),
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Get detailed learning metrics for session."""
    try:
        metrics = await service.get_learning_metrics(session_id, window_size)
        return {
            "session_id": session_id,
            "metrics": metrics,
            "window_size": window_size,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/checkpoint")
async def create_checkpoint(
    session_id: str,
    description: Optional[str] = None,
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Create a checkpoint of current model state."""
    try:
        checkpoint_id = await service.create_checkpoint(session_id, description)
        return {
            "session_id": session_id,
            "checkpoint_id": checkpoint_id,
            "description": description,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/rollback")
async def rollback_to_checkpoint(
    session_id: str,
    checkpoint_id: str,
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Rollback model to a previous checkpoint."""
    try:
        await service.rollback_to_checkpoint(session_id, checkpoint_id)
        return {
            "session_id": session_id,
            "checkpoint_id": checkpoint_id,
            "status": "rolled_back",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/drift")
async def get_drift_status(
    session_id: str,
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Get drift detection status and history."""
    try:
        drift_info = await service.get_drift_status(session_id)
        return {
            "session_id": session_id,
            "drift_detected": drift_info.get("drift_detected", False),
            "drift_score": drift_info.get("drift_score", 0.0),
            "drift_events": drift_info.get("drift_events", []),
            "last_check": drift_info.get("last_check"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(
    active_only: bool = Query(True, description="Only show active sessions"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """List all learning sessions."""
    try:
        sessions = await service.list_sessions(
            active_only=active_only,
            agent_id=agent_id
        )
        
        return {
            "sessions": sessions,
            "count": len(sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def stop_session(
    session_id: str,
    save_checkpoint: bool = Query(True, description="Save final checkpoint"),
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Stop and cleanup learning session."""
    try:
        await service.stop_session(session_id, save_checkpoint)
        return {
            "session_id": session_id,
            "status": "stopped",
            "checkpoint_saved": save_checkpoint,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-predict")
async def batch_predict(
    session_ids: List[str],
    features: Dict[str, Any],
    user=Depends(get_current_user),
    service=Depends(get_online_learning_service)
):
    """Get predictions from multiple models (ensemble)."""
    try:
        predictions = await service.batch_predict(session_ids, features)
        return {
            "session_ids": session_ids,
            "predictions": predictions,
            "ensemble_prediction": predictions.get("ensemble"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))