"""
Market regime detection API endpoints.

Provides endpoints for accessing current market regime state,
historical regime data, transition probabilities, and regime-based
trading recommendations.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from alpha_pulse.api.dependencies import get_current_user, get_regime_detection_service

router = APIRouter()


class RegimeStateResponse(BaseModel):
    """Response model for current regime state."""
    current_regime: str
    confidence: float
    timestamp: datetime
    features: Dict[str, float]
    transition_probabilities: Dict[str, float]
    stability_score: float
    

class RegimeHistoryResponse(BaseModel):
    """Response model for regime history."""
    regimes: List[Dict[str, Any]]
    transitions: List[Dict[str, Any]]
    total_regimes: int
    average_duration_days: float
    

class RegimeAnalysisResponse(BaseModel):
    """Response model for regime analysis."""
    regime_type: str
    characteristics: Dict[str, Any]
    recommended_strategies: List[str]
    risk_adjustments: Dict[str, float]
    historical_performance: Dict[str, float]


@router.get("/current", response_model=RegimeStateResponse)
async def get_current_regime(
    current_user: Dict = Depends(get_current_user),
    regime_service = Depends(get_regime_detection_service)
) -> RegimeStateResponse:
    """
    Get current market regime state.
    
    Returns the current regime classification with confidence scores
    and transition probabilities to other regimes.
    """
    try:
        # Get current regime info
        regime_info = regime_service.current_regime_info
        
        if not regime_info:
            raise HTTPException(
                status_code=404,
                detail="No regime state available yet"
            )
        
        # Get current state from state manager
        current_state = regime_service.state_manager.get_current_state()
        
        # Map regime index to name
        regime_names = ["Bull", "Bear", "Sideways", "High Volatility", "Crisis"]
        current_regime_name = regime_names[regime_info.current_regime] if regime_info.current_regime < len(regime_names) else "Unknown"
        
        # Calculate transition probabilities
        transition_probs = {}
        if hasattr(regime_info, 'transition_matrix') and regime_info.transition_matrix is not None:
            current_idx = regime_info.current_regime
            for i, name in enumerate(regime_names):
                if i < len(regime_info.transition_matrix[current_idx]):
                    transition_probs[name] = float(regime_info.transition_matrix[current_idx][i])
        
        return RegimeStateResponse(
            current_regime=current_regime_name,
            confidence=float(regime_info.confidence),
            timestamp=regime_info.timestamp,
            features=regime_info.features,
            transition_probabilities=transition_probs,
            stability_score=float(current_state.stability_score) if current_state else 0.8
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=RegimeHistoryResponse)
async def get_regime_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    current_user: Dict = Depends(get_current_user),
    regime_service = Depends(get_regime_detection_service)
) -> RegimeHistoryResponse:
    """
    Get historical regime data.
    
    Returns regime history including transitions and duration statistics.
    """
    try:
        # Get historical states
        start_date = datetime.now() - timedelta(days=days)
        historical_states = regime_service.state_manager.get_historical_states(
            start_date, datetime.now()
        )
        
        if not historical_states:
            return RegimeHistoryResponse(
                regimes=[],
                transitions=[],
                total_regimes=0,
                average_duration_days=0
            )
        
        # Process regime data
        regime_names = ["Bull", "Bear", "Sideways", "High Volatility", "Crisis"]
        regimes = []
        transitions = []
        
        for i, state in enumerate(historical_states):
            regime_data = {
                "regime": regime_names[state.regime] if state.regime < len(regime_names) else "Unknown",
                "start_time": state.timestamp.isoformat(),
                "confidence": float(state.confidence),
                "volatility": float(state.features.get('volatility', 0))
            }
            
            # Calculate duration if not the last regime
            if i < len(historical_states) - 1:
                duration = (historical_states[i+1].timestamp - state.timestamp).total_seconds() / 86400
                regime_data["duration_days"] = duration
                
                # Record transition
                transitions.append({
                    "from_regime": regime_data["regime"],
                    "to_regime": regime_names[historical_states[i+1].regime] if historical_states[i+1].regime < len(regime_names) else "Unknown",
                    "timestamp": historical_states[i+1].timestamp.isoformat(),
                    "confidence": float(historical_states[i+1].confidence)
                })
            
            regimes.append(regime_data)
        
        # Calculate average duration
        durations = [r.get("duration_days", 0) for r in regimes if "duration_days" in r]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return RegimeHistoryResponse(
            regimes=regimes,
            transitions=transitions,
            total_regimes=len(regimes),
            average_duration_days=avg_duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{regime_type}", response_model=RegimeAnalysisResponse)
async def get_regime_analysis(
    regime_type: str,
    current_user: Dict = Depends(get_current_user),
    regime_service = Depends(get_regime_detection_service)
) -> RegimeAnalysisResponse:
    """
    Get detailed analysis for a specific regime type.
    
    Returns characteristics, recommended strategies, and historical
    performance for the specified regime.
    """
    try:
        # Validate regime type
        valid_regimes = ["bull", "bear", "sideways", "high_volatility", "crisis"]
        if regime_type.lower() not in valid_regimes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regime type. Must be one of: {valid_regimes}"
            )
        
        # Define regime characteristics and strategies
        regime_data = {
            "bull": {
                "characteristics": {
                    "trend": "Strong upward",
                    "volatility": "Low to moderate",
                    "sentiment": "Positive",
                    "volume": "Increasing"
                },
                "recommended_strategies": [
                    "Trend following",
                    "Momentum trading",
                    "Buy and hold",
                    "Growth investing"
                ],
                "risk_adjustments": {
                    "leverage": 1.5,
                    "position_size": 1.2,
                    "stop_loss": 0.08
                }
            },
            "bear": {
                "characteristics": {
                    "trend": "Strong downward",
                    "volatility": "High",
                    "sentiment": "Negative",
                    "volume": "High on declines"
                },
                "recommended_strategies": [
                    "Short selling",
                    "Defensive positioning",
                    "Value investing",
                    "Hedging strategies"
                ],
                "risk_adjustments": {
                    "leverage": 0.5,
                    "position_size": 0.6,
                    "stop_loss": 0.05
                }
            },
            "sideways": {
                "characteristics": {
                    "trend": "Range-bound",
                    "volatility": "Low",
                    "sentiment": "Neutral",
                    "volume": "Average"
                },
                "recommended_strategies": [
                    "Mean reversion",
                    "Range trading",
                    "Options strategies",
                    "Arbitrage"
                ],
                "risk_adjustments": {
                    "leverage": 1.0,
                    "position_size": 0.8,
                    "stop_loss": 0.06
                }
            },
            "high_volatility": {
                "characteristics": {
                    "trend": "Uncertain",
                    "volatility": "Very high",
                    "sentiment": "Mixed/Fearful",
                    "volume": "Erratic"
                },
                "recommended_strategies": [
                    "Volatility trading",
                    "Options strategies",
                    "Risk parity",
                    "Defensive positioning"
                ],
                "risk_adjustments": {
                    "leverage": 0.3,
                    "position_size": 0.4,
                    "stop_loss": 0.03
                }
            },
            "crisis": {
                "characteristics": {
                    "trend": "Sharp decline",
                    "volatility": "Extreme",
                    "sentiment": "Panic",
                    "volume": "Extreme spikes"
                },
                "recommended_strategies": [
                    "Cash preservation",
                    "Tail risk hedging",
                    "Safe haven assets",
                    "Systematic de-risking"
                ],
                "risk_adjustments": {
                    "leverage": 0.0,
                    "position_size": 0.2,
                    "stop_loss": 0.02
                }
            }
        }
        
        regime_info = regime_data[regime_type.lower()]
        
        # Get historical performance if available
        historical_performance = {
            "average_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }
        
        if hasattr(regime_service, 'performance_tracker'):
            perf_data = regime_service.performance_tracker.get_regime_performance(
                regime_type.lower()
            )
            if perf_data:
                historical_performance.update(perf_data)
        
        return RegimeAnalysisResponse(
            regime_type=regime_type.title(),
            characteristics=regime_info["characteristics"],
            recommended_strategies=regime_info["recommended_strategies"],
            risk_adjustments=regime_info["risk_adjustments"],
            historical_performance=historical_performance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_regime_alerts(
    hours: int = Query(24, ge=1, le=168, description="Hours of alert history"),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get recent regime-related alerts.
    
    Returns alerts for regime transitions and confidence changes.
    """
    try:
        # This would integrate with the alert system
        # For now, return a sample structure
        return {
            "regime_transition_alerts": [],
            "confidence_alerts": [],
            "stability_alerts": [],
            "total_alerts": 0,
            "time_range_hours": hours
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))