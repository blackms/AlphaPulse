"""
Explainable AI API endpoints for trading decision explanations.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from loguru import logger
from datetime import datetime

from ..dependencies import get_agent_manager
from ...agents.manager import AgentManager
from ...services.explainability_service import ExplainabilityService
from ...models.explanation_result import ExplanationResult, GlobalExplanation
from ...compliance.explanation_compliance import ExplanationComplianceManager, ComplianceRequirement

router = APIRouter(prefix="/explainability", tags=["explainability"])

# Initialize explainability service and compliance manager
explainability_service = ExplainabilityService()
compliance_manager = ExplanationComplianceManager()


class ExplanationRequest(BaseModel):
    """Request model for explanations."""
    agent_type: str
    symbol: str
    method: Optional[str] = "shap"  # shap, lime, feature_importance
    explanation_type: Optional[str] = "local"  # local, global
    include_visualization: Optional[bool] = True


class TradeDecisionExplanationRequest(BaseModel):
    """Request model for trade decision explanations."""
    trade_id: Optional[str] = None
    symbol: str
    signal_data: Dict[str, Any]
    method: Optional[str] = "shap"
    include_counterfactuals: Optional[bool] = False


@router.post("/explain-decision", response_model=Dict[str, Any])
async def explain_trade_decision(
    request: TradeDecisionExplanationRequest,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Generate explanation for a specific trading decision.
    
    Args:
        request: Explanation request parameters
        
    Returns:
        Comprehensive explanation of the trading decision
    """
    try:
        logger.info(f"Generating explanation for {request.symbol} using {request.method}")
        
        # Get the trading signal data
        signal_data = request.signal_data
        symbol = request.symbol
        
        # Generate explanation using the explainability service
        explanation = await explainability_service.explain_prediction(
            model_type="trading_signal",
            prediction_data=signal_data,
            method=request.method,
            symbol=symbol,
            include_visualization=True
        )
        
        # Enhanced explanation with trading context
        enhanced_explanation = {
            "explanation_id": explanation.explanation_id,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "signal_details": {
                "direction": signal_data.get("direction"),
                "confidence": signal_data.get("confidence"),
                "target_price": signal_data.get("target_price"),
                "stop_loss": signal_data.get("stop_loss")
            },
            "feature_importance": explanation.feature_importance,
            "explanation_text": explanation.explanation_text,
            "confidence_score": explanation.confidence,
            "visualization_data": explanation.visualization_data,
            "key_factors": _extract_key_factors(explanation),
            "risk_factors": _extract_risk_factors(explanation),
            "compliance_notes": explanation.compliance_notes
        }
        
        # Add counterfactual analysis if requested
        if request.include_counterfactuals:
            counterfactuals = await explainability_service.generate_counterfactuals(
                model_type="trading_signal",
                original_data=signal_data,
                symbol=symbol
            )
            enhanced_explanation["counterfactuals"] = [
                {
                    "scenario": cf.scenario_description,
                    "changed_features": cf.changed_features,
                    "predicted_outcome": cf.predicted_outcome,
                    "likelihood": cf.likelihood
                }
                for cf in counterfactuals
            ]
        
        logger.info(f"Generated explanation for {symbol} with {len(explanation.feature_importance)} features")
        return enhanced_explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation for {request.symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@router.get("/agent-explanation/{agent_type}", response_model=Dict[str, Any])
async def get_agent_explanation(
    agent_type: str,
    symbol: str = Query(..., description="Trading symbol to explain"),
    method: str = Query("shap", description="Explanation method"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Get explanation for how a specific agent makes decisions for a symbol.
    
    Args:
        agent_type: Type of trading agent (technical, fundamental, sentiment, etc.)
        symbol: Trading symbol
        method: Explanation method to use
        
    Returns:
        Agent-specific decision explanation
    """
    try:
        if agent_type not in agent_manager.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent type '{agent_type}' not found"
            )
        
        # Get agent-specific explanation
        agent = agent_manager.agents[agent_type]
        
        # Generate global explanation for the agent's decision process
        global_explanation = await explainability_service.explain_model_globally(
            model_type=f"{agent_type}_agent",
            method=method,
            symbol=symbol
        )
        
        # Enhanced agent explanation
        agent_explanation = {
            "agent_type": agent_type,
            "symbol": symbol,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "decision_factors": global_explanation.top_features,
            "feature_interactions": global_explanation.feature_interactions,
            "typical_patterns": global_explanation.typical_patterns,
            "visualization_data": global_explanation.visualization_data,
            "agent_weights": agent_manager.agent_weights.get(agent_type, 0),
            "performance_influence": _get_agent_performance_influence(agent_type, agent_manager),
            "explanation_summary": global_explanation.summary
        }
        
        return agent_explanation
        
    except Exception as e:
        logger.error(f"Error getting agent explanation for {agent_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent explanation: {str(e)}"
        )


@router.get("/feature-importance/{symbol}", response_model=Dict[str, Any])
async def get_feature_importance(
    symbol: str,
    method: str = Query("permutation", description="Importance method"),
    top_n: int = Query(20, description="Number of top features to return")
) -> Dict[str, Any]:
    """
    Get feature importance analysis for a specific symbol.
    
    Args:
        symbol: Trading symbol
        method: Importance calculation method
        top_n: Number of top features to return
        
    Returns:
        Feature importance rankings and analysis
    """
    try:
        # Calculate feature importance
        importance_result = await explainability_service.calculate_feature_importance(
            symbol=symbol,
            method=method,
            top_n=top_n
        )
        
        return {
            "symbol": symbol,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "feature_importance": importance_result.feature_importance[:top_n],
            "importance_scores": importance_result.importance_scores[:top_n],
            "feature_descriptions": _get_feature_descriptions(importance_result.feature_importance),
            "stability_score": importance_result.stability_score,
            "visualization_data": importance_result.visualization_data
        }
        
    except Exception as e:
        logger.error(f"Error calculating feature importance for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate feature importance: {str(e)}"
        )


@router.get("/explanation-history/{symbol}", response_model=List[Dict[str, Any]])
async def get_explanation_history(
    symbol: str,
    limit: int = Query(50, description="Maximum number of explanations to return"),
    method: Optional[str] = Query(None, description="Filter by explanation method")
) -> List[Dict[str, Any]]:
    """
    Get historical explanations for a symbol.
    
    Args:
        symbol: Trading symbol
        limit: Maximum number of explanations
        method: Optional method filter
        
    Returns:
        List of historical explanations
    """
    try:
        explanations = await explainability_service.get_explanation_history(
            symbol=symbol,
            limit=limit,
            method=method
        )
        
        return [
            {
                "explanation_id": exp.explanation_id,
                "timestamp": exp.timestamp.isoformat() if exp.timestamp else None,
                "method": exp.method,
                "confidence": exp.confidence,
                "key_features": exp.feature_importance[:5],  # Top 5 features
                "explanation_summary": exp.explanation_text[:200] + "..." if len(exp.explanation_text) > 200 else exp.explanation_text
            }
            for exp in explanations
        ]
        
    except Exception as e:
        logger.error(f"Error getting explanation history for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get explanation history: {str(e)}"
        )


@router.get("/methods", response_model=Dict[str, Any])
async def get_available_methods() -> Dict[str, Any]:
    """
    Get available explanation methods and their capabilities.
    
    Returns:
        Available explanation methods and descriptions
    """
    return {
        "methods": {
            "shap": {
                "name": "SHAP (SHapley Additive exPlanations)",
                "description": "Game theory based explanations with feature attributions",
                "supports_local": True,
                "supports_global": True,
                "best_for": ["tree_models", "neural_networks", "general_models"]
            },
            "lime": {
                "name": "LIME (Local Interpretable Model-agnostic Explanations)",
                "description": "Local explanations using interpretable surrogate models",
                "supports_local": True,
                "supports_global": False,
                "best_for": ["local_explanations", "tabular_data", "time_series"]
            },
            "feature_importance": {
                "name": "Feature Importance",
                "description": "Model-agnostic feature importance using permutation or dropout",
                "supports_local": False,
                "supports_global": True,
                "best_for": ["global_understanding", "feature_selection"]
            },
            "decision_trees": {
                "name": "Decision Tree Surrogates",
                "description": "Interpretable decision tree approximations of complex models",
                "supports_local": True,
                "supports_global": True,
                "best_for": ["rule_extraction", "simple_explanations"]
            }
        },
        "capabilities": {
            "real_time_explanations": True,
            "historical_analysis": True,
            "counterfactual_generation": True,
            "compliance_reporting": True,
            "visualization_export": True,
            "multi_method_consensus": True
        }
    }


@router.get("/compliance-report", response_model=Dict[str, Any])
async def generate_compliance_report(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    symbol: Optional[str] = Query(None, description="Optional symbol filter"),
    regulatory_requirement: Optional[str] = Query(None, description="Regulatory requirement (mifid_ii, sox, gdpr)")
) -> Dict[str, Any]:
    """
    Generate comprehensive compliance report with explanation audit trail.
    
    Args:
        start_date: Start date for report period
        end_date: End date for report period
        symbol: Optional symbol filter
        regulatory_requirement: Optional regulatory requirement filter
        
    Returns:
        Comprehensive compliance report with explanation metrics
    """
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Parse regulatory requirement
        req = None
        if regulatory_requirement:
            try:
                req = ComplianceRequirement(regulatory_requirement)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid regulatory requirement: {regulatory_requirement}"
                )
        
        # Generate comprehensive compliance report
        report = await compliance_manager.generate_compliance_report(
            start_date=start_dt,
            end_date=end_dt,
            regulatory_requirement=req,
            symbol=symbol
        )
        
        return report
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate compliance report: {str(e)}"
        )


@router.get("/compliance-export", response_model=Dict[str, Any])
async def export_compliance_data(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    format_type: str = Query("json", description="Export format (json, csv, xml)"),
    regulatory_requirement: Optional[str] = Query(None, description="Regulatory requirement filter")
) -> Dict[str, Any]:
    """
    Export compliance data for regulatory submission.
    
    Args:
        start_date: Start date for export
        end_date: End date for export
        format_type: Export format
        regulatory_requirement: Optional regulatory requirement filter
        
    Returns:
        Exported compliance data
    """
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Parse regulatory requirement
        req = None
        if regulatory_requirement:
            try:
                req = ComplianceRequirement(regulatory_requirement)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid regulatory requirement: {regulatory_requirement}"
                )
        
        # Export compliance data
        export_data = await compliance_manager.export_compliance_data(
            start_date=start_dt,
            end_date=end_dt,
            format_type=format_type,
            regulatory_requirement=req
        )
        
        return export_data
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error exporting compliance data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export compliance data: {str(e)}"
        )


# Helper functions
def _extract_key_factors(explanation: ExplanationResult) -> List[Dict[str, Any]]:
    """Extract key factors from explanation."""
    return [
        {
            "feature": feature,
            "importance": importance,
            "direction": "positive" if importance > 0 else "negative",
            "description": _get_feature_description(feature)
        }
        for feature, importance in explanation.feature_importance[:5]
    ]


def _extract_risk_factors(explanation: ExplanationResult) -> List[str]:
    """Extract risk factors from explanation."""
    risk_factors = []
    
    # Look for negative importance features (risk factors)
    for feature, importance in explanation.feature_importance:
        if importance < -0.1:  # Significant negative impact
            risk_factors.append(f"High {feature} indicates increased risk")
    
    return risk_factors[:3]  # Top 3 risk factors


def _get_agent_performance_influence(agent_type: str, agent_manager: AgentManager) -> Dict[str, Any]:
    """Get agent performance influence metrics."""
    performance = agent_manager.performance_metrics.get(agent_type)
    if performance:
        return {
            "signal_accuracy": performance.signal_accuracy,
            "profit_factor": performance.profit_factor,
            "sharpe_ratio": performance.sharpe_ratio,
            "weight_influence": agent_manager.agent_weights.get(agent_type, 0)
        }
    return {}


def _get_feature_descriptions(features: List[str]) -> Dict[str, str]:
    """Get human-readable descriptions for features."""
    descriptions = {
        "rsi": "Relative Strength Index - momentum oscillator",
        "macd": "Moving Average Convergence Divergence - trend indicator",
        "bollinger_bands": "Bollinger Bands - volatility indicator",
        "volume": "Trading volume - market activity indicator",
        "price_change": "Price change - directional movement",
        "volatility": "Price volatility - risk indicator"
    }
    
    return {feature: descriptions.get(feature, f"Technical indicator: {feature}") 
            for feature in features}


def _get_feature_description(feature: str) -> str:
    """Get description for a single feature."""
    descriptions = _get_feature_descriptions([feature])
    return descriptions.get(feature, f"Technical indicator: {feature}")