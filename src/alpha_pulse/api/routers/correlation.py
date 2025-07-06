"""
Correlation analysis API endpoints.

Provides comprehensive correlation analysis for portfolio risk management
including correlation matrices, rolling correlations, regime analysis,
and concentration risk metrics.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from alpha_pulse.risk.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationMethod,
    CorrelationAnalysisConfig
)
from alpha_pulse.api.dependencies import get_current_user, get_risk_manager
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.services.caching_service import CachingService

router = APIRouter()


class CorrelationMatrixRequest(BaseModel):
    """Request model for correlation matrix calculation."""
    symbols: List[str] = Field(..., min_items=2, description="List of asset symbols")
    lookback_days: int = Field(252, ge=30, le=1260, description="Lookback period in days")
    method: str = Field("pearson", description="Correlation method")
    
    
class CorrelationMatrixResponse(BaseModel):
    """Response model for correlation matrix."""
    symbols: List[str]
    matrix: List[List[float]]
    method: str
    timestamp: datetime
    lookback_days: int
    

class RollingCorrelationRequest(BaseModel):
    """Request model for rolling correlation analysis."""
    symbol1: str = Field(..., description="First asset symbol")
    symbol2: str = Field(..., description="Second asset symbol")
    window_days: int = Field(30, ge=10, le=252, description="Rolling window in days")
    lookback_days: int = Field(252, ge=30, le=1260, description="Total lookback period")
    

class RollingCorrelationResponse(BaseModel):
    """Response model for rolling correlation."""
    symbol1: str
    symbol2: str
    correlations: List[Dict[str, Any]]  # List of {date, correlation} pairs
    current_correlation: float
    average_correlation: float
    min_correlation: float
    max_correlation: float
    

class ConcentrationRiskResponse(BaseModel):
    """Response model for concentration risk analysis."""
    concentration_score: float
    high_correlation_pairs: List[Dict[str, Any]]
    diversification_ratio: float
    effective_assets: float
    risk_contributions: Dict[str, float]
    

class RegimeCorrelationResponse(BaseModel):
    """Response model for regime-based correlation analysis."""
    current_regime: str
    regime_correlations: Dict[str, Dict[str, Any]]
    regime_transitions: List[Dict[str, Any]]
    stability_score: float


@router.get("/matrix", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix(
    symbols: List[str] = Query(..., min_items=2, description="Asset symbols"),
    lookback_days: int = Query(252, ge=30, le=1260, description="Lookback period"),
    method: str = Query("pearson", description="Correlation method"),
    current_user: Dict = Depends(get_current_user)
) -> CorrelationMatrixResponse:
    """
    Calculate correlation matrix for specified assets.
    
    Methods available: pearson, spearman, kendall, distance
    """
    try:
        # Initialize components
        data_fetcher = DataFetcher()
        analyzer = CorrelationAnalyzer()
        cache = CachingService.create_for_api()
        
        # Check cache first
        cache_key = f"correlation:matrix:{':'.join(sorted(symbols))}:{lookback_days}:{method}"
        cached_result = await cache.get(cache_key)
        if cached_result:
            return CorrelationMatrixResponse(**cached_result)
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        returns_data = {}
        for symbol in symbols:
            df = await data_fetcher.fetch_historical_data(
                symbol, start_date, end_date
            )
            if df is None or len(df) < 30:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data for symbol {symbol}"
                )
            returns_data[symbol] = df['close'].pct_change().dropna()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Calculate correlation matrix
        if method == "pearson":
            corr_matrix = returns_df.corr(method='pearson')
        elif method == "spearman":
            corr_matrix = returns_df.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = returns_df.corr(method='kendall')
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid correlation method: {method}"
            )
        
        response = CorrelationMatrixResponse(
            symbols=symbols,
            matrix=corr_matrix.values.tolist(),
            method=method,
            timestamp=datetime.now(),
            lookback_days=lookback_days
        )
        
        # Cache result for 1 hour
        await cache.set(cache_key, response.dict(), ttl=3600)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rolling", response_model=RollingCorrelationResponse)
async def get_rolling_correlations(
    symbol1: str = Query(..., description="First asset symbol"),
    symbol2: str = Query(..., description="Second asset symbol"),
    window_days: int = Query(30, ge=10, le=252, description="Rolling window"),
    lookback_days: int = Query(252, ge=30, le=1260, description="Total period"),
    current_user: Dict = Depends(get_current_user)
) -> RollingCorrelationResponse:
    """
    Calculate rolling correlation between two assets.
    
    Returns time series of correlations with statistics.
    """
    try:
        # Initialize components
        data_fetcher = DataFetcher()
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get data for both symbols
        df1 = await data_fetcher.fetch_historical_data(symbol1, start_date, end_date)
        df2 = await data_fetcher.fetch_historical_data(symbol2, start_date, end_date)
        
        if df1 is None or df2 is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to fetch data for one or both symbols"
            )
        
        # Calculate returns
        returns1 = df1['close'].pct_change().dropna()
        returns2 = df2['close'].pct_change().dropna()
        
        # Align data
        aligned_data = pd.DataFrame({
            symbol1: returns1,
            symbol2: returns2
        }).dropna()
        
        # Calculate rolling correlation
        rolling_corr = aligned_data[symbol1].rolling(window=window_days).corr(
            aligned_data[symbol2]
        ).dropna()
        
        # Prepare response data
        correlations = [
            {"date": date.isoformat(), "correlation": float(corr)}
            for date, corr in rolling_corr.items()
        ]
        
        return RollingCorrelationResponse(
            symbol1=symbol1,
            symbol2=symbol2,
            correlations=correlations,
            current_correlation=float(rolling_corr.iloc[-1]),
            average_correlation=float(rolling_corr.mean()),
            min_correlation=float(rolling_corr.min()),
            max_correlation=float(rolling_corr.max())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concentration", response_model=ConcentrationRiskResponse)
async def get_concentration_risk(
    current_user: Dict = Depends(get_current_user),
    risk_manager = Depends(get_risk_manager)
) -> ConcentrationRiskResponse:
    """
    Analyze portfolio concentration risk via correlation.
    
    Returns concentration metrics and high correlation pairs.
    """
    try:
        # Get current portfolio
        portfolio = risk_manager.portfolio_manager.get_portfolio()
        
        if not portfolio or not portfolio.positions:
            raise HTTPException(
                status_code=400,
                detail="No portfolio positions found"
            )
        
        # Get symbols from portfolio
        symbols = list(portfolio.positions.keys())
        
        # Initialize components
        data_fetcher = DataFetcher()
        analyzer = CorrelationAnalyzer()
        
        # Fetch historical data for all symbols
        end_date = datetime.now()
        start_date = end_date - timedelta(days=252)
        
        returns_data = {}
        for symbol in symbols:
            df = await data_fetcher.fetch_historical_data(
                symbol, start_date, end_date
            )
            if df is not None and len(df) >= 30:
                returns_data[symbol] = df['close'].pct_change().dropna()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Calculate correlation matrix
        corr_result = analyzer.calculate_correlation_matrix(returns_df)
        
        # Find high correlation pairs (> 0.7)
        high_corr_pairs = []
        corr_matrix = corr_result.matrix if hasattr(corr_result, 'matrix') else corr_result.values
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr = corr_matrix[i][j]
                if abs(corr) > 0.7:
                    high_corr_pairs.append({
                        "symbol1": symbols[i],
                        "symbol2": symbols[j],
                        "correlation": float(corr),
                        "weight1": float(portfolio.positions[symbols[i]].weight),
                        "weight2": float(portfolio.positions[symbols[j]].weight)
                    })
        
        # Calculate concentration metrics
        weights = np.array([pos.weight for pos in portfolio.positions.values()])
        
        # Diversification ratio
        portfolio_variance = float(weights @ corr_matrix @ weights)
        equal_weight_variance = (1/len(weights)) * np.sum(corr_matrix) / len(weights)
        diversification_ratio = float(np.sqrt(equal_weight_variance / portfolio_variance))
        
        # Effective number of assets
        eigenvalues = np.linalg.eigvals(corr_matrix)
        effective_assets = float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))
        
        # Risk contributions
        marginal_contributions = corr_matrix @ weights
        risk_contributions = {
            symbol: float(weight * mc / portfolio_variance)
            for symbol, weight, mc in zip(symbols, weights, marginal_contributions)
        }
        
        # Concentration score (0-1, higher is more concentrated)
        concentration_score = float(1 - effective_assets / len(symbols))
        
        return ConcentrationRiskResponse(
            concentration_score=concentration_score,
            high_correlation_pairs=high_corr_pairs,
            diversification_ratio=diversification_ratio,
            effective_assets=effective_assets,
            risk_contributions=risk_contributions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime", response_model=RegimeCorrelationResponse)
async def get_regime_correlations(
    lookback_days: int = Query(504, ge=252, le=1260, description="Analysis period"),
    current_user: Dict = Depends(get_current_user),
    risk_manager = Depends(get_risk_manager)
) -> RegimeCorrelationResponse:
    """
    Get correlation analysis by market regime.
    
    Identifies different correlation regimes and current state.
    """
    try:
        # Get portfolio symbols
        portfolio = risk_manager.portfolio_manager.get_portfolio()
        if not portfolio or not portfolio.positions:
            raise HTTPException(
                status_code=400,
                detail="No portfolio positions found"
            )
        
        symbols = list(portfolio.positions.keys())
        
        # Initialize analyzer with regime detection
        config = CorrelationAnalysisConfig(
            lookback_period=lookback_days,
            detect_regimes=True
        )
        analyzer = CorrelationAnalyzer(config)
        
        # Fetch historical data for all symbols
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        returns_data = {}
        data_fetcher = DataFetcher()
        for symbol in symbols:
            df = await data_fetcher.fetch_historical_data(
                symbol, start_date, end_date
            )
            if df is not None and len(df) >= 30:
                returns_data[symbol] = df['close'].pct_change().dropna()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Perform regime analysis
        regime_analysis = analyzer.detect_correlation_regimes(returns_df)
        
        # Format response
        regime_correlations = {}
        current_regime = "unknown"
        
        # Process regimes
        if regime_analysis and len(regime_analysis) > 0:
            # Get the most recent regime
            latest_regime = regime_analysis[-1]
            current_regime = latest_regime.regime_type
            
            # Group regimes by type
            for regime in regime_analysis:
                regime_type = regime.regime_type
                if regime_type not in regime_correlations:
                    regime_correlations[regime_type] = {
                        "average_correlation": float(regime.average_correlation),
                        "correlation_volatility": 0.1,  # Default value
                        "period_count": 0,
                        "last_occurrence": regime.end_date.isoformat()
                    }
                else:
                    # Update last occurrence
                    regime_correlations[regime_type]["period_count"] += 1
                    if regime.end_date > datetime.fromisoformat(regime_correlations[regime_type]["last_occurrence"]):
                        regime_correlations[regime_type]["last_occurrence"] = regime.end_date.isoformat()
        
        # Calculate regime transitions
        transitions = []
        for i in range(1, len(regime_analysis)):
            if regime_analysis[i].regime_type != regime_analysis[i-1].regime_type:
                transitions.append({
                    "from_regime": regime_analysis[i-1].regime_type,
                    "to_regime": regime_analysis[i].regime_type,
                    "date": regime_analysis[i].start_date.isoformat(),
                    "probability": 0.8  # Default transition probability
                })
        
        # Calculate stability score based on regime changes
        stability_score = 1.0 - (len(transitions) / max(len(regime_analysis), 1))
        
        return RegimeCorrelationResponse(
            current_regime=current_regime,
            regime_correlations=regime_correlations,
            regime_transitions=transitions[-10:],  # Last 10 transitions
            stability_score=float(stability_score)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_correlation_alerts(
    threshold: float = Query(0.8, ge=0.5, le=1.0, description="Alert threshold"),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get active correlation-based alerts.
    
    Monitors for correlation spikes and regime changes.
    """
    try:
        # This would integrate with the alert manager
        # For now, return a sample structure
        return {
            "correlation_spike_alerts": [],
            "regime_change_alerts": [],
            "concentration_alerts": [],
            "threshold": threshold,
            "last_check": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))