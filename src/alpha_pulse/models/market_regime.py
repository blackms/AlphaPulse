"""
Data models for market regime detection and classification.

Defines structures for market regimes, indicators, and transitions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class RegimeIndicatorType(Enum):
    """Types of regime indicators."""
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class RegimeIndicator:
    """Individual regime indicator with value and signal."""
    name: str
    indicator_type: RegimeIndicatorType
    value: float
    normalized_value: float  # 0-1 scale
    signal: float  # -1 to 1 (bearish to bullish)
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_weighted_signal(self) -> float:
        """Get weighted signal contribution."""
        return self.signal * self.weight


@dataclass
class MarketRegime:
    """Represents a market regime with characteristics."""
    regime_type: RegimeType
    start_date: datetime
    end_date: Optional[datetime] = None
    confidence: float = 0.0
    probability: float = 1.0
    
    # Regime characteristics
    volatility_level: str = "normal"  # low, normal, high, extreme
    momentum_direction: str = "neutral"  # negative, neutral, positive
    liquidity_condition: str = "normal"  # poor, normal, abundant
    sentiment_score: float = 0.0  # -1 to 1
    
    # Risk parameters
    suggested_leverage: float = 1.0
    max_position_size: float = 0.10
    stop_loss_multiplier: float = 1.0
    
    # Strategy preferences
    preferred_strategies: List[str] = field(default_factory=list)
    avoided_strategies: List[str] = field(default_factory=list)
    
    # Supporting data
    indicators: Dict[str, RegimeIndicator] = field(default_factory=dict)
    transition_probabilities: Dict[RegimeType, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_days(self) -> Optional[int]:
        """Calculate regime duration in days."""
        if self.end_date:
            return (self.end_date - self.start_date).days
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if regime is currently active."""
        return self.end_date is None
    
    def get_risk_appetite(self) -> str:
        """Determine risk appetite for this regime."""
        if self.regime_type == RegimeType.BULL:
            return "aggressive"
        elif self.regime_type == RegimeType.BEAR:
            return "defensive"
        elif self.regime_type == RegimeType.CRISIS:
            return "minimal"
        elif self.regime_type == RegimeType.RECOVERY:
            return "moderate"
        else:
            return "neutral"


@dataclass
class RegimeTransition:
    """Represents a transition between market regimes."""
    from_regime: RegimeType
    to_regime: RegimeType
    transition_date: datetime
    transition_probability: float
    actual_occurred: bool = False
    
    # Transition triggers
    trigger_indicators: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # Impact metrics
    expected_volatility_change: float = 0.0
    expected_correlation_change: float = 0.0
    recommended_actions: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeDetectionResult:
    """Result of regime detection analysis."""
    current_regime: MarketRegime
    regime_probabilities: Dict[RegimeType, float]
    transition_risk: float  # 0-1 probability of regime change
    
    # Historical context
    previous_regime: Optional[MarketRegime] = None
    regime_history: List[MarketRegime] = field(default_factory=list)
    
    # Indicators summary
    indicator_summary: Dict[RegimeIndicatorType, float] = field(default_factory=dict)
    composite_score: float = 0.0  # Overall market health score
    
    # Predictions
    next_likely_regime: Optional[RegimeType] = None
    regime_change_probability: Dict[str, float] = field(default_factory=dict)
    
    # Risk recommendations
    recommended_risk_level: float = 1.0
    position_sizing_multiplier: float = 1.0
    hedging_recommendations: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_regime_confidence(self) -> float:
        """Get confidence in current regime classification."""
        if self.current_regime.regime_type in self.regime_probabilities:
            return self.regime_probabilities[self.current_regime.regime_type]
        return self.current_regime.confidence
    
    def is_transition_likely(self, threshold: float = 0.3) -> bool:
        """Check if regime transition is likely."""
        return self.transition_risk >= threshold


@dataclass
class RegimeParameters:
    """Parameters for regime detection and classification."""
    
    # Volatility thresholds
    vix_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 15,
        "normal": 20,
        "high": 30,
        "extreme": 40
    })
    
    # Momentum thresholds
    momentum_periods: List[int] = field(default_factory=lambda: [20, 60, 120])
    momentum_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "strong_negative": -0.10,
        "negative": -0.05,
        "neutral_low": -0.02,
        "neutral_high": 0.02,
        "positive": 0.05,
        "strong_positive": 0.10
    })
    
    # Liquidity parameters
    liquidity_metrics: List[str] = field(default_factory=lambda: [
        "bid_ask_spread",
        "market_depth",
        "volume_ratio",
        "price_impact"
    ])
    
    # Sentiment thresholds
    sentiment_sources: List[str] = field(default_factory=lambda: [
        "put_call_ratio",
        "vix_term_structure",
        "news_sentiment",
        "social_sentiment"
    ])
    
    # Regime classification weights
    indicator_weights: Dict[RegimeIndicatorType, float] = field(default_factory=lambda: {
        RegimeIndicatorType.VOLATILITY: 0.30,
        RegimeIndicatorType.MOMENTUM: 0.25,
        RegimeIndicatorType.LIQUIDITY: 0.20,
        RegimeIndicatorType.SENTIMENT: 0.15,
        RegimeIndicatorType.TECHNICAL: 0.10
    })
    
    # Transition detection
    min_regime_duration_days: int = 20
    transition_smoothing_days: int = 5
    confidence_threshold: float = 0.60
    
    # Model parameters
    lookback_window: int = 252  # 1 year
    update_frequency: str = "daily"
    use_ensemble: bool = True
    
    def get_total_weight(self) -> float:
        """Get total indicator weights (should sum to 1.0)."""
        return sum(self.indicator_weights.values())
    
    def normalize_weights(self):
        """Normalize indicator weights to sum to 1.0."""
        total = self.get_total_weight()
        if total > 0:
            for key in self.indicator_weights:
                self.indicator_weights[key] /= total


@dataclass
class RegimeAnalytics:
    """Analytics and statistics for regime analysis."""
    
    # Regime statistics
    regime_durations: Dict[RegimeType, List[int]] = field(default_factory=dict)
    regime_frequencies: Dict[RegimeType, float] = field(default_factory=dict)
    
    # Transition matrix
    transition_matrix: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    regime_labels: List[str] = field(default_factory=list)
    
    # Performance by regime
    avg_returns: Dict[RegimeType, float] = field(default_factory=dict)
    avg_volatility: Dict[RegimeType, float] = field(default_factory=dict)
    sharpe_ratios: Dict[RegimeType, float] = field(default_factory=dict)
    max_drawdowns: Dict[RegimeType, float] = field(default_factory=dict)
    
    # Indicator effectiveness
    indicator_accuracy: Dict[str, float] = field(default_factory=dict)
    indicator_lead_time: Dict[str, int] = field(default_factory=dict)
    
    # Model performance
    regime_prediction_accuracy: float = 0.0
    transition_prediction_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Time period
    analysis_start: Optional[datetime] = None
    analysis_end: Optional[datetime] = None
    total_observations: int = 0
    
    def calculate_average_regime_duration(self, regime_type: RegimeType) -> float:
        """Calculate average duration for a regime type."""
        if regime_type in self.regime_durations and self.regime_durations[regime_type]:
            return np.mean(self.regime_durations[regime_type])
        return 0.0
    
    def get_most_frequent_regime(self) -> Optional[RegimeType]:
        """Get the most frequently occurring regime."""
        if self.regime_frequencies:
            return max(self.regime_frequencies, key=self.regime_frequencies.get)
        return None
    
    def get_regime_sharpe_ranking(self) -> List[Tuple[RegimeType, float]]:
        """Get regimes ranked by Sharpe ratio."""
        rankings = [(regime, sharpe) for regime, sharpe in self.sharpe_ratios.items()]
        return sorted(rankings, key=lambda x: x[1], reverse=True)