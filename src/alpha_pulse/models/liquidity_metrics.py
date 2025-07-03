"""
Liquidity metrics data models.

Defines data structures for liquidity measurement, market depth,
and trading volume analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class LiquidityTier(Enum):
    """Liquidity tier classification."""
    ULTRA_LIQUID = "ultra_liquid"
    LIQUID = "liquid"
    MODERATE = "moderate"
    ILLIQUID = "illiquid"
    DISTRESSED = "distressed"


class SpreadType(Enum):
    """Types of spread measurements."""
    QUOTED = "quoted"
    EFFECTIVE = "effective"
    REALIZED = "realized"
    ROLL = "roll"
    CORWIN_SCHULTZ = "corwin_schultz"


class LiquidityIndicatorType(Enum):
    """Types of liquidity indicators."""
    SPREAD = "spread"
    VOLUME = "volume"
    DEPTH = "depth"
    IMPACT = "impact"
    RESILIENCE = "resilience"
    TURNOVER = "turnover"


@dataclass
class MarketDepth:
    """Market depth at multiple price levels."""
    timestamp: datetime
    symbol: str
    bid_levels: List[Tuple[float, float]]  # (price, size) pairs
    ask_levels: List[Tuple[float, float]]  # (price, size) pairs
    total_bid_depth: float
    total_ask_depth: float
    
    def get_depth_at_bps(self, basis_points: float) -> Tuple[float, float]:
        """Get bid/ask depth at specified basis points from mid."""
        if not self.bid_levels or not self.ask_levels:
            return 0.0, 0.0
        
        mid_price = (self.bid_levels[0][0] + self.ask_levels[0][0]) / 2
        threshold = mid_price * basis_points / 10000
        
        bid_depth = sum(
            size for price, size in self.bid_levels
            if mid_price - price <= threshold
        )
        ask_depth = sum(
            size for price, size in self.ask_levels
            if price - mid_price <= threshold
        )
        
        return bid_depth, ask_depth
    
    def get_imbalance(self) -> float:
        """Calculate order book imbalance."""
        if self.total_bid_depth + self.total_ask_depth == 0:
            return 0.0
        return (self.total_bid_depth - self.total_ask_depth) / (
            self.total_bid_depth + self.total_ask_depth
        )


@dataclass
class SpreadMeasurement:
    """Spread measurements for a security."""
    timestamp: datetime
    symbol: str
    spread_type: SpreadType
    value: float  # In basis points
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    mid_price: Optional[float] = None
    
    def to_percentage(self) -> float:
        """Convert basis points to percentage."""
        return self.value / 100


@dataclass
class VolumeProfile:
    """Intraday volume profile."""
    symbol: str
    date: datetime
    hourly_volumes: Dict[int, float]  # Hour -> Volume
    total_volume: float
    average_volume: float
    volume_curve: Optional[np.ndarray] = None  # Normalized volume curve
    
    def get_participation_rate(self, hour: int, target_volume: float) -> float:
        """Calculate recommended participation rate for given hour."""
        if hour not in self.hourly_volumes or self.total_volume == 0:
            return 0.0
        
        hourly_pct = self.hourly_volumes[hour] / self.total_volume
        if hourly_pct == 0:
            return 0.0
        
        # Target participation rate (e.g., 10% of hourly volume)
        return min(0.1, target_volume / (self.hourly_volumes[hour] * 0.1))


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics for a security."""
    symbol: str
    timestamp: datetime
    
    # Spread metrics
    quoted_spread: Optional[float] = None  # Basis points
    effective_spread: Optional[float] = None
    realized_spread: Optional[float] = None
    roll_spread: Optional[float] = None
    
    # Volume metrics
    daily_volume: Optional[float] = None
    average_daily_volume: Optional[float] = None
    turnover_ratio: Optional[float] = None
    volume_volatility: Optional[float] = None
    
    # Depth metrics
    market_depth: Optional[MarketDepth] = None
    depth_imbalance: Optional[float] = None
    
    # Impact metrics
    amihud_illiquidity: Optional[float] = None
    kyle_lambda: Optional[float] = None
    hasbrouck_lambda: Optional[float] = None
    price_impact_10bps: Optional[float] = None  # Impact for 10bps participation
    
    # Advanced metrics
    liquidity_score: Optional[float] = None  # 0-100
    liquidity_tier: Optional[LiquidityTier] = None
    resilience_score: Optional[float] = None
    
    # Intraday patterns
    intraday_spread_pattern: Optional[Dict[int, float]] = None
    intraday_volume_pattern: Optional[Dict[int, float]] = None
    
    def calculate_liquidity_score(self) -> float:
        """Calculate composite liquidity score (0-100)."""
        scores = []
        weights = []
        
        # Spread score (lower is better)
        if self.quoted_spread is not None:
            spread_score = max(0, 100 - self.quoted_spread)
            scores.append(spread_score)
            weights.append(0.3)
        
        # Volume score (higher is better, normalized)
        if self.turnover_ratio is not None:
            volume_score = min(100, self.turnover_ratio * 100)
            scores.append(volume_score)
            weights.append(0.3)
        
        # Impact score (lower is better)
        if self.amihud_illiquidity is not None:
            impact_score = max(0, 100 - min(100, self.amihud_illiquidity * 1000))
            scores.append(impact_score)
            weights.append(0.2)
        
        # Depth score
        if self.depth_imbalance is not None:
            depth_score = 100 * (1 - abs(self.depth_imbalance))
            scores.append(depth_score)
            weights.append(0.2)
        
        if not scores:
            return 50.0  # Default neutral score
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            self.liquidity_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            self.liquidity_score = 50.0
        
        # Determine tier
        if self.liquidity_score >= 80:
            self.liquidity_tier = LiquidityTier.ULTRA_LIQUID
        elif self.liquidity_score >= 60:
            self.liquidity_tier = LiquidityTier.LIQUID
        elif self.liquidity_score >= 40:
            self.liquidity_tier = LiquidityTier.MODERATE
        elif self.liquidity_score >= 20:
            self.liquidity_tier = LiquidityTier.ILLIQUID
        else:
            self.liquidity_tier = LiquidityTier.DISTRESSED
        
        return self.liquidity_score


@dataclass
class LiquiditySnapshot:
    """Point-in-time liquidity snapshot."""
    timestamp: datetime
    metrics: Dict[str, LiquidityMetrics]  # Symbol -> Metrics
    market_conditions: str  # "normal", "stressed", "crisis"
    aggregate_liquidity_score: float
    
    def get_liquid_symbols(self, min_score: float = 60) -> List[str]:
        """Get symbols meeting liquidity threshold."""
        return [
            symbol for symbol, metrics in self.metrics.items()
            if metrics.liquidity_score and metrics.liquidity_score >= min_score
        ]
    
    def get_illiquid_symbols(self, max_score: float = 40) -> List[str]:
        """Get symbols below liquidity threshold."""
        return [
            symbol for symbol, metrics in self.metrics.items()
            if metrics.liquidity_score and metrics.liquidity_score < max_score
        ]


@dataclass
class IntradayLiquidity:
    """Intraday liquidity patterns and metrics."""
    symbol: str
    date: datetime
    
    # Time-of-day patterns
    hourly_spreads: Dict[int, float]  # Hour -> Spread (bps)
    hourly_volumes: Dict[int, float]  # Hour -> Volume
    hourly_depths: Dict[int, Tuple[float, float]]  # Hour -> (bid_depth, ask_depth)
    
    # Liquidity events
    liquidity_gaps: List[Tuple[datetime, datetime]]  # Periods of low liquidity
    spread_spikes: List[Tuple[datetime, float]]  # Time, spread value
    volume_surges: List[Tuple[datetime, float]]  # Time, volume
    
    # Statistics
    avg_spread: float
    spread_volatility: float
    volume_concentration: float  # Herfindahl index of hourly volumes
    liquidity_factor: float  # Composite liquidity factor
    
    def get_optimal_trading_hours(self, min_liquidity_factor: float = 0.7) -> List[int]:
        """Get hours with sufficient liquidity for trading."""
        optimal_hours = []
        
        for hour in range(24):
            if hour not in self.hourly_spreads:
                continue
            
            # Calculate hourly liquidity factor
            spread_score = 1 - (self.hourly_spreads[hour] / 100)  # Lower spread is better
            volume_score = self.hourly_volumes.get(hour, 0) / max(self.hourly_volumes.values())
            
            hourly_factor = (spread_score + volume_score) / 2
            
            if hourly_factor >= min_liquidity_factor:
                optimal_hours.append(hour)
        
        return optimal_hours


@dataclass
class LiquidityEvent:
    """Liquidity-related market event."""
    timestamp: datetime
    symbol: str
    event_type: str  # "spread_widening", "volume_drop", "depth_depletion", etc.
    severity: str  # "low", "medium", "high", "critical"
    metrics_before: LiquidityMetrics
    metrics_after: LiquidityMetrics
    duration_seconds: Optional[int] = None
    impact_estimate: Optional[float] = None  # Estimated price impact
    
    def get_liquidity_degradation(self) -> float:
        """Calculate liquidity score degradation."""
        score_before = self.metrics_before.liquidity_score or 50
        score_after = self.metrics_after.liquidity_score or 50
        return max(0, score_before - score_after)


@dataclass
class CrossAssetLiquidity:
    """Cross-asset liquidity comparison."""
    timestamp: datetime
    asset_type: str  # "equity", "forex", "crypto", "futures"
    
    # Aggregated metrics by asset class
    avg_spreads: Dict[str, float]  # Asset class -> Average spread
    avg_volumes: Dict[str, float]  # Asset class -> Average volume
    avg_impacts: Dict[str, float]  # Asset class -> Average impact
    
    # Relative liquidity
    liquidity_rankings: List[Tuple[str, float]]  # Symbol, score
    correlation_matrix: Optional[np.ndarray] = None  # Liquidity correlations
    
    def get_most_liquid_assets(self, n: int = 10) -> List[str]:
        """Get top N most liquid assets."""
        return [symbol for symbol, _ in self.liquidity_rankings[:n]]
    
    def get_liquidity_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """Get liquidity correlation between two assets."""
        # Implementation would use correlation matrix
        return None