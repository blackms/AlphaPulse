"""
Liquidity risk management configuration parameters.

Defines thresholds, model parameters, and execution settings for
liquidity risk management system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class LiquidityRegime(Enum):
    """Market liquidity regime types."""
    NORMAL = "normal"
    STRESSED = "stressed"
    CRISIS = "crisis"


@dataclass
class LiquidityRiskThresholds:
    """Thresholds for liquidity risk assessment."""
    # Concentration limits (as % of ADV)
    max_position_concentration: float = 0.20      # 20% of ADV
    warning_concentration: float = 0.15           # 15% triggers warning
    
    # Market impact limits (in basis points)
    max_acceptable_impact: float = 50.0           # 50 bps max impact
    critical_impact_threshold: float = 100.0      # 100 bps critical
    
    # Spread thresholds (in basis points)
    normal_spread_threshold: float = 20.0         # Normal conditions
    wide_spread_threshold: float = 50.0           # Wide spread warning
    critical_spread_threshold: float = 100.0      # Critical spread
    
    # Liquidity score thresholds
    min_acceptable_liquidity_score: float = 30.0  # Minimum score
    preferred_liquidity_score: float = 60.0       # Preferred minimum
    
    # Volume thresholds
    min_daily_volume: float = 1e5                 # $100k minimum
    min_turnover_ratio: float = 0.001             # 0.1% of shares outstanding
    
    # Depth thresholds
    max_depth_imbalance: float = 0.5             # 50% imbalance
    min_depth_levels: int = 5                     # Minimum order book levels
    
    # Timing thresholds
    max_liquidation_days: float = 10.0           # Maximum days to liquidate
    urgent_liquidation_hours: float = 4.0         # Hours for urgent liquidation


@dataclass
class SlippageModelParameters:
    """Parameters for slippage prediction models."""
    # Linear model parameters
    linear_temporary_impact: float = 0.10         # 10% of participation
    linear_permanent_impact: float = 0.05         # 5% of participation
    
    # Square-root model parameters
    sqrt_liquidity_factor: float = 0.5            # Liquidity adjustment
    sqrt_volatility_factor: float = 1.0           # Volatility adjustment
    sqrt_urgency_factor: float = 1.5              # Urgency multiplier
    
    # Almgren-Chriss parameters
    ac_temporary_impact_power: float = 1.0        # Linear temporary impact
    ac_permanent_impact_power: float = 0.5        # Square-root permanent
    ac_risk_aversion_default: float = 1.0         # Default risk aversion
    
    # Machine learning parameters
    ml_feature_window: int = 20                   # Days of history
    ml_min_training_samples: int = 1000           # Minimum samples
    ml_retrain_frequency_days: int = 30           # Retrain monthly
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'linear': 0.15,
        'square_root': 0.35,
        'almgren_chriss': 0.30,
        'machine_learning': 0.20
    })


@dataclass
class ExecutionParameters:
    """Parameters for order execution."""
    # Participation rate limits
    min_participation_rate: float = 0.01          # 1% minimum
    max_participation_rate: float = 0.30          # 30% maximum
    preferred_participation_rate: float = 0.10     # 10% preferred
    
    # Time constraints
    min_execution_minutes: int = 5                 # 5 minutes minimum
    max_execution_minutes: int = 390               # Full trading day
    preferred_execution_minutes: int = 60          # 1 hour preferred
    
    # Slice parameters
    min_slice_size_pct: float = 0.01              # 1% of order
    max_slices_per_order: int = 100               # Maximum slices
    preferred_slice_duration_minutes: int = 5      # 5-minute slices
    
    # Strategy selection thresholds
    aggressive_strategy_threshold: float = 0.05    # < 5% of ADV
    passive_strategy_threshold: float = 0.20       # > 20% of ADV
    adaptive_strategy_min_size: float = 0.10       # > 10% of ADV
    
    # Urgency multipliers
    urgency_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.5,
        'medium': 1.0,
        'high': 1.5,
        'critical': 2.0
    })


@dataclass
class LiquidityIndicatorSettings:
    """Settings for liquidity indicator calculations."""
    # Spread calculation settings
    spread_outlier_threshold: float = 3.0          # Z-score threshold
    spread_smoothing_window: int = 20              # Moving average window
    
    # Volume profile settings
    volume_bucket_minutes: int = 30                # 30-minute buckets
    volume_profile_days: int = 20                  # 20-day profile
    volume_seasonality_adjustment: bool = True     # Adjust for day of week
    
    # Market depth settings
    depth_levels_to_analyze: int = 10              # Top 10 levels
    depth_imbalance_threshold: float = 0.3         # 30% imbalance
    depth_aggregation_method: str = "weighted"     # weighted, simple
    
    # Impact measure settings
    amihud_lookback_days: int = 30                # 30-day calculation
    kyle_lambda_window: int = 60                   # 60-minute window
    hasbrouck_lambda_lags: int = 5                 # 5 lags
    
    # Intraday pattern settings
    intraday_buckets: int = 48                     # 30-minute buckets
    pattern_smoothing: bool = True                 # Smooth patterns
    detect_auctions: bool = True                   # Detect opening/closing
    
    # Event detection settings
    liquidity_gap_minutes: int = 5                 # 5-minute gaps
    spread_spike_threshold: float = 3.0            # 3 std devs
    volume_surge_threshold: float = 3.0            # 3 std devs
    event_cooldown_minutes: int = 30               # 30-minute cooldown


@dataclass
class StressTestScenarios:
    """Predefined stress test scenarios."""
    scenarios: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            'name': 'moderate_stress',
            'description': 'Moderate market stress conditions',
            'spread_multiplier': 2.0,
            'volume_multiplier': 0.5,
            'depth_reduction': 0.7,
            'volatility_multiplier': 1.5,
            'severity': 'medium',
            'stress_premium': 1.2
        },
        {
            'name': 'severe_stress',
            'description': 'Severe market stress (2008-like)',
            'spread_multiplier': 5.0,
            'volume_multiplier': 0.2,
            'depth_reduction': 0.3,
            'volatility_multiplier': 3.0,
            'severity': 'high',
            'stress_premium': 2.0
        },
        {
            'name': 'flash_crash',
            'description': 'Flash crash scenario',
            'spread_multiplier': 10.0,
            'volume_multiplier': 0.1,
            'depth_reduction': 0.1,
            'volatility_multiplier': 5.0,
            'severity': 'critical',
            'stress_premium': 3.0
        },
        {
            'name': 'liquidity_drought',
            'description': 'Prolonged liquidity drought',
            'spread_multiplier': 3.0,
            'volume_multiplier': 0.3,
            'depth_reduction': 0.5,
            'volatility_multiplier': 2.0,
            'severity': 'high',
            'stress_premium': 1.5
        }
    ])


@dataclass
class MonitoringSettings:
    """Settings for liquidity monitoring and alerts."""
    # Real-time monitoring
    monitoring_interval_seconds: int = 60           # Check every minute
    fast_market_interval_seconds: int = 10          # 10 seconds in fast markets
    
    # Alert thresholds
    liquidity_score_alert_threshold: float = 30.0   # Alert below 30
    spread_widening_alert_pct: float = 50.0         # 50% widening
    volume_drop_alert_pct: float = 70.0             # 70% volume drop
    
    # Alert cooldowns
    alert_cooldown_minutes: int = 60                # 1 hour between alerts
    critical_alert_cooldown_minutes: int = 15       # 15 minutes for critical
    
    # Metrics retention
    intraday_metrics_retention_days: int = 7        # 7 days
    daily_metrics_retention_days: int = 90          # 90 days
    alert_history_retention_days: int = 30          # 30 days


@dataclass
class LiquidityRiskConfig:
    """Main configuration for liquidity risk management."""
    # Component settings
    risk_thresholds: LiquidityRiskThresholds = field(default_factory=LiquidityRiskThresholds)
    slippage_parameters: SlippageModelParameters = field(default_factory=SlippageModelParameters)
    execution_parameters: ExecutionParameters = field(default_factory=ExecutionParameters)
    indicator_settings: LiquidityIndicatorSettings = field(default_factory=LiquidityIndicatorSettings)
    stress_scenarios: StressTestScenarios = field(default_factory=StressTestScenarios)
    monitoring_settings: MonitoringSettings = field(default_factory=MonitoringSettings)
    
    # Global settings
    enabled: bool = True
    debug_mode: bool = False
    
    # Cache settings
    cache_ttl_seconds: int = 300                    # 5-minute cache
    max_cache_size: int = 1000                      # Max cached items
    
    # Performance settings
    max_concurrent_calculations: int = 10           # Parallel calculations
    calculation_timeout_seconds: int = 30           # Timeout per calculation
    
    # Data quality settings
    min_data_points_required: int = 100             # Minimum data points
    max_data_staleness_minutes: int = 15            # 15 minutes max staleness
    outlier_removal_method: str = "iqr"             # IQR-based outlier removal
    
    def get_regime_adjustments(self, regime: LiquidityRegime) -> Dict[str, float]:
        """Get parameter adjustments for different liquidity regimes."""
        if regime == LiquidityRegime.NORMAL:
            return {
                'spread_multiplier': 1.0,
                'impact_multiplier': 1.0,
                'participation_adjustment': 1.0
            }
        elif regime == LiquidityRegime.STRESSED:
            return {
                'spread_multiplier': 1.5,
                'impact_multiplier': 1.3,
                'participation_adjustment': 0.7
            }
        elif regime == LiquidityRegime.CRISIS:
            return {
                'spread_multiplier': 2.5,
                'impact_multiplier': 2.0,
                'participation_adjustment': 0.3
            }
        else:
            return {
                'spread_multiplier': 1.0,
                'impact_multiplier': 1.0,
                'participation_adjustment': 1.0
            }
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        # Validate thresholds
        if self.risk_thresholds.max_position_concentration <= 0:
            errors.append("max_position_concentration must be positive")
        
        if self.risk_thresholds.max_acceptable_impact < 0:
            errors.append("max_acceptable_impact cannot be negative")
        
        # Validate execution parameters
        if self.execution_parameters.min_participation_rate >= self.execution_parameters.max_participation_rate:
            errors.append("min_participation_rate must be less than max_participation_rate")
        
        if self.execution_parameters.min_execution_minutes <= 0:
            errors.append("min_execution_minutes must be positive")
        
        # Validate model parameters
        total_weight = sum(self.slippage_parameters.ensemble_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            errors.append(f"ensemble_weights must sum to 1.0, got {total_weight}")
        
        return errors


# Default configuration instance
DEFAULT_LIQUIDITY_CONFIG = LiquidityRiskConfig()


# Preset configurations for different trading styles
CONSERVATIVE_LIQUIDITY_CONFIG = LiquidityRiskConfig(
    risk_thresholds=LiquidityRiskThresholds(
        max_position_concentration=0.10,  # 10% of ADV
        max_acceptable_impact=30.0,       # 30 bps
        min_acceptable_liquidity_score=50.0
    ),
    execution_parameters=ExecutionParameters(
        max_participation_rate=0.15,      # 15% max
        preferred_participation_rate=0.05  # 5% preferred
    )
)

AGGRESSIVE_LIQUIDITY_CONFIG = LiquidityRiskConfig(
    risk_thresholds=LiquidityRiskThresholds(
        max_position_concentration=0.30,  # 30% of ADV
        max_acceptable_impact=100.0,      # 100 bps
        min_acceptable_liquidity_score=20.0
    ),
    execution_parameters=ExecutionParameters(
        max_participation_rate=0.50,      # 50% max
        preferred_participation_rate=0.20  # 20% preferred
    )
)


def get_liquidity_config(style: str = "default") -> LiquidityRiskConfig:
    """Get liquidity configuration for a given trading style."""
    configs = {
        "default": DEFAULT_LIQUIDITY_CONFIG,
        "conservative": CONSERVATIVE_LIQUIDITY_CONFIG,
        "aggressive": AGGRESSIVE_LIQUIDITY_CONFIG
    }
    
    return configs.get(style, DEFAULT_LIQUIDITY_CONFIG)