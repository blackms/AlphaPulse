"""
Configuration parameters for market regime detection and risk budgeting.

Defines thresholds, weights, and parameters for regime classification
and dynamic risk allocation.
"""

from typing import Dict, List, Any
from alpha_pulse.models.market_regime import RegimeType, RegimeIndicatorType


# Regime Detection Parameters
REGIME_DETECTION_PARAMS = {
    "lookback_window": 252,  # Trading days for analysis
    "min_regime_duration": 20,  # Minimum days for regime stability
    "transition_smoothing": 5,  # Days for smoothing transitions
    "confidence_threshold": 0.60,  # Minimum confidence for regime classification
    
    # Indicator weights for regime detection
    "indicator_weights": {
        RegimeIndicatorType.VOLATILITY: 0.30,
        RegimeIndicatorType.MOMENTUM: 0.25,
        RegimeIndicatorType.LIQUIDITY: 0.20,
        RegimeIndicatorType.SENTIMENT: 0.15,
        RegimeIndicatorType.TECHNICAL: 0.10
    },
    
    # Update frequencies
    "update_frequencies": {
        "real_time": ["volatility", "liquidity"],
        "minute": ["momentum", "technical"],
        "hourly": ["sentiment", "regime_classification"],
        "daily": ["model_retraining", "parameter_calibration"]
    }
}

# Volatility Regime Thresholds
VOLATILITY_THRESHOLDS = {
    "vix_levels": {
        "low": 15,
        "normal": 20,
        "high": 30,
        "extreme": 40
    },
    
    "realized_vol_levels": {
        "low": 0.10,  # 10% annualized
        "normal": 0.16,
        "high": 0.25,
        "extreme": 0.35
    },
    
    "vol_of_vol_levels": {
        "stable": 0.05,
        "normal": 0.10,
        "unstable": 0.20
    },
    
    # Term structure thresholds
    "term_structure": {
        "backwardation": -0.10,  # Front month 10% higher
        "normal_range": (-0.05, 0.05),
        "contango": 0.10
    }
}

# Momentum Regime Parameters
MOMENTUM_THRESHOLDS = {
    "return_thresholds": {
        "strong_negative": -0.10,  # -10% over period
        "negative": -0.05,
        "neutral_low": -0.02,
        "neutral_high": 0.02,
        "positive": 0.05,
        "strong_positive": 0.10
    },
    
    "trend_strength": {
        "no_trend": 20,  # ADX < 20
        "weak_trend": 25,
        "strong_trend": 40,
        "very_strong_trend": 50
    },
    
    "momentum_periods": [20, 60, 120, 252],  # Days
    
    "moving_average_periods": {
        "short": 50,
        "medium": 100,
        "long": 200
    }
}

# Liquidity Parameters
LIQUIDITY_PARAMS = {
    "bid_ask_thresholds": {
        "tight": 0.0005,  # 5 bps
        "normal": 0.0010,  # 10 bps
        "wide": 0.0020,  # 20 bps
        "very_wide": 0.0050  # 50 bps
    },
    
    "volume_ratios": {
        "low": 0.5,  # 50% of average
        "normal": (0.8, 1.2),
        "high": 1.5,
        "spike": 2.0
    },
    
    "market_impact_thresholds": {
        "minimal": 0.0001,
        "low": 0.0005,
        "moderate": 0.0010,
        "high": 0.0025
    }
}

# Sentiment Indicators
SENTIMENT_PARAMS = {
    "put_call_ratios": {
        "extreme_bullish": 0.5,
        "bullish": 0.7,
        "neutral": (0.8, 1.2),
        "bearish": 1.5,
        "extreme_bearish": 2.0
    },
    
    "sentiment_scores": {
        "extreme_negative": -0.7,
        "negative": -0.3,
        "neutral": (-0.2, 0.2),
        "positive": 0.3,
        "extreme_positive": 0.7
    },
    
    "fear_greed_levels": {
        "extreme_fear": 20,
        "fear": 35,
        "neutral": (45, 55),
        "greed": 65,
        "extreme_greed": 80
    }
}

# Regime Classification Rules
REGIME_CLASSIFICATION_RULES = {
    RegimeType.BULL: {
        "conditions": {
            "vix_max": 20,
            "momentum_min": 0.05,  # 3-month momentum
            "sentiment_min": 0.3,
            "liquidity_min": 0.0,
            "trend_strength_min": 25
        },
        "confidence_boost": {
            "all_conditions_met": 0.2,
            "momentum_strong": 0.1,
            "low_volatility": 0.1
        }
    },
    
    RegimeType.BEAR: {
        "conditions": {
            "vix_min": 30,
            "momentum_max": -0.05,
            "sentiment_max": -0.3,
            "liquidity_max": -0.3,
            "volume_spike": True
        },
        "confidence_boost": {
            "high_volatility": 0.2,
            "negative_momentum": 0.15,
            "poor_sentiment": 0.1
        }
    },
    
    RegimeType.SIDEWAYS: {
        "conditions": {
            "vix_range": (20, 30),
            "momentum_range": (-0.02, 0.02),
            "sentiment_range": (-0.2, 0.2),
            "trend_strength_max": 25
        },
        "confidence_boost": {
            "low_momentum": 0.2,
            "neutral_sentiment": 0.1,
            "range_bound": 0.15
        }
    },
    
    RegimeType.CRISIS: {
        "conditions": {
            "vix_min": 40,
            "momentum_max": -0.10,  # 1-month momentum
            "liquidity_crisis": True,
            "correlation_breakdown": True
        },
        "confidence_boost": {
            "extreme_volatility": 0.3,
            "liquidity_issues": 0.2,
            "panic_sentiment": 0.2
        }
    },
    
    RegimeType.RECOVERY: {
        "conditions": {
            "previous_regime": RegimeType.CRISIS,
            "vix_decreasing": True,
            "momentum_improving": True,
            "sentiment_improving": True
        },
        "confidence_boost": {
            "from_crisis": 0.3,
            "momentum_positive": 0.15,
            "volatility_declining": 0.15
        }
    }
}

# Risk Budget Parameters by Regime
RISK_BUDGET_PARAMS = {
    RegimeType.BULL: {
        "volatility_target_multiplier": 1.2,
        "max_leverage": 1.5,
        "position_limits": {
            "max_single_position": 0.15,
            "max_sector_concentration": 0.40,
            "min_positions": 8
        },
        "risk_allocation": {
            "equity_allocation": 0.80,
            "defensive_allocation": 0.10,
            "alternatives_allocation": 0.10
        },
        "rebalancing": {
            "frequency": "weekly",
            "threshold": 0.10,  # 10% drift
            "urgency": "low"
        }
    },
    
    RegimeType.BEAR: {
        "volatility_target_multiplier": 0.7,
        "max_leverage": 0.5,
        "position_limits": {
            "max_single_position": 0.08,
            "max_sector_concentration": 0.25,
            "min_positions": 15
        },
        "risk_allocation": {
            "equity_allocation": 0.40,
            "defensive_allocation": 0.40,
            "alternatives_allocation": 0.20
        },
        "rebalancing": {
            "frequency": "daily",
            "threshold": 0.05,
            "urgency": "high"
        }
    },
    
    RegimeType.SIDEWAYS: {
        "volatility_target_multiplier": 1.0,
        "max_leverage": 1.0,
        "position_limits": {
            "max_single_position": 0.10,
            "max_sector_concentration": 0.35,
            "min_positions": 10
        },
        "risk_allocation": {
            "equity_allocation": 0.60,
            "defensive_allocation": 0.25,
            "alternatives_allocation": 0.15
        },
        "rebalancing": {
            "frequency": "weekly",
            "threshold": 0.08,
            "urgency": "medium"
        }
    },
    
    RegimeType.CRISIS: {
        "volatility_target_multiplier": 0.4,
        "max_leverage": 0.2,
        "position_limits": {
            "max_single_position": 0.05,
            "max_sector_concentration": 0.15,
            "min_positions": 20
        },
        "risk_allocation": {
            "equity_allocation": 0.20,
            "defensive_allocation": 0.60,
            "alternatives_allocation": 0.20
        },
        "rebalancing": {
            "frequency": "real-time",
            "threshold": 0.02,
            "urgency": "critical"
        }
    },
    
    RegimeType.RECOVERY: {
        "volatility_target_multiplier": 0.8,
        "max_leverage": 0.8,
        "position_limits": {
            "max_single_position": 0.10,
            "max_sector_concentration": 0.30,
            "min_positions": 12
        },
        "risk_allocation": {
            "equity_allocation": 0.50,
            "defensive_allocation": 0.30,
            "alternatives_allocation": 0.20
        },
        "rebalancing": {
            "frequency": "daily",
            "threshold": 0.06,
            "urgency": "high"
        }
    }
}

# Strategy Preferences by Regime
REGIME_STRATEGY_PREFERENCES = {
    RegimeType.BULL: {
        "preferred": [
            "momentum",
            "growth",
            "trend_following",
            "breakout",
            "sector_rotation",
            "small_cap"
        ],
        "avoided": [
            "mean_reversion",
            "defensive",
            "value_deep",
            "short_selling"
        ],
        "factor_tilts": {
            "momentum": 1.5,
            "growth": 1.3,
            "quality": 1.1,
            "value": 0.8,
            "low_volatility": 0.7
        }
    },
    
    RegimeType.BEAR: {
        "preferred": [
            "defensive",
            "quality",
            "low_volatility",
            "value",
            "short_selling",
            "cash"
        ],
        "avoided": [
            "momentum",
            "growth",
            "small_cap",
            "high_beta",
            "leverage"
        ],
        "factor_tilts": {
            "momentum": 0.5,
            "growth": 0.6,
            "quality": 1.5,
            "value": 1.2,
            "low_volatility": 1.8
        }
    },
    
    RegimeType.SIDEWAYS: {
        "preferred": [
            "mean_reversion",
            "pairs_trading",
            "market_neutral",
            "volatility_selling",
            "range_trading"
        ],
        "avoided": [
            "trend_following",
            "breakout",
            "momentum",
            "directional_bets"
        ],
        "factor_tilts": {
            "momentum": 0.8,
            "growth": 0.9,
            "quality": 1.2,
            "value": 1.1,
            "low_volatility": 1.0
        }
    },
    
    RegimeType.CRISIS: {
        "preferred": [
            "cash",
            "tail_hedging",
            "defensive",
            "quality",
            "liquidity"
        ],
        "avoided": [
            "leverage",
            "concentration",
            "illiquid",
            "high_beta",
            "emerging_markets"
        ],
        "factor_tilts": {
            "momentum": 0.3,
            "growth": 0.4,
            "quality": 2.0,
            "value": 0.8,
            "low_volatility": 2.5
        }
    },
    
    RegimeType.RECOVERY: {
        "preferred": [
            "quality_growth",
            "selective_value",
            "moderate_momentum",
            "sector_leaders"
        ],
        "avoided": [
            "excessive_risk",
            "leverage",
            "speculative",
            "low_quality"
        ],
        "factor_tilts": {
            "momentum": 1.0,
            "growth": 1.1,
            "quality": 1.4,
            "value": 1.2,
            "low_volatility": 0.9
        }
    }
}

# Transition Probabilities (Historical)
REGIME_TRANSITION_MATRIX = {
    # From -> To probabilities
    RegimeType.BULL: {
        RegimeType.BULL: 0.85,
        RegimeType.BEAR: 0.05,
        RegimeType.SIDEWAYS: 0.08,
        RegimeType.CRISIS: 0.01,
        RegimeType.RECOVERY: 0.01
    },
    RegimeType.BEAR: {
        RegimeType.BULL: 0.10,
        RegimeType.BEAR: 0.70,
        RegimeType.SIDEWAYS: 0.10,
        RegimeType.CRISIS: 0.08,
        RegimeType.RECOVERY: 0.02
    },
    RegimeType.SIDEWAYS: {
        RegimeType.BULL: 0.20,
        RegimeType.BEAR: 0.15,
        RegimeType.SIDEWAYS: 0.60,
        RegimeType.CRISIS: 0.03,
        RegimeType.RECOVERY: 0.02
    },
    RegimeType.CRISIS: {
        RegimeType.BULL: 0.02,
        RegimeType.BEAR: 0.20,
        RegimeType.SIDEWAYS: 0.08,
        RegimeType.CRISIS: 0.50,
        RegimeType.RECOVERY: 0.20
    },
    RegimeType.RECOVERY: {
        RegimeType.BULL: 0.30,
        RegimeType.BEAR: 0.10,
        RegimeType.SIDEWAYS: 0.20,
        RegimeType.CRISIS: 0.05,
        RegimeType.RECOVERY: 0.35
    }
}

# Model Training Parameters
MODEL_TRAINING_PARAMS = {
    "hmm": {
        "n_states": 5,
        "covariance_type": "full",
        "n_iter": 100,
        "tol": 0.01
    },
    
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt"
    },
    
    "gmm": {
        "n_components": 5,
        "covariance_type": "full",
        "max_iter": 100,
        "n_init": 10
    },
    
    "ensemble": {
        "voting": "soft",
        "weights": {
            "hmm": 0.35,
            "random_forest": 0.35,
            "gmm": 0.20,
            "rule_based": 0.10
        }
    },
    
    "retraining": {
        "frequency": "monthly",
        "min_data_points": 1000,
        "validation_split": 0.2,
        "cross_validation_folds": 5
    }
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "regime_change": {
        "confidence_threshold": 0.70,
        "severity_map": {
            RegimeType.CRISIS: "critical",
            RegimeType.BEAR: "high",
            RegimeType.RECOVERY: "medium",
            RegimeType.SIDEWAYS: "low",
            RegimeType.BULL: "info"
        }
    },
    
    "risk_budget": {
        "utilization_warning": 0.80,
        "utilization_critical": 0.95,
        "volatility_breach": 0.02,  # 2% above target
        "concentration_warning": 0.35,
        "concentration_critical": 0.45
    },
    
    "rebalancing": {
        "drift_warning": 0.08,
        "drift_critical": 0.15,
        "turnover_warning": 0.20,
        "turnover_critical": 0.40
    }
}

# Performance Metrics
PERFORMANCE_METRICS = {
    "tracking": {
        "metrics": [
            "regime_prediction_accuracy",
            "transition_detection_lag",
            "risk_budget_utilization",
            "volatility_targeting_accuracy",
            "rebalancing_effectiveness"
        ],
        "frequency": "daily",
        "retention_days": 365
    },
    
    "benchmarks": {
        "regime_detection_accuracy": 0.70,
        "volatility_forecast_accuracy": 0.80,
        "risk_budget_efficiency": 0.85,
        "rebalancing_cost_ratio": 0.001  # 10 bps
    }
}


def get_regime_parameters(regime_type: RegimeType) -> Dict[str, Any]:
    """Get all parameters for a specific regime."""
    return {
        "classification_rules": REGIME_CLASSIFICATION_RULES.get(regime_type, {}),
        "risk_budget": RISK_BUDGET_PARAMS.get(regime_type, {}),
        "strategy_preferences": REGIME_STRATEGY_PREFERENCES.get(regime_type, {}),
        "transition_probabilities": REGIME_TRANSITION_MATRIX.get(regime_type, {})
    }


def get_indicator_thresholds(indicator_type: str) -> Dict[str, Any]:
    """Get thresholds for a specific indicator type."""
    threshold_map = {
        "volatility": VOLATILITY_THRESHOLDS,
        "momentum": MOMENTUM_THRESHOLDS,
        "liquidity": LIQUIDITY_PARAMS,
        "sentiment": SENTIMENT_PARAMS
    }
    
    return threshold_map.get(indicator_type, {})