"""
AlphaPulse Feature Engineering Package

This package provides tools and utilities for financial feature engineering,
including technical indicators, rolling statistics, and feature management.
"""

from .feature_engineering import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_rolling_stats,
    FeatureStore
)

__all__ = [
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_rolling_stats',
    'FeatureStore'
]