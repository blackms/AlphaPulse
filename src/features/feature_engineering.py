"""
Feature Engineering Module

This module provides functions and classes for financial feature engineering,
including technical indicators and a feature store for caching computed features.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import joblib


def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price series data
        window: Rolling window size

    Returns:
        pd.Series: Simple Moving Average series
    """
    return data.rolling(window=window).mean()


def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price series data
        window: Rolling window size

    Returns:
        pd.Series: Exponential Moving Average series
    """
    return data.ewm(span=window, adjust=False, min_periods=window).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        data: Price series data
        window: Rolling window size (default: 14)

    Returns:
        pd.Series: RSI values series
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        data: Price series data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple containing:
        - MACD line
        - Signal line
        - MACD histogram
    """
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        data: Price series data
        window: Rolling window size (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Tuple containing:
        - Upper band
        - Middle band (SMA)
        - Lower band
    """
    middle_band = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_rolling_stats(
    data: pd.Series,
    window: int
) -> Dict[str, pd.Series]:
    """
    Calculate various rolling window statistics.

    Args:
        data: Price series data
        window: Rolling window size

    Returns:
        Dictionary containing various rolling statistics
    """
    return {
        'mean': data.rolling(window=window).mean(),
        'std': data.rolling(window=window).std(),
        'min': data.rolling(window=window).min(),
        'max': data.rolling(window=window).max(),
        'median': data.rolling(window=window).median(),
        'skew': data.rolling(window=window).skew(),
        'kurt': data.rolling(window=window).kurt()
    }


class FeatureStore:
    """
    A class to manage and cache computed features.
    
    This class provides functionality to compute, store, and retrieve
    technical indicators and other features, with caching capabilities
    to improve performance.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize FeatureStore.

        Args:
            cache_dir: Directory to store cached features (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.features: Dict[str, pd.DataFrame] = {}

    def compute_technical_indicators(
        self,
        price_data: pd.Series,
        windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Compute multiple technical indicators for given price data.

        Args:
            price_data: Price series data
            windows: List of window sizes for indicators

        Returns:
            DataFrame containing all computed indicators
        """
        features = pd.DataFrame(index=price_data.index)
        
        for window in windows:
            # Moving averages
            features[f'sma_{window}'] = calculate_sma(price_data, window)
            features[f'ema_{window}'] = calculate_ema(price_data, window)
            
            # Bollinger Bands
            upper, middle, lower = calculate_bollinger_bands(price_data, window)
            features[f'bb_upper_{window}'] = upper
            features[f'bb_middle_{window}'] = middle
            features[f'bb_lower_{window}'] = lower
            
            # Rolling statistics
            stats = calculate_rolling_stats(price_data, window)
            for stat_name, stat_values in stats.items():
                features[f'{stat_name}_{window}'] = stat_values
        
        # RSI
        features['rsi'] = calculate_rsi(price_data)
        
        # MACD
        macd, signal, hist = calculate_macd(price_data)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        return features

    def add_features(self, name: str, features: pd.DataFrame) -> None:
        """
        Add features to the store.

        Args:
            name: Identifier for the feature set
            features: DataFrame containing the features
        """
        self.features[name] = features
        if self.cache_dir:
            self._save_to_cache(name, features)

    def get_features(self, name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve features from the store.

        Args:
            name: Identifier for the feature set

        Returns:
            DataFrame containing the features if found, None otherwise
        """
        if name in self.features:
            return self.features[name]
        elif self.cache_dir:
            return self._load_from_cache(name)
        return None

    def _save_to_cache(self, name: str, features: pd.DataFrame) -> None:
        """Save features to cache directory."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{name}.joblib"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(features, cache_path)

    def _load_from_cache(self, name: str) -> Optional[pd.DataFrame]:
        """Load features from cache directory."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{name}.joblib"
            if cache_path.exists():
                return joblib.load(cache_path)
        return None