"""
Feature engineering module for AlphaPulse.
"""
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from loguru import logger


def calculate_rolling_stats(
    data: Union[pd.Series, np.ndarray],
    window: int
) -> Dict[str, np.ndarray]:
    """
    Calculate rolling statistics.

    Args:
        data: Input data
        window: Window size for rolling calculations

    Returns:
        Dict containing rolling statistics (mean, std, min, max, etc.)
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    stats = {
        'mean': data.rolling(window=window).mean().values,
        'std': data.rolling(window=window).std().values,
        'min': data.rolling(window=window).min().values,
        'max': data.rolling(window=window).max().values,
        'median': data.rolling(window=window).median().values,
        'skew': data.rolling(window=window).skew().values,
        'kurt': data.rolling(window=window).kurt().values,
    }
    
    return stats


def calculate_sma(prices: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Args:
        prices: Price data
        window: Window size for moving average

    Returns:
        Array of SMA values
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return prices.rolling(window=window).mean().values


def calculate_ema(prices: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Price data
        window: Window size for moving average

    Returns:
        Array of EMA values
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Initialize with NaN for the first value
    ema = pd.Series(index=prices.index, dtype=float)
    ema.iloc[0] = np.nan
    
    # Calculate alpha (smoothing factor)
    alpha = 2 / (window + 1)
    
    # Calculate EMA starting from the second value
    ema.iloc[1:] = prices.iloc[1:].ewm(
        span=window,
        adjust=False,
        min_periods=window
    ).mean()
    
    return ema.values


def calculate_rsi(prices: Union[pd.Series, np.ndarray], window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Price data
        window: Window size for RSI calculation

    Returns:
        Array of RSI values
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values


def calculate_macd(
    prices: Union[pd.Series, np.ndarray],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Args:
        prices: Price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Calculate EMAs
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values


def calculate_bollinger_bands(
    prices: Union[pd.Series, np.ndarray],
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Price data
        window: Window size for moving average
        num_std: Number of standard deviations for bands

    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Calculate middle band (SMA)
    middle_band = prices.rolling(window=window).mean()
    
    # Calculate standard deviation
    std = prices.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band.values, middle_band.values, lower_band.values


class FeatureStore:
    """Manages feature computation and caching."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize feature store.

        Args:
            cache_dir: Directory to cache computed features
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path('feature_cache')
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized FeatureStore with cache dir: {self.cache_dir}")

    def compute_technical_indicators(
        self,
        data: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute technical indicators for price data.

        Args:
            data: DataFrame with OHLCV data
            windows: List of lookback periods for indicators

        Returns:
            DataFrame with computed indicators
        """
        if windows is None:
            windows = [12, 26, 50, 100, 200]
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Rolling statistics
        stats = calculate_rolling_stats(data['close'], window=20)
        features['volatility'] = stats['std']
        features['rolling_mean'] = stats['mean']
        features['rolling_min'] = stats['min']
        features['rolling_max'] = stats['max']
        features['rolling_median'] = stats['median']
        features['rolling_skew'] = stats['skew']
        features['rolling_kurt'] = stats['kurt']
        
        # Moving averages
        for window in [12, 26, 50, 100, 200]:
            features[f'sma_{window}'] = calculate_sma(data['close'], window)
            features[f'ema_{window}'] = calculate_ema(data['close'], window)
        
        # RSI
        features['rsi'] = calculate_rsi(data['close'])
        
        # MACD
        macd, signal, hist = calculate_macd(data['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(data['close'])
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        
        # Price momentum
        for window in [5, 10, 20]:
            features[f'momentum_{window}'] = data['close'].pct_change(window)
        
        logger.info(f"Generated {len(features.columns)} features")
        return features

    def add_features(self, name: str, features: pd.DataFrame) -> None:
        """
        Add features to cache.

        Args:
            name: Name to save features under
            features: DataFrame of features to cache
        """
        save_path = self.cache_dir / f"{name}.joblib"
        joblib.dump(features, save_path)

    def get_features(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get features from cache.

        Args:
            name: Name of features to retrieve

        Returns:
            DataFrame of features or None if not found
        """
        load_path = self.cache_dir / f"{name}.joblib"
        if not load_path.exists():
            return None
        return joblib.load(load_path)