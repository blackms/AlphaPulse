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


def calculate_sma(prices: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        prices: Price data
        window: Window size for moving average

    Returns:
        Series of SMA values
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return prices.rolling(window=window).mean()


def calculate_ema(prices: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Price data
        window: Window size for moving average

    Returns:
        Series of EMA values
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return prices.ewm(span=window, adjust=False, min_periods=window).mean()


def calculate_rsi(prices: Union[pd.Series, np.ndarray], window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Price data
        window: Window size for RSI calculation

    Returns:
        Series of RSI values
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
    
    # Use smaller window if data length is less than window size
    actual_window = min(window, len(prices) - 1)
    if actual_window < 2:
        return pd.Series(np.nan, index=prices.index)
    
    # Calculate initial averages using simple moving average
    avg_gains = gains.rolling(window=actual_window, min_periods=1).mean()
    avg_losses = losses.rolling(window=actual_window, min_periods=1).mean()
    
    # Calculate subsequent values using exponential moving average
    avg_gains = gains.ewm(alpha=1/actual_window, min_periods=1, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/actual_window, min_periods=1, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge cases
    rsi[avg_losses == 0] = 100
    rsi[avg_gains == 0] = 0
    rsi[0] = np.nan  # First value should be NaN
    
    return rsi


def calculate_macd(
    prices: Union[pd.Series, np.ndarray],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: Union[pd.Series, np.ndarray],
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
    
    # Convert to pandas Series if numpy array
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Use smaller window if data length is less than window size
    actual_window = min(window, len(prices))
    if actual_window < 2:
        return (pd.Series(np.nan, index=prices.index),
                pd.Series(np.nan, index=prices.index),
                pd.Series(np.nan, index=prices.index))
    
    # Calculate middle band (SMA)
    middle_band = prices.rolling(window=actual_window, min_periods=2).mean()
    
    # Calculate standard deviation
    std = prices.rolling(window=actual_window, min_periods=2).std()
    
    # Calculate standard deviation based band width
    band_width = np.abs(std * num_std)  # Use absolute value to ensure positive width
    
    # Calculate bands ensuring proper relationships
    upper_band = pd.Series(middle_band + band_width, index=prices.index)
    lower_band = pd.Series(middle_band - band_width, index=prices.index)
    
    # Handle NaN values
    upper_band.iloc[0] = np.nan
    middle_band.iloc[0] = np.nan
    lower_band.iloc[0] = np.nan
    
    return upper_band, middle_band, lower_band


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for price data.

    Args:
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with computed indicators
    """
    if len(data) == 0:
        return pd.DataFrame()
    
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    
    # Rolling statistics
    stats = calculate_rolling_stats(data['close'], window=20)
    features['volatility'] = stats['std']
    
    # Moving averages
    features['sma_20'] = calculate_sma(data['close'], 20)
    features['sma_50'] = calculate_sma(data['close'], 50)
    features['ema_20'] = calculate_ema(data['close'], 20)
    
    # RSI
    features['rsi'] = calculate_rsi(data['close'])
    
    # MACD
    macd, signal, hist = calculate_macd(data['close'])
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = hist
    
    # Bollinger Bands
    upper, _, lower = calculate_bollinger_bands(data['close'])
    features['bollinger_upper'] = upper
    features['bollinger_lower'] = lower
    
    # ATR (Average True Range)
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    features['atr'] = true_range.rolling(window=14).mean()
    
    logger.info(f"Generated {len(features.columns)} technical indicators")
    return features


def add_target_column(
    data: pd.DataFrame,
    target_column: str = 'close',
    periods: int = 1,
    pct_change: bool = True
) -> pd.DataFrame:
    """
    Add target column for prediction.

    Args:
        data: DataFrame with price data
        target_column: Column to use for target calculation
        periods: Number of periods to look ahead
        pct_change: If True, calculate percentage change, otherwise absolute change

    Returns:
        DataFrame with added target column
    """
    if pct_change:
        data['target'] = data[target_column].pct_change(periods).shift(-periods)
    else:
        data['target'] = data[target_column].diff(periods).shift(-periods)
    
    logger.info(f"Added target column (periods={periods}, pct_change={pct_change})")
    return data


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