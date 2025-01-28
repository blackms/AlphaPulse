"""
Feature engineering module for AlphaPulse.
"""
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from loguru import logger


def ensure_series(data: Union[pd.Series, np.ndarray]) -> pd.Series:
    """Convert input to pandas Series if it isn't already."""
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data)


def calculate_rolling_stats(
    data: Union[pd.Series, np.ndarray],
    window: int
) -> Dict[str, pd.Series]:
    """
    Calculate rolling statistics.

    Args:
        data: Input data
        window: Window size for rolling calculations

    Returns:
        Dict containing rolling statistics (mean, std, min, max, etc.)
    """
    series = ensure_series(data)
    
    stats = {
        'mean': series.rolling(window=window).mean(),
        'std': series.rolling(window=window).std(),
        'min': series.rolling(window=window).min(),
        'max': series.rolling(window=window).max(),
        'median': series.rolling(window=window).median(),
        'skew': series.rolling(window=window).skew(),
        'kurt': series.rolling(window=window).kurt(),
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
    series = ensure_series(prices)
    return series.rolling(window=window).mean()


def calculate_ema(prices: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Price data
        window: Window size for moving average

    Returns:
        Series of EMA values
    """
    series = ensure_series(prices)
    return series.ewm(span=window).mean()


def calculate_rsi(prices: Union[pd.Series, np.ndarray], window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Price data
        window: Window size for RSI calculation

    Returns:
        Series of RSI values
    """
    series = ensure_series(prices)
    
    # Calculate price changes
    delta = series.diff()
    
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
    series = ensure_series(prices)
    
    # Calculate EMAs
    fast_ema = series.ewm(span=fast_period).mean()
    slow_ema = series.ewm(span=slow_period).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period).mean()
    
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
    series = ensure_series(prices)
    
    # Calculate middle band (SMA)
    middle_band = series.rolling(window=window).mean()
    
    # Calculate standard deviation
    std = series.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


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
        price_data: Union[pd.Series, pd.DataFrame],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute technical indicators for price data.

        Args:
            price_data: Series of price data or DataFrame with OHLCV data
            windows: List of lookback periods for indicators

        Returns:
            DataFrame with computed indicators
        """
        if isinstance(price_data, pd.Series):
            data = pd.DataFrame({'close': price_data})
        else:
            data = price_data.copy()

        if len(data) == 0:
            return pd.DataFrame()

        if windows is None:
            windows = [12, 26, 50, 100, 200]

        feature_engineer = FeatureEngineer()
        return feature_engineer.create_features(data)

    def add_features(self, name: str, features: pd.DataFrame) -> None:
        """
        Add features to cache.

        Args:
            name: Name to save features under
            features: DataFrame of computed features
        """
        save_path = self.cache_dir / f"{name}.joblib"
        joblib.dump(features, save_path)
        logger.info(f"Saved features to {save_path}")

    def get_features(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get features from cache.

        Args:
            name: Name of features to load

        Returns:
            DataFrame of features or None if not found
        """
        load_path = self.cache_dir / f"{name}.joblib"
        if not load_path.exists():
            return None
        
        features = joblib.load(load_path)
        logger.info(f"Loaded features from {load_path}")
        return features


class FeatureEngineer:
    """Handles feature engineering and technical indicator calculation."""

    def __init__(self):
        """Initialize feature engineer."""
        logger.info("Initializing FeatureEngineer")

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical features from price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with computed features
        """
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
        
        # Volume features
        if 'volume' in data.columns:
            vol_stats = calculate_rolling_stats(data['volume'], window=20)
            features['volume_ma'] = vol_stats['mean']
            features['volume_std'] = vol_stats['std']
        
        logger.info(f"Generated {len(features.columns)} features")
        return features