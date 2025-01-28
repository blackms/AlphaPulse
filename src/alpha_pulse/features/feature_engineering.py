"""
Feature engineering module for AlphaPulse.
"""
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from loguru import logger
import talib

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
    Calculate Simple Moving Average using TA-Lib.

    Args:
        prices: Price data
        window: Window size for moving average

    Returns:
        Series of SMA values
    """
    if isinstance(prices, pd.Series):
        index = prices.index
        prices = prices.values.astype(np.float64)
    else:
        prices = np.array(prices, dtype=np.float64)
        index = None
    
    sma = talib.SMA(prices, timeperiod=window)
    return pd.Series(sma, index=index)

def calculate_ema(prices: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average using TA-Lib.

    Args:
        prices: Price data
        window: Window size for moving average

    Returns:
        Series of EMA values
    """
    if isinstance(prices, pd.Series):
        index = prices.index
        prices = prices.values.astype(np.float64)
    else:
        prices = np.array(prices, dtype=np.float64)
        index = None
    
    ema = talib.EMA(prices, timeperiod=window)
    return pd.Series(ema, index=index)

def calculate_rsi(prices: Union[pd.Series, np.ndarray], window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index using TA-Lib.

    Args:
        prices: Price data
        window: Window size for RSI calculation

    Returns:
        Series of RSI values
    """
    if isinstance(prices, pd.Series):
        index = prices.index
        prices = prices.values.astype(np.float64)
    else:
        prices = np.array(prices, dtype=np.float64)
        index = None
    
    rsi = talib.RSI(prices, timeperiod=window)
    return pd.Series(rsi, index=index)

def calculate_macd(
    prices: Union[pd.Series, np.ndarray],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence using TA-Lib.

    Args:
        prices: Price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    if isinstance(prices, pd.Series):
        index = prices.index
        prices = prices.values.astype(np.float64)
    else:
        prices = np.array(prices, dtype=np.float64)
        index = None
    
    macd, signal, hist = talib.MACD(
        prices,
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period
    )
    
    return (
        pd.Series(macd, index=index),
        pd.Series(signal, index=index),
        pd.Series(hist, index=index)
    )

def calculate_bollinger_bands(
    prices: Union[pd.Series, np.ndarray],
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands using TA-Lib.

    Args:
        prices: Price data
        window: Window size for moving average
        num_std: Number of standard deviations for bands

    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    if isinstance(prices, pd.Series):
        index = prices.index
        prices = prices.values.astype(np.float64)
    else:
        prices = np.array(prices, dtype=np.float64)
        index = None
    
    upper, middle, lower = talib.BBANDS(
        prices,
        timeperiod=window,
        nbdevup=num_std,
        nbdevdn=num_std,
        matype=talib.MA_Type.SMA
    )
    
    return (
        pd.Series(upper, index=index),
        pd.Series(middle, index=index),
        pd.Series(lower, index=index)
    )

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for price data using TA-Lib.

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
    # Convert OHLC data to float64
    high = data['high'].values.astype(np.float64)
    low = data['low'].values.astype(np.float64)
    close = data['close'].values.astype(np.float64)
    
    features['atr'] = talib.ATR(
        high,
        low,
        close,
        timeperiod=14
    )
    
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