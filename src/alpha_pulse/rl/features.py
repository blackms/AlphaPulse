"""
Feature engineering module for RL trading.

This module provides advanced technical analysis and feature engineering
capabilities for the RL trading environment.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import talib.abstract as ta
from scipy.ndimage import gaussian_filter1d
import pywt
from loguru import logger


class FeatureEngineer:
    """Advanced feature engineering for RL trading."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize feature engineer.
        
        Args:
            window_size: Size of the rolling window for DWT
        """
        self.window_size = window_size
        
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated features
        """
        df = data.copy()
        
        # Basic features
        self._add_basic_features(df)
        
        # Technical indicators
        self._add_trend_indicators(df)
        self._add_momentum_indicators(df)
        self._add_volatility_indicators(df)
        self._add_volume_indicators(df)
        
        # Advanced features
        self._add_wavelet_features(df)
        self._add_market_cipher_features(df)
        
        # Clean up and normalize
        df = self._cleanup_features(df)
        
        return df
        
    def _add_basic_features(self, df: pd.DataFrame) -> None:
        """Add basic price and volume features."""
        # Price changes
        # Calculate returns
        df['pct_change'] = df['close'].pct_change()
        
        # Convert percentage change to numpy array and handle NaN/Inf values
        pct_change_array = df['pct_change'].fillna(0).replace([np.inf, -np.inf], 0).astype(float).values
        df['log_return'] = np.log1p(pct_change_array)
        
        # Price levels
        df['price_level'] = (df['close'] - df['close'].rolling(50).mean()) / \
            df['close'].rolling(50).std()
            
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_level'] = df['volume'] / df['volume_ma']
        
        # OHLC relationships
        df['hl_diff'] = (df['high'] - df['low']) / df['close']
        df['co_diff'] = (df['close'] - df['open']) / df['open']
        
    def _add_trend_indicators(self, df: pd.DataFrame) -> None:
        """Add trend-following indicators."""
        # Moving averages
        for period in [10, 20, 50, 100]:
            df[f'sma_{period}'] = ta.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = ta.EMA(df['close'], timeperiod=period)
            
        # MACD (returns macd, signal, hist)
        macd_line, signal_line, hist_line = ta.MACD(df['close'])
        df['macd'] = macd_line
        df['macdsignal'] = signal_line
        df['macdhist'] = hist_line
        
        # ADX
        df['adx'] = ta.ADX(df[['high', 'low', 'close']], timeperiod=14)
        
        # Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(9).max() + 
                           df['low'].rolling(9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(26).max() + 
                          df['low'].rolling(26).min()) / 2
                          
    def _add_momentum_indicators(self, df: pd.DataFrame) -> None:
        """Add momentum indicators."""
        # RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        
        # Stochastic
        stoch = ta.STOCH(df[['high', 'low', 'close']])
        df['stoch_k'] = stoch['slowk']
        df['stoch_d'] = stoch['slowd']
        
        # CCI
        df['cci'] = ta.CCI(df[['high', 'low', 'close']], timeperiod=20)
        
        # ROC
        df['roc'] = ta.ROC(df['close'], timeperiod=10)
        
        # Williams %R
        df['willr'] = ta.WILLR(df[['high', 'low', 'close']], timeperiod=14)
        
    def _add_volatility_indicators(self, df: pd.DataFrame) -> None:
        """Add volatility indicators."""
        # ATR
        df['atr'] = ta.ATR(df[['high', 'low', 'close']], timeperiod=14)
        df['natr'] = df['atr'] / df['close']
        
        # Bollinger Bands
        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # Historical Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
    def _add_volume_indicators(self, df: pd.DataFrame) -> None:
        """Add volume-based indicators."""
        # OBV
        df['obv'] = ta.OBV(df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['cmf'] = self._chaikin_money_flow(df)
        
        # Volume Force Index
        df['vfi'] = (df['close'] - df['close'].shift(1)) * df['volume']
        
        # Ease of Movement
        df['eom'] = (df['high'] - df['low']) / (2 * df['volume'])
        df['eom_sma'] = df['eom'].rolling(14).mean()
        
    def _add_wavelet_features(self, df: pd.DataFrame) -> None:
        """Add wavelet transform features."""
        try:
            # Apply DWT to price series
            close_values = df['close'].values
            wavelet = 'db8'
            level = 3
            
            # Decompose signal
            coeffs = pywt.wavedec(close_values, wavelet, level=level)
            
            # Reconstruct components
            for i in range(level):
                coeff_list = [np.zeros_like(c) for c in coeffs]
                coeff_list[i] = coeffs[i]
                reconstructed = pywt.waverec(coeff_list, wavelet)
                
                # Trim to match original length
                if len(reconstructed) > len(close_values):
                    reconstructed = reconstructed[:len(close_values)]
                    
                df[f'wavelet_level_{i+1}'] = reconstructed
                
        except Exception as e:
            logger.warning(f"Failed to calculate wavelet features: {str(e)}")
            
    def _add_market_cipher_features(self, df: pd.DataFrame) -> None:
        """Add Market Cipher inspired features."""
        # Wave Trend
        ap = (df['high'] + df['low'] + df['close']) / 3
        esa = ta.EMA(ap, timeperiod=10)
        d = ta.EMA((ap - esa).abs(), timeperiod=10)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.EMA(ci, timeperiod=21)
        
        df['wt1'] = tci
        df['wt2'] = ta.SMA(df['wt1'], timeperiod=4)
        
        # Additional indicators
        df['mfi'] = ta.MFI(df[['high', 'low', 'close', 'volume']], timeperiod=14)
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['mom'] = ta.MOM(df['close'], timeperiod=14)
        
    def _chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
            (df['high'] - df['low'])
        mfv = mfm * df['volume']
        return mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
        
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up and normalize features."""
        # Remove features with too many NaN values
        nan_threshold = 0.1
        nan_ratios = df.isna().sum() / len(df)
        valid_columns = nan_ratios[nan_ratios < nan_threshold].index
        df = df[valid_columns]
        
        # Forward fill remaining NaN values
        df = df.fillna(method='ffill')
        
        # Replace any remaining NaN with 0
        df = df.fillna(0)
        
        # Remove inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df