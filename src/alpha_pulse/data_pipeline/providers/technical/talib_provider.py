"""
Technical analysis provider implementation using TA-Lib.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import talib
from loguru import logger
from datetime import datetime

from ...interfaces import TechnicalIndicators
from ..base import BaseDataProvider


class TALibProvider(BaseDataProvider):
    """
    Technical analysis provider implementation using TA-Lib.
    
    Features:
    - Trend indicators (SMA, EMA, MACD)
    - Momentum indicators (RSI, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV, AD)
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize TA-Lib provider.

        Args:
            max_workers: Maximum number of worker threads
        """
        super().__init__("talib", "technical", None)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _calculate_trend_indicators(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Calculate trend indicators."""
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            
            # SMA - Simple Moving Average
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
            
            # EMA - Exponential Moving Average
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            # MACD - Moving Average Convergence Divergence
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            
            # ADX - Average Directional Index
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            return {
                'sma_20': [float(x) if not np.isnan(x) else None for x in sma_20],
                'sma_50': [float(x) if not np.isnan(x) else None for x in sma_50],
                'sma_200': [float(x) if not np.isnan(x) else None for x in sma_200],
                'ema_12': [float(x) if not np.isnan(x) else None for x in ema_12],
                'ema_26': [float(x) if not np.isnan(x) else None for x in ema_26],
                'macd': [float(x) if not np.isnan(x) else None for x in macd],
                'macd_signal': [float(x) if not np.isnan(x) else None for x in macd_signal],
                'macd_hist': [float(x) if not np.isnan(x) else None for x in macd_hist],
                'adx': [float(x) if not np.isnan(x) else None for x in adx]
            }
        except Exception as e:
            logger.exception(f"Error calculating trend indicators: {str(e)}")
            raise

    def _calculate_momentum_indicators(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Calculate momentum indicators."""
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            volume = df['volume'].values.astype(float)
            
            # RSI - Relative Strength Index
            rsi = talib.RSI(close, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                high,
                low,
                close,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            
            # MFI - Money Flow Index
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # CCI - Commodity Channel Index
            cci = talib.CCI(high, low, close, timeperiod=14)
            
            return {
                'rsi': [float(x) if not np.isnan(x) else None for x in rsi],
                'stoch_k': [float(x) if not np.isnan(x) else None for x in slowk],
                'stoch_d': [float(x) if not np.isnan(x) else None for x in slowd],
                'mfi': [float(x) if not np.isnan(x) else None for x in mfi],
                'cci': [float(x) if not np.isnan(x) else None for x in cci]
            }
        except Exception as e:
            logger.exception(f"Error calculating momentum indicators: {str(e)}")
            raise

    def _calculate_volatility_indicators(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Calculate volatility indicators."""
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            
            # ATR - Average True Range
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # Standard Deviation
            stddev = talib.STDDEV(close, timeperiod=20, nbdev=2)
            
            return {
                'bb_upper': [float(x) if not np.isnan(x) else None for x in bb_upper],
                'bb_middle': [float(x) if not np.isnan(x) else None for x in bb_middle],
                'bb_lower': [float(x) if not np.isnan(x) else None for x in bb_lower],
                'atr': [float(x) if not np.isnan(x) else None for x in atr],
                'stddev': [float(x) if not np.isnan(x) else None for x in stddev]
            }
        except Exception as e:
            logger.exception(f"Error calculating volatility indicators: {str(e)}")
            raise

    def _calculate_volume_indicators(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Calculate volume indicators."""
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            volume = df['volume'].values.astype(float)
            
            # OBV - On Balance Volume
            obv = talib.OBV(close, volume)
            
            # AD - Chaikin A/D Line
            ad = talib.AD(high, low, close, volume)
            
            # ADOSC - Chaikin A/D Oscillator
            adosc = talib.ADOSC(
                high,
                low,
                close,
                volume,
                fastperiod=3,
                slowperiod=10
            )
            
            return {
                'obv': [float(x) if not np.isnan(x) else None for x in obv],
                'ad': [float(x) if not np.isnan(x) else None for x in ad],
                'adosc': [float(x) if not np.isnan(x) else None for x in adosc]
            }
        except Exception as e:
            logger.exception(f"Error calculating volume indicators: {str(e)}")
            raise

    def get_technical_indicators(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TechnicalIndicators:
        """
        Calculate technical indicators.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol

        Returns:
            TechnicalIndicators object
        """
        try:
            logger.debug(f"Calculating technical indicators for {symbol} with {len(df)} data points")
            
            # Ensure DataFrame is properly sorted
            df = df.sort_index()
            
            # Log data range
            logger.debug(f"Data range for {symbol}: {df.index[0]} to {df.index[-1]}")
            
            # Calculate all indicators
            trend = self._calculate_trend_indicators(df)
            momentum = self._calculate_momentum_indicators(df)
            volatility = self._calculate_volatility_indicators(df)
            volume = self._calculate_volume_indicators(df)
            
            # Log some debug information
            logger.debug(f"Latest RSI for {symbol}: {momentum['rsi'][-1] if momentum['rsi'] else 'N/A'}")
            logger.debug(f"Latest MACD for {symbol}: {trend['macd'][-1] if trend['macd'] else 'N/A'}")
            
            return TechnicalIndicators(
                symbol=symbol,
                timestamp=datetime.now(),
                trend=trend,
                momentum=momentum,
                volatility=volatility,
                volume=volume,
                source="talib"
            )
            
        except Exception as e:
            logger.exception(f"Error calculating indicators for {symbol}: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        try:
            self._executor.shutdown(wait=True)
        except Exception as e:
            logger.exception(f"Error shutting down executor: {str(e)}")