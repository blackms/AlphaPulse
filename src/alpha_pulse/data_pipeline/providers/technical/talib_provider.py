"""
Technical analysis provider implementation using TA-Lib.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import talib
import logging

from ...interfaces import TechnicalIndicators
from ..base import BaseDataProvider

logger = logging.getLogger(__name__)


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

    async def _execute_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute request (not used in this provider)."""
        pass  # Not needed for TA-Lib

    async def _process_response(self, response: Any) -> Any:
        """Process response (not used in this provider)."""
        pass  # Not needed for TA-Lib

    def _calculate_trend_indicators(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, np.ndarray]:
        """Calculate trend indicators."""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
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
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'adx': adx
            }
        except Exception as e:
            logger.error(f"Error calculating trend indicators for {symbol}: {str(e)}")
            raise

    def _calculate_momentum_indicators(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, np.ndarray]:
        """Calculate momentum indicators."""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
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
            volume = data['volume'].values
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # CCI - Commodity Channel Index
            cci = talib.CCI(high, low, close, timeperiod=14)
            
            return {
                'rsi': rsi,
                'stoch_k': slowk,
                'stoch_d': slowd,
                'mfi': mfi,
                'cci': cci
            }
        except Exception as e:
            logger.error(f"Error calculating momentum indicators for {symbol}: {str(e)}")
            raise

    def _calculate_volatility_indicators(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, np.ndarray]:
        """Calculate volatility indicators."""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
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
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'atr': atr,
                'stddev': stddev
            }
        except Exception as e:
            logger.error(f"Error calculating volatility indicators for {symbol}: {str(e)}")
            raise

    def _calculate_volume_indicators(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, np.ndarray]:
        """Calculate volume indicators."""
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
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
                'obv': obv,
                'ad': ad,
                'adosc': adosc
            }
        except Exception as e:
            logger.error(f"Error calculating volume indicators for {symbol}: {str(e)}")
            raise

    def get_technical_indicators(
        self,
        data: Dict[str, List[Any]],
        symbol: Optional[str] = None
    ) -> Dict[str, TechnicalIndicators]:
        """
        Calculate technical indicators for market data.

        Args:
            data: Dictionary mapping symbols to their market data
            symbol: Optional specific symbol to calculate for

        Returns:
            Dictionary mapping symbols to their technical indicators
        """
        results = {}
        
        # Process all symbols if none specified
        symbols = [symbol] if symbol else list(data.keys())
        
        for sym in symbols:
            if sym not in data:
                logger.warning(f"No data available for {sym}")
                continue
                
            try:
                # Convert market data to DataFrame
                df = pd.DataFrame([
                    {
                        'open': d.open,
                        'high': d.high,
                        'low': d.low,
                        'close': d.close,
                        'volume': d.volume
                    }
                    for d in data[sym]
                ])
                
                if df.empty:
                    logger.warning(f"Empty data for {sym}")
                    continue
                
                # Calculate all indicators
                trend = self._calculate_trend_indicators(df, sym)
                momentum = self._calculate_momentum_indicators(df, sym)
                volatility = self._calculate_volatility_indicators(df, sym)
                volume = self._calculate_volume_indicators(df, sym)
                
                # Create TechnicalIndicators object
                results[sym] = TechnicalIndicators(
                    symbol=sym,
                    timestamp=data[sym][-1].timestamp,
                    trend=trend,
                    momentum=momentum,
                    volatility=volatility,
                    volume=volume
                )
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {sym}: {str(e)}")
                continue
        
        return results

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        self._executor.shutdown(wait=True)