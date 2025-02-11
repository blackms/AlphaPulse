"""
Technical analysis provider implementation using TA-Lib.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from ...interfaces import TechnicalIndicators
from ..base import BaseDataProvider, CacheMixin


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    name: str
    function: callable
    parameters: Dict[str, Any]
    category: str


class TALibProvider(BaseDataProvider, CacheMixin):
    """
    Technical analysis provider implementation using TA-Lib.
    
    Features:
    - Comprehensive technical indicator calculation
    - Parallel processing for performance
    - Indicator customization
    - Result caching
    """

    def __init__(
        self,
        cache_ttl: int = 300,  # 5 minutes cache for technical indicators
        max_workers: int = 4
    ):
        """
        Initialize TA-Lib provider.

        Args:
            cache_ttl: Cache time-to-live in seconds
            max_workers: Maximum number of worker threads
        """
        super().__init__("talib", "technical")
        CacheMixin.__init__(self, cache_ttl)
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._indicators = self._initialize_indicators()

    def _initialize_indicators(self) -> Dict[str, IndicatorConfig]:
        """Initialize technical indicators configuration."""
        return {
            # Trend Indicators
            "sma": IndicatorConfig(
                name="Simple Moving Average",
                function=talib.SMA,
                parameters={"timeperiod": 20},
                category="trend"
            ),
            "ema": IndicatorConfig(
                name="Exponential Moving Average",
                function=talib.EMA,
                parameters={"timeperiod": 20},
                category="trend"
            ),
            "macd": IndicatorConfig(
                name="MACD",
                function=talib.MACD,
                parameters={
                    "fastperiod": 12,
                    "slowperiod": 26,
                    "signalperiod": 9
                },
                category="trend"
            ),
            
            # Momentum Indicators
            "rsi": IndicatorConfig(
                name="Relative Strength Index",
                function=talib.RSI,
                parameters={"timeperiod": 14},
                category="momentum"
            ),
            "stoch": IndicatorConfig(
                name="Stochastic",
                function=talib.STOCH,
                parameters={
                    "fastk_period": 14,
                    "slowk_period": 3,
                    "slowd_period": 3
                },
                category="momentum"
            ),
            "adx": IndicatorConfig(
                name="Average Directional Index",
                function=talib.ADX,
                parameters={"timeperiod": 14},
                category="momentum"
            ),
            
            # Volatility Indicators
            "bbands": IndicatorConfig(
                name="Bollinger Bands",
                function=talib.BBANDS,
                parameters={
                    "timeperiod": 20,
                    "nbdevup": 2,
                    "nbdevdn": 2
                },
                category="volatility"
            ),
            "atr": IndicatorConfig(
                name="Average True Range",
                function=talib.ATR,
                parameters={"timeperiod": 14},
                category="volatility"
            ),
            
            # Volume Indicators
            "obv": IndicatorConfig(
                name="On Balance Volume",
                function=talib.OBV,
                parameters={},
                category="volume"
            ),
            "ad": IndicatorConfig(
                name="Chaikin A/D Line",
                function=talib.AD,
                parameters={},
                category="volume"
            ),
            "adosc": IndicatorConfig(
                name="Chaikin A/D Oscillator",
                function=talib.ADOSC,
                parameters={
                    "fastperiod": 3,
                    "slowperiod": 10
                },
                category="volume"
            )
        }

    def _prepare_data(
        self,
        data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare OHLCV data for TA-Lib."""
        return (
            data['open'].values,
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values
        )

    def _calculate_single_indicator(
        self,
        config: IndicatorConfig,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate a single technical indicator."""
        try:
            if config.function.__name__ in ['STOCH', 'STOCHF']:
                result = config.function(
                    highs, lows, closes,
                    **config.parameters
                )
            elif config.function.__name__ in ['BBANDS']:
                result = config.function(closes, **config.parameters)
            elif config.function.__name__ in ['AD', 'ADOSC']:
                result = config.function(
                    highs, lows, closes, volumes,
                    **config.parameters
                )
            elif config.function.__name__ == 'OBV':
                result = config.function(closes, volumes)
            elif config.function.__name__ == 'MACD':
                result = config.function(closes, **config.parameters)
            else:
                result = config.function(closes, **config.parameters)

            if isinstance(result, tuple):
                return {f"{config.name}_{i}": r for i, r in enumerate(result)}
            return {config.name: result}
            
        except Exception as e:
            logger.error(
                f"Error calculating {config.name}: {str(e)}",
                exc_info=True
            )
            return {}

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators for the given data.

        Args:
            data: OHLCV DataFrame
            indicators: List of indicator names to calculate (None for all)

        Returns:
            Dictionary of calculated indicators
        """
        if data.empty:
            return {}

        cache_key = f"indicators_{data.index[-1]}_{','.join(indicators or [])}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Prepare data
        opens, highs, lows, closes, volumes = self._prepare_data(data)

        # Select indicators to calculate
        if indicators:
            selected_indicators = {
                name: config for name, config in self._indicators.items()
                if name in indicators
            }
        else:
            selected_indicators = self._indicators

        # Calculate indicators in parallel
        futures = []
        for config in selected_indicators.values():
            future = self._executor.submit(
                self._calculate_single_indicator,
                config, opens, highs, lows, closes, volumes
            )
            futures.append((config.category, future))

        # Collect results
        results: Dict[str, Dict[str, np.ndarray]] = {
            "trend": {},
            "momentum": {},
            "volatility": {},
            "volume": {}
        }

        for category, future in futures:
            try:
                result = future.result()
                results[category].update(result)
            except Exception as e:
                logger.error(f"Error in indicator calculation: {str(e)}")

        # Convert results to pandas Series
        processed_results = {}
        for category, indicators in results.items():
            for name, values in indicators.items():
                if values is not None and len(values) > 0:
                    processed_results[f"{category}_{name}"] = pd.Series(
                        values,
                        index=data.index
                    )

        # Cache results
        self._store_in_cache(cache_key, processed_results)
        return processed_results

    def get_technical_indicators(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> TechnicalIndicators:
        """
        Get technical indicators for a symbol.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            TechnicalIndicators object
        """
        indicators = self.calculate_indicators(data)
        
        # Group indicators by category
        trend_indicators = {
            k.replace('trend_', ''): v.iloc[-1]
            for k, v in indicators.items()
            if k.startswith('trend_')
        }
        
        momentum_indicators = {
            k.replace('momentum_', ''): v.iloc[-1]
            for k, v in indicators.items()
            if k.startswith('momentum_')
        }
        
        volatility_indicators = {
            k.replace('volatility_', ''): v.iloc[-1]
            for k, v in indicators.items()
            if k.startswith('volatility_')
        }
        
        volume_indicators = {
            k.replace('volume_', ''): v.iloc[-1]
            for k, v in indicators.items()
            if k.startswith('volume_')
        }

        return TechnicalIndicators(
            symbol=symbol,
            timestamp=pd.Timestamp.now(),
            trend_indicators=trend_indicators,
            momentum_indicators=momentum_indicators,
            volatility_indicators=volatility_indicators,
            volume_indicators=volume_indicators,
            metadata={
                "lookback_period": len(data),
                "last_price": data['close'].iloc[-1],
                "last_volume": data['volume'].iloc[-1]
            }
        )

    def __del__(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)