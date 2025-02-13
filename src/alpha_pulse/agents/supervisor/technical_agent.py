"""
Self-supervised technical analysis agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger
import talib
from scipy import stats

from ..interfaces import MarketData, TradeSignal, SignalDirection
from .base import BaseSelfSupervisedAgent


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division handling zeros and NaN."""
    try:
        if b == 0 or np.isnan(b) or np.isnan(a):
            return default
        result = a / b
        return result if not np.isinf(result) else default
    except Exception:
        return default


class SelfSupervisedTechnicalAgent(BaseSelfSupervisedAgent):
    """
    Self-supervised technical analysis agent that can optimize its own parameters
    and adapt to changing market conditions.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize self-supervised technical agent."""
        super().__init__(agent_id, config)
        
        # Technical analysis parameters
        self.timeframes = {
            'short': self.config.get("timeframes", {}).get("short", 14),
            'medium': self.config.get("timeframes", {}).get("medium", 50),
            'long': self.config.get("timeframes", {}).get("long", 180)
        }
        
        # Technical indicators configuration
        self.indicator_weights = {
            'trend': 0.3,
            'momentum': 0.2,
            'volatility': 0.2,
            'volume': 0.15,
            'pattern': 0.15
        }
        
        # Performance tracking
        self._indicator_performance = {
            'trend': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'pattern': []
        }
        self._market_regime = "unknown"
        self._optimization_history = []
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate trading signals with self-supervision capabilities."""
        try:
            # Update market regime
            self._market_regime = await self._detect_market_regime(market_data)
            logger.info(f"Current market regime: {self._market_regime}")
            
            signals = []
            
            for symbol in market_data.prices.columns:
                prices = market_data.prices[symbol].dropna()
                volumes = market_data.volumes[symbol].dropna() if market_data.volumes is not None else None
                
                if len(prices) < self.timeframes['long']:
                    continue
                    
                try:
                    # Convert to numpy arrays for TA-Lib
                    prices_array = np.array(prices.values, dtype=np.float64)
                    volumes_array = np.array(volumes.values, dtype=np.float64) if volumes is not None else None
                    
                    # Handle NaN values
                    prices_array = np.nan_to_num(prices_array, nan=prices_array[~np.isnan(prices_array)].mean())
                    if volumes_array is not None:
                        volumes_array = np.nan_to_num(volumes_array, nan=volumes_array[~np.isnan(volumes_array)].mean())
                    
                    # Calculate technical scores
                    trend_score = await self._analyze_trend(prices_array)
                    momentum_score = await self._analyze_momentum(prices_array)
                    volatility_score = await self._analyze_volatility(prices_array)
                    volume_score = await self._analyze_volume(prices_array, volumes_array) if volumes_array is not None else 0
                    
                    # Calculate weighted technical score
                    technical_score = (
                        trend_score * self.indicator_weights['trend'] +
                        momentum_score * self.indicator_weights['momentum'] +
                        volatility_score * self.indicator_weights['volatility'] +
                        volume_score * self.indicator_weights['volume']
                    )
                    
                    # Generate signal if score is significant
                    if abs(technical_score) > 0.2:  # Threshold for signal generation
                        direction = (
                            SignalDirection.BUY if technical_score > 0
                            else SignalDirection.SELL
                        )
                        
                        # Calculate confidence based on score strength and regime
                        confidence = min(abs(technical_score), 1.0)
                        
                        # Calculate target price and stop loss
                        current_price = float(prices.iloc[-1])
                        atr = talib.ATR(
                            prices_array,
                            prices_array,
                            prices_array,
                            timeperiod=14
                        )[-1]
                        
                        if direction == SignalDirection.BUY:
                            target_price = current_price * (1 + abs(technical_score))
                            stop_loss = current_price - (2 * atr if not np.isnan(atr) else current_price * 0.02)
                        else:
                            target_price = current_price * (1 - abs(technical_score))
                            stop_loss = current_price + (2 * atr if not np.isnan(atr) else current_price * 0.02)
                            
                        signal = TradeSignal(
                            agent_id=self.agent_id,
                            symbol=symbol,
                            direction=direction,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            target_price=target_price,
                            stop_loss=stop_loss,
                            metadata={
                                "strategy": "technical",
                                "market_regime": self._market_regime,
                                "scores": {
                                    "trend": float(trend_score),
                                    "momentum": float(momentum_score),
                                    "volatility": float(volatility_score),
                                    "volume": float(volume_score),
                                    "total": float(technical_score)
                                }
                            }
                        )
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
                    
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals in technical agent: {str(e)}")
            raise
            
    async def _analyze_trend(self, prices: np.ndarray) -> float:
        """Analyze price trends."""
        try:
            scores = []
            
            # Moving averages
            for period in self.timeframes.values():
                if len(prices) >= period:
                    ma = talib.SMA(prices, timeperiod=period)
                    if not np.isnan(ma[-1]):
                        trend = safe_divide(prices[-1] - ma[-1], ma[-1])
                        scores.append(np.tanh(trend * 5))  # Scale and normalize
                        
            # Moving average crossovers
            if len(prices) >= self.timeframes['long']:
                short_ma = talib.SMA(prices, timeperiod=self.timeframes['short'])
                long_ma = talib.SMA(prices, timeperiod=self.timeframes['long'])
                
                if not (np.isnan(short_ma[-1]) or np.isnan(long_ma[-1])):
                    crossover = safe_divide(short_ma[-1] - long_ma[-1], long_ma[-1])
                    scores.append(np.tanh(crossover * 10))
                    
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return 0.0
            
    async def _analyze_momentum(self, prices: np.ndarray) -> float:
        """Analyze price momentum."""
        try:
            scores = []
            
            # RSI
            rsi = talib.RSI(prices, timeperiod=14)
            if not np.isnan(rsi[-1]):
                rsi_score = safe_divide(rsi[-1] - 50, 50)  # Normalize to [-1, 1]
                scores.append(rsi_score)
                
            # MACD
            macd, signal, _ = talib.MACD(prices)
            if not (np.isnan(macd[-1]) or np.isnan(signal[-1])):
                macd_score = np.tanh(safe_divide(macd[-1] - signal[-1], abs(signal[-1]) + 1e-8) * 5)
                scores.append(macd_score)
                
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {str(e)}")
            return 0.0
            
    async def _analyze_volatility(self, prices: np.ndarray) -> float:
        """Analyze price volatility."""
        try:
            scores = []
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(prices)
            if not (np.isnan(upper[-1]) or np.isnan(middle[-1]) or np.isnan(lower[-1])):
                bandwidth = safe_divide(upper[-1] - lower[-1], middle[-1])
                position = safe_divide(prices[-1] - lower[-1], upper[-1] - lower[-1])
                scores.append(2 * position - 1)  # Normalize to [-1, 1]
                
            # ATR
            atr = talib.ATR(prices, prices, prices, timeperiod=14)
            if not np.isnan(atr[-1]):
                atr_score = 1 - safe_divide(2 * atr[-1], prices[-1])
                scores.append(atr_score)
                
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            return 0.0
            
    async def _analyze_volume(self, prices: np.ndarray, volumes: Optional[np.ndarray]) -> float:
        """Analyze volume patterns."""
        try:
            if volumes is None or len(volumes) < self.timeframes['short']:
                return 0.0
                
            scores = []
            
            # Volume trend
            vol_ma = talib.SMA(volumes, timeperiod=self.timeframes['short'])
            if not np.isnan(vol_ma[-1]):
                vol_trend = safe_divide(volumes[-1], vol_ma[-1]) - 1
                scores.append(np.tanh(vol_trend))
                
            # Price-volume correlation
            if len(prices) == len(volumes):
                price_changes = np.diff(prices)[-20:]
                volume_changes = np.diff(volumes)[-20:]
                if len(price_changes) > 0 and len(volume_changes) > 0:
                    correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                    if not np.isnan(correlation):
                        scores.append(correlation)
                
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return 0.0
            
    async def _detect_market_regime(self, market_data: MarketData) -> str:
        """Detect the current market regime."""
        try:
            regimes = []
            
            for symbol in market_data.prices.columns:
                prices = market_data.prices[symbol].dropna()
                if len(prices) < self.timeframes['long']:
                    continue
                    
                # Convert to numpy array and handle NaN values
                prices_array = np.array(prices.values, dtype=np.float64)
                prices_array = np.nan_to_num(prices_array, nan=prices_array[~np.isnan(prices_array)].mean())
                
                # Calculate trend strength
                long_ma = talib.SMA(prices_array, timeperiod=self.timeframes['long'])
                if np.isnan(long_ma[-1]):
                    continue
                    
                trend = safe_divide(prices_array[-1] - long_ma[-1], long_ma[-1])
                
                # Calculate volatility
                atr = talib.ATR(
                    prices_array,
                    prices_array,
                    prices_array,
                    timeperiod=14
                )
                if np.isnan(atr[-1]):
                    continue
                    
                vol_level = safe_divide(atr[-1], prices_array[-1])
                
                # Determine regime
                if abs(trend) > 0.1:  # Strong trend
                    if trend > 0:
                        regimes.append("uptrend")
                    else:
                        regimes.append("downtrend")
                elif vol_level > 0.02:  # High volatility
                    regimes.append("volatile")
                else:
                    regimes.append("ranging")
                    
            # Return most common regime
            if regimes:
                return max(set(regimes), key=regimes.count)
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"