"""
Technical analysis agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import talib
from collections import defaultdict
from loguru import logger

from .interfaces import (
    BaseTradeAgent,
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)


class TechnicalAgent(BaseTradeAgent):
    """
    Implements technical analysis strategies focusing on price patterns,
    technical indicators, and chart analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize technical analysis agent."""
        super().__init__("technical_agent", config)
        self.indicator_weights = {
            'trend': self.config.get("trend_weight", 0.3),
            'momentum': self.config.get("momentum_weight", 0.2),
            'volatility': self.config.get("volatility_weight", 0.2),
            'volume': self.config.get("volume_weight", 0.15),
            'pattern': self.config.get("pattern_weight", 0.15)
        }
        self.timeframes = {
            'short': self.config.get("timeframes", {}).get("short", 14),
            'medium': self.config.get("timeframes", {}).get("medium", 50),
            'long': self.config.get("timeframes", {}).get("long", 180)
        }
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration."""
        await super().initialize(config)
        self.pattern_recognition = config.get("pattern_recognition", True)
        self.min_pattern_confidence = config.get("min_pattern_confidence", 0.6)
        self.indicator_signals = defaultdict(dict)
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate trading signals based on technical analysis.
        
        Args:
            market_data: Market data including prices and volumes
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not isinstance(market_data.prices, pd.DataFrame) or market_data.prices.empty:
            logger.warning("No price data available")
            return signals
            
        for symbol in market_data.prices.columns:
            prices = market_data.prices[symbol].dropna()
            volumes = market_data.volumes[symbol].dropna() if market_data.volumes is not None else None
            
            logger.debug(f"Analyzing {symbol} with {len(prices)} data points")
            
            if len(prices) < self.timeframes['long']:
                logger.warning(f"Insufficient data for {symbol}: {len(prices)} < {self.timeframes['long']}")
                continue
                
            try:
                # Calculate technical indicators
                trend_score = await self._analyze_trends(prices)
                logger.debug(f"{symbol} trend score: {trend_score:.2f}")
                
                momentum_score = await self._analyze_momentum(prices)
                logger.debug(f"{symbol} momentum score: {momentum_score:.2f}")
                
                volatility_score = await self._analyze_volatility(prices)
                logger.debug(f"{symbol} volatility score: {volatility_score:.2f}")
                
                volume_score = await self._analyze_volume(prices, volumes) if volumes is not None else 0
                logger.debug(f"{symbol} volume score: {volume_score:.2f}")
                
                pattern_score = await self._analyze_patterns(prices) if self.pattern_recognition else 0
                logger.debug(f"{symbol} pattern score: {pattern_score:.2f}")
                
                # Calculate weighted technical score
                technical_score = (
                    trend_score * self.indicator_weights['trend'] +
                    momentum_score * self.indicator_weights['momentum'] +
                    volatility_score * self.indicator_weights['volatility'] +
                    volume_score * self.indicator_weights['volume'] +
                    pattern_score * self.indicator_weights['pattern']
                )
                logger.debug(f"{symbol} technical score components:")
                logger.debug(f"  - Trend: {trend_score:.2f} * {self.indicator_weights['trend']:.2f} = {trend_score * self.indicator_weights['trend']:.2f}")
                logger.debug(f"  - Momentum: {momentum_score:.2f} * {self.indicator_weights['momentum']:.2f} = {momentum_score * self.indicator_weights['momentum']:.2f}")
                logger.debug(f"  - Volatility: {volatility_score:.2f} * {self.indicator_weights['volatility']:.2f} = {volatility_score * self.indicator_weights['volatility']:.2f}")
                logger.debug(f"  - Volume: {volume_score:.2f} * {self.indicator_weights['volume']:.2f} = {volume_score * self.indicator_weights['volume']:.2f}")
                logger.debug(f"  - Pattern: {pattern_score:.2f} * {self.indicator_weights['pattern']:.2f} = {pattern_score * self.indicator_weights['pattern']:.2f}")
                logger.debug(f"{symbol} final technical score: {technical_score:.2f}")
                
                # Store indicator signals
                self.indicator_signals[symbol] = {
                    'trend': trend_score,
                    'momentum': momentum_score,
                    'volatility': volatility_score,
                    'volume': volume_score,
                    'pattern': pattern_score
                }
                
                # Generate signal based on technical score
                signal = await self._generate_technical_signal(
                    symbol,
                    technical_score,
                    prices,
                    volumes
                )
                
                if signal:
                    logger.info(f"Generated signal for {symbol}: {signal.direction.value} with confidence {signal.confidence:.2f}")
                    signals.append(signal)
                else:
                    logger.debug(f"No signal generated for {symbol} (score: {technical_score:.2f})")
                    
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                continue
                
        return signals
        
    async def _analyze_trends(self, prices: pd.Series) -> float:
        """Analyze price trends using multiple timeframes."""
        scores = []
        
        # Moving averages for different timeframes
        for period in self.timeframes.values():
            if len(prices) >= period:
                ma = talib.SMA(prices.values, timeperiod=period)
                current_price = prices.iloc[-1]
                ma_current = ma[-1]
                
                if not np.isnan(ma_current):
                    # Calculate trend strength and direction
                    trend_strength = abs(current_price - ma_current) / ma_current
                    trend_direction = np.sign(current_price - ma_current)
                    scores.append(trend_strength * trend_direction)
                    
        # Moving average crossovers
        if len(prices) >= self.timeframes['long']:
            short_ma = talib.SMA(prices.values, timeperiod=self.timeframes['short'])
            long_ma = talib.SMA(prices.values, timeperiod=self.timeframes['long'])
            
            if not (np.isnan(short_ma[-1]) or np.isnan(long_ma[-1])):
                crossover_signal = np.sign(short_ma[-1] - long_ma[-1])
                scores.append(crossover_signal)
                
        return np.tanh(np.mean(scores)) if scores else 0
        
    async def _analyze_momentum(self, prices: pd.Series) -> float:
        """Analyze price momentum using multiple indicators."""
        scores = []
        
        # RSI
        rsi = talib.RSI(prices.values, timeperiod=14)
        if not np.isnan(rsi[-1]):
            rsi_score = (rsi[-1] - 50) / 50  # Normalize to [-1, 1]
            scores.append(rsi_score)
            
        # MACD
        macd, signal, _ = talib.MACD(prices.values)
        if not (np.isnan(macd[-1]) or np.isnan(signal[-1])):
            macd_score = np.sign(macd[-1] - signal[-1])
            scores.append(macd_score)
            
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            prices.values,
            prices.values,
            prices.values
        )
        if not (np.isnan(slowk[-1]) or np.isnan(slowd[-1])):
            stoch_score = (slowk[-1] - 50) / 50  # Normalize to [-1, 1]
            scores.append(stoch_score)
            
        return np.tanh(np.mean(scores)) if scores else 0
        
    async def _analyze_volatility(self, prices: pd.Series) -> float:
        """Analyze price volatility using multiple indicators."""
        scores = []
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(prices.values)
        if not (np.isnan(upper[-1]) or np.isnan(middle[-1]) or np.isnan(lower[-1])):
            bandwidth = (upper[-1] - lower[-1]) / middle[-1]
            position = (prices.iloc[-1] - lower[-1]) / (upper[-1] - lower[-1])
            scores.append(2 * position - 1)  # Normalize to [-1, 1]
            
        # ATR
        atr = talib.ATR(
            prices.values,
            prices.values,
            prices.values
        )
        if not np.isnan(atr[-1]):
            atr_score = 1 - (2 * atr[-1] / prices.iloc[-1])  # Inverse and normalize
            scores.append(atr_score)
            
        return np.tanh(np.mean(scores)) if scores else 0
        
    async def _analyze_volume(self, prices: pd.Series, volumes: pd.Series) -> float:
        """Analyze volume patterns and trends."""
        if volumes is None or len(volumes) < self.timeframes['short']:
            return 0
            
        scores = []
        
        # Volume trend
        vol_ma = talib.SMA(volumes.values, timeperiod=self.timeframes['short'])
        if not np.isnan(vol_ma[-1]):
            vol_trend = volumes.iloc[-1] / vol_ma[-1] - 1
            scores.append(np.tanh(vol_trend))
            
        # Price-volume correlation
        if len(prices) == len(volumes):
            correlation = stats.pearsonr(
                prices.pct_change().fillna(0)[-20:],
                volumes.pct_change().fillna(0)[-20:]
            )[0]
            scores.append(correlation)
            
        # On-balance volume
        obv = talib.OBV(prices.values, volumes.values)
        if len(obv) > 1:
            obv_trend = np.sign(obv[-1] - obv[-2])
            scores.append(obv_trend)
            
        return np.tanh(np.mean(scores)) if scores else 0
        
    async def _analyze_patterns(self, prices: pd.Series) -> float:
        """Analyze chart patterns."""
        scores = []
        
        # Candlestick patterns
        pattern_functions = [
            (talib.CDLENGULFING, 1.0),
            (talib.CDLHARAMI, 0.8),
            (talib.CDLDOJI, 0.6),
            (talib.CDLHAMMER, 0.8),
            (talib.CDLSHOOTINGSTAR, 0.8)
        ]
        
        for pattern_func, weight in pattern_functions:
            pattern = pattern_func(
                prices.values,
                prices.values,
                prices.values,
                prices.values
            )
            if pattern[-1] != 0:
                scores.append(np.sign(pattern[-1]) * weight)
                
        # Support and resistance levels
        levels = await self._find_support_resistance(prices)
        if levels:
            current_price = prices.iloc[-1]
            nearest_level = min(levels, key=lambda x: abs(x - current_price))
            level_score = np.sign(current_price - nearest_level)
            scores.append(level_score)
            
        return np.tanh(np.mean(scores)) if scores else 0
        
    async def _find_support_resistance(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> List[float]:
        """Find support and resistance levels."""
        levels = []
        
        # Find local maxima and minima
        for i in range(window, len(prices) - window):
            if len(prices.iloc[i-window:i+window]) < 2 * window:
                continue
                
            price_window = prices.iloc[i-window:i+window]
            if prices.iloc[i] == max(price_window):
                levels.append(prices.iloc[i])  # Resistance
            elif prices.iloc[i] == min(price_window):
                levels.append(prices.iloc[i])  # Support
                
        return levels
        
    async def _generate_technical_signal(
        self,
        symbol: str,
        technical_score: float,
        prices: pd.Series,
        volumes: Optional[pd.Series]
    ) -> Optional[TradeSignal]:
        """Generate trading signal based on technical analysis."""
        logger.debug(f"Generating signal for {symbol} with technical score: {technical_score:.2f}")
        
        logger.debug(f"Evaluating signal for {symbol} with technical score: {technical_score:.2f}")
        if abs(technical_score) <= 0.15:  # Lower minimum conviction threshold
            logger.debug(f"Technical score {technical_score:.2f} below minimum threshold 0.15 (need |score| > 0.15)")
            return None
        logger.debug(f"Technical score {technical_score:.2f} passed minimum threshold")
            
        # Determine signal direction
        direction = SignalDirection.BUY if technical_score > 0 else SignalDirection.SELL
        logger.debug(f"Signal direction: {direction.value}")
        
        # Calculate confidence based on indicator agreement
        indicator_scores = list(self.indicator_signals[symbol].values())
        confidence = abs(technical_score) * (1 - np.std(indicator_scores))
        logger.debug(f"Confidence score: {confidence:.2f} (std: {np.std(indicator_scores):.2f})")
        
        # Calculate target price and stop loss
        current_price = prices.iloc[-1]
        atr = talib.ATR(prices.values, prices.values, prices.values)[-1]
        logger.debug(f"Current price: {current_price:.2f}, ATR: {atr:.2f}")
        
        if direction == SignalDirection.BUY:
            target_price = current_price * (1 + abs(technical_score))
            stop_loss = current_price - (2 * atr)
        else:
            target_price = current_price * (1 - abs(technical_score))
            stop_loss = current_price + (2 * atr)
            
        return TradeSignal(
            agent_id=self.agent_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(),
            target_price=target_price,
            stop_loss=stop_loss,
            metadata={
                "strategy": "technical",
                "technical_score": technical_score,
                "indicators": self.indicator_signals[symbol],
                "support_resistance": await self._find_support_resistance(prices),
                "volume_profile": await self._get_volume_profile(volumes) if volumes is not None else None
            }
        )
        
    async def _get_volume_profile(self, volumes: pd.Series) -> Dict:
        """Calculate volume profile metrics."""
        if volumes is None or len(volumes) < self.timeframes['short']:
            return {}
            
        return {
            "average_volume": float(volumes[-self.timeframes['short']:].mean()),
            "volume_trend": float(volumes.pct_change()[-5:].mean()),
            "volume_volatility": float(volumes.pct_change()[-20:].std())
        }