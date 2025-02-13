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


class SelfSupervisedTechnicalAgent(BaseSelfSupervisedAgent):
    """
    Self-supervised technical analysis agent that can optimize its own parameters
    and adapt to changing market conditions.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize self-supervised technical agent."""
        super().__init__(agent_id, config)
        
        # Technical analysis parameters
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
        
        # Self-supervision parameters
        self._indicator_performance = {
            'trend': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'pattern': []
        }
        self._optimization_history = []
        self._market_regime = "unknown"
        
    async def optimize(self) -> None:
        """
        Optimize technical analysis parameters based on performance metrics.
        This includes:
        1. Adjusting indicator weights based on their predictive power
        2. Optimizing timeframes based on market regime
        3. Tuning threshold parameters
        """
        await super().optimize()
        
        try:
            # Calculate indicator performance scores
            performance_scores = {}
            for indicator, history in self._indicator_performance.items():
                if history:
                    # Calculate predictive accuracy
                    accuracy = np.mean([1 if h['predicted'] == h['actual'] else 0 for h in history[-50:]])
                    # Calculate profit factor
                    wins = sum(h['profit'] for h in history[-50:] if h['profit'] > 0)
                    losses = abs(sum(h['profit'] for h in history[-50:] if h['profit'] < 0))
                    profit_factor = wins / losses if losses > 0 else 1
                    
                    performance_scores[indicator] = accuracy * profit_factor
                    
            if performance_scores:
                # Normalize scores
                total_score = sum(performance_scores.values())
                if total_score > 0:
                    new_weights = {
                        k: v / total_score
                        for k, v in performance_scores.items()
                    }
                    
                    # Apply smoothing to weight updates
                    smoothing = 0.7  # 70% old weights, 30% new weights
                    self.indicator_weights = {
                        k: smoothing * self.indicator_weights[k] + (1 - smoothing) * new_weights[k]
                        for k in self.indicator_weights
                    }
                    
                    logger.info(f"Updated indicator weights: {self.indicator_weights}")
                    
            # Optimize timeframes based on market regime
            if self._market_regime == "trending":
                # Extend timeframes for trend following
                self.timeframes['short'] = max(14, int(self.timeframes['short'] * 1.2))
                self.timeframes['medium'] = max(50, int(self.timeframes['medium'] * 1.1))
            elif self._market_regime == "ranging":
                # Shorten timeframes for range trading
                self.timeframes['short'] = max(10, int(self.timeframes['short'] * 0.9))
                self.timeframes['medium'] = max(30, int(self.timeframes['medium'] * 0.9))
                
            logger.info(f"Updated timeframes: {self.timeframes}")
            
            # Store optimization result
            self._optimization_history.append({
                'timestamp': datetime.now(),
                'weights': self.indicator_weights.copy(),
                'timeframes': self.timeframes.copy(),
                'market_regime': self._market_regime,
                'performance_scores': performance_scores
            })
            
        except Exception as e:
            logger.error(f"Error in technical agent optimization: {str(e)}")
            raise
            
    async def self_evaluate(self) -> Dict[str, float]:
        """
        Evaluate agent's performance and market adaptation.
        Returns performance metrics specific to technical analysis.
        """
        metrics = await super().self_evaluate()
        
        try:
            # Calculate indicator-specific metrics
            for indicator, history in self._indicator_performance.items():
                if history:
                    recent_history = history[-50:]  # Look at last 50 signals
                    metrics[f"{indicator}_accuracy"] = np.mean([
                        1 if h['predicted'] == h['actual'] else 0
                        for h in recent_history
                    ])
                    
            # Calculate market regime metrics
            if self._optimization_history:
                recent_regimes = [
                    h['market_regime']
                    for h in self._optimization_history[-10:]
                ]
                metrics['regime_stability'] = len(set(recent_regimes)) / len(recent_regimes)
                
            # Calculate adaptation metrics
            if len(self._optimization_history) >= 2:
                last_opt = self._optimization_history[-1]
                prev_opt = self._optimization_history[-2]
                
                # Calculate weight change magnitude
                weight_change = np.mean([
                    abs(last_opt['weights'][k] - prev_opt['weights'][k])
                    for k in last_opt['weights']
                ])
                metrics['adaptation_rate'] = weight_change
                
            logger.debug(f"Technical agent self-evaluation metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error in technical agent self-evaluation: {str(e)}")
            raise
            
        return metrics
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate trading signals with self-supervision capabilities."""
        try:
            # Update market regime
            self._market_regime = await self._detect_market_regime(market_data)
            logger.info(f"Current market regime: {self._market_regime}")
            
            # Generate signals using parent implementation
            signals = await super().generate_signals(market_data)
            
            # Track indicator performance
            for signal in signals:
                current_price = market_data.prices[signal.symbol].iloc[-1]
                
                for indicator, score in signal.metadata.get("indicators", {}).items():
                    predicted_direction = np.sign(score)
                    self._indicator_performance[indicator].append({
                        'timestamp': signal.timestamp,
                        'predicted': predicted_direction,
                        'actual': None,  # Will be updated in next iteration
                        'profit': None
                    })
                    
                # Update previous predictions with actual outcomes
                for indicator, history in self._indicator_performance.items():
                    for record in history:
                        if record['actual'] is None and record['timestamp'] < signal.timestamp:
                            price_change = current_price - market_data.prices[signal.symbol].iloc[-2]
                            record['actual'] = np.sign(price_change)
                            record['profit'] = price_change if record['predicted'] == np.sign(price_change) else -price_change
                            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals in technical agent: {str(e)}")
            raise
            
    async def _detect_market_regime(self, market_data: MarketData) -> str:
        """
        Detect the current market regime (trending, ranging, volatile).
        This helps adapt the strategy to different market conditions.
        """
        try:
            regimes = []
            
            for symbol in market_data.prices.columns:
                prices = market_data.prices[symbol].dropna()
                if len(prices) < self.timeframes['long']:
                    continue
                    
                # Calculate trend strength
                atr = talib.ATR(
                    prices.values,
                    prices.values,
                    prices.values,
                    timeperiod=14
                )
                
                # Calculate directional movement
                adx = talib.ADX(
                    prices.values,
                    prices.values,
                    prices.values,
                    timeperiod=14
                )
                
                if not (np.isnan(atr[-1]) or np.isnan(adx[-1])):
                    # High ADX indicates trending market
                    if adx[-1] > 25:
                        regimes.append("trending")
                    # Low ADX with low ATR indicates ranging market
                    elif atr[-1] < np.mean(atr) * 0.8:
                        regimes.append("ranging")
                    # High ATR indicates volatile market
                    elif atr[-1] > np.mean(atr) * 1.2:
                        regimes.append("volatile")
                    else:
                        regimes.append("unknown")
                        
            # Return most common regime
            if regimes:
                return max(set(regimes), key=regimes.count)
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"