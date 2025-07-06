"""
Market sentiment analysis agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from textblob import TextBlob
from collections import defaultdict
from loguru import logger

from .interfaces import (
    BaseTradeAgent,
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)
from .regime_mixin import RegimeAwareMixin
from alpha_pulse.decorators.audit_decorators import audit_agent_signal
from alpha_pulse.services.regime_detection_service import RegimeDetectionService
from alpha_pulse.ml.regime.regime_classifier import RegimeInfo, RegimeType


class SentimentAgent(RegimeAwareMixin, BaseTradeAgent):
    """
    Implements sentiment analysis strategies focusing on market psychology,
    social media trends, and news sentiment with regime awareness.
    """
    
    def __init__(self, config: Dict[str, Any] = None, regime_service: Optional[RegimeDetectionService] = None):
        """Initialize sentiment analysis agent."""
        super().__init__("sentiment_agent", config, regime_service=regime_service)
        self.sentiment_sources = {
            'news': self.config.get("news_weight", 0.3),
            'social_media': self.config.get("social_media_weight", 0.25),
            'market_data': self.config.get("market_data_weight", 0.25),
            'analyst_ratings': self.config.get("analyst_weight", 0.2)
        }
        self.sentiment_thresholds = {
            'strong_buy': 0.7,
            'buy': 0.5,
            'neutral': 0.0,
            'sell': -0.5,
            'strong_sell': -0.7
        }
        self.momentum_window = self.config.get("momentum_window", 14)
        self.volume_window = self.config.get("volume_window", 5)
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize agent with configuration."""
        await super().initialize(config)
        self.sentiment_history = defaultdict(list)
        self.volume_history = defaultdict(list)
        self.momentum_history = defaultdict(list)
        
    @audit_agent_signal(agent_type='sentiment', include_market_data=True)
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate trading signals based on sentiment analysis with regime awareness.
        
        Args:
            market_data: Market data including sentiment indicators and news
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not market_data.sentiment:
            return signals
        
        # Get current market regime
        regime_info = await self.get_current_regime()
        regime_context = self.get_regime_strategy_context(regime_info)
        
        logger.debug(f"Sentiment agent operating in {regime_context['regime_type']} regime "
                    f"(mode: {regime_context['strategy_mode']})")
            
        for symbol, sentiment_data in market_data.sentiment.items():
            # Calculate composite sentiment score
            news_score = await self._analyze_news_sentiment(sentiment_data)
            social_score = await self._analyze_social_sentiment(sentiment_data)
            market_score = await self._analyze_market_sentiment(symbol, market_data)
            analyst_score = await self._analyze_analyst_sentiment(sentiment_data)
            
            # Weighted sentiment score
            base_sentiment_score = (
                news_score * self.sentiment_sources['news'] +
                social_score * self.sentiment_sources['social_media'] +
                market_score * self.sentiment_sources['market_data'] +
                analyst_score * self.sentiment_sources['analyst_ratings']
            )
            
            # Update sentiment history
            self.sentiment_history[symbol].append(base_sentiment_score)
            
            # Apply regime-based adjustments
            base_confidence = abs(base_sentiment_score)
            adjusted_score, adjusted_confidence = self.adjust_signal_for_regime(
                base_sentiment_score, base_confidence, regime_info
            )
            
            # Log regime adjustment if significant
            await self.log_regime_based_decision(symbol, base_sentiment_score, adjusted_score, regime_info)
            
            # Generate signal based on sentiment score and regime context
            signal = await self._generate_regime_aware_sentiment_signal(
                symbol,
                adjusted_score,
                adjusted_confidence,
                market_data,
                regime_context
            )
            
            if signal:
                signals.append(signal)
                
        return signals
    
    async def _generate_regime_aware_sentiment_signal(
        self,
        symbol: str,
        sentiment_score: float,
        confidence: float,
        market_data: MarketData,
        regime_context: Dict[str, Any]
    ) -> Optional[TradeSignal]:
        """Generate sentiment signal based on regime context."""
        strategy_mode = regime_context['strategy_mode']
        
        # Adjust sentiment thresholds based on regime
        if strategy_mode == "defensive":
            # More conservative thresholds in bear markets
            buy_threshold = 0.8
            sell_threshold = -0.8
        elif strategy_mode == "trend_following":
            # Standard thresholds in bull markets
            buy_threshold = 0.7
            sell_threshold = -0.7
        elif strategy_mode == "mean_reversion":
            # Contrarian approach in volatile markets
            buy_threshold = -0.5  # Buy on negative sentiment (contrarian)
            sell_threshold = 0.5   # Sell on positive sentiment (contrarian)
            sentiment_score = -sentiment_score  # Reverse sentiment
        else:  # range_trading or neutral
            buy_threshold = 0.6
            sell_threshold = -0.6
        
        # Get sentiment trend
        recent_sentiment = self.sentiment_history[symbol][-20:] if self.sentiment_history[symbol] else []
        sentiment_trend = np.mean(recent_sentiment) if recent_sentiment else sentiment_score
        
        # Calculate confidence based on sentiment consistency
        sentiment_std = np.std(recent_sentiment) if len(recent_sentiment) > 1 else 1
        consistency_factor = 1 / (1 + sentiment_std)
        final_confidence = confidence * consistency_factor
        
        # Determine signal direction based on adjusted thresholds
        if sentiment_score > buy_threshold and sentiment_trend > (buy_threshold * 0.7):
            direction = SignalDirection.BUY
        elif sentiment_score < sell_threshold and sentiment_trend < (sell_threshold * 0.7):
            direction = SignalDirection.SELL
        else:
            return None
            
        # Calculate target price and stop loss
        current_price = market_data.prices[symbol].iloc[-1] if symbol in market_data.prices else 0
        if current_price > 0:
            # Adjust risk based on regime
            risk_multiplier = self._get_risk_multiplier_for_sentiment(regime_context['risk_tolerance'])
            
            if direction == SignalDirection.BUY:
                target_price = current_price * (1 + abs(sentiment_score) * risk_multiplier)
                stop_loss = current_price * (1 - abs(sentiment_score) * 0.5 * risk_multiplier)
            else:
                target_price = current_price * (1 - abs(sentiment_score) * risk_multiplier)
                stop_loss = current_price * (1 + abs(sentiment_score) * 0.5 * risk_multiplier)
        else:
            target_price = stop_loss = None
            
        return TradeSignal(
            agent_id=self.agent_id,
            symbol=symbol,
            direction=direction,
            confidence=final_confidence,
            timestamp=datetime.now(),
            target_price=target_price,
            stop_loss=stop_loss,
            metadata={
                "strategy": "sentiment",
                "sentiment_score": sentiment_score,
                "sentiment_trend": sentiment_trend,
                "sentiment_std": sentiment_std,
                "news_sentiment": await self._get_sentiment_breakdown(symbol, market_data),
                "market_indicators": await self._get_market_indicators(symbol),
                "regime_context": regime_context,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "risk_multiplier": self._get_risk_multiplier_for_sentiment(regime_context['risk_tolerance'])
            }
        )
    
    def _get_risk_multiplier_for_sentiment(self, risk_tolerance: str) -> float:
        """Get risk multiplier for sentiment-based signals."""
        multipliers = {
            "very_low": 0.3,
            "low": 0.5,
            "moderate": 1.0,
            "high": 1.5,
            "very_high": 2.0
        }
        return multipliers.get(risk_tolerance, 1.0)
    
    async def _fallback_regime_detection(self) -> Optional[RegimeInfo]:
        """
        Fallback regime detection using sentiment indicators.
        
        This is used when the centralized regime service is unavailable.
        """
        try:
            # Simple fallback based on overall market sentiment
            # This would ideally use historical sentiment patterns
            
            from dataclasses import dataclass
            
            @dataclass
            class FallbackRegimeInfo:
                regime_type: RegimeType = RegimeType.RANGING
                current_regime: int = 0
                confidence: float = 0.5
                expected_remaining_duration: float = 10.0
                transition_probability: float = 0.1
            
            # Default to ranging regime for sentiment analysis
            return FallbackRegimeInfo(
                regime_type=RegimeType.RANGING,
                current_regime=0,
                confidence=0.5
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment fallback regime detection: {e}")
            return None

    async def _analyze_news_sentiment(self, sentiment_data: Dict) -> float:
        """Analyze news sentiment."""
        news_items = sentiment_data.get("news", [])
        if not news_items:
            return 0
            
        scores = []
        for item in news_items:
            # Calculate base sentiment
            text_blob = TextBlob(item.get("title", "") + " " + item.get("content", ""))
            base_sentiment = text_blob.sentiment.polarity
            
            # Apply source credibility weight
            credibility = item.get("source_credibility", 0.5)
            weighted_sentiment = base_sentiment * credibility
            
            # Apply recency weight
            age_hours = (datetime.now() - item.get("timestamp", datetime.now())).total_seconds() / 3600
            recency_weight = np.exp(-age_hours / 24)  # Exponential decay over 24 hours
            
            scores.append(weighted_sentiment * recency_weight)
            
        return np.mean(scores) if scores else 0
        
    async def _analyze_social_sentiment(self, sentiment_data: Dict) -> float:
        """Analyze social media sentiment."""
        social_data = sentiment_data.get("social_media", {})
        if not social_data:
            return 0
            
        platform_weights = {
            "twitter": 0.4,
            "reddit": 0.3,
            "stocktwits": 0.3
        }
        
        weighted_scores = []
        for platform, weight in platform_weights.items():
            platform_data = social_data.get(platform, [])
            if platform_data:
                # Calculate engagement-weighted sentiment
                sentiments = []
                for post in platform_data:
                    sentiment = post.get("sentiment", 0)
                    engagement = post.get("engagement", 1)
                    sentiments.append(sentiment * np.log1p(engagement))
                platform_score = np.mean(sentiments) if sentiments else 0
                weighted_scores.append(platform_score * weight)
                
        return sum(weighted_scores) if weighted_scores else 0
        
    async def _analyze_market_sentiment(self, symbol: str, market_data: MarketData) -> float:
        """Analyze market sentiment using price and volume data."""
        if symbol not in market_data.prices.columns:
            return 0
            
        prices = market_data.prices[symbol].dropna()
        volumes = market_data.volumes[symbol].dropna() if market_data.volumes is not None else None
        
        if len(prices) < self.momentum_window:
            return 0
            
        scores = []
        
        # Momentum indicator
        momentum = (prices.iloc[-1] / prices.iloc[-self.momentum_window] - 1)
        momentum_score = np.tanh(momentum * 5)  # Scale and normalize
        scores.append(momentum_score)
        
        # Volume analysis
        if volumes is not None and len(volumes) >= self.volume_window:
            volume_ratio = volumes.iloc[-1] / volumes.iloc[-self.volume_window:].mean()
            volume_score = np.tanh(np.log(volume_ratio))  # Log scale and normalize
            scores.append(volume_score)
            
        # Update histories
        self.momentum_history[symbol].append(momentum_score)
        if volumes is not None:
            self.volume_history[symbol].append(volume_ratio)
            
        return np.mean(scores) if scores else momentum_score
        
    async def _analyze_analyst_sentiment(self, sentiment_data: Dict) -> float:
        """Analyze analyst ratings and price targets."""
        analyst_data = sentiment_data.get("analyst_ratings", {})
        if not analyst_data:
            return 0
            
        scores = []
        
        # Consensus rating score
        ratings = analyst_data.get("ratings", {})
        if ratings:
            rating_weights = {
                "strong_buy": 1.0,
                "buy": 0.5,
                "hold": 0.0,
                "sell": -0.5,
                "strong_sell": -1.0
            }
            weighted_sum = sum(rating_weights[rating] * count 
                             for rating, count in ratings.items())
            total_ratings = sum(ratings.values())
            if total_ratings > 0:
                scores.append(weighted_sum / total_ratings)
                
        # Price target analysis
        current_price = analyst_data.get("current_price", 0)
        target_price = analyst_data.get("consensus_target", current_price)
        if current_price > 0:
            price_target_score = np.tanh((target_price / current_price - 1) * 2)
            scores.append(price_target_score)
            
        return np.mean(scores) if scores else 0
        
    async def _get_sentiment_breakdown(self, symbol: str, market_data: MarketData) -> Dict:
        """Get detailed sentiment breakdown for signal metadata."""
        sentiment_data = market_data.sentiment.get(symbol, {})
        return {
            "news": await self._analyze_news_sentiment(sentiment_data),
            "social": await self._analyze_social_sentiment(sentiment_data),
            "analyst": await self._analyze_analyst_sentiment(sentiment_data)
        }
        
    async def _get_market_indicators(self, symbol: str) -> Dict:
        """Get market indicators for signal metadata."""
        return {
            "momentum": self.momentum_history[symbol][-1] if self.momentum_history[symbol] else 0,
            "volume_ratio": self.volume_history[symbol][-1] if self.volume_history[symbol] else 1,
            "sentiment_ma": np.mean(self.sentiment_history[symbol][-20:]) \
                if self.sentiment_history[symbol] else 0
        }