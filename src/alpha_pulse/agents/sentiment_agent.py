"""
Market sentiment analysis agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from textblob import TextBlob
from collections import defaultdict

from .interfaces import (
    BaseTradeAgent,
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)


class SentimentAgent(BaseTradeAgent):
    """
    Implements sentiment analysis strategies focusing on market psychology,
    social media trends, and news sentiment.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize sentiment analysis agent."""
        super().__init__("sentiment_agent", config)
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
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate trading signals based on sentiment analysis.
        
        Args:
            market_data: Market data including sentiment indicators and news
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not market_data.sentiment:
            return signals
            
        for symbol, sentiment_data in market_data.sentiment.items():
            # Calculate composite sentiment score
            news_score = await self._analyze_news_sentiment(sentiment_data)
            social_score = await self._analyze_social_sentiment(sentiment_data)
            market_score = await self._analyze_market_sentiment(symbol, market_data)
            analyst_score = await self._analyze_analyst_sentiment(sentiment_data)
            
            # Weighted sentiment score
            sentiment_score = (
                news_score * self.sentiment_sources['news'] +
                social_score * self.sentiment_sources['social_media'] +
                market_score * self.sentiment_sources['market_data'] +
                analyst_score * self.sentiment_sources['analyst_ratings']
            )
            
            # Update sentiment history
            self.sentiment_history[symbol].append(sentiment_score)
            
            # Generate signal based on sentiment score and momentum
            signal = await self._generate_sentiment_signal(
                symbol,
                sentiment_score,
                market_data
            )
            
            if signal:
                signals.append(signal)
                
        return signals
        
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
        
    async def _generate_sentiment_signal(
        self,
        symbol: str,
        sentiment_score: float,
        market_data: MarketData
    ) -> Optional[TradeSignal]:
        """Generate trading signal based on sentiment score."""
        # Get sentiment trend
        recent_sentiment = self.sentiment_history[symbol][-20:] if self.sentiment_history[symbol] else []
        sentiment_trend = np.mean(recent_sentiment) if recent_sentiment else sentiment_score
        
        # Calculate confidence based on sentiment consistency
        sentiment_std = np.std(recent_sentiment) if len(recent_sentiment) > 1 else 1
        confidence = 1 / (1 + sentiment_std)  # Higher consistency = higher confidence
        
        # Determine signal direction
        if sentiment_score > self.sentiment_thresholds['strong_buy'] and \
           sentiment_trend > self.sentiment_thresholds['buy']:
            direction = SignalDirection.BUY
        elif sentiment_score < self.sentiment_thresholds['strong_sell'] and \
             sentiment_trend < self.sentiment_thresholds['sell']:
            direction = SignalDirection.SELL
        else:
            return None
            
        # Calculate target price and stop loss
        current_price = market_data.prices[symbol].iloc[-1] if symbol in market_data.prices else 0
        if current_price > 0:
            if direction == SignalDirection.BUY:
                target_price = current_price * (1 + abs(sentiment_score))
                stop_loss = current_price * (1 - abs(sentiment_score) * 0.5)
            else:
                target_price = current_price * (1 - abs(sentiment_score))
                stop_loss = current_price * (1 + abs(sentiment_score) * 0.5)
        else:
            target_price = stop_loss = None
            
        return TradeSignal(
            agent_id=self.agent_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now(),
            target_price=target_price,
            stop_loss=stop_loss,
            metadata={
                "strategy": "sentiment",
                "sentiment_score": sentiment_score,
                "sentiment_trend": sentiment_trend,
                "sentiment_std": sentiment_std,
                "news_sentiment": await self._get_sentiment_breakdown(symbol, market_data),
                "market_indicators": await self._get_market_indicators(symbol)
            }
        )
        
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