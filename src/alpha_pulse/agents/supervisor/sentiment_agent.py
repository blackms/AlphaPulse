"""
Self-supervised sentiment analysis agent implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from textblob import TextBlob
from collections import defaultdict
from loguru import logger

from ..interfaces import MarketData, TradeSignal, SignalDirection
from .base import BaseSelfSupervisedAgent


class SelfSupervisedSentimentAgent(BaseSelfSupervisedAgent):
    """
    Self-supervised sentiment analysis agent that can optimize its own parameters
    and adapt to changing market conditions.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize self-supervised sentiment agent."""
        super().__init__(agent_id, config)
        
        # Sentiment analysis parameters
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
        
        # Self-supervision parameters
        self._source_performance = {
            source: [] for source in self.sentiment_sources.keys()
        }
        self._sentiment_predictions = []
        self._market_mood = "unknown"
        self._source_correlations = defaultdict(float)
        self._optimization_history = []
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate trading signals with self-supervision capabilities."""
        try:
            # Check for sentiment data
            if not hasattr(market_data, 'sentiment') or not market_data.sentiment:
                self._market_mood = "unknown (no sentiment data)"
                return []
                
            # Update market mood
            self._market_mood = await self._detect_market_mood(market_data)
            if self._market_mood not in ["neutral", "unknown"]:
                logger.debug(f"Market mood: {self._market_mood}")
            
            signals = []
            
            # Process each symbol
            for symbol in market_data.prices.columns:
                try:
                    # Get sentiment data
                    sentiment_data = market_data.sentiment.get(symbol, {})
                    if not sentiment_data:
                        continue
                        
                    # Calculate sentiment scores
                    news_score = await self._analyze_source_sentiment('news', sentiment_data)
                    social_score = await self._analyze_source_sentiment('social_media', sentiment_data)
                    market_score = await self._analyze_source_sentiment('market_data', sentiment_data)
                    analyst_score = await self._analyze_source_sentiment('analyst_ratings', sentiment_data)
                    
                    # Check if we have enough sentiment signals
                    valid_scores = [
                        score for score in [news_score, social_score, market_score, analyst_score]
                        if score['score'] != 0 or score['confidence'] > 0
                    ]
                    
                    if not valid_scores:
                        continue
                        
                    # Calculate weighted sentiment score
                    sentiment_score = (
                        news_score['score'] * self.sentiment_sources['news'] +
                        social_score['score'] * self.sentiment_sources['social_media'] +
                        market_score['score'] * self.sentiment_sources['market_data'] +
                        analyst_score['score'] * self.sentiment_sources['analyst_ratings']
                    )
                    
                    # Calculate confidence
                    confidence = np.mean([
                        score['confidence']
                        for score in valid_scores
                    ])
                    
                    # Generate signal if sentiment is strong enough
                    if abs(sentiment_score) > self.sentiment_thresholds['buy']:
                        direction = (
                            SignalDirection.BUY if sentiment_score > 0
                            else SignalDirection.SELL
                        )
                        
                        # Calculate target price and stop loss
                        current_price = float(market_data.prices[symbol].iloc[-1])
                        target_price = current_price * (1 + abs(sentiment_score))
                        stop_loss = current_price * (1 - abs(sentiment_score) * 0.5)
                        
                        signal = TradeSignal(
                            agent_id=self.agent_id,
                            symbol=symbol,
                            direction=direction,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            target_price=target_price,
                            stop_loss=stop_loss,
                            metadata={
                                "strategy": "sentiment",
                                "market_mood": self._market_mood,
                                "scores": {
                                    "news": float(news_score['score'] or 0),
                                    "social": float(social_score['score'] or 0),
                                    "market": float(market_score['score'] or 0),
                                    "analyst": float(analyst_score['score'] or 0),
                                    "total": float(sentiment_score)
                                }
                            }
                        )
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {str(e)}")
                    continue
                    
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals in sentiment agent: {str(e)}")
            raise
            
    async def _analyze_source_sentiment(
        self,
        source: str,
        sentiment_data: Dict
    ) -> Dict[str, Any]:
        """Analyze sentiment from a specific source."""
        try:
            score = None
            confidence = 0.0  # Start with zero confidence
            
            if source == 'news':
                news_items = sentiment_data.get("news", [])
                if news_items:
                    scores = []
                    for item in news_items:
                        text = item.get("title", "") + " " + item.get("content", "")
                        blob = TextBlob(text)
                        scores.append(blob.sentiment.polarity)
                    score = np.mean(scores)
                    confidence = 1 - np.std(scores) if len(scores) > 1 else 0.5
                    
            elif source == 'social_media':
                social_data = sentiment_data.get("social_media", {})
                if social_data:
                    platform_scores = []
                    for platform_data in social_data.values():
                        if platform_data:
                            platform_scores.extend([
                                post.get("sentiment", 0) * np.log1p(post.get("engagement", 1))
                                for post in platform_data
                            ])
                    if platform_scores:
                        score = np.mean(platform_scores)
                        confidence = 1 - np.std(platform_scores) if len(platform_scores) > 1 else 0.5
                        
            elif source == 'market_data':
                # Use price momentum as market sentiment
                prices = sentiment_data.get("prices", [])
                if prices:
                    returns = np.diff(prices) / prices[:-1]
                    score = np.tanh(np.mean(returns[-5:]) * 10)  # Scale recent returns
                    confidence = 1 - np.std(returns[-5:])
                    
            elif source == 'analyst_ratings':
                analyst_data = sentiment_data.get("analyst_ratings", {})
                if analyst_data:
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
                            score = weighted_sum / total_ratings
                            confidence = min(total_ratings / 10, 1.0)  # Scale with number of ratings
                            
            return {
                'score': float(score) if score is not None else 0.0,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {source} sentiment: {str(e)}")
            return {'score': 0.0, 'confidence': 0.0}
            
    async def _detect_market_mood(self, market_data: MarketData) -> str:
        """Detect the current market mood."""
        try:
            if not market_data.sentiment:
                return "unknown (no sentiment data)"
                
            # Aggregate sentiment scores
            total_sentiment = 0
            count = 0
            
            for sentiment_data in market_data.sentiment.values():
                # News sentiment
                news_score = await self._analyze_source_sentiment('news', sentiment_data)
                if news_score['score'] != 0:
                    total_sentiment += news_score['score']
                    count += 1
                    
                # Social media sentiment
                social_score = await self._analyze_source_sentiment('social_media', sentiment_data)
                if social_score['score'] != 0:
                    total_sentiment += social_score['score']
                    count += 1
                    
            if count > 0:
                avg_sentiment = total_sentiment / count
                if avg_sentiment > 0.3:
                    return "bullish"
                elif avg_sentiment < -0.3:
                    return "bearish"
                    
            return "neutral (insufficient signals)"
            
        except Exception as e:
            logger.error(f"Error detecting market mood: {str(e)}")
            return "unknown (error)"