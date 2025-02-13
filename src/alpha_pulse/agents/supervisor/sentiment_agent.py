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
        self._market_mood = "neutral"  # bullish, neutral, bearish
        self._source_correlations = defaultdict(float)
        
    async def optimize(self) -> None:
        """
        Optimize sentiment analysis parameters based on performance metrics.
        This includes:
        1. Adjusting source weights based on predictive power
        2. Optimizing sentiment thresholds based on market mood
        3. Adapting source credibility weights
        """
        await super().optimize()
        
        try:
            # Optimize source weights based on performance
            performance_scores = {}
            for source, history in self._source_performance.items():
                if len(history) >= 50:  # Need sufficient history
                    recent_history = history[-50:]
                    
                    # Calculate predictive accuracy
                    accuracy = np.mean([
                        1 if h['predicted_direction'] == h['actual_direction'] else 0
                        for h in recent_history
                    ])
                    
                    # Calculate correlation with price movements
                    correlation = self._source_correlations.get(source, 0)
                    
                    # Combined score
                    performance_scores[source] = accuracy * (1 + abs(correlation))
                    
            if performance_scores:
                # Normalize scores to weights
                total_score = sum(performance_scores.values())
                if total_score > 0:
                    new_weights = {
                        k: v / total_score
                        for k, v in performance_scores.items()
                    }
                    
                    # Apply smoothing to weight updates
                    smoothing = 0.7  # 70% old weights, 30% new weights
                    self.sentiment_sources = {
                        k: smoothing * self.sentiment_sources[k] + (1 - smoothing) * new_weights[k]
                        for k in self.sentiment_sources
                    }
                    
                    logger.info(f"Updated sentiment source weights: {self.sentiment_sources}")
                    
            # Optimize sentiment thresholds based on market mood
            if self._market_mood == "bullish":
                # More aggressive in bullish markets
                self.sentiment_thresholds.update({
                    'strong_buy': 0.6,  # Lower threshold for strong buy
                    'buy': 0.4,
                    'sell': -0.6,
                    'strong_sell': -0.8  # Higher threshold for strong sell
                })
            elif self._market_mood == "bearish":
                # More conservative in bearish markets
                self.sentiment_thresholds.update({
                    'strong_buy': 0.8,  # Higher threshold for strong buy
                    'buy': 0.6,
                    'sell': -0.4,
                    'strong_sell': -0.6  # Lower threshold for strong sell
                })
            else:  # neutral
                # Balanced thresholds
                self.sentiment_thresholds.update({
                    'strong_buy': 0.7,
                    'buy': 0.5,
                    'sell': -0.5,
                    'strong_sell': -0.7
                })
                
            logger.info(f"Updated sentiment thresholds for {self._market_mood} market: {self.sentiment_thresholds}")
            
            # Store optimization result
            self._optimization_history.append({
                'timestamp': datetime.now(),
                'market_mood': self._market_mood,
                'source_weights': self.sentiment_sources.copy(),
                'thresholds': self.sentiment_thresholds.copy(),
                'performance_scores': performance_scores
            })
            
        except Exception as e:
            logger.error(f"Error in sentiment agent optimization: {str(e)}")
            raise
            
    async def self_evaluate(self) -> Dict[str, float]:
        """
        Evaluate agent's performance and market adaptation.
        Returns performance metrics specific to sentiment analysis.
        """
        metrics = await super().self_evaluate()
        
        try:
            # Calculate source-specific performance
            for source, history in self._source_performance.items():
                if history:
                    recent_history = history[-50:]  # Look at last 50 predictions
                    metrics[f"{source}_accuracy"] = np.mean([
                        1 if h['predicted_direction'] == h['actual_direction'] else 0
                        for h in recent_history
                    ])
                    metrics[f"{source}_correlation"] = self._source_correlations.get(source, 0)
                    
            # Calculate sentiment prediction metrics
            if self._sentiment_predictions:
                recent_predictions = self._sentiment_predictions[-50:]
                metrics['sentiment_accuracy'] = np.mean([
                    1 if p['predicted'] == p['actual'] else 0
                    for p in recent_predictions
                ])
                
                # Calculate directional accuracy
                metrics['directional_accuracy'] = np.mean([
                    1 if np.sign(p['predicted_score']) == np.sign(p['actual_return'])
                    else 0
                    for p in recent_predictions
                ])
                
            # Calculate adaptation metrics
            if len(self._optimization_history) >= 2:
                last_opt = self._optimization_history[-1]
                prev_opt = self._optimization_history[-2]
                
                # Calculate weight change magnitude
                weight_changes = [
                    abs(last_opt['source_weights'][k] - prev_opt['source_weights'][k])
                    for k in last_opt['source_weights']
                ]
                metrics['weight_adaptation'] = np.mean(weight_changes)
                
                # Calculate threshold change magnitude
                threshold_changes = [
                    abs(last_opt['thresholds'][k] - prev_opt['thresholds'][k])
                    for k in last_opt['thresholds']
                ]
                metrics['threshold_adaptation'] = np.mean(threshold_changes)
                
            logger.debug(f"Sentiment agent self-evaluation metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error in sentiment agent self-evaluation: {str(e)}")
            raise
            
        return metrics
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate trading signals with self-supervision capabilities."""
        try:
            # Update market mood
            self._market_mood = await self._detect_market_mood(market_data)
            logger.info(f"Current market mood: {self._market_mood}")
            
            # Generate signals using parent implementation
            signals = await super().generate_signals(market_data)
            
            # Track sentiment predictions
            if market_data.sentiment:
                for symbol, sentiment_data in market_data.sentiment.items():
                    # Calculate individual source predictions
                    for source, weight in self.sentiment_sources.items():
                        prediction = await self._analyze_source_sentiment(
                            source,
                            sentiment_data,
                            symbol,
                            market_data
                        )
                        
                        self._source_performance[source].append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'predicted_score': prediction['score'],
                            'predicted_direction': prediction['direction'],
                            'actual_direction': None,  # Will be updated in next iteration
                            'confidence': prediction['confidence']
                        })
                        
            # Update previous predictions with actual outcomes
            current_prices = {
                symbol: prices.iloc[-1]
                for symbol, prices in market_data.prices.items()
            }
            
            for source, history in self._source_performance.items():
                for record in history:
                    if record['actual_direction'] is None and record['symbol'] in current_prices:
                        symbol = record['symbol']
                        if len(market_data.prices[symbol]) > 1:
                            price_change = (
                                current_prices[symbol] -
                                market_data.prices[symbol].iloc[-2]
                            )
                            record['actual_direction'] = np.sign(price_change)
                            
                            # Update source correlation
                            predictions = [
                                h['predicted_score']
                                for h in history[-50:]
                                if h['actual_direction'] is not None
                            ]
                            returns = [
                                h['actual_direction']
                                for h in history[-50:]
                                if h['actual_direction'] is not None
                            ]
                            if len(predictions) >= 10:
                                correlation = np.corrcoef(predictions, returns)[0, 1]
                                self._source_correlations[source] = correlation
                                
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals in sentiment agent: {str(e)}")
            raise
            
    async def _detect_market_mood(self, market_data: MarketData) -> str:
        """
        Detect the current market mood using sentiment indicators.
        Returns: "bullish", "neutral", or "bearish"
        """
        try:
            if not market_data.sentiment:
                return self._market_mood
                
            # Aggregate sentiment scores
            total_sentiment = 0
            count = 0
            
            for sentiment_data in market_data.sentiment.values():
                # News sentiment
                news_score = await self._analyze_source_sentiment(
                    'news',
                    sentiment_data,
                    None,
                    market_data
                )
                if news_score['score'] is not None:
                    total_sentiment += news_score['score']
                    count += 1
                    
                # Social media sentiment
                social_score = await self._analyze_source_sentiment(
                    'social_media',
                    sentiment_data,
                    None,
                    market_data
                )
                if social_score['score'] is not None:
                    total_sentiment += social_score['score']
                    count += 1
                    
            if count > 0:
                avg_sentiment = total_sentiment / count
                if avg_sentiment > 0.3:
                    return "bullish"
                elif avg_sentiment < -0.3:
                    return "bearish"
                    
            return "neutral"
            
        except Exception as e:
            logger.error(f"Error detecting market mood: {str(e)}")
            return self._market_mood
            
    async def _analyze_source_sentiment(
        self,
        source: str,
        sentiment_data: Dict,
        symbol: Optional[str],
        market_data: MarketData
    ) -> Dict[str, Any]:
        """Analyze sentiment from a specific source."""
        try:
            score = None
            confidence = 0.5
            direction = SignalDirection.HOLD
            
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
                            
            if score is not None:
                direction = (
                    SignalDirection.BUY if score > 0
                    else SignalDirection.SELL if score < 0
                    else SignalDirection.HOLD
                )
                
            return {
                'score': score,
                'confidence': confidence,
                'direction': direction
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {source} sentiment: {str(e)}")
            return {
                'score': None,
                'confidence': 0.0,
                'direction': SignalDirection.HOLD
            }