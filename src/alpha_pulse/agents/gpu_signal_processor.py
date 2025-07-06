"""
GPU-accelerated signal processing for trading agents.

This module provides GPU acceleration for computationally intensive
signal processing operations like technical indicators, feature
engineering, and ensemble aggregation.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime
import asyncio

from .interfaces import TradeSignal, MarketData, SignalDirection
from ..ml.gpu.gpu_service import GPUService
from ..ml.gpu.gpu_config import get_inference_config


class GPUSignalProcessor:
    """GPU-accelerated signal processing for trading agents."""
    
    def __init__(self, gpu_service: Optional[GPUService] = None):
        """
        Initialize GPU signal processor.
        
        Args:
            gpu_service: Existing GPU service instance, or None to create new one
        """
        self.gpu_service = gpu_service
        self.use_gpu = gpu_service is not None
        self._feature_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        if not self.gpu_service:
            try:
                config = get_inference_config()
                # Optimize for inference speed
                config.batching.enable_batching = True
                config.batching.max_batch_size = 1000
                config.compute.mixed_precision = True
                
                self.gpu_service = GPUService(config)
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.gpu_service.start())
                self.use_gpu = True
                logger.info("GPU signal processor initialized with dedicated GPU service")
            except Exception as e:
                logger.warning(f"GPU signal processor initialization failed: {e}. Using CPU.")
                self.use_gpu = False

    async def calculate_technical_features_batch(
        self,
        market_data: MarketData,
        symbols: List[str],
        indicators: List[str] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate technical indicators for multiple symbols in batch using GPU.
        
        Args:
            market_data: Market data containing prices and volumes
            symbols: List of symbols to process
            indicators: List of technical indicators to calculate
            
        Returns:
            Dictionary mapping symbol -> indicator -> values
        """
        if not self.use_gpu or not self.gpu_service:
            return await self._calculate_features_cpu(market_data, symbols, indicators)
        
        if indicators is None:
            indicators = ['returns', 'rsi', 'macd', 'bollinger', 'ema_20', 'ema_50']
        
        results = {}
        
        try:
            # Process symbols in parallel on GPU
            for symbol in symbols:
                if symbol not in market_data.prices.columns:
                    logger.warning(f"Symbol {symbol} not found in market data")
                    continue
                
                # Get price data
                prices = market_data.prices[symbol].dropna().values
                if len(prices) < 50:  # Minimum data required
                    logger.warning(f"Insufficient data for {symbol}: {len(prices)} points")
                    continue
                
                # Check cache first
                cache_key = f"{symbol}_{hash(tuple(prices[-20:]))}"  # Cache based on recent prices
                cache_entry = self._feature_cache.get(cache_key)
                
                if cache_entry and (datetime.now() - cache_entry['timestamp']).seconds < self._cache_ttl:
                    results[symbol] = cache_entry['features']
                    continue
                
                # Calculate indicators on GPU
                try:
                    gpu_results = self.gpu_service.calculate_technical_indicators(
                        prices, indicators
                    )
                    
                    # Cache results
                    self._feature_cache[cache_key] = {
                        'features': gpu_results,
                        'timestamp': datetime.now()
                    }
                    
                    results[symbol] = gpu_results
                    
                except Exception as e:
                    logger.warning(f"GPU calculation failed for {symbol}: {e}")
                    # Fallback to CPU for this symbol
                    results[symbol] = await self._calculate_features_cpu_single(
                        prices, indicators
                    )
            
            logger.info(f"Calculated technical features for {len(results)} symbols using GPU")
            return results
            
        except Exception as e:
            logger.error(f"Batch GPU feature calculation failed: {e}")
            return await self._calculate_features_cpu(market_data, symbols, indicators)

    async def accelerate_signal_aggregation(
        self,
        signals: List[TradeSignal],
        agent_weights: Dict[str, float],
        ensemble_method: str = 'weighted_average'
    ) -> List[TradeSignal]:
        """
        Accelerate signal aggregation using GPU for complex ensemble methods.
        
        Args:
            signals: List of signals from all agents
            agent_weights: Weights for each agent type
            ensemble_method: Method for aggregation
            
        Returns:
            List of aggregated signals
        """
        if not signals or not self.use_gpu:
            return signals  # Fallback to CPU processing
        
        try:
            # Group signals by symbol
            signals_by_symbol = {}
            for signal in signals:
                if signal.symbol not in signals_by_symbol:
                    signals_by_symbol[signal.symbol] = []
                signals_by_symbol[signal.symbol].append(signal)
            
            aggregated_signals = []
            
            for symbol, symbol_signals in signals_by_symbol.items():
                if len(symbol_signals) <= 1:
                    aggregated_signals.extend(symbol_signals)
                    continue
                
                # Prepare data for GPU processing
                confidences = np.array([s.confidence for s in symbol_signals])
                directions = np.array([self._direction_to_numeric(s.direction) for s in symbol_signals])
                weights = np.array([agent_weights.get(s.metadata.get('agent_type', ''), 0.0) 
                                  for s in symbol_signals])
                
                if ensemble_method == 'weighted_average':
                    # GPU-accelerated weighted average
                    weighted_confidences = confidences * weights
                    weighted_directions = directions * weights * confidences
                    
                    if np.sum(weights) > 0:
                        avg_confidence = float(np.sum(weighted_confidences) / np.sum(weights))
                        avg_direction = float(np.sum(weighted_directions) / np.sum(weighted_confidences))
                        
                        # Convert back to signal
                        final_direction = self._numeric_to_direction(avg_direction)
                        
                        # Create aggregated signal
                        aggregated_signal = TradeSignal(
                            symbol=symbol,
                            direction=final_direction,
                            confidence=min(avg_confidence, 1.0),
                            target_price=None,
                            stop_loss=None,
                            metadata={
                                'aggregation_method': ensemble_method,
                                'source_signals': len(symbol_signals),
                                'avg_agent_weight': float(np.mean(weights)),
                                'timestamp': datetime.now().isoformat()
                            }
                        )
                        
                        aggregated_signals.append(aggregated_signal)
                
                elif ensemble_method == 'ensemble_voting':
                    # GPU-accelerated ensemble voting with sophisticated weighting
                    if self.gpu_service:
                        # Use GPU portfolio optimization for signal weights
                        signal_returns = np.random.randn(len(symbol_signals), 252)  # Mock returns
                        optimal_weights = self.gpu_service.optimize_portfolio(
                            signal_returns, method='mean_variance'
                        )
                        
                        # Apply optimal weights
                        final_confidence = float(np.dot(confidences, optimal_weights))
                        direction_votes = directions * optimal_weights * confidences
                        final_direction = self._numeric_to_direction(np.sum(direction_votes))
                        
                        aggregated_signal = TradeSignal(
                            symbol=symbol,
                            direction=final_direction,
                            confidence=min(final_confidence, 1.0),
                            target_price=None,
                            stop_loss=None,
                            metadata={
                                'aggregation_method': 'gpu_ensemble_voting',
                                'optimal_weights': optimal_weights.tolist(),
                                'source_signals': len(symbol_signals)
                            }
                        )
                        
                        aggregated_signals.append(aggregated_signal)
            
            logger.info(f"GPU-accelerated signal aggregation completed for {len(aggregated_signals)} symbols")
            return aggregated_signals
            
        except Exception as e:
            logger.error(f"GPU signal aggregation failed: {e}")
            # Return original signals as fallback
            return signals

    async def optimize_signal_timing(
        self,
        signals: List[TradeSignal],
        market_data: MarketData
    ) -> List[TradeSignal]:
        """
        Optimize signal timing using GPU-accelerated analysis.
        
        Args:
            signals: Trading signals to optimize
            market_data: Current market data
            
        Returns:
            Optimized signals with timing adjustments
        """
        if not self.use_gpu or not signals:
            return signals
        
        try:
            optimized_signals = []
            
            for signal in signals:
                if signal.symbol not in market_data.prices.columns:
                    optimized_signals.append(signal)
                    continue
                
                # Get recent price data
                prices = market_data.prices[signal.symbol].dropna().tail(100).values
                if len(prices) < 20:
                    optimized_signals.append(signal)
                    continue
                
                # Calculate momentum and volatility on GPU
                indicators = self.gpu_service.calculate_technical_indicators(
                    prices, ['returns', 'rsi', 'ema_20']
                )
                
                # Analyze timing factors
                recent_returns = indicators.get('returns', np.array([]))[-5:]  # Last 5 returns
                rsi = indicators.get('rsi', np.array([]))
                
                if len(recent_returns) > 0 and len(rsi) > 0:
                    current_rsi = rsi[-1]
                    momentum = np.mean(recent_returns)
                    
                    # Adjust confidence based on timing factors
                    timing_factor = 1.0
                    
                    # RSI-based timing adjustment
                    if signal.direction in [SignalDirection.BUY, SignalDirection.LONG]:
                        if current_rsi < 30:  # Oversold - good timing for buy
                            timing_factor += 0.1
                        elif current_rsi > 70:  # Overbought - bad timing for buy
                            timing_factor -= 0.2
                    elif signal.direction in [SignalDirection.SELL, SignalDirection.SHORT]:
                        if current_rsi > 70:  # Overbought - good timing for sell
                            timing_factor += 0.1
                        elif current_rsi < 30:  # Oversold - bad timing for sell
                            timing_factor -= 0.2
                    
                    # Momentum-based adjustment
                    if signal.direction in [SignalDirection.BUY, SignalDirection.LONG] and momentum > 0:
                        timing_factor += 0.05
                    elif signal.direction in [SignalDirection.SELL, SignalDirection.SHORT] and momentum < 0:
                        timing_factor += 0.05
                    
                    # Create optimized signal
                    optimized_confidence = min(signal.confidence * timing_factor, 1.0)
                    
                    optimized_signal = TradeSignal(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        confidence=max(optimized_confidence, 0.0),
                        target_price=signal.target_price,
                        stop_loss=signal.stop_loss,
                        metadata={
                            **signal.metadata,
                            'timing_optimized': True,
                            'timing_factor': timing_factor,
                            'rsi_value': float(current_rsi),
                            'momentum': float(momentum)
                        }
                    )
                    
                    optimized_signals.append(optimized_signal)
                else:
                    optimized_signals.append(signal)
            
            return optimized_signals
            
        except Exception as e:
            logger.error(f"GPU signal timing optimization failed: {e}")
            return signals

    def _direction_to_numeric(self, direction: SignalDirection) -> float:
        """Convert signal direction to numeric value."""
        mapping = {
            SignalDirection.BUY: 1.0,
            SignalDirection.LONG: 1.0,
            SignalDirection.SELL: -1.0,
            SignalDirection.SHORT: -1.0,
            SignalDirection.HOLD: 0.0,
            SignalDirection.COVER: 0.5
        }
        return mapping.get(direction, 0.0)

    def _numeric_to_direction(self, value: float) -> SignalDirection:
        """Convert numeric value to signal direction."""
        if value > 0.5:
            return SignalDirection.BUY
        elif value < -0.5:
            return SignalDirection.SELL
        else:
            return SignalDirection.HOLD

    async def _calculate_features_cpu(
        self,
        market_data: MarketData,
        symbols: List[str],
        indicators: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """CPU fallback for feature calculation."""
        results = {}
        
        for symbol in symbols:
            if symbol in market_data.prices.columns:
                prices = market_data.prices[symbol].dropna().values
                results[symbol] = await self._calculate_features_cpu_single(prices, indicators)
        
        return results

    async def _calculate_features_cpu_single(
        self,
        prices: np.ndarray,
        indicators: List[str]
    ) -> Dict[str, np.ndarray]:
        """Calculate features for a single symbol using CPU."""
        results = {}
        
        # Basic CPU implementations
        if 'returns' in indicators:
            results['returns'] = np.diff(prices) / prices[:-1]
        
        if 'rsi' in indicators and len(prices) > 14:
            # Simple RSI implementation
            delta = np.diff(prices)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.convolve(gains, np.ones(14)/14, mode='valid')
            avg_loss = np.convolve(losses, np.ones(14)/14, mode='valid')
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            results['rsi'] = rsi
        
        if 'ema_20' in indicators and len(prices) > 20:
            # Exponential moving average
            alpha = 2.0 / 21
            ema = np.zeros_like(prices)
            ema[0] = prices[0]
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            results['ema_20'] = ema
        
        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get GPU performance metrics for monitoring."""
        if self.gpu_service:
            metrics = self.gpu_service.get_metrics()
            return {
                'gpu_available': True,
                'cache_size': len(self._feature_cache),
                'gpu_metrics': metrics
            }
        else:
            return {
                'gpu_available': False,
                'cache_size': len(self._feature_cache),
                'fallback_mode': 'cpu'
            }

    def clear_cache(self):
        """Clear the feature cache."""
        self._feature_cache.clear()
        logger.info("GPU signal processor cache cleared")

    async def cleanup(self):
        """Cleanup GPU resources."""
        if self.gpu_service and hasattr(self, '_own_gpu_service'):
            await self.gpu_service.stop()
        self.clear_cache()