"""
Agent manager for coordinating multiple trading agents.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from loguru import logger

from .interfaces import (
    BaseTradeAgent,
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)
from .factory import AgentFactory
from alpha_pulse.decorators.audit_decorators import (
    audit_agent_signal,
    audit_trade_decision
)
from alpha_pulse.services.ensemble_service import EnsembleService
from alpha_pulse.models.ensemble_model import AgentSignalCreate
from .gpu_signal_processor import GPUSignalProcessor
from alpha_pulse.services.explainability_service import ExplainabilityService


class AgentManager:
    """
    Manages multiple trading agents and coordinates their signals.
    Provides a unified interface for the risk management system.
    """
    
    def __init__(self, config: Dict[str, Any] = None, ensemble_service: EnsembleService = None, gpu_service=None):
        """Initialize agent manager."""
        self.config = config or {}
        self.agents: Dict[str, BaseTradeAgent] = {}
        self.agent_weights: Dict[str, float] = self.config.get("agent_weights", {
            "activist": 0.15,
            "value": 0.20,
            "fundamental": 0.20,
            "sentiment": 0.15,
            "technical": 0.15,
            "valuation": 0.15
        })
        self.signal_history: Dict[str, List[TradeSignal]] = defaultdict(list)
        self.performance_metrics: Dict[str, AgentMetrics] = {}
        
        # Ensemble integration
        self.ensemble_service = ensemble_service
        self.ensemble_id = None
        self.agent_registry: Dict[str, str] = {}  # agent_type -> agent_id mapping
        self.use_ensemble = config.get("use_ensemble", True) if ensemble_service else False
        
        # GPU acceleration
        self.gpu_processor = GPUSignalProcessor(gpu_service)
        self.use_gpu_acceleration = config.get("use_gpu_acceleration", True)
        
        # Explainable AI
        self.explainability_service = ExplainabilityService()
        self.enable_explanations = config.get("enable_explanations", True)
        
    async def initialize(self) -> None:
        """Initialize all agents."""
        # Create agent instances
        self.agents = await AgentFactory.create_all_agents(
            self.config.get("agent_configs", {})
        )
        
        # Normalize agent weights
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            self.agent_weights = {
                k: v / total_weight
                for k, v in self.agent_weights.items()
            }
            
        # Initialize ensemble if available
        if self.ensemble_service and self.use_ensemble:
            await self._initialize_ensemble()
            
    @audit_trade_decision(extract_reasoning=True, include_market_data=True)
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Generate and aggregate signals from all agents.
        
        Args:
            market_data: Market data object with prices and volumes
            
        Returns:
            List of aggregated trading signals
        """
        all_signals = []

        try:
            # Pre-calculate GPU-accelerated features if enabled
            symbols = list(market_data.prices.columns) if hasattr(market_data.prices, 'columns') else []
            gpu_features = {}
            
            if self.use_gpu_acceleration and symbols:
                gpu_features = await self.gpu_processor.calculate_technical_features_batch(
                    market_data, symbols, 
                    indicators=['returns', 'rsi', 'macd', 'bollinger', 'ema_20', 'ema_50']
                )
                logger.info(f"Pre-calculated GPU features for {len(gpu_features)} symbols")

            # Collect signals from each agent
            for agent_type, agent in self.agents.items():
                try:
                    # Enhance market data with GPU features if available
                    enhanced_market_data = market_data
                    if gpu_features and hasattr(enhanced_market_data, 'metadata'):
                        enhanced_market_data.metadata = getattr(enhanced_market_data, 'metadata', {})
                        enhanced_market_data.metadata['gpu_features'] = gpu_features
                    
                    signals = await agent.generate_signals(enhanced_market_data)
                    logger.debug(f"Agent {agent_type} generated {len(signals)} signals")
                    for signal in signals:
                        signal.metadata["agent_type"] = agent_type
                        signal.metadata["agent_weight"] = self.agent_weights.get(agent_type, 0)
                        logger.debug(f"Signal: {signal.symbol} {signal.direction.value} (confidence: {signal.confidence:.2f})")
                    all_signals.extend(signals)
                    
                    # Update signal history
                    for signal in signals:
                        self.signal_history[signal.symbol].append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signals for {agent_type}: {str(e)}")
                    continue
            
            logger.debug(f"Total signals collected: {len(all_signals)}")
            
            # Apply GPU-accelerated signal optimization
            if self.use_gpu_acceleration and all_signals:
                all_signals = await self.gpu_processor.optimize_signal_timing(
                    all_signals, market_data
                )
                logger.debug(f"Applied GPU signal timing optimization")
            
            # Aggregate signals using ensemble, GPU acceleration, or fallback to basic method
            if self.use_ensemble and self.ensemble_service and self.ensemble_id:
                aggregated = await self._aggregate_signals_with_ensemble(all_signals)
                logger.info(f"Ensemble-aggregated signals: {len(aggregated)}")
            elif self.use_gpu_acceleration and len(all_signals) > 5:
                # Use GPU-accelerated aggregation for larger signal sets
                aggregated = await self.gpu_processor.accelerate_signal_aggregation(
                    all_signals, self.agent_weights, ensemble_method='weighted_average'
                )
                logger.info(f"GPU-accelerated aggregation: {len(aggregated)} signals")
            else:
                aggregated = await self._aggregate_signals(all_signals)
                logger.debug(f"Basic-aggregated signals: {len(aggregated)}")
            
            # Generate explanations for aggregated signals if enabled
            if self.enable_explanations and aggregated:
                await self._generate_signal_explanations(aggregated, market_data)
            
            return aggregated

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            return []
    
    async def _generate_signal_explanations(
        self, 
        signals: List[TradeSignal], 
        market_data: MarketData
    ) -> None:
        """
        Generate explanations for trading signals.
        
        Args:
            signals: List of trading signals to explain
            market_data: Market data used for signal generation
        """
        try:
            for signal in signals:
                # Prepare signal data for explanation
                signal_data = {
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "confidence": signal.confidence,
                    "target_price": signal.target_price,
                    "stop_loss": signal.stop_loss,
                    "agent_type": signal.metadata.get("agent_type"),
                    "agent_weight": signal.metadata.get("agent_weight"),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add market context if available
                if hasattr(market_data, 'prices') and signal.symbol in market_data.prices.columns:
                    recent_prices = market_data.prices[signal.symbol].tail(20)
                    signal_data.update({
                        "current_price": float(recent_prices.iloc[-1]) if not recent_prices.empty else None,
                        "price_change_1d": float((recent_prices.iloc[-1] / recent_prices.iloc[-2] - 1) * 100) 
                                         if len(recent_prices) >= 2 else None,
                        "volatility": float(recent_prices.pct_change().std() * 100) 
                                    if len(recent_prices) > 1 else None
                    })
                
                # Generate explanation asynchronously (non-blocking)
                try:
                    explanation = await self.explainability_service.explain_prediction(
                        model_type="trading_signal",
                        prediction_data=signal_data,
                        method="shap",  # Default to SHAP for real-time explanations
                        symbol=signal.symbol,
                        include_visualization=False  # Skip viz for real-time to save time
                    )
                    
                    # Add explanation to signal metadata
                    signal.metadata.update({
                        "explanation_id": explanation.explanation_id,
                        "key_factors": explanation.feature_importance[:3],  # Top 3 factors
                        "explanation_confidence": explanation.confidence,
                        "explanation_text": explanation.explanation_text[:200] + "..." 
                                          if len(explanation.explanation_text) > 200 
                                          else explanation.explanation_text
                    })
                    
                    logger.debug(f"Generated explanation for {signal.symbol} signal")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate explanation for {signal.symbol}: {e}")
                    # Don't fail the signal generation if explanation fails
                    signal.metadata["explanation_error"] = str(e)
                    
        except Exception as e:
            logger.error(f"Error in explanation generation process: {e}")
            # Don't fail signal generation if explanation process fails
        
    async def update_agent_weights(self, performance_data: Dict[str, pd.DataFrame]) -> None:
        """
        Update agent weights based on performance.
        
        Args:
            performance_data: Performance data for each agent
        """
        performance_scores = {}
        
        # Calculate performance metrics for each agent
        for agent_type, agent in self.agents.items():
            if agent_type in performance_data:
                metrics = await agent.update_metrics(performance_data[agent_type])
                self.performance_metrics[agent_type] = metrics
                
                # Calculate composite performance score
                performance_scores[agent_type] = (
                    metrics.signal_accuracy * 0.3 +
                    metrics.profit_factor * 0.3 +
                    metrics.sharpe_ratio * 0.2 +
                    (1 - metrics.max_drawdown) * 0.2
                )
                
        if performance_scores:
            # Update weights using softmax
            scores = np.array(list(performance_scores.values()))
            exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
            softmax = exp_scores / exp_scores.sum()
            
            # Update weights with smoothing
            smoothing = 0.7  # 70% old weights, 30% new weights
            for agent_type, new_weight in zip(performance_scores.keys(), softmax):
                old_weight = self.agent_weights.get(agent_type, 0)
                self.agent_weights[agent_type] = (
                    old_weight * smoothing + new_weight * (1 - smoothing)
                )
                
    async def _aggregate_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        Aggregate signals from multiple agents.
        
        Args:
            signals: List of signals from all agents
            
        Returns:
            List of aggregated signals
        """
        if not signals:
            logger.debug("No signals to aggregate")
            return []
            
        # Group signals by symbol
        signals_by_symbol = defaultdict(list)
        for signal in signals:
            signals_by_symbol[signal.symbol].append(signal)
        
        logger.debug(f"Aggregating signals for {len(signals_by_symbol)} symbols")
        aggregated_signals = []
        
        for symbol, symbol_signals in signals_by_symbol.items():
            logger.debug(f"Processing {len(symbol_signals)} signals for {symbol}")
            
            # Calculate weighted signal strength for each direction
            direction_strength = defaultdict(float)
            total_weight = 0
            
            for signal in symbol_signals:
                agent_type = signal.metadata.get("agent_type")
                agent_weight = self.agent_weights.get(agent_type, 0)
                
                # Weight the signal by agent weight and confidence
                weighted_strength = signal.confidence * agent_weight
                direction_strength[signal.direction] += weighted_strength
                total_weight += agent_weight
                logger.debug(f"Agent {agent_type} signal: direction={signal.direction.value}, confidence={signal.confidence:.2f}, weight={agent_weight:.2f}")
                
            if total_weight > 0:
                # Normalize strengths
                direction_strength = {
                    k: v / total_weight
                    for k, v in direction_strength.items()
                }
                logger.debug(f"Normalized direction strengths: {direction_strength}")
                
                # Find dominant direction
                dominant_direction = max(
                    direction_strength.items(),
                    key=lambda x: x[1]
                )
                logger.debug(f"Dominant direction: {dominant_direction[0].value} with strength {dominant_direction[1]:.2f}")
                
                if dominant_direction[1] >= 0.05:  # Very low threshold since we only have one agent with 0.15 weight
                    logger.debug(f"Signal passed consensus threshold")
                    # Calculate aggregate target price and stop loss
                    prices = [(s.target_price, s.stop_loss) for s in symbol_signals
                             if s.direction == dominant_direction[0]]
                    
                    target_prices = [p[0] for p in prices if p[0] is not None]
                    stop_losses = [p[1] for p in prices if p[1] is not None]
                    
                    target_price = np.median(target_prices) if target_prices else None
                    stop_loss = np.median(stop_losses) if stop_losses else None
                    
                    # Create aggregated signal
                    aggregated_signals.append(TradeSignal(
                        agent_id="agent_manager",
                        symbol=symbol,
                        direction=dominant_direction[0],
                        confidence=dominant_direction[1],
                        timestamp=datetime.now(),
                        target_price=target_price,
                        stop_loss=stop_loss,
                        metadata={
                            "strategy": "multi_agent",
                            "contributing_agents": len(symbol_signals),
                            "direction_strength": dict(direction_strength),
                            "agent_signals": [
                                {
                                    "agent_type": s.metadata.get("agent_type"),
                                    "confidence": s.confidence,
                                    "direction": s.direction
                                }
                                for s in symbol_signals
                            ]
                        }
                    ))
                    
        return aggregated_signals
        
    def get_agent_performance(self) -> Dict[str, AgentMetrics]:
        """
        Get performance metrics for all agents.
        
        Returns:
            Dictionary of agent performance metrics
        """
        return self.performance_metrics
        
    def get_signal_history(self, symbol: str) -> List[TradeSignal]:
        """
        Get signal history for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of historical signals
        """
        return self.signal_history.get(symbol, [])
        
    async def validate_signal(self, signal: TradeSignal) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        validations = []
        
        # Get contributing agents
        contributing_agents = signal.metadata.get("agent_signals", [])
        
        # Validate with each contributing agent
        for agent_signal in contributing_agents:
            agent_type = agent_signal.get("agent_type")
            if agent_type in self.agents:
                agent = self.agents[agent_type]
                validation = await agent.validate_signal(signal)
                validations.append(validation)
                
        # Signal is valid if majority of agents validate it
        return sum(validations) > len(validations) / 2 if validations else False
    
    async def _initialize_ensemble(self) -> None:
        """Initialize ensemble configuration and register agents."""
        try:
            # Create ensemble configuration
            ensemble_config = {
                "aggregation_method": "adaptive",
                "confidence_threshold": 0.6,
                "outlier_detection": True,
                "temporal_weighting": True,
                "performance_tracking": True
            }
            
            # Create ensemble
            self.ensemble_id = self.ensemble_service.create_ensemble(
                name="trading_agent_ensemble",
                ensemble_type="voting",  # Use voting ensemble as default
                config=ensemble_config
            )
            
            # Register each agent with ensemble service
            for agent_type in self.agents.keys():
                agent_id = self.ensemble_service.register_agent(
                    name=f"{agent_type}_agent",
                    agent_type=agent_type,
                    config={"weight": self.agent_weights.get(agent_type, 0)}
                )
                self.agent_registry[agent_type] = agent_id
                
            logger.info(f"Ensemble initialized with ID: {self.ensemble_id}")
            logger.info(f"Registered {len(self.agent_registry)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {str(e)}")
            self.use_ensemble = False
    
    async def _aggregate_signals_with_ensemble(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Aggregate signals using ensemble methods."""
        if not signals:
            return []
            
        try:
            # Group signals by symbol
            signals_by_symbol = defaultdict(list)
            for signal in signals:
                signals_by_symbol[signal.symbol].append(signal)
            
            aggregated_signals = []
            
            for symbol, symbol_signals in signals_by_symbol.items():
                # Convert to ensemble signal format
                ensemble_signals = []
                for signal in symbol_signals:
                    agent_type = signal.metadata.get("agent_type")
                    agent_id = self.agent_registry.get(agent_type)
                    
                    if agent_id:
                        ensemble_signal = AgentSignalCreate(
                            agent_id=agent_id,
                            signal=signal.direction.value,
                            confidence=signal.confidence,
                            metadata={
                                "symbol": signal.symbol,
                                "target_price": signal.target_price,
                                "stop_loss": signal.stop_loss,
                                "agent_type": agent_type
                            }
                        )
                        ensemble_signals.append(ensemble_signal)
                
                if ensemble_signals:
                    # Get ensemble prediction
                    ensemble_result = self.ensemble_service.get_ensemble_prediction(
                        self.ensemble_id,
                        ensemble_signals
                    )
                    
                    # Convert back to TradeSignal format
                    if ensemble_result.confidence >= 0.5:  # Minimum confidence threshold
                        # Calculate aggregate target price and stop loss
                        prices = [s.metadata.get("target_price", 0) for s in symbol_signals 
                                if s.metadata.get("target_price")]
                        stops = [s.metadata.get("stop_loss", 0) for s in symbol_signals 
                               if s.metadata.get("stop_loss")]
                        
                        avg_target = np.mean(prices) if prices else None
                        avg_stop = np.mean(stops) if stops else None
                        
                        ensemble_trade_signal = TradeSignal(
                            symbol=symbol,
                            direction=SignalDirection(ensemble_result.signal),
                            confidence=ensemble_result.confidence,
                            target_price=avg_target,
                            stop_loss=avg_stop,
                            reason=f"Ensemble prediction with {len(ensemble_result.contributing_agents)} agents",
                            metadata={
                                "ensemble_prediction": True,
                                "ensemble_id": self.ensemble_id,
                                "contributing_agents": ensemble_result.contributing_agents,
                                "ensemble_weights": ensemble_result.weights,
                                "execution_time_ms": ensemble_result.execution_time_ms,
                                "agent_signals": [
                                    {
                                        "agent_type": s.metadata.get("agent_type"),
                                        "signal": s.direction.value,
                                        "confidence": s.confidence
                                    }
                                    for s in symbol_signals
                                ]
                            }
                        )
                        aggregated_signals.append(ensemble_trade_signal)
                        
                        logger.info(
                            f"Ensemble signal for {symbol}: {ensemble_result.signal} "
                            f"(confidence: {ensemble_result.confidence:.2f}, "
                            f"agents: {len(ensemble_result.contributing_agents)})"
                        )
            
            return aggregated_signals
            
        except Exception as e:
            logger.error(f"Ensemble aggregation failed: {str(e)}")
            # Fallback to basic aggregation
            return await self._aggregate_signals(signals)