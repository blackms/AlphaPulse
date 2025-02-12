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


class AgentManager:
    """
    Manages multiple trading agents and coordinates their signals.
    Provides a unified interface for the risk management system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
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

            # Collect signals from each agent
            for agent_type, agent in self.agents.items():
                try:
                    signals = await agent.generate_signals(market_data)
                    for signal in signals:
                        signal.metadata["agent_type"] = agent_type
                        signal.metadata["agent_weight"] = self.agent_weights.get(agent_type, 0)
                    all_signals.extend(signals)
                    
                    # Update signal history
                    for signal in signals:
                        self.signal_history[signal.symbol].append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signals for {agent_type}: {str(e)}")
                    continue
                    
            # Aggregate signals
            return await self._aggregate_signals(all_signals)

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            return []
        
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
            return []
            
        # Group signals by symbol
        signals_by_symbol = defaultdict(list)
        for signal in signals:
            signals_by_symbol[signal.symbol].append(signal)
            
        aggregated_signals = []
        
        for symbol, symbol_signals in signals_by_symbol.items():
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
                
            if total_weight > 0:
                # Normalize strengths
                direction_strength = {
                    k: v / total_weight
                    for k, v in direction_strength.items()
                }
                
                # Find dominant direction
                dominant_direction = max(
                    direction_strength.items(),
                    key=lambda x: x[1]
                )
                
                if dominant_direction[1] >= 0.2:  # Lower consensus threshold since we only have one agent
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