"""Ensemble manager for combining trading agent signals."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from collections import defaultdict

from alpha_pulse.ml.ensemble.signal_aggregation import SignalAggregator
from alpha_pulse.utils.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    """Trading signal from an individual agent."""
    agent_id: str
    timestamp: datetime
    signal: float  # -1 to 1 (sell to buy)
    confidence: float  # 0 to 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class EnsembleSignal:
    """Combined signal from ensemble."""
    timestamp: datetime
    signal: float
    confidence: float
    contributing_agents: List[str]
    weights: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal,
            'confidence': self.confidence,
            'contributing_agents': self.contributing_agents,
            'weights': self.weights,
            'metadata': self.metadata
        }


class BaseEnsemble(ABC):
    """Base class for ensemble methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train the ensemble on historical data."""
        pass
        
    @abstractmethod
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate ensemble prediction from agent signals."""
        pass
        
    @abstractmethod
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update agent weights based on performance."""
        pass


class EnsembleManager:
    """Manages multiple ensemble methods and agent lifecycle."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensembles: Dict[str, BaseEnsemble] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.performance_tracker = PerformanceTracker()
        self.signal_aggregator = SignalAggregator(config.get('aggregation', {}))
        self.max_agents = config.get('max_agents', 20)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        
    def register_agent(self, agent_id: str, metadata: Dict[str, Any]) -> None:
        """Register a new trading agent."""
        self.agent_registry[agent_id] = {
            'metadata': metadata,
            'registered_at': datetime.now(),
            'status': 'inactive',
            'performance_score': 0.0,
            'signal_count': 0
        }
        logger.info(f"Registered agent: {agent_id}")
        
    def activate_agent(self, agent_id: str) -> None:
        """Activate an agent for ensemble participation."""
        if agent_id not in self.agent_registry:
            raise ValueError(f"Agent {agent_id} not registered")
            
        if len(self.active_agents) >= self.max_agents:
            # Remove lowest performing agent
            self._retire_worst_agent()
            
        self.active_agents[agent_id] = self.agent_registry[agent_id]
        self.active_agents[agent_id]['status'] = 'active'
        self.active_agents[agent_id]['activated_at'] = datetime.now()
        logger.info(f"Activated agent: {agent_id}")
        
    def deactivate_agent(self, agent_id: str) -> None:
        """Deactivate an agent."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
            self.agent_registry[agent_id]['status'] = 'inactive'
            logger.info(f"Deactivated agent: {agent_id}")
            
    def add_ensemble(self, name: str, ensemble: BaseEnsemble) -> None:
        """Add an ensemble method."""
        self.ensembles[name] = ensemble
        logger.info(f"Added ensemble method: {name}")
        
    def collect_signals(self, agents: Dict[str, Any], market_data: pd.DataFrame) -> List[AgentSignal]:
        """Collect signals from all active agents in parallel."""
        signals = []
        futures = []
        
        for agent_id, agent in agents.items():
            if agent_id in self.active_agents:
                future = self.executor.submit(
                    self._get_agent_signal, agent_id, agent, market_data
                )
                futures.append(future)
                
        for future in as_completed(futures):
            try:
                signal = future.result(timeout=5.0)
                if signal and signal.confidence >= self.min_confidence:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error collecting signal: {e}")
                
        return signals
        
    def _get_agent_signal(self, agent_id: str, agent: Any, market_data: pd.DataFrame) -> Optional[AgentSignal]:
        """Get signal from a single agent."""
        try:
            # Call agent's predict method
            prediction = agent.predict(market_data)
            
            return AgentSignal(
                agent_id=agent_id,
                timestamp=datetime.now(),
                signal=prediction.get('signal', 0.0),
                confidence=prediction.get('confidence', 0.0),
                metadata=prediction.get('metadata', {})
            )
        except Exception as e:
            logger.error(f"Error getting signal from agent {agent_id}: {e}")
            return None
            
    def generate_ensemble_signal(self, signals: List[AgentSignal], 
                               method: str = 'weighted_average') -> EnsembleSignal:
        """Generate ensemble signal from agent signals."""
        if not signals:
            return EnsembleSignal(
                timestamp=datetime.now(),
                signal=0.0,
                confidence=0.0,
                contributing_agents=[],
                weights={},
                metadata={'reason': 'no_signals'}
            )
            
        if method in self.ensembles:
            ensemble = self.ensembles[method]
            return ensemble.predict(signals)
        else:
            # Default weighted average
            return self._weighted_average_ensemble(signals)
            
    def _weighted_average_ensemble(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Simple weighted average ensemble."""
        weights = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for signal in signals:
            weight = self.agent_weights.get(signal.agent_id, 1.0) * signal.confidence
            weights[signal.agent_id] = weight
            weighted_sum += signal.signal * weight
            weight_sum += weight
            
        ensemble_signal = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        ensemble_confidence = np.mean([s.confidence for s in signals])
        
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=ensemble_signal,
            confidence=ensemble_confidence,
            contributing_agents=[s.agent_id for s in signals],
            weights=weights,
            metadata={
                'method': 'weighted_average',
                'signal_count': len(signals)
            }
        )
        
    def update_performance(self, agent_id: str, performance: float) -> None:
        """Update agent performance metrics."""
        if agent_id in self.agent_registry:
            self.performance_history[agent_id].append(performance)
            # Update rolling performance score
            recent_performance = self.performance_history[agent_id][-20:]
            self.agent_registry[agent_id]['performance_score'] = np.mean(recent_performance)
            
    def _retire_worst_agent(self) -> None:
        """Retire the worst performing active agent."""
        if not self.active_agents:
            return
            
        worst_agent = min(
            self.active_agents.keys(),
            key=lambda x: self.agent_registry[x]['performance_score']
        )
        
        self.deactivate_agent(worst_agent)
        logger.info(f"Retired worst performing agent: {worst_agent}")
        
    def get_agent_rankings(self) -> List[Tuple[str, float]]:
        """Get agents ranked by performance."""
        rankings = [
            (agent_id, data['performance_score'])
            for agent_id, data in self.agent_registry.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
        
    def get_ensemble_diversity(self, signals: List[AgentSignal]) -> float:
        """Calculate diversity of ensemble predictions."""
        if len(signals) < 2:
            return 0.0
            
        signal_values = [s.signal for s in signals]
        return np.std(signal_values)
        
    def optimize_ensemble_weights(self, historical_data: pd.DataFrame) -> None:
        """Optimize ensemble weights based on historical performance."""
        for name, ensemble in self.ensembles.items():
            if hasattr(ensemble, 'optimize_weights'):
                ensemble.optimize_weights(historical_data)
                logger.info(f"Optimized weights for ensemble: {name}")


class PerformanceTracker:
    """Track and analyze ensemble performance."""
    
    def __init__(self):
        self.predictions: List[Dict[str, Any]] = []
        self.outcomes: List[float] = []
        self.metrics_history: List[Dict[str, float]] = []
        
    def record_prediction(self, ensemble_signal: EnsembleSignal, 
                         actual_outcome: Optional[float] = None) -> None:
        """Record ensemble prediction and outcome."""
        self.predictions.append(ensemble_signal.to_dict())
        if actual_outcome is not None:
            self.outcomes.append(actual_outcome)
            
    def calculate_metrics(self, window: int = 100) -> Dict[str, float]:
        """Calculate performance metrics over recent window."""
        if len(self.predictions) < window or len(self.outcomes) < window:
            return {}
            
        recent_predictions = self.predictions[-window:]
        recent_outcomes = self.outcomes[-window:]
        
        pred_signals = [p['signal'] for p in recent_predictions]
        
        metrics = {
            'accuracy': self._calculate_accuracy(pred_signals, recent_outcomes),
            'sharpe_ratio': self._calculate_sharpe(pred_signals, recent_outcomes),
            'hit_rate': self._calculate_hit_rate(pred_signals, recent_outcomes),
            'avg_confidence': np.mean([p['confidence'] for p in recent_predictions])
        }
        
        self.metrics_history.append(metrics)
        return metrics
        
    def _calculate_accuracy(self, predictions: List[float], outcomes: List[float]) -> float:
        """Calculate directional accuracy."""
        correct = sum(
            1 for p, o in zip(predictions, outcomes)
            if (p > 0 and o > 0) or (p < 0 and o < 0)
        )
        return correct / len(predictions)
        
    def _calculate_sharpe(self, predictions: List[float], outcomes: List[float]) -> float:
        """Calculate Sharpe ratio of predictions."""
        returns = [p * o for p, o in zip(predictions, outcomes)]
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
        
    def _calculate_hit_rate(self, predictions: List[float], outcomes: List[float]) -> float:
        """Calculate hit rate (profitable predictions)."""
        profitable = sum(1 for p, o in zip(predictions, outcomes) if p * o > 0)
        return profitable / len(predictions)