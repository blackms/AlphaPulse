"""Service layer for ensemble methods."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
import uuid

from alpha_pulse.ml.ensemble.ensemble_manager import EnsembleManager, AgentSignal, EnsembleSignal
from alpha_pulse.ml.ensemble.voting_classifiers import (
    HardVotingEnsemble, SoftVotingEnsemble, WeightedMajorityVoting
)
from alpha_pulse.ml.ensemble.stacking_methods import StackingEnsemble, HierarchicalStacking
from alpha_pulse.ml.ensemble.boosting_algorithms import (
    AdaptiveBoosting, GradientBoosting, XGBoostEnsemble, LightGBMEnsemble, OnlineBoosting
)
from alpha_pulse.ml.ensemble.signal_aggregation import SignalAggregator, TemporalAggregator, ConsensusAggregator
from alpha_pulse.utils.ensemble_validation import EnsembleValidator, EnsembleMonitor
from alpha_pulse.models.ensemble_model import (
    EnsembleConfig, TradingAgent, AgentSignalRecord, EnsemblePrediction,
    AgentWeight, EnsemblePerformance, AgentSignalCreate, EnsemblePredictionResponse,
    EnsembleOptimizationResponse
)
from alpha_pulse.utils.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class EnsembleService:
    """Service for managing ensemble trading strategies."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.ensemble_managers = {}
        self.temporal_aggregators = {}
        self.ensemble_monitors = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def create_ensemble(self, name: str, ensemble_type: str, config: Dict[str, Any]) -> str:
        """Create a new ensemble configuration."""
        # Generate unique ID
        ensemble_id = str(uuid.uuid4())
        
        # Create database record
        ensemble_config = EnsembleConfig(
            id=ensemble_id,
            name=name,
            ensemble_type=ensemble_type,
            config=config
        )
        self.db.add(ensemble_config)
        self.db.commit()
        
        # Initialize ensemble manager
        self._initialize_ensemble_manager(ensemble_id, ensemble_type, config)
        
        logger.info(f"Created ensemble {name} with ID {ensemble_id}")
        return ensemble_id
        
    def _initialize_ensemble_manager(self, ensemble_id: str, ensemble_type: str, 
                                   config: Dict[str, Any]) -> None:
        """Initialize ensemble manager with specified type."""
        manager = EnsembleManager(config)
        
        # Add appropriate ensemble methods
        if ensemble_type == 'voting':
            manager.add_ensemble('hard_voting', HardVotingEnsemble(config))
            manager.add_ensemble('soft_voting', SoftVotingEnsemble(config))
            manager.add_ensemble('weighted_majority', WeightedMajorityVoting(config))
        elif ensemble_type == 'stacking':
            manager.add_ensemble('stacking', StackingEnsemble(config))
            manager.add_ensemble('hierarchical', HierarchicalStacking(config))
        elif ensemble_type == 'boosting':
            manager.add_ensemble('adaptive', AdaptiveBoosting(config))
            manager.add_ensemble('gradient', GradientBoosting(config))
            manager.add_ensemble('xgboost', XGBoostEnsemble(config))
            manager.add_ensemble('lightgbm', LightGBMEnsemble(config))
            manager.add_ensemble('online', OnlineBoosting(config))
            
        self.ensemble_managers[ensemble_id] = manager
        
        # Initialize aggregators and monitors
        self.temporal_aggregators[ensemble_id] = TemporalAggregator(config)
        self.ensemble_monitors[ensemble_id] = EnsembleMonitor(config)
        
    def register_agent(self, name: str, agent_type: str, config: Dict[str, Any]) -> str:
        """Register a new trading agent."""
        agent_id = str(uuid.uuid4())
        
        agent = TradingAgent(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            config=config
        )
        self.db.add(agent)
        self.db.commit()
        
        # Register with all ensemble managers
        for manager in self.ensemble_managers.values():
            manager.register_agent(agent_id, {'name': name, 'type': agent_type})
            
        logger.info(f"Registered agent {name} with ID {agent_id}")
        return agent_id
        
    def activate_agent(self, agent_id: str, ensemble_id: Optional[str] = None) -> None:
        """Activate agent for ensemble participation."""
        # Update database
        agent = self.db.query(TradingAgent).filter_by(id=agent_id).first()
        if agent:
            agent.status = 'active'
            self.db.commit()
            
        # Activate in ensemble managers
        if ensemble_id and ensemble_id in self.ensemble_managers:
            self.ensemble_managers[ensemble_id].activate_agent(agent_id)
        else:
            # Activate in all ensembles
            for manager in self.ensemble_managers.values():
                manager.activate_agent(agent_id)
                
    async def generate_ensemble_prediction(self, ensemble_id: str, 
                                         agent_signals: List[AgentSignalCreate]) -> EnsemblePredictionResponse:
        """Generate ensemble prediction from agent signals."""
        if ensemble_id not in self.ensemble_managers:
            raise ValueError(f"Ensemble {ensemble_id} not found")
            
        manager = self.ensemble_managers[ensemble_id]
        
        # Convert to internal signal format
        signals = []
        for signal_data in agent_signals:
            # Record signal in database
            signal_record = AgentSignalRecord(
                agent_id=signal_data.agent_id,
                timestamp=datetime.now(),
                signal=signal_data.signal,
                confidence=signal_data.confidence,
                metadata=signal_data.metadata
            )
            self.db.add(signal_record)
            
            # Create signal object
            signal = AgentSignal(
                agent_id=signal_data.agent_id,
                timestamp=datetime.now(),
                signal=signal_data.signal,
                confidence=signal_data.confidence,
                metadata=signal_data.metadata or {}
            )
            signals.append(signal)
            
            # Update temporal aggregator
            self.temporal_aggregators[ensemble_id].add_signal(
                signal_data.agent_id, signal_data.signal, 
                signal_data.confidence, datetime.now()
            )
            
        # Generate ensemble prediction
        start_time = datetime.now()
        ensemble_signal = manager.generate_ensemble_signal(signals)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Record prediction
        prediction_record = EnsemblePrediction(
            ensemble_id=ensemble_id,
            timestamp=ensemble_signal.timestamp,
            signal=ensemble_signal.signal,
            confidence=ensemble_signal.confidence,
            contributing_agents=ensemble_signal.contributing_agents,
            weights=ensemble_signal.weights,
            metadata=ensemble_signal.metadata,
            execution_time_ms=execution_time
        )
        self.db.add(prediction_record)
        self.db.commit()
        
        # Update monitor
        self.ensemble_monitors[ensemble_id].update(ensemble_signal)
        
        return EnsemblePredictionResponse(
            id=prediction_record.id,
            ensemble_id=ensemble_id,
            timestamp=ensemble_signal.timestamp,
            signal=ensemble_signal.signal,
            confidence=ensemble_signal.confidence,
            contributing_agents=ensemble_signal.contributing_agents,
            weights=ensemble_signal.weights,
            metadata=ensemble_signal.metadata,
            execution_time_ms=execution_time
        )
    
    async def get_ensemble_prediction(
        self,
        ensemble_id: str,
        agent_signals: List[AgentSignalCreate | Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnsemblePredictionResponse:
        """
        Backward-compatible wrapper for generating ensemble predictions.
        
        Accepts either AgentSignalCreate models or raw dictionaries, normalizes
        them, and delegates to generate_ensemble_prediction.
        """
        normalized_signals: List[AgentSignalCreate] = []
        for signal in agent_signals:
            if isinstance(signal, AgentSignalCreate):
                normalized_signals.append(signal)
            else:
                normalized_signals.append(AgentSignalCreate(**signal))
        
        return await self.generate_ensemble_prediction(ensemble_id, normalized_signals)
        
    def update_agent_performance(self, agent_id: str, performance: float) -> None:
        """Update agent performance metrics."""
        # Update database
        agent = self.db.query(TradingAgent).filter_by(id=agent_id).first()
        if agent:
            agent.performance_score = performance
            agent.last_signal_at = datetime.now()
            agent.signal_count += 1
            self.db.commit()
            
        # Update ensemble managers
        for manager in self.ensemble_managers.values():
            manager.update_performance(agent_id, performance)
            
    def optimize_ensemble_weights(self, ensemble_id: str, 
                                lookback_days: int = 30) -> EnsembleOptimizationResponse:
        """Optimize ensemble weights based on historical performance."""
        if ensemble_id not in self.ensemble_managers:
            raise ValueError(f"Ensemble {ensemble_id} not found")
            
        # Get historical data
        start_date = datetime.now() - timedelta(days=lookback_days)
        
        # Get historical signals and outcomes
        signals = self.db.query(AgentSignalRecord).filter(
            AgentSignalRecord.timestamp >= start_date
        ).all()
        
        predictions = self.db.query(EnsemblePrediction).filter(
            and_(
                EnsemblePrediction.ensemble_id == ensemble_id,
                EnsemblePrediction.timestamp >= start_date
            )
        ).all()
        
        # Convert to format for optimization
        signal_objects = [
            AgentSignal(
                agent_id=s.agent_id,
                timestamp=s.timestamp,
                signal=s.signal,
                confidence=s.confidence,
                metadata=s.metadata or {}
            )
            for s in signals
        ]
        
        # Mock outcomes for demonstration (in production, use actual market outcomes)
        outcomes = np.random.randn(len(predictions))
        
        # Store old weights
        manager = self.ensemble_managers[ensemble_id]
        old_weights = manager.agent_weights.copy()
        
        # Optimize weights
        manager.optimize_ensemble_weights(pd.DataFrame())  # Placeholder
        
        # Calculate expected improvement
        expected_improvement = 0.1  # Placeholder
        
        # Record weight updates
        for agent_id, new_weight in manager.agent_weights.items():
            if agent_id in old_weights:
                weight_record = AgentWeight(
                    agent_id=agent_id,
                    ensemble_id=ensemble_id,
                    weight=new_weight,
                    reason='optimization'
                )
                self.db.add(weight_record)
                
        self.db.commit()
        
        return EnsembleOptimizationResponse(
            ensemble_id=ensemble_id,
            old_weights=old_weights,
            new_weights=manager.agent_weights,
            expected_improvement=expected_improvement,
            optimization_metrics={
                'lookback_days': lookback_days,
                'signal_count': len(signals),
                'prediction_count': len(predictions)
            }
        )
        
    def get_ensemble_performance(self, ensemble_id: str, 
                               days: int = 30) -> Dict[str, Any]:
        """Get ensemble performance metrics."""
        start_date = datetime.now() - timedelta(days=days)
        
        # Query performance records
        performance_records = self.db.query(EnsemblePerformance).filter(
            and_(
                EnsemblePerformance.ensemble_id == ensemble_id,
                EnsemblePerformance.timestamp >= start_date
            )
        ).order_by(EnsemblePerformance.timestamp.desc()).all()
        
        if not performance_records:
            return {}
            
        # Get latest performance
        latest = performance_records[0]
        
        # Calculate trends
        metrics_trend = {}
        if len(performance_records) > 1:
            for metric in ['accuracy', 'sharpe_ratio', 'hit_rate']:
                values = [getattr(p, metric) for p in performance_records if getattr(p, metric) is not None]
                if values:
                    metrics_trend[f"{metric}_trend"] = values[0] - values[-1]
                    
        # Get monitor summary
        monitor_summary = {}
        if ensemble_id in self.ensemble_monitors:
            monitor_summary = self.ensemble_monitors[ensemble_id].get_performance_summary()
            
        return {
            'latest_metrics': {
                'accuracy': latest.accuracy,
                'sharpe_ratio': latest.sharpe_ratio,
                'hit_rate': latest.hit_rate,
                'avg_confidence': latest.avg_confidence,
                'signal_count': latest.signal_count,
                'profit_loss': latest.profit_loss,
                'max_drawdown': latest.max_drawdown
            },
            'trends': metrics_trend,
            'monitor_summary': monitor_summary,
            'period_days': days
        }
        
    def get_agent_rankings(self, ensemble_id: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Get agents ranked by performance."""
        # Query all agents
        agents = self.db.query(TradingAgent).filter_by(status='active').all()
        
        rankings = []
        for agent in agents:
            # Get recent performance
            recent_signals = self.db.query(func.count(AgentSignalRecord.id)).filter(
                and_(
                    AgentSignalRecord.agent_id == agent.id,
                    AgentSignalRecord.timestamp >= datetime.now() - timedelta(days=7)
                )
            ).scalar()
            
            # Get ensemble weight if specified
            weight = 0.0
            if ensemble_id and ensemble_id in self.ensemble_managers:
                weight = self.ensemble_managers[ensemble_id].agent_weights.get(agent.id, 0.0)
                
            rankings.append((
                agent.id,
                {
                    'name': agent.name,
                    'type': agent.agent_type,
                    'performance_score': agent.performance_score,
                    'recent_signals': recent_signals,
                    'ensemble_weight': weight,
                    'status': agent.status
                }
            ))
            
        # Sort by performance score
        rankings.sort(key=lambda x: x[1]['performance_score'], reverse=True)
        
        return rankings
        
    def backtest_ensemble(self, ensemble_id: str, start_date: datetime, 
                         end_date: datetime, initial_capital: float = 100000) -> Dict[str, Any]:
        """Backtest ensemble strategy."""
        if ensemble_id not in self.ensemble_managers:
            raise ValueError(f"Ensemble {ensemble_id} not found")
            
        # Get historical predictions
        predictions = self.db.query(EnsemblePrediction).filter(
            and_(
                EnsemblePrediction.ensemble_id == ensemble_id,
                EnsemblePrediction.timestamp >= start_date,
                EnsemblePrediction.timestamp <= end_date
            )
        ).order_by(EnsemblePrediction.timestamp).all()
        
        if not predictions:
            return {'error': 'No predictions found for specified period'}
            
        # Simple backtest simulation
        capital = initial_capital
        positions = 0.0
        trades = []
        equity_curve = [capital]
        
        for pred in predictions:
            # Simple position sizing based on signal strength
            position_size = capital * 0.1 * abs(pred.signal)
            
            if pred.signal > 0.1 and positions <= 0:
                # Buy signal
                positions = position_size
                trades.append({
                    'type': 'buy',
                    'size': position_size,
                    'timestamp': pred.timestamp
                })
            elif pred.signal < -0.1 and positions >= 0:
                # Sell signal
                positions = -position_size
                trades.append({
                    'type': 'sell',
                    'size': position_size,
                    'timestamp': pred.timestamp
                })
                
            # Mock returns (in production, use actual market data)
            returns = np.random.randn() * 0.02
            capital += positions * returns
            equity_curve.append(capital)
            
        # Calculate metrics
        returns_array = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = PerformanceMetrics.calculate_comprehensive_metrics(
            returns_array,
            np.array(equity_curve)
        )
        
        return {
            'period': {
                'start': start_date,
                'end': end_date
            },
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital,
            'total_trades': len(trades),
            'metrics': metrics,
            'equity_curve': equity_curve[-100:]  # Last 100 points
        }
        
    async def validate_ensemble(self, ensemble_id: str) -> Dict[str, Any]:
        """Validate ensemble performance and configuration."""
        if ensemble_id not in self.ensemble_managers:
            raise ValueError(f"Ensemble {ensemble_id} not found")
            
        # Get ensemble config
        config = self.db.query(EnsembleConfig).filter_by(id=ensemble_id).first()
        if not config:
            return {'error': 'Ensemble configuration not found'}
            
        # Initialize validator
        validator = EnsembleValidator(config.config)
        
        # Get recent signals for validation
        signals = self.db.query(AgentSignalRecord).filter(
            AgentSignalRecord.timestamp >= datetime.now() - timedelta(days=30)
        ).all()
        
        # Convert to signal objects
        signal_objects = [
            AgentSignal(
                agent_id=s.agent_id,
                timestamp=s.timestamp,
                signal=s.signal,
                confidence=s.confidence,
                metadata=s.metadata or {}
            )
            for s in signals
        ]
        
        # Mock outcomes for validation
        outcomes = np.random.randn(len(signals))
        
        # Validate signal quality
        ensemble_signals = []
        predictions = self.db.query(EnsemblePrediction).filter(
            and_(
                EnsemblePrediction.ensemble_id == ensemble_id,
                EnsemblePrediction.timestamp >= datetime.now() - timedelta(days=30)
            )
        ).all()
        
        for pred in predictions:
            ensemble_signals.append(
                EnsembleSignal(
                    timestamp=pred.timestamp,
                    signal=pred.signal,
                    confidence=pred.confidence,
                    contributing_agents=pred.contributing_agents,
                    weights=pred.weights,
                    metadata=pred.metadata
                )
            )
            
        quality_metrics = validator.validate_signal_quality(ensemble_signals, outcomes)
        
        # Validate agent contributions
        manager = self.ensemble_managers[ensemble_id]
        ensemble_method = manager.ensembles.get(config.config.get('default_method', 'weighted_average'))
        
        if ensemble_method:
            agent_metrics = validator.validate_agent_contributions(
                ensemble_method, signal_objects, outcomes
            )
        else:
            agent_metrics = {}
            
        return {
            'ensemble_id': ensemble_id,
            'validation_date': datetime.now(),
            'signal_quality': quality_metrics,
            'agent_contributions': agent_metrics,
            'config_valid': True,
            'recommendations': self._generate_recommendations(quality_metrics, agent_metrics)
        }
        
    def _generate_recommendations(self, quality_metrics: Dict[str, Any], 
                                agent_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check signal quality
        if quality_metrics.get('accuracy', 0) < 0.5:
            recommendations.append("Consider retraining ensemble - accuracy below 50%")
            
        if quality_metrics.get('signal_stability', 1) < 0.7:
            recommendations.append("Signal stability low - consider smoothing or filtering")
            
        if quality_metrics.get('warnings'):
            for warning in quality_metrics['warnings']:
                recommendations.append(f"Warning: {warning}")
                
        # Check agent contributions
        low_performing_agents = [
            agent_id for agent_id, metrics in agent_metrics.items()
            if metrics.get('accuracy', 0) < 0.45
        ]
        
        if low_performing_agents:
            recommendations.append(
                f"Consider removing or retraining agents: {', '.join(low_performing_agents[:3])}"
            )
            
        return recommendations
        
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data from database."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Delete old signals
        deleted_signals = self.db.query(AgentSignalRecord).filter(
            AgentSignalRecord.timestamp < cutoff_date
        ).delete()
        
        # Delete old predictions
        deleted_predictions = self.db.query(EnsemblePrediction).filter(
            EnsemblePrediction.timestamp < cutoff_date
        ).delete()
        
        # Delete old performance records
        deleted_performance = self.db.query(EnsemblePerformance).filter(
            EnsemblePerformance.timestamp < cutoff_date
        ).delete()
        
        self.db.commit()
        
        return {
            'deleted_signals': deleted_signals,
            'deleted_predictions': deleted_predictions,
            'deleted_performance': deleted_performance,
            'cutoff_date': cutoff_date
        }
