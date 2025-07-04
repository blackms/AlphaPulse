"""Comprehensive tests for ensemble methods."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha_pulse.ml.ensemble.ensemble_manager import (
    EnsembleManager, AgentSignal, EnsembleSignal, PerformanceTracker
)
from alpha_pulse.ml.ensemble.voting_classifiers import (
    HardVotingEnsemble, SoftVotingEnsemble, WeightedMajorityVoting
)
from alpha_pulse.ml.ensemble.stacking_methods import (
    StackingEnsemble, HierarchicalStacking
)
from alpha_pulse.ml.ensemble.boosting_algorithms import (
    AdaptiveBoosting, GradientBoosting, XGBoostEnsemble, LightGBMEnsemble, OnlineBoosting
)
from alpha_pulse.ml.ensemble.signal_aggregation import (
    SignalAggregator, TemporalAggregator, ConsensusAggregator
)
from alpha_pulse.utils.ensemble_validation import EnsembleValidator, EnsembleMonitor
from alpha_pulse.services.ensemble_service import EnsembleService
from alpha_pulse.models.ensemble_model import Base, AgentSignalCreate


@pytest.fixture
def sample_signals():
    """Generate sample agent signals for testing."""
    signals = []
    agents = ['agent1', 'agent2', 'agent3', 'agent4', 'agent5']
    
    for i in range(100):
        for agent in agents:
            signal = AgentSignal(
                agent_id=agent,
                timestamp=datetime.now() + timedelta(seconds=i),
                signal=np.random.uniform(-1, 1),
                confidence=np.random.uniform(0.5, 1),
                metadata={'iteration': i}
            )
            signals.append(signal)
            
    return signals


@pytest.fixture
def sample_outcomes():
    """Generate sample outcomes for testing."""
    return np.random.randn(100) * 0.1


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestEnsembleManager:
    """Test ensemble manager functionality."""
    
    def test_initialization(self):
        """Test ensemble manager initialization."""
        config = {'max_agents': 10, 'min_confidence': 0.6}
        manager = EnsembleManager(config)
        
        assert manager.max_agents == 10
        assert manager.min_confidence == 0.6
        assert len(manager.ensembles) == 0
        assert len(manager.active_agents) == 0
        
    def test_agent_registration(self):
        """Test agent registration."""
        manager = EnsembleManager({})
        
        manager.register_agent('agent1', {'type': 'technical'})
        assert 'agent1' in manager.agent_registry
        assert manager.agent_registry['agent1']['status'] == 'inactive'
        
    def test_agent_activation(self):
        """Test agent activation and deactivation."""
        manager = EnsembleManager({'max_agents': 2})
        
        # Register and activate agents
        manager.register_agent('agent1', {})
        manager.register_agent('agent2', {})
        manager.register_agent('agent3', {})
        
        manager.activate_agent('agent1')
        manager.activate_agent('agent2')
        
        assert len(manager.active_agents) == 2
        assert 'agent1' in manager.active_agents
        assert 'agent2' in manager.active_agents
        
        # Try to activate third agent (should remove worst)
        manager.activate_agent('agent3')
        assert len(manager.active_agents) == 2
        
    def test_ensemble_signal_generation(self, sample_signals):
        """Test ensemble signal generation."""
        manager = EnsembleManager({})
        
        # Register agents
        for agent_id in ['agent1', 'agent2', 'agent3']:
            manager.register_agent(agent_id, {})
            manager.activate_agent(agent_id)
            
        # Generate ensemble signal
        signals = sample_signals[:3]  # Use first 3 signals
        ensemble_signal = manager.generate_ensemble_signal(signals)
        
        assert isinstance(ensemble_signal, EnsembleSignal)
        assert -1 <= ensemble_signal.signal <= 1
        assert 0 <= ensemble_signal.confidence <= 1
        assert len(ensemble_signal.contributing_agents) == 3


class TestVotingClassifiers:
    """Test voting ensemble methods."""
    
    def test_hard_voting(self, sample_signals, sample_outcomes):
        """Test hard voting ensemble."""
        config = {'vote_threshold': 0.1, 'tie_breaker': 'confidence'}
        ensemble = HardVotingEnsemble(config)
        
        # Fit ensemble
        ensemble.fit(sample_signals[:50], sample_outcomes[:10])
        assert ensemble.is_fitted
        
        # Make prediction
        test_signals = [
            AgentSignal('agent1', datetime.now(), 0.8, 0.9, {}),
            AgentSignal('agent2', datetime.now(), 0.7, 0.8, {}),
            AgentSignal('agent3', datetime.now(), -0.3, 0.7, {})
        ]
        
        prediction = ensemble.predict(test_signals)
        assert isinstance(prediction, EnsembleSignal)
        assert prediction.signal > 0  # Majority vote should be positive
        
    def test_soft_voting(self, sample_signals, sample_outcomes):
        """Test soft voting ensemble."""
        config = {'temperature': 1.0, 'use_bayesian': True}
        ensemble = SoftVotingEnsemble(config)
        
        # Fit ensemble
        ensemble.fit(sample_signals[:50], sample_outcomes[:10])
        assert ensemble.is_fitted
        
        # Test probability conversion
        test_signals = [
            AgentSignal('agent1', datetime.now(), 0.5, 0.8, {}),
            AgentSignal('agent2', datetime.now(), -0.2, 0.7, {}),
            AgentSignal('agent3', datetime.now(), 0.3, 0.9, {})
        ]
        
        prediction = ensemble.predict(test_signals)
        assert -1 <= prediction.signal <= 1
        assert 0 <= prediction.confidence <= 1
        
    def test_weighted_majority(self):
        """Test weighted majority voting."""
        config = {'consensus_threshold': 0.6, 'super_majority': 0.8}
        ensemble = WeightedMajorityVoting(config)
        
        # Test with strong consensus
        strong_signals = [
            AgentSignal(f'agent{i}', datetime.now(), 0.8, 0.9, {})
            for i in range(5)
        ]
        
        prediction = ensemble.predict(strong_signals)
        assert prediction.signal > 0.5  # Strong buy consensus
        assert prediction.metadata['decision'] == 'strong_buy'


class TestStackingMethods:
    """Test stacking ensemble methods."""
    
    def test_stacking_ensemble(self, sample_signals, sample_outcomes):
        """Test basic stacking ensemble."""
        config = {
            'meta_model': 'xgboost',
            'use_cv': True,
            'cv_folds': 3
        }
        ensemble = StackingEnsemble(config)
        
        # Fit ensemble
        ensemble.fit(sample_signals[:100], sample_outcomes[:20])
        assert ensemble.is_fitted
        assert ensemble.meta_model is not None
        
        # Make prediction
        test_signals = sample_signals[100:105]
        prediction = ensemble.predict(test_signals)
        
        assert isinstance(prediction, EnsembleSignal)
        assert -1 <= prediction.signal <= 1
        
    def test_hierarchical_stacking(self, sample_signals, sample_outcomes):
        """Test hierarchical stacking."""
        config = {
            'levels': 2,
            'level_configs': {
                'technical': {'meta_model': 'linear'},
                'level2': {'meta_model': 'xgboost'}
            }
        }
        ensemble = HierarchicalStacking(config)
        
        # Modify agent IDs to include types
        for signal in sample_signals[:50]:
            if 'agent1' in signal.agent_id or 'agent2' in signal.agent_id:
                signal.agent_id = f"technical_{signal.agent_id}"
                
        ensemble.fit(sample_signals[:50], sample_outcomes[:10])
        
        # Test prediction
        test_signals = [
            AgentSignal('technical_agent1', datetime.now(), 0.5, 0.8, {}),
            AgentSignal('fundamental_agent2', datetime.now(), -0.2, 0.7, {})
        ]
        
        prediction = ensemble.predict(test_signals)
        assert isinstance(prediction, EnsembleSignal)


class TestBoostingAlgorithms:
    """Test boosting ensemble methods."""
    
    def test_adaptive_boosting(self, sample_signals, sample_outcomes):
        """Test adaptive boosting."""
        config = {
            'n_estimators': 10,
            'learning_rate': 1.0,
            'base_estimator': 'decision_tree'
        }
        ensemble = AdaptiveBoosting(config)
        
        ensemble.fit(sample_signals[:50], sample_outcomes[:10])
        assert ensemble.is_fitted
        assert len(ensemble.estimators) > 0
        
        # Test prediction
        prediction = ensemble.predict(sample_signals[50:55])
        assert isinstance(prediction, EnsembleSignal)
        
    def test_gradient_boosting(self, sample_signals, sample_outcomes):
        """Test gradient boosting."""
        config = {
            'n_estimators': 10,
            'learning_rate': 0.1,
            'max_depth': 3
        }
        ensemble = GradientBoosting(config)
        
        ensemble.fit(sample_signals[:50], sample_outcomes[:10])
        assert ensemble.is_fitted
        assert ensemble.model is not None
        
    @pytest.mark.skipif(not pytest.importorskip("xgboost"), reason="xgboost not installed")
    def test_xgboost_ensemble(self, sample_signals, sample_outcomes):
        """Test XGBoost ensemble."""
        config = {'n_estimators': 10}
        ensemble = XGBoostEnsemble(config)
        
        ensemble.fit(sample_signals[:50], sample_outcomes[:10])
        assert ensemble.is_fitted
        
        prediction = ensemble.predict(sample_signals[50:55])
        assert isinstance(prediction, EnsembleSignal)
        
    def test_online_boosting(self, sample_signals, sample_outcomes):
        """Test online boosting."""
        config = {
            'window_size': 50,
            'update_frequency': 10,
            'base_model': 'gradient'
        }
        ensemble = OnlineBoosting(config)
        
        # Add samples incrementally
        for i in range(20):
            ensemble.add_sample(sample_signals[i*5:(i+1)*5], sample_outcomes[i])
            
        assert ensemble.is_fitted
        
        prediction = ensemble.predict(sample_signals[100:105])
        assert isinstance(prediction, EnsembleSignal)


class TestSignalAggregation:
    """Test signal aggregation methods."""
    
    def test_signal_aggregator(self):
        """Test various aggregation methods."""
        config = {'method': 'weighted_average', 'outlier_detection': True}
        aggregator = SignalAggregator(config)
        
        signals = [0.5, 0.3, 0.7, -0.2, 0.1]
        confidences = [0.8, 0.7, 0.9, 0.6, 0.8]
        weights = [1.0, 0.8, 1.2, 0.7, 0.9]
        
        result = aggregator.aggregate(signals, confidences, weights)
        
        assert 'signal' in result
        assert 'confidence' in result
        assert 'diversity' in result
        assert -1 <= result['signal'] <= 1
        
    def test_temporal_aggregator(self):
        """Test temporal signal aggregation."""
        config = {'window_size': 5, 'decay_rate': 0.9}
        aggregator = TemporalAggregator(config)
        
        # Add signals over time
        for i in range(10):
            aggregator.add_signal(
                'agent1',
                np.sin(i * 0.5),
                0.8,
                datetime.now() + timedelta(seconds=i)
            )
            
        result = aggregator.get_temporal_aggregate('agent1')
        assert 'signal' in result
        assert 'confidence' in result
        assert result['history_length'] <= 5
        
        # Test momentum calculation
        momentum = aggregator.get_signal_momentum('agent1')
        assert isinstance(momentum, float)
        
    def test_consensus_aggregator(self):
        """Test consensus aggregation."""
        config = {
            'consensus_threshold': 0.6,
            'super_majority': 0.8,
            'require_quorum': True,
            'quorum_size': 0.5
        }
        aggregator = ConsensusAggregator(config)
        
        # Test with consensus
        signals = [
            {'signal': 0.8, 'confidence': 0.9},
            {'signal': 0.7, 'confidence': 0.8},
            {'signal': 0.6, 'confidence': 0.7},
            {'signal': 0.5, 'confidence': 0.8}
        ]
        
        result = aggregator.calculate_consensus(signals, 5)
        assert result['consensus'] == True
        assert result['consensus_type'] == 'super_majority_buy'


class TestEnsembleValidation:
    """Test ensemble validation utilities."""
    
    def test_cross_validation(self, sample_signals, sample_outcomes):
        """Test cross-validation."""
        config = {'n_splits': 3}
        validator = EnsembleValidator(config)
        
        ensemble = HardVotingEnsemble({})
        cv_results = validator.cross_validate_ensemble(
            ensemble, sample_signals[:50], sample_outcomes[:10]
        )
        
        assert 'train_scores' in cv_results
        assert 'test_scores' in cv_results
        assert len(cv_results['train_scores']) == 3
        
    def test_signal_quality_validation(self):
        """Test signal quality validation."""
        validator = EnsembleValidator({})
        
        # Create ensemble signals
        ensemble_signals = [
            EnsembleSignal(
                datetime.now(),
                np.random.uniform(-1, 1),
                np.random.uniform(0.5, 1),
                ['agent1', 'agent2'],
                {'agent1': 0.5, 'agent2': 0.5},
                {}
            )
            for _ in range(50)
        ]
        
        outcomes = np.random.randn(50)
        
        quality_metrics = validator.validate_signal_quality(ensemble_signals, outcomes)
        
        assert 'accuracy' in quality_metrics
        assert 'stability' in quality_metrics
        assert 'warnings' in quality_metrics
        
    def test_ensemble_monitor(self):
        """Test ensemble monitoring."""
        config = {'performance_window': 10}
        monitor = EnsembleMonitor(config)
        
        # Add signals and outcomes
        for i in range(20):
            signal = EnsembleSignal(
                datetime.now(),
                np.random.uniform(-1, 1),
                0.8,
                ['agent1'],
                {'agent1': 1.0},
                {'diversity': 0.5}
            )
            outcome = np.random.randn()
            
            result = monitor.update(signal, outcome)
            
            if i >= 10:
                assert 'metrics' in result
                assert result['metrics'].get('accuracy') is not None


class TestEnsembleService:
    """Test ensemble service layer."""
    
    def test_create_ensemble(self, db_session):
        """Test ensemble creation."""
        service = EnsembleService(db_session)
        
        ensemble_id = service.create_ensemble(
            'test_ensemble',
            'voting',
            {'method': 'soft_voting'}
        )
        
        assert ensemble_id is not None
        assert ensemble_id in service.ensemble_managers
        
    def test_register_and_activate_agent(self, db_session):
        """Test agent registration and activation."""
        service = EnsembleService(db_session)
        
        # Create ensemble
        ensemble_id = service.create_ensemble('test', 'voting', {})
        
        # Register agent
        agent_id = service.register_agent(
            'test_agent',
            'technical',
            {'indicator': 'SMA'}
        )
        
        assert agent_id is not None
        
        # Activate agent
        service.activate_agent(agent_id, ensemble_id)
        
    @pytest.mark.asyncio
    async def test_generate_prediction(self, db_session):
        """Test ensemble prediction generation."""
        service = EnsembleService(db_session)
        
        # Setup
        ensemble_id = service.create_ensemble('test', 'voting', {})
        agent_id = service.register_agent('agent1', 'technical', {})
        service.activate_agent(agent_id)
        
        # Create signals
        signals = [
            AgentSignalCreate(
                agent_id=agent_id,
                signal=0.5,
                confidence=0.8
            )
        ]
        
        # Generate prediction
        prediction = await service.generate_ensemble_prediction(ensemble_id, signals)
        
        assert prediction is not None
        assert prediction.ensemble_id == ensemble_id
        assert -1 <= prediction.signal <= 1
        
    def test_performance_tracking(self, db_session):
        """Test performance tracking."""
        service = EnsembleService(db_session)
        
        ensemble_id = service.create_ensemble('test', 'voting', {})
        
        # Get initial performance (should be empty)
        performance = service.get_ensemble_performance(ensemble_id, days=30)
        
        assert isinstance(performance, dict)


class TestIntegration:
    """Integration tests for ensemble system."""
    
    @pytest.mark.integration
    def test_full_ensemble_workflow(self, db_session, sample_signals, sample_outcomes):
        """Test complete ensemble workflow."""
        service = EnsembleService(db_session)
        
        # 1. Create ensemble
        ensemble_id = service.create_ensemble(
            'integrated_ensemble',
            'stacking',
            {'meta_model': 'xgboost', 'use_cv': True}
        )
        
        # 2. Register multiple agents
        agent_ids = []
        for i in range(5):
            agent_id = service.register_agent(
                f'agent_{i}',
                'technical' if i < 3 else 'fundamental',
                {'id': i}
            )
            agent_ids.append(agent_id)
            service.activate_agent(agent_id)
            
        # 3. Train ensemble (mock)
        manager = service.ensemble_managers[ensemble_id]
        ensemble = StackingEnsemble({'meta_model': 'xgboost'})
        manager.add_ensemble('stacking', ensemble)
        
        # 4. Validate ensemble
        validator = EnsembleValidator({})
        cv_results = validator.cross_validate_ensemble(
            ensemble,
            sample_signals[:50],
            sample_outcomes[:10]
        )
        
        assert cv_results['mean_test_score'] > -1  # Basic sanity check
        
        # 5. Generate predictions
        test_signal = EnsembleSignal(
            datetime.now(),
            0.5,
            0.8,
            agent_ids,
            {agent_id: 0.2 for agent_id in agent_ids},
            {}
        )
        
        assert test_signal.signal == 0.5
        assert len(test_signal.contributing_agents) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])