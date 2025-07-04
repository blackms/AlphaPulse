"""Comprehensive tests for online learning module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
from sqlalchemy.orm import Session

from alpha_pulse.ml.online import (
    BaseOnlineLearner, OnlineLearnerEnsemble, AdaptiveLearningController,
    OnlineDataPoint, LearningState,
    IncrementalSGD, IncrementalNaiveBayes, HoeffdingTree, AdaptiveRandomForest,
    AdaptiveLearningRateScheduler, AdaptiveOptimizer, MultiArmedBandit,
    ConceptDriftDetector, ADWIN, DDM, PageHinkley, KSWIN,
    MemoryManager, SlidingWindowBuffer, ReservoirSampler,
    StreamingValidator, PrequentialEvaluator,
    OnlineLearningService,
    OnlineDataPointModel, StreamingBatch, LearningSessionRequest
)


class TestOnlineLearner:
    """Test base online learner functionality."""
    
    def test_incremental_sgd_regression(self):
        """Test incremental SGD for regression."""
        config = {
            'learning_rate': 0.01,
            'penalty': 'l2',
            'alpha': 0.0001
        }
        
        learner = IncrementalSGD(config, task_type='regression')
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(n_samples) * 0.1
        
        # Train incrementally
        for i in range(n_samples):
            data_point = OnlineDataPoint(
                timestamp=datetime.now(),
                features=X[i],
                label=y[i]
            )
            prediction = learner.learn_one(data_point)
            
            # After some samples, predictions should improve
            if i > 50 and prediction is not None:
                error = abs(prediction - y[i])
                assert error < 2.0  # Reasonable error bound
                
        # Test batch prediction
        test_X = np.random.randn(10, 5)
        predictions = learner.predict(test_X)
        assert predictions.shape == (10,)
        
    def test_incremental_naive_bayes_classification(self):
        """Test incremental Naive Bayes for classification."""
        config = {'var_smoothing': 1e-9}
        
        learner = IncrementalNaiveBayes(config)
        
        # Generate synthetic classification data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Train incrementally
        accuracy_sum = 0
        accuracy_count = 0
        
        for i in range(n_samples):
            data_point = OnlineDataPoint(
                timestamp=datetime.now(),
                features=X[i],
                label=y[i]
            )
            prediction = learner.learn_one(data_point)
            
            if prediction is not None:
                accuracy_sum += int(round(prediction) == y[i])
                accuracy_count += 1
                
        # Check reasonable accuracy
        if accuracy_count > 0:
            accuracy = accuracy_sum / accuracy_count
            assert accuracy > 0.6  # Better than random
            
        # Test probability prediction
        test_X = np.random.randn(5, 3)
        probas = learner.predict_proba(test_X)
        assert probas.shape[0] == 5
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        
    def test_hoeffding_tree(self):
        """Test Hoeffding tree for streaming classification."""
        config = {
            'grace_period': 50,
            'split_confidence': 1e-6,
            'tie_threshold': 0.05
        }
        
        learner = HoeffdingTree(config)
        
        # Generate data with clear pattern
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 4)
        # Simple rule: if feature 0 > 0 and feature 1 > 0, then class 1
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
        
        # Train tree
        for i in range(n_samples):
            learner.partial_fit(X[i:i+1], y[i:i+1])
            
        # Test predictions
        test_X = np.array([[1, 1, 0, 0], [-1, -1, 0, 0], [1, -1, 0, 0]])
        predictions = learner.predict(test_X)
        
        # Check learned pattern
        assert predictions[0] == 1  # Both positive
        assert predictions[1] == 0  # Both negative
        
    def test_adaptive_random_forest(self):
        """Test adaptive random forest with drift detection."""
        config = {
            'n_estimators': 3,
            'max_features': 'sqrt',
            'drift_detection': True
        }
        
        learner = AdaptiveRandomForest(config)
        
        # Generate data with concept drift
        np.random.seed(42)
        n_samples = 300
        X = np.random.randn(n_samples, 6)
        
        # First concept: feature 0 determines class
        y1 = (X[:150, 0] > 0).astype(int)
        # Second concept: feature 1 determines class (drift)
        y2 = (X[150:, 1] > 0).astype(int)
        y = np.concatenate([y1, y2])
        
        # Train with drift
        drift_detected = False
        for i in range(n_samples):
            learner.partial_fit(X[i:i+1], y[i:i+1])
            
            # Check if any tree detected drift
            if i > 150:
                for detector in learner.tree_drift_detectors:
                    if detector.detected_change():
                        drift_detected = True
                        break
                        
        # Should detect drift at some point
        assert drift_detected
        
    def test_online_learner_ensemble(self):
        """Test ensemble of online learners."""
        config = {
            'max_models': 3,
            'combination_method': 'weighted_average'
        }
        
        ensemble = OnlineLearnerEnsemble(config)
        
        # Add different learners
        sgd_config = {'learning_rate': 0.01}
        nb_config = {}
        
        ensemble.add_learner(IncrementalSGD(sgd_config))
        ensemble.add_learner(IncrementalNaiveBayes(nb_config))
        
        # Generate regression data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 4)
        y = X[:, 0] * 2 + np.random.randn(n_samples) * 0.1
        
        # Train ensemble
        for i in range(n_samples):
            data_point = OnlineDataPoint(
                timestamp=datetime.now(),
                features=X[i],
                label=y[i]
            )
            prediction = ensemble.learn_one(data_point)
            
        # Test ensemble prediction
        test_X = np.random.randn(5, 4)
        predictions = ensemble.predict(test_X)
        assert predictions.shape == (5,)
        
        # Check weights are being updated
        ensemble.update_weights()
        assert len(ensemble.learner_weights) == 2
        assert sum(ensemble.learner_weights) == pytest.approx(1.0)


class TestAdaptiveAlgorithms:
    """Test adaptive learning algorithms."""
    
    def test_adaptive_learning_rate_scheduler(self):
        """Test different learning rate schedules."""
        config = {
            'schedule_type': 'exponential_decay',
            'initial_rate': 0.1,
            'decay_rate': 0.95,
            'decay_steps': 100,
            'min_rate': 1e-6,
            'max_rate': 1.0
        }
        
        scheduler = AdaptiveLearningRateScheduler(config)
        
        # Test exponential decay
        rates = []
        for step in range(500):
            rate = scheduler.step()
            rates.append(rate)
            
        # Check decay
        assert rates[0] > rates[-1]
        assert all(r >= config['min_rate'] for r in rates)
        assert all(r <= config['max_rate'] for r in rates)
        
        # Test cosine annealing
        config['schedule_type'] = 'cosine_annealing'
        config['T_max'] = 100
        scheduler = AdaptiveLearningRateScheduler(config)
        scheduler.reset()
        
        rates = []
        for step in range(100):
            rate = scheduler.step()
            rates.append(rate)
            
        # Should decrease then potentially restart
        assert rates[0] > rates[50]
        
    def test_adaptive_optimizer(self):
        """Test adaptive optimization algorithms."""
        config = {
            'optimizer_type': 'adam',
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
        
        optimizer = AdaptiveOptimizer(config)
        
        # Test parameter update
        params = {'w': np.array([1.0, 2.0, 3.0])}
        gradients = {'w': np.array([0.1, -0.2, 0.3])}
        
        # Multiple updates
        for _ in range(10):
            params = optimizer.update(params, gradients)
            
        # Parameters should change
        assert not np.array_equal(params['w'], np.array([1.0, 2.0, 3.0]))
        
        # Test different optimizers
        for opt_type in ['rmsprop', 'adagrad', 'momentum']:
            config['optimizer_type'] = opt_type
            optimizer = AdaptiveOptimizer(config)
            optimizer.reset()
            
            params = {'w': np.array([1.0, 2.0, 3.0])}
            new_params = optimizer.update(params, gradients)
            assert 'w' in new_params
            
    def test_multi_armed_bandit(self):
        """Test multi-armed bandit algorithms."""
        config = {
            'n_arms': 4,
            'algorithm': 'ucb',
            'c': 2.0
        }
        
        bandit = MultiArmedBandit(config)
        
        # Simulate rewards
        true_rewards = [0.2, 0.5, 0.3, 0.8]  # Arm 3 is best
        
        selections = []
        for _ in range(1000):
            arm = bandit.select_arm()
            reward = np.random.binomial(1, true_rewards[arm])
            bandit.update(arm, reward)
            selections.append(arm)
            
        # Best arm should be selected most frequently
        arm_counts = [selections.count(i) for i in range(4)]
        best_arm = np.argmax(arm_counts)
        assert best_arm == 3  # Should learn arm 3 is best
        
        # Test Thompson sampling
        config['algorithm'] = 'thompson'
        bandit = MultiArmedBandit(config)
        bandit.reset()
        
        for _ in range(100):
            arm = bandit.select_arm()
            assert 0 <= arm < 4
            bandit.update(arm, np.random.random())


class TestConceptDriftDetection:
    """Test concept drift detection algorithms."""
    
    def test_adwin_drift_detection(self):
        """Test ADWIN algorithm for drift detection."""
        config = {'delta': 0.002}
        detector = ADWIN(config)
        
        # Generate data with drift
        np.random.seed(42)
        
        # Stable phase
        for _ in range(100):
            value = np.random.normal(0, 1)
            detector.add_element(value)
            
        assert not detector.detected_change()
        
        # Drift phase - different mean
        drift_detected = False
        for _ in range(100):
            value = np.random.normal(3, 1)  # Shifted mean
            detector.add_element(value)
            if detector.detected_change():
                drift_detected = True
                break
                
        assert drift_detected
        
    def test_ddm_drift_detection(self):
        """Test DDM algorithm for drift detection."""
        config = {
            'warning_level': 2.0,
            'drift_level': 3.0,
            'min_samples': 30
        }
        detector = DDM(config)
        
        # Simulate classification errors
        # Low error rate initially
        for _ in range(50):
            error = np.random.binomial(1, 0.1)  # 10% error rate
            detector.add_element(error)
            
        assert not detector.detected_change()
        
        # Increase error rate (drift)
        warning_detected = False
        drift_detected = False
        
        for _ in range(50):
            error = np.random.binomial(1, 0.5)  # 50% error rate
            detector.add_element(error)
            
            if detector.detected_warning():
                warning_detected = True
            if detector.detected_change():
                drift_detected = True
                break
                
        assert warning_detected or drift_detected
        
    def test_page_hinkley_drift_detection(self):
        """Test Page-Hinkley test for drift detection."""
        config = {
            'delta': 0.005,
            'threshold': 50.0,
            'alpha': 0.999
        }
        detector = PageHinkley(config)
        
        # Generate data with incremental drift
        np.random.seed(42)
        
        for i in range(200):
            # Gradually increasing values
            value = np.random.normal(i * 0.01, 1)
            detector.add_element(value)
            
            if detector.detected_change():
                assert i > 50  # Should detect after some samples
                break
                
    def test_kswin_drift_detection(self):
        """Test KSWIN algorithm for distributional drift."""
        config = {
            'window_size': 50,
            'alpha': 0.05,
            'stat_threshold': 0.3
        }
        detector = KSWIN(config)
        
        # Generate data with distributional change
        np.random.seed(42)
        
        # Normal distribution
        for _ in range(100):
            value = np.random.normal(0, 1)
            detector.add_element(value)
            
        # Change to exponential distribution
        drift_detected = False
        for _ in range(100):
            value = np.random.exponential(1)
            detector.add_element(value)
            
            if detector.detected_change():
                drift_detected = True
                break
                
        assert drift_detected
        
    def test_ensemble_drift_detection(self):
        """Test ensemble drift detection."""
        config = {
            'method': 'adwin',
            'ensemble_mode': True
        }
        
        detector = ConceptDriftDetector(config)
        
        # Add values with drift
        np.random.seed(42)
        
        for _ in range(100):
            detector.add_element(np.random.normal(0, 1))
            
        # Add drifted values
        for _ in range(100):
            detector.add_element(np.random.normal(5, 1))
            
            if detector.detected_change():
                # Get drift info
                info = detector.get_drift_info()
                assert 'ensemble_status' in info
                break


class TestMemoryManagement:
    """Test memory management utilities."""
    
    def test_memory_manager(self):
        """Test memory manager with different eviction policies."""
        config = {
            'max_memory_mb': 0.1,  # Small for testing
            'eviction_policy': 'lru'
        }
        
        manager = MemoryManager(config)
        
        # Add items until eviction needed
        for i in range(100):
            data = np.random.randn(100, 100)  # ~80KB each
            manager.store(f'item_{i}', data, importance=np.random.random())
            
        # Should have evicted some items
        assert len(manager._memory) < 100
        
        # Test retrieval
        stored_keys = list(manager._memory.keys())
        if stored_keys:
            data = manager.retrieve(stored_keys[0])
            assert data is not None
            
        # Test different eviction policies
        for policy in ['lfu', 'importance', 'adaptive']:
            config['eviction_policy'] = policy
            manager = MemoryManager(config)
            
            for i in range(20):
                manager.store(f'item_{i}', np.random.randn(10, 10))
                
            assert manager.get_memory_usage()['managed_items'] > 0
            
    def test_sliding_window_buffer(self):
        """Test sliding window buffer."""
        buffer = SlidingWindowBuffer(window_size=10)
        
        # Fill buffer
        for i in range(15):
            buffer.append(float(i), datetime.now())
            
        # Check window
        window = buffer.get_window()
        assert len(window) == 10
        assert window[-1] == 14  # Latest value
        
        # Test get_recent
        recent = buffer.get_recent(5)
        assert len(recent) == 5
        assert list(recent) == [10, 11, 12, 13, 14]
        
    def test_reservoir_sampler(self):
        """Test reservoir sampling."""
        sampler = ReservoirSampler(reservoir_size=10)
        
        # Add many items
        for i in range(1000):
            sampler.add(i)
            
        # Check reservoir
        sample = sampler.get_sample()
        assert len(sample) == 10
        assert all(0 <= item < 1000 for item in sample)
        
        # Check statistics
        stats = sampler.get_statistics()
        assert stats['total_seen'] == 1000
        assert stats['current_size'] == 10


class TestStreamingValidation:
    """Test streaming validation utilities."""
    
    def test_prequential_evaluator(self):
        """Test prequential evaluation."""
        config = {
            'window_size': 100,
            'fade_factor': 0.99,
            'task_type': 'regression'
        }
        
        evaluator = PrequentialEvaluator(config)
        
        # Add predictions and actuals
        np.random.seed(42)
        for i in range(200):
            prediction = np.random.randn()
            actual = prediction + np.random.randn() * 0.1  # Small error
            metrics = evaluator.add_result(prediction, actual)
            
            if i > 10:  # After warm-up
                assert 'mse' in metrics
                assert 'mae' in metrics
                assert metrics['mse'] < 1.0  # Reasonable error
                
        # Get performance
        performance = evaluator.get_current_performance()
        assert performance['n_samples'] == 200
        assert 'metrics' in performance
        
        # Test classification evaluator
        config['task_type'] = 'classification'
        evaluator = PrequentialEvaluator(config)
        
        for i in range(100):
            prediction = np.random.random()
            actual = float(np.random.binomial(1, 0.7))
            metrics = evaluator.add_result(prediction, actual)
            
            if i > 10:
                assert 'accuracy' in metrics
                assert 'f1' in metrics
                
    def test_streaming_validator(self):
        """Test streaming validator."""
        config = {
            'validation_interval': 10,
            'stability_threshold': 0.05,
            'task_type': 'regression'
        }
        
        validator = StreamingValidator(config)
        
        # Generate predictions and actuals
        np.random.seed(42)
        predictions = np.random.randn(50) + 1
        actuals = predictions + np.random.randn(50) * 0.2
        
        # Validate stream
        result = validator.validate_stream('model_1', predictions, actuals)
        
        assert result['model_id'] == 'model_1'
        assert 'performance' in result
        assert 'is_stable' in result
        assert 'anomalies' in result
        
        # Compare models
        predictions2 = actuals + np.random.randn(50) * 0.5  # Worse model
        validator.validate_stream('model_2', predictions2, actuals)
        
        comparison = validator.compare_models(['model_1', 'model_2'])
        assert 'mse' in comparison
        assert comparison['mse']['best_model'] == 'model_1'


@pytest.mark.asyncio
class TestOnlineLearningService:
    """Test online learning service."""
    
    async def test_start_session(self):
        """Test starting an online learning session."""
        # Mock database
        db = MagicMock(spec=Session)
        
        config = {'checkpoint_dir': '/tmp/test_checkpoints'}
        service = OnlineLearningService(db, config)
        
        # Start session
        request = LearningSessionRequest(
            agent_id='test_agent',
            strategy='adaptive',
            config={'model_type': 'sgd'}
        )
        
        response = await service.start_session(request)
        
        assert response.session_id
        assert response.status == 'active'
        assert response.start_time
        
        # Check session is tracked
        assert response.session_id in service.active_sessions
        assert response.session_id in service.learners
        
        # Cleanup
        await service.shutdown()
        
    async def test_process_batch(self):
        """Test processing a data batch."""
        db = MagicMock(spec=Session)
        config = {'checkpoint_dir': '/tmp/test_checkpoints'}
        service = OnlineLearningService(db, config)
        
        # Start session
        request = LearningSessionRequest(
            agent_id='test_agent',
            strategy='adaptive',
            config={'model_type': 'sgd', 'task_type': 'regression'}
        )
        response = await service.start_session(request)
        session_id = response.session_id
        
        # Create batch
        data_points = []
        for i in range(10):
            dp = OnlineDataPointModel(
                timestamp=datetime.now(),
                features=[np.random.randn() for _ in range(5)],
                label=np.random.randn()
            )
            data_points.append(dp)
            
        batch = StreamingBatch(
            batch_id='batch_1',
            data_points=data_points,
            source='test'
        )
        
        # Process batch
        result = await service.process_batch(session_id, batch)
        
        assert result['session_id'] == session_id
        assert result['n_processed'] == 10
        assert 'predictions' in result
        
        # Cleanup
        await service.shutdown()
        
    async def test_drift_handling(self):
        """Test drift detection and handling."""
        db = MagicMock(spec=Session)
        config = {
            'checkpoint_dir': '/tmp/test_checkpoints',
            'drift_detection': {'method': 'adwin'}
        }
        service = OnlineLearningService(db, config)
        
        # Start session
        request = LearningSessionRequest(
            agent_id='test_agent',
            strategy='adaptive',
            config={'model_type': 'sgd'}
        )
        response = await service.start_session(request)
        
        # Simulate drift by mocking learner state
        learner = service.learners[response.session_id]
        if hasattr(learner, 'state'):
            learner.state.drift_detected = True
            learner.state.n_samples_seen = 100
            
        # Monitor should handle drift
        await service._handle_drift(response.session_id)
        
        # Check drift event was recorded
        assert db.add.called
        
        # Cleanup
        await service.shutdown()


def test_integration_online_learning_pipeline():
    """Integration test for complete online learning pipeline."""
    # Create components
    config = {
        'learning_rate': 0.01,
        'batch_size': 10,
        'drift_detection': {'method': 'ddm'},
        'memory': {'max_memory_mb': 100},
        'validation': {'window_size': 50}
    }
    
    # Initialize learner with drift detection
    learner = IncrementalSGD(config, task_type='regression')
    validator = PrequentialEvaluator({'task_type': 'regression'})
    
    # Generate streaming data with concept drift
    np.random.seed(42)
    
    # Phase 1: Linear relationship
    for i in range(100):
        x = np.random.randn(5)
        y = 2 * x[0] + x[1] + np.random.randn() * 0.1
        
        dp = OnlineDataPoint(
            timestamp=datetime.now(),
            features=x,
            label=y
        )
        
        prediction = learner.learn_one(dp)
        if prediction is not None:
            validator.add_result(prediction, y)
            
    # Check performance before drift
    perf1 = validator.get_current_performance()
    
    # Phase 2: Changed relationship (drift)
    for i in range(100):
        x = np.random.randn(5)
        y = -1 * x[0] + 3 * x[2] + np.random.randn() * 0.1  # Different pattern
        
        dp = OnlineDataPoint(
            timestamp=datetime.now(),
            features=x,
            label=y
        )
        
        prediction = learner.learn_one(dp)
        if prediction is not None:
            validator.add_result(prediction, y)
            
    # Check if drift was detected
    assert learner.drift_detector.detected_change()
    
    # Performance should adapt
    perf2 = validator.get_current_performance()
    
    # Model should have adapted to new concept
    assert learner.state.n_samples_seen == 200
    assert learner.state.n_updates > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])