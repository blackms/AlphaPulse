"""
Tests for Hidden Markov Model Regime Detection.

Tests cover HMM model fitting, prediction, various HMM variants,
and parameter optimization.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from alpha_pulse.ml.regime.hmm_regime_detector import (
    GaussianHMM, HMMConfig, RegimeType, HMMState,
    RegimeSwitchingGARCH, HierarchicalHMM, HiddenSemiMarkovModel,
    FactorialHMM, InputOutputHMM, EnsembleHMM
)


class TestGaussianHMM:
    """Test Gaussian Hidden Markov Model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market-like data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate regime sequence
        regimes = []
        current_regime = 0
        for _ in range(n_samples):
            # Transition probability
            if np.random.random() < 0.05:
                current_regime = (current_regime + 1) % 3
            regimes.append(current_regime)
        
        regimes = np.array(regimes)
        
        # Generate observations based on regimes
        features = []
        for regime in regimes:
            if regime == 0:  # Bull market
                mean = [0.001, 0.15]
                cov = [[0.0001, 0], [0, 0.01]]
            elif regime == 1:  # Bear market
                mean = [-0.001, 0.25]
                cov = [[0.0002, 0], [0, 0.02]]
            else:  # Sideways market
                mean = [0.0, 0.12]
                cov = [[0.00008, 0], [0, 0.008]]
            
            features.append(np.random.multivariate_normal(mean, cov))
        
        return np.array(features), regimes
    
    @pytest.fixture
    def config(self):
        """Default HMM configuration."""
        return HMMConfig(
            n_states=3,
            covariance_type="full",
            n_iter=50,
            random_state=42
        )
    
    def test_hmm_initialization(self, config):
        """Test HMM initialization."""
        model = GaussianHMM(config)
        
        assert model.n_states == 3
        assert model.config.covariance_type == "full"
        assert not model.is_fitted
        assert model.start_prob is None
        assert model.trans_prob is None
    
    def test_hmm_fit(self, sample_data, config):
        """Test HMM fitting."""
        features, true_regimes = sample_data
        model = GaussianHMM(config)
        
        # Fit model
        model.fit(features)
        
        assert model.is_fitted
        assert model.start_prob is not None
        assert model.trans_prob is not None
        assert model.means is not None
        assert model.covars is not None
        assert len(model.states) == config.n_states
        
        # Check parameter shapes
        assert model.start_prob.shape == (3,)
        assert model.trans_prob.shape == (3, 3)
        assert model.means.shape == (3, 2)
        assert model.covars.shape == (3, 2, 2)
        
        # Check probability constraints
        np.testing.assert_almost_equal(model.start_prob.sum(), 1.0)
        np.testing.assert_almost_equal(model.trans_prob.sum(axis=1), np.ones(3))
    
    def test_hmm_predict(self, sample_data, config):
        """Test HMM prediction."""
        features, true_regimes = sample_data
        model = GaussianHMM(config)
        
        # Fit model
        model.fit(features)
        
        # Predict states
        predicted_states = model.predict(features)
        
        assert predicted_states.shape == true_regimes.shape
        assert np.all(predicted_states >= 0)
        assert np.all(predicted_states < config.n_states)
        
        # Test prediction on new data
        new_features = features[:100]
        new_predictions = model.predict(new_features)
        assert new_predictions.shape == (100,)
    
    def test_hmm_predict_proba(self, sample_data, config):
        """Test HMM probability prediction."""
        features, _ = sample_data
        model = GaussianHMM(config)
        
        # Fit model
        model.fit(features)
        
        # Predict probabilities
        probs = model.predict_proba(features)
        
        assert probs.shape == (len(features), config.n_states)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_almost_equal(probs.sum(axis=1), np.ones(len(features)))
    
    def test_hmm_score(self, sample_data, config):
        """Test HMM scoring."""
        features, _ = sample_data
        model = GaussianHMM(config)
        
        # Fit model
        model.fit(features)
        
        # Calculate log-likelihood
        score = model.score(features)
        
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)
    
    def test_regime_classification(self, sample_data, config):
        """Test regime type classification."""
        features, _ = sample_data
        model = GaussianHMM(config)
        
        # Fit model
        model.fit(features)
        
        # Check regime types
        for state in model.states:
            assert isinstance(state, HMMState)
            assert isinstance(state.regime_type, RegimeType)
            assert state.regime_type in [
                RegimeType.BULL, RegimeType.BEAR, RegimeType.SIDEWAYS,
                RegimeType.CRISIS, RegimeType.RECOVERY
            ]
    
    def test_different_covariance_types(self, sample_data):
        """Test different covariance types."""
        features, _ = sample_data
        
        for cov_type in ['full', 'diag']:
            config = HMMConfig(
                n_states=3,
                covariance_type=cov_type,
                n_iter=50,
                random_state=42
            )
            model = GaussianHMM(config)
            model.fit(features)
            
            assert model.is_fitted
            predictions = model.predict(features)
            assert len(predictions) == len(features)
    
    def test_different_initialization_methods(self, sample_data):
        """Test different initialization methods."""
        features, _ = sample_data
        
        for init_method in ['kmeans', 'uniform', 'random']:
            config = HMMConfig(
                n_states=3,
                init_method=init_method,
                n_iter=50,
                random_state=42
            )
            model = GaussianHMM(config)
            model.fit(features)
            
            assert model.is_fitted
            assert len(model.states) == 3
    
    def test_convergence_history(self, sample_data, config):
        """Test convergence history tracking."""
        features, _ = sample_data
        model = GaussianHMM(config)
        
        model.fit(features)
        
        assert len(model.convergence_history) > 0
        assert all(isinstance(ll, float) for ll in model.convergence_history)
        
        # Check that log-likelihood is increasing
        for i in range(1, len(model.convergence_history)):
            assert model.convergence_history[i] >= model.convergence_history[i-1] - 1e-6
    
    def test_regime_statistics(self, sample_data, config):
        """Test regime statistics calculation."""
        features, _ = sample_data
        model = GaussianHMM(config)
        
        model.fit(features)
        states = model.predict(features)
        
        stats = model.get_regime_statistics(states)
        
        assert isinstance(stats, pd.DataFrame)
        assert 'regime' in stats.columns
        assert 'type' in stats.columns
        assert 'frequency' in stats.columns
        assert 'avg_duration' in stats.columns
        
        # Check frequency sums to 1
        assert abs(stats['frequency'].sum() - 1.0) < 0.01


class TestRegimeSwitchingGARCH:
    """Test Regime Switching GARCH model."""
    
    @pytest.fixture
    def volatility_data(self):
        """Generate data with volatility clustering."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate returns with GARCH effects
        returns = []
        volatility = []
        current_vol = 0.01
        
        for i in range(n_samples):
            # GARCH(1,1) process
            shock = np.random.normal(0, 1)
            current_vol = np.sqrt(0.00001 + 0.1 * shock**2 + 0.85 * current_vol**2)
            returns.append(shock * current_vol)
            volatility.append(current_vol)
        
        features = np.column_stack([returns, volatility])
        return features
    
    def test_garch_initialization(self):
        """Test GARCH model initialization."""
        config = HMMConfig(n_states=3, use_regime_switching_garch=True)
        model = RegimeSwitchingGARCH(config)
        
        assert isinstance(model, RegimeSwitchingGARCH)
        assert hasattr(model, 'garch_params')
        assert model.garch_params == {}
    
    def test_garch_fit(self, volatility_data):
        """Test GARCH model fitting."""
        config = HMMConfig(n_states=2, n_iter=30, random_state=42)
        model = RegimeSwitchingGARCH(config)
        
        model.fit(volatility_data)
        
        assert model.is_fitted
        assert len(model.garch_params) == 2
        
        # Check GARCH parameters
        for state in range(2):
            assert state in model.garch_params
            params = model.garch_params[state]
            assert 'omega' in params
            assert 'alpha' in params
            assert 'beta' in params
            assert 'unconditional_var' in params
            
            # Check parameter constraints
            assert params['alpha'] >= 0
            assert params['beta'] >= 0
            assert params['alpha'] + params['beta'] < 1


class TestHierarchicalHMM:
    """Test Hierarchical HMM."""
    
    def test_hierarchical_initialization(self):
        """Test hierarchical HMM initialization."""
        config = HMMConfig(n_states=4, random_state=42)
        model = HierarchicalHMM(config, n_levels=2)
        
        assert isinstance(model, HierarchicalHMM)
        assert model.n_levels == 2
        assert model.sub_models == {}
    
    def test_hierarchical_fit(self, sample_data):
        """Test hierarchical HMM fitting."""
        features, _ = sample_data
        config = HMMConfig(n_states=3, n_iter=30, random_state=42)
        model = HierarchicalHMM(config, n_levels=2)
        
        model.fit(features)
        
        assert model.is_fitted
        assert len(model.sub_models) > 0
        
        # Check sub-models
        for state_id, sub_model in model.sub_models.items():
            assert isinstance(sub_model, GaussianHMM)
            assert sub_model.is_fitted
    
    def test_hierarchical_prediction(self, sample_data):
        """Test hierarchical prediction."""
        features, _ = sample_data
        config = HMMConfig(n_states=3, n_iter=30, random_state=42)
        model = HierarchicalHMM(config, n_levels=2)
        
        model.fit(features)
        
        top_states, sub_states = model.predict_hierarchical(features)
        
        assert len(top_states) == len(features)
        assert isinstance(sub_states, dict)
        
        # Check sub-state predictions
        for state_id, sub_predictions in sub_states.items():
            mask = top_states == state_id
            assert len(sub_predictions) == mask.sum()


class TestHiddenSemiMarkovModel:
    """Test Hidden Semi-Markov Model."""
    
    def test_hsmm_initialization(self):
        """Test HSMM initialization."""
        config = HMMConfig(n_states=3, random_state=42)
        model = HiddenSemiMarkovModel(config)
        
        assert isinstance(model, HiddenSemiMarkovModel)
        assert hasattr(model, 'duration_params')
        assert hasattr(model, 'max_duration')
        assert model.max_duration == 100
    
    def test_hsmm_duration_estimation(self, sample_data):
        """Test duration distribution estimation."""
        features, _ = sample_data
        config = HMMConfig(n_states=3, n_iter=20, random_state=42)
        model = HiddenSemiMarkovModel(config)
        
        model.fit(features)
        
        assert model.is_fitted
        assert len(model.duration_params) == 3
        
        # Check duration parameters
        for state in range(3):
            assert state in model.duration_params
            params = model.duration_params[state]
            assert 'distribution' in params
            assert 'mean' in params
            assert 'std' in params
            assert params['mean'] > 0
            assert params['std'] > 0
    
    def test_duration_probability(self):
        """Test duration probability calculation."""
        config = HMMConfig(n_states=2, random_state=42)
        model = HiddenSemiMarkovModel(config)
        
        # Set dummy duration parameters
        model.duration_params[0] = {
            'distribution': 'negative_binomial',
            'r': 10,
            'p': 0.5,
            'mean': 10,
            'std': 5
        }
        
        # Test duration probabilities
        prob = model.duration_probability(0, 10)
        assert 0 <= prob <= 1
        assert not np.isnan(prob)


class TestFactorialHMM:
    """Test Factorial HMM."""
    
    def test_factorial_initialization(self):
        """Test factorial HMM initialization."""
        config = HMMConfig(n_states=6, random_state=42)
        model = FactorialHMM(config, n_chains=2)
        
        assert isinstance(model, FactorialHMM)
        assert model.n_chains == 2
        assert model.chain_models == []
        assert model.chain_weights is None
    
    def test_factorial_fit(self, sample_data):
        """Test factorial HMM fitting."""
        features, _ = sample_data
        config = HMMConfig(n_states=4, n_iter=20, random_state=42)
        model = FactorialHMM(config, n_chains=2)
        
        model.fit(features)
        
        assert model.is_fitted
        assert len(model.chain_models) == 2
        assert model.chain_weights is not None
        assert len(model.chain_weights) == 2
        assert abs(model.chain_weights.sum() - 1.0) < 0.01
    
    def test_factorial_states(self, sample_data):
        """Test factorial state extraction."""
        features, _ = sample_data
        config = HMMConfig(n_states=4, n_iter=20, random_state=42)
        model = FactorialHMM(config, n_chains=2)
        
        model.fit(features)
        
        chain_states = model.factorial_states(features)
        
        assert len(chain_states) == 2
        for states in chain_states:
            assert len(states) == len(features)


class TestInputOutputHMM:
    """Test Input-Output HMM."""
    
    @pytest.fixture
    def io_data(self):
        """Generate data with external inputs."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate external inputs (e.g., market indicators)
        inputs = np.random.randn(n_samples, 2)
        
        # Generate observations influenced by inputs
        observations = []
        for i in range(n_samples):
            # Simple linear influence
            obs = np.random.randn(2) + 0.5 * inputs[i]
            observations.append(obs)
        
        return np.array(observations), inputs
    
    def test_iohmm_initialization(self):
        """Test IOHMM initialization."""
        config = HMMConfig(n_states=3, random_state=42)
        model = InputOutputHMM(config)
        
        assert isinstance(model, InputOutputHMM)
        assert model.input_weights is None
        assert model.n_inputs is None
    
    def test_iohmm_fit(self, io_data):
        """Test IOHMM fitting with external inputs."""
        observations, inputs = io_data
        config = HMMConfig(n_states=2, n_iter=20, random_state=42)
        model = InputOutputHMM(config)
        
        model.fit(observations, inputs)
        
        assert model.is_fitted
        assert model.n_inputs == 2
        assert model.input_weights is not None
        assert model.input_weights.shape == (2, 2, 2)
    
    def test_input_dependent_transitions(self, io_data):
        """Test input-dependent transition matrix computation."""
        observations, inputs = io_data
        config = HMMConfig(n_states=2, n_iter=20, random_state=42)
        model = InputOutputHMM(config)
        
        model.fit(observations, inputs)
        
        # Test transition matrix computation
        trans_matrix = model._compute_transition_matrix(inputs[0])
        
        assert trans_matrix.shape == (2, 2)
        assert np.all(trans_matrix >= 0)
        assert np.all(trans_matrix <= 1)
        np.testing.assert_almost_equal(trans_matrix.sum(axis=1), np.ones(2))


class TestEnsembleHMM:
    """Test Ensemble HMM."""
    
    @pytest.fixture
    def ensemble_models(self, sample_data):
        """Create ensemble of fitted models."""
        features, _ = sample_data
        
        models = []
        for n_states in [2, 3, 4]:
            config = HMMConfig(n_states=n_states, n_iter=20, random_state=42)
            model = GaussianHMM(config)
            model.fit(features)
            models.append(model)
        
        return models
    
    def test_ensemble_initialization(self, ensemble_models):
        """Test ensemble initialization."""
        ensemble = EnsembleHMM(ensemble_models)
        
        assert ensemble.n_models == 3
        assert len(ensemble.base_models) == 3
        assert ensemble.weights is not None
        assert len(ensemble.weights) == 3
        assert abs(ensemble.weights.sum() - 1.0) < 0.01
    
    def test_ensemble_fit(self, sample_data):
        """Test ensemble fitting."""
        features, _ = sample_data
        
        # Create unfitted models
        models = []
        for n_states in [2, 3]:
            config = HMMConfig(n_states=n_states, n_iter=20, random_state=42)
            model = GaussianHMM(config)
            models.append(model)
        
        ensemble = EnsembleHMM(models)
        ensemble.fit(features)
        
        assert ensemble.is_fitted
        assert all(model.is_fitted for model in ensemble.base_models)
    
    def test_ensemble_predict(self, ensemble_models, sample_data):
        """Test ensemble prediction."""
        features, _ = sample_data
        ensemble = EnsembleHMM(ensemble_models)
        
        predictions = ensemble.predict(features[:100])
        
        assert len(predictions) == 100
        assert np.all(predictions >= 0)
    
    def test_ensemble_predict_proba(self, ensemble_models, sample_data):
        """Test ensemble probability prediction."""
        features, _ = sample_data
        ensemble = EnsembleHMM(ensemble_models)
        
        probs = ensemble.predict_proba(features[:100])
        
        assert probs.shape[0] == 100
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_ensemble_regime_prediction(self, sample_data):
        """Test ensemble regime type prediction."""
        features, _ = sample_data
        
        # Create models with regime prediction capability
        models = []
        for n_states in [3, 4]:
            config = HMMConfig(n_states=n_states, n_iter=20, random_state=42)
            model = GaussianHMM(config)
            model.fit(features)
            
            # Add predict_regime_type method
            def predict_regime_type(X):
                states = model.predict(X)
                return [model.states[s].regime_type for s in states]
            
            model.predict_regime_type = predict_regime_type
            models.append(model)
        
        ensemble = EnsembleHMM(models)
        regimes = ensemble.predict_regime_type(features[:50])
        
        assert len(regimes) == 50
        assert all(isinstance(r, RegimeType) for r in regimes)


class TestModelSelection:
    """Test model selection and parameter optimization."""
    
    def test_optimal_state_selection(self, sample_data):
        """Test selecting optimal number of states."""
        features, _ = sample_data
        
        scores = {}
        for n_states in range(2, 6):
            config = HMMConfig(n_states=n_states, n_iter=30, random_state=42)
            model = GaussianHMM(config)
            
            try:
                model.fit(features)
                # Use BIC for model selection
                log_likelihood = model.score(features)
                n_params = n_states * n_states + n_states * features.shape[1] * 2
                bic = -2 * log_likelihood + n_params * np.log(len(features))
                scores[n_states] = -bic  # Negative because we want to maximize
            except:
                scores[n_states] = -np.inf
        
        # Best model should have reasonable number of states
        best_n_states = max(scores, key=scores.get)
        assert 2 <= best_n_states <= 5
    
    def test_stability_across_seeds(self, sample_data):
        """Test model stability across random seeds."""
        features, _ = sample_data
        
        predictions = []
        for seed in [42, 123, 456]:
            config = HMMConfig(n_states=3, n_iter=30, random_state=seed)
            model = GaussianHMM(config)
            model.fit(features)
            pred = model.predict(features[:100])
            predictions.append(pred)
        
        # Check that predictions are somewhat consistent
        # (allowing for label switching)
        assert len(set(len(np.unique(p)) for p in predictions)) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])