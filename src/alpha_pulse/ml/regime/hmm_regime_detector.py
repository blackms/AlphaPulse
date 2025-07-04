"""
Hidden Markov Model for Market Regime Detection.

This module implements various HMM architectures for identifying market regimes,
including Gaussian HMM, regime-switching models, and advanced variants.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class HMMConfig:
    """Configuration for Hidden Markov Model."""
    n_states: int = 5
    covariance_type: str = "full"  # 'spherical', 'diag', 'full', 'tied'
    n_iter: int = 100
    tol: float = 1e-4
    random_state: int = 42
    init_method: str = "kmeans"  # 'random', 'kmeans', 'uniform'
    min_covar: float = 1e-3
    verbose: bool = False
    
    # Advanced HMM variants
    use_regime_switching_garch: bool = False
    use_smooth_transition: bool = False
    use_hierarchical: bool = False
    
    # Regime constraints
    min_regime_duration: int = 5
    transition_penalty: float = 0.01


@dataclass
class HMMState:
    """State representation for HMM."""
    mean: np.ndarray
    covariance: np.ndarray
    regime_type: RegimeType
    typical_duration: float
    transition_probs: np.ndarray
    
    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """Calculate Mahalanobis distance to state center."""
        diff = x - self.mean
        inv_cov = np.linalg.inv(self.covariance)
        return np.sqrt(diff.T @ inv_cov @ diff)


class GaussianHMM:
    """Gaussian Hidden Markov Model for regime detection."""
    
    def __init__(self, config: HMMConfig):
        self.config = config
        self.n_states = config.n_states
        self.n_features = None
        
        # Model parameters
        self.start_prob = None
        self.trans_prob = None
        self.means = None
        self.covars = None
        
        # State information
        self.states: List[HMMState] = []
        self.convergence_history = []
        self.is_fitted = False
        
        # Initialize random state
        self.random_state = np.random.RandomState(config.random_state)
    
    def fit(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> 'GaussianHMM':
        """
        Fit HMM to observation sequences.
        
        Args:
            X: Observations of shape (n_samples, n_features)
            lengths: Length of each sequence (for multiple sequences)
            
        Returns:
            Self
        """
        X = self._validate_input(X)
        self.n_features = X.shape[1]
        
        if lengths is None:
            lengths = [X.shape[0]]
        
        # Initialize parameters
        self._initialize_parameters(X, lengths)
        
        # Run Expectation-Maximization
        log_likelihood = -np.inf
        
        for iteration in range(self.config.n_iter):
            prev_log_likelihood = log_likelihood
            
            # E-step: compute state responsibilities
            log_likelihood, posteriors = self._e_step(X, lengths)
            
            # M-step: update parameters
            self._m_step(X, posteriors, lengths)
            
            # Check convergence
            if self.config.verbose:
                logger.info(f"Iteration {iteration}: log-likelihood = {log_likelihood:.4f}")
            
            self.convergence_history.append(log_likelihood)
            
            if log_likelihood - prev_log_likelihood < self.config.tol:
                if self.config.verbose:
                    logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        # Create state representations
        self._create_states()
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Find most likely state sequence using Viterbi algorithm.
        
        Args:
            X: Observations of shape (n_samples, n_features)
            
        Returns:
            State sequence of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)
        _, state_sequence = self._viterbi(X)
        
        return state_sequence
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probability of each state.
        
        Args:
            X: Observations of shape (n_samples, n_features)
            
        Returns:
            State probabilities of shape (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)
        _, posteriors = self._forward_backward(X)
        
        return posteriors
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.
        
        Args:
            X: Observations of shape (n_samples, n_features)
            
        Returns:
            Log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X = self._validate_input(X)
        log_likelihood, _ = self._forward_backward(X)
        
        return log_likelihood
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and prepare input data."""
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.n_features is not None and X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        return X
    
    def _initialize_parameters(self, X: np.ndarray, lengths: List[int]):
        """Initialize HMM parameters."""
        n_samples = X.shape[0]
        
        # Initialize start probabilities
        self.start_prob = np.ones(self.n_states) / self.n_states
        
        # Initialize transition matrix with slight diagonal bias
        self.trans_prob = np.ones((self.n_states, self.n_states)) / self.n_states
        np.fill_diagonal(self.trans_prob, 0.7)
        self.trans_prob /= self.trans_prob.sum(axis=1, keepdims=True)
        
        # Initialize emission parameters
        if self.config.init_method == "kmeans":
            self._kmeans_init(X)
        elif self.config.init_method == "uniform":
            self._uniform_init(X)
        else:
            self._random_init(X)
    
    def _kmeans_init(self, X: np.ndarray):
        """Initialize using k-means clustering."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.config.random_state)
        labels = kmeans.fit_predict(X)
        
        self.means = np.zeros((self.n_states, self.n_features))
        
        if self.config.covariance_type == "full":
            self.covars = np.zeros((self.n_states, self.n_features, self.n_features))
        elif self.config.covariance_type == "diag":
            self.covars = np.zeros((self.n_states, self.n_features))
        
        for i in range(self.n_states):
            mask = labels == i
            if mask.sum() > 0:
                self.means[i] = X[mask].mean(axis=0)
                
                if self.config.covariance_type == "full":
                    self.covars[i] = np.cov(X[mask].T) + self.config.min_covar * np.eye(self.n_features)
                elif self.config.covariance_type == "diag":
                    self.covars[i] = X[mask].var(axis=0) + self.config.min_covar
    
    def _uniform_init(self, X: np.ndarray):
        """Initialize with uniform spacing."""
        # Sort data by first principal component
        mean = X.mean(axis=0)
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = X @ eigvecs[:, -1]
        
        # Divide into equal segments
        percentiles = np.linspace(0, 100, self.n_states + 1)
        boundaries = np.percentile(pc1, percentiles)
        
        self.means = np.zeros((self.n_states, self.n_features))
        
        if self.config.covariance_type == "full":
            self.covars = np.zeros((self.n_states, self.n_features, self.n_features))
        elif self.config.covariance_type == "diag":
            self.covars = np.zeros((self.n_states, self.n_features))
        
        for i in range(self.n_states):
            mask = (pc1 >= boundaries[i]) & (pc1 < boundaries[i + 1])
            if i == self.n_states - 1:
                mask = pc1 >= boundaries[i]
            
            if mask.sum() > 0:
                self.means[i] = X[mask].mean(axis=0)
                
                if self.config.covariance_type == "full":
                    self.covars[i] = np.cov(X[mask].T) + self.config.min_covar * np.eye(self.n_features)
                elif self.config.covariance_type == "diag":
                    self.covars[i] = X[mask].var(axis=0) + self.config.min_covar
    
    def _random_init(self, X: np.ndarray):
        """Random initialization."""
        indices = self.random_state.choice(X.shape[0], self.n_states, replace=False)
        self.means = X[indices].copy()
        
        global_cov = np.cov(X.T)
        if self.config.covariance_type == "full":
            self.covars = np.array([global_cov + self.config.min_covar * np.eye(self.n_features) 
                                   for _ in range(self.n_states)])
        elif self.config.covariance_type == "diag":
            self.covars = np.array([np.diag(global_cov) + self.config.min_covar 
                                   for _ in range(self.n_states)])
    
    def _e_step(self, X: np.ndarray, lengths: List[int]) -> Tuple[float, np.ndarray]:
        """Expectation step of EM algorithm."""
        curr_idx = 0
        log_likelihood = 0.0
        posteriors = np.zeros((X.shape[0], self.n_states))
        
        for length in lengths:
            X_seq = X[curr_idx:curr_idx + length]
            log_prob, post = self._forward_backward(X_seq)
            
            log_likelihood += log_prob
            posteriors[curr_idx:curr_idx + length] = post
            curr_idx += length
        
        return log_likelihood, posteriors
    
    def _m_step(self, X: np.ndarray, posteriors: np.ndarray, lengths: List[int]):
        """Maximization step of EM algorithm."""
        # Update start probabilities
        self.start_prob = posteriors[0].copy()
        
        # Update transition probabilities
        curr_idx = 0
        trans_num = np.zeros((self.n_states, self.n_states))
        trans_den = np.zeros(self.n_states)
        
        for length in lengths:
            if length <= 1:
                continue
                
            post_seq = posteriors[curr_idx:curr_idx + length]
            
            for t in range(length - 1):
                trans_num += np.outer(post_seq[t], post_seq[t + 1])
                trans_den += post_seq[t]
            
            curr_idx += length
        
        # Add transition penalty to encourage stability
        trans_num += self.config.transition_penalty
        trans_den += self.config.transition_penalty * self.n_states
        
        self.trans_prob = trans_num / trans_den[:, np.newaxis]
        
        # Update emission parameters
        post_sum = posteriors.sum(axis=0)
        
        for i in range(self.n_states):
            if post_sum[i] < 1e-10:
                continue
            
            # Update mean
            self.means[i] = (posteriors[:, i:i+1] * X).sum(axis=0) / post_sum[i]
            
            # Update covariance
            diff = X - self.means[i]
            
            if self.config.covariance_type == "full":
                weighted_diff = diff * posteriors[:, i:i+1]
                self.covars[i] = (weighted_diff.T @ diff) / post_sum[i]
                self.covars[i] += self.config.min_covar * np.eye(self.n_features)
            elif self.config.covariance_type == "diag":
                self.covars[i] = (posteriors[:, i:i+1] * diff**2).sum(axis=0) / post_sum[i]
                self.covars[i] += self.config.min_covar
    
    def _forward_backward(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """Forward-backward algorithm for computing posteriors."""
        n_samples = X.shape[0]
        
        # Compute log emission probabilities
        log_emit_prob = self._compute_log_likelihood(X)
        
        # Forward pass
        log_alpha = np.zeros((n_samples, self.n_states))
        log_alpha[0] = np.log(self.start_prob) + log_emit_prob[0]
        
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + np.log(self.trans_prob[:, j])
                ) + log_emit_prob[t, j]
        
        # Log likelihood
        log_likelihood = logsumexp(log_alpha[-1])
        
        # Backward pass
        log_beta = np.zeros((n_samples, self.n_states))
        log_beta[-1] = 0  # log(1) = 0
        
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.trans_prob[i]) + log_emit_prob[t+1] + log_beta[t+1]
                )
        
        # Compute posteriors
        log_gamma = log_alpha + log_beta - log_likelihood
        posteriors = np.exp(log_gamma)
        
        return log_likelihood, posteriors
    
    def _viterbi(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """Viterbi algorithm for finding most likely state sequence."""
        n_samples = X.shape[0]
        
        # Compute log emission probabilities
        log_emit_prob = self._compute_log_likelihood(X)
        
        # Initialize
        log_delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)
        
        log_delta[0] = np.log(self.start_prob) + log_emit_prob[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                trans_scores = log_delta[t-1] + np.log(self.trans_prob[:, j])
                psi[t, j] = np.argmax(trans_scores)
                log_delta[t, j] = trans_scores[psi[t, j]] + log_emit_prob[t, j]
        
        # Backward pass
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        log_likelihood = log_delta[-1, states[-1]]
        
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return log_likelihood, states
    
    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities."""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_states))
        
        for i in range(self.n_states):
            if self.config.covariance_type == "full":
                log_prob[:, i] = stats.multivariate_normal.logpdf(
                    X, self.means[i], self.covars[i]
                )
            elif self.config.covariance_type == "diag":
                # Diagonal covariance
                log_prob[:, i] = -0.5 * (
                    self.n_features * np.log(2 * np.pi) +
                    np.sum(np.log(self.covars[i])) +
                    np.sum((X - self.means[i])**2 / self.covars[i], axis=1)
                )
        
        return log_prob
    
    def _create_states(self):
        """Create HMMState objects from fitted parameters."""
        self.states = []
        
        # Classify states based on their characteristics
        regime_types = self._classify_regimes()
        
        # Calculate typical durations
        durations = self._estimate_state_durations()
        
        for i in range(self.n_states):
            if self.config.covariance_type == "diag":
                covar = np.diag(self.covars[i])
            else:
                covar = self.covars[i]
            
            state = HMMState(
                mean=self.means[i],
                covariance=covar,
                regime_type=regime_types[i],
                typical_duration=durations[i],
                transition_probs=self.trans_prob[i]
            )
            self.states.append(state)
    
    def _classify_regimes(self) -> List[RegimeType]:
        """Classify states into regime types based on characteristics."""
        regime_types = []
        
        # Extract key characteristics (assuming first feature is returns)
        returns = self.means[:, 0] if self.n_features > 0 else np.zeros(self.n_states)
        
        # Extract volatility (assuming second feature or from covariance)
        if self.n_features > 1:
            volatilities = self.means[:, 1]
        else:
            if self.config.covariance_type == "full":
                volatilities = np.sqrt(np.diag(self.covars[:, 0, 0]))
            else:
                volatilities = np.sqrt(self.covars[:, 0])
        
        # Sort states by return/volatility characteristics
        for i in range(self.n_states):
            if returns[i] > 0.05 and volatilities[i] < np.median(volatilities):
                regime_types.append(RegimeType.BULL)
            elif returns[i] < -0.05 and volatilities[i] > np.median(volatilities):
                regime_types.append(RegimeType.BEAR)
            elif returns[i] < -0.10 and volatilities[i] > np.percentile(volatilities, 75):
                regime_types.append(RegimeType.CRISIS)
            elif returns[i] > 0 and volatilities[i] < np.percentile(volatilities, 75):
                regime_types.append(RegimeType.RECOVERY)
            else:
                regime_types.append(RegimeType.SIDEWAYS)
        
        return regime_types
    
    def _estimate_state_durations(self) -> np.ndarray:
        """Estimate typical state durations from transition matrix."""
        # Expected duration = 1 / (1 - self-transition probability)
        self_trans_probs = np.diag(self.trans_prob)
        durations = 1.0 / (1.0 - self_trans_probs + 1e-10)
        
        return durations
    
    def get_regime_statistics(self, states: np.ndarray) -> pd.DataFrame:
        """Calculate statistics for each regime."""
        stats_list = []
        
        for i in range(self.n_states):
            mask = states == i
            if mask.sum() == 0:
                continue
            
            # Calculate regime statistics
            stats_dict = {
                'regime': i,
                'type': self.states[i].regime_type.value,
                'frequency': mask.mean(),
                'avg_duration': self.states[i].typical_duration,
                'return_mean': self.states[i].mean[0] if self.n_features > 0 else 0,
                'volatility': np.sqrt(self.states[i].covariance[0, 0]) if self.n_features > 0 else 0
            }
            
            stats_list.append(stats_dict)
        
        return pd.DataFrame(stats_list)


class RegimeSwitchingGARCH(GaussianHMM):
    """Markov Switching GARCH model for volatility regime detection."""
    
    def __init__(self, config: HMMConfig):
        super().__init__(config)
        self.garch_params = {}
        
    def _m_step(self, X: np.ndarray, posteriors: np.ndarray, lengths: List[int]):
        """Extended M-step with GARCH parameter estimation."""
        # First do standard M-step
        super()._m_step(X, posteriors, lengths)
        
        # Then estimate GARCH parameters for each state
        for i in range(self.n_states):
            self._estimate_garch_params(X, posteriors[:, i], i)
    
    def _estimate_garch_params(self, returns: np.ndarray, weights: np.ndarray, state: int):
        """Estimate GARCH(1,1) parameters for a given state."""
        # Simplified GARCH parameter estimation
        # In practice, use arch package or similar
        
        # Weight observations by state probability
        weighted_returns = returns * np.sqrt(weights[:, np.newaxis])
        
        # Estimate unconditional variance
        variance = np.average((returns ** 2), weights=weights, axis=0)
        
        # Store simplified GARCH parameters
        self.garch_params[state] = {
            'omega': variance * 0.1,  # Long-run variance weight
            'alpha': 0.1,  # ARCH coefficient
            'beta': 0.85,  # GARCH coefficient
            'unconditional_var': variance
        }


class HierarchicalHMM(GaussianHMM):
    """Hierarchical HMM for multi-scale regime detection."""
    
    def __init__(self, config: HMMConfig, n_levels: int = 2):
        super().__init__(config)
        self.n_levels = n_levels
        self.sub_models = {}
        
    def fit(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> 'HierarchicalHMM':
        """Fit hierarchical HMM."""
        # Fit top-level model
        super().fit(X, lengths)
        
        # Fit sub-models for each top-level state
        states = self.predict(X)
        
        for i in range(self.n_states):
            mask = states == i
            if mask.sum() > 10:  # Minimum samples for sub-model
                sub_config = HMMConfig(
                    n_states=3,  # Fewer states for sub-models
                    covariance_type=self.config.covariance_type,
                    n_iter=self.config.n_iter,
                    random_state=self.config.random_state + i
                )
                
                sub_model = GaussianHMM(sub_config)
                sub_model.fit(X[mask])
                self.sub_models[i] = sub_model
        
        return self
    
    def predict_hierarchical(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Predict states at all hierarchy levels."""
        # Top-level prediction
        top_states = self.predict(X)
        
        # Sub-level predictions
        sub_states = {}
        for i in range(self.n_states):
            if i in self.sub_models:
                mask = top_states == i
                if mask.sum() > 0:
                    sub_pred = self.sub_models[i].predict(X[mask])
                    sub_states[i] = sub_pred
        
        return top_states, sub_states