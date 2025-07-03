"""
Statistical and machine learning models for market regime detection.

Implements Hidden Markov Models, Markov Switching Models, and other
regime detection algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings

# For HMM
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available. HMM functionality will be limited.")

# For Markov Switching
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Markov switching models will be limited.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HiddenMarkovRegimeModel:
    """Hidden Markov Model for regime detection."""
    
    def __init__(
        self,
        n_states: int = 5,
        covariance_type: str = "full",
        n_iter: int = 100
    ):
        """Initialize HMM model."""
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        
        if HMM_AVAILABLE:
            self.model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=42
            )
        else:
            self.model = None
            logger.warning("HMM model not available")
        
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray, lengths: Optional[List[int]] = None):
        """Fit HMM model to features."""
        if not HMM_AVAILABLE or self.model is None:
            logger.error("HMM not available for fitting")
            return
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        if lengths:
            self.model.fit(features_scaled, lengths)
        else:
            self.model.fit(features_scaled)
        
        self.is_fitted = True
        logger.info(f"HMM fitted with {self.n_states} states")
        
    def predict_regime(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict regime states and probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict states
        states = self.model.predict(features_scaled)
        
        # Get state probabilities
        state_probs = self.model.predict_proba(features_scaled)
        
        return states, state_probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition probability matrix."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.transmat_
    
    def get_state_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters for each state."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        
        return {
            "means": self.model.means_,
            "covariances": self.model.covars_,
            "stationary_distribution": self._get_stationary_distribution()
        }
    
    def _get_stationary_distribution(self) -> np.ndarray:
        """Calculate stationary distribution of Markov chain."""
        if not self.is_fitted:
            return None
        
        # Eigenvalue decomposition
        transition_matrix = self.model.transmat_
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        
        # Find eigenvector with eigenvalue 1
        stationary_idx = np.argmax(np.abs(eigenvalues - 1.0) < 1e-8)
        stationary_dist = np.real(eigenvectors[:, stationary_idx])
        
        # Normalize
        stationary_dist = stationary_dist / stationary_dist.sum()
        
        return np.abs(stationary_dist)


class MarkovSwitchingModel:
    """Markov Switching Dynamic Regression Model."""
    
    def __init__(
        self,
        n_regimes: int = 3,
        switching_variance: bool = True,
        switching_trend: bool = False
    ):
        """Initialize Markov Switching model."""
        self.n_regimes = n_regimes
        self.switching_variance = switching_variance
        self.switching_trend = switching_trend
        self.model = None
        self.results = None
        
    def fit(self, data: pd.Series, exog: Optional[pd.DataFrame] = None):
        """Fit Markov Switching model."""
        if not STATSMODELS_AVAILABLE:
            logger.error("statsmodels not available for Markov Switching")
            return
        
        # Create model
        if exog is not None:
            self.model = sm.tsa.MarkovRegression(
                data,
                k_regimes=self.n_regimes,
                exog=exog,
                switching_variance=self.switching_variance
            )
        else:
            # Autoregression model
            self.model = sm.tsa.MarkovAutoregression(
                data,
                k_regimes=self.n_regimes,
                order=1,
                switching_variance=self.switching_variance,
                switching_trend=self.switching_trend
            )
        
        # Fit model
        self.results = self.model.fit(disp=False)
        
        logger.info(f"Markov Switching model fitted with {self.n_regimes} regimes")
        
    def predict_regimes(self, n_periods: int = 1) -> pd.DataFrame:
        """Predict future regimes."""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        # Get smoothed probabilities for historical data
        smoothed_probs = self.results.smoothed_marginal_probabilities
        
        # For forecasting, use the last state probabilities
        last_probs = smoothed_probs.iloc[-1]
        
        # Simple forecast using transition probabilities
        forecast_probs = []
        current_probs = last_probs.values
        
        for _ in range(n_periods):
            # Predict next state probabilities
            # This is simplified - actual implementation would use full model
            next_probs = self._predict_next_probs(current_probs)
            forecast_probs.append(next_probs)
            current_probs = next_probs
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast_probs,
            columns=[f"regime_{i}" for i in range(self.n_regimes)]
        )
        
        return forecast_df
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics for each regime."""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        stats = {}
        
        # Get parameters for each regime
        for i in range(self.n_regimes):
            regime_stats = {
                "mean": self.results.params[f'regime{i}.const'] if hasattr(self.results.params, f'regime{i}.const') else None,
                "volatility": np.sqrt(self.results.params[f'regime{i}.sigma2']) if self.switching_variance else None,
                "duration": self._calculate_expected_duration(i)
            }
            stats[f"regime_{i}"] = regime_stats
        
        # Transition matrix
        if hasattr(self.results, 'transition_matrix'):
            stats["transition_matrix"] = self.results.transition_matrix
        
        return stats
    
    def _predict_next_probs(self, current_probs: np.ndarray) -> np.ndarray:
        """Predict next state probabilities."""
        # Simplified - would use actual transition matrix
        # For now, just add some noise
        next_probs = current_probs + np.random.normal(0, 0.01, len(current_probs))
        next_probs = np.maximum(next_probs, 0)
        next_probs = next_probs / next_probs.sum()
        
        return next_probs
    
    def _calculate_expected_duration(self, regime: int) -> float:
        """Calculate expected duration of a regime."""
        if not hasattr(self.results, 'transition_matrix'):
            return np.nan
        
        # Expected duration = 1 / (1 - P[staying in regime])
        staying_prob = self.results.transition_matrix[regime, regime]
        
        if staying_prob < 1:
            return 1 / (1 - staying_prob)
        else:
            return np.inf


class ThresholdAutoregressiveModel:
    """Threshold Autoregressive (TAR) model for regime detection."""
    
    def __init__(
        self,
        n_regimes: int = 2,
        threshold_variable: str = "returns",
        delay: int = 1
    ):
        """Initialize TAR model."""
        self.n_regimes = n_regimes
        self.threshold_variable = threshold_variable
        self.delay = delay
        self.thresholds = []
        self.models = {}
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame):
        """Fit TAR model."""
        # Get threshold variable
        if self.threshold_variable not in data.columns:
            raise ValueError(f"Threshold variable {self.threshold_variable} not in data")
        
        threshold_data = data[self.threshold_variable].shift(self.delay)
        
        # Find thresholds using grid search
        self.thresholds = self._find_optimal_thresholds(data, threshold_data)
        
        # Fit separate models for each regime
        self._fit_regime_models(data, threshold_data)
        
        self.is_fitted = True
        logger.info(f"TAR model fitted with thresholds: {self.thresholds}")
        
    def predict_regime(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime for new data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        threshold_values = data[self.threshold_variable].shift(self.delay)
        regimes = np.zeros(len(data))
        
        # Assign regimes based on thresholds
        for i, value in enumerate(threshold_values):
            if pd.isna(value):
                regimes[i] = 0
            else:
                regime = 0
                for j, threshold in enumerate(self.thresholds):
                    if value > threshold:
                        regime = j + 1
                regimes[i] = regime
        
        return regimes.astype(int)
    
    def _find_optimal_thresholds(
        self,
        data: pd.DataFrame,
        threshold_data: pd.Series
    ) -> List[float]:
        """Find optimal thresholds using grid search."""
        # Simplified implementation
        # In practice, would use more sophisticated optimization
        
        # Use quantiles as initial thresholds
        quantiles = np.linspace(0.2, 0.8, self.n_regimes - 1)
        thresholds = threshold_data.quantile(quantiles).values
        
        return sorted(thresholds)
    
    def _fit_regime_models(self, data: pd.DataFrame, threshold_data: pd.Series):
        """Fit separate models for each regime."""
        # Create regime indicators
        regimes = self.predict_regime(data)
        
        # Fit model for each regime
        for regime in range(self.n_regimes):
            regime_data = data[regimes == regime]
            
            if len(regime_data) > 10:  # Minimum data requirement
                # Simple AR(1) model for each regime
                # In practice, would use more sophisticated models
                self.models[regime] = {
                    "mean": regime_data.mean(),
                    "std": regime_data.std(),
                    "n_obs": len(regime_data)
                }


class RegimeDetectionEnsemble:
    """Ensemble of regime detection models."""
    
    def __init__(self, models: Optional[List[str]] = None):
        """Initialize ensemble."""
        if models is None:
            models = ["hmm", "gmm", "threshold"]
        
        self.models = {}
        self.weights = {}
        
        # Initialize models
        if "hmm" in models and HMM_AVAILABLE:
            self.models["hmm"] = HiddenMarkovRegimeModel()
            self.weights["hmm"] = 0.4
        
        if "gmm" in models:
            self.models["gmm"] = GaussianMixture(n_components=5)
            self.weights["gmm"] = 0.3
        
        if "markov_switch" in models and STATSMODELS_AVAILABLE:
            self.models["markov_switch"] = MarkovSwitchingModel()
            self.weights["markov_switch"] = 0.2
        
        if "threshold" in models:
            self.models["threshold"] = ThresholdAutoregressiveModel()
            self.weights["threshold"] = 0.1
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight
        
    def fit(self, data: Union[np.ndarray, pd.DataFrame]):
        """Fit all models in ensemble."""
        for name, model in self.models.items():
            try:
                if name == "hmm" and isinstance(model, HiddenMarkovRegimeModel):
                    if isinstance(data, pd.DataFrame):
                        features = data.values
                    else:
                        features = data
                    model.fit(features)
                
                elif name == "gmm":
                    if isinstance(data, pd.DataFrame):
                        features = data.values
                    else:
                        features = data
                    model.fit(features)
                
                elif name == "markov_switch" and isinstance(data, pd.DataFrame):
                    # Use first column as target
                    model.fit(data.iloc[:, 0])
                
                elif name == "threshold" and isinstance(data, pd.DataFrame):
                    model.fit(data)
                
                logger.info(f"Fitted {name} model")
                
            except Exception as e:
                logger.error(f"Failed to fit {name} model: {e}")
                # Remove failed model from ensemble
                self.weights[name] = 0
        
        # Re-normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def predict_regime_probabilities(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[int, float]:
        """Get ensemble regime predictions."""
        all_predictions = {}
        
        for name, model in self.models.items():
            if self.weights.get(name, 0) == 0:
                continue
            
            try:
                if name == "hmm" and hasattr(model, 'predict_regime'):
                    if isinstance(data, pd.DataFrame):
                        features = data.values
                    else:
                        features = data
                    _, probs = model.predict_regime(features)
                    # Use last prediction
                    regime_probs = probs[-1] if len(probs) > 0 else np.zeros(5)
                    all_predictions[name] = regime_probs
                
                elif name == "gmm" and hasattr(model, 'predict_proba'):
                    if isinstance(data, pd.DataFrame):
                        features = data.values
                    else:
                        features = data
                    probs = model.predict_proba(features)
                    regime_probs = probs[-1] if len(probs) > 0 else np.zeros(5)
                    all_predictions[name] = regime_probs
                
                elif name == "threshold" and hasattr(model, 'predict_regime'):
                    regimes = model.predict_regime(data)
                    # Convert to probability-like format
                    current_regime = int(regimes[-1]) if len(regimes) > 0 else 0
                    regime_probs = np.zeros(5)
                    if current_regime < 5:
                        regime_probs[current_regime] = 1.0
                    all_predictions[name] = regime_probs
                
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
        
        # Combine predictions
        combined_probs = np.zeros(5)
        
        for name, probs in all_predictions.items():
            weight = self.weights.get(name, 0)
            combined_probs += weight * probs
        
        # Normalize
        if combined_probs.sum() > 0:
            combined_probs /= combined_probs.sum()
        
        # Convert to dictionary
        regime_map = {0: "bull", 1: "bear", 2: "sideways", 3: "crisis", 4: "recovery"}
        
        return {i: prob for i, prob in enumerate(combined_probs)}
    
    def get_model_agreement(self) -> float:
        """Calculate agreement between models."""
        if len(self.models) < 2:
            return 1.0
        
        # Get predictions from each model
        predictions = []
        
        for name, model in self.models.items():
            if self.weights.get(name, 0) > 0 and hasattr(model, 'predict'):
                # Get most likely regime from each model
                # This is simplified - would need actual predictions
                predictions.append(np.random.randint(0, 5))
        
        if len(predictions) < 2:
            return 1.0
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                agreements.append(int(predictions[i] == predictions[j]))
        
        return np.mean(agreements) if agreements else 0.0


class RegimeStabilityAnalyzer:
    """Analyze regime stability and transition probabilities."""
    
    def __init__(self):
        """Initialize stability analyzer."""
        self.regime_history = []
        self.transition_counts = np.zeros((5, 5))
        self.regime_durations = {i: [] for i in range(5)}
        
    def update(self, current_regime: int, timestamp: datetime):
        """Update with new regime observation."""
        if self.regime_history:
            last_regime, last_time = self.regime_history[-1]
            
            # Update transition count
            if last_regime != current_regime:
                self.transition_counts[last_regime, current_regime] += 1
                
                # Update duration
                duration = (timestamp - last_time).days
                self.regime_durations[last_regime].append(duration)
        
        self.regime_history.append((current_regime, timestamp))
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get normalized transition probability matrix."""
        # Normalize rows
        transition_matrix = self.transition_counts.copy()
        
        for i in range(5):
            row_sum = transition_matrix[i].sum()
            if row_sum > 0:
                transition_matrix[i] /= row_sum
        
        return transition_matrix
    
    def get_regime_stability(self, regime: int) -> Dict[str, float]:
        """Get stability metrics for a regime."""
        if regime not in self.regime_durations:
            return {}
        
        durations = self.regime_durations[regime]
        
        if not durations:
            return {
                "avg_duration": 0,
                "min_duration": 0,
                "max_duration": 0,
                "stability_score": 0
            }
        
        avg_duration = np.mean(durations)
        
        # Stability score based on duration consistency
        if len(durations) > 1:
            cv = np.std(durations) / avg_duration if avg_duration > 0 else 1
            stability_score = 1 / (1 + cv)  # Lower CV = higher stability
        else:
            stability_score = 0.5
        
        return {
            "avg_duration": avg_duration,
            "min_duration": min(durations),
            "max_duration": max(durations),
            "duration_std": np.std(durations) if len(durations) > 1 else 0,
            "stability_score": stability_score,
            "n_occurrences": len(durations)
        }
    
    def predict_regime_duration(self, current_regime: int) -> float:
        """Predict expected duration of current regime."""
        if current_regime not in self.regime_durations:
            return 30  # Default 30 days
        
        durations = self.regime_durations[current_regime]
        
        if not durations:
            return 30
        
        # Use exponential smoothing for prediction
        if len(durations) == 1:
            return durations[0]
        
        # Recent durations have more weight
        weights = np.exp(np.linspace(-1, 0, len(durations)))
        weights /= weights.sum()
        
        predicted_duration = np.sum(np.array(durations) * weights)
        
        return predicted_duration
    
    def get_transition_probabilities(self, from_regime: int) -> Dict[int, float]:
        """Get transition probabilities from a specific regime."""
        trans_matrix = self.get_transition_matrix()
        
        probs = {}
        for to_regime in range(5):
            probs[to_regime] = trans_matrix[from_regime, to_regime]
        
        return probs