"""
HMM Optimization Utilities.

This module provides optimization utilities for Hidden Markov Models including
hyperparameter tuning, model selection, and performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import silhouette_score
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from ..ml.regime.hmm_regime_detector import GaussianHMM, HMMConfig
from ..ml.regime.regime_features import RegimeFeatureEngineer, RegimeFeatureConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class HMMOptimizationResult:
    """Results from HMM optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_history: List[float]
    convergence_plot: Optional[Any] = None
    
    def get_best_config(self) -> HMMConfig:
        """Create HMMConfig from best parameters."""
        config = HMMConfig()
        for key, value in self.best_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class HMMOptimizer:
    """Optimize HMM hyperparameters for regime detection."""
    
    def __init__(self, 
                 feature_engineer: Optional[RegimeFeatureEngineer] = None,
                 n_jobs: int = -1):
        self.feature_engineer = feature_engineer or RegimeFeatureEngineer()
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.optimization_results: List[HMMOptimizationResult] = []
    
    def optimize_n_states(self,
                         features: np.ndarray,
                         min_states: int = 2,
                         max_states: int = 10,
                         cv_splits: int = 3) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of hidden states using cross-validation.
        
        Args:
            features: Feature matrix
            min_states: Minimum number of states to test
            max_states: Maximum number of states to test
            cv_splits: Number of cross-validation splits
            
        Returns:
            Tuple of (optimal_n_states, scores_dict)
        """
        scores = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        for n_states in range(min_states, max_states + 1):
            logger.info(f"Testing {n_states} states...")
            
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(features):
                train_data = features[train_idx]
                val_data = features[val_idx]
                
                # Fit model
                config = HMMConfig(n_states=n_states, n_iter=50, verbose=False)
                model = GaussianHMM(config)
                
                try:
                    model.fit(train_data)
                    
                    # Calculate validation score (log-likelihood)
                    val_score = model.score(val_data)
                    
                    # Add BIC penalty for model complexity
                    n_params = self._count_parameters(model)
                    bic_penalty = 0.5 * n_params * np.log(len(val_data))
                    
                    cv_scores.append(val_score - bic_penalty)
                    
                except Exception as e:
                    logger.warning(f"Failed to fit model with {n_states} states: {e}")
                    cv_scores.append(-np.inf)
            
            avg_score = np.mean(cv_scores)
            scores[n_states] = avg_score
            logger.info(f"  Average score: {avg_score:.3f}")
        
        # Find optimal number of states
        optimal_n_states = max(scores, key=scores.get)
        
        return optimal_n_states, scores
    
    def optimize_hyperparameters(self,
                               features: np.ndarray,
                               param_grid: Optional[Dict[str, List[Any]]] = None,
                               n_trials: int = 100,
                               use_optuna: bool = True) -> HMMOptimizationResult:
        """
        Optimize HMM hyperparameters using grid search or Bayesian optimization.
        
        Args:
            features: Feature matrix
            param_grid: Parameter grid for grid search
            n_trials: Number of trials for Optuna
            use_optuna: Whether to use Optuna (True) or grid search (False)
            
        Returns:
            HMMOptimizationResult
        """
        if param_grid is None:
            param_grid = {
                'n_states': [3, 4, 5, 6],
                'covariance_type': ['full', 'diag'],
                'init_method': ['kmeans', 'uniform'],
                'min_covar': [1e-4, 1e-3, 1e-2]
            }
        
        if use_optuna:
            return self._optimize_with_optuna(features, n_trials)
        else:
            return self._optimize_with_grid_search(features, param_grid)
    
    def _optimize_with_optuna(self, 
                            features: np.ndarray,
                            n_trials: int) -> HMMOptimizationResult:
        """Optimize using Optuna Bayesian optimization."""
        
        def objective(trial):
            # Sample hyperparameters
            n_states = trial.suggest_int('n_states', 3, 8)
            covariance_type = trial.suggest_categorical(
                'covariance_type', ['full', 'diag']
            )
            init_method = trial.suggest_categorical(
                'init_method', ['kmeans', 'uniform', 'random']
            )
            min_covar = trial.suggest_loguniform('min_covar', 1e-5, 1e-2)
            transition_penalty = trial.suggest_uniform('transition_penalty', 0.0, 0.1)
            
            # Create config
            config = HMMConfig(
                n_states=n_states,
                covariance_type=covariance_type,
                init_method=init_method,
                min_covar=min_covar,
                transition_penalty=transition_penalty,
                n_iter=100,
                verbose=False
            )
            
            # Evaluate using cross-validation
            score = self._evaluate_config(config, features)
            
            return score
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)
        
        # Get results
        best_params = study.best_params
        best_score = study.best_value
        
        # Collect all results
        all_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = trial.params.copy()
                result['score'] = trial.value
                all_results.append(result)
        
        # Get optimization history
        optimization_history = [t.value for t in study.trials 
                              if t.state == optuna.trial.TrialState.COMPLETE]
        
        return HMMOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_history=optimization_history
        )
    
    def _optimize_with_grid_search(self,
                                 features: np.ndarray,
                                 param_grid: Dict[str, List[Any]]) -> HMMOptimizationResult:
        """Optimize using grid search."""
        all_results = []
        best_score = -np.inf
        best_params = None
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Parallel evaluation
        if self.n_jobs:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {}
                
                for params in param_combinations:
                    config = HMMConfig(**params)
                    future = executor.submit(self._evaluate_config, config, features)
                    futures[future] = params
                
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        score = future.result()
                        result = params.copy()
                        result['score'] = score
                        all_results.append(result)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            
                    except Exception as e:
                        logger.warning(f"Failed to evaluate params {params}: {e}")
        else:
            # Sequential evaluation
            for params in param_combinations:
                config = HMMConfig(**params)
                score = self._evaluate_config(config, features)
                
                result = params.copy()
                result['score'] = score
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        optimization_history = [r['score'] for r in all_results]
        
        return HMMOptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_history=optimization_history
        )
    
    def _evaluate_config(self, config: HMMConfig, features: np.ndarray) -> float:
        """Evaluate a single configuration using cross-validation."""
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(features):
            train_data = features[train_idx]
            val_data = features[val_idx]
            
            model = GaussianHMM(config)
            
            try:
                model.fit(train_data)
                
                # Multiple evaluation metrics
                log_likelihood = model.score(val_data)
                
                # Penalize model complexity
                n_params = self._count_parameters(model)
                bic_penalty = 0.5 * n_params * np.log(len(val_data))
                
                # Add stability score (penalize frequent regime changes)
                predicted_states = model.predict(val_data)
                regime_changes = np.diff(predicted_states) != 0
                stability_penalty = -0.1 * (regime_changes.sum() / len(regime_changes))
                
                score = log_likelihood - bic_penalty + stability_penalty
                scores.append(score)
                
            except Exception as e:
                logger.debug(f"Model evaluation failed: {e}")
                scores.append(-np.inf)
        
        return np.mean(scores)
    
    def _count_parameters(self, model: GaussianHMM) -> int:
        """Count number of parameters in HMM."""
        n_states = model.n_states
        n_features = model.n_features
        
        # Start probabilities
        n_params = n_states - 1
        
        # Transition probabilities
        n_params += n_states * (n_states - 1)
        
        # Emission parameters
        n_params += n_states * n_features  # Means
        
        if model.config.covariance_type == 'full':
            n_params += n_states * n_features * (n_features + 1) // 2
        elif model.config.covariance_type == 'diag':
            n_params += n_states * n_features
        
        return n_params
    
    def analyze_model_stability(self, 
                              model: GaussianHMM,
                              features: np.ndarray,
                              n_bootstrap: int = 20) -> Dict[str, Any]:
        """
        Analyze model stability using bootstrap resampling.
        
        Args:
            model: Fitted HMM model
            features: Feature matrix
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with stability metrics
        """
        n_samples = len(features)
        state_predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            indices.sort()  # Maintain temporal order
            bootstrap_features = features[indices]
            
            # Refit model
            bootstrap_model = GaussianHMM(model.config)
            try:
                bootstrap_model.fit(bootstrap_features)
                states = bootstrap_model.predict(features)
                state_predictions.append(states)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
        
        if not state_predictions:
            return {'stability_score': 0.0, 'error': 'All bootstrap iterations failed'}
        
        # Calculate stability metrics
        state_predictions = np.array(state_predictions)
        
        # Agreement between predictions
        mode_states = []
        agreement_scores = []
        
        for t in range(n_samples):
            states_at_t = state_predictions[:, t]
            unique, counts = np.unique(states_at_t, return_counts=True)
            mode_state = unique[np.argmax(counts)]
            mode_states.append(mode_state)
            agreement_scores.append(np.max(counts) / len(states_at_t))
        
        stability_metrics = {
            'stability_score': np.mean(agreement_scores),
            'min_agreement': np.min(agreement_scores),
            'max_agreement': np.max(agreement_scores),
            'unstable_periods': np.sum(np.array(agreement_scores) < 0.7),
            'mode_states': mode_states
        }
        
        return stability_metrics
    
    def optimize_feature_selection(self,
                                 features: pd.DataFrame,
                                 base_config: HMMConfig,
                                 min_features: int = 5) -> Tuple[List[str], float]:
        """
        Optimize feature selection for HMM.
        
        Args:
            features: Feature DataFrame
            base_config: Base HMM configuration
            min_features: Minimum number of features to keep
            
        Returns:
            Tuple of (selected_features, score)
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # Fit initial model with all features
        initial_model = GaussianHMM(base_config)
        initial_model.fit(features.values)
        initial_states = initial_model.predict(features.values)
        
        # Calculate mutual information between features and states
        mi_scores = []
        for col in features.columns:
            mi = mutual_info_regression(
                features[[col]].values, 
                initial_states, 
                random_state=42
            )[0]
            mi_scores.append((col, mi))
        
        # Sort by importance
        mi_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Try different feature subsets
        best_score = -np.inf
        best_features = []
        
        for n_features in range(min_features, len(features.columns) + 1):
            selected_features = [f[0] for f in mi_scores[:n_features]]
            selected_data = features[selected_features].values
            
            score = self._evaluate_config(base_config, selected_data)
            
            if score > best_score:
                best_score = score
                best_features = selected_features
        
        return best_features, best_score


class HMMModelSelector:
    """Select best HMM variant for given data."""
    
    def __init__(self):
        self.models = {}
        self.scores = {}
    
    def compare_models(self,
                      features: np.ndarray,
                      base_config: HMMConfig,
                      model_variants: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare different HMM model variants.
        
        Args:
            features: Feature matrix
            base_config: Base configuration
            model_variants: List of variants to test
            
        Returns:
            Dictionary with comparison results
        """
        if model_variants is None:
            model_variants = ['gaussian', 'garch', 'hierarchical']
        
        results = {}
        
        for variant in model_variants:
            logger.info(f"Testing {variant} HMM...")
            
            try:
                if variant == 'gaussian':
                    from ..ml.regime.hmm_regime_detector import GaussianHMM
                    model = GaussianHMM(base_config)
                
                elif variant == 'garch':
                    from ..ml.regime.hmm_regime_detector import RegimeSwitchingGARCH
                    garch_config = base_config
                    garch_config.use_regime_switching_garch = True
                    model = RegimeSwitchingGARCH(garch_config)
                
                elif variant == 'hierarchical':
                    from ..ml.regime.hmm_regime_detector import HierarchicalHMM
                    model = HierarchicalHMM(base_config, n_levels=2)
                
                else:
                    logger.warning(f"Unknown model variant: {variant}")
                    continue
                
                # Fit and evaluate
                model.fit(features)
                
                # Calculate metrics
                log_likelihood = model.score(features)
                n_params = self._count_model_parameters(model, features.shape[1])
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(len(features)) - 2 * log_likelihood
                
                # Predict states
                states = model.predict(features)
                
                # Calculate silhouette score if enough states
                unique_states = np.unique(states)
                if len(unique_states) > 1:
                    silhouette = silhouette_score(features, states)
                else:
                    silhouette = 0.0
                
                results[variant] = {
                    'model': model,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'silhouette_score': silhouette,
                    'n_parameters': n_params
                }
                
                self.models[variant] = model
                self.scores[variant] = bic  # Use BIC for selection
                
            except Exception as e:
                logger.error(f"Failed to evaluate {variant} model: {e}")
                results[variant] = {'error': str(e)}
        
        # Select best model
        if self.scores:
            best_variant = min(self.scores, key=self.scores.get)
            results['best_model'] = best_variant
        
        return results
    
    def _count_model_parameters(self, model: Any, n_features: int) -> int:
        """Count parameters for different model types."""
        if hasattr(model, 'n_states'):
            n_states = model.n_states
        else:
            n_states = 5  # Default
        
        # Base HMM parameters
        n_params = n_states - 1  # Start probabilities
        n_params += n_states * (n_states - 1)  # Transition matrix
        n_params += n_states * n_features  # Means
        
        # Covariance parameters
        if hasattr(model, 'config') and hasattr(model.config, 'covariance_type'):
            if model.config.covariance_type == 'full':
                n_params += n_states * n_features * (n_features + 1) // 2
            else:
                n_params += n_states * n_features
        else:
            n_params += n_states * n_features
        
        # Additional parameters for variants
        if hasattr(model, 'garch_params'):
            n_params += n_states * 4  # GARCH parameters per state
        
        if hasattr(model, 'sub_models'):
            n_params += len(model.sub_models) * 20  # Approximate sub-model params
        
        return n_params