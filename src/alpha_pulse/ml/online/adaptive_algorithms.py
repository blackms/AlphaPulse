"""Adaptive algorithms for online learning with dynamic parameter adjustment."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import math
from enum import Enum

logger = logging.getLogger(__name__)


class AdaptiveStrategy(Enum):
    """Adaptive learning strategies."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    VOLATILITY_BASED = "volatility_based"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class LearningRateSchedule:
    """Learning rate schedule configuration."""
    initial_rate: float
    decay_rate: float
    decay_steps: int
    min_rate: float
    max_rate: float
    warmup_steps: int = 0


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduling for online learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.schedule_type = config.get('schedule_type', 'exponential_decay')
        self.initial_rate = config.get('initial_rate', 0.01)
        self.decay_rate = config.get('decay_rate', 0.95)
        self.decay_steps = config.get('decay_steps', 1000)
        self.min_rate = config.get('min_rate', 1e-6)
        self.max_rate = config.get('max_rate', 1.0)
        self.warmup_steps = config.get('warmup_steps', 0)
        
        self.current_step = 0
        self.current_rate = self.initial_rate
        self.performance_history = deque(maxlen=100)
        self.rate_history = deque(maxlen=1000)
        
        # Adaptive parameters
        self.adapt_to_performance = config.get('adapt_to_performance', True)
        self.adapt_to_volatility = config.get('adapt_to_volatility', True)
        self.momentum = config.get('momentum', 0.9)
        
        # State for adaptive adjustments
        self.velocity = 0.0
        self.best_performance = -float('inf')
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        
    def get_learning_rate(self, step: Optional[int] = None) -> float:
        """Get current learning rate."""
        if step is None:
            step = self.current_step
            
        # Warmup phase
        if step < self.warmup_steps:
            warmup_rate = self.initial_rate * (step / self.warmup_steps)
            return max(self.min_rate, warmup_rate)
            
        # Apply schedule
        if self.schedule_type == 'exponential_decay':
            rate = self._exponential_decay(step)
        elif self.schedule_type == 'polynomial_decay':
            rate = self._polynomial_decay(step)
        elif self.schedule_type == 'cosine_annealing':
            rate = self._cosine_annealing(step)
        elif self.schedule_type == 'step_decay':
            rate = self._step_decay(step)
        elif self.schedule_type == 'adaptive':
            rate = self._adaptive_schedule(step)
        else:
            rate = self.initial_rate
            
        # Apply bounds
        rate = np.clip(rate, self.min_rate, self.max_rate)
        
        # Store history
        self.rate_history.append((step, rate))
        self.current_rate = rate
        
        return rate
        
    def step(self, performance: Optional[float] = None) -> float:
        """Advance scheduler and update learning rate."""
        self.current_step += 1
        
        if performance is not None:
            self.performance_history.append(performance)
            
        return self.get_learning_rate()
        
    def _exponential_decay(self, step: int) -> float:
        """Exponential decay schedule."""
        decay_factor = self.decay_rate ** (step / self.decay_steps)
        return self.initial_rate * decay_factor
        
    def _polynomial_decay(self, step: int) -> float:
        """Polynomial decay schedule."""
        power = self.config.get('power', 1.0)
        end_learning_rate = self.config.get('end_learning_rate', self.min_rate)
        
        global_step = min(step, self.decay_steps)
        decayed = (self.initial_rate - end_learning_rate) * \
                  (1 - global_step / self.decay_steps) ** power + end_learning_rate
                  
        return decayed
        
    def _cosine_annealing(self, step: int) -> float:
        """Cosine annealing schedule."""
        T_max = self.config.get('T_max', self.decay_steps)
        eta_min = self.config.get('eta_min', self.min_rate)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / T_max))
        decayed = eta_min + (self.initial_rate - eta_min) * cosine_decay
        
        return decayed
        
    def _step_decay(self, step: int) -> float:
        """Step decay schedule."""
        drop_rate = self.config.get('drop_rate', 0.5)
        epochs_drop = self.config.get('epochs_drop', 10)
        
        drops = step // epochs_drop
        decayed = self.initial_rate * (drop_rate ** drops)
        
        return decayed
        
    def _adaptive_schedule(self, step: int) -> float:
        """Adaptive learning rate based on performance."""
        base_rate = self._exponential_decay(step)
        
        if not self.performance_history:
            return base_rate
            
        # Performance-based adjustment
        if self.adapt_to_performance:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            
            if recent_performance > self.best_performance:
                self.best_performance = recent_performance
                self.patience_counter = 0
                # Increase learning rate slightly
                base_rate *= 1.05
            else:
                self.patience_counter += 1
                if self.patience_counter > self.patience:
                    # Reduce learning rate
                    base_rate *= 0.8
                    self.patience_counter = 0
                    
        # Volatility-based adjustment
        if self.adapt_to_volatility and len(self.performance_history) > 10:
            volatility = np.std(list(self.performance_history)[-20:])
            # Higher volatility -> lower learning rate
            volatility_factor = 1.0 / (1.0 + volatility)
            base_rate *= volatility_factor
            
        return base_rate
        
    def reset(self) -> None:
        """Reset scheduler state."""
        self.current_step = 0
        self.current_rate = self.initial_rate
        self.performance_history.clear()
        self.rate_history.clear()
        self.velocity = 0.0
        self.best_performance = -float('inf')
        self.patience_counter = 0


class AdaptiveOptimizer:
    """Adaptive optimizer with momentum and adaptive learning rates."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer_type = config.get('optimizer_type', 'adam')
        self.learning_rate = config.get('learning_rate', 0.001)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.999)
        self.epsilon = config.get('epsilon', 1e-8)
        
        # State variables
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
        
        # Adaptive parameters
        self.adapt_betas = config.get('adapt_betas', True)
        self.gradient_clipping = config.get('gradient_clipping', 1.0)
        
    def update(self, params: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray],
               learning_rate: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Update parameters using adaptive optimization."""
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        self.t += 1
        updated_params = {}
        
        for key in params:
            if key not in gradients:
                updated_params[key] = params[key]
                continue
                
            grad = gradients[key]
            
            # Gradient clipping
            if self.gradient_clipping:
                grad = np.clip(grad, -self.gradient_clipping, self.gradient_clipping)
                
            if self.optimizer_type == 'adam':
                updated_params[key] = self._adam_update(
                    params[key], grad, key, learning_rate
                )
            elif self.optimizer_type == 'rmsprop':
                updated_params[key] = self._rmsprop_update(
                    params[key], grad, key, learning_rate
                )
            elif self.optimizer_type == 'adagrad':
                updated_params[key] = self._adagrad_update(
                    params[key], grad, key, learning_rate
                )
            elif self.optimizer_type == 'momentum':
                updated_params[key] = self._momentum_update(
                    params[key], grad, key, learning_rate
                )
            else:
                # Simple SGD
                updated_params[key] = params[key] - learning_rate * grad
                
        return updated_params
        
    def _adam_update(self, param: np.ndarray, grad: np.ndarray, 
                     key: str, learning_rate: float) -> np.ndarray:
        """Adam optimizer update."""
        # Initialize moments
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
            
        # Update biased moments
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2
        
        # Bias correction
        m_hat = self.m[key] / (1 - self.beta1**self.t)
        v_hat = self.v[key] / (1 - self.beta2**self.t)
        
        # Update parameters
        update = learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return param - update
        
    def _rmsprop_update(self, param: np.ndarray, grad: np.ndarray,
                        key: str, learning_rate: float) -> np.ndarray:
        """RMSprop optimizer update."""
        if key not in self.v:
            self.v[key] = np.zeros_like(param)
            
        # Update moving average of squared gradients
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2
        
        # Update parameters
        update = learning_rate * grad / (np.sqrt(self.v[key]) + self.epsilon)
        
        return param - update
        
    def _adagrad_update(self, param: np.ndarray, grad: np.ndarray,
                        key: str, learning_rate: float) -> np.ndarray:
        """AdaGrad optimizer update."""
        if key not in self.v:
            self.v[key] = np.zeros_like(param)
            
        # Accumulate squared gradients
        self.v[key] += grad**2
        
        # Update parameters
        update = learning_rate * grad / (np.sqrt(self.v[key]) + self.epsilon)
        
        return param - update
        
    def _momentum_update(self, param: np.ndarray, grad: np.ndarray,
                         key: str, learning_rate: float) -> np.ndarray:
        """Momentum optimizer update."""
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            
        # Update velocity
        self.m[key] = self.beta1 * self.m[key] + learning_rate * grad
        
        # Update parameters
        return param - self.m[key]
        
    def adapt_hyperparameters(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt optimizer hyperparameters based on performance."""
        if not self.adapt_betas:
            return
            
        # Adapt beta1 based on gradient variance
        if 'gradient_variance' in performance_metrics:
            grad_var = performance_metrics['gradient_variance']
            # High variance -> lower beta1 (less momentum)
            self.beta1 = 0.9 * (1.0 / (1.0 + grad_var))
            
        # Adapt beta2 based on gradient magnitude
        if 'gradient_magnitude' in performance_metrics:
            grad_mag = performance_metrics['gradient_magnitude']
            # Large gradients -> higher beta2 (more stable)
            self.beta2 = 0.999 * (1.0 - np.exp(-grad_mag))
            
    def reset(self) -> None:
        """Reset optimizer state."""
        self.m.clear()
        self.v.clear()
        self.t = 0


class MultiArmedBandit:
    """Multi-armed bandit for adaptive strategy selection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_arms = config.get('n_arms', 4)
        self.algorithm = config.get('algorithm', 'ucb')
        self.epsilon = config.get('epsilon', 0.1)
        self.c = config.get('c', 2.0)  # UCB exploration parameter
        self.tau = config.get('tau', 1.0)  # Temperature for softmax
        
        # Initialize statistics
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_count = 0
        
        # For Thompson Sampling
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """Select an arm (strategy) to play."""
        self.total_count += 1
        
        if self.algorithm == 'epsilon_greedy':
            return self._epsilon_greedy()
        elif self.algorithm == 'ucb':
            return self._upper_confidence_bound()
        elif self.algorithm == 'thompson':
            return self._thompson_sampling()
        elif self.algorithm == 'softmax':
            return self._softmax_selection()
        elif self.algorithm == 'gradient':
            return self._gradient_bandit(context)
        else:
            return np.random.randint(self.n_arms)
            
    def _epsilon_greedy(self) -> int:
        """Epsilon-greedy selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)
            
    def _upper_confidence_bound(self) -> int:
        """Upper Confidence Bound selection."""
        ucb_values = np.zeros(self.n_arms)
        
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                ucb_values[i] = float('inf')
            else:
                exploration_term = self.c * np.sqrt(
                    np.log(self.total_count) / self.counts[i]
                )
                ucb_values[i] = self.values[i] + exploration_term
                
        return np.argmax(ucb_values)
        
    def _thompson_sampling(self) -> int:
        """Thompson Sampling selection."""
        samples = np.zeros(self.n_arms)
        
        for i in range(self.n_arms):
            samples[i] = np.random.beta(self.alpha[i], self.beta[i])
            
        return np.argmax(samples)
        
    def _softmax_selection(self) -> int:
        """Softmax selection with temperature."""
        exp_values = np.exp(self.values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        
        return np.random.choice(self.n_arms, p=probabilities)
        
    def _gradient_bandit(self, context: Optional[np.ndarray]) -> int:
        """Gradient bandit with optional context."""
        if context is None:
            # Use preferences directly
            exp_prefs = np.exp(self.values)
            probabilities = exp_prefs / np.sum(exp_prefs)
        else:
            # Linear contextual bandit
            # Simplified - would need weight matrix in practice
            scores = self.values + np.dot(context, np.ones((len(context), self.n_arms)))
            exp_scores = np.exp(scores)
            probabilities = exp_scores / np.sum(exp_scores)
            
        return np.random.choice(self.n_arms, p=probabilities)
        
    def update(self, arm: int, reward: float) -> None:
        """Update arm statistics with observed reward."""
        self.counts[arm] += 1
        
        # Update value estimate
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm] = new_value
        
        # Update Thompson Sampling parameters
        if reward > 0:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += 1 - reward
            
    def get_arm_stats(self) -> Dict[str, Any]:
        """Get statistics for all arms."""
        return {
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'total_count': self.total_count,
            'best_arm': int(np.argmax(self.values)),
            'exploration_rate': self._calculate_exploration_rate()
        }
        
    def _calculate_exploration_rate(self) -> float:
        """Calculate current exploration rate."""
        if self.total_count == 0:
            return 1.0
            
        best_arm = np.argmax(self.values)
        exploitation_count = self.counts[best_arm]
        
        return 1.0 - (exploitation_count / self.total_count)
        
    def reset(self) -> None:
        """Reset bandit statistics."""
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_count = 0
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)


class AdaptiveMetaLearner:
    """Meta-learner for adaptive algorithm selection and hyperparameter tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_learning_rate = config.get('meta_learning_rate', 0.01)
        self.adaptation_window = config.get('adaptation_window', 100)
        
        # Strategy pool
        self.strategies = {
            'conservative': {'learning_rate': 0.001, 'batch_size': 32},
            'aggressive': {'learning_rate': 0.1, 'batch_size': 1},
            'balanced': {'learning_rate': 0.01, 'batch_size': 8},
            'adaptive': {'learning_rate': 'adaptive', 'batch_size': 'adaptive'}
        }
        
        # Performance tracking
        self.strategy_performance = {s: deque(maxlen=self.adaptation_window) 
                                   for s in self.strategies}
        self.current_strategy = 'balanced'
        
        # Multi-armed bandit for strategy selection
        bandit_config = {'n_arms': len(self.strategies), 'algorithm': 'thompson'}
        self.strategy_bandit = MultiArmedBandit(bandit_config)
        self.strategy_names = list(self.strategies.keys())
        
    def select_strategy(self, market_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Select learning strategy based on market conditions."""
        # Use bandit to select strategy
        arm = self.strategy_bandit.select_arm()
        self.current_strategy = self.strategy_names[arm]
        
        # Get base configuration
        config = self.strategies[self.current_strategy].copy()
        
        # Adapt based on market conditions
        if config.get('learning_rate') == 'adaptive':
            config['learning_rate'] = self._adapt_learning_rate(market_conditions)
            
        if config.get('batch_size') == 'adaptive':
            config['batch_size'] = self._adapt_batch_size(market_conditions)
            
        return config
        
    def _adapt_learning_rate(self, conditions: Dict[str, float]) -> float:
        """Adapt learning rate based on market conditions."""
        base_rate = 0.01
        
        # Volatility adjustment
        if 'volatility' in conditions:
            volatility = conditions['volatility']
            # Higher volatility -> lower learning rate
            base_rate *= (1.0 / (1.0 + volatility))
            
        # Trend strength adjustment
        if 'trend_strength' in conditions:
            trend = conditions['trend_strength']
            # Strong trend -> higher learning rate
            base_rate *= (1.0 + trend * 0.5)
            
        # Market regime adjustment
        if 'regime' in conditions:
            regime = conditions['regime']
            if regime == 'crisis':
                base_rate *= 0.5
            elif regime == 'bull':
                base_rate *= 1.2
                
        return np.clip(base_rate, 1e-5, 0.1)
        
    def _adapt_batch_size(self, conditions: Dict[str, float]) -> int:
        """Adapt batch size based on market conditions."""
        base_size = 8
        
        # Data availability
        if 'data_rate' in conditions:
            data_rate = conditions['data_rate']
            # High data rate -> larger batches
            base_size = int(base_size * (1 + data_rate))
            
        # Stability requirement
        if 'stability_required' in conditions:
            if conditions['stability_required'] > 0.7:
                base_size = max(base_size, 16)
                
        return np.clip(base_size, 1, 64)
        
    def update_performance(self, strategy: str, performance: float) -> None:
        """Update strategy performance."""
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy].append(performance)
            
            # Update bandit
            arm = self.strategy_names.index(strategy)
            self.strategy_bandit.update(arm, performance)
            
    def get_meta_insights(self) -> Dict[str, Any]:
        """Get insights from meta-learning."""
        insights = {
            'current_strategy': self.current_strategy,
            'strategy_stats': self.strategy_bandit.get_arm_stats(),
            'performance_summary': {}
        }
        
        for strategy, perfs in self.strategy_performance.items():
            if perfs:
                insights['performance_summary'][strategy] = {
                    'mean': np.mean(list(perfs)),
                    'std': np.std(list(perfs)),
                    'recent': list(perfs)[-10:]
                }
                
        return insights