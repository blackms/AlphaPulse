"""
Machine learning-based optimization for self-supervised agents.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from datetime import datetime
from loguru import logger


class MLOptimizer:
    """
    Machine learning-based optimizer for agent parameters.
    Uses Random Forest for feature importance and Optuna for hyperparameter optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML optimizer."""
        self.config = config or {}
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self._feature_importance: Dict[str, float] = {}
        self._optimization_history: List[Dict[str, Any]] = []
        
    async def optimize_parameters(
        self,
        parameters: Dict[str, float],
        performance_history: List[Dict[str, Any]],
        target_metric: str = "performance_score"
    ) -> Dict[str, float]:
        """
        Optimize parameters using ML-based approach.
        
        Args:
            parameters: Current parameter values to optimize
            performance_history: Historical performance data
            target_metric: Metric to optimize for
            
        Returns:
            Optimized parameter values
        """
        try:
            if len(performance_history) < 50:  # Need sufficient history
                logger.warning("Insufficient history for ML optimization")
                return parameters
                
            # Prepare training data
            X, y = await self._prepare_training_data(
                performance_history,
                parameters.keys(),
                target_metric
            )
            
            if X.shape[0] < 2:
                logger.warning("Insufficient training data")
                return parameters
                
            # Train model and get feature importance
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Update feature importance
            feature_names = list(parameters.keys())
            importance = self.model.feature_importances_
            self._feature_importance = dict(zip(feature_names, importance))
            
            # Use Optuna for parameter optimization
            study = optuna.create_study(direction="maximize")
            
            def objective(trial):
                # Create trial parameters
                trial_params = {}
                for name, value in parameters.items():
                    # Define parameter range based on current value
                    lower = value * 0.5
                    upper = value * 1.5
                    trial_params[name] = trial.suggest_float(name, lower, upper)
                    
                # Predict performance with trial parameters
                trial_features = np.array([list(trial_params.values())])
                trial_features_scaled = self.scaler.transform(trial_features)
                return float(self.model.predict(trial_features_scaled)[0])
                
            # Run optimization
            study.optimize(objective, n_trials=100)
            
            # Get optimized parameters
            optimized_params = study.best_params
            
            # Apply smoothing to parameter updates
            smoothing = 0.7  # 70% old values, 30% new values
            final_params = {
                name: smoothing * parameters[name] + (1 - smoothing) * optimized_params[name]
                for name in parameters
            }
            
            # Store optimization result
            self._optimization_history.append({
                'timestamp': datetime.now(),
                'original_params': parameters.copy(),
                'optimized_params': optimized_params.copy(),
                'final_params': final_params.copy(),
                'feature_importance': self._feature_importance.copy(),
                'best_score': study.best_value
            })
            
            logger.info(f"ML optimization completed. Feature importance: {self._feature_importance}")
            return final_params
            
        except Exception as e:
            logger.error(f"Error in ML optimization: {str(e)}")
            return parameters
            
    async def get_parameter_importance(self) -> Dict[str, float]:
        """Get current parameter importance scores."""
        return self._feature_importance.copy()
        
    async def _prepare_training_data(
        self,
        history: List[Dict[str, Any]],
        parameter_names: List[str],
        target_metric: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from performance history."""
        X_data = []
        y_data = []
        
        for record in history:
            # Extract features (parameters)
            features = [
                record.get('parameters', {}).get(name, 0)
                for name in parameter_names
            ]
            
            # Extract target (performance metric)
            target = record.get('metrics', {}).get(target_metric, 0)
            
            if all(x is not None for x in features) and target is not None:
                X_data.append(features)
                y_data.append(target)
                
        return np.array(X_data), np.array(y_data)
        
    async def analyze_optimization_results(self) -> Dict[str, Any]:
        """
        Analyze optimization history and results.
        
        Returns:
            Dictionary containing optimization analytics
        """
        if not self._optimization_history:
            return {}
            
        try:
            # Calculate parameter stability
            param_changes = []
            for i in range(1, len(self._optimization_history)):
                prev = self._optimization_history[i-1]['final_params']
                curr = self._optimization_history[i]['final_params']
                
                changes = [
                    abs(curr[param] - prev[param]) / prev[param]
                    for param in prev
                    if prev[param] != 0
                ]
                param_changes.append(np.mean(changes))
                
            # Calculate improvement trends
            scores = [h['best_score'] for h in self._optimization_history]
            score_changes = np.diff(scores)
            
            return {
                'parameter_stability': 1 - np.mean(param_changes) if param_changes else 1.0,
                'score_trend': np.mean(score_changes) if len(scores) > 1 else 0.0,
                'best_score': max(scores),
                'optimization_count': len(self._optimization_history),
                'latest_importance': self._feature_importance,
                'latest_score': scores[-1] if scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing optimization results: {str(e)}")
            return {}