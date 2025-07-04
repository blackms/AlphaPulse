"""Stacking ensemble methods with meta-learning."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import joblib
import os

from alpha_pulse.ml.ensemble.ensemble_manager import BaseEnsemble, AgentSignal, EnsembleSignal

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble with multiple meta-learners."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.meta_model_type = config.get('meta_model', 'xgboost')
        self.use_cross_validation = config.get('use_cv', True)
        self.cv_folds = config.get('cv_folds', 5)
        self.blend_meta_models = config.get('blend_meta_models', False)
        self.feature_engineering = config.get('feature_engineering', True)
        self.model_save_path = config.get('model_save_path', 'models/stacking/')
        
        self.meta_model = None
        self.blend_models = {}
        self.feature_stats = {}
        
    def _create_meta_model(self) -> Any:
        """Create meta-learner based on configuration."""
        if self.meta_model_type == 'linear':
            return Ridge(alpha=1.0, random_state=42)
        elif self.meta_model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.meta_model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.meta_model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.meta_model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta model type: {self.meta_model_type}")
            
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train stacking ensemble with cross-validation."""
        # Create feature matrix from signals
        X, agent_ids = self._create_feature_matrix(signals)
        y = outcomes[:len(X)]
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No data to fit stacking ensemble")
            return
            
        # Store feature statistics for normalization
        self.feature_stats['mean'] = np.mean(X, axis=0)
        self.feature_stats['std'] = np.std(X, axis=0) + 1e-8
        
        # Normalize features
        X_normalized = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        if self.use_cross_validation:
            # Train with cross-validation
            self._train_with_cv(X_normalized, y, agent_ids)
        else:
            # Simple train
            self.meta_model = self._create_meta_model()
            self.meta_model.fit(X_normalized, y)
            
        # Train blend models if enabled
        if self.blend_meta_models:
            self._train_blend_models(X_normalized, y)
            
        self.is_fitted = True
        logger.info(f"Fitted stacking ensemble with {len(agent_ids)} base models")
        
    def _train_with_cv(self, X: np.ndarray, y: np.ndarray, agent_ids: List[str]) -> None:
        """Train meta-model with cross-validation."""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Store out-of-fold predictions for blending
        oof_predictions = np.zeros(len(y))
        
        # Train multiple models on different folds
        cv_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model for this fold
            model = self._create_meta_model()
            model.fit(X_train, y_train)
            
            # Store out-of-fold predictions
            oof_predictions[val_idx] = model.predict(X_val)
            cv_models.append(model)
            
            # Log fold performance
            val_score = np.corrcoef(oof_predictions[val_idx], y_val)[0, 1]
            logger.info(f"Fold {fold+1} validation correlation: {val_score:.4f}")
            
        # Train final model on all data
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(X, y)
        
        # Calculate feature importance if available
        if hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_
            for i, agent_id in enumerate(agent_ids):
                if i < len(importances):
                    self.agent_weights[agent_id] = float(importances[i])
                    
    def _train_blend_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train multiple meta-models for blending."""
        model_types = ['linear', 'random_forest', 'xgboost']
        
        for model_type in model_types:
            if model_type != self.meta_model_type:
                self.meta_model_type = model_type
                model = self._create_meta_model()
                model.fit(X, y)
                self.blend_models[model_type] = model
                
        # Reset to original model type
        self.meta_model_type = self.config.get('meta_model', 'xgboost')
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate prediction using stacking."""
        if not self.is_fitted or self.meta_model is None:
            return self._create_neutral_signal(signals, "model_not_fitted")
            
        # Create feature matrix
        X, agent_ids = self._create_feature_matrix(signals)
        
        if len(X) == 0:
            return self._create_neutral_signal(signals, "no_features")
            
        # Normalize features
        X_normalized = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Make prediction
        if self.blend_meta_models and self.blend_models:
            # Blend predictions from multiple models
            predictions = []
            weights = []
            
            # Main model prediction
            predictions.append(self.meta_model.predict(X_normalized))
            weights.append(2.0)  # Higher weight for main model
            
            # Blend model predictions
            for model_name, model in self.blend_models.items():
                predictions.append(model.predict(X_normalized))
                weights.append(1.0)
                
            # Weighted average
            ensemble_signal = np.average(predictions, weights=weights, axis=0)
        else:
            ensemble_signal = self.meta_model.predict(X_normalized)
            
        # Handle array output
        if isinstance(ensemble_signal, np.ndarray):
            ensemble_signal = float(ensemble_signal[0])
            
        # Calculate confidence based on prediction variance
        confidence = self._calculate_prediction_confidence(X_normalized, signals)
        
        # Get contributing agents and their importance
        contributing_agents = [s.agent_id for s in signals]
        
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=float(np.clip(ensemble_signal, -1, 1)),
            confidence=confidence,
            contributing_agents=contributing_agents,
            weights=self.agent_weights,
            metadata={
                'method': 'stacking',
                'meta_model': self.meta_model_type,
                'feature_count': X.shape[1],
                'blended': self.blend_meta_models
            }
        )
        
    def _create_feature_matrix(self, signals: List[AgentSignal]) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from agent signals."""
        if not signals:
            return np.array([]), []
            
        # Group signals by timestamp
        signal_groups = {}
        for signal in signals:
            timestamp = signal.timestamp.replace(microsecond=0)  # Round to second
            if timestamp not in signal_groups:
                signal_groups[timestamp] = {}
            signal_groups[timestamp][signal.agent_id] = signal
            
        # Get unique agent IDs
        agent_ids = sorted(set(s.agent_id for s in signals))
        
        # Create feature matrix
        features = []
        
        for timestamp, agent_signals in signal_groups.items():
            row = []
            
            for agent_id in agent_ids:
                if agent_id in agent_signals:
                    signal = agent_signals[agent_id]
                    # Basic features
                    row.extend([
                        signal.signal,
                        signal.confidence,
                        signal.signal * signal.confidence  # Weighted signal
                    ])
                    
                    if self.feature_engineering:
                        # Additional engineered features
                        row.extend([
                            abs(signal.signal),  # Signal magnitude
                            signal.signal ** 2,  # Squared signal
                            np.sign(signal.signal),  # Signal direction
                            signal.confidence ** 2,  # Confidence squared
                            np.log1p(signal.confidence),  # Log confidence
                        ])
                else:
                    # Missing signal - use zeros
                    num_features = 8 if self.feature_engineering else 3
                    row.extend([0.0] * num_features)
                    
            features.append(row)
            
        return np.array(features), agent_ids
        
    def _calculate_prediction_confidence(self, X: np.ndarray, 
                                       signals: List[AgentSignal]) -> float:
        """Calculate confidence in stacking prediction."""
        # Base confidence from signal agreement
        signal_values = [s.signal for s in signals]
        signal_std = np.std(signal_values)
        agreement_confidence = 1.0 / (1.0 + signal_std)
        
        # Model confidence (if available)
        model_confidence = 0.8  # Default
        
        if hasattr(self.meta_model, 'predict_proba'):
            try:
                # For probabilistic models
                proba = self.meta_model.predict_proba(X)
                model_confidence = np.max(proba)
            except:
                pass
                
        # Average confidence from signals
        avg_signal_confidence = np.mean([s.confidence for s in signals])
        
        # Combine confidences
        return float(np.mean([agreement_confidence, model_confidence, avg_signal_confidence]))
        
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'stacking', 'reason': reason}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update weights based on performance (feature importance)."""
        if hasattr(self.meta_model, 'feature_importances_'):
            # Update based on feature importance
            importances = self.meta_model.feature_importances_
            agent_ids = sorted(self.agent_weights.keys())
            
            for i, agent_id in enumerate(agent_ids):
                if i < len(importances):
                    # Blend old weight with new importance
                    old_weight = self.agent_weights.get(agent_id, 1.0)
                    new_weight = importances[i]
                    self.agent_weights[agent_id] = 0.7 * old_weight + 0.3 * new_weight
                    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save trained model to disk."""
        if not self.is_fitted:
            logger.warning("Model not fitted, nothing to save")
            return
            
        save_path = path or self.model_save_path
        os.makedirs(save_path, exist_ok=True)
        
        # Save main model
        model_file = os.path.join(save_path, f"stacking_{self.meta_model_type}.pkl")
        joblib.dump(self.meta_model, model_file)
        
        # Save blend models
        if self.blend_models:
            for name, model in self.blend_models.items():
                blend_file = os.path.join(save_path, f"blend_{name}.pkl")
                joblib.dump(model, blend_file)
                
        # Save feature stats and weights
        metadata = {
            'feature_stats': self.feature_stats,
            'agent_weights': self.agent_weights,
            'meta_model_type': self.meta_model_type
        }
        metadata_file = os.path.join(save_path, "stacking_metadata.pkl")
        joblib.dump(metadata, metadata_file)
        
        logger.info(f"Saved stacking model to {save_path}")
        
    def load_model(self, path: Optional[str] = None) -> None:
        """Load trained model from disk."""
        load_path = path or self.model_save_path
        
        # Load main model
        model_file = os.path.join(load_path, f"stacking_{self.meta_model_type}.pkl")
        if os.path.exists(model_file):
            self.meta_model = joblib.load(model_file)
            
        # Load blend models
        for model_type in ['linear', 'random_forest', 'xgboost']:
            blend_file = os.path.join(load_path, f"blend_{model_type}.pkl")
            if os.path.exists(blend_file):
                self.blend_models[model_type] = joblib.load(blend_file)
                
        # Load metadata
        metadata_file = os.path.join(load_path, "stacking_metadata.pkl")
        if os.path.exists(metadata_file):
            metadata = joblib.load(metadata_file)
            self.feature_stats = metadata['feature_stats']
            self.agent_weights = metadata['agent_weights']
            
        self.is_fitted = True
        logger.info(f"Loaded stacking model from {load_path}")


class HierarchicalStacking(BaseEnsemble):
    """Hierarchical stacking with multiple levels."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.levels = config.get('levels', 2)
        self.level_models = {}
        self.level_configs = config.get('level_configs', {})
        
    def fit(self, signals: List[AgentSignal], outcomes: np.ndarray) -> None:
        """Train hierarchical stacking ensemble."""
        # Level 1: Group agents by type/strategy
        agent_groups = self._group_agents(signals)
        
        # Train level 1 models for each group
        level1_predictions = {}
        
        for group_name, group_signals in agent_groups.items():
            if group_signals:
                # Create stacking model for this group
                group_config = self.level_configs.get(group_name, {})
                group_model = StackingEnsemble(group_config)
                group_model.fit(group_signals, outcomes)
                
                self.level_models[f"level1_{group_name}"] = group_model
                
                # Get predictions for level 2
                predictions = []
                for i in range(len(outcomes)):
                    # Get signals at timestamp i
                    timestamp_signals = [s for s in group_signals if s.timestamp.second == i]
                    if timestamp_signals:
                        pred = group_model.predict(timestamp_signals)
                        predictions.append(pred.signal)
                    else:
                        predictions.append(0.0)
                        
                level1_predictions[group_name] = predictions
                
        # Level 2: Meta-model on level 1 predictions
        if level1_predictions:
            # Create feature matrix from level 1 predictions
            X_level2 = np.column_stack(list(level1_predictions.values()))
            
            # Train level 2 model
            level2_config = self.level_configs.get('level2', {'meta_model': 'xgboost'})
            self.level_models['level2'] = StackingEnsemble(level2_config)
            self.level_models['level2'].fit(self._create_synthetic_signals(X_level2), outcomes)
            
        self.is_fitted = True
        
    def _group_agents(self, signals: List[AgentSignal]) -> Dict[str, List[AgentSignal]]:
        """Group agents by type or strategy."""
        groups = {
            'technical': [],
            'fundamental': [],
            'sentiment': [],
            'ml_based': [],
            'other': []
        }
        
        for signal in signals:
            # Group based on agent_id prefix or metadata
            if 'technical' in signal.agent_id.lower():
                groups['technical'].append(signal)
            elif 'fundamental' in signal.agent_id.lower():
                groups['fundamental'].append(signal)
            elif 'sentiment' in signal.agent_id.lower():
                groups['sentiment'].append(signal)
            elif 'ml' in signal.agent_id.lower() or 'nn' in signal.agent_id.lower():
                groups['ml_based'].append(signal)
            else:
                groups['other'].append(signal)
                
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
        
    def _create_synthetic_signals(self, predictions: np.ndarray) -> List[AgentSignal]:
        """Create synthetic signals from predictions for level 2."""
        synthetic_signals = []
        
        for i in range(len(predictions)):
            for j, pred in enumerate(predictions[i]):
                signal = AgentSignal(
                    agent_id=f"level1_group{j}",
                    timestamp=datetime.now(),
                    signal=float(pred),
                    confidence=0.8,
                    metadata={'level': 1, 'group': j}
                )
                synthetic_signals.append(signal)
                
        return synthetic_signals
        
    def predict(self, signals: List[AgentSignal]) -> EnsembleSignal:
        """Generate hierarchical prediction."""
        if not self.is_fitted:
            return self._create_neutral_signal(signals, "model_not_fitted")
            
        # Level 1 predictions
        agent_groups = self._group_agents(signals)
        level1_predictions = []
        level1_confidences = []
        
        for group_name, group_signals in agent_groups.items():
            model_key = f"level1_{group_name}"
            if model_key in self.level_models and group_signals:
                pred = self.level_models[model_key].predict(group_signals)
                level1_predictions.append(pred.signal)
                level1_confidences.append(pred.confidence)
                
        if not level1_predictions:
            return self._create_neutral_signal(signals, "no_level1_predictions")
            
        # Level 2 prediction
        if 'level2' in self.level_models:
            # Create synthetic signals for level 2
            synthetic_signals = [
                AgentSignal(
                    agent_id=f"level1_pred{i}",
                    timestamp=datetime.now(),
                    signal=pred,
                    confidence=conf,
                    metadata={'level': 1}
                )
                for i, (pred, conf) in enumerate(zip(level1_predictions, level1_confidences))
            ]
            
            final_prediction = self.level_models['level2'].predict(synthetic_signals)
            
            return EnsembleSignal(
                timestamp=datetime.now(),
                signal=final_prediction.signal,
                confidence=final_prediction.confidence,
                contributing_agents=[s.agent_id for s in signals],
                weights=self.agent_weights,
                metadata={
                    'method': 'hierarchical_stacking',
                    'levels': self.levels,
                    'group_count': len(agent_groups),
                    'level1_predictions': level1_predictions
                }
            )
        else:
            # Average level 1 predictions
            return EnsembleSignal(
                timestamp=datetime.now(),
                signal=float(np.mean(level1_predictions)),
                confidence=float(np.mean(level1_confidences)),
                contributing_agents=[s.agent_id for s in signals],
                weights=self.agent_weights,
                metadata={
                    'method': 'hierarchical_stacking',
                    'levels': 1,
                    'group_count': len(agent_groups)
                }
            )
            
    def _create_neutral_signal(self, signals: List[AgentSignal], reason: str) -> EnsembleSignal:
        """Create neutral signal."""
        return EnsembleSignal(
            timestamp=datetime.now(),
            signal=0.0,
            confidence=0.0,
            contributing_agents=[s.agent_id for s in signals],
            weights={},
            metadata={'method': 'hierarchical_stacking', 'reason': reason}
        )
        
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update weights at all levels."""
        # Update level 1 models
        for model_name, model in self.level_models.items():
            if model_name.startswith('level1_'):
                model.update_weights(performance_metrics)
                
        # Update level 2 model with aggregated metrics
        if 'level2' in self.level_models:
            # Create synthetic metrics for level 2
            level2_metrics = {}
            for i, group in enumerate(self._group_agents([]).keys()):
                # Average metrics for agents in this group
                group_agents = [a for a in performance_metrics.keys() if group in a.lower()]
                if group_agents:
                    avg_metric = np.mean([performance_metrics[a] for a in group_agents])
                    level2_metrics[f"level1_pred{i}"] = avg_metric
                    
            self.level_models['level2'].update_weights(level2_metrics)