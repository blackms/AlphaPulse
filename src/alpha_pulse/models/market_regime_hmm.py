"""
Market Regime Hidden Markov Model Implementation.

This module provides the main interface for market regime detection using
Hidden Markov Models with comprehensive feature engineering and real-time
classification capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import joblib
import json

from ..ml.regime.regime_features import RegimeFeatureEngineer, RegimeFeatureConfig
from ..ml.regime.hmm_regime_detector import (
    GaussianHMM, HMMConfig, RegimeType, 
    RegimeSwitchingGARCH, HierarchicalHMM
)
from ..ml.regime.regime_classifier import RegimeClassifier, RegimeInfo
from ..ml.regime.regime_transitions import RegimeTransitionAnalyzer, TransitionEvent

logger = logging.getLogger(__name__)


@dataclass
class MarketRegimeConfig:
    """Configuration for market regime detection system."""
    # Feature engineering
    feature_config: RegimeFeatureConfig = field(default_factory=RegimeFeatureConfig)
    
    # HMM configuration
    hmm_config: HMMConfig = field(default_factory=lambda: HMMConfig(
        n_states=5,
        covariance_type="full",
        n_iter=100,
        init_method="kmeans"
    ))
    
    # Classifier configuration
    classification_window: int = 100
    min_confidence: float = 0.6
    transition_confirmation_window: int = 5
    
    # Model variants
    use_regime_switching_garch: bool = False
    use_hierarchical_hmm: bool = False
    hierarchical_levels: int = 2
    
    # Real-time settings
    update_frequency: str = "1h"  # How often to update regime classification
    feature_lag: int = 1  # Lag for feature calculation to avoid look-ahead bias
    
    # Persistence
    model_save_path: Optional[str] = None
    save_frequency: int = 24  # Save model every N updates


class MarketRegimeHMM:
    """
    Complete market regime detection system using Hidden Markov Models.
    
    This class integrates feature engineering, HMM training, real-time
    classification, and transition analysis into a unified interface.
    """
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        self.config = config or MarketRegimeConfig()
        
        # Initialize components
        self.feature_engineer = RegimeFeatureEngineer(self.config.feature_config)
        
        # Select appropriate HMM variant
        if self.config.use_regime_switching_garch:
            self.hmm_model = RegimeSwitchingGARCH(self.config.hmm_config)
        elif self.config.use_hierarchical_hmm:
            self.hmm_model = HierarchicalHMM(
                self.config.hmm_config, 
                n_levels=self.config.hierarchical_levels
            )
        else:
            self.hmm_model = GaussianHMM(self.config.hmm_config)
        
        # Initialize classifier and analyzer
        self.classifier: Optional[RegimeClassifier] = None
        self.transition_analyzer = RegimeTransitionAnalyzer()
        
        # State tracking
        self.is_fitted = False
        self.last_update_time: Optional[datetime] = None
        self.update_count = 0
        self.training_data_info = {}
        
        # Performance tracking
        self.performance_metrics = {
            'classification_accuracy': [],
            'transition_accuracy': [],
            'regime_stability': []
        }
    
    def fit(self, 
            data: pd.DataFrame,
            additional_data: Optional[Dict[str, pd.DataFrame]] = None,
            validation_split: float = 0.2) -> 'MarketRegimeHMM':
        """
        Fit the regime detection model.
        
        Args:
            data: Market data with OHLCV columns
            additional_data: Optional additional data (VIX, sentiment, etc.)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Self
        """
        logger.info("Starting market regime model training...")
        
        # Extract features
        features = self.feature_engineer.extract_features(data, additional_data)
        if features.empty:
            raise ValueError("No features could be extracted from the data")
        
        # Split data for validation
        n_samples = len(features)
        n_train = int(n_samples * (1 - validation_split))
        
        train_features = features.iloc[:n_train]
        val_features = features.iloc[n_train:] if validation_split > 0 else None
        
        # Fit HMM model
        logger.info(f"Training HMM with {len(train_features)} samples...")
        self.hmm_model.fit(train_features.values)
        
        # Initialize classifier
        self.classifier = RegimeClassifier(
            self.hmm_model,
            self.feature_engineer,
            window_size=self.config.classification_window,
            min_confidence=self.config.min_confidence
        )
        
        # Validate if split provided
        if val_features is not None:
            self._validate_model(val_features)
        
        # Store training info
        self.training_data_info = {
            'n_samples': n_samples,
            'n_features': features.shape[1],
            'feature_names': features.columns.tolist(),
            'date_range': (data.index[0], data.index[-1]),
            'convergence_iterations': len(self.hmm_model.convergence_history)
        }
        
        self.is_fitted = True
        logger.info("Market regime model training completed")
        
        # Analyze historical regimes
        self._analyze_historical_regimes(data, features)
        
        return self
    
    def predict_regime(self, 
                      data: pd.DataFrame,
                      additional_data: Optional[Dict[str, pd.DataFrame]] = None,
                      return_probabilities: bool = False) -> Union[RegimeInfo, Tuple[RegimeInfo, np.ndarray]]:
        """
        Predict current market regime.
        
        Args:
            data: Recent market data
            additional_data: Optional additional data
            return_probabilities: Whether to return regime probabilities
            
        Returns:
            RegimeInfo or tuple of (RegimeInfo, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Classify current regime
        regime_info = self.classifier.classify(data, additional_data)
        
        # Update tracking
        self.last_update_time = datetime.now()
        self.update_count += 1
        
        # Save model periodically
        if self.config.model_save_path and self.update_count % self.config.save_frequency == 0:
            self.save(self.config.model_save_path)
        
        if return_probabilities:
            return regime_info, regime_info.regime_probabilities
        return regime_info
    
    def get_regime_forecast(self, horizon: int = 10) -> pd.DataFrame:
        """
        Forecast regime probabilities over given horizon.
        
        Args:
            horizon: Forecast horizon in periods
            
        Returns:
            DataFrame with forecasted regime probabilities
        """
        if not self.classifier:
            raise ValueError("Model must be fitted and have made predictions")
        
        return self.classifier.get_regime_forecast(horizon)
    
    def get_transition_analysis(self) -> Dict[str, Any]:
        """Get comprehensive transition analysis."""
        if not self.classifier:
            return {}
        
        # Get transition statistics
        trans_stats = self.classifier.get_transition_statistics()
        
        # Get patterns from analyzer
        patterns = self.transition_analyzer.identify_transition_patterns()
        
        # Get current regime info
        current_info = None
        if self.classifier.current_regime_info:
            current_info = {
                'regime': self.classifier.current_regime_info.current_regime,
                'type': self.classifier.current_regime_info.regime_type.value,
                'confidence': self.classifier.current_regime_info.confidence,
                'duration': self.classifier.current_regime_info.duration,
                'transition_probability': self.classifier.current_regime_info.transition_probability
            }
        
        return {
            'current_regime': current_info,
            'transition_statistics': trans_stats.to_dict() if not trans_stats.empty else {},
            'patterns': patterns,
            'regime_characteristics': self._get_all_regime_characteristics()
        }
    
    def _get_all_regime_characteristics(self) -> Dict[int, Dict[str, Any]]:
        """Get characteristics for all regimes."""
        characteristics = {}
        
        for i in range(self.hmm_model.n_states):
            characteristics[i] = self.classifier.get_regime_characteristics(i)
        
        return characteristics
    
    def _validate_model(self, val_features: pd.DataFrame):
        """Validate model performance on held-out data."""
        # Predict regimes
        val_regimes = self.hmm_model.predict(val_features.values)
        
        # Calculate regime stability
        regime_changes = np.diff(val_regimes) != 0
        stability_score = 1 - (regime_changes.sum() / len(regime_changes))
        
        self.performance_metrics['regime_stability'].append(stability_score)
        
        logger.info(f"Validation stability score: {stability_score:.3f}")
    
    def _analyze_historical_regimes(self, data: pd.DataFrame, features: pd.DataFrame):
        """Analyze historical regime patterns."""
        # Get historical regime sequence
        regimes = self.hmm_model.predict(features.values)
        
        # Create regime change events
        regime_changes = []
        current_regime = regimes[0]
        regime_start = 0
        
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # Extract features at transition
                trans_features = features.iloc[i].to_dict()
                
                # Identify potential triggers
                triggers = self._identify_triggers(
                    features.iloc[max(0, i-5):i+1],
                    current_regime,
                    regimes[i]
                )
                
                event = TransitionEvent(
                    from_regime=current_regime,
                    to_regime=regimes[i],
                    timestamp=data.index[i],
                    duration_in_from=i - regime_start,
                    market_conditions=trans_features,
                    trigger_factors=triggers,
                    transition_speed=1.0,  # Simplified
                    stability_score=0.8  # Simplified
                )
                
                regime_changes.append(event)
                self.transition_analyzer.add_transition(event)
                
                current_regime = regimes[i]
                regime_start = i
    
    def _identify_triggers(self, 
                         features: pd.DataFrame,
                         from_regime: int,
                         to_regime: int) -> List[str]:
        """Identify potential triggers for regime transition."""
        triggers = []
        
        if features.empty:
            return triggers
        
        # Check for volatility spike
        if 'volatility_5d' in features.columns:
            vol_change = features['volatility_5d'].iloc[-1] / features['volatility_5d'].iloc[0]
            if vol_change > 1.5:
                triggers.append("volatility_spike")
        
        # Check for return extremes
        if 'return_5d' in features.columns:
            recent_return = features['return_5d'].iloc[-1]
            if abs(recent_return) > 0.05:
                triggers.append("extreme_return")
        
        # Check for VIX levels
        if 'vix_level' in features.columns:
            vix = features['vix_level'].iloc[-1]
            if vix > 30:
                triggers.append("high_vix")
            elif vix < 15:
                triggers.append("low_vix")
        
        # Check for technical indicators
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14'].iloc[-1]
            if rsi > 70:
                triggers.append("overbought")
            elif rsi < 30:
                triggers.append("oversold")
        
        return triggers
    
    def get_regime_trading_signals(self, 
                                 regime_info: RegimeInfo,
                                 risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """
        Generate trading signals based on current regime.
        
        Args:
            regime_info: Current regime information
            risk_tolerance: Risk tolerance level ("conservative", "moderate", "aggressive")
            
        Returns:
            Dictionary with trading recommendations
        """
        signals = {
            'regime_type': regime_info.regime_type.value,
            'confidence': regime_info.confidence,
            'position_sizing': 1.0,
            'recommended_strategies': [],
            'risk_adjustments': {}
        }
        
        # Adjust signals based on regime type
        if regime_info.regime_type == RegimeType.BULL:
            signals['position_sizing'] = 1.2 if risk_tolerance == "aggressive" else 1.0
            signals['recommended_strategies'] = ["trend_following", "momentum", "buy_dips"]
            signals['risk_adjustments'] = {"stop_loss": 0.03, "take_profit": 0.06}
            
        elif regime_info.regime_type == RegimeType.BEAR:
            signals['position_sizing'] = 0.5 if risk_tolerance == "conservative" else 0.7
            signals['recommended_strategies'] = ["short_selling", "volatility_trading", "defensive"]
            signals['risk_adjustments'] = {"stop_loss": 0.02, "take_profit": 0.04}
            
        elif regime_info.regime_type == RegimeType.CRISIS:
            signals['position_sizing'] = 0.2 if risk_tolerance != "aggressive" else 0.4
            signals['recommended_strategies'] = ["cash", "hedging", "safe_haven"]
            signals['risk_adjustments'] = {"stop_loss": 0.01, "reduce_leverage": True}
            
        elif regime_info.regime_type == RegimeType.SIDEWAYS:
            signals['position_sizing'] = 0.8
            signals['recommended_strategies'] = ["mean_reversion", "range_trading", "options"]
            signals['risk_adjustments'] = {"stop_loss": 0.025, "take_profit": 0.03}
            
        elif regime_info.regime_type == RegimeType.RECOVERY:
            signals['position_sizing'] = 1.0 if risk_tolerance != "conservative" else 0.8
            signals['recommended_strategies'] = ["value_investing", "gradual_accumulation"]
            signals['risk_adjustments'] = {"stop_loss": 0.04, "scale_in": True}
        
        # Adjust for confidence
        if regime_info.confidence < 0.7:
            signals['position_sizing'] *= 0.8
            signals['risk_adjustments']['reduce_size'] = True
        
        # Add transition risk warning
        if regime_info.transition_probability > 0.3:
            signals['warnings'] = ["High regime transition probability - consider reducing positions"]
        
        return signals
    
    def save(self, path: str):
        """Save the complete regime detection model."""
        model_data = {
            'config': self.config,
            'hmm_model': self.hmm_model,
            'feature_engineer': self.feature_engineer,
            'classifier_state': self.classifier.__dict__ if self.classifier else None,
            'transition_analyzer': self.transition_analyzer,
            'training_info': self.training_data_info,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MarketRegimeHMM':
        """Load a saved regime detection model."""
        model_data = joblib.load(path)
        
        # Create instance with loaded config
        instance = cls(model_data['config'])
        
        # Restore components
        instance.hmm_model = model_data['hmm_model']
        instance.feature_engineer = model_data['feature_engineer']
        instance.transition_analyzer = model_data['transition_analyzer']
        instance.training_data_info = model_data['training_info']
        instance.performance_metrics = model_data['performance_metrics']
        instance.is_fitted = True
        
        # Restore classifier if available
        if model_data['classifier_state']:
            instance.classifier = RegimeClassifier(
                instance.hmm_model,
                instance.feature_engineer
            )
            instance.classifier.__dict__.update(model_data['classifier_state'])
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def get_model_summary(self) -> str:
        """Get a summary of the model configuration and performance."""
        summary = []
        summary.append("=" * 60)
        summary.append("MARKET REGIME HMM MODEL SUMMARY")
        summary.append("=" * 60)
        
        # Configuration
        summary.append("\nCONFIGURATION:")
        summary.append(f"  Number of regimes: {self.config.hmm_config.n_states}")
        summary.append(f"  Covariance type: {self.config.hmm_config.covariance_type}")
        summary.append(f"  Model variant: {'GARCH' if self.config.use_regime_switching_garch else 'Standard'}")
        
        # Training info
        if self.training_data_info:
            summary.append("\nTRAINING INFO:")
            summary.append(f"  Training samples: {self.training_data_info['n_samples']}")
            summary.append(f"  Number of features: {self.training_data_info['n_features']}")
            summary.append(f"  Date range: {self.training_data_info['date_range'][0]} to {self.training_data_info['date_range'][1]}")
            summary.append(f"  Convergence iterations: {self.training_data_info['convergence_iterations']}")
        
        # Current state
        if self.classifier and self.classifier.current_regime_info:
            summary.append("\nCURRENT STATE:")
            info = self.classifier.current_regime_info
            summary.append(f"  Current regime: {info.regime_type.value}")
            summary.append(f"  Confidence: {info.confidence:.2%}")
            summary.append(f"  Duration: {info.duration} periods")
            summary.append(f"  Transition probability: {info.transition_probability:.2%}")
        
        # Performance metrics
        if self.performance_metrics['regime_stability']:
            summary.append("\nPERFORMANCE:")
            summary.append(f"  Regime stability: {np.mean(self.performance_metrics['regime_stability']):.3f}")
        
        # Regime statistics
        summary.append("\nREGIME STATISTICS:")
        if self.hmm_model.is_fitted:
            regime_stats = self.hmm_model.get_regime_statistics(
                np.arange(self.hmm_model.n_states)
            )
            for _, row in regime_stats.iterrows():
                summary.append(f"  {row['type']}: frequency={row['frequency']:.1%}, avg_duration={row['avg_duration']:.1f}")
        
        return "\n".join(summary)