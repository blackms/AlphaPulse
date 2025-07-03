"""
Market regime detection using multiple indicators and models.

Implements regime classification using volatility, momentum, liquidity,
and sentiment indicators with machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import warnings

from alpha_pulse.models.market_regime import (
    MarketRegime, RegimeType, RegimeIndicator, RegimeIndicatorType,
    RegimeDetectionResult, RegimeParameters, RegimeTransition
)
from alpha_pulse.utils.regime_indicators import RegimeIndicatorCalculator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Advanced market regime detection system."""
    
    def __init__(
        self,
        parameters: Optional[RegimeParameters] = None,
        indicator_calculator: Optional[RegimeIndicatorCalculator] = None
    ):
        """Initialize regime detector."""
        self.parameters = parameters or RegimeParameters()
        self.parameters.normalize_weights()
        
        self.indicator_calculator = indicator_calculator or RegimeIndicatorCalculator()
        
        # Initialize models
        self.hmm_model = None
        self.rf_classifier = None
        self.gmm_model = None
        self.scaler = StandardScaler()
        
        # State tracking
        self.current_regime = None
        self.regime_history = []
        self.indicator_history = {}
        
        # Model training data
        self.training_data = None
        self.is_trained = False
        
    def detect_regime(
        self,
        market_data: pd.DataFrame,
        additional_indicators: Optional[Dict[str, float]] = None
    ) -> RegimeDetectionResult:
        """Detect current market regime from market data."""
        logger.info("Starting regime detection")
        
        # Calculate all indicators
        indicators = self._calculate_indicators(market_data, additional_indicators)
        
        # Get regime probabilities from ensemble
        regime_probs = self._ensemble_regime_prediction(indicators)
        
        # Determine current regime
        current_regime_type = max(regime_probs, key=regime_probs.get)
        confidence = regime_probs[current_regime_type]
        
        # Create regime object
        current_regime = self._create_regime(
            current_regime_type,
            indicators,
            confidence
        )
        
        # Calculate transition risk
        transition_risk = self._calculate_transition_risk(
            current_regime,
            self.current_regime,
            indicators
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_regime,
            transition_risk
        )
        
        # Create result
        result = RegimeDetectionResult(
            current_regime=current_regime,
            regime_probabilities=regime_probs,
            transition_risk=transition_risk,
            previous_regime=self.current_regime,
            regime_history=self.regime_history[-10:],  # Last 10 regimes
            indicator_summary=self._summarize_indicators(indicators),
            composite_score=self._calculate_composite_score(indicators),
            next_likely_regime=self._predict_next_regime(current_regime, indicators),
            recommended_risk_level=recommendations["risk_level"],
            position_sizing_multiplier=recommendations["position_sizing"],
            hedging_recommendations=recommendations["hedging"]
        )
        
        # Update state
        self._update_state(current_regime, indicators)
        
        return result
    
    def train_models(
        self,
        historical_data: pd.DataFrame,
        labeled_regimes: Optional[pd.Series] = None
    ):
        """Train regime detection models on historical data."""
        logger.info("Training regime detection models")
        
        # Prepare training data
        X, y = self._prepare_training_data(historical_data, labeled_regimes)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self._train_random_forest(X_scaled, y)
        
        # Train GMM for unsupervised detection
        self._train_gmm(X_scaled)
        
        # Train HMM for regime transitions
        self._train_hmm(X_scaled, y)
        
        self.is_trained = True
        logger.info("Model training completed")
    
    def _calculate_indicators(
        self,
        market_data: pd.DataFrame,
        additional_indicators: Optional[Dict[str, float]] = None
    ) -> Dict[str, RegimeIndicator]:
        """Calculate all regime indicators."""
        indicators = {}
        
        # Volatility indicators
        vol_indicators = self.indicator_calculator.calculate_volatility_indicators(
            market_data
        )
        indicators.update(vol_indicators)
        
        # Momentum indicators
        mom_indicators = self.indicator_calculator.calculate_momentum_indicators(
            market_data
        )
        indicators.update(mom_indicators)
        
        # Liquidity indicators
        liq_indicators = self.indicator_calculator.calculate_liquidity_indicators(
            market_data
        )
        indicators.update(liq_indicators)
        
        # Sentiment indicators
        sent_indicators = self.indicator_calculator.calculate_sentiment_indicators(
            market_data,
            additional_indicators
        )
        indicators.update(sent_indicators)
        
        # Technical indicators
        tech_indicators = self.indicator_calculator.calculate_technical_indicators(
            market_data
        )
        indicators.update(tech_indicators)
        
        return indicators
    
    def _ensemble_regime_prediction(
        self,
        indicators: Dict[str, RegimeIndicator]
    ) -> Dict[RegimeType, float]:
        """Get regime predictions from ensemble of models."""
        # Convert indicators to feature vector
        features = self._indicators_to_features(indicators)
        
        predictions = {}
        weights = {}
        
        # Rule-based prediction
        rule_based = self._rule_based_regime_detection(indicators)
        predictions["rule_based"] = rule_based
        weights["rule_based"] = 0.3
        
        # ML predictions if trained
        if self.is_trained:
            features_scaled = self.scaler.transform([features])
            
            # Random Forest prediction
            if self.rf_classifier:
                rf_probs = self.rf_classifier.predict_proba(features_scaled)[0]
                rf_pred = self._probs_to_regime_dict(rf_probs)
                predictions["random_forest"] = rf_pred
                weights["random_forest"] = 0.4
            
            # GMM prediction
            if self.gmm_model:
                gmm_probs = self.gmm_model.predict_proba(features_scaled)[0]
                gmm_pred = self._gmm_to_regime_probs(gmm_probs)
                predictions["gmm"] = gmm_pred
                weights["gmm"] = 0.3
        else:
            # If not trained, rely more on rule-based
            weights["rule_based"] = 1.0
        
        # Combine predictions
        ensemble_probs = self._combine_predictions(predictions, weights)
        
        return ensemble_probs
    
    def _rule_based_regime_detection(
        self,
        indicators: Dict[str, RegimeIndicator]
    ) -> Dict[RegimeType, float]:
        """Rule-based regime detection using thresholds."""
        scores = {
            RegimeType.BULL: 0.0,
            RegimeType.BEAR: 0.0,
            RegimeType.SIDEWAYS: 0.0,
            RegimeType.CRISIS: 0.0,
            RegimeType.RECOVERY: 0.0
        }
        
        # Get key indicators
        vix_level = self._get_indicator_value(indicators, "vix_level")
        momentum_1m = self._get_indicator_value(indicators, "momentum_1m")
        momentum_3m = self._get_indicator_value(indicators, "momentum_3m")
        liquidity_score = self._get_indicator_value(indicators, "liquidity_composite")
        sentiment_score = self._get_indicator_value(indicators, "sentiment_composite")
        
        # Bull market rules
        if vix_level < 20 and momentum_3m > 0.05 and sentiment_score > 0.3:
            scores[RegimeType.BULL] += 0.8
            if liquidity_score > 0.5:
                scores[RegimeType.BULL] += 0.2
        
        # Bear market rules
        if vix_level > 30 and momentum_3m < -0.05 and sentiment_score < -0.3:
            scores[RegimeType.BEAR] += 0.8
            if liquidity_score < -0.3:
                scores[RegimeType.BEAR] += 0.2
        
        # Crisis rules
        if vix_level > 40 and momentum_1m < -0.10:
            scores[RegimeType.CRISIS] += 0.9
            if liquidity_score < -0.5:
                scores[RegimeType.CRISIS] += 0.1
        
        # Sideways market rules
        if 20 <= vix_level <= 30 and abs(momentum_3m) < 0.02:
            scores[RegimeType.SIDEWAYS] += 0.7
            if abs(sentiment_score) < 0.2:
                scores[RegimeType.SIDEWAYS] += 0.3
        
        # Recovery rules (after crisis)
        if self.current_regime and self.current_regime.regime_type == RegimeType.CRISIS:
            if vix_level < 35 and momentum_1m > 0.02:
                scores[RegimeType.RECOVERY] += 0.8
        
        # Normalize scores to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            for regime in scores:
                scores[regime] /= total_score
        else:
            # Default to sideways if no clear signal
            scores[RegimeType.SIDEWAYS] = 1.0
        
        return scores
    
    def _create_regime(
        self,
        regime_type: RegimeType,
        indicators: Dict[str, RegimeIndicator],
        confidence: float
    ) -> MarketRegime:
        """Create market regime object with characteristics."""
        # Determine regime characteristics
        vix_level = self._get_indicator_value(indicators, "vix_level")
        
        if vix_level < 15:
            volatility_level = "low"
        elif vix_level < 25:
            volatility_level = "normal"
        elif vix_level < 35:
            volatility_level = "high"
        else:
            volatility_level = "extreme"
        
        # Momentum direction
        momentum = self._get_indicator_value(indicators, "momentum_3m")
        if momentum > 0.05:
            momentum_direction = "positive"
        elif momentum < -0.05:
            momentum_direction = "negative"
        else:
            momentum_direction = "neutral"
        
        # Liquidity condition
        liquidity = self._get_indicator_value(indicators, "liquidity_composite")
        if liquidity > 0.3:
            liquidity_condition = "abundant"
        elif liquidity < -0.3:
            liquidity_condition = "poor"
        else:
            liquidity_condition = "normal"
        
        # Sentiment score
        sentiment = self._get_indicator_value(indicators, "sentiment_composite")
        
        # Risk parameters based on regime
        risk_params = self._get_regime_risk_parameters(regime_type)
        
        # Strategy preferences
        strategies = self._get_regime_strategies(regime_type)
        
        regime = MarketRegime(
            regime_type=regime_type,
            start_date=datetime.utcnow(),
            confidence=confidence,
            volatility_level=volatility_level,
            momentum_direction=momentum_direction,
            liquidity_condition=liquidity_condition,
            sentiment_score=sentiment,
            suggested_leverage=risk_params["leverage"],
            max_position_size=risk_params["max_position"],
            stop_loss_multiplier=risk_params["stop_loss"],
            preferred_strategies=strategies["preferred"],
            avoided_strategies=strategies["avoided"],
            indicators=indicators
        )
        
        return regime
    
    def _calculate_transition_risk(
        self,
        current_regime: MarketRegime,
        previous_regime: Optional[MarketRegime],
        indicators: Dict[str, RegimeIndicator]
    ) -> float:
        """Calculate probability of regime transition."""
        if not previous_regime:
            return 0.0
        
        # Base transition risk on regime stability
        base_risk = 0.1
        
        # Check indicator divergences
        divergence_count = 0
        for indicator in indicators.values():
            if abs(indicator.signal) > 0.7:  # Strong signal
                if (indicator.signal > 0 and current_regime.regime_type == RegimeType.BEAR) or \
                   (indicator.signal < 0 and current_regime.regime_type == RegimeType.BULL):
                    divergence_count += 1
        
        # Increase risk based on divergences
        risk = base_risk + (divergence_count * 0.1)
        
        # Consider regime duration
        if previous_regime.duration_days and previous_regime.duration_days < 20:
            risk += 0.2  # Recent regime change increases transition risk
        
        # Consider confidence
        if current_regime.confidence < 0.6:
            risk += 0.2
        
        return min(risk, 0.9)  # Cap at 90%
    
    def _generate_recommendations(
        self,
        regime: MarketRegime,
        transition_risk: float
    ) -> Dict[str, Any]:
        """Generate risk management recommendations."""
        recommendations = {
            "risk_level": 1.0,
            "position_sizing": 1.0,
            "hedging": []
        }
        
        # Adjust risk level based on regime
        risk_levels = {
            RegimeType.BULL: 1.2,
            RegimeType.BEAR: 0.6,
            RegimeType.SIDEWAYS: 0.8,
            RegimeType.CRISIS: 0.3,
            RegimeType.RECOVERY: 0.7
        }
        
        recommendations["risk_level"] = risk_levels.get(regime.regime_type, 1.0)
        
        # Adjust for transition risk
        if transition_risk > 0.5:
            recommendations["risk_level"] *= 0.8
            recommendations["hedging"].append("Increase hedging due to high transition risk")
        
        # Position sizing
        recommendations["position_sizing"] = regime.suggested_leverage
        
        # Regime-specific hedging
        if regime.regime_type == RegimeType.BEAR:
            recommendations["hedging"].extend([
                "Consider put options for downside protection",
                "Increase allocation to defensive sectors",
                "Reduce leverage on high-beta positions"
            ])
        elif regime.regime_type == RegimeType.CRISIS:
            recommendations["hedging"].extend([
                "Maximize cash position",
                "Use VIX calls for tail hedging",
                "Focus on capital preservation"
            ])
        elif regime.volatility_level == "extreme":
            recommendations["hedging"].append("Reduce position sizes due to extreme volatility")
        
        return recommendations
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest classifier."""
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        
        self.rf_classifier.fit(X, y)
        
        # Feature importance
        feature_importance = self.rf_classifier.feature_importances_
        logger.info(f"Top feature importances: {feature_importance[:5]}")
    
    def _train_gmm(self, X: np.ndarray):
        """Train Gaussian Mixture Model."""
        self.gmm_model = GaussianMixture(
            n_components=5,  # 5 regime types
            covariance_type='full',
            max_iter=100,
            random_state=42
        )
        
        self.gmm_model.fit(X)
    
    def _train_hmm(self, X: np.ndarray, y: np.ndarray):
        """Train Hidden Markov Model for regime transitions."""
        # Simplified HMM training - in practice would use hmmlearn
        # For now, calculate transition matrix from observed transitions
        n_regimes = 5
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(1, len(y)):
            prev_regime = int(y[i-1])
            curr_regime = int(y[i])
            transition_matrix[prev_regime, curr_regime] += 1
        
        # Normalize rows
        for i in range(n_regimes):
            if transition_matrix[i].sum() > 0:
                transition_matrix[i] /= transition_matrix[i].sum()
        
        self.transition_matrix = transition_matrix
    
    def _prepare_training_data(
        self,
        historical_data: pd.DataFrame,
        labeled_regimes: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for models."""
        # Calculate indicators for all historical periods
        features_list = []
        labels_list = []
        
        window = self.parameters.lookback_window
        
        for i in range(window, len(historical_data)):
            window_data = historical_data.iloc[i-window:i]
            
            # Calculate indicators
            indicators = self._calculate_indicators(window_data)
            features = self._indicators_to_features(indicators)
            features_list.append(features)
            
            # Get label
            if labeled_regimes is not None:
                label = labeled_regimes.iloc[i]
            else:
                # Use rule-based detection as pseudo-labels
                regime_probs = self._rule_based_regime_detection(indicators)
                label = max(regime_probs, key=regime_probs.get).value
            
            labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        return X, y
    
    def _indicators_to_features(
        self,
        indicators: Dict[str, RegimeIndicator]
    ) -> np.ndarray:
        """Convert indicators to feature vector."""
        # Fixed order of features
        feature_names = [
            "vix_level", "vix_term_structure", "realized_vol",
            "momentum_1m", "momentum_3m", "momentum_6m",
            "trend_strength", "liquidity_composite",
            "bid_ask_spread", "volume_ratio",
            "sentiment_composite", "put_call_ratio"
        ]
        
        features = []
        for name in feature_names:
            if name in indicators:
                features.append(indicators[name].normalized_value)
            else:
                features.append(0.5)  # Default neutral value
        
        return np.array(features)
    
    def _get_indicator_value(
        self,
        indicators: Dict[str, RegimeIndicator],
        name: str,
        default: float = 0.0
    ) -> float:
        """Get indicator value with default."""
        if name in indicators:
            return indicators[name].value
        return default
    
    def _get_regime_risk_parameters(
        self,
        regime_type: RegimeType
    ) -> Dict[str, float]:
        """Get risk parameters for regime type."""
        params = {
            RegimeType.BULL: {
                "leverage": 1.5,
                "max_position": 0.15,
                "stop_loss": 0.8
            },
            RegimeType.BEAR: {
                "leverage": 0.5,
                "max_position": 0.08,
                "stop_loss": 1.5
            },
            RegimeType.SIDEWAYS: {
                "leverage": 1.0,
                "max_position": 0.10,
                "stop_loss": 1.0
            },
            RegimeType.CRISIS: {
                "leverage": 0.2,
                "max_position": 0.05,
                "stop_loss": 2.0
            },
            RegimeType.RECOVERY: {
                "leverage": 0.8,
                "max_position": 0.10,
                "stop_loss": 1.2
            }
        }
        
        return params.get(regime_type, {
            "leverage": 1.0,
            "max_position": 0.10,
            "stop_loss": 1.0
        })
    
    def _get_regime_strategies(
        self,
        regime_type: RegimeType
    ) -> Dict[str, List[str]]:
        """Get preferred and avoided strategies for regime."""
        strategies = {
            RegimeType.BULL: {
                "preferred": ["momentum", "growth", "breakout", "trend_following"],
                "avoided": ["mean_reversion", "value", "defensive"]
            },
            RegimeType.BEAR: {
                "preferred": ["defensive", "value", "quality", "short_selling"],
                "avoided": ["momentum", "growth", "leverage"]
            },
            RegimeType.SIDEWAYS: {
                "preferred": ["mean_reversion", "pairs_trading", "volatility_selling"],
                "avoided": ["trend_following", "breakout"]
            },
            RegimeType.CRISIS: {
                "preferred": ["cash", "hedging", "tail_protection"],
                "avoided": ["leverage", "concentration", "illiquid"]
            },
            RegimeType.RECOVERY: {
                "preferred": ["quality", "value", "selective_momentum"],
                "avoided": ["excessive_risk", "leverage"]
            }
        }
        
        return strategies.get(regime_type, {
            "preferred": [],
            "avoided": []
        })
    
    def _summarize_indicators(
        self,
        indicators: Dict[str, RegimeIndicator]
    ) -> Dict[RegimeIndicatorType, float]:
        """Summarize indicators by type."""
        summary = {}
        
        for ind_type in RegimeIndicatorType:
            type_indicators = [
                ind for ind in indicators.values()
                if ind.indicator_type == ind_type
            ]
            
            if type_indicators:
                # Weighted average of signals
                total_weight = sum(ind.weight for ind in type_indicators)
                if total_weight > 0:
                    weighted_signal = sum(
                        ind.signal * ind.weight for ind in type_indicators
                    ) / total_weight
                    summary[ind_type] = weighted_signal
                else:
                    summary[ind_type] = 0.0
            else:
                summary[ind_type] = 0.0
        
        return summary
    
    def _calculate_composite_score(
        self,
        indicators: Dict[str, RegimeIndicator]
    ) -> float:
        """Calculate overall market health score."""
        # Get weighted signals by type
        type_summary = self._summarize_indicators(indicators)
        
        # Apply type weights
        composite = 0.0
        for ind_type, signal in type_summary.items():
            weight = self.parameters.indicator_weights.get(ind_type, 0.0)
            composite += signal * weight
        
        # Normalize to 0-1 scale
        return (composite + 1) / 2
    
    def _predict_next_regime(
        self,
        current_regime: MarketRegime,
        indicators: Dict[str, RegimeIndicator]
    ) -> Optional[RegimeType]:
        """Predict most likely next regime."""
        if not hasattr(self, 'transition_matrix'):
            return None
        
        current_idx = list(RegimeType).index(current_regime.regime_type)
        transition_probs = self.transition_matrix[current_idx]
        
        # Adjust based on current indicators
        # (simplified - in practice would use more sophisticated model)
        momentum = self._get_indicator_value(indicators, "momentum_3m")
        
        if momentum > 0.05:
            # Increase probability of bullish transitions
            if RegimeType.BULL in RegimeType:
                bull_idx = list(RegimeType).index(RegimeType.BULL)
                transition_probs[bull_idx] *= 1.2
        elif momentum < -0.05:
            # Increase probability of bearish transitions
            if RegimeType.BEAR in RegimeType:
                bear_idx = list(RegimeType).index(RegimeType.BEAR)
                transition_probs[bear_idx] *= 1.2
        
        # Normalize
        if transition_probs.sum() > 0:
            transition_probs /= transition_probs.sum()
        
        # Get most likely
        next_idx = np.argmax(transition_probs)
        return list(RegimeType)[next_idx]
    
    def _update_state(
        self,
        new_regime: MarketRegime,
        indicators: Dict[str, RegimeIndicator]
    ):
        """Update detector state."""
        # Check if regime changed
        if self.current_regime is None or \
           self.current_regime.regime_type != new_regime.regime_type:
            # End previous regime
            if self.current_regime:
                self.current_regime.end_date = datetime.utcnow()
                self.regime_history.append(self.current_regime)
            
            # Start new regime
            self.current_regime = new_regime
            logger.info(f"Regime change detected: {new_regime.regime_type}")
        else:
            # Update current regime confidence
            self.current_regime.confidence = new_regime.confidence
        
        # Update indicator history
        for name, indicator in indicators.items():
            if name not in self.indicator_history:
                self.indicator_history[name] = []
            
            self.indicator_history[name].append({
                "timestamp": datetime.utcnow(),
                "value": indicator.value,
                "signal": indicator.signal
            })
            
            # Keep only recent history
            if len(self.indicator_history[name]) > 1000:
                self.indicator_history[name] = self.indicator_history[name][-1000:]
    
    def _combine_predictions(
        self,
        predictions: Dict[str, Dict[RegimeType, float]],
        weights: Dict[str, float]
    ) -> Dict[RegimeType, float]:
        """Combine predictions from multiple models."""
        combined = {regime: 0.0 for regime in RegimeType}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return combined
        
        norm_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted average
        for model, weight in norm_weights.items():
            if model in predictions:
                for regime, prob in predictions[model].items():
                    combined[regime] += prob * weight
        
        # Ensure probabilities sum to 1
        total = sum(combined.values())
        if total > 0:
            for regime in combined:
                combined[regime] /= total
        
        return combined
    
    def _probs_to_regime_dict(self, probs: np.ndarray) -> Dict[RegimeType, float]:
        """Convert probability array to regime dictionary."""
        regime_list = list(RegimeType)[:5]  # Exclude UNKNOWN
        return {regime: probs[i] for i, regime in enumerate(regime_list)}
    
    def _gmm_to_regime_probs(self, gmm_probs: np.ndarray) -> Dict[RegimeType, float]:
        """Map GMM components to regime probabilities."""
        # Simple mapping - in practice would use more sophisticated approach
        return self._probs_to_regime_dict(gmm_probs)
    
    def get_regime_analytics(self) -> Dict[str, Any]:
        """Get analytics on regime detection performance."""
        if not self.regime_history:
            return {}
        
        analytics = {
            "total_regimes": len(self.regime_history),
            "regime_durations": {},
            "regime_frequencies": {},
            "average_confidence": 0.0
        }
        
        # Calculate statistics
        total_days = 0
        confidence_sum = 0
        
        for regime in self.regime_history:
            regime_type = regime.regime_type
            
            # Duration
            if regime.duration_days:
                if regime_type not in analytics["regime_durations"]:
                    analytics["regime_durations"][regime_type] = []
                analytics["regime_durations"][regime_type].append(regime.duration_days)
                total_days += regime.duration_days
            
            # Confidence
            confidence_sum += regime.confidence
        
        # Frequencies
        for regime_type in RegimeType:
            count = sum(1 for r in self.regime_history if r.regime_type == regime_type)
            analytics["regime_frequencies"][regime_type] = count / len(self.regime_history)
        
        # Average confidence
        analytics["average_confidence"] = confidence_sum / len(self.regime_history)
        
        # Average durations
        analytics["average_durations"] = {}
        for regime_type, durations in analytics["regime_durations"].items():
            if durations:
                analytics["average_durations"][regime_type] = np.mean(durations)
        
        return analytics