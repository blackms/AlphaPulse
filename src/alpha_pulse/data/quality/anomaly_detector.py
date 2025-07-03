"""
Statistical and machine learning-based anomaly detection for market data.

Provides:
- Z-score and IQR statistical anomaly detection
- Isolation Forest for multivariate anomaly detection
- LSTM autoencoder for time series anomaly detection
- One-class SVM for outlier detection
- Ensemble anomaly detection methods
- Real-time anomaly scoring and alerts
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import deque
import joblib
from loguru import logger

# ML libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report
    import tensorflow as tf
    from tensorflow import keras
    SKLEARN_AVAILABLE = True
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML libraries not available: {e}")
    SKLEARN_AVAILABLE = False
    TENSORFLOW_AVAILABLE = False

from alpha_pulse.models.market_data import MarketDataPoint, TimeSeriesData, OHLCV
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class AnomalyMethod(Enum):
    """Anomaly detection methods."""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    ENSEMBLE = "ensemble"
    MOVING_AVERAGE = "moving_average"
    BOLLINGER_BANDS = "bollinger_bands"


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    timestamp: datetime
    symbol: str
    method: AnomalyMethod
    anomaly_score: float
    is_anomaly: bool
    severity: AnomalySeverity
    confidence: float
    details: Dict[str, Any]
    affected_fields: List[str]
    suggested_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'method': self.method.value,
            'anomaly_score': self.anomaly_score,
            'is_anomaly': self.is_anomaly,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'details': self.details,
            'affected_fields': self.affected_fields,
            'suggested_action': self.suggested_action
        }


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detectors."""
    # Statistical thresholds
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # ML model parameters
    isolation_forest_contamination: float = 0.1
    isolation_forest_n_estimators: int = 100
    svm_nu: float = 0.1
    
    # LSTM autoencoder parameters
    lstm_sequence_length: int = 50
    lstm_encoding_dim: int = 32
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'z_score': 0.2,
        'iqr': 0.2,
        'isolation_forest': 0.3,
        'one_class_svm': 0.3
    })
    
    # General parameters
    minimum_data_points: int = 30
    historical_window_size: int = 200
    retrain_interval_hours: int = 24


class StatisticalAnomalyDetector:
    """Statistical anomaly detection methods."""
    
    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config
        self.historical_data: Dict[str, deque] = {}
    
    def detect_z_score_anomaly(
        self, 
        data_point: MarketDataPoint,
        historical_context: List[MarketDataPoint]
    ) -> AnomalyResult:
        """Detect anomalies using z-score analysis."""
        if len(historical_context) < self.config.minimum_data_points:
            return self._create_no_anomaly_result(data_point, AnomalyMethod.Z_SCORE)
        
        anomalies = []
        max_score = 0.0
        affected_fields = []
        
        if data_point.ohlcv:
            # Price anomaly detection
            prices = [p.ohlcv.close for p in historical_context if p.ohlcv]
            if len(prices) >= self.config.minimum_data_points:
                price_mean = statistics.mean(prices)
                price_std = statistics.stdev(prices)
                
                if price_std > 0:
                    z_score = abs((float(data_point.ohlcv.close) - price_mean) / price_std)
                    if z_score > self.config.z_score_threshold:
                        anomalies.append({
                            'field': 'close_price',
                            'z_score': z_score,
                            'threshold': self.config.z_score_threshold,
                            'value': float(data_point.ohlcv.close),
                            'mean': price_mean,
                            'std': price_std
                        })
                        affected_fields.append('close_price')
                        max_score = max(max_score, z_score / self.config.z_score_threshold)
            
            # Volume anomaly detection
            volumes = [float(p.ohlcv.volume) for p in historical_context if p.ohlcv and p.ohlcv.volume > 0]
            if len(volumes) >= self.config.minimum_data_points:
                volume_mean = statistics.mean(volumes)
                volume_std = statistics.stdev(volumes)
                
                if volume_std > 0:
                    z_score = abs((float(data_point.ohlcv.volume) - volume_mean) / volume_std)
                    if z_score > self.config.z_score_threshold:
                        anomalies.append({
                            'field': 'volume',
                            'z_score': z_score,
                            'threshold': self.config.z_score_threshold,
                            'value': float(data_point.ohlcv.volume),
                            'mean': volume_mean,
                            'std': volume_std
                        })
                        affected_fields.append('volume')
                        max_score = max(max_score, z_score / self.config.z_score_threshold)
        
        is_anomaly = len(anomalies) > 0
        severity = self._calculate_severity(max_score)
        
        return AnomalyResult(
            timestamp=data_point.timestamp,
            symbol=data_point.symbol,
            method=AnomalyMethod.Z_SCORE,
            anomaly_score=max_score,
            is_anomaly=is_anomaly,
            severity=severity,
            confidence=min(1.0, len(historical_context) / self.config.minimum_data_points),
            details={'anomalies': anomalies},
            affected_fields=affected_fields,
            suggested_action="Review data source for potential issues" if is_anomaly else "No action required"
        )
    
    def detect_iqr_anomaly(
        self, 
        data_point: MarketDataPoint,
        historical_context: List[MarketDataPoint]
    ) -> AnomalyResult:
        """Detect anomalies using Interquartile Range (IQR) method."""
        if len(historical_context) < self.config.minimum_data_points:
            return self._create_no_anomaly_result(data_point, AnomalyMethod.IQR)
        
        anomalies = []
        max_score = 0.0
        affected_fields = []
        
        if data_point.ohlcv:
            # Price IQR analysis
            prices = [float(p.ohlcv.close) for p in historical_context if p.ohlcv]
            if len(prices) >= self.config.minimum_data_points:
                q1 = np.percentile(prices, 25)
                q3 = np.percentile(prices, 75)
                iqr = q3 - q1
                lower_bound = q1 - self.config.iqr_multiplier * iqr
                upper_bound = q3 + self.config.iqr_multiplier * iqr
                
                current_price = float(data_point.ohlcv.close)
                if current_price < lower_bound or current_price > upper_bound:
                    deviation = min(abs(current_price - lower_bound), abs(current_price - upper_bound))
                    score = deviation / iqr if iqr > 0 else 0
                    
                    anomalies.append({
                        'field': 'close_price',
                        'value': current_price,
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'deviation': deviation
                    })
                    affected_fields.append('close_price')
                    max_score = max(max_score, score)
            
            # Volume IQR analysis
            volumes = [float(p.ohlcv.volume) for p in historical_context if p.ohlcv and p.ohlcv.volume > 0]
            if len(volumes) >= self.config.minimum_data_points:
                q1 = np.percentile(volumes, 25)
                q3 = np.percentile(volumes, 75)
                iqr = q3 - q1
                lower_bound = q1 - self.config.iqr_multiplier * iqr
                upper_bound = q3 + self.config.iqr_multiplier * iqr
                
                current_volume = float(data_point.ohlcv.volume)
                if current_volume < lower_bound or current_volume > upper_bound:
                    deviation = min(abs(current_volume - lower_bound), abs(current_volume - upper_bound))
                    score = deviation / iqr if iqr > 0 else 0
                    
                    anomalies.append({
                        'field': 'volume',
                        'value': current_volume,
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'deviation': deviation
                    })
                    affected_fields.append('volume')
                    max_score = max(max_score, score)
        
        is_anomaly = len(anomalies) > 0
        severity = self._calculate_severity(max_score)
        
        return AnomalyResult(
            timestamp=data_point.timestamp,
            symbol=data_point.symbol,
            method=AnomalyMethod.IQR,
            anomaly_score=max_score,
            is_anomaly=is_anomaly,
            severity=severity,
            confidence=min(1.0, len(historical_context) / self.config.minimum_data_points),
            details={'anomalies': anomalies},
            affected_fields=affected_fields,
            suggested_action="Investigate outlier values" if is_anomaly else "No action required"
        )
    
    def detect_moving_average_anomaly(
        self, 
        data_point: MarketDataPoint,
        historical_context: List[MarketDataPoint],
        window_size: int = 20
    ) -> AnomalyResult:
        """Detect anomalies using moving average deviation."""
        if len(historical_context) < window_size:
            return self._create_no_anomaly_result(data_point, AnomalyMethod.MOVING_AVERAGE)
        
        anomalies = []
        max_score = 0.0
        affected_fields = []
        
        if data_point.ohlcv:
            # Price moving average
            recent_prices = [float(p.ohlcv.close) for p in historical_context[-window_size:] if p.ohlcv]
            if len(recent_prices) == window_size:
                ma = statistics.mean(recent_prices)
                std = statistics.stdev(recent_prices)
                current_price = float(data_point.ohlcv.close)
                
                if std > 0:
                    deviation = abs(current_price - ma) / std
                    if deviation > 2.0:  # 2 standard deviations
                        anomalies.append({
                            'field': 'close_price',
                            'value': current_price,
                            'moving_average': ma,
                            'std': std,
                            'deviation': deviation
                        })
                        affected_fields.append('close_price')
                        max_score = max(max_score, deviation / 2.0)
        
        is_anomaly = len(anomalies) > 0
        severity = self._calculate_severity(max_score)
        
        return AnomalyResult(
            timestamp=data_point.timestamp,
            symbol=data_point.symbol,
            method=AnomalyMethod.MOVING_AVERAGE,
            anomaly_score=max_score,
            is_anomaly=is_anomaly,
            severity=severity,
            confidence=min(1.0, len(historical_context) / window_size),
            details={'anomalies': anomalies, 'window_size': window_size},
            affected_fields=affected_fields,
            suggested_action="Check for trend breaks or market events" if is_anomaly else "No action required"
        )
    
    def _create_no_anomaly_result(self, data_point: MarketDataPoint, method: AnomalyMethod) -> AnomalyResult:
        """Create a no-anomaly result."""
        return AnomalyResult(
            timestamp=data_point.timestamp,
            symbol=data_point.symbol,
            method=method,
            anomaly_score=0.0,
            is_anomaly=False,
            severity=AnomalySeverity.LOW,
            confidence=0.0,
            details={'reason': 'insufficient_data'},
            affected_fields=[],
            suggested_action="Collect more historical data for accurate detection"
        )
    
    def _calculate_severity(self, score: float) -> AnomalySeverity:
        """Calculate anomaly severity based on score."""
        if score >= 3.0:
            return AnomalySeverity.CRITICAL
        elif score >= 2.0:
            return AnomalySeverity.HIGH
        elif score >= 1.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class MLAnomalyDetector:
    """Machine learning-based anomaly detection."""
    
    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.is_trained: Dict[str, bool] = {}
        self.last_training: Dict[str, datetime] = {}
        
    def detect_isolation_forest_anomaly(
        self, 
        data_point: MarketDataPoint,
        historical_context: List[MarketDataPoint]
    ) -> AnomalyResult:
        """Detect anomalies using Isolation Forest."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for Isolation Forest")
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ISOLATION_FOREST)
        
        symbol = data_point.symbol
        
        # Check if model needs training/retraining
        if (symbol not in self.is_trained or 
            not self.is_trained[symbol] or
            self._needs_retraining(symbol)):
            
            success = self._train_isolation_forest(symbol, historical_context + [data_point])
            if not success:
                return self._create_no_anomaly_result(data_point, AnomalyMethod.ISOLATION_FOREST)
        
        if symbol not in self.models or symbol not in self.scalers:
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ISOLATION_FOREST)
        
        # Prepare data point for prediction
        features = self._extract_features(data_point)
        if features is None:
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ISOLATION_FOREST)
        
        try:
            # Scale features
            scaled_features = self.scalers[symbol].transform([features])
            
            # Predict anomaly
            prediction = self.models[symbol].predict(scaled_features)[0]
            anomaly_score = self.models[symbol].decision_function(scaled_features)[0]
            
            # Convert score to positive and normalize
            normalized_score = max(0, -anomaly_score)
            is_anomaly = prediction == -1
            
            severity = self._calculate_severity(normalized_score * 2)  # Scale for severity calculation
            
            return AnomalyResult(
                timestamp=data_point.timestamp,
                symbol=data_point.symbol,
                method=AnomalyMethod.ISOLATION_FOREST,
                anomaly_score=normalized_score,
                is_anomaly=is_anomaly,
                severity=severity,
                confidence=0.8,  # Isolation Forest typically has good confidence
                details={
                    'prediction': int(prediction),
                    'raw_score': float(anomaly_score),
                    'features': features
                },
                affected_fields=['multivariate'],
                suggested_action="Investigate multivariate anomaly pattern" if is_anomaly else "No action required"
            )
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest prediction: {e}")
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ISOLATION_FOREST)
    
    def detect_one_class_svm_anomaly(
        self, 
        data_point: MarketDataPoint,
        historical_context: List[MarketDataPoint]
    ) -> AnomalyResult:
        """Detect anomalies using One-Class SVM."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available for One-Class SVM")
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ONE_CLASS_SVM)
        
        symbol = data_point.symbol
        model_key = f"{symbol}_svm"
        
        # Check if model needs training/retraining
        if (model_key not in self.is_trained or 
            not self.is_trained[model_key] or
            self._needs_retraining(model_key)):
            
            success = self._train_one_class_svm(symbol, historical_context + [data_point])
            if not success:
                return self._create_no_anomaly_result(data_point, AnomalyMethod.ONE_CLASS_SVM)
        
        if model_key not in self.models or model_key not in self.scalers:
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ONE_CLASS_SVM)
        
        # Prepare data point for prediction
        features = self._extract_features(data_point)
        if features is None:
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ONE_CLASS_SVM)
        
        try:
            # Scale features
            scaled_features = self.scalers[model_key].transform([features])
            
            # Predict anomaly
            prediction = self.models[model_key].predict(scaled_features)[0]
            decision_score = self.models[model_key].decision_function(scaled_features)[0]
            
            # Normalize score
            normalized_score = max(0, -decision_score)
            is_anomaly = prediction == -1
            
            severity = self._calculate_severity(normalized_score * 2)
            
            return AnomalyResult(
                timestamp=data_point.timestamp,
                symbol=data_point.symbol,
                method=AnomalyMethod.ONE_CLASS_SVM,
                anomaly_score=normalized_score,
                is_anomaly=is_anomaly,
                severity=severity,
                confidence=0.75,
                details={
                    'prediction': int(prediction),
                    'decision_score': float(decision_score),
                    'features': features
                },
                affected_fields=['multivariate'],
                suggested_action="Investigate SVM-detected anomaly pattern" if is_anomaly else "No action required"
            )
            
        except Exception as e:
            logger.error(f"Error in One-Class SVM prediction: {e}")
            return self._create_no_anomaly_result(data_point, AnomalyMethod.ONE_CLASS_SVM)
    
    def _train_isolation_forest(self, symbol: str, data_points: List[MarketDataPoint]) -> bool:
        """Train Isolation Forest model."""
        try:
            # Extract features
            features_list = []
            for point in data_points:
                features = self._extract_features(point)
                if features is not None:
                    features_list.append(features)
            
            if len(features_list) < self.config.minimum_data_points:
                return False
            
            # Create and train model
            model = IsolationForest(
                contamination=self.config.isolation_forest_contamination,
                n_estimators=self.config.isolation_forest_n_estimators,
                random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_list)
            
            # Train model
            model.fit(scaled_features)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.is_trained[symbol] = True
            self.last_training[symbol] = datetime.utcnow()
            
            logger.info(f"Trained Isolation Forest for {symbol} with {len(features_list)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train Isolation Forest for {symbol}: {e}")
            return False
    
    def _train_one_class_svm(self, symbol: str, data_points: List[MarketDataPoint]) -> bool:
        """Train One-Class SVM model."""
        try:
            # Extract features
            features_list = []
            for point in data_points:
                features = self._extract_features(point)
                if features is not None:
                    features_list.append(features)
            
            if len(features_list) < self.config.minimum_data_points:
                return False
            
            model_key = f"{symbol}_svm"
            
            # Create and train model
            model = OneClassSVM(nu=self.config.svm_nu, kernel='rbf', gamma='scale')
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_list)
            
            # Train model
            model.fit(scaled_features)
            
            # Store model and scaler
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.is_trained[model_key] = True
            self.last_training[model_key] = datetime.utcnow()
            
            logger.info(f"Trained One-Class SVM for {symbol} with {len(features_list)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train One-Class SVM for {symbol}: {e}")
            return False
    
    def _extract_features(self, data_point: MarketDataPoint) -> Optional[List[float]]:
        """Extract features from a data point for ML models."""
        if not data_point.ohlcv:
            return None
        
        try:
            features = [
                float(data_point.ohlcv.open),
                float(data_point.ohlcv.high),
                float(data_point.ohlcv.low),
                float(data_point.ohlcv.close),
                float(data_point.ohlcv.volume),
            ]
            
            # Add derived features
            ohlc_range = float(data_point.ohlcv.high - data_point.ohlcv.low)
            body_size = abs(float(data_point.ohlcv.close - data_point.ohlcv.open))
            upper_shadow = float(data_point.ohlcv.high - max(data_point.ohlcv.open, data_point.ohlcv.close))
            lower_shadow = float(min(data_point.ohlcv.open, data_point.ohlcv.close) - data_point.ohlcv.low)
            
            features.extend([ohlc_range, body_size, upper_shadow, lower_shadow])
            
            # Add VWAP if available
            if data_point.ohlcv.vwap:
                features.append(float(data_point.ohlcv.vwap))
            else:
                features.append(float(data_point.ohlcv.close))  # Use close as fallback
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _needs_retraining(self, model_key: str) -> bool:
        """Check if model needs retraining."""
        if model_key not in self.last_training:
            return True
        
        last_train = self.last_training[model_key]
        hours_since_training = (datetime.utcnow() - last_train).total_seconds() / 3600
        
        return hours_since_training >= self.config.retrain_interval_hours
    
    def _create_no_anomaly_result(self, data_point: MarketDataPoint, method: AnomalyMethod) -> AnomalyResult:
        """Create a no-anomaly result."""
        return AnomalyResult(
            timestamp=data_point.timestamp,
            symbol=data_point.symbol,
            method=method,
            anomaly_score=0.0,
            is_anomaly=False,
            severity=AnomalySeverity.LOW,
            confidence=0.0,
            details={'reason': 'model_unavailable'},
            affected_fields=[],
            suggested_action="Model training required"
        )
    
    def _calculate_severity(self, score: float) -> AnomalySeverity:
        """Calculate anomaly severity based on score."""
        if score >= 2.0:
            return AnomalySeverity.CRITICAL
        elif score >= 1.5:
            return AnomalySeverity.HIGH
        elif score >= 1.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class AnomalyDetector:
    """Main anomaly detector orchestrating multiple detection methods."""
    
    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self.statistical_detector = StatisticalAnomalyDetector(self.config)
        self.ml_detector = MLAnomalyDetector(self.config)
        self.audit_logger = get_audit_logger()
        
        # Detection history for tracking
        self.detection_history: Dict[str, List[AnomalyResult]] = {}
    
    async def detect_anomalies(
        self, 
        data_point: MarketDataPoint,
        historical_context: List[MarketDataPoint],
        methods: Optional[List[AnomalyMethod]] = None
    ) -> List[AnomalyResult]:
        """
        Detect anomalies using multiple methods.
        
        Args:
            data_point: Current data point to analyze
            historical_context: Historical data for context
            methods: List of methods to use (if None, uses all available)
        
        Returns:
            List of anomaly detection results
        """
        if methods is None:
            methods = [
                AnomalyMethod.Z_SCORE,
                AnomalyMethod.IQR,
                AnomalyMethod.MOVING_AVERAGE
            ]
            
            # Add ML methods if libraries are available
            if SKLEARN_AVAILABLE:
                methods.extend([
                    AnomalyMethod.ISOLATION_FOREST,
                    AnomalyMethod.ONE_CLASS_SVM
                ])
        
        results = []
        
        # Run statistical methods
        if AnomalyMethod.Z_SCORE in methods:
            result = self.statistical_detector.detect_z_score_anomaly(data_point, historical_context)
            results.append(result)
        
        if AnomalyMethod.IQR in methods:
            result = self.statistical_detector.detect_iqr_anomaly(data_point, historical_context)
            results.append(result)
        
        if AnomalyMethod.MOVING_AVERAGE in methods:
            result = self.statistical_detector.detect_moving_average_anomaly(data_point, historical_context)
            results.append(result)
        
        # Run ML methods
        if SKLEARN_AVAILABLE:
            if AnomalyMethod.ISOLATION_FOREST in methods:
                result = self.ml_detector.detect_isolation_forest_anomaly(data_point, historical_context)
                results.append(result)
            
            if AnomalyMethod.ONE_CLASS_SVM in methods:
                result = self.ml_detector.detect_one_class_svm_anomaly(data_point, historical_context)
                results.append(result)
        
        # Run ensemble method if requested
        if AnomalyMethod.ENSEMBLE in methods and len(results) > 1:
            ensemble_result = self._create_ensemble_result(data_point, results)
            results.append(ensemble_result)
        
        # Store detection history
        symbol = data_point.symbol
        if symbol not in self.detection_history:
            self.detection_history[symbol] = []
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.detection_history[symbol] = [
            r for r in self.detection_history[symbol] 
            if r.timestamp > cutoff_time
        ]
        
        # Add current results
        self.detection_history[symbol].extend(results)
        
        # Log anomalies
        anomaly_results = [r for r in results if r.is_anomaly]
        if anomaly_results:
            await self._log_anomalies(data_point, anomaly_results)
        
        return results
    
    def _create_ensemble_result(
        self, 
        data_point: MarketDataPoint, 
        individual_results: List[AnomalyResult]
    ) -> AnomalyResult:
        """Create ensemble anomaly result from individual method results."""
        weights = self.config.ensemble_weights
        weighted_score = 0.0
        total_weight = 0.0
        anomaly_count = 0
        affected_fields = set()
        details = {}
        
        for result in individual_results:
            method_name = result.method.value
            weight = weights.get(method_name, 0.1)
            
            weighted_score += result.anomaly_score * weight
            total_weight += weight
            
            if result.is_anomaly:
                anomaly_count += 1
            
            affected_fields.update(result.affected_fields)
            details[method_name] = {
                'score': result.anomaly_score,
                'is_anomaly': result.is_anomaly,
                'severity': result.severity.value
            }
        
        # Calculate final ensemble score
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine if ensemble detects anomaly (majority vote with score weighting)
        anomaly_ratio = anomaly_count / len(individual_results)
        is_anomaly = anomaly_ratio >= 0.5 or final_score > 1.0
        
        severity = self._calculate_ensemble_severity(final_score, anomaly_ratio)
        confidence = min(1.0, total_weight)
        
        return AnomalyResult(
            timestamp=data_point.timestamp,
            symbol=data_point.symbol,
            method=AnomalyMethod.ENSEMBLE,
            anomaly_score=final_score,
            is_anomaly=is_anomaly,
            severity=severity,
            confidence=confidence,
            details={
                'individual_results': details,
                'anomaly_ratio': anomaly_ratio,
                'total_methods': len(individual_results)
            },
            affected_fields=list(affected_fields),
            suggested_action="Review multiple detection methods indicating anomaly" if is_anomaly else "No action required"
        )
    
    def _calculate_ensemble_severity(self, score: float, anomaly_ratio: float) -> AnomalySeverity:
        """Calculate ensemble severity."""
        combined_score = score + anomaly_ratio
        
        if combined_score >= 2.0:
            return AnomalySeverity.CRITICAL
        elif combined_score >= 1.5:
            return AnomalySeverity.HIGH
        elif combined_score >= 1.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    async def _log_anomalies(
        self, 
        data_point: MarketDataPoint, 
        anomaly_results: List[AnomalyResult]
    ) -> None:
        """Log detected anomalies."""
        for result in anomaly_results:
            severity_map = {
                AnomalySeverity.LOW: AuditSeverity.INFO,
                AnomalySeverity.MEDIUM: AuditSeverity.WARNING,
                AnomalySeverity.HIGH: AuditSeverity.WARNING,
                AnomalySeverity.CRITICAL: AuditSeverity.ERROR
            }
            
            self.audit_logger.log(
                event_type=AuditEventType.DATA_ANOMALY,
                event_data={
                    "symbol": data_point.symbol,
                    "method": result.method.value,
                    "anomaly_score": result.anomaly_score,
                    "severity": result.severity.value,
                    "confidence": result.confidence,
                    "affected_fields": result.affected_fields,
                    "details": result.details
                },
                severity=severity_map.get(result.severity, AuditSeverity.WARNING)
            )
    
    def get_anomaly_statistics(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get anomaly detection statistics for a symbol."""
        if symbol not in self.detection_history:
            return {}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_results = [
            r for r in self.detection_history[symbol] 
            if r.timestamp > cutoff_time
        ]
        
        if not recent_results:
            return {}
        
        # Calculate statistics by method
        method_stats = {}
        for method in AnomalyMethod:
            method_results = [r for r in recent_results if r.method == method]
            if method_results:
                anomaly_count = sum(1 for r in method_results if r.is_anomaly)
                avg_score = statistics.mean(r.anomaly_score for r in method_results)
                
                method_stats[method.value] = {
                    'total_detections': len(method_results),
                    'anomaly_count': anomaly_count,
                    'anomaly_rate': anomaly_count / len(method_results),
                    'avg_score': avg_score
                }
        
        # Overall statistics
        total_anomalies = sum(1 for r in recent_results if r.is_anomaly)
        severity_counts = {}
        for severity in AnomalySeverity:
            severity_counts[severity.value] = sum(
                1 for r in recent_results 
                if r.is_anomaly and r.severity == severity
            )
        
        return {
            'symbol': symbol,
            'time_period_hours': hours,
            'total_detections': len(recent_results),
            'total_anomalies': total_anomalies,
            'overall_anomaly_rate': total_anomalies / len(recent_results),
            'method_statistics': method_stats,
            'severity_distribution': severity_counts
        }


# Global detector instance
_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector(config: Optional[AnomalyDetectorConfig] = None) -> AnomalyDetector:
    """Get the global anomaly detector instance."""
    global _anomaly_detector
    
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector(config)
    
    return _anomaly_detector