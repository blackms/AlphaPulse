"""
Real-time performance monitoring with ML-based anomaly detection.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


class PerformanceMonitor:
    """
    Real-time performance monitoring system with ML-based anomaly detection
    and performance degradation prediction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance monitor."""
        self.config = config or {}
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self._performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._anomaly_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._prediction_window = self.config.get("prediction_window", 20)
        self._alert_thresholds = {
            'critical': 0.8,
            'warning': 0.6,
            'info': 0.4
        }
        
    async def monitor_performance(
        self,
        agent_id: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Monitor agent performance and detect anomalies.
        
        Args:
            agent_id: Identifier of the agent to monitor
            metrics: Current performance metrics
            timestamp: Timestamp of the metrics (default: now)
            
        Returns:
            Dictionary containing monitoring results
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Store performance data
            self._performance_history[agent_id].append({
                'timestamp': timestamp,
                'metrics': metrics.copy()
            })
            
            # Prepare data for anomaly detection
            X = await self._prepare_monitoring_data(agent_id)
            if len(X) < self._prediction_window:
                return {'status': 'insufficient_data'}
                
            # Detect anomalies
            X_scaled = self.scaler.fit_transform(X)
            anomaly_scores = self.anomaly_detector.fit_predict(X_scaled)
            
            # Get latest anomaly score
            latest_score = anomaly_scores[-1]
            
            # Predict performance trends
            trend_prediction = await self._predict_performance_trend(agent_id)
            
            # Determine alert level
            alert_level = await self._calculate_alert_level(
                latest_score,
                trend_prediction
            )
            
            # Store anomaly if detected
            if alert_level != 'normal':
                self._anomaly_history[agent_id].append({
                    'timestamp': timestamp,
                    'metrics': metrics.copy(),
                    'anomaly_score': latest_score,
                    'alert_level': alert_level,
                    'trend_prediction': trend_prediction
                })
                
            result = {
                'status': 'ok',
                'alert_level': alert_level,
                'anomaly_score': float(latest_score),
                'trend_prediction': trend_prediction,
                'metrics_analyzed': list(metrics.keys()),
                'timestamp': timestamp
            }
            
            logger.debug(f"Performance monitoring result for {agent_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error monitoring performance for {agent_id}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
            
    async def get_performance_analytics(
        self,
        agent_id: str,
        window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get detailed performance analytics for an agent.
        
        Args:
            agent_id: Agent identifier
            window: Time window to analyze (default: all history)
            
        Returns:
            Dictionary containing performance analytics
        """
        try:
            history = self._performance_history[agent_id]
            if not history:
                return {}
                
            # Filter by time window if specified
            if window:
                cutoff = datetime.now() - window
                history = [
                    h for h in history
                    if h['timestamp'] >= cutoff
                ]
                
            if not history:
                return {}
                
            # Convert to DataFrame for analysis
            df = pd.DataFrame([
                {
                    'timestamp': h['timestamp'],
                    **h['metrics']
                }
                for h in history
            ])
            
            # Calculate trends
            trends = {}
            for column in df.columns:
                if column != 'timestamp':
                    trends[column] = await self._calculate_metric_trend(df[column])
                    
            # Calculate stability scores
            stability = {}
            for column in df.columns:
                if column != 'timestamp':
                    stability[column] = 1 - df[column].std() / df[column].mean() \
                        if df[column].mean() != 0 else 0
                        
            # Get anomaly statistics
            anomalies = self._anomaly_history[agent_id]
            if window:
                anomalies = [
                    a for a in anomalies
                    if a['timestamp'] >= cutoff
                ]
                
            return {
                'trends': trends,
                'stability': stability,
                'anomaly_count': len(anomalies),
                'alert_levels': {
                    level: len([a for a in anomalies if a['alert_level'] == level])
                    for level in ['critical', 'warning', 'info']
                },
                'latest_metrics': history[-1]['metrics'] if history else {},
                'metric_correlations': df.corr().to_dict() if len(df.columns) > 1 else {},
                'timespan': {
                    'start': history[0]['timestamp'],
                    'end': history[-1]['timestamp']
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics for {agent_id}: {str(e)}")
            return {}
            
    async def _prepare_monitoring_data(self, agent_id: str) -> np.ndarray:
        """Prepare data for anomaly detection."""
        history = self._performance_history[agent_id]
        if not history:
            return np.array([])
            
        # Extract features from all metrics
        features = []
        for record in history:
            metrics = record['metrics']
            features.append(list(metrics.values()))
            
        return np.array(features)
        
    async def _predict_performance_trend(self, agent_id: str) -> Dict[str, float]:
        """Predict performance trends for each metric."""
        history = self._performance_history[agent_id]
        if len(history) < self._prediction_window:
            return {}
            
        trends = {}
        recent_history = history[-self._prediction_window:]
        
        # Get all metric names
        metric_names = set()
        for record in recent_history:
            metric_names.update(record['metrics'].keys())
            
        # Calculate trend for each metric
        for metric in metric_names:
            values = [
                record['metrics'].get(metric, 0)
                for record in recent_history
            ]
            trend = await self._calculate_metric_trend(pd.Series(values))
            trends[metric] = trend
            
        return trends
        
    async def _calculate_metric_trend(self, series: pd.Series) -> float:
        """Calculate trend score for a metric."""
        try:
            # Use linear regression to calculate trend
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, deg=1)
            slope = coeffs[0]
            
            # Normalize trend score to [-1, 1]
            trend_score = np.tanh(slope * 10)  # Scale slope for better normalization
            return float(trend_score)
            
        except Exception:
            return 0.0
            
    async def _calculate_alert_level(
        self,
        anomaly_score: float,
        trend_prediction: Dict[str, float]
    ) -> str:
        """Calculate alert level based on anomaly score and trends."""
        # Convert isolation forest score to anomaly probability
        anomaly_prob = 1 - (anomaly_score + 1) / 2
        
        # Consider trend direction in alert level
        trend_factor = np.mean(list(trend_prediction.values())) \
            if trend_prediction else 0
            
        # Adjust anomaly probability based on trend
        adjusted_prob = anomaly_prob * (1 + abs(trend_factor))
        
        # Determine alert level
        if adjusted_prob >= self._alert_thresholds['critical']:
            return 'critical'
        elif adjusted_prob >= self._alert_thresholds['warning']:
            return 'warning'
        elif adjusted_prob >= self._alert_thresholds['info']:
            return 'info'
        return 'normal'