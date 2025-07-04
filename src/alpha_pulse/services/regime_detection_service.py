"""
Real-time Market Regime Detection Service.

This service provides real-time regime detection capabilities with integration
to the AlphaPulse trading system, including monitoring, alerting, and performance tracking.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from prometheus_client import Counter, Gauge, Histogram, Summary
import aioredis
import json

from ..models.market_regime_hmm import MarketRegimeHMM, MarketRegimeConfig
from ..models.regime_state import RegimeStateManager, RegimeState
from ..ml.regime.regime_classifier import RegimeInfo
from ..data_pipeline.data_fetcher import DataPipeline
from ..monitoring.alert_manager import AlertManager, Alert, AlertLevel

logger = logging.getLogger(__name__)


# Prometheus metrics
regime_classification_counter = Counter(
    'regime_classifications_total',
    'Total number of regime classifications',
    ['regime_type']
)

regime_transition_counter = Counter(
    'regime_transitions_total',
    'Total number of regime transitions',
    ['from_regime', 'to_regime']
)

current_regime_gauge = Gauge(
    'current_market_regime',
    'Current market regime (0-4)'
)

regime_confidence_gauge = Gauge(
    'regime_confidence',
    'Confidence in current regime classification'
)

regime_detection_latency = Histogram(
    'regime_detection_latency_seconds',
    'Latency of regime detection'
)

regime_stability_gauge = Gauge(
    'regime_stability_score',
    'Stability score of current regime'
)


@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection service."""
    # Model configuration
    model_config: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    
    # Update settings
    update_interval_minutes: int = 60
    min_data_points: int = 500
    lookback_days: int = 252
    
    # Alerting
    enable_alerts: bool = True
    transition_alert_threshold: float = 0.3
    confidence_alert_threshold: float = 0.5
    
    # Performance tracking
    track_performance: bool = True
    performance_window_days: int = 30
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    
    # Model persistence
    model_checkpoint_interval: int = 24  # hours
    model_checkpoint_path: str = "/tmp/regime_model_checkpoint.pkl"


class RegimeDetectionService:
    """
    Real-time market regime detection service.
    
    This service continuously monitors market conditions, detects regime changes,
    and provides trading signals based on the current market regime.
    """
    
    def __init__(self,
                 config: Optional[RegimeDetectionConfig] = None,
                 data_pipeline: Optional[DataPipeline] = None,
                 alert_manager: Optional[AlertManager] = None):
        self.config = config or RegimeDetectionConfig()
        self.data_pipeline = data_pipeline
        self.alert_manager = alert_manager
        
        # Initialize components
        self.regime_model = MarketRegimeHMM(self.config.model_config)
        self.state_manager = RegimeStateManager()
        
        # State tracking
        self.is_running = False
        self.last_update_time: Optional[datetime] = None
        self.current_regime_info: Optional[RegimeInfo] = None
        self.update_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_tracker = RegimePerformanceTracker()
        
        # Redis connection
        self.redis: Optional[aioredis.Redis] = None
        
        # Callbacks
        self.regime_change_callbacks: List[Callable] = []
        
    async def initialize(self):
        """Initialize the service and its components."""
        logger.info("Initializing regime detection service...")
        
        # Connect to Redis
        try:
            self.redis = await aioredis.create_redis_pool(self.config.redis_url)
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
        
        # Load or train model
        await self._initialize_model()
        
        logger.info("Regime detection service initialized")
    
    async def start(self):
        """Start the regime detection service."""
        if self.is_running:
            logger.warning("Service is already running")
            return
        
        self.is_running = True
        logger.info("Starting regime detection service...")
        
        # Start update loop
        self.update_task = asyncio.create_task(self._update_loop())
        
        # Initial detection
        await self._detect_regime()
        
    async def stop(self):
        """Stop the regime detection service."""
        logger.info("Stopping regime detection service...")
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Save model checkpoint
        if self.regime_model.is_fitted:
            self.regime_model.save(self.config.model_checkpoint_path)
        
        # Close Redis connection
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        
        logger.info("Regime detection service stopped")
    
    async def _update_loop(self):
        """Main update loop for continuous regime detection."""
        while self.is_running:
            try:
                # Wait for next update interval
                await asyncio.sleep(self.config.update_interval_minutes * 60)
                
                # Detect regime
                await self._detect_regime()
                
                # Save checkpoint periodically
                if (datetime.now() - self.last_update_time).total_seconds() > \
                   self.config.model_checkpoint_interval * 3600:
                    self.regime_model.save(self.config.model_checkpoint_path)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry
    
    async def _initialize_model(self):
        """Initialize or load the regime detection model."""
        try:
            # Try to load existing model
            self.regime_model = MarketRegimeHMM.load(self.config.model_checkpoint_path)
            logger.info("Loaded existing regime model")
        except Exception as e:
            logger.info(f"No existing model found, training new model: {e}")
            
            # Fetch training data
            if self.data_pipeline:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.config.lookback_days * 2)
                
                data = await self.data_pipeline.fetch_historical_data(
                    symbols=["SPY"],  # Use market index for regime detection
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d"
                )
                
                if not data.empty:
                    # Train model
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.regime_model.fit, data
                    )
                    logger.info("Trained new regime model")
                else:
                    logger.error("No data available for model training")
    
    async def _detect_regime(self):
        """Perform regime detection."""
        if not self.regime_model.is_fitted:
            logger.warning("Model not fitted, skipping regime detection")
            return
        
        start_time = datetime.now()
        
        try:
            # Fetch recent data
            data = await self._fetch_recent_data()
            if data.empty:
                logger.warning("No recent data available")
                return
            
            # Detect regime
            with regime_detection_latency.time():
                regime_info = await asyncio.get_event_loop().run_in_executor(
                    None, self.regime_model.predict_regime, data
                )
            
            # Check for regime change
            regime_changed = await self._check_regime_change(regime_info)
            
            # Update metrics
            self._update_metrics(regime_info, regime_changed)
            
            # Cache results
            await self._cache_regime_info(regime_info)
            
            # Update state manager
            self.state_manager.update_state(
                regime_info.current_regime,
                data['close'].pct_change().dropna().values,
                pd.DataFrame()  # Features would go here
            )
            
            # Track performance
            if self.config.track_performance:
                self.performance_tracker.update(regime_info, data)
            
            self.last_update_time = datetime.now()
            logger.info(
                f"Regime detection completed: {regime_info.regime_type.value} "
                f"(confidence: {regime_info.confidence:.2%})"
            )
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}", exc_info=True)
            await self._send_alert(
                "Regime Detection Error",
                f"Failed to detect market regime: {str(e)}",
                AlertLevel.ERROR
            )
    
    async def _fetch_recent_data(self) -> pd.DataFrame:
        """Fetch recent market data for regime detection."""
        if not self.data_pipeline:
            return pd.DataFrame()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        data = await self.data_pipeline.fetch_historical_data(
            symbols=["SPY"],
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        # Add additional data if available (VIX, etc.)
        additional_data = {}
        
        try:
            vix_data = await self.data_pipeline.fetch_historical_data(
                symbols=["^VIX"],
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            if not vix_data.empty:
                additional_data['vix'] = vix_data
        except Exception as e:
            logger.debug(f"Failed to fetch VIX data: {e}")
        
        return data
    
    async def _check_regime_change(self, new_regime_info: RegimeInfo) -> bool:
        """Check if regime has changed."""
        if self.current_regime_info is None:
            self.current_regime_info = new_regime_info
            return True
        
        regime_changed = (
            new_regime_info.current_regime != self.current_regime_info.current_regime
        )
        
        if regime_changed:
            # Send alerts
            if self.config.enable_alerts:
                await self._send_regime_change_alert(
                    self.current_regime_info,
                    new_regime_info
                )
            
            # Call callbacks
            for callback in self.regime_change_callbacks:
                try:
                    await callback(self.current_regime_info, new_regime_info)
                except Exception as e:
                    logger.error(f"Error in regime change callback: {e}")
            
            # Update state manager
            self.state_manager.transition_to(new_regime_info.current_regime)
        
        # Check for high transition probability
        elif new_regime_info.transition_probability > self.config.transition_alert_threshold:
            await self._send_alert(
                "High Regime Transition Probability",
                f"Transition probability: {new_regime_info.transition_probability:.1%}",
                AlertLevel.WARNING
            )
        
        self.current_regime_info = new_regime_info
        return regime_changed
    
    def _update_metrics(self, regime_info: RegimeInfo, regime_changed: bool):
        """Update Prometheus metrics."""
        # Update gauges
        current_regime_gauge.set(regime_info.current_regime)
        regime_confidence_gauge.set(regime_info.confidence)
        
        # Update counters
        regime_classification_counter.labels(
            regime_type=regime_info.regime_type.value
        ).inc()
        
        if regime_changed and self.current_regime_info:
            regime_transition_counter.labels(
                from_regime=self.current_regime_info.regime_type.value,
                to_regime=regime_info.regime_type.value
            ).inc()
        
        # Update stability
        if hasattr(regime_info, 'expected_remaining_duration'):
            stability_score = min(
                1.0,
                regime_info.expected_remaining_duration / regime_info.duration
            )
            regime_stability_gauge.set(stability_score)
    
    async def _cache_regime_info(self, regime_info: RegimeInfo):
        """Cache regime information in Redis."""
        if not self.redis:
            return
        
        try:
            cache_data = {
                'regime': regime_info.current_regime,
                'regime_type': regime_info.regime_type.value,
                'confidence': regime_info.confidence,
                'timestamp': datetime.now().isoformat(),
                'probabilities': regime_info.regime_probabilities.tolist()
            }
            
            await self.redis.setex(
                'regime:current',
                self.config.cache_ttl_seconds,
                json.dumps(cache_data)
            )
            
            # Cache historical
            await self.redis.lpush(
                'regime:history',
                json.dumps(cache_data)
            )
            await self.redis.ltrim('regime:history', 0, 1000)
            
        except Exception as e:
            logger.error(f"Failed to cache regime info: {e}")
    
    async def _send_regime_change_alert(self,
                                      old_regime: RegimeInfo,
                                      new_regime: RegimeInfo):
        """Send alert for regime change."""
        message = (
            f"Market regime changed from {old_regime.regime_type.value} "
            f"to {new_regime.regime_type.value}\n"
            f"Confidence: {new_regime.confidence:.1%}\n"
            f"Expected duration: {new_regime.expected_remaining_duration:.0f} periods"
        )
        
        await self._send_alert(
            "Market Regime Change",
            message,
            AlertLevel.HIGH
        )
    
    async def _send_alert(self, title: str, message: str, level: AlertLevel):
        """Send alert through alert manager."""
        if not self.alert_manager or not self.config.enable_alerts:
            return
        
        alert = Alert(
            title=title,
            message=message,
            level=level,
            source="regime_detection",
            timestamp=datetime.now()
        )
        
        await self.alert_manager.send_alert(alert)
    
    def register_regime_change_callback(self, callback: Callable):
        """Register callback for regime changes."""
        self.regime_change_callbacks.append(callback)
    
    async def get_current_regime(self) -> Optional[RegimeInfo]:
        """Get current regime information."""
        # Try cache first
        if self.redis:
            try:
                cached = await self.redis.get('regime:current')
                if cached:
                    data = json.loads(cached)
                    # Return cached info (simplified)
                    return self.current_regime_info
            except Exception as e:
                logger.debug(f"Failed to get cached regime: {e}")
        
        return self.current_regime_info
    
    async def get_regime_forecast(self, horizon: int = 10) -> pd.DataFrame:
        """Get regime forecast."""
        if not self.regime_model.is_fitted:
            return pd.DataFrame()
        
        return await asyncio.get_event_loop().run_in_executor(
            None, self.regime_model.get_regime_forecast, horizon
        )
    
    async def get_trading_signals(self,
                                risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """Get trading signals based on current regime."""
        if not self.current_regime_info:
            return {}
        
        return self.regime_model.get_regime_trading_signals(
            self.current_regime_info,
            risk_tolerance
        )
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get regime detection performance report."""
        return self.performance_tracker.get_report()


class RegimePerformanceTracker:
    """Track performance of regime detection."""
    
    def __init__(self):
        self.regime_history: List[Tuple[datetime, RegimeInfo]] = []
        self.prediction_accuracy: List[float] = []
        self.transition_timing: List[Dict[str, Any]] = []
        
    def update(self, regime_info: RegimeInfo, market_data: pd.DataFrame):
        """Update performance tracking."""
        self.regime_history.append((datetime.now(), regime_info))
        
        # Track prediction accuracy (simplified)
        if len(self.regime_history) > 10:
            # Look at past predictions vs actual market performance
            past_regime = self.regime_history[-10][1]
            actual_return = market_data['close'].pct_change(10).iloc[-1]
            
            # Simple accuracy check
            if past_regime.regime_type.value == "bull" and actual_return > 0:
                self.prediction_accuracy.append(1.0)
            elif past_regime.regime_type.value == "bear" and actual_return < 0:
                self.prediction_accuracy.append(1.0)
            else:
                self.prediction_accuracy.append(0.0)
    
    def get_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.prediction_accuracy:
            return {}
        
        return {
            'average_accuracy': np.mean(self.prediction_accuracy),
            'total_predictions': len(self.prediction_accuracy),
            'regime_distribution': self._get_regime_distribution(),
            'transition_frequency': self._get_transition_frequency()
        }
    
    def _get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of regimes."""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for _, info in self.regime_history:
            regime_type = info.regime_type.value
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        total = sum(regime_counts.values())
        return {k: v / total for k, v in regime_counts.items()}
    
    def _get_transition_frequency(self) -> float:
        """Calculate regime transition frequency."""
        if len(self.regime_history) < 2:
            return 0.0
        
        transitions = 0
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i][1].current_regime != \
               self.regime_history[i-1][1].current_regime:
                transitions += 1
        
        return transitions / (len(self.regime_history) - 1)