"""
Data Quality Pipeline orchestrator for comprehensive market data validation.

Provides:
- Real-time data quality monitoring and validation
- Automated quality checks and anomaly detection
- Quality metrics collection and reporting
- Alert generation and notification
- Quality dashboard integration
- Performance monitoring and optimization
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
from loguru import logger

from alpha_pulse.models.market_data import MarketDataPoint, TimeSeriesData
from alpha_pulse.data.quality.data_validator import (
    DataQualityValidator, ValidationResult, QualityScore
)
from alpha_pulse.data.quality.anomaly_detector import (
    AnomalyDetector, AnomalyResult, AnomalyMethod, AnomalyDetectorConfig
)
from alpha_pulse.data.quality.quality_metrics import (
    QualityMetricsService, QualityMetric, QualityAlert, QualityReport
)
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class PipelineMode(Enum):
    """Data quality pipeline modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HYBRID = "hybrid"


class ProcessingStatus(Enum):
    """Processing status for data points."""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    QUARANTINED = "quarantined"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for data quality pipeline."""
    mode: PipelineMode = PipelineMode.REAL_TIME
    
    # Validation settings
    enable_validation: bool = True
    validation_timeout_ms: int = 5000
    
    # Anomaly detection settings
    enable_anomaly_detection: bool = True
    anomaly_methods: List[AnomalyMethod] = field(default_factory=lambda: [
        AnomalyMethod.Z_SCORE,
        AnomalyMethod.IQR,
        AnomalyMethod.ISOLATION_FOREST
    ])
    
    # Metrics collection settings
    enable_metrics_collection: bool = True
    metrics_interval_seconds: int = 300  # 5 minutes
    
    # Performance settings
    max_concurrent_validations: int = 10
    batch_size: int = 100
    processing_queue_size: int = 1000
    
    # Storage settings
    historical_data_retention_days: int = 30
    quality_metrics_retention_days: int = 90
    
    # Alert settings
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 15


@dataclass
class ProcessingResult:
    """Result of data quality processing."""
    data_point: MarketDataPoint
    validation_result: Optional[ValidationResult]
    anomaly_results: List[AnomalyResult]
    quality_metrics: List[QualityMetric]
    processing_time_ms: float
    status: ProcessingStatus
    errors: List[str] = field(default_factory=list)


class DataQualityPipeline:
    """Orchestrator for comprehensive data quality processing."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Core components
        self.validator = DataQualityValidator() if self.config.enable_validation else None
        self.anomaly_detector = AnomalyDetector() if self.config.enable_anomaly_detection else None
        self.metrics_service = QualityMetricsService() if self.config.enable_metrics_collection else None
        
        # Processing state
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.processing_queue_size)
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.processing_stats = {
            'total_processed': 0,
            'validation_count': 0,
            'anomaly_detection_count': 0,
            'quarantine_count': 0,
            'error_count': 0,
            'avg_processing_time_ms': 0.0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Callbacks and handlers
        self.data_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Rate limiting and performance
        self.processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_validations)
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Audit logging
        self.audit_logger = get_audit_logger()
    
    async def start(self) -> None:
        """Start the data quality pipeline."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        logger.info("Starting data quality pipeline")
        
        try:
            # Start background tasks
            if self.config.mode in [PipelineMode.REAL_TIME, PipelineMode.HYBRID]:
                # Real-time processing task
                processing_task = asyncio.create_task(self._processing_loop())
                self.background_tasks.append(processing_task)
            
            if self.config.enable_metrics_collection:
                # Metrics collection task
                metrics_task = asyncio.create_task(self._metrics_collection_loop())
                self.background_tasks.append(metrics_task)
            
            # Performance monitoring task
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.background_tasks.append(monitoring_task)
            
            # Cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.background_tasks.append(cleanup_task)
            
            self.is_running = True
            logger.info("Data quality pipeline started successfully")
            
            # Log pipeline start
            self.audit_logger.log(
                event_type=AuditEventType.SYSTEM_START,
                event_data={
                    "component": "data_quality_pipeline",
                    "mode": self.config.mode.value,
                    "config": {
                        "validation_enabled": self.config.enable_validation,
                        "anomaly_detection_enabled": self.config.enable_anomaly_detection,
                        "metrics_collection_enabled": self.config.enable_metrics_collection
                    }
                },
                severity=AuditSeverity.INFO
            )
            
        except Exception as e:
            logger.error(f"Failed to start data quality pipeline: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the data quality pipeline."""
        logger.info("Stopping data quality pipeline")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        
        logger.info("Data quality pipeline stopped")
    
    async def process_data_point(self, data_point: MarketDataPoint) -> ProcessingResult:
        """Process a single data point through the quality pipeline."""
        start_time = time.time()
        
        try:
            # Add to processing queue if in real-time mode
            if self.config.mode == PipelineMode.REAL_TIME:
                await self.processing_queue.put(data_point)
                # Return immediately in real-time mode
                return ProcessingResult(
                    data_point=data_point,
                    validation_result=None,
                    anomaly_results=[],
                    quality_metrics=[],
                    processing_time_ms=0.0,
                    status=ProcessingStatus.PENDING
                )
            else:
                # Process immediately in batch mode
                return await self._process_data_point_sync(data_point)
                
        except Exception as e:
            logger.error(f"Error processing data point for {data_point.symbol}: {e}")
            return ProcessingResult(
                data_point=data_point,
                validation_result=None,
                anomaly_results=[],
                quality_metrics=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )
    
    async def process_time_series(self, time_series: TimeSeriesData) -> List[ProcessingResult]:
        """Process a time series through the quality pipeline."""
        if self.config.mode == PipelineMode.REAL_TIME:
            logger.warning("Time series processing not optimal in real-time mode")
        
        results = []
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        for i in range(0, len(time_series.data_points), batch_size):
            batch = time_series.data_points[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [self._process_data_point_sync(dp) for dp in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                else:
                    results.append(result)
        
        return results
    
    async def _process_data_point_sync(self, data_point: MarketDataPoint) -> ProcessingResult:
        """Synchronously process a data point through all quality checks."""
        async with self.processing_semaphore:
            start_time = time.time()
            validation_result = None
            anomaly_results = []
            quality_metrics = []
            errors = []
            status = ProcessingStatus.PROCESSING
            
            try:
                # Get historical context
                historical_context = list(self.historical_data[data_point.symbol])
                
                # Step 1: Data validation
                if self.validator:
                    try:
                        validation_result = await asyncio.wait_for(
                            self.validator.validate_data_point(data_point, historical_context),
                            timeout=self.config.validation_timeout_ms / 1000
                        )
                        self.processing_stats['validation_count'] += 1
                        
                        # Check if data should be quarantined
                        if validation_result.is_quarantined:
                            status = ProcessingStatus.QUARANTINED
                            self.processing_stats['quarantine_count'] += 1
                        
                    except asyncio.TimeoutError:
                        errors.append("Validation timeout")
                        logger.warning(f"Validation timeout for {data_point.symbol}")
                    except Exception as e:
                        errors.append(f"Validation error: {str(e)}")
                        logger.error(f"Validation error for {data_point.symbol}: {e}")
                
                # Step 2: Anomaly detection
                if self.anomaly_detector and status != ProcessingStatus.QUARANTINED:
                    try:
                        anomaly_results = await self.anomaly_detector.detect_anomalies(
                            data_point, historical_context, self.config.anomaly_methods
                        )
                        self.processing_stats['anomaly_detection_count'] += 1
                        
                        # Check for critical anomalies
                        critical_anomalies = [r for r in anomaly_results if r.is_anomaly and r.severity.value in ['high', 'critical']]
                        if critical_anomalies:
                            await self._handle_critical_anomalies(data_point, critical_anomalies)
                        
                    except Exception as e:
                        errors.append(f"Anomaly detection error: {str(e)}")
                        logger.error(f"Anomaly detection error for {data_point.symbol}: {e}")
                
                # Step 3: Update historical data
                self.historical_data[data_point.symbol].append(data_point)
                
                # Step 4: Calculate quality metrics (if enabled and sufficient data)
                if self.metrics_service and len(historical_context) > 10:
                    try:
                        validation_results = [validation_result] if validation_result else []
                        quality_metrics = await self.metrics_service.calculate_quality_metrics(
                            data_point.symbol, validation_results, anomaly_results
                        )
                    except Exception as e:
                        errors.append(f"Metrics calculation error: {str(e)}")
                        logger.error(f"Metrics calculation error for {data_point.symbol}: {e}")
                
                # Update status
                if status == ProcessingStatus.PROCESSING:
                    status = ProcessingStatus.VALIDATED
                
                # Update statistics
                processing_time_ms = (time.time() - start_time) * 1000
                self.processing_stats['total_processed'] += 1
                self._update_avg_processing_time(processing_time_ms)
                
                # Trigger callbacks
                result = ProcessingResult(
                    data_point=data_point,
                    validation_result=validation_result,
                    anomaly_results=anomaly_results,
                    quality_metrics=quality_metrics,
                    processing_time_ms=processing_time_ms,
                    status=status,
                    errors=errors
                )
                
                await self._trigger_callbacks(result)
                
                return result
                
            except Exception as e:
                self.processing_stats['error_count'] += 1
                logger.error(f"Unexpected error processing {data_point.symbol}: {e}")
                
                return ProcessingResult(
                    data_point=data_point,
                    validation_result=validation_result,
                    anomaly_results=anomaly_results,
                    quality_metrics=quality_metrics,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    status=ProcessingStatus.FAILED,
                    errors=errors + [str(e)]
                )
    
    async def _processing_loop(self) -> None:
        """Background processing loop for real-time mode."""
        logger.info("Starting real-time processing loop")
        
        while self.is_running:
            try:
                # Get data point from queue with timeout
                data_point = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=1.0
                )
                
                # Process the data point
                await self._process_data_point_sync(data_point)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # No data to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_collection_loop(self) -> None:
        """Background loop for metrics collection and reporting."""
        logger.info("Starting metrics collection loop")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config.metrics_interval_seconds)
                
                if not self.metrics_service:
                    continue
                
                # Collect metrics for all symbols with recent activity
                symbols_with_data = list(self.historical_data.keys())
                
                for symbol in symbols_with_data:
                    try:
                        # Generate quality report for the last interval
                        end_time = datetime.utcnow()
                        start_time = end_time - timedelta(seconds=self.config.metrics_interval_seconds)
                        
                        report = await self.metrics_service.generate_quality_report(
                            symbol, start_time, end_time
                        )
                        
                        # Process any alerts
                        if report.alerts:
                            await self._handle_quality_alerts(report.alerts)
                        
                    except Exception as e:
                        logger.error(f"Error collecting metrics for {symbol}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self) -> None:
        """Background loop for performance monitoring."""
        logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
                # Log performance statistics
                self.audit_logger.log(
                    event_type=AuditEventType.SYSTEM_PERFORMANCE,
                    event_data={
                        "component": "data_quality_pipeline",
                        "stats": self.processing_stats.copy(),
                        "queue_size": self.processing_queue.qsize(),
                        "active_symbols": len(self.historical_data),
                        "memory_usage": {
                            "historical_data_points": sum(len(deque_) for deque_ in self.historical_data.values())
                        }
                    },
                    severity=AuditSeverity.INFO
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Background loop for data cleanup."""
        logger.info("Starting cleanup loop")
        
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Clean up old historical data
                cutoff_time = datetime.utcnow() - timedelta(days=self.config.historical_data_retention_days)
                
                for symbol in list(self.historical_data.keys()):
                    # Remove old data points
                    deque_data = self.historical_data[symbol]
                    while deque_data and deque_data[0].timestamp < cutoff_time:
                        deque_data.popleft()
                    
                    # Remove empty deques
                    if not deque_data:
                        del self.historical_data[symbol]
                
                logger.info(f"Cleanup completed. Active symbols: {len(self.historical_data)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _handle_critical_anomalies(self, data_point: MarketDataPoint, anomalies: List[AnomalyResult]) -> None:
        """Handle critical anomalies with immediate alerting."""
        for anomaly in anomalies:
            alert_key = f"{data_point.symbol}_{anomaly.method.value}"
            
            # Check alert cooldown
            if alert_key in self.last_alert_times:
                time_since_last = datetime.utcnow() - self.last_alert_times[alert_key]
                if time_since_last.total_seconds() < self.config.alert_cooldown_minutes * 60:
                    continue
            
            # Log critical anomaly
            self.audit_logger.log(
                event_type=AuditEventType.DATA_ANOMALY,
                event_data={
                    "symbol": data_point.symbol,
                    "anomaly_type": anomaly.method.value,
                    "severity": anomaly.severity.value,
                    "score": anomaly.anomaly_score,
                    "details": anomaly.details
                },
                severity=AuditSeverity.ERROR
            )
            
            self.last_alert_times[alert_key] = datetime.utcnow()
    
    async def _handle_quality_alerts(self, alerts: List[QualityAlert]) -> None:
        """Handle quality alerts."""
        for alert in alerts:
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    async def _trigger_callbacks(self, result: ProcessingResult) -> None:
        """Trigger data processing callbacks."""
        for callback in self.data_callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def _update_avg_processing_time(self, processing_time_ms: float) -> None:
        """Update average processing time statistic."""
        total_processed = self.processing_stats['total_processed']
        current_avg = self.processing_stats['avg_processing_time_ms']
        
        # Update running average
        new_avg = ((current_avg * (total_processed - 1)) + processing_time_ms) / total_processed
        self.processing_stats['avg_processing_time_ms'] = new_avg
    
    def add_data_callback(self, callback: Callable[[ProcessingResult], None]) -> None:
        """Add callback for processed data."""
        self.data_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]) -> None:
        """Add callback for quality alerts."""
        self.alert_callbacks.append(callback)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "is_running": self.is_running,
            "mode": self.config.mode.value,
            "processing_stats": self.processing_stats.copy(),
            "queue_size": self.processing_queue.qsize() if hasattr(self.processing_queue, 'qsize') else 0,
            "active_symbols": len(self.historical_data),
            "historical_data_points": sum(len(deque_) for deque_ in self.historical_data.values()),
            "background_tasks": len(self.background_tasks),
            "components": {
                "validator_enabled": self.validator is not None,
                "anomaly_detector_enabled": self.anomaly_detector is not None,
                "metrics_service_enabled": self.metrics_service is not None
            }
        }
    
    async def generate_quality_report(self, symbol: str, hours: int = 24) -> Optional[QualityReport]:
        """Generate quality report for a symbol."""
        if not self.metrics_service:
            return None
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        return await self.metrics_service.generate_quality_report(symbol, start_time, end_time)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global pipeline instance
_data_quality_pipeline: Optional[DataQualityPipeline] = None


def get_data_quality_pipeline(config: Optional[PipelineConfig] = None) -> DataQualityPipeline:
    """Get the global data quality pipeline instance."""
    global _data_quality_pipeline
    
    if _data_quality_pipeline is None:
        _data_quality_pipeline = DataQualityPipeline(config)
    
    return _data_quality_pipeline