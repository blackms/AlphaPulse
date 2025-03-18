"""
Enhanced metrics collection and storage.
"""
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timezone, timedelta
import asyncio
import logging
import time
import functools
import inspect
import platform
import psutil
import numpy as np
from collections import defaultdict, deque

from ..portfolio.data_models import PortfolioData, Position
from .storage.interfaces import TimeSeriesStorage, MetricsStorageFactory
from .config import MonitoringConfig, load_config
from .metrics_calculations import (
    calculate_performance_metrics,
    calculate_risk_metrics,
    calculate_trade_metrics
)
from .integrations.alerting import (
    initialize_alerting,
    process_metrics_for_alerts
)

# Global metrics storage
API_LATENCY = defaultdict(list)
API_ERRORS = defaultdict(int)
API_CALLS = defaultdict(int)


def track_latency(api_name: str):
    """
    Decorator to track API call latencies.
    
    Args:
        api_name: Name of the API being called
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                API_CALLS[api_name] += 1
                return result
            except Exception as e:
                API_ERRORS[api_name] += 1
                raise
            finally:
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                API_LATENCY[api_name].append(latency)
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                API_CALLS[api_name] += 1
                return result
            except Exception as e:
                API_ERRORS[api_name] += 1
                raise
            finally:
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                API_LATENCY[api_name].append(latency)
        
        # Choose the appropriate wrapper based on whether the function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


class EnhancedMetricsCollector:
    """
    Enhanced metrics collector with time series storage.
    """
    
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        storage: Optional[TimeSeriesStorage] = None,
        risk_free_rate: float = 0.0,
        enable_alerting: bool = True,
        alerting_config_path: Optional[str] = None
    ):
        """
        Initialize the enhanced metrics collector.
        
        Args:
            config: Monitoring configuration
            storage: Optional time series storage (if not provided, created from config)
            risk_free_rate: Risk-free rate for calculations
            enable_alerting: Whether to enable the alerting system
            alerting_config_path: Optional path to alerting configuration file
        """
        self.config = config or load_config()
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger("metrics_collector")
        
        # Initialize storage
        self.storage = storage
        if not self.storage:
            storage_config = self.config.storage.get_storage_config()
            self.storage = MetricsStorageFactory.create_storage(
                self.config.storage.type, storage_config
            )
        
        # Initialize metric history for calculations
        self._metrics_history = {}
        self._agent_signals = defaultdict(list)
        self._trade_history = []
        
        # For real-time monitoring
        self._collection_task = None
        self._running = False
        self._last_collection_time = None
        
        # System metrics
        self._process = psutil.Process()
        
        # Initialize API metrics history
        self._api_metrics_history = defaultdict(lambda: deque(maxlen=100))
        
        # Alerting system
        self.enable_alerting = enable_alerting
        self.alerting_config_path = alerting_config_path
        self.alerting_initialized = False
    
    async def start(self):
        """
        Start the metrics collector.
        """
        if self._running:
            return
            
        # Connect to storage
        await self.storage.connect()
        
        # Initialize alerting system if enabled
        if self.enable_alerting:
            try:
                self.alerting_initialized = await initialize_alerting(self.alerting_config_path)
                if self.alerting_initialized:
                    self.logger.info("Alerting system initialized")
                else:
                    self.logger.warning("Failed to initialize alerting system")
            except Exception as e:
                self.logger.error(f"Error initializing alerting system: {str(e)}")
        
        # Start collection task if real-time is enabled
        if self.config.enable_realtime:
            self._running = True
            self._collection_task = asyncio.create_task(self._collection_loop())
            self.logger.info("Started metrics collection task")
    
    async def stop(self):
        """
        Stop the metrics collector.
        """
        if not self._running:
            return
            
        # Stop collection task
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
        
        # Shut down alerting system if initialized
        if self.enable_alerting and self.alerting_initialized:
            try:
                from .integrations.alerting import shutdown_alerting
                await shutdown_alerting()
                self.logger.info("Alerting system shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down alerting system: {str(e)}")
        
        # Disconnect from storage
        await self.storage.disconnect()
        self.logger.info("Stopped metrics collection")
    
    async def collect_and_store(
        self, 
        portfolio_data: Optional[PortfolioData] = None,
        trade_data: Optional[Dict[str, Any]] = None,
        agent_data: Optional[Dict[str, Any]] = None,
        system_data: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect and store metrics.
        
        Args:
            portfolio_data: Optional portfolio data
            trade_data: Optional trade execution data
            agent_data: Optional agent performance data
            system_data: Whether to collect system metrics
            
        Returns:
            Dictionary of collected metrics by type
        """
        now = datetime.now(timezone.utc)
        metrics = {}
        
        # Collect performance metrics if portfolio data is provided
        if portfolio_data:
            performance_metrics = self._calculate_performance_metrics(portfolio_data)
            if performance_metrics:
                metrics["performance"] = performance_metrics
                await self.storage.store_metrics("performance", now, performance_metrics)
        
        # Collect risk metrics if portfolio data is provided
        if portfolio_data:
            risk_metrics = self._calculate_risk_metrics(portfolio_data)
            if risk_metrics:
                metrics["risk"] = risk_metrics
                await self.storage.store_metrics("risk", now, risk_metrics)
        
        # Collect trade metrics if trade data is provided
        if trade_data:
            self._trade_history.append(trade_data)
            trade_metrics = self._calculate_trade_metrics(trade_data)
            if trade_metrics:
                metrics["trade"] = trade_metrics
                await self.storage.store_metrics("trade", now, trade_metrics)
        
        # Collect agent metrics if agent data is provided
        if agent_data:
            for agent_name, signal_data in agent_data.items():
                self._agent_signals[agent_name].append(signal_data)
            
            agent_metrics = self._calculate_agent_metrics(agent_data)
            if agent_metrics:
                metrics["agent"] = agent_metrics
                await self.storage.store_metrics("agent", now, agent_metrics)
        
        # Collect API metrics
        if self.config.collect_api_latency:
            api_metrics = self._calculate_api_metrics()
            if api_metrics:
                metrics["api"] = api_metrics
                await self.storage.store_metrics("api", now, api_metrics)
        
        # Collect system metrics
        if system_data:
            system_metrics = self._calculate_system_metrics()
            if system_metrics:
                metrics["system"] = system_metrics
                await self.storage.store_metrics("system", now, system_metrics)
        
        self._last_collection_time = now
        
        # Process metrics for alerts if alerting is enabled and initialized
        if self.enable_alerting and self.alerting_initialized:
            try:
                # Combine all metrics into a single dictionary for alerting
                all_metrics = {}
                for metric_type, metric_data in metrics.items():
                    for key, value in metric_data.items():
                        all_metrics[f"{key}"] = value
                
                # Process metrics for alerts
                alerts = await process_metrics_for_alerts(all_metrics)
                
                if alerts:
                    self.logger.info(f"Generated {len(alerts)} alerts from metrics")
                    
                    # Log alert details at debug level
                    for alert in alerts:
                        self.logger.debug(
                            f"Alert: {alert.severity.value.upper()} - {alert.message} "
                            f"(Metric: {alert.metric_name}={alert.metric_value})"
                        )
            except Exception as e:
                self.logger.error(f"Error processing metrics for alerts: {str(e)}")
        
        return metrics
    
    async def get_metrics_history(
        self, 
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics from storage.
        
        Args:
            metric_type: Type of metrics to retrieve
            start_time: Optional start time (default: 24 hours ago)
            end_time: Optional end time (default: now)
            aggregation: Optional aggregation interval
            
        Returns:
            List of metric data points
        """
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # Query storage
        return await self.storage.query_metrics(
            metric_type, start_time, end_time, aggregation
        )
    
    async def get_latest_metrics(
        self, metric_type: str, limit: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get the latest metrics of a specific type.
        
        Args:
            metric_type: Type of metrics to retrieve
            limit: Maximum number of data points to return
            
        Returns:
            List of the latest metric data points
        """
        return await self.storage.get_latest_metrics(metric_type, limit)
    
    def _calculate_performance_metrics(
        self, portfolio_data: PortfolioData
    ) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            portfolio_data: Current portfolio state
            
        Returns:
            Dictionary of performance metrics
        """
        return calculate_performance_metrics(portfolio_data, self.risk_free_rate)
    
    def _calculate_risk_metrics(
        self, portfolio_data: PortfolioData
    ) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            portfolio_data: Current portfolio state
            
        Returns:
            Dictionary of risk metrics
        """
        return calculate_risk_metrics(portfolio_data)
    
    def _calculate_trade_metrics(
        self, trade_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate trade execution metrics.
        
        Args:
            trade_data: Trade execution data
            
        Returns:
            Dictionary of trade metrics
        """
        return calculate_trade_metrics(trade_data)
    
    def _calculate_agent_metrics(
        self, agent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate agent performance metrics.
        
        Args:
            agent_data: Agent performance data
            
        Returns:
            Dictionary of agent metrics
        """
        metrics = {}
        
        # Process each agent's signals
        for agent_name, signal_data in agent_data.items():
            # Get historical signals for this agent
            agent_history = self._agent_signals[agent_name]
            if len(agent_history) < 2:
                continue
            
            # Calculate signal quality if actual outcome is available
            if 'actual_outcome' in signal_data and 'prediction' in signal_data:
                actual = signal_data['actual_outcome']
                prediction = signal_data['prediction']
                
                # Simple accuracy for directional predictions
                if isinstance(actual, bool) and isinstance(prediction, bool):
                    correct = actual == prediction
                    
                    # Update accuracy
                    correct_count = sum(1 for s in agent_history[-10:] 
                                      if s.get('actual_outcome') == s.get('prediction'))
                    accuracy = correct_count / min(10, len(agent_history))
                    
                    # Store metrics with both formats for compatibility
                    metrics[f'{agent_name}_accuracy'] = accuracy
                    metrics[f'{agent_name}_correct'] = correct
                
                # For numeric predictions, calculate error metrics
                elif isinstance(actual, (int, float)) and isinstance(prediction, (int, float)):
                    error = abs(prediction - actual)
                    pct_error = error / abs(actual) if actual != 0 else float('inf')
                    
                    # Update error metrics
                    recent_errors = [abs(s.get('prediction', 0) - s.get('actual_outcome', 0)) 
                                    for s in agent_history[-10:] 
                                    if 'actual_outcome' in s and 'prediction' in s]
                    
                    if recent_errors:
                        avg_error = sum(recent_errors) / len(recent_errors)
                        metrics[f'{agent_name}_avg_error'] = avg_error
                    
                    metrics[f'{agent_name}_error'] = error
                    metrics[f'{agent_name}_pct_error'] = pct_error
            
            # Include confidence if available
            if 'confidence' in signal_data:
                # Store both with prefix and without for test compatibility
                metrics[f'{agent_name}_confidence'] = signal_data['confidence']
                # Add direct key for test compatibility
                metrics[agent_name + '_confidence'] = signal_data['confidence']
            
            # Include signal strength if available
            if 'strength' in signal_data:
                metrics[f'{agent_name}_strength'] = signal_data['strength']
        
        # Calculate overall agent metrics
        if agent_data:
            # Average confidence across all agents
            confidences = [data.get('confidence', 0) for data in agent_data.values() 
                          if 'confidence' in data]
            if confidences:
                metrics['avg_confidence'] = sum(confidences) / len(confidences)
            
            # Signal agreement (how many agents agree on direction)
            directions = [data.get('direction', 0) for data in agent_data.values() 
                         if 'direction' in data]
            if directions:
                positive = sum(1 for d in directions if d > 0)
                negative = sum(1 for d in directions if d < 0)
                neutral = len(directions) - positive - negative
                
                metrics['signal_agreement'] = max(positive, negative, neutral) / len(directions)
                metrics['signal_direction'] = 1 if positive > negative else (-1 if negative > positive else 0)
        
        return metrics
    
    def _calculate_api_metrics(self) -> Dict[str, Any]:
        """
        Calculate API performance metrics.
        
        Returns:
            Dictionary of API metrics
        """
        metrics = {}
        
        # Process each API's latency data
        for api_name, latencies in API_LATENCY.items():
            if not latencies:
                continue
            
            # Calculate latency statistics
            metrics[f'{api_name}_mean_latency'] = np.mean(latencies)
            metrics[f'{api_name}_median_latency'] = np.median(latencies)
            metrics[f'{api_name}_p95_latency'] = np.percentile(latencies, 95)
            metrics[f'{api_name}_p99_latency'] = np.percentile(latencies, 99)
            metrics[f'{api_name}_max_latency'] = max(latencies)
            metrics[f'{api_name}_min_latency'] = min(latencies)
            
            # Calculate error rate
            calls = API_CALLS.get(api_name, 0)
            errors = API_ERRORS.get(api_name, 0)
            if calls > 0:
                metrics[f'{api_name}_error_rate'] = errors / calls
            
            # Store history for trend analysis
            self._api_metrics_history[api_name].append({
                'timestamp': datetime.now(timezone.utc),
                'mean_latency': metrics[f'{api_name}_mean_latency'],
                'error_rate': metrics.get(f'{api_name}_error_rate', 0)
            })
            
            # Calculate latency trend (increasing or decreasing)
            history = self._api_metrics_history[api_name]
            if len(history) >= 5:
                recent_means = [h['mean_latency'] for h in list(history)[-5:]]
                if len(recent_means) >= 2:
                    trend = (recent_means[-1] - recent_means[0]) / recent_means[0] if recent_means[0] > 0 else 0
                    metrics[f'{api_name}_latency_trend'] = trend
            
            # Reset latency data to avoid memory growth
            API_LATENCY[api_name] = latencies[-100:] if len(latencies) > 100 else latencies
        
        return metrics
    
    def _calculate_system_metrics(self) -> Dict[str, Any]:
        """
        Calculate system performance metrics.
        
        Returns:
            Dictionary of system metrics
        """
        metrics = {}
        
        # CPU usage
        metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics['memory_total'] = memory.total
        metrics['memory_available'] = memory.available
        metrics['memory_used'] = memory.used
        metrics['memory_usage_percent'] = memory.percent
        
        # Process-specific metrics
        try:
            process = self._process
            with process.oneshot():
                metrics['process_cpu_percent'] = process.cpu_percent(interval=None)
                metrics['process_memory_percent'] = process.memory_percent()
                metrics['process_threads'] = process.num_threads()
                
                # IO counters
                try:
                    io = process.io_counters()
                    metrics['process_read_bytes'] = io.read_bytes
                    metrics['process_write_bytes'] = io.write_bytes
                except (psutil.AccessDenied, AttributeError):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Disk usage
        try:
            disk = psutil.disk_usage('/')
            metrics['disk_total'] = disk.total
            metrics['disk_used'] = disk.used
            metrics['disk_free'] = disk.free
            metrics['disk_usage_percent'] = disk.percent
        except (FileNotFoundError, PermissionError):
            pass
        
        # Network IO
        try:
            net_io = psutil.net_io_counters()
            metrics['net_bytes_sent'] = net_io.bytes_sent
            metrics['net_bytes_recv'] = net_io.bytes_recv
            metrics['net_packets_sent'] = net_io.packets_sent
            metrics['net_packets_recv'] = net_io.packets_recv
        except (psutil.AccessDenied, AttributeError):
            pass
        
        # System information
        metrics['system_platform'] = platform.system()
        metrics['python_version'] = platform.python_version()
        
        return metrics
    
    async def _collection_loop(self):
        """
        Background task for periodic metrics collection.
        """
        while self._running:
            try:
                # Collect system metrics
                await self.collect_and_store(system_data=True)
                
                # Sleep until next collection
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Short sleep on error
