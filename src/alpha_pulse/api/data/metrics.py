"""Metrics data access module."""
from typing import Dict, List, Optional
from datetime import datetime
import logging

from alpha_pulse.monitoring.storage import get_storage_manager
from alpha_pulse.monitoring.metrics_calculations import calculate_derived_metrics


class MetricsDataAccessor:
    """Access metrics data from storage."""
    
    def __init__(self):
        """Initialize metrics accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.metrics")
        self.storage_manager = get_storage_manager()
    
    async def get_metrics(
        self, 
        metric_type: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: Optional[str] = None
    ) -> List[Dict]:
        """
        Get metrics data from storage.
        
        Args:
            metric_type: Type of metric to retrieve
            start_time: Start time for query
            end_time: End time for query
            aggregation: Aggregation method (e.g., "mean", "sum")
            
        Returns:
            List of metric data points
        """
        try:
            # Get raw metrics from storage
            raw_metrics = await self.storage_manager.query_metrics(
                metric_names=[metric_type] if metric_type != "all" else None,
                start_time=start_time,
                end_time=end_time,
                aggregation=aggregation
            )
            
            # Transform to API format
            result = []
            for metric in raw_metrics:
                result.append({
                    "name": metric.name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "labels": metric.labels
                })
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving metrics: {str(e)}")
            return []
    
    async def get_latest_metrics(self, metric_type: str) -> Dict:
        """
        Get latest metrics of a specific type.
        
        Args:
            metric_type: Type of metric to retrieve
            
        Returns:
            Dictionary of latest metrics
        """
        try:
            # Get latest metrics
            latest_metrics = await self.storage_manager.get_latest_metrics(
                metric_names=[metric_type] if metric_type != "all" else None
            )
            
            # Transform to API format
            result = {}
            for metric in latest_metrics:
                result[metric.name] = {
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "labels": metric.labels
                }
                
            # Add derived metrics if appropriate
            if metric_type in ["performance", "portfolio", "all"]:
                derived = calculate_derived_metrics(latest_metrics)
                for name, value in derived.items():
                    result[name] = {
                        "value": value,
                        "timestamp": datetime.now().isoformat(),
                        "labels": {"derived": "true"}
                    }
                    
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving latest metrics: {str(e)}")
            return {}