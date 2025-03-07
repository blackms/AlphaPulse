"""Metrics data access module."""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import random


class MetricsDataAccessor:
    """Access metrics data from storage."""
    
    def __init__(self):
        """Initialize metrics accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.metrics")
    
    async def get_metrics(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: Optional[str] = None,
        aggregation: str = 'avg'
    ) -> List[Dict[str, Any]]:
        """
        Get metrics data from storage.
        
        Args:
            metric_type: Type of metric to retrieve
            start_time: Start time for query
            end_time: End time for query
            interval: Time interval for aggregation
            aggregation: Aggregation method (e.g., "avg", "sum")
            
        Returns:
            List of metric data points
        """
        try:
            # Set default time range if not provided
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=1)
            
            # Generate mock data
            result = []
            current_time = start_time
            
            # Determine step size based on interval
            if interval == '1h':
                step = timedelta(hours=1)
            elif interval == '1d':
                step = timedelta(days=1)
            elif interval == '1w':
                step = timedelta(weeks=1)
            else:
                # Default to 1 hour
                step = timedelta(hours=1)
            
            # Generate data points
            while current_time <= end_time:
                # Generate different values based on metric type
                if metric_type == 'portfolio_value':
                    base_value = 1000000.0
                    variation = random.uniform(-50000, 50000)
                    value = base_value + variation
                elif metric_type == 'sharpe_ratio':
                    value = random.uniform(1.5, 2.5)
                elif metric_type == 'volatility':
                    value = random.uniform(0.1, 0.3)
                elif metric_type == 'drawdown':
                    value = random.uniform(0.05, 0.2)
                else:
                    value = random.uniform(0, 100)
                
                result.append({
                    'timestamp': current_time.isoformat(),
                    'value': value,
                    'labels': {
                        'metric_type': metric_type,
                        'aggregation': aggregation
                    }
                })
                
                current_time += step
            
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving metrics: {str(e)}")
            return []
    
    async def get_latest_metric(self, metric_type: str) -> Dict[str, Any]:
        """
        Get latest metric of a specific type.
        
        Args:
            metric_type: Type of metric to retrieve
            
        Returns:
            Dictionary with latest metric data
        """
        try:
            # Generate mock data based on metric type
            if metric_type == 'portfolio_value':
                value = random.uniform(950000, 1050000)
            elif metric_type == 'sharpe_ratio':
                value = random.uniform(1.5, 2.5)
            elif metric_type == 'volatility':
                value = random.uniform(0.1, 0.3)
            elif metric_type == 'drawdown':
                value = random.uniform(0.05, 0.2)
            else:
                value = random.uniform(0, 100)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'value': value,
                'labels': {
                    'metric_type': metric_type
                }
            }
        except Exception as e:
            self.logger.error(f"Error retrieving latest metric: {str(e)}")
            return {}