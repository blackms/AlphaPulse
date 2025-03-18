"""
Metrics router.

This module defines the API endpoints for metrics.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import (
    require_view_metrics,
    get_metric_accessor
)
from ..data import MetricsDataAccessor

logger = logging.getLogger(__name__)

# Module-level metrics accessor for easier mocking in tests
metrics_accessor = MetricsDataAccessor()

router = APIRouter()


@router.get("/metrics/{metric_type}")
async def get_metrics(
    metric_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    aggregation: str = Query("avg", pattern="^(avg|min|max|sum|count)$"),
    _: Dict[str, Any] = Depends(require_view_metrics),
    metric_accessor: MetricsDataAccessor = Depends(lambda: metrics_accessor)
) -> List[Dict[str, Any]]:
    """
    Get metrics data.
    
    Args:
        metric_type: Type of metric to retrieve
        start_time: Start time for the query (default: 24 hours ago)
        end_time: End time for the query (default: now)
        aggregation: Aggregation function (avg, min, max, sum, count)
    
    Returns:
        List of metric data points
    """
    try:
        return await metric_accessor.get_metrics(
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time,
            aggregation=aggregation
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return []


@router.get("/metrics/{metric_type}/latest", response_model=Dict[str, Any])
async def get_latest_metrics(
    metric_type: str,
    _: Dict[str, Any] = Depends(require_view_metrics),
    metric_accessor: MetricsDataAccessor = Depends(lambda: metrics_accessor)
) -> Dict[str, Any]:
    """
    Get latest metrics of a specific type.
    
    Args:
        metric_type: Type of metric to retrieve
        
    Returns:
        Latest metric data point
    """
    try:
        result = await metric_accessor.get_latest_metrics(metric_type)
        return result if result else {}
    except Exception as e:
        logger.error(f"Error getting latest metrics: {e}")
        return {}