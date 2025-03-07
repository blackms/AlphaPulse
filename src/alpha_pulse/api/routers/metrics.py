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
from ..data import MetricDataAccessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics/{metric_type}")
async def get_metrics(
    metric_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    interval: Optional[str] = None,
    aggregation: str = Query("avg", regex="^(avg|min|max|sum|count)$"),
    _: Dict[str, Any] = Depends(require_view_metrics),
    metric_accessor: MetricDataAccessor = Depends(get_metric_accessor)
) -> List[Dict[str, Any]]:
    """
    Get metrics data.
    
    Args:
        metric_type: Type of metric to retrieve
        start_time: Start time for the query (default: 24 hours ago)
        end_time: End time for the query (default: now)
        interval: Time interval for aggregation (e.g., '1h', '1d')
        aggregation: Aggregation function (avg, min, max, sum, count)
    
    Returns:
        List of metric data points
    """
    return await metric_accessor.get_metrics(
        metric_type=metric_type,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
        aggregation=aggregation
    )


@router.get("/metrics/{metric_type}/latest")
async def get_latest_metric(
    metric_type: str,
    _: Dict[str, Any] = Depends(require_view_metrics),
    metric_accessor: MetricDataAccessor = Depends(get_metric_accessor)
) -> Dict[str, Any]:
    """
    Get the latest metric value.
    
    Args:
        metric_type: Type of metric to retrieve
    
    Returns:
        Latest metric data point
    """
    result = await metric_accessor.get_latest_metric(metric_type)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Metric {metric_type} not found")
    return result