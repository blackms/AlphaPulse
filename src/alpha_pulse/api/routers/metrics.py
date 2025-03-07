"""Metrics router."""
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import MetricsDataAccessor
from ..cache import get_cache

router = APIRouter()
metrics_accessor = MetricsDataAccessor()


@router.get("/{metric_type}", response_model=List[Dict])
async def get_metrics(
    metric_type: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    aggregation: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get metrics data.
    
    Args:
        metric_type: Type of metric to retrieve
        start_time: Start time for query
        end_time: End time for query
        aggregation: Aggregation method (e.g., "mean", "sum")
        
    Returns:
        List of metric data points
    """
    # Check permissions
    if not has_permission(current_user, "view_metrics"):
        raise HTTPException(status_code=403, detail="Not authorized to view metrics")
    
    # Use cache for common queries
    cache = await get_cache()
    cache_key = f"metrics:{metric_type}:{start_time}:{end_time}:{aggregation}"
    
    # Try to get from cache
    cached_data = await cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Get from data accessor
    data = await metrics_accessor.get_metrics(
        metric_type=metric_type,
        start_time=start_time,
        end_time=end_time,
        aggregation=aggregation
    )
    
    # Cache for future requests
    await cache.set(cache_key, data, expiry=300)  # 5 minutes
    
    return data


@router.get("/{metric_type}/latest", response_model=Dict)
async def get_latest_metrics(
    metric_type: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get latest metrics of a specific type.
    
    Args:
        metric_type: Type of metric to retrieve
        
    Returns:
        Dictionary of latest metrics
    """
    # Check permissions
    if not has_permission(current_user, "view_metrics"):
        raise HTTPException(status_code=403, detail="Not authorized to view metrics")
    
    # Use cache with short TTL for latest metrics
    cache = await get_cache()
    cache_key = f"metrics:{metric_type}:latest"
    
    # Try to get from cache
    cached_data = await cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Get from data accessor
    data = await metrics_accessor.get_latest_metrics(metric_type)
    
    # Cache for future requests (short TTL for latest data)
    await cache.set(cache_key, data, expiry=60)  # 1 minute
    
    return data