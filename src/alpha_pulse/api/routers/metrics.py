"""
Metrics router.

This module defines the API endpoints for metrics.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException, Request

from ..dependencies import (
    require_view_metrics,
    get_metric_accessor
)
from ..data import MetricsDataAccessor
from ..middleware.tenant_context import get_current_tenant_id

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
    tenant_id: str = Depends(get_current_tenant_id),
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
        logger.info(f"[Tenant: {tenant_id}] Retrieving metrics: {metric_type}, aggregation={aggregation}")
        return await metric_accessor.get_metrics(
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time,
            aggregation=aggregation
        )
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting metrics {metric_type}: {e}")
        return []


@router.get("/metrics/{metric_type}/latest", response_model=Dict[str, Any])
async def get_latest_metrics(
    metric_type: str,
    tenant_id: str = Depends(get_current_tenant_id),
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
        logger.info(f"[Tenant: {tenant_id}] Retrieving latest metrics: {metric_type}")
        result = await metric_accessor.get_latest_metrics(metric_type)
        return result if result else {}
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting latest metrics {metric_type}: {e}")
        return {}


@router.get("/metrics/cache", response_model=Dict[str, Any])
async def get_cache_metrics(
    request: Request,
    tenant_id: str = Depends(get_current_tenant_id),
    _: Dict[str, Any] = Depends(require_view_metrics)
) -> Dict[str, Any]:
    """
    Get cache performance metrics.

    Returns:
        Cache hit rates, latency, memory usage, and other cache metrics
    """
    try:
        logger.info(f"[Tenant: {tenant_id}] Retrieving cache metrics")
        if hasattr(request.app.state, 'caching_service') and request.app.state.caching_service:
            metrics = await request.app.state.caching_service.get_metrics()
            return {
                "status": "active",
                "metrics": metrics,
                "message": "CachingService is active and optimizing performance"
            }
        else:
            logger.warning(f"[Tenant: {tenant_id}] CachingService is not initialized")
            return {
                "status": "inactive",
                "metrics": {},
                "message": "CachingService is not initialized"
            }
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting cache metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/database", response_model=Dict[str, Any])
async def get_database_metrics(
    request: Request,
    tenant_id: str = Depends(get_current_tenant_id),
    _: Dict[str, Any] = Depends(require_view_metrics)
) -> Dict[str, Any]:
    """
    Get database performance metrics.

    Returns:
        Query performance, connection pool stats, slow queries, and optimization metrics
    """
    try:
        logger.info(f"[Tenant: {tenant_id}] Retrieving database metrics")
        if hasattr(request.app.state, 'db_optimization_service') and request.app.state.db_optimization_service:
            metrics = await request.app.state.db_optimization_service.get_performance_metrics()
            return {
                "status": "active",
                "metrics": metrics,
                "message": "DatabaseOptimizationService is active and monitoring performance"
            }
        else:
            logger.warning(f"[Tenant: {tenant_id}] DatabaseOptimizationService is not initialized")
            return {
                "status": "inactive",
                "metrics": {},
                "message": "DatabaseOptimizationService is not initialized"
            }
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error getting database metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))