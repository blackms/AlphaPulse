"""
Alerts router.

This module defines the API endpoints for alerts.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from ..dependencies import (
    require_view_alerts,
    require_acknowledge_alerts,
    get_alert_accessor,
    get_current_user
)
from ..data import AlertDataAccessor
from ..middleware.tenant_context import get_current_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/alerts")
async def get_alerts(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    source: Optional[str] = None,
    tenant_id: str = Depends(get_current_tenant_id),
    _: Dict[str, Any] = Depends(require_view_alerts),
    alert_accessor: AlertDataAccessor = Depends(get_alert_accessor)
) -> List[Dict[str, Any]]:
    """
    Get alerts.

    Args:
        start_time: Start time for the query
        end_time: End time for the query
        severity: Filter by severity (info, warning, critical)
        acknowledged: Filter by acknowledgment status
        source: Filter by source

    Returns:
        List of alerts
    """
    # Build filters
    filters = {}
    if severity:
        filters["severity"] = severity
    if acknowledged is not None:
        filters["acknowledged"] = acknowledged
    if source:
        filters["source"] = source

    try:
        logger.info(f"[Tenant: {tenant_id}] Retrieving alerts with filters: {filters}")
        return await alert_accessor.get_alerts(
            start_time=start_time,
            end_time=end_time,
            filters=filters
        )
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error retrieving alerts: {e}")
        # Return empty list on error as specified in the test
        return []



@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int = Path(..., description="The ID of the alert to acknowledge"),
    tenant_id: str = Depends(get_current_tenant_id),
    user: Dict[str, Any] = Depends(require_acknowledge_alerts),
    alert_accessor: AlertDataAccessor = Depends(get_alert_accessor)
) -> Dict[str, Any]:
    """
    Acknowledge an alert.

    Args:
        alert_id: ID of the alert to acknowledge

    Returns:
        Acknowledgment result
    """
    try:
        logger.info(f"[Tenant: {tenant_id}] Acknowledging alert {alert_id} by user {user.get('username')}")
        result = await alert_accessor.acknowledge_alert(
            alert_id=alert_id,
            user=user["username"]
        )

        if not result["success"]:
            logger.warning(f"[Tenant: {tenant_id}] Alert {alert_id} not found or already acknowledged")
            raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")

        logger.info(f"[Tenant: {tenant_id}] Alert {alert_id} acknowledged successfully")
        return result
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")