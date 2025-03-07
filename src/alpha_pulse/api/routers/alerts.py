"""Alerts router."""
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException

from ..dependencies import get_current_user, has_permission
from ..data import AlertDataAccessor

router = APIRouter()
alert_accessor = AlertDataAccessor()


@router.get("/", response_model=List[Dict])
async def get_alerts(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    severity: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get alert history.
    
    Args:
        start_time: Filter alerts after this time
        end_time: Filter alerts before this time
        severity: Filter by severity
        acknowledged: Filter by acknowledgment status
        
    Returns:
        List of alert data
    """
    # Check permissions
    if not has_permission(current_user, "view_alerts"):
        raise HTTPException(status_code=403, detail="Not authorized to view alerts")
    
    # Build filters
    filters = {}
    if severity:
        filters["severity"] = severity
    if acknowledged is not None:
        filters["acknowledged"] = acknowledged
    
    # Get from data accessor
    return await alert_accessor.get_alerts(
        start_time=start_time,
        end_time=end_time,
        filters=filters
    )


@router.post("/{alert_id}/acknowledge", response_model=Dict)
async def acknowledge_alert(
    alert_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Acknowledge an alert.
    
    Args:
        alert_id: ID of the alert to acknowledge
        
    Returns:
        Updated alert data
    """
    # Check permissions
    if not has_permission(current_user, "acknowledge_alerts"):
        raise HTTPException(status_code=403, detail="Not authorized to acknowledge alerts")
    
    # Acknowledge alert
    result = await alert_accessor.acknowledge_alert(
        alert_id=alert_id,
        user=current_user["username"]
    )
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result