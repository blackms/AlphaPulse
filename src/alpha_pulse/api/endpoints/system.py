"""
System endpoints for the API.

This module provides endpoints for system metrics and status.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import (
    get_system_accessor,
    get_alert_manager,
    require_view_system
)

# Create router
router = APIRouter(
    prefix="/system",
    tags=["system"],
    dependencies=[Depends(require_view_system)],
)


@router.get("", response_model=Dict[str, Any])
async def get_system_metrics(
    system_accessor=Depends(get_system_accessor),
    alert_manager=Depends(get_alert_manager)
):
    """
    Get system metrics.
    
    Returns:
        System metrics including CPU, memory, and disk usage
    """
    # Get system metrics
    metrics = await system_accessor.get_system_metrics()
    
    # Process metrics and generate alerts if needed
    if "error" not in metrics:
        # Create a dictionary of metrics for the alert system
        alert_metrics = {
            "cpu_usage_percent": metrics["cpu"]["usage_percent"],
            "memory_usage_percent": metrics["memory"]["percent"],
            "disk_usage_percent": metrics["disk"]["percent"],
        }
        
        # Process metrics through the alert system
        try:
            alerts = await alert_manager.process_metrics(alert_metrics)
            if alerts:
                # Include triggered alert information
                metrics["alerts"] = [
                    {
                        "id": alert.alert_id,
                        "message": alert.message,
                        "severity": alert.severity.value
                    } 
                    for alert in alerts
                ]
        except Exception as e:
            # Log error but continue to return metrics
            metrics["alert_processing_error"] = str(e)
    
    return metrics