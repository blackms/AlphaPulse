"""
System router.

This module defines the API endpoints for system data.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import (
    require_view_system,
    get_system_accessor,
    get_alert_manager,
    get_portfolio_accessor
)
from ..data import SystemDataAccessor, PortfolioDataAccessor
from ..exchange_sync_integration import trigger_exchange_sync

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system")
async def get_system_metrics(
    _: Dict[str, Any] = Depends(require_view_system),
    system_accessor: SystemDataAccessor = Depends(get_system_accessor),
    alert_manager = Depends(get_alert_manager)
) -> Dict[str, Any]:
    """
    Get system metrics.
    
    Returns:
        System metrics data
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
            logger.error(f"Error processing alerts: {str(e)}")
            metrics["alert_processing_error"] = str(e)
    
    return metrics

@router.post("/system/exchange/reload")
async def force_exchange_sync(
    _: Dict[str, Any] = Depends(require_view_system),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    """
    Force synchronization of exchange data.
    
    This endpoint triggers an immediate synchronization of exchange data,
    bypassing the scheduled synchronization mechanism.
    
    Returns:
        Dict with operation status and results
    """
    try:
        return await trigger_exchange_sync(portfolio_accessor._exchange_id)
    except Exception as e:
        logger.error(f"Error forcing exchange sync: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to sync exchange data: {str(e)}")