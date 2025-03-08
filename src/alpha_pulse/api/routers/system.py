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
from ...data_pipeline.scheduler import ExchangeDataSynchronizer

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
async def force_exchange_data_reload(
    _: Dict[str, Any] = Depends(require_view_system),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
) -> Dict[str, Any]:
    """
    Force reload of exchange data.
    
    This endpoint triggers an immediate refresh of exchange data,
    bypassing the scheduled refresh mechanism.
    
    Returns:
        Dict with operation status and timestamp
    """
    try:
        # Use the portfolio accessor to get the exchange ID
        exchange_id = portfolio_accessor._exchange_id
        
        # Get the global exchange data synchronizer instance
        from ...data_pipeline.scheduler import exchange_data_synchronizer
        
        # Trigger immediate sync for all data types
        from ...data_pipeline.scheduler import DataType
        success = exchange_data_synchronizer.trigger_sync(exchange_id, DataType.ALL)
        
        if not success:
            raise Exception(f"Failed to trigger synchronization for {exchange_id}")
        
        # Return successful result
        from datetime import datetime, timezone
        sync_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "next_sync": None  # This is handled by the synchronizer
        }
        
        # Refresh the portfolio accessor's cache if needed
        if hasattr(portfolio_accessor, '_get_portfolio_from_exchange'):
            portfolio_data = await portfolio_accessor._get_portfolio_from_exchange()
            
        return {
            "status": "success",
            "message": f"Exchange data for {exchange_id} reloaded successfully",
            "timestamp": sync_result.get("timestamp", "N/A"),
            "next_sync": sync_result.get("next_sync", "N/A")
        }
    except Exception as e:
        logger.error(f"Error forcing exchange data reload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload exchange data: {str(e)}"
        )