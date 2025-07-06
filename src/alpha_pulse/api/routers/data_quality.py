"""
Data Quality API endpoints for monitoring and managing data quality.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from loguru import logger

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


class QualityThresholdUpdate(BaseModel):
    """Model for updating quality thresholds."""
    symbol: Optional[str] = None
    metric: str
    threshold: float
    enabled: bool = True


class QuarantineAction(BaseModel):
    """Model for quarantine actions."""
    action: str  # release, extend, delete
    reason: Optional[str] = None


@router.get("/metrics", response_model=Dict[str, Any])
async def get_quality_metrics(
    symbol: str = Query("ALL", description="Symbol filter (ALL for aggregate)"),
    time_range: str = Query("24h", description="Time range (1h, 24h, 7d, 30d)")
) -> Dict[str, Any]:
    """
    Get comprehensive data quality metrics.
    
    Args:
        symbol: Symbol to filter by (ALL for aggregate metrics)
        time_range: Time range for metrics calculation
        
    Returns:
        Comprehensive quality metrics
    """
    try:
        # Import here to avoid circular imports
        from ...data.quality.quality_metrics import QualityMetricsCalculator
        
        calculator = QualityMetricsCalculator()
        
        # Parse time range
        time_delta_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        if time_range not in time_delta_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid time range: {time_range}"
            )
        
        end_time = datetime.now()
        start_time = end_time - time_delta_map[time_range]
        
        # Calculate metrics
        if symbol == "ALL":
            metrics = await calculator.calculate_aggregate_metrics(start_time, end_time)
        else:
            metrics = await calculator.calculate_symbol_metrics(symbol, start_time, end_time)
        
        # Format response
        response = {
            "symbol": symbol,
            "time_range": time_range,
            "calculation_time": datetime.now().isoformat(),
            "overall_score": metrics.get("overall_score", 0.0),
            "completeness": metrics.get("completeness", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "consistency": metrics.get("consistency", 0.0),
            "timeliness": metrics.get("timeliness", 0.0),
            "validity": metrics.get("validity", 0.0),
            "uniqueness": metrics.get("uniqueness", 0.0),
            "anomaly_rate": metrics.get("anomaly_rate", 0.0),
            "data_volume": metrics.get("data_volume", 0),
            "source_reliability": metrics.get("source_reliability", 0.0),
            "quality_trend": metrics.get("quality_trend", "stable"),
            "last_updated": metrics.get("last_updated", datetime.now().isoformat()),
            "sla_compliance": metrics.get("sla_compliance", True),
            "recommendations": metrics.get("recommendations", [])
        }
        
        logger.info(f"Retrieved quality metrics for {symbol} over {time_range}")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving quality metrics for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality metrics: {str(e)}"
        )


@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_quality_alerts(
    limit: int = Query(50, description="Maximum number of alerts to return"),
    severity: Optional[str] = Query(None, description="Filter by severity (critical, warning, info)"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)")
) -> List[Dict[str, Any]]:
    """
    Get quality alerts with optional filtering.
    
    Args:
        limit: Maximum number of alerts to return
        severity: Optional severity filter
        symbol: Optional symbol filter
        start_date: Optional start date filter
        
    Returns:
        List of quality alerts
    """
    try:
        from ...data.quality.quality_metrics import QualityAlertManager
        
        alert_manager = QualityAlertManager()
        
        # Build filter criteria
        filters = {"limit": limit}
        if severity:
            if severity not in ["critical", "warning", "info"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid severity: {severity}"
                )
            filters["severity"] = severity
        
        if symbol and symbol != "ALL":
            filters["symbol"] = symbol
        
        if start_date:
            try:
                filters["start_date"] = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid date format: {start_date}"
                )
        
        # Get alerts
        alerts = await alert_manager.get_alerts(**filters)
        
        # Format alerts
        formatted_alerts = []
        for alert in alerts:
            formatted_alerts.append({
                "id": alert.get("id", ""),
                "symbol": alert.get("symbol", ""),
                "metric": alert.get("metric", ""),
                "severity": alert.get("severity", "info"),
                "message": alert.get("message", ""),
                "timestamp": alert.get("timestamp", datetime.now().isoformat()),
                "threshold_violated": alert.get("threshold_violated", 0.0),
                "current_value": alert.get("current_value", 0.0),
                "source": alert.get("source", "unknown"),
                "resolved": alert.get("resolved", False),
                "resolution_time": alert.get("resolution_time"),
                "metadata": alert.get("metadata", {})
            })
        
        logger.info(f"Retrieved {len(formatted_alerts)} quality alerts")
        return formatted_alerts
        
    except Exception as e:
        logger.error(f"Error retrieving quality alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality alerts: {str(e)}"
        )


@router.get("/sources", response_model=List[Dict[str, Any]])
async def get_data_source_status() -> List[Dict[str, Any]]:
    """
    Get status of all data sources.
    
    Returns:
        List of data source status information
    """
    try:
        from ...data.quality.data_validator import DataValidator
        
        validator = DataValidator()
        source_status = await validator.get_source_status()
        
        # Format source status
        formatted_sources = []
        for source in source_status:
            formatted_sources.append({
                "source_id": source.get("source_id", ""),
                "source_name": source.get("source_name", ""),
                "status": source.get("status", "unknown"),
                "reliability_score": source.get("reliability_score", 0.0),
                "last_update": source.get("last_update", datetime.now().isoformat()),
                "quality_score": source.get("quality_score", 0.0),
                "data_volume": source.get("data_volume", 0),
                "error_count": source.get("error_count", 0),
                "latency_ms": source.get("latency_ms", 0),
                "uptime_percentage": source.get("uptime_percentage", 0.0),
                "last_error": source.get("last_error"),
                "configuration": source.get("configuration", {})
            })
        
        logger.info(f"Retrieved status for {len(formatted_sources)} data sources")
        return formatted_sources
        
    except Exception as e:
        logger.error(f"Error retrieving data source status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data source status: {str(e)}"
        )


@router.get("/quarantine", response_model=List[Dict[str, Any]])
async def get_quarantined_data(
    limit: int = Query(100, description="Maximum number of quarantine entries"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    reason: Optional[str] = Query(None, description="Filter by quarantine reason")
) -> List[Dict[str, Any]]:
    """
    Get quarantined data entries.
    
    Args:
        limit: Maximum number of entries to return
        symbol: Optional symbol filter
        reason: Optional reason filter
        
    Returns:
        List of quarantined data entries
    """
    try:
        from ...data.quality.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Build filter criteria
        filters = {"limit": limit}
        if symbol:
            filters["symbol"] = symbol
        if reason:
            filters["reason"] = reason
        
        quarantine_data = await validator.get_quarantine_data(**filters)
        
        # Format quarantine data
        formatted_data = []
        for entry in quarantine_data:
            formatted_data.append({
                "id": entry.get("id", ""),
                "symbol": entry.get("symbol", ""),
                "reason": entry.get("reason", ""),
                "quarantine_time": entry.get("quarantine_time", datetime.now().isoformat()),
                "quality_score": entry.get("quality_score", 0.0),
                "data_points": entry.get("data_points", 0),
                "source": entry.get("source", "unknown"),
                "expires_at": entry.get("expires_at"),
                "metadata": entry.get("metadata", {}),
                "can_release": entry.get("can_release", False),
                "review_required": entry.get("review_required", False)
            })
        
        logger.info(f"Retrieved {len(formatted_data)} quarantined data entries")
        return formatted_data
        
    except Exception as e:
        logger.error(f"Error retrieving quarantined data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quarantined data: {str(e)}"
        )


@router.post("/quarantine/{quarantine_id}/action")
async def manage_quarantine(
    quarantine_id: str,
    action: QuarantineAction
) -> Dict[str, Any]:
    """
    Manage quarantined data (release, extend, delete).
    
    Args:
        quarantine_id: ID of quarantined data
        action: Action to perform
        
    Returns:
        Action result
    """
    try:
        from ...data.quality.data_validator import DataValidator
        
        validator = DataValidator()
        
        if action.action not in ["release", "extend", "delete"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {action.action}"
            )
        
        result = await validator.manage_quarantine(
            quarantine_id=quarantine_id,
            action=action.action,
            reason=action.reason
        )
        
        logger.info(f"Performed {action.action} on quarantine {quarantine_id}")
        
        return {
            "quarantine_id": quarantine_id,
            "action": action.action,
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error managing quarantine {quarantine_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to manage quarantine: {str(e)}"
        )


@router.get("/thresholds", response_model=Dict[str, Any])
async def get_quality_thresholds(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
) -> Dict[str, Any]:
    """
    Get current quality thresholds.
    
    Args:
        symbol: Optional symbol filter
        
    Returns:
        Quality thresholds configuration
    """
    try:
        from ...config.quality_rules import QualityRulesManager
        
        rules_manager = QualityRulesManager()
        
        if symbol:
            thresholds = await rules_manager.get_symbol_thresholds(symbol)
        else:
            thresholds = await rules_manager.get_default_thresholds()
        
        return {
            "symbol": symbol,
            "thresholds": thresholds,
            "profiles_available": ["strict", "standard", "relaxed", "custom"],
            "current_profile": thresholds.get("profile", "standard"),
            "last_updated": thresholds.get("last_updated", datetime.now().isoformat()),
            "auto_adjustment": thresholds.get("auto_adjustment", True)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving quality thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality thresholds: {str(e)}"
        )


@router.put("/thresholds")
async def update_quality_thresholds(
    threshold_update: QualityThresholdUpdate
) -> Dict[str, Any]:
    """
    Update quality thresholds.
    
    Args:
        threshold_update: Threshold update parameters
        
    Returns:
        Update confirmation
    """
    try:
        from ...config.quality_rules import QualityRulesManager
        
        rules_manager = QualityRulesManager()
        
        result = await rules_manager.update_threshold(
            symbol=threshold_update.symbol,
            metric=threshold_update.metric,
            threshold=threshold_update.threshold,
            enabled=threshold_update.enabled
        )
        
        logger.info(f"Updated quality threshold for {threshold_update.metric}")
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "updated_threshold": {
                "symbol": threshold_update.symbol,
                "metric": threshold_update.metric,
                "threshold": threshold_update.threshold,
                "enabled": threshold_update.enabled
            },
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating quality thresholds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update quality thresholds: {str(e)}"
        )


@router.get("/reports/summary", response_model=Dict[str, Any])
async def get_quality_summary_report(
    time_range: str = Query("24h", description="Time range for report"),
    include_trends: bool = Query(True, description="Include trend analysis")
) -> Dict[str, Any]:
    """
    Get comprehensive quality summary report.
    
    Args:
        time_range: Time range for the report
        include_trends: Whether to include trend analysis
        
    Returns:
        Quality summary report
    """
    try:
        from ...data.quality.quality_metrics import QualityReportGenerator
        
        report_generator = QualityReportGenerator()
        
        # Parse time range
        time_delta_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        if time_range not in time_delta_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid time range: {time_range}"
            )
        
        end_time = datetime.now()
        start_time = end_time - time_delta_map[time_range]
        
        report = await report_generator.generate_summary_report(
            start_time=start_time,
            end_time=end_time,
            include_trends=include_trends
        )
        
        return {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_range": time_range,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            },
            "quality_summary": report.get("summary", {}),
            "key_insights": report.get("insights", []),
            "alerts_summary": report.get("alerts", {}),
            "source_performance": report.get("sources", {}),
            "recommendations": report.get("recommendations", []),
            "trends": report.get("trends", {}) if include_trends else {},
            "sla_compliance": report.get("sla_compliance", {})
        }
        
    except Exception as e:
        logger.error(f"Error generating quality summary report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate quality report: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_quality_system_health() -> Dict[str, Any]:
    """
    Get data quality system health status.
    
    Returns:
        System health information
    """
    try:
        from ...pipelines.data_quality_pipeline import DataQualityPipeline
        
        pipeline = DataQualityPipeline()
        health_status = await pipeline.get_health_status()
        
        return {
            "system_status": health_status.get("status", "unknown"),
            "uptime": health_status.get("uptime", 0),
            "last_update": health_status.get("last_update", datetime.now().isoformat()),
            "pipeline_status": {
                "running": health_status.get("pipeline_running", False),
                "performance": health_status.get("performance", {}),
                "errors": health_status.get("errors", []),
                "throughput": health_status.get("throughput", 0)
            },
            "component_status": {
                "validator": health_status.get("validator_status", "unknown"),
                "anomaly_detector": health_status.get("anomaly_detector_status", "unknown"),
                "metrics_calculator": health_status.get("metrics_calculator_status", "unknown"),
                "alert_manager": health_status.get("alert_manager_status", "unknown")
            },
            "resource_usage": {
                "cpu_usage": health_status.get("cpu_usage", 0.0),
                "memory_usage": health_status.get("memory_usage", 0.0),
                "queue_size": health_status.get("queue_size", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving quality system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system health: {str(e)}"
        )