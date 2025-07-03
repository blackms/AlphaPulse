"""
API routes for audit log access and reporting.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from alpha_pulse.api.auth import get_current_active_user, User, PermissionChecker
from alpha_pulse.utils.audit_queries import (
    AuditQueryBuilder,
    AuditReporter,
    AuditReportType,
    generate_audit_summary
)
from alpha_pulse.utils.audit_logger import AuditEventType, AuditSeverity

router = APIRouter(prefix="/audit", tags=["audit"])

# Permission checker for audit log access
require_audit_access = PermissionChecker(["view_audit_logs"])


class AuditLogResponse(BaseModel):
    """Response model for audit log entries."""
    id: int
    timestamp: datetime
    event_type: str
    severity: str
    user_id: Optional[str]
    ip_address: Optional[str]
    success: bool
    duration_ms: Optional[float]
    error_message: Optional[str]
    event_summary: Dict[str, Any]


class AuditReportResponse(BaseModel):
    """Response model for audit reports."""
    report_type: str
    period: Dict[str, str]
    generated_at: datetime
    data: Dict[str, Any]


class AuditSummaryResponse(BaseModel):
    """Response model for audit summary."""
    summary: str
    period_days: int
    generated_at: datetime


@router.get("/logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    start_date: Optional[datetime] = Query(None, description="Start date for logs"),
    end_date: Optional[datetime] = Query(None, description="End date for logs"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    severity: Optional[str] = Query(None, description="Minimum severity level"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    user: User = Depends(require_audit_access)
):
    """
    Retrieve audit logs with filtering options.
    
    Requires: view_audit_logs permission
    """
    # Build query
    builder = AuditQueryBuilder()
    
    # Apply filters
    if start_date:
        builder.time_range(start_date, end_date)
    else:
        # Default to last 24 hours
        builder.time_range(datetime.utcnow() - timedelta(hours=24))
        
    if event_type:
        try:
            event_enum = AuditEventType(event_type)
            builder.event_types(event_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
            
    if user_id:
        builder.user(user_id)
        
    if severity:
        try:
            severity_enum = AuditSeverity(severity)
            builder.severity(severity_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
            
    # Apply ordering and pagination
    builder.order_by_time(descending=True)
    logs = builder.query.offset(offset).limit(limit).all()
    
    # Convert to response model
    responses = []
    for log in logs:
        # Extract key fields from event_data for summary
        event_summary = {}
        if log.event_data:
            for key in ['symbol', 'action', 'method', 'path', 'agent', 'error']:
                if key in log.event_data:
                    event_summary[key] = log.event_data[key]
                    
        responses.append(AuditLogResponse(
            id=log.id,
            timestamp=log.timestamp,
            event_type=log.event_type,
            severity=log.severity,
            user_id=log.user_id,
            ip_address=log.ip_address,
            success=log.success,
            duration_ms=log.duration_ms,
            error_message=log.error_message,
            event_summary=event_summary
        ))
        
    return responses


@router.get("/reports/security", response_model=AuditReportResponse)
async def get_security_report(
    days: int = Query(7, ge=1, le=90, description="Number of days to include"),
    user: User = Depends(require_audit_access)
):
    """
    Generate security audit report.
    
    Includes:
    - Authentication events
    - Failed login attempts
    - API security events
    - Secret access logs
    
    Requires: view_audit_logs permission
    """
    reporter = AuditReporter()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    report_data = reporter.security_summary(start_time, end_time)
    
    return AuditReportResponse(
        report_type="security_summary",
        period={
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        generated_at=datetime.utcnow(),
        data=report_data
    )


@router.get("/reports/trading", response_model=AuditReportResponse)
async def get_trading_report(
    days: int = Query(7, ge=1, le=90, description="Number of days to include"),
    user_id: Optional[str] = Query(None, description="Filter by specific user"),
    user: User = Depends(require_audit_access)
):
    """
    Generate trading activity audit report.
    
    Includes:
    - Trading decisions
    - Trade executions
    - Success rates
    - Performance metrics
    
    Requires: view_audit_logs permission
    """
    reporter = AuditReporter()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    report_data = reporter.trading_activity(start_time, end_time, user_id)
    
    return AuditReportResponse(
        report_type="trading_activity",
        period={
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        generated_at=datetime.utcnow(),
        data=report_data
    )


@router.get("/reports/risk", response_model=AuditReportResponse)
async def get_risk_report(
    days: int = Query(7, ge=1, le=90, description="Number of days to include"),
    user: User = Depends(require_audit_access)
):
    """
    Generate risk management audit report.
    
    Includes:
    - Risk limits triggered
    - Risk overrides
    - Stop loss events
    - Drawdown alerts
    
    Requires: view_audit_logs permission
    """
    reporter = AuditReporter()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    report_data = reporter.risk_events(start_time, end_time)
    
    return AuditReportResponse(
        report_type="risk_events",
        period={
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        generated_at=datetime.utcnow(),
        data=report_data
    )


@router.get("/reports/compliance", response_model=AuditReportResponse)
async def get_compliance_report(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    regulations: Optional[List[str]] = Query(None, description="Specific regulations to report on"),
    user: User = Depends(require_audit_access)
):
    """
    Generate compliance audit report.
    
    Includes events flagged for:
    - GDPR compliance
    - SOX compliance
    - PCI compliance
    - Other regulatory requirements
    
    Requires: view_audit_logs permission
    """
    reporter = AuditReporter()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    report_data = reporter.compliance_report(start_time, end_time, regulations)
    
    return AuditReportResponse(
        report_type="compliance_report",
        period={
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        generated_at=datetime.utcnow(),
        data=report_data
    )


@router.get("/anomalies", response_model=List[Dict[str, Any]])
async def get_anomalies(
    lookback_days: int = Query(7, ge=1, le=30, description="Days of history to analyze"),
    threshold: float = Query(3.0, ge=1.0, le=5.0, description="Standard deviations for anomaly threshold"),
    user: User = Depends(require_audit_access)
):
    """
    Detect anomalous patterns in audit logs.
    
    Looks for:
    - Unusual login patterns
    - Abnormal API usage
    - Suspicious activity patterns
    
    Requires: view_audit_logs permission
    """
    reporter = AuditReporter()
    anomalies = reporter.detect_anomalies(lookback_days, threshold)
    
    return anomalies


@router.get("/summary", response_model=AuditSummaryResponse)
async def get_audit_summary(
    days: int = Query(30, ge=1, le=90, description="Number of days to summarize"),
    user: User = Depends(require_audit_access)
):
    """
    Get human-readable audit summary.
    
    Provides high-level overview of:
    - Security events
    - Trading activity
    - Risk events
    - Detected anomalies
    
    Requires: view_audit_logs permission
    """
    summary_text = generate_audit_summary(days)
    
    return AuditSummaryResponse(
        summary=summary_text,
        period_days=days,
        generated_at=datetime.utcnow()
    )


@router.post("/export")
async def export_audit_logs(
    start_date: datetime = Query(..., description="Start date for export"),
    end_date: datetime = Query(..., description="End date for export"),
    format: str = Query("csv", regex="^(csv|json|parquet)$", description="Export format"),
    user: User = Depends(require_audit_access)
):
    """
    Export audit logs for external analysis.
    
    Formats:
    - CSV: For spreadsheet analysis
    - JSON: For programmatic processing
    - Parquet: For big data analysis
    
    Requires: view_audit_logs permission
    """
    # Validate date range
    if end_date <= start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")
        
    if (end_date - start_date).days > 365:
        raise HTTPException(status_code=400, detail="Export range cannot exceed 365 days")
        
    # Generate export
    reporter = AuditReporter()
    
    # Create temporary file for export
    import tempfile
    import os
    
    suffix = f".{format}"
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmp:
        output_path = tmp.name
        
    try:
        reporter.export_for_analysis(output_path, start_date, end_date, format)
        
        # Return file info (in production, would return download URL)
        file_size = os.path.getsize(output_path)
        
        return {
            "status": "success",
            "format": format,
            "file_size": file_size,
            "rows_exported": "Check file",  # Would count in production
            "download_url": f"/audit/download/{os.path.basename(output_path)}"
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/event-types")
async def get_event_types(
    user: User = Depends(require_audit_access)
):
    """
    Get list of available audit event types.
    
    Requires: view_audit_logs permission
    """
    return {
        "event_types": [
            {
                "value": event_type.value,
                "name": event_type.name,
                "category": event_type.value.split('.')[0]
            }
            for event_type in AuditEventType
        ]
    }


@router.get("/severity-levels")
async def get_severity_levels(
    user: User = Depends(require_audit_access)
):
    """
    Get list of audit severity levels.
    
    Requires: view_audit_logs permission
    """
    return {
        "severity_levels": [
            {
                "value": severity.value,
                "name": severity.name,
                "level": i
            }
            for i, severity in enumerate(AuditSeverity)
        ]
    }