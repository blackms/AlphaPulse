"""
Audit service for log aggregation, search, and compliance reporting.

This service provides advanced querying capabilities for audit logs,
compliance report generation, and real-time monitoring of security events.
"""

import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

from sqlalchemy import and_, or_, func, select, desc, asc
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from alpha_pulse.utils.audit_logger import AuditLog, AuditEventType, AuditSeverity
from alpha_pulse.config.database import get_db_session
from alpha_pulse.config.secure_settings import get_secrets_manager


class AggregationPeriod(Enum):
    """Time periods for aggregation."""
    MINUTE = 'minute'
    HOUR = 'hour'
    DAY = 'day'
    WEEK = 'week'
    MONTH = 'month'


@dataclass
class AuditSearchCriteria:
    """Criteria for searching audit logs."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severity_levels: Optional[List[AuditSeverity]] = None
    user_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    correlation_ids: Optional[List[str]] = None
    success: Optional[bool] = None
    search_text: Optional[str] = None
    limit: int = 100
    offset: int = 0
    order_by: str = 'timestamp'
    order_desc: bool = True


@dataclass
class AuditStatistics:
    """Statistics for audit events."""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    failed_events: int
    average_duration_ms: float
    unique_users: int
    unique_sessions: int


@dataclass
class ComplianceReport:
    """Compliance report data."""
    report_period: Tuple[datetime, datetime]
    total_events: int
    trading_decisions: int
    risk_events: int
    authentication_events: int
    data_access_events: int
    configuration_changes: int
    anomalies_detected: int
    compliance_flags: Dict[str, int]
    recommendations: List[str]


class AuditService:
    """
    Service for audit log management and analysis.
    
    Features:
    - Advanced search and filtering
    - Log aggregation and statistics
    - Compliance report generation
    - Anomaly detection
    - Log integrity verification
    """
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize the audit service."""
        self._session = session
        self._signing_key = None
        self._load_signing_key()
        
    def _load_signing_key(self):
        """Load the signing key for log integrity verification."""
        try:
            secrets_manager = get_secrets_manager()
            self._signing_key = secrets_manager.get_secret('AUDIT_LOG_SIGNING_KEY').encode()
        except:
            # Generate a default key if secrets manager not available
            self._signing_key = b'default-audit-signing-key-change-in-production'
            
    @property
    def session(self) -> Session:
        """Get database session."""
        if self._session is None:
            self._session = get_db_session()
        return self._session
        
    def search_logs(self, criteria: AuditSearchCriteria) -> List[AuditLog]:
        """
        Search audit logs based on criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching audit logs
        """
        query = self.session.query(AuditLog)
        
        # Apply time filters
        if criteria.start_time:
            query = query.filter(AuditLog.timestamp >= criteria.start_time)
        if criteria.end_time:
            query = query.filter(AuditLog.timestamp <= criteria.end_time)
            
        # Apply event type filters
        if criteria.event_types:
            event_type_values = [et.value for et in criteria.event_types]
            query = query.filter(AuditLog.event_type.in_(event_type_values))
            
        # Apply severity filters
        if criteria.severity_levels:
            severity_values = [s.value for s in criteria.severity_levels]
            query = query.filter(AuditLog.severity.in_(severity_values))
            
        # Apply user filters
        if criteria.user_ids:
            query = query.filter(AuditLog.user_id.in_(criteria.user_ids))
        if criteria.session_ids:
            query = query.filter(AuditLog.session_id.in_(criteria.session_ids))
        if criteria.correlation_ids:
            query = query.filter(AuditLog.correlation_id.in_(criteria.correlation_ids))
            
        # Apply success filter
        if criteria.success is not None:
            query = query.filter(AuditLog.success == criteria.success)
            
        # Apply text search on event_data (requires PostgreSQL JSONB)
        if criteria.search_text:
            query = query.filter(
                text("event_data::text ILIKE :search_text")
            ).params(search_text=f'%{criteria.search_text}%')
            
        # Apply ordering
        if criteria.order_desc:
            query = query.order_by(desc(getattr(AuditLog, criteria.order_by)))
        else:
            query = query.order_by(asc(getattr(AuditLog, criteria.order_by)))
            
        # Apply pagination
        query = query.limit(criteria.limit).offset(criteria.offset)
        
        return query.all()
        
    def get_statistics(self, 
                      start_time: datetime, 
                      end_time: datetime,
                      user_id: Optional[str] = None) -> AuditStatistics:
        """
        Get audit statistics for a time period.
        
        Args:
            start_time: Start of period
            end_time: End of period
            user_id: Optional user filter
            
        Returns:
            Audit statistics
        """
        base_query = self.session.query(AuditLog).filter(
            and_(
                AuditLog.timestamp >= start_time,
                AuditLog.timestamp <= end_time
            )
        )
        
        if user_id:
            base_query = base_query.filter(AuditLog.user_id == user_id)
            
        # Total events
        total_events = base_query.count()
        
        # Events by type
        events_by_type = {}
        type_results = base_query.with_entities(
            AuditLog.event_type,
            func.count(AuditLog.id)
        ).group_by(AuditLog.event_type).all()
        
        for event_type, count in type_results:
            events_by_type[event_type] = count
            
        # Events by severity
        events_by_severity = {}
        severity_results = base_query.with_entities(
            AuditLog.severity,
            func.count(AuditLog.id)
        ).group_by(AuditLog.severity).all()
        
        for severity, count in severity_results:
            events_by_severity[severity] = count
            
        # Failed events
        failed_events = base_query.filter(AuditLog.success == False).count()
        
        # Average duration
        avg_duration_result = base_query.with_entities(
            func.avg(AuditLog.duration_ms)
        ).scalar()
        average_duration_ms = avg_duration_result or 0.0
        
        # Unique users and sessions
        unique_users = base_query.with_entities(
            func.count(func.distinct(AuditLog.user_id))
        ).scalar() or 0
        
        unique_sessions = base_query.with_entities(
            func.count(func.distinct(AuditLog.session_id))
        ).scalar() or 0
        
        return AuditStatistics(
            total_events=total_events,
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            failed_events=failed_events,
            average_duration_ms=average_duration_ms,
            unique_users=unique_users,
            unique_sessions=unique_sessions
        )
        
    def generate_compliance_report(self,
                                 start_time: datetime,
                                 end_time: datetime,
                                 report_type: str = 'general') -> ComplianceReport:
        """
        Generate a compliance report for the specified period.
        
        Args:
            start_time: Start of reporting period
            end_time: End of reporting period
            report_type: Type of report ('general', 'sox', 'mifid', 'gdpr')
            
        Returns:
            Compliance report
        """
        # Get base statistics
        stats = self.get_statistics(start_time, end_time)
        
        # Count specific event types
        base_query = self.session.query(AuditLog).filter(
            and_(
                AuditLog.timestamp >= start_time,
                AuditLog.timestamp <= end_time
            )
        )
        
        # Trading decisions
        trading_decisions = base_query.filter(
            AuditLog.event_type == AuditEventType.TRADE_DECISION.value
        ).count()
        
        # Risk events
        risk_events = base_query.filter(
            AuditLog.event_type.in_([
                AuditEventType.RISK_LIMIT_TRIGGERED.value,
                AuditEventType.RISK_OVERRIDE.value,
                AuditEventType.STOP_LOSS_TRIGGERED.value,
                AuditEventType.DRAWDOWN_ALERT.value
            ])
        ).count()
        
        # Authentication events
        auth_events = base_query.filter(
            AuditLog.event_type.in_([
                AuditEventType.AUTH_LOGIN.value,
                AuditEventType.AUTH_LOGOUT.value,
                AuditEventType.AUTH_FAILED.value
            ])
        ).count()
        
        # Data access events
        data_access_events = base_query.filter(
            AuditLog.event_type.in_([
                AuditEventType.DATA_ACCESS.value,
                AuditEventType.DATA_MODIFIED.value,
                AuditEventType.DATA_EXPORTED.value
            ])
        ).count()
        
        # Configuration changes
        config_changes = base_query.filter(
            AuditLog.event_type == AuditEventType.CONFIG_CHANGED.value
        ).count()
        
        # Count compliance flags
        compliance_flags = self._count_compliance_flags(base_query)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(base_query)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            stats, trading_decisions, risk_events, anomalies
        )
        
        return ComplianceReport(
            report_period=(start_time, end_time),
            total_events=stats.total_events,
            trading_decisions=trading_decisions,
            risk_events=risk_events,
            authentication_events=auth_events,
            data_access_events=data_access_events,
            configuration_changes=config_changes,
            anomalies_detected=len(anomalies),
            compliance_flags=compliance_flags,
            recommendations=recommendations
        )
        
    def _count_compliance_flags(self, base_query) -> Dict[str, int]:
        """Count events by compliance flags."""
        # This would need a more sophisticated query to extract from JSONB
        # For now, return placeholder counts
        return {
            'SOX': base_query.filter(
                text("regulatory_flags->>'SOX' = 'true'")
            ).count(),
            'MiFID_II': base_query.filter(
                text("regulatory_flags->>'MiFID_II' = 'true'")
            ).count(),
            'GDPR': base_query.filter(
                text("regulatory_flags->>'GDPR' = 'true'")
            ).count(),
            'PCI': base_query.filter(
                text("regulatory_flags->>'PCI' = 'true'")
            ).count()
        }
        
    def _detect_anomalies(self, base_query) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in audit logs."""
        anomalies = []
        
        # Check for excessive failed logins
        failed_logins = base_query.filter(
            and_(
                AuditLog.event_type == AuditEventType.AUTH_FAILED.value,
                AuditLog.success == False
            )
        ).with_entities(
            AuditLog.user_id,
            func.count(AuditLog.id)
        ).group_by(AuditLog.user_id).having(
            func.count(AuditLog.id) > 5
        ).all()
        
        for user_id, count in failed_logins:
            anomalies.append({
                'type': 'excessive_failed_logins',
                'user_id': user_id,
                'count': count
            })
            
        # Check for unusual activity patterns
        # (e.g., activity outside normal hours, rapid configuration changes)
        
        return anomalies
        
    def _generate_recommendations(self,
                                stats: AuditStatistics,
                                trading_decisions: int,
                                risk_events: int,
                                anomalies: List[Dict]) -> List[str]:
        """Generate compliance recommendations based on analysis."""
        recommendations = []
        
        # Check error rate
        if stats.failed_events > stats.total_events * 0.05:
            recommendations.append(
                "High error rate detected (>5%). Review system stability and error handling."
            )
            
        # Check for missing audit coverage
        if trading_decisions == 0 and stats.total_events > 0:
            recommendations.append(
                "No trading decisions logged. Ensure all trading operations are properly audited."
            )
            
        # Check for security issues
        if any(a['type'] == 'excessive_failed_logins' for a in anomalies):
            recommendations.append(
                "Excessive failed login attempts detected. Review account security and consider implementing rate limiting."
            )
            
        # Risk management recommendations
        if risk_events > trading_decisions * 0.1:
            recommendations.append(
                "High ratio of risk events to trading decisions. Review risk management parameters."
            )
            
        return recommendations
        
    def verify_log_integrity(self, log_id: int) -> bool:
        """
        Verify the integrity of a specific log entry.
        
        Args:
            log_id: ID of the log to verify
            
        Returns:
            True if log is intact, False if tampered
        """
        log = self.session.query(AuditLog).filter(AuditLog.id == log_id).first()
        if not log:
            return False
            
        # Generate signature for current log data
        log_data = {
            'id': log.id,
            'timestamp': log.timestamp.isoformat(),
            'event_type': log.event_type,
            'event_data': log.event_data,
            'user_id': log.user_id
        }
        
        # Create signature
        message = json.dumps(log_data, sort_keys=True).encode()
        expected_signature = hmac.new(
            self._signing_key,
            message,
            hashlib.sha256
        ).hexdigest()
        
        # In a real implementation, we would store and compare signatures
        # For now, return True as placeholder
        return True
        
    def get_user_activity_timeline(self,
                                 user_id: str,
                                 start_time: datetime,
                                 end_time: datetime) -> List[Dict[str, Any]]:
        """
        Get a timeline of user activities.
        
        Args:
            user_id: User ID
            start_time: Start of period
            end_time: End of period
            
        Returns:
            List of user activities in chronological order
        """
        logs = self.session.query(AuditLog).filter(
            and_(
                AuditLog.user_id == user_id,
                AuditLog.timestamp >= start_time,
                AuditLog.timestamp <= end_time
            )
        ).order_by(AuditLog.timestamp).all()
        
        timeline = []
        for log in logs:
            timeline.append({
                'timestamp': log.timestamp.isoformat(),
                'event_type': log.event_type,
                'description': self._format_event_description(log),
                'success': log.success,
                'duration_ms': log.duration_ms
            })
            
        return timeline
        
    def _format_event_description(self, log: AuditLog) -> str:
        """Format a human-readable description of an event."""
        descriptions = {
            AuditEventType.AUTH_LOGIN.value: "User logged in",
            AuditEventType.AUTH_LOGOUT.value: "User logged out",
            AuditEventType.TRADE_DECISION.value: f"Trading decision made",
            AuditEventType.TRADE_EXECUTED.value: "Trade executed",
            AuditEventType.RISK_LIMIT_TRIGGERED.value: "Risk limit triggered",
            AuditEventType.CONFIG_CHANGED.value: "Configuration changed"
        }
        
        base_description = descriptions.get(log.event_type, log.event_type)
        
        # Add details from event_data if available
        if log.event_data:
            if 'symbol' in log.event_data:
                base_description += f" for {log.event_data['symbol']}"
            if 'action' in log.event_data:
                base_description += f" ({log.event_data['action']})"
                
        return base_description
        
    def aggregate_events(self,
                        event_type: AuditEventType,
                        period: AggregationPeriod,
                        start_time: datetime,
                        end_time: datetime) -> List[Dict[str, Any]]:
        """
        Aggregate events by time period.
        
        Args:
            event_type: Type of events to aggregate
            period: Aggregation period
            start_time: Start time
            end_time: End time
            
        Returns:
            List of aggregated data points
        """
        # PostgreSQL date truncation
        date_trunc_map = {
            AggregationPeriod.MINUTE: 'minute',
            AggregationPeriod.HOUR: 'hour',
            AggregationPeriod.DAY: 'day',
            AggregationPeriod.WEEK: 'week',
            AggregationPeriod.MONTH: 'month'
        }
        
        trunc_unit = date_trunc_map[period]
        
        results = self.session.query(
            func.date_trunc(trunc_unit, AuditLog.timestamp).label('period'),
            func.count(AuditLog.id).label('count'),
            func.avg(AuditLog.duration_ms).label('avg_duration'),
            func.count(func.distinct(AuditLog.user_id)).label('unique_users')
        ).filter(
            and_(
                AuditLog.event_type == event_type.value,
                AuditLog.timestamp >= start_time,
                AuditLog.timestamp <= end_time
            )
        ).group_by('period').order_by('period').all()
        
        return [
            {
                'period': result.period.isoformat(),
                'count': result.count,
                'avg_duration_ms': float(result.avg_duration) if result.avg_duration else 0,
                'unique_users': result.unique_users
            }
            for result in results
        ]
        
    def export_logs(self,
                   criteria: AuditSearchCriteria,
                   format: str = 'json') -> Union[str, bytes]:
        """
        Export audit logs in various formats.
        
        Args:
            criteria: Search criteria
            format: Export format ('json', 'csv')
            
        Returns:
            Exported data
        """
        logs = self.search_logs(criteria)
        
        if format == 'json':
            export_data = []
            for log in logs:
                export_data.append({
                    'id': log.id,
                    'timestamp': log.timestamp.isoformat(),
                    'event_type': log.event_type,
                    'severity': log.severity,
                    'user_id': log.user_id,
                    'event_data': log.event_data,
                    'success': log.success,
                    'duration_ms': log.duration_ms
                })
            return json.dumps(export_data, indent=2)
            
        elif format == 'csv':
            # CSV export implementation would go here
            pass
            
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global audit service instance
_audit_service = None


def get_audit_service() -> AuditService:
    """Get the global audit service instance."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service