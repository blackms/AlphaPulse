"""
Audit log query and reporting utilities.

Provides functions for querying, analyzing, and reporting on audit logs.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import pandas as pd
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from alpha_pulse.utils.audit_logger import AuditLog, AuditEventType, AuditSeverity
from alpha_pulse.config.database import get_db_session


class AuditReportType(Enum):
    """Types of audit reports."""
    SECURITY_SUMMARY = "security_summary"
    TRADING_ACTIVITY = "trading_activity"
    RISK_EVENTS = "risk_events"
    API_USAGE = "api_usage"
    USER_ACTIVITY = "user_activity"
    COMPLIANCE_REPORT = "compliance_report"
    ANOMALY_DETECTION = "anomaly_detection"


class AuditQueryBuilder:
    """Build complex queries for audit logs."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize with database session."""
        self.session = session or get_db_session()
        self.query = self.session.query(AuditLog)
        
    def time_range(self, start: datetime, end: Optional[datetime] = None):
        """Filter by time range."""
        self.query = self.query.filter(AuditLog.timestamp >= start)
        if end:
            self.query = self.query.filter(AuditLog.timestamp <= end)
        return self
        
    def event_types(self, *types: AuditEventType):
        """Filter by event types."""
        type_values = [t.value for t in types]
        self.query = self.query.filter(AuditLog.event_type.in_(type_values))
        return self
        
    def user(self, user_id: str):
        """Filter by user ID."""
        self.query = self.query.filter(AuditLog.user_id == user_id)
        return self
        
    def severity(self, min_severity: AuditSeverity):
        """Filter by minimum severity."""
        severities = {
            AuditSeverity.DEBUG: 0,
            AuditSeverity.INFO: 1,
            AuditSeverity.WARNING: 2,
            AuditSeverity.ERROR: 3,
            AuditSeverity.CRITICAL: 4
        }
        
        min_level = severities[min_severity]
        allowed_severities = [
            s.value for s, level in severities.items() 
            if level >= min_level
        ]
        
        self.query = self.query.filter(AuditLog.severity.in_(allowed_severities))
        return self
        
    def failed_only(self):
        """Filter to show only failed operations."""
        self.query = self.query.filter(AuditLog.success == False)
        return self
        
    def ip_address(self, ip: str):
        """Filter by IP address."""
        self.query = self.query.filter(AuditLog.ip_address == ip)
        return self
        
    def correlation_id(self, correlation_id: str):
        """Filter by correlation ID to trace related events."""
        self.query = self.query.filter(AuditLog.correlation_id == correlation_id)
        return self
        
    def regulatory_flag(self, flag: str):
        """Filter by regulatory compliance flag."""
        self.query = self.query.filter(
            func.json_extract(AuditLog.regulatory_flags, f'$.{flag}') == True
        )
        return self
        
    def order_by_time(self, descending: bool = True):
        """Order results by timestamp."""
        if descending:
            self.query = self.query.order_by(AuditLog.timestamp.desc())
        else:
            self.query = self.query.order_by(AuditLog.timestamp.asc())
        return self
        
    def limit(self, limit: int):
        """Limit number of results."""
        self.query = self.query.limit(limit)
        return self
        
    def execute(self) -> List[AuditLog]:
        """Execute the query and return results."""
        return self.query.all()
        
    def count(self) -> int:
        """Get count of matching records."""
        return self.query.count()
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        results = self.execute()
        
        data = []
        for log in results:
            row = {
                'timestamp': log.timestamp,
                'event_type': log.event_type,
                'severity': log.severity,
                'user_id': log.user_id,
                'session_id': log.session_id,
                'ip_address': log.ip_address,
                'success': log.success,
                'duration_ms': log.duration_ms,
                'error_message': log.error_message,
                'data_classification': log.data_classification
            }
            
            # Add selected event data fields
            if log.event_data:
                for key in ['symbol', 'action', 'quantity', 'method', 'path']:
                    if key in log.event_data:
                        row[f'event_{key}'] = log.event_data[key]
                        
            data.append(row)
            
        return pd.DataFrame(data)


class AuditReporter:
    """Generate audit reports and analytics."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize with database session."""
        self.session = session or get_db_session()
        
    def security_summary(self, 
                        start_time: datetime,
                        end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate security summary report.
        
        Returns:
            Summary of authentication events, failed attempts, etc.
        """
        builder = AuditQueryBuilder(self.session)
        builder.time_range(start_time, end_time)
        
        # Authentication stats
        auth_events = (
            builder.event_types(
                AuditEventType.AUTH_LOGIN,
                AuditEventType.AUTH_FAILED,
                AuditEventType.AUTH_LOGOUT
            )
            .to_dataframe()
        )
        
        # Failed login analysis
        failed_logins = auth_events[auth_events['event_type'] == AuditEventType.AUTH_FAILED.value]
        failed_by_user = failed_logins.groupby('user_id').size().to_dict()
        failed_by_ip = failed_logins.groupby('ip_address').size().to_dict()
        
        # API security events
        api_builder = AuditQueryBuilder(self.session)
        api_security = (
            api_builder
            .time_range(start_time, end_time)
            .event_types(
                AuditEventType.API_ERROR,
                AuditEventType.API_RATE_LIMITED
            )
            .count()
        )
        
        # Secret access
        secret_builder = AuditQueryBuilder(self.session)
        secret_access = (
            secret_builder
            .time_range(start_time, end_time)
            .event_types(AuditEventType.SECRET_ACCESSED)
            .count()
        )
        
        return {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat() if end_time else 'current'
            },
            'authentication': {
                'successful_logins': len(auth_events[auth_events['success'] == True]),
                'failed_logins': len(failed_logins),
                'unique_users': auth_events['user_id'].nunique(),
                'failed_by_user': failed_by_user,
                'failed_by_ip': failed_by_ip
            },
            'api_security': {
                'errors': api_security,
                'rate_limit_hits': api_security
            },
            'secret_access_count': secret_access,
            'top_ips': auth_events['ip_address'].value_counts().head(10).to_dict()
        }
        
    def trading_activity(self,
                        start_time: datetime,
                        end_time: Optional[datetime] = None,
                        user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate trading activity report.
        
        Returns:
            Summary of trading decisions and executions.
        """
        builder = AuditQueryBuilder(self.session)
        builder.time_range(start_time, end_time)
        
        if user_id:
            builder.user(user_id)
            
        # Trading events
        trade_events = (
            builder.event_types(
                AuditEventType.TRADE_DECISION,
                AuditEventType.TRADE_EXECUTED,
                AuditEventType.TRADE_FAILED,
                AuditEventType.TRADE_CANCELLED
            )
            .to_dataframe()
        )
        
        if trade_events.empty:
            return {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat() if end_time else 'current'
                },
                'summary': 'No trading activity in period'
            }
            
        # Analysis
        decisions = trade_events[trade_events['event_type'] == AuditEventType.TRADE_DECISION.value]
        executions = trade_events[trade_events['event_type'] == AuditEventType.TRADE_EXECUTED.value]
        failures = trade_events[trade_events['event_type'] == AuditEventType.TRADE_FAILED.value]
        
        # Success rate
        total_attempts = len(executions) + len(failures)
        success_rate = len(executions) / total_attempts if total_attempts > 0 else 0
        
        # Performance metrics
        avg_duration = executions['duration_ms'].mean() if not executions.empty else 0
        
        return {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat() if end_time else 'current'
            },
            'summary': {
                'total_decisions': len(decisions),
                'total_executions': len(executions),
                'total_failures': len(failures),
                'success_rate': success_rate,
                'avg_execution_time_ms': avg_duration
            },
            'by_symbol': trade_events.groupby('event_symbol').size().to_dict() if 'event_symbol' in trade_events else {},
            'by_user': trade_events.groupby('user_id').size().to_dict() if not user_id else None,
            'failure_reasons': failures['error_message'].value_counts().to_dict() if not failures.empty else {}
        }
        
    def risk_events(self,
                   start_time: datetime,
                   end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate risk management events report.
        
        Returns:
            Summary of risk limits triggered, overrides, etc.
        """
        builder = AuditQueryBuilder(self.session)
        
        risk_events = (
            builder
            .time_range(start_time, end_time)
            .event_types(
                AuditEventType.RISK_LIMIT_TRIGGERED,
                AuditEventType.RISK_OVERRIDE,
                AuditEventType.STOP_LOSS_TRIGGERED,
                AuditEventType.DRAWDOWN_ALERT
            )
            .to_dataframe()
        )
        
        if risk_events.empty:
            return {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat() if end_time else 'current'
                },
                'summary': 'No risk events in period'
            }
            
        # Group by event type
        by_type = risk_events.groupby('event_type').size().to_dict()
        
        # Risk overrides (potential concern)
        overrides = risk_events[risk_events['event_type'] == AuditEventType.RISK_OVERRIDE.value]
        
        return {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat() if end_time else 'current'
            },
            'summary': {
                'total_events': len(risk_events),
                'by_type': by_type,
                'total_overrides': len(overrides)
            },
            'overrides': {
                'count': len(overrides),
                'by_user': overrides.groupby('user_id').size().to_dict() if not overrides.empty else {}
            },
            'timeline': risk_events.groupby(
                pd.Grouper(key='timestamp', freq='1H')
            ).size().to_dict()
        }
        
    def compliance_report(self,
                         start_time: datetime,
                         end_time: Optional[datetime] = None,
                         regulations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate compliance report for regulatory requirements.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            regulations: List of regulations to report on (GDPR, SOX, PCI)
            
        Returns:
            Compliance-focused audit summary
        """
        if regulations is None:
            regulations = ['GDPR', 'SOX', 'PCI']
            
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat() if end_time else 'current'
            },
            'regulations': {}
        }
        
        for regulation in regulations:
            builder = AuditQueryBuilder(self.session)
            
            # Get events with this regulatory flag
            events = (
                builder
                .time_range(start_time, end_time)
                .regulatory_flag(regulation)
                .to_dataframe()
            )
            
            if not events.empty:
                report['regulations'][regulation] = {
                    'total_events': len(events),
                    'by_type': events.groupby('event_type').size().to_dict(),
                    'by_classification': events.groupby('data_classification').size().to_dict(),
                    'failed_operations': len(events[events['success'] == False])
                }
            else:
                report['regulations'][regulation] = {
                    'total_events': 0,
                    'note': 'No events with this regulatory flag'
                }
                
        return report
        
    def detect_anomalies(self,
                        lookback_days: int = 7,
                        threshold_multiplier: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect anomalous patterns in audit logs.
        
        Args:
            lookback_days: Days of history to analyze
            threshold_multiplier: Standard deviations for anomaly threshold
            
        Returns:
            List of detected anomalies
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        
        anomalies = []
        
        # Check for unusual failed login patterns
        builder = AuditQueryBuilder(self.session)
        failed_logins = (
            builder
            .time_range(start_time, end_time)
            .event_types(AuditEventType.AUTH_FAILED)
            .to_dataframe()
        )
        
        if not failed_logins.empty:
            # Group by hour
            hourly_failures = failed_logins.groupby(
                pd.Grouper(key='timestamp', freq='1H')
            ).size()
            
            mean_failures = hourly_failures.mean()
            std_failures = hourly_failures.std()
            threshold = mean_failures + (threshold_multiplier * std_failures)
            
            # Find anomalous hours
            for timestamp, count in hourly_failures.items():
                if count > threshold:
                    anomalies.append({
                        'type': 'excessive_failed_logins',
                        'timestamp': timestamp.isoformat(),
                        'count': int(count),
                        'threshold': float(threshold),
                        'severity': 'high'
                    })
                    
        # Check for unusual API activity
        api_builder = AuditQueryBuilder(self.session)
        api_events = (
            api_builder
            .time_range(start_time, end_time)
            .event_types(AuditEventType.API_REQUEST)
            .to_dataframe()
        )
        
        if not api_events.empty:
            # Check for unusual user activity
            user_activity = api_events.groupby(['user_id', pd.Grouper(key='timestamp', freq='1H')]).size()
            
            for (user_id, _), count in user_activity.items():
                user_mean = user_activity[user_id].mean()
                user_std = user_activity[user_id].std()
                
                if user_std > 0:
                    z_score = (count - user_mean) / user_std
                    if abs(z_score) > threshold_multiplier:
                        anomalies.append({
                            'type': 'unusual_api_activity',
                            'user_id': user_id,
                            'count': int(count),
                            'z_score': float(z_score),
                            'severity': 'medium'
                        })
                        
        return anomalies
        
    def export_for_analysis(self,
                           output_path: str,
                           start_time: datetime,
                           end_time: Optional[datetime] = None,
                           format: str = 'csv'):
        """
        Export audit logs for external analysis.
        
        Args:
            output_path: Path to save the export
            start_time: Export start time
            end_time: Export end time
            format: Export format (csv, json, parquet)
        """
        builder = AuditQueryBuilder(self.session)
        df = builder.time_range(start_time, end_time).to_dataframe()
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


def generate_audit_summary(days: int = 30) -> str:
    """
    Generate a human-readable audit summary.
    
    Args:
        days: Number of days to include in summary
        
    Returns:
        Formatted summary text
    """
    reporter = AuditReporter()
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    # Get various reports
    security = reporter.security_summary(start_time, end_time)
    trading = reporter.trading_activity(start_time, end_time)
    risk = reporter.risk_events(start_time, end_time)
    anomalies = reporter.detect_anomalies(lookback_days=days)
    
    # Format summary
    lines = [
        f"=== Audit Summary for Last {days} Days ===",
        f"Period: {start_time.date()} to {end_time.date()}",
        "",
        "Security Summary:",
        f"  - Successful logins: {security['authentication']['successful_logins']}",
        f"  - Failed logins: {security['authentication']['failed_logins']}",
        f"  - Unique users: {security['authentication']['unique_users']}",
        f"  - Secret accesses: {security['secret_access_count']}",
        "",
        "Trading Activity:",
        f"  - Total decisions: {trading['summary']['total_decisions']}",
        f"  - Successful trades: {trading['summary']['total_executions']}",
        f"  - Failed trades: {trading['summary']['total_failures']}",
        f"  - Success rate: {trading['summary']['success_rate']:.1%}",
        "",
        "Risk Events:",
        f"  - Total events: {risk['summary']['total_events']}",
        f"  - Risk overrides: {risk['summary']['total_overrides']}",
        ""
    ]
    
    if anomalies:
        lines.extend([
            "Detected Anomalies:",
            f"  - Total anomalies: {len(anomalies)}"
        ])
        
        for anomaly in anomalies[:5]:  # Show first 5
            lines.append(f"  - {anomaly['type']}: {anomaly.get('severity', 'unknown')} severity")
            
    return "\n".join(lines)