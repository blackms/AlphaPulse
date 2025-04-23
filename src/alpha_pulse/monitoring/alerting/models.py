"""
Data models for the alerting system.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertRule:
    """Rule definition for generating alerts."""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        metric_name: str,
        condition: str,
        severity: AlertSeverity,
        message_template: str,
        channels: List[str],
        cooldown_period: int = 3600,
        enabled: bool = True
    ):
        """Initialize an alert rule.
        
        Args:
            rule_id: Unique identifier for the rule
            name: Human-readable name
            description: Description of the rule
            metric_name: Name of the metric to monitor
            condition: Condition expression (e.g., "> 0.8", "< 100")
            severity: Severity level
            message_template: Template for alert message with {value} placeholder
            channels: List of notification channel names to use
            cooldown_period: Minimum time between alerts in seconds (default: 1 hour)
            enabled: Whether the rule is active
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.metric_name = metric_name
        self.condition = condition
        self.severity = severity if isinstance(severity, AlertSeverity) else AlertSeverity(severity)
        self.message_template = message_template
        self.channels = channels
        self.cooldown_period = cooldown_period
        self.enabled = enabled
        self.last_triggered_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "severity": self.severity.value,
            "message_template": self.message_template,
            "channels": self.channels,
            "cooldown_period": self.cooldown_period,
            "enabled": self.enabled,
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertRule":
        """Create a rule from a dictionary."""
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data["description"],
            metric_name=data["metric_name"],
            condition=data["condition"],
            severity=data["severity"],
            message_template=data["message_template"],
            channels=data["channels"],
            cooldown_period=data.get("cooldown_period", 3600),
            enabled=data.get("enabled", True)
        )


class Alert:
    """Alert model for triggered alerts."""
    
    def __init__(
        self,
        rule_id: str,
        metric_name: str,
        metric_value: Any,
        severity: AlertSeverity,
        message: str,
        alert_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        acknowledged: bool = False,
        acknowledged_by: Optional[str] = None,
        acknowledged_at: Optional[datetime] = None
    ):
        """Initialize an alert.
        
        Args:
            rule_id: ID of the rule that triggered the alert
            metric_name: Name of the metric that triggered the alert
            metric_value: Value of the metric
            severity: Severity level
            message: Alert message
            alert_id: Unique identifier (generated if not provided)
            timestamp: When the alert was generated (now if not provided)
            acknowledged: Whether the alert has been acknowledged
            acknowledged_by: Who acknowledged the alert
            acknowledged_at: When the alert was acknowledged
        """
        self.alert_id = alert_id or str(uuid.uuid4())
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.severity = severity if isinstance(severity, AlertSeverity) else AlertSeverity(severity)
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.acknowledged = acknowledged
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = acknowledged_at
    
    def acknowledge(self, user: str) -> None:
        """Acknowledge this alert.
        
        Args:
            user: User who acknowledged the alert
        """
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary."""
        return {
            "id": self.id,  # Add integer ID
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Create an alert from a dictionary."""
        return cls(
            id=data.get("id"),  # Get integer ID
            alert_id=data["alert_id"],
            rule_id=data["rule_id"],
            metric_name=data["metric_name"],
            metric_value=data["metric_value"],
            severity=data["severity"],
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            acknowledged=data["acknowledged"],
            acknowledged_by=data.get("acknowledged_by"),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None
        )