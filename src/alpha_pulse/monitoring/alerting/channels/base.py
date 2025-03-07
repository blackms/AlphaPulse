"""
Base notification channel interface for the alerting system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..models import Alert


class NotificationChannel(ABC):
    """Base interface for notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the channel with configuration.
        
        Args:
            config: Channel-specific configuration
        """
        self.config = config
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the channel.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert.
        
        Args:
            alert: The alert to send notification for
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the channel and release resources."""
        pass
    
    def get_notification_title(self, alert: Alert) -> str:
        """Get the title for a notification.
        
        Args:
            alert: The alert to generate title for
            
        Returns:
            str: Notification title
        """
        severity = alert.severity.value.upper()
        return f"{severity} ALERT: {alert.metric_name}"
    
    def get_notification_body(self, alert: Alert) -> str:
        """Get the body content for a notification.
        
        Args:
            alert: The alert to generate body for
            
        Returns:
            str: Notification body
        """
        return (
            f"Message: {alert.message}\n"
            f"Metric: {alert.metric_name}\n"
            f"Value: {alert.metric_value}\n"
            f"Severity: {alert.severity.value}\n"
            f"Time: {alert.timestamp.isoformat()}"
        )