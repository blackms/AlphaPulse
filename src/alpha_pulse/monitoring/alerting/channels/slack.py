"""
Slack notification channel for the alerting system.
"""
import json
import logging
from typing import Any, Dict, Optional
import aiohttp

from ..models import Alert, AlertSeverity
from .base import NotificationChannel


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Slack channel configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger("alpha_pulse.alerting.slack")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> bool:
        """Initialize the Slack client.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            webhook_url = self.config.get("webhook_url")
            
            if not webhook_url:
                self.logger.error("Slack webhook URL not configured")
                return False
            
            self.session = aiohttp.ClientSession()
            self.initialized = True
            self.logger.info("Slack notification channel initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Slack channel: {str(e)}")
            return False
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification for an alert.
        
        Args:
            alert: The alert to send notification for
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.initialized or not self.session:
            self.logger.warning("Slack channel not initialized")
            return False
        
        try:
            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                self.logger.error("Slack webhook URL not configured")
                return False
            
            # Create Slack message payload
            payload = self._create_slack_payload(alert)
            
            # Send message to webhook
            async with self.session.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    self.logger.info(f"Sent Slack notification for alert: {alert.alert_id}")
                    return True
                else:
                    response_text = await response.text()
                    self.logger.error(
                        f"Failed to send Slack notification. Status: {response.status}, "
                        f"Response: {response_text}"
                    )
                    return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _create_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """Create Slack message payload.
        
        Args:
            alert: The alert to create payload for
            
        Returns:
            Dict[str, Any]: Slack message payload
        """
        # Determine color based on severity
        color = self._get_severity_color(alert.severity)
        
        # Create attachment
        attachment = {
            "color": color,
            "title": self.get_notification_title(alert),
            "text": alert.message,
            "fields": [
                {
                    "title": "Metric",
                    "value": alert.metric_name,
                    "short": True
                },
                {
                    "title": "Value",
                    "value": str(alert.metric_value),
                    "short": True
                },
                {
                    "title": "Severity",
                    "value": alert.severity.value.upper(),
                    "short": True
                },
                {
                    "title": "Time",
                    "value": alert.timestamp.isoformat(),
                    "short": True
                }
            ],
            "footer": "AlphaPulse Alerting System",
            "ts": int(alert.timestamp.timestamp())
        }
        
        # Create payload
        payload = {
            "username": self.config.get("username", "AlphaPulse Alerting"),
            "channel": self.config.get("channel", "#alerts"),
            "attachments": [attachment]
        }
        
        return payload
    
    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Get Slack color for severity level.
        
        Args:
            severity: Alert severity
            
        Returns:
            str: Slack color code
        """
        if severity == AlertSeverity.INFO:
            return "#2196F3"  # Blue
        elif severity == AlertSeverity.WARNING:
            return "#FFC107"  # Yellow
        elif severity == AlertSeverity.ERROR:
            return "#FF5722"  # Orange
        elif severity == AlertSeverity.CRITICAL:
            return "#F44336"  # Red
        else:
            return "#9E9E9E"  # Grey