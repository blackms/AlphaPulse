"""
Web notification channel for the alerting system.

This channel stores alerts in memory for retrieval by the dashboard or API.
"""
import logging
from typing import Any, Dict, List, Optional
from collections import deque
from datetime import datetime

from ..models import Alert
from .base import NotificationChannel


class WebNotificationChannel(NotificationChannel):
    """Web notification channel implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Web channel configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger("alpha_pulse.alerting.web")
        self.max_alerts = config.get("max_alerts", 100)
        self.alerts: deque = deque(maxlen=self.max_alerts)
        self.subscribers: List[Any] = []  # Will hold WebSocket connections
    
    async def initialize(self) -> bool:
        """Initialize the web notification channel.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.initialized = True
            self.logger.info(f"Web notification channel initialized (max_alerts={self.max_alerts})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize web channel: {str(e)}")
            return False
    
    async def send_notification(self, alert: Alert) -> bool:
        """Store alert and notify subscribers.
        
        Args:
            alert: The alert to send notification for
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.initialized:
            self.logger.warning("Web channel not initialized")
            return False
        
        try:
            # Store alert
            self.alerts.append(alert)
            
            # Notify subscribers
            await self._notify_subscribers(alert)
            
            self.logger.info(f"Stored web notification for alert: {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send web notification: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close the channel."""
        self.subscribers.clear()
    
    def get_alerts(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        acknowledged: Optional[bool] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get stored alerts with optional filtering.
        
        Args:
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            acknowledged: Filter by acknowledgment status
            severity: Filter by severity
            limit: Maximum number of alerts to return
            
        Returns:
            List[Alert]: Filtered alerts
        """
        filtered_alerts = list(self.alerts)
        
        # Apply time filters
        if start_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= start_time]
        if end_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp <= end_time]
        
        # Apply acknowledgment filter
        if acknowledged is not None:
            filtered_alerts = [a for a in filtered_alerts if a.acknowledged == acknowledged]
        
        # Apply severity filter
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity.value == severity]
        
        # Sort by timestamp (newest first) and apply limit
        filtered_alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return filtered_alerts[:limit]
    
    def add_subscriber(self, subscriber: Any) -> None:
        """Add a subscriber for real-time notifications.
        
        Args:
            subscriber: WebSocket connection or other subscriber object
        """
        if subscriber not in self.subscribers:
            self.subscribers.append(subscriber)
            self.logger.info(f"Added web notification subscriber (total: {len(self.subscribers)})")
    
    def remove_subscriber(self, subscriber: Any) -> None:
        """Remove a subscriber.
        
        Args:
            subscriber: WebSocket connection or other subscriber object
        """
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)
            self.logger.info(f"Removed web notification subscriber (total: {len(self.subscribers)})")
    
    async def _notify_subscribers(self, alert: Alert) -> None:
        """Notify all subscribers about a new alert.
        
        Args:
            alert: The alert to notify about
        """
        if not self.subscribers:
            return
            
        # Convert alert to dict for JSON serialization
        alert_data = alert.to_dict()
        
        # Create notification message
        message = {
            "type": "alert",
            "data": alert_data
        }
        
        # Send to all subscribers
        for subscriber in list(self.subscribers):
            try:
                # This assumes subscribers have a send_json method (like WebSocket connections)
                if hasattr(subscriber, 'send_json'):
                    await subscriber.send_json(message)
                elif hasattr(subscriber, 'send'):
                    await subscriber.send(message)
            except Exception as e:
                self.logger.error(f"Failed to notify subscriber: {str(e)}")
                # Remove failed subscriber
                self.remove_subscriber(subscriber)