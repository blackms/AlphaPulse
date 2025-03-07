"""
Web notification channel for the alerting system.

This channel is used to send alerts to web clients via WebSockets.
"""
import logging
from typing import List, Callable, Coroutine, Any

from ..models import Alert
from .base import NotificationChannel


class WebNotificationChannel(NotificationChannel):
    """
    Web notification channel.
    
    This channel sends alerts to web clients via WebSockets.
    It doesn't actually send the alerts directly, but instead
    notifies registered handlers that can then broadcast the alerts
    to connected WebSocket clients.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the web notification channel.
        
        Args:
            config: Channel configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger("alpha_pulse.alerting.channels.web")
        self.handlers: List[Callable[[Alert], Coroutine[Any, Any, None]]] = []
    
    async def initialize(self) -> bool:
        """
        Initialize the channel.
        
        Returns:
            bool: True if initialization was successful
        """
        self.logger.info("Initialized web notification channel")
        return True
    
    async def close(self) -> None:
        """Close the channel."""
        self.logger.info("Closed web notification channel")
    
    async def send_notification(self, alert: Alert) -> bool:
        """
        Send a notification.
        
        This method notifies all registered handlers about the alert.
        
        Args:
            alert: The alert to send
            
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            # Notify all handlers
            for handler in self.handlers:
                await handler(alert)
            
            self.logger.debug(f"Notified {len(self.handlers)} handlers about alert: {alert.alert_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send web notification: {str(e)}")
            return False
    
    def add_handler(self, handler: Callable[[Alert], Coroutine[Any, Any, None]]) -> None:
        """
        Add a notification handler.
        
        Args:
            handler: The handler function to add
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
            self.logger.debug("Added notification handler")
    
    def remove_handler(self, handler: Callable[[Alert], Coroutine[Any, Any, None]]) -> None:
        """
        Remove a notification handler.
        
        Args:
            handler: The handler function to remove
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            self.logger.debug("Removed notification handler")