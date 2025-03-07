"""
Email notification channel for the alerting system.
"""
import aiosmtplib
from email.message import EmailMessage
import logging
from typing import Any, Dict, List, Optional

from ..models import Alert
from .base import NotificationChannel


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Email channel configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger("alpha_pulse.alerting.email")
        self.smtp_client: Optional[aiosmtplib.SMTP] = None
    
    async def initialize(self) -> bool:
        """Initialize the email client.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            smtp_server = self.config.get("smtp_server")
            smtp_port = self.config.get("smtp_port", 587)
            
            if not smtp_server:
                self.logger.error("SMTP server not configured")
                return False
            
            self.smtp_client = aiosmtplib.SMTP(
                hostname=smtp_server,
                port=smtp_port,
                use_tls=self.config.get("use_tls", True)
            )
            
            self.initialized = True
            self.logger.info(f"Email notification channel initialized: {smtp_server}:{smtp_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize email channel: {str(e)}")
            return False
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification for an alert.
        
        Args:
            alert: The alert to send notification for
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.initialized or not self.smtp_client:
            self.logger.warning("Email channel not initialized")
            return False
        
        try:
            # Create email message
            message = EmailMessage()
            message["From"] = self.config.get("from_address")
            
            # Get recipients
            to_addresses = self.config.get("to_addresses", [])
            if not to_addresses:
                self.logger.error("No recipients configured for email notification")
                return False
                
            message["To"] = ", ".join(to_addresses)
            message["Subject"] = self.get_notification_title(alert)
            message.set_content(self.get_notification_body(alert))
            
            # Connect to SMTP server
            await self.smtp_client.connect()
            
            # Login if credentials provided
            username = self.config.get("smtp_user")
            password = self.config.get("smtp_password")
            if username and password:
                await self.smtp_client.login(username, password)
            
            # Send message
            await self.smtp_client.send_message(message)
            
            # Disconnect
            await self.smtp_client.quit()
            
            self.logger.info(f"Sent email notification for alert: {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close SMTP connection."""
        if self.smtp_client:
            try:
                await self.smtp_client.quit()
            except Exception:
                pass
            self.smtp_client = None