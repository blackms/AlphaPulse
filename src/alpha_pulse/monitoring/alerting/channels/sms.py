"""
SMS notification channel for the alerting system.

This implementation uses Twilio as the SMS provider.
"""
import logging
from typing import Any, Dict, Optional
import aiohttp

from ..models import Alert
from .base import NotificationChannel


class SMSNotificationChannel(NotificationChannel):
    """SMS notification channel implementation using Twilio."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: SMS channel configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger("alpha_pulse.alerting.sms")
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Twilio configuration
        self.account_sid = config.get("account_sid")
        self.auth_token = config.get("auth_token")
        self.from_number = config.get("from_number")
        self.to_numbers = config.get("to_numbers", [])
        self.api_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
    
    async def initialize(self) -> bool:
        """Initialize the SMS client.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self.account_sid or not self.auth_token or not self.from_number:
                self.logger.error("Missing required Twilio configuration")
                return False
            
            if not self.to_numbers:
                self.logger.error("No recipient phone numbers configured")
                return False
            
            self.session = aiohttp.ClientSession()
            self.initialized = True
            self.logger.info("SMS notification channel initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SMS channel: {str(e)}")
            return False
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send SMS notification for an alert.
        
        Args:
            alert: The alert to send notification for
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.initialized or not self.session:
            self.logger.warning("SMS channel not initialized")
            return False
        
        try:
            # Create message content
            message = self._create_sms_content(alert)
            
            # Send to all recipients
            success = True
            for to_number in self.to_numbers:
                result = await self._send_sms(to_number, message)
                if not result:
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send SMS notification: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _create_sms_content(self, alert: Alert) -> str:
        """Create SMS message content.
        
        Args:
            alert: The alert to create content for
            
        Returns:
            str: SMS message content
        """
        # Keep SMS content concise due to length limitations
        severity = alert.severity.value.upper()
        return (
            f"{severity} ALERT: {alert.message}\n"
            f"Metric: {alert.metric_name} = {alert.metric_value}"
        )
    
    async def _send_sms(self, to_number: str, message: str) -> bool:
        """Send SMS to a single recipient.
        
        Args:
            to_number: Recipient phone number
            message: SMS content
            
        Returns:
            bool: True if SMS was sent successfully, False otherwise
        """
        if not self.session:
            return False
        
        try:
            # Prepare request data
            data = {
                "From": self.from_number,
                "To": to_number,
                "Body": message
            }
            
            # Send request to Twilio API
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            async with self.session.post(
                self.api_url,
                data=data,
                auth=auth
            ) as response:
                if response.status == 201:  # Twilio returns 201 Created on success
                    response_data = await response.json()
                    self.logger.info(
                        f"Sent SMS notification to {to_number} "
                        f"(SID: {response_data.get('sid')})"
                    )
                    return True
                else:
                    response_text = await response.text()
                    self.logger.error(
                        f"Failed to send SMS to {to_number}. "
                        f"Status: {response.status}, Response: {response_text}"
                    )
                    return False
                
        except Exception as e:
            self.logger.error(f"Error sending SMS to {to_number}: {str(e)}")
            return False