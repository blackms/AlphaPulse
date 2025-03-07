"""
Unit tests for the SMS notification channel.
"""
import unittest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime

from alpha_pulse.monitoring.alerting.models import Alert, AlertSeverity
from alpha_pulse.monitoring.alerting.channels.sms import SMSNotificationChannel


class TestSMSNotificationChannel(unittest.TestCase):
    """Test cases for the SMS notification channel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "account_sid": "test_sid",
            "auth_token": "test_token",
            "from_number": "+15551234567",
            "to_numbers": ["+15557654321", "+15559876543"]
        }
        self.channel = SMSNotificationChannel(self.config)
        
        # Create a test alert
        self.alert = Alert(
            rule_id="test_rule",
            metric_name="test_metric",
            metric_value=0.75,
            severity=AlertSeverity.ERROR,
            message="Test alert message"
        )
    
    def test_create_sms_content(self):
        """Test creating SMS content from an alert."""
        content = self.channel._create_sms_content(self.alert)
        
        # Check that content includes severity, message, and metric info
        self.assertIn("ERROR ALERT", content)
        self.assertIn("Test alert message", content)
        self.assertIn("test_metric", content)
        self.assertIn("0.75", content)
    
    @patch('aiohttp.ClientSession')
    def test_initialize(self, mock_session):
        """Test channel initialization."""
        # Run the coroutine
        result = asyncio.run(self.channel.initialize())
        
        # Check that initialization was successful
        self.assertTrue(result)
        self.assertTrue(self.channel.initialized)
        
        # Check that session was created
        self.assertIsNotNone(self.channel.session)
    
    @patch('aiohttp.ClientSession')
    def test_initialize_missing_config(self, mock_session):
        """Test initialization with missing configuration."""
        # Create channel with missing config
        channel = SMSNotificationChannel({})
        
        # Run the coroutine
        result = asyncio.run(channel.initialize())
        
        # Check that initialization failed
        self.assertFalse(result)
        self.assertFalse(channel.initialized)
    
    @patch('aiohttp.ClientSession.post')
    async def test_send_sms(self, mock_post):
        """Test sending SMS to a recipient."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json.return_value = {"sid": "test_message_sid"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Initialize the channel
        self.channel.session = MagicMock()
        self.channel.initialized = True
        
        # Call the method
        result = await self.channel._send_sms("+15557654321", "Test message")
        
        # Check that the method returned success
        self.assertTrue(result)
        
        # Check that post was called with correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], f"https://api.twilio.com/2010-04-01/Accounts/test_sid/Messages.json")
        self.assertEqual(kwargs["data"]["From"], "+15551234567")
        self.assertEqual(kwargs["data"]["To"], "+15557654321")
        self.assertEqual(kwargs["data"]["Body"], "Test message")
    
    @patch('aiohttp.ClientSession.post')
    async def test_send_notification(self, mock_post):
        """Test sending notification to all recipients."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json.return_value = {"sid": "test_message_sid"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Initialize the channel
        self.channel.session = MagicMock()
        self.channel.initialized = True
        
        # Call the method
        result = await self.channel.send_notification(self.alert)
        
        # Check that the method returned success
        self.assertTrue(result)
        
        # Check that post was called twice (once for each recipient)
        self.assertEqual(mock_post.call_count, 2)
    
    @patch('aiohttp.ClientSession.post')
    async def test_send_notification_failure(self, mock_post):
        """Test handling of send failure."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text.return_value = "Error"
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Initialize the channel
        self.channel.session = MagicMock()
        self.channel.initialized = True
        
        # Call the method
        result = await self.channel.send_notification(self.alert)
        
        # Check that the method returned failure
        self.assertFalse(result)
    
    async def test_close(self):
        """Test closing the channel."""
        # Create a mock session
        mock_session = MagicMock()
        self.channel.session = mock_session
        
        # Call the method
        await self.channel.close()
        
        # Check that session.close was called
        mock_session.close.assert_called_once()
        self.assertIsNone(self.channel.session)


if __name__ == '__main__':
    unittest.main()