"""
Notification channels for the alerting system.

This module provides various notification channels for sending alerts,
including email, SMS, Slack, and web notifications.
"""

from .base import NotificationChannel
from .email import EmailNotificationChannel
from .slack import SlackNotificationChannel
from .web import WebNotificationChannel

__all__ = [
    'NotificationChannel',
    'EmailNotificationChannel',
    'SlackNotificationChannel',
    'WebNotificationChannel'
]