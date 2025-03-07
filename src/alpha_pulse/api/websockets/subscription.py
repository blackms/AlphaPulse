"""
WebSocket subscription manager.

This module provides a subscription manager for WebSocket connections.
"""
import logging
import asyncio
import json
from typing import Dict, Any, Set, List, Callable, Coroutine
from datetime import datetime

from alpha_pulse.monitoring.alerting import AlertManager
from alpha_pulse.monitoring.alerting.models import Alert

from .manager import connection_manager

logger = logging.getLogger(__name__)


class AlertsSubscription:
    """
    Subscription for alert updates.
    
    This class handles alert notifications from the alerting system
    and broadcasts them to WebSocket clients.
    """
    
    def __init__(self, alert_manager: AlertManager):
        """
        Initialize the alerts subscription.
        
        Args:
            alert_manager: The alerting system manager
        """
        self.alert_manager = alert_manager
        self.notification_handlers = []
    
    async def start(self):
        """Start listening for alert notifications."""
        # Register a notification handler with the web channel
        web_channel = self.alert_manager.get_channel("web")
        if web_channel:
            web_channel.add_handler(self._handle_alert)
            logger.info("Registered alert notification handler with web channel")
        else:
            logger.warning("Web notification channel not found, alerts won't be sent to WebSocket clients")
    
    async def stop(self):
        """Stop listening for alert notifications."""
        # Unregister the notification handler
        web_channel = self.alert_manager.get_channel("web")
        if web_channel:
            web_channel.remove_handler(self._handle_alert)
            logger.info("Unregistered alert notification handler from web channel")
    
    async def _handle_alert(self, alert: Alert):
        """
        Handle an alert notification.
        
        Args:
            alert: The alert that was triggered
        """
        try:
            # Format the alert for the WebSocket API
            alert_data = {
                "type": "alert",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "id": alert.alert_id,
                    "title": f"{alert.metric_name} Alert",
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "source": alert.metric_name,
                    "created_at": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "acknowledged_by": alert.acknowledged_by,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "tags": [alert.metric_name, alert.severity.value]
                }
            }
            
            # Broadcast to all subscribers
            await connection_manager.broadcast("alerts", alert_data)
            logger.debug(f"Broadcasted alert to WebSocket clients: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error handling alert notification: {e}")


class SubscriptionManager:
    """
    WebSocket subscription manager.
    
    This class manages subscriptions and broadcasts updates to subscribers.
    """
    
    def __init__(self):
        """Initialize the subscription manager."""
        self.running = False
        self.tasks = []
        self.alert_manager = None
        self.alerts_subscription = None
    
    def set_alert_manager(self, alert_manager: AlertManager):
        """
        Set the alert manager.
        
        Args:
            alert_manager: The alerting system manager
        """
        self.alert_manager = alert_manager
        self.alerts_subscription = AlertsSubscription(alert_manager)
    
    async def start(self):
        """Start the subscription manager."""
        if self.running:
            return
        
        self.running = True
        
        # Start the alerts subscription if available
        if self.alerts_subscription:
            await self.alerts_subscription.start()
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._metrics_updater()),
            asyncio.create_task(self._portfolio_updater()),
            asyncio.create_task(self._trades_updater())
        ]
        
        logger.info("Subscription manager started")
    
    async def stop(self):
        """Stop the subscription manager."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop the alerts subscription if available
        if self.alerts_subscription:
            await self.alerts_subscription.stop()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        
        logger.info("Subscription manager stopped")
    
    async def _metrics_updater(self):
        """Update metrics periodically."""
        try:
            while self.running:
                # In a real implementation, this would get actual metrics
                # For now, send demo data
                await connection_manager.broadcast("metrics", {
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "portfolio_value": {
                            "value": 1050000.0,
                            "timestamp": datetime.now().isoformat(),
                            "labels": {"currency": "USD"}
                        },
                        "sharpe_ratio": {
                            "value": 1.9,
                            "timestamp": datetime.now().isoformat(),
                            "labels": {"window": "30d"}
                        }
                    }
                })
                
                # Wait for next update
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Metrics updater cancelled")
        except Exception as e:
            logger.error(f"Error in metrics updater: {e}")
    
    async def _portfolio_updater(self):
        """Update portfolio periodically."""
        try:
            while self.running:
                # In a real implementation, this would get actual portfolio data
                # For now, send demo data
                await connection_manager.broadcast("portfolio", {
                    "type": "portfolio",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "total_value": 1250000.0,
                        "cash": 250000.0,
                        "positions_value": 1000000.0
                    }
                })
                
                # Wait for next update
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("Portfolio updater cancelled")
        except Exception as e:
            logger.error(f"Error in portfolio updater: {e}")
    
    async def _trades_updater(self):
        """Update trades periodically."""
        try:
            while self.running:
                # In a real implementation, this would get actual trade data
                # For now, send demo data
                await connection_manager.broadcast("trades", {
                    "type": "trade",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "id": 456,
                        "symbol": "BTC-USD",
                        "side": "buy",
                        "quantity": 0.5,
                        "price": 45000.0,
                        "executed_at": datetime.now().isoformat()
                    }
                })
                
                # Wait for next update (longer interval for trades)
                await asyncio.sleep(20)
        except asyncio.CancelledError:
            logger.info("Trades updater cancelled")
        except Exception as e:
            logger.error(f"Error in trades updater: {e}")


# Create a singleton instance
subscription_manager = SubscriptionManager()