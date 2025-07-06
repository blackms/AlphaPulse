"""
AlertManager implementation for the alerting system.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
import asyncio
import logging
import importlib

from .models import Alert, AlertRule, AlertSeverity
from .evaluator import RuleEvaluator
from .history import AlertHistoryStorage, create_alert_history
from .channels.base import NotificationChannel
from .config import AlertingConfig


class AlertManager:
    """Central component that coordinates alerting activities."""
    
    def __init__(self, config: AlertingConfig):
        """Initialize with configuration.
        
        Args:
            config: Alerting system configuration
        """
        self.config = config
        self.logger = logging.getLogger("alpha_pulse.alerting")
        self.channels: Dict[str, NotificationChannel] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.evaluator = RuleEvaluator()
        self.running = False
        self.check_task = None
        
        # Initialize alert history storage
        history_config = getattr(config, "history", {"type": "memory"})
        self.alert_history = create_alert_history(history_config)
        
        # Load rules from configuration
        for rule in config.rules:
            self.rules[rule.rule_id] = rule
        
        # Initialize notification channels
        self._initialize_channels()
    
    def _initialize_channels(self) -> None:
        """Initialize notification channels from configuration."""
        for channel_name, channel_config in self.config.channels.items():
            try:
                # Determine channel type
                channel_type = channel_config.get("type", channel_name)
                
                # Import the appropriate channel class
                if channel_type == "email":
                    from .channels.email import EmailNotificationChannel
                    channel_class = EmailNotificationChannel
                elif channel_type == "slack":
                    from .channels.slack import SlackNotificationChannel
                    channel_class = SlackNotificationChannel
                elif channel_type == "sms":
                    from .channels.sms import SMSNotificationChannel
                    channel_class = SMSNotificationChannel
                elif channel_type == "web":
                    from .channels.web import WebNotificationChannel
                    channel_class = WebNotificationChannel
                else:
                    # Try to dynamically import custom channel
                    try:
                        module_path = f".channels.{channel_type}"
                        module = importlib.import_module(module_path, package="alpha_pulse.monitoring.alerting")
                        channel_class = getattr(module, f"{channel_type.capitalize()}NotificationChannel")
                    except (ImportError, AttributeError):
                        self.logger.error(f"Unsupported notification channel type: {channel_type}")
                        continue
                
                # Create channel instance
                channel = channel_class(channel_config)
                self.channels[channel_name] = channel
                self.logger.info(f"Initialized notification channel: {channel_name} ({channel_type})")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize channel {channel_name}: {str(e)}")
    
    async def start(self) -> None:
        """Start the alert manager."""
        if self.running:
            return
            
        self.logger.info("Starting AlertManager")
        
        # Initialize notification channels
        for channel_name, channel_instance in self.channels.items():
            try:
                success = await channel_instance.initialize()
                if not success:
                    self.logger.warning(f"Failed to initialize notification channel: {channel_name}")
            except Exception as e:
                self.logger.error(f"Error initializing channel {channel_name}: {str(e)}")
        
        # Start periodic check task if enabled
        if self.config.enabled:
            self.running = True
            self.check_task = asyncio.create_task(self._check_loop())
            self.logger.info(f"Started alert check loop (interval: {self.config.check_interval}s)")
        
        self.logger.info("AlertManager started")
    
    async def stop(self) -> None:
        """Stop the alert manager."""
        if not self.running:
            return
            
        self.logger.info("Stopping AlertManager")
        
        # Stop check task
        self.running = False
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
            self.check_task = None
        
        # Close notification channels
        for channel_name, channel_instance in self.channels.items():
            try:
                await channel_instance.close()
            except Exception as e:
                self.logger.error(f"Error closing channel {channel_name}: {str(e)}")
        
        self.logger.info("AlertManager stopped")
    
    def register_channel(self, name: str, channel: NotificationChannel) -> None:
        """Register a notification channel.
        
        Args:
            name: Name of the channel
            channel: The notification channel instance
        """
        self.channels[name] = channel
        self.logger.info(f"Registered notification channel: {name}")
    
    async def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule.
        
        Args:
            rule: The rule to add
        """
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added rule: {rule.name} ({rule.rule_id})")
    
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule.
        
        Args:
            rule_id: ID of the rule to remove
            
        Returns:
            bool: True if rule was removed, False if rule not found
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed rule: {rule_id}")
            return True
        return False
    
    async def process_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Process metrics and generate alerts if needed.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            List[Alert]: List of triggered alerts
        """
        if not self.running:
            self.logger.warning("AlertManager not running, metrics not processed")
            return []
            
        triggered_alerts: List[Alert] = []
        
        # Check each rule against applicable metrics
        for rule_id, rule in self.rules.items():
            if rule.metric_name in metrics:
                metric_value = metrics[rule.metric_name]
                alert = self.evaluator.evaluate(rule, metric_value)
                
                if alert:
                    self.logger.info(f"Rule triggered: {rule.name} for metric {rule.metric_name}")
                    
                    # Add to alert history
                    await self.alert_history.store_alert(alert)
                    
                    # Send notifications
                    await self._send_notifications(alert, rule.channels)
                    
                    # Add to triggered alerts
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: User acknowledging the alert
            
        Returns:
            bool: True if alert was acknowledged, False if alert not found
        """
        # Update alert in history
        return await self.alert_history.update_alert(
            alert_id, 
            {
                "acknowledged": True,
                "acknowledged_by": user,
                "acknowledged_at": datetime.now()
            }
        )
    
    async def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """Get alert history with optional filtering.
        
        Args:
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            filters: Additional filters (severity, acknowledged, etc.)
            
        Returns:
            List[Alert]: Filtered alert history
        """
        return await self.alert_history.get_alerts(start_time, end_time, filters)
    
    async def _send_notifications(self, alert: Alert, channel_names: List[str]) -> None:
        """Send notifications for an alert through specified channels.
        
        Args:
            alert: The alert to send notifications for
            channel_names: List of channel names to use
        """
        for channel_name in channel_names:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                try:
                    success = await channel.send_notification(alert)
                    if not success:
                        self.logger.warning(f"Failed to send notification through channel: {channel_name}")
                except Exception as e:
                    self.logger.error(f"Error sending notification through {channel_name}: {str(e)}")
            else:
                self.logger.warning(f"Notification channel not found: {channel_name}")
    
    async def _check_loop(self) -> None:
        """Background task for periodic alert checking."""
        while self.running:
            try:
                # This is a placeholder - in the actual implementation,
                # this would check metrics from the metrics collector
                # For now, we'll just log that we're checking
                self.logger.debug("Alert check loop running")
                
                # Sleep until next check
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert check loop: {str(e)}")
                await asyncio.sleep(5)  # Short sleep on error
    
    def get_channel(self, name: str) -> Optional[NotificationChannel]:
        """Get a notification channel by name.
        
        Args:
            name: Name of the channel
            
        Returns:
            Optional[NotificationChannel]: The channel if found, None otherwise
        """
        return self.channels.get(name)
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get a rule by ID.
        
        Args:
            rule_id: ID of the rule
            
        Returns:
            Optional[AlertRule]: The rule if found, None otherwise
        """
        return self.rules.get(rule_id)
    
    def get_rules(self) -> List[AlertRule]:
        """Get all rules.
        
        Returns:
            List[AlertRule]: List of all rules
        """
        return list(self.rules.values())
    
    async def send_quality_alert(
        self, 
        symbol: str, 
        metric_type: str, 
        severity: str, 
        current_value: float, 
        threshold_value: float, 
        description: str,
        suggested_action: str = None,
        channels: List[str] = None
    ) -> Alert:
        """
        Send a data quality alert directly.
        
        Args:
            symbol: Trading symbol
            metric_type: Type of quality metric
            severity: Alert severity (critical, warning, info)
            current_value: Current metric value
            threshold_value: Violated threshold value
            description: Alert description
            suggested_action: Suggested remediation action
            channels: List of channels to send to (uses defaults if None)
            
        Returns:
            Alert: The created alert
        """
        from uuid import uuid4
        
        # Map string severity to AlertSeverity enum
        severity_map = {
            "critical": AlertSeverity.CRITICAL,
            "emergency": AlertSeverity.CRITICAL,
            "warning": AlertSeverity.WARNING,
            "info": AlertSeverity.INFO
        }
        
        alert_severity = severity_map.get(severity.lower(), AlertSeverity.WARNING)
        
        # Create alert
        alert = Alert(
            alert_id=str(uuid4()),
            rule_id=f"data_quality_{metric_type}",
            rule_name=f"Data Quality: {metric_type}",
            metric_name=f"quality.{metric_type}",
            metric_value=current_value,
            threshold=threshold_value,
            severity=alert_severity,
            message=description,
            timestamp=datetime.now(),
            acknowledged=False,
            additional_data={
                "symbol": symbol,
                "metric_type": metric_type,
                "suggested_action": suggested_action or "Review data quality",
                "source": "data_quality_system"
            }
        )
        
        # Add to alert history
        await self.alert_history.store_alert(alert)
        
        # Send notifications
        notification_channels = channels or ["web", "email"]  # Default channels
        await self._send_notifications(alert, notification_channels)
        
        self.logger.info(
            f"Data quality alert sent: {symbol} {metric_type} {severity} "
            f"(value: {current_value:.3f}, threshold: {threshold_value:.3f})"
        )
        
        return alert