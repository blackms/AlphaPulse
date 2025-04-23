"""
Tests for the alerting system.
"""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from alpha_pulse.monitoring.alerting import (
    Alert, 
    AlertRule, 
    AlertSeverity, 
    AlertManager, 
    AlertingConfig
)
from alpha_pulse.monitoring.alerting.evaluator import RuleEvaluator
from alpha_pulse.monitoring.alerting.channels.base import NotificationChannel


class MockNotificationChannel(NotificationChannel):
    """Mock notification channel for testing."""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.notifications = []
        self.initialized = False
        self.closed = False
    
    async def initialize(self) -> bool:
        self.initialized = True
        return True
    
    async def send_notification(self, alert: Alert) -> bool:
        self.notifications.append(alert)
        return True
    
    async def close(self) -> None:
        self.closed = True


class TestAlertRule(unittest.TestCase):
    """Tests for the AlertRule class."""
    
    def test_init(self):
        """Test initialization."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric_name="test_metric",
            condition="> 10",
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}",
            channels=["email", "slack"],
            cooldown_period=3600,
            enabled=True
        )
        
        self.assertEqual(rule.rule_id, "test_rule")
        self.assertEqual(rule.name, "Test Rule")
        self.assertEqual(rule.description, "A test rule")
        self.assertEqual(rule.metric_name, "test_metric")
        self.assertEqual(rule.condition, "> 10")
        self.assertEqual(rule.severity, AlertSeverity.WARNING)
        self.assertEqual(rule.message_template, "Value is {value}")
        self.assertEqual(rule.channels, ["email", "slack"])
        self.assertEqual(rule.cooldown_period, 3600)
        self.assertTrue(rule.enabled)
        self.assertIsNone(rule.last_triggered_at)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric_name="test_metric",
            condition="> 10",
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}",
            channels=["email", "slack"],
            cooldown_period=3600,
            enabled=True
        )
        
        rule_dict = rule.to_dict()
        self.assertEqual(rule_dict["rule_id"], "test_rule")
        self.assertEqual(rule_dict["name"], "Test Rule")
        self.assertEqual(rule_dict["description"], "A test rule")
        self.assertEqual(rule_dict["metric_name"], "test_metric")
        self.assertEqual(rule_dict["condition"], "> 10")
        self.assertEqual(rule_dict["severity"], "warning")
        self.assertEqual(rule_dict["message_template"], "Value is {value}")
        self.assertEqual(rule_dict["channels"], ["email", "slack"])
        self.assertEqual(rule_dict["cooldown_period"], 3600)
        self.assertTrue(rule_dict["enabled"])
        self.assertIsNone(rule_dict["last_triggered_at"])
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        rule_dict = {
            "rule_id": "test_rule",
            "name": "Test Rule",
            "description": "A test rule",
            "metric_name": "test_metric",
            "condition": "> 10",
            "severity": "warning",
            "message_template": "Value is {value}",
            "channels": ["email", "slack"],
            "cooldown_period": 3600,
            "enabled": True
        }
        
        rule = AlertRule.from_dict(rule_dict)
        self.assertEqual(rule.rule_id, "test_rule")
        self.assertEqual(rule.name, "Test Rule")
        self.assertEqual(rule.description, "A test rule")
        self.assertEqual(rule.metric_name, "test_metric")
        self.assertEqual(rule.condition, "> 10")
        self.assertEqual(rule.severity, AlertSeverity.WARNING)
        self.assertEqual(rule.message_template, "Value is {value}")
        self.assertEqual(rule.channels, ["email", "slack"])
        self.assertEqual(rule.cooldown_period, 3600)
        self.assertTrue(rule.enabled)


class TestAlert(unittest.TestCase):
    """Tests for the Alert class."""
    
    def test_init(self):
        """Test initialization."""
        alert = Alert(
            rule_id="test_rule",
            metric_name="test_metric",
            metric_value=15,
            severity=AlertSeverity.WARNING,
            message="Value is 15",
            alert_id="test_alert",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            acknowledged=False,
            acknowledged_by=None,
            acknowledged_at=None
        )
        
        self.assertEqual(alert.alert_id, "test_alert")
        self.assertEqual(alert.rule_id, "test_rule")
        self.assertEqual(alert.metric_name, "test_metric")
        self.assertEqual(alert.metric_value, 15)
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertEqual(alert.message, "Value is 15")
        self.assertEqual(alert.timestamp, datetime(2025, 1, 1, 12, 0, 0))
        self.assertFalse(alert.acknowledged)
        self.assertIsNone(alert.acknowledged_by)
        self.assertIsNone(alert.acknowledged_at)
    
    def test_acknowledge(self):
        """Test alert acknowledgment."""
        alert = Alert(
            rule_id="test_rule",
            metric_name="test_metric",
            metric_value=15,
            severity=AlertSeverity.WARNING,
            message="Value is 15"
        )
        
        self.assertFalse(alert.acknowledged)
        self.assertIsNone(alert.acknowledged_by)
        self.assertIsNone(alert.acknowledged_at)
        
        alert.acknowledge("test_user")
        
        self.assertTrue(alert.acknowledged)
        self.assertEqual(alert.acknowledged_by, "test_user")
        self.assertIsNotNone(alert.acknowledged_at)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        alert = Alert(
            rule_id="test_rule",
            metric_name="test_metric",
            metric_value=15,
            severity=AlertSeverity.WARNING,
            message="Value is 15",
            alert_id="test_alert",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            acknowledged=False,
            acknowledged_by=None,
            acknowledged_at=None
        )
        
        alert_dict = alert.to_dict()
        self.assertEqual(alert_dict["alert_id"], "test_alert")
        self.assertEqual(alert_dict["rule_id"], "test_rule")
        self.assertEqual(alert_dict["metric_name"], "test_metric")
        self.assertEqual(alert_dict["metric_value"], 15)
        self.assertEqual(alert_dict["severity"], "warning")
        self.assertEqual(alert_dict["message"], "Value is 15")
        self.assertEqual(alert_dict["timestamp"], "2025-01-01T12:00:00")
        self.assertFalse(alert_dict["acknowledged"])
        self.assertIsNone(alert_dict["acknowledged_by"])
        self.assertIsNone(alert_dict["acknowledged_at"])


class TestRuleEvaluator(unittest.TestCase):
    """Tests for the RuleEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RuleEvaluator()
    
    def test_parse_condition_greater_than(self):
        """Test parsing '>' condition."""
        condition_func = self.evaluator.parse_condition("> 10")
        self.assertTrue(condition_func(15))
        self.assertFalse(condition_func(5))
        self.assertFalse(condition_func(10))
    
    def test_parse_condition_greater_equal(self):
        """Test parsing '>=' condition."""
        condition_func = self.evaluator.parse_condition(">= 10")
        self.assertTrue(condition_func(15))
        self.assertTrue(condition_func(10))
        self.assertFalse(condition_func(5))
    
    def test_parse_condition_less_than(self):
        """Test parsing '<' condition."""
        condition_func = self.evaluator.parse_condition("< 10")
        self.assertTrue(condition_func(5))
        self.assertFalse(condition_func(15))
        self.assertFalse(condition_func(10))
    
    def test_parse_condition_less_equal(self):
        """Test parsing '<=' condition."""
        condition_func = self.evaluator.parse_condition("<= 10")
        self.assertTrue(condition_func(5))
        self.assertTrue(condition_func(10))
        self.assertFalse(condition_func(15))
    
    def test_parse_condition_equal(self):
        """Test parsing '==' condition."""
        condition_func = self.evaluator.parse_condition("== 10")
        self.assertTrue(condition_func(10))
        self.assertFalse(condition_func(5))
        self.assertFalse(condition_func(15))
    
    def test_parse_condition_not_equal(self):
        """Test parsing '!=' condition."""
        condition_func = self.evaluator.parse_condition("!= 10")
        self.assertTrue(condition_func(5))
        self.assertTrue(condition_func(15))
        self.assertFalse(condition_func(10))
    
    def test_parse_condition_invalid(self):
        """Test parsing invalid condition."""
        with self.assertRaises(ValueError):
            self.evaluator.parse_condition("invalid")
    
    def test_evaluate_rule_triggered(self):
        """Test rule evaluation that triggers an alert."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric_name="test_metric",
            condition="> 10",
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}",
            channels=["email", "slack"],
            cooldown_period=3600,
            enabled=True
        )
        
        alert = self.evaluator.evaluate(rule, 15)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.rule_id, "test_rule")
        self.assertEqual(alert.metric_name, "test_metric")
        self.assertEqual(alert.metric_value, 15)
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertEqual(alert.message, "Value is 15")
    
    def test_evaluate_rule_not_triggered(self):
        """Test rule evaluation that doesn't trigger an alert."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric_name="test_metric",
            condition="> 10",
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}",
            channels=["email", "slack"],
            cooldown_period=3600,
            enabled=True
        )
        
        alert = self.evaluator.evaluate(rule, 5)
        self.assertIsNone(alert)
    
    def test_evaluate_rule_disabled(self):
        """Test evaluation of disabled rule."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric_name="test_metric",
            condition="> 10",
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}",
            channels=["email", "slack"],
            cooldown_period=3600,
            enabled=False
        )
        
        alert = self.evaluator.evaluate(rule, 15)
        self.assertIsNone(alert)
    
    def test_evaluate_rule_cooldown(self):
        """Test rule cooldown period."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric_name="test_metric",
            condition="> 10",
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}",
            channels=["email", "slack"],
            cooldown_period=3600,
            enabled=True
        )
        
        # First evaluation should trigger
        alert1 = self.evaluator.evaluate(rule, 15)
        self.assertIsNotNone(alert1)
        
        # Second evaluation should not trigger due to cooldown
        alert2 = self.evaluator.evaluate(rule, 15)
        self.assertIsNone(alert2)
        
        # Set last_triggered_at to more than cooldown period ago
        rule.last_triggered_at = datetime.now() - timedelta(seconds=3601)
        
        # Third evaluation should trigger again
        alert3 = self.evaluator.evaluate(rule, 15)
        self.assertIsNotNone(alert3)


class TestAlertManager(unittest.IsolatedAsyncioTestCase):
    """Tests for the AlertManager class."""
    
    async def asyncSetUp(self):
        """Set up test fixtures."""
        # Create mock channels
        self.email_channel = MockNotificationChannel()
        self.slack_channel = MockNotificationChannel()
        self.web_channel = MockNotificationChannel()
        
        # Create rules
        self.rules = [
            AlertRule(
                rule_id="test_rule_1",
                name="Test Rule 1",
                description="A test rule",
                metric_name="metric_1",
                condition="> 10",
                severity=AlertSeverity.WARNING,
                message_template="Metric 1 is {value}",
                channels=["email", "slack"],
                cooldown_period=0,  # No cooldown for testing
                enabled=True
            ),
            AlertRule(
                rule_id="test_rule_2",
                name="Test Rule 2",
                description="Another test rule",
                metric_name="metric_2",
                condition="< 5",
                severity=AlertSeverity.ERROR,
                message_template="Metric 2 is {value}",
                channels=["slack", "web"],
                cooldown_period=0,  # No cooldown for testing
                enabled=True
            )
        ]
        
        # Create config
        self.config = AlertingConfig({
            "enabled": True,
            "check_interval": 5,
            "channels": {
                "email": {"enabled": True},
                "slack": {"enabled": True},
                "web": {"enabled": True}
            },
            "rules": [rule.to_dict() for rule in self.rules]
        })
        
        # Create alert manager
        self.alert_manager = AlertManager(self.config)
        
        # Register mock channels
        self.alert_manager.register_channel("email", self.email_channel)
        self.alert_manager.register_channel("slack", self.slack_channel)
        self.alert_manager.register_channel("web", self.web_channel)
    
    async def test_start_stop(self):
        """Test starting and stopping the alert manager."""
        self.assertFalse(self.alert_manager.running)
        
        await self.alert_manager.start()
        self.assertTrue(self.alert_manager.running)
        self.assertTrue(self.email_channel.initialized)
        self.assertTrue(self.slack_channel.initialized)
        self.assertTrue(self.web_channel.initialized)
        
        await self.alert_manager.stop()
        self.assertFalse(self.alert_manager.running)
        self.assertTrue(self.email_channel.closed)
        self.assertTrue(self.slack_channel.closed)
        self.assertTrue(self.web_channel.closed)
    
    async def test_process_metrics_no_alerts(self):
        """Test processing metrics that don't trigger alerts."""
        await self.alert_manager.start()
        
        metrics = {
            "metric_1": 5,  # Below threshold for rule 1
            "metric_2": 10  # Above threshold for rule 2
        }
        
        alerts = await self.alert_manager.process_metrics(metrics)
        self.assertEqual(len(alerts), 0)
        self.assertEqual(len(self.email_channel.notifications), 0)
        self.assertEqual(len(self.slack_channel.notifications), 0)
        self.assertEqual(len(self.web_channel.notifications), 0)
        
        await self.alert_manager.stop()
    
    async def test_process_metrics_with_alerts(self):
        """Test processing metrics that trigger alerts."""
        await self.alert_manager.start()
        
        metrics = {
            "metric_1": 15,  # Above threshold for rule 1
            "metric_2": 3    # Below threshold for rule 2
        }
        
        alerts = await self.alert_manager.process_metrics(metrics)
        self.assertEqual(len(alerts), 2)
        
        # Check email channel (should have 1 notification from rule 1)
        self.assertEqual(len(self.email_channel.notifications), 1)
        self.assertEqual(self.email_channel.notifications[0].rule_id, "test_rule_1")
        
        # Check slack channel (should have 2 notifications, from both rules)
        self.assertEqual(len(self.slack_channel.notifications), 2)
        rule_ids = [alert.rule_id for alert in self.slack_channel.notifications]
        self.assertIn("test_rule_1", rule_ids)
        self.assertIn("test_rule_2", rule_ids)
        
        # Check web channel (should have 1 notification from rule 2)
        self.assertEqual(len(self.web_channel.notifications), 1)
        self.assertEqual(self.web_channel.notifications[0].rule_id, "test_rule_2")
        
        await self.alert_manager.stop()
    
    async def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        await self.alert_manager.start()
        
        # Process metrics to generate an alert
        metrics = {"metric_1": 15}
        alerts = await self.alert_manager.process_metrics(metrics)
        self.assertEqual(len(alerts), 1)
        
        # Get the alert ID
        alert_id = alerts[0].alert_id
        
        # Mock the alert history update method
        self.alert_manager.alert_history.update_alert = AsyncMock(return_value=True)
        
        # Acknowledge the alert
        result = await self.alert_manager.acknowledge_alert(alert_id, "test_user")
        self.assertTrue(result)
        
        # Verify update_alert was called with correct parameters
        self.alert_manager.alert_history.update_alert.assert_called_once()
        call_args = self.alert_manager.alert_history.update_alert.call_args[0]
        self.assertEqual(call_args[0], alert_id)
        self.assertEqual(call_args[1]["acknowledged"], True)
        self.assertEqual(call_args[1]["acknowledged_by"], "test_user")
        
        await self.alert_manager.stop()


if __name__ == "__main__":
    unittest.main()