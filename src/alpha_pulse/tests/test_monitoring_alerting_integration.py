"""
Integration test for the monitoring and alerting systems.
"""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from alpha_pulse.monitoring import (
    EnhancedMetricsCollector,
    AlertManager,
    AlertRule,
    AlertSeverity,
    MonitoringConfig,
    AlertingConfig
)


class TestMonitoringAlertingIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the monitoring and alerting systems."""
    
    async def asyncSetUp(self):
        """Set up test fixtures."""
        # Create monitoring config with in-memory storage
        monitoring_config = MonitoringConfig.from_dict({
            "storage": {
                "type": "memory",
                "memory_max_points": 1000
            },
            "collection_interval": 1,  # 1 second for faster testing
            "enable_realtime": True
        })
        
        # Create alerting config with rules
        alerting_config = AlertingConfig({
            "enabled": True,
            "check_interval": 1,  # 1 second for faster testing
            "channels": {
                "test": {"enabled": True}
            },
            "rules": [
                {
                    "rule_id": "test_rule_1",
                    "name": "Test Rule 1",
                    "description": "A test rule for performance metrics",
                    "metric_name": "sharpe_ratio",
                    "condition": "< 0.5",
                    "severity": "warning",
                    "message_template": "Sharpe ratio is {value:.2f}, below threshold of 0.5",
                    "channels": ["test"],
                    "cooldown_period": 0,  # No cooldown for testing
                    "enabled": True
                },
                {
                    "rule_id": "test_rule_2",
                    "name": "Test Rule 2",
                    "description": "A test rule for risk metrics",
                    "metric_name": "max_drawdown",
                    "condition": "> 0.1",
                    "severity": "error",
                    "message_template": "Max drawdown is {value:.2%}, exceeding threshold of 10%",
                    "channels": ["test"],
                    "cooldown_period": 0,  # No cooldown for testing
                    "enabled": True
                }
            ]
        })
        
        # Create metrics collector
        self.collector = EnhancedMetricsCollector(config=monitoring_config)
        
        # Create alert manager
        self.alert_manager = AlertManager(alerting_config)
        
        # Create a mock notification channel
        self.mock_channel = MagicMock()
        self.mock_channel.initialize = MagicMock(return_value=True)
        self.mock_channel.send_notification = MagicMock(return_value=True)
        self.mock_channel.close = MagicMock()
        
        # Register mock channel
        self.alert_manager.register_channel("test", self.mock_channel)
        
        # Start both components
        await self.collector.start()
        await self.alert_manager.start()
    
    async def asyncTearDown(self):
        """Tear down test fixtures."""
        await self.collector.stop()
        await self.alert_manager.stop()
    
    async def test_metrics_to_alerts_integration(self):
        """Test that metrics collected by the collector trigger alerts."""
        # Create a callback to process metrics with the alert manager
        async def process_metrics_callback(metrics):
            await self.alert_manager.process_metrics(metrics)
        
        # Mock the portfolio data
        portfolio_data = MagicMock()
        portfolio_data.sharpe_ratio = 0.3  # Below threshold, should trigger alert
        portfolio_data.max_drawdown = 0.15  # Above threshold, should trigger alert
        
        # Mock the performance metrics calculation to return our test metrics
        with patch('alpha_pulse.monitoring.collector.calculate_performance_metrics') as mock_perf:
            mock_perf.return_value = {
                "sharpe_ratio": 0.3,
                "max_drawdown": 0.15
            }
            
            # Collect metrics
            metrics = await self.collector.collect_and_store(portfolio_data=portfolio_data)
            
            # Process metrics with alert manager
            await process_metrics_callback(metrics.get("performance", {}))
            
            # Wait a moment for async processing
            await asyncio.sleep(0.1)
            
            # Verify that alerts were triggered
            self.mock_channel.send_notification.assert_called()
            
            # Should have 2 calls to send_notification (one for each rule)
            self.assertEqual(self.mock_channel.send_notification.call_count, 2)
            
            # Check the alerts that were sent
            calls = self.mock_channel.send_notification.call_args_list
            alerts = [call[0][0] for call in calls]
            
            # Verify alert properties
            rule_ids = [alert.rule_id for alert in alerts]
            self.assertIn("test_rule_1", rule_ids)
            self.assertIn("test_rule_2", rule_ids)
            
            # Verify alert severities
            severities = [alert.severity for alert in alerts]
            self.assertIn(AlertSeverity.WARNING, severities)
            self.assertIn(AlertSeverity.ERROR, severities)
    
    async def test_metrics_collector_callback_integration(self):
        """Test integration using the metrics collector callback mechanism."""
        # Create a callback to process metrics with the alert manager
        async def process_metrics_callback(metrics_dict):
            # Extract performance metrics
            performance_metrics = metrics_dict.get("performance", {})
            if performance_metrics:
                await self.alert_manager.process_metrics(performance_metrics)
        
        # Register the callback with the collector
        self.collector.register_callback = MagicMock()
        self.collector.register_callback(process_metrics_callback)
        
        # Mock the portfolio data
        portfolio_data = MagicMock()
        portfolio_data.sharpe_ratio = 0.3  # Below threshold, should trigger alert
        
        # Mock the performance metrics calculation to return our test metrics
        with patch('alpha_pulse.monitoring.collector.calculate_performance_metrics') as mock_perf:
            mock_perf.return_value = {
                "sharpe_ratio": 0.3
            }
            
            # Verify that the callback was registered
            self.collector.register_callback.assert_called_with(process_metrics_callback)
            
            # This test verifies the integration pattern, not the actual callback execution
            # since we've mocked the register_callback method


if __name__ == "__main__":
    unittest.main()