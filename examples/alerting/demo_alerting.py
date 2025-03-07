#!/usr/bin/env python
"""
Demo script for the AlphaPulse alerting system.

This script demonstrates how to:
1. Configure and initialize the alerting system
2. Create and register alert rules
3. Process metrics and generate alerts
4. Acknowledge alerts
5. Retrieve alert history

Usage:
    python demo_alerting.py
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
import random

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from alpha_pulse.monitoring.alerting.models import AlertRule, AlertSeverity
from alpha_pulse.monitoring.alerting.manager import AlertManager
from alpha_pulse.monitoring.alerting.config import AlertingConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alerting_demo")


async def demo_alerting():
    """Run the alerting system demo."""
    logger.info("Starting alerting system demo")
    
    # Create alerting configuration
    config = {
        "enabled": True,
        "check_interval": 10,  # seconds
        
        # Configure channels
        "channels": {
            "console": {
                "enabled": True,
                "type": "console"  # Simple console output for demo
            },
            "web": {
                "enabled": True,
                "max_alerts": 100
            }
        },
        
        # Configure alert history storage
        "history": {
            "type": "memory",
            "max_alerts": 1000
        },
        
        # Define alert rules
        "rules": [
            {
                "rule_id": "sharpe_ratio_low",
                "name": "Low Sharpe Ratio",
                "description": "Alerts when the Sharpe ratio falls below threshold",
                "metric_name": "sharpe_ratio",
                "condition": "< 0.5",
                "severity": "warning",
                "message_template": "Sharpe ratio is {value}, below threshold of 0.5",
                "channels": ["console", "web"],
                "cooldown_period": 60,  # 1 minute for demo
                "enabled": True
            },
            {
                "rule_id": "drawdown_high",
                "name": "High Drawdown",
                "description": "Alerts when drawdown exceeds threshold",
                "metric_name": "max_drawdown",
                "condition": "> 0.1",
                "severity": "error",
                "message_template": "Drawdown is {value}, exceeding threshold of 10%",
                "channels": ["console", "web"],
                "cooldown_period": 60,  # 1 minute for demo
                "enabled": True
            },
            {
                "rule_id": "portfolio_value_low",
                "name": "Low Portfolio Value",
                "description": "Alerts when portfolio value falls below threshold",
                "metric_name": "portfolio_value",
                "condition": "< 100000",
                "severity": "critical",
                "message_template": "Portfolio value is ${value}, below threshold of $100,000",
                "channels": ["console", "web"],
                "cooldown_period": 60,  # 1 minute for demo
                "enabled": True
            }
        ]
    }
    
    # Create a simple console notification channel for the demo
    from alpha_pulse.monitoring.alerting.channels.base import NotificationChannel
    from alpha_pulse.monitoring.alerting.models import Alert
    
    class ConsoleNotificationChannel(NotificationChannel):
        """Simple console notification channel for demo purposes."""
        
        async def initialize(self) -> bool:
            """Initialize the channel."""
            logger.info("Console notification channel initialized")
            return True
        
        async def send_notification(self, alert: Alert) -> bool:
            """Send notification to console."""
            severity = alert.severity.value.upper()
            print(f"\n[{severity} ALERT] {alert.message}")
            print(f"  Metric: {alert.metric_name} = {alert.metric_value}")
            print(f"  Time: {alert.timestamp}")
            print(f"  ID: {alert.alert_id}\n")
            return True
        
        async def close(self) -> None:
            """Close the channel."""
            pass
    
    # Initialize the alerting system
    alerting_config = AlertingConfig(config)
    alert_manager = AlertManager(alerting_config)
    
    # Register the console notification channel
    alert_manager.register_channel("console", ConsoleNotificationChannel({}))
    
    # Start the alert manager
    await alert_manager.start()
    
    try:
        # Simulate metrics for demo
        for i in range(10):
            logger.info(f"Simulating metrics batch {i+1}/10")
            
            # Generate random metrics
            metrics = {
                "sharpe_ratio": random.uniform(0.1, 1.0),
                "max_drawdown": random.uniform(0.05, 0.2),
                "portfolio_value": random.uniform(80000, 120000),
                "win_rate": random.uniform(0.4, 0.7),
                "volatility": random.uniform(0.1, 0.3)
            }
            
            logger.info(f"Metrics: {metrics}")
            
            # Process metrics
            alerts = await alert_manager.process_metrics(metrics)
            
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts")
                
                # Acknowledge some alerts randomly
                for alert in alerts:
                    if random.random() > 0.5:
                        logger.info(f"Acknowledging alert: {alert.alert_id}")
                        await alert_manager.acknowledge_alert(alert.alert_id, "demo_user")
            else:
                logger.info("No alerts generated")
            
            # Wait before next batch
            await asyncio.sleep(2)
        
        # Retrieve and display alert history
        logger.info("\nRetrieving alert history:")
        all_alerts = await alert_manager.get_alert_history()
        print(f"\nAlert History ({len(all_alerts)} alerts):")
        for alert in all_alerts:
            ack_status = "Acknowledged" if alert.acknowledged else "Unacknowledged"
            print(f"- [{alert.severity.value.upper()}] {alert.message} ({ack_status})")
        
        # Filter for unacknowledged alerts
        unack_alerts = await alert_manager.get_alert_history(
            filters={"acknowledged": False}
        )
        print(f"\nUnacknowledged Alerts ({len(unack_alerts)} alerts):")
        for alert in unack_alerts:
            print(f"- [{alert.severity.value.upper()}] {alert.message}")
        
        # Filter by severity
        critical_alerts = await alert_manager.get_alert_history(
            filters={"severity": "critical"}
        )
        print(f"\nCritical Alerts ({len(critical_alerts)} alerts):")
        for alert in critical_alerts:
            ack_status = "Acknowledged" if alert.acknowledged else "Unacknowledged"
            print(f"- {alert.message} ({ack_status})")
        
    finally:
        # Stop the alert manager
        await alert_manager.stop()
        logger.info("Alerting system demo completed")


# Add a custom rule at runtime
async def add_custom_rule(alert_manager):
    """Add a custom rule at runtime."""
    custom_rule = AlertRule(
        rule_id="custom_volatility_high",
        name="High Volatility",
        description="Alerts when volatility exceeds threshold",
        metric_name="volatility",
        condition="> 0.25",
        severity=AlertSeverity.WARNING,
        message_template="Volatility is {value}, exceeding threshold of 0.25",
        channels=["console", "web"],
        cooldown_period=60,
        enabled=True
    )
    
    await alert_manager.add_rule(custom_rule)
    logger.info(f"Added custom rule: {custom_rule.name}")


if __name__ == "__main__":
    asyncio.run(demo_alerting())