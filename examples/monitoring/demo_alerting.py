"""
Demo of the AlphaPulse Alerting System.

This example demonstrates how to use the alerting system to monitor metrics
and send notifications when predefined conditions are met.
"""
import asyncio
import logging
import os
import sys
import random
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.alpha_pulse.monitoring.alerting import (
    Alert, 
    AlertRule, 
    AlertSeverity, 
    AlertManager, 
    AlertingConfig
)
from src.alpha_pulse.monitoring.alerting.channels.email import EmailNotificationChannel
from src.alpha_pulse.monitoring.alerting.channels.slack import SlackNotificationChannel
from src.alpha_pulse.monitoring.alerting.channels.web import WebNotificationChannel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alerting_demo")


async def simulate_metrics(alert_manager: AlertManager, duration: int = 60):
    """
    Simulate metrics for testing the alerting system.
    
    Args:
        alert_manager: The alert manager to process metrics
        duration: Duration to run the simulation in seconds
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration)
    
    logger.info(f"Starting metrics simulation for {duration} seconds")
    
    # Define some metrics to simulate
    metrics = {
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.05,
        "var_95": 0.03,
        "leverage": 1.2,
        "api_latency": 200,
        "error_rate": 0.02,
        "memory_usage_percent": 60,
        "cpu_usage_percent": 30
    }
    
    # Simulate metrics changing over time
    while datetime.now() < end_time:
        # Update metrics with some random variation
        current_metrics = {}
        for key, value in metrics.items():
            # Add random variation (±20%)
            variation = value * 0.2
            new_value = value + random.uniform(-variation, variation)
            current_metrics[key] = new_value
            
        # Occasionally trigger alerts by setting extreme values
        if random.random() < 0.2:  # 20% chance
            # Pick a random metric to spike
            metric_to_spike = random.choice(list(metrics.keys()))
            
            if metric_to_spike == "sharpe_ratio":
                # Low Sharpe ratio (bad)
                current_metrics[metric_to_spike] = 0.3
            elif metric_to_spike == "max_drawdown":
                # High drawdown (bad)
                current_metrics[metric_to_spike] = 0.15
            elif metric_to_spike == "var_95":
                # High VaR (bad)
                current_metrics[metric_to_spike] = 0.08
            elif metric_to_spike == "leverage":
                # High leverage (bad)
                current_metrics[metric_to_spike] = 1.8
            elif metric_to_spike == "api_latency":
                # High latency (bad)
                current_metrics[metric_to_spike] = 1200
            elif metric_to_spike == "error_rate":
                # High error rate (bad)
                current_metrics[metric_to_spike] = 0.08
            elif metric_to_spike == "memory_usage_percent":
                # High memory usage (bad)
                current_metrics[metric_to_spike] = 92
            elif metric_to_spike == "cpu_usage_percent":
                # High CPU usage (bad)
                current_metrics[metric_to_spike] = 95
            
            logger.info(f"Simulating spike in {metric_to_spike}: {current_metrics[metric_to_spike]}")
        
        # Process metrics with the alert manager
        alerts = await alert_manager.process_metrics(current_metrics)
        
        if alerts:
            logger.info(f"Generated {len(alerts)} alerts")
            for alert in alerts:
                logger.info(f"  - {alert.severity.value.upper()}: {alert.message}")
        
        # Wait before next update
        await asyncio.sleep(2)
    
    logger.info("Metrics simulation completed")


async def main():
    """Main function to run the alerting demo."""
    logger.info("Starting alerting system demo")
    
    # Create alert rules
    rules = [
        AlertRule(
            rule_id="sharpe_ratio_low",
            name="Low Sharpe Ratio",
            description="Alerts when the Sharpe ratio falls below threshold",
            metric_name="sharpe_ratio",
            condition="< 0.5",
            severity=AlertSeverity.WARNING,
            message_template="Sharpe ratio is {value:.2f}, below threshold of 0.5",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        ),
        AlertRule(
            rule_id="drawdown_high",
            name="High Drawdown",
            description="Alerts when drawdown exceeds threshold",
            metric_name="max_drawdown",
            condition="> 0.1",
            severity=AlertSeverity.ERROR,
            message_template="Drawdown is {value:.2%}, exceeding threshold of 10%",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        ),
        AlertRule(
            rule_id="var_high",
            name="High Value at Risk",
            description="Alerts when VaR exceeds threshold",
            metric_name="var_95",
            condition="> 0.05",
            severity=AlertSeverity.WARNING,
            message_template="Value at Risk (95%) is {value:.2%}, exceeding threshold of 5%",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        ),
        AlertRule(
            rule_id="leverage_high",
            name="High Leverage",
            description="Alerts when leverage exceeds threshold",
            metric_name="leverage",
            condition="> 1.5",
            severity=AlertSeverity.ERROR,
            message_template="Portfolio leverage is {value:.2f}x, exceeding threshold of 1.5x",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        ),
        AlertRule(
            rule_id="api_latency_high",
            name="High API Latency",
            description="Alerts when API latency exceeds threshold",
            metric_name="api_latency",
            condition="> 1000",
            severity=AlertSeverity.WARNING,
            message_template="API latency is {value:.0f}ms, exceeding threshold of 1000ms",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        ),
        AlertRule(
            rule_id="error_rate_high",
            name="High Error Rate",
            description="Alerts when error rate exceeds threshold",
            metric_name="error_rate",
            condition="> 0.05",
            severity=AlertSeverity.ERROR,
            message_template="Error rate is {value:.2%}, exceeding threshold of 5%",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        ),
        AlertRule(
            rule_id="memory_usage_high",
            name="High Memory Usage",
            description="Alerts when memory usage exceeds threshold",
            metric_name="memory_usage_percent",
            condition="> 90",
            severity=AlertSeverity.WARNING,
            message_template="Memory usage is {value:.1f}%, exceeding threshold of 90%",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        ),
        AlertRule(
            rule_id="cpu_usage_high",
            name="High CPU Usage",
            description="Alerts when CPU usage exceeds threshold",
            metric_name="cpu_usage_percent",
            condition="> 90",
            severity=AlertSeverity.WARNING,
            message_template="CPU usage is {value:.1f}%, exceeding threshold of 90%",
            channels=["console", "web"],
            cooldown_period=10  # Short cooldown for demo
        )
    ]
    
    # Create a simple configuration
    config = AlertingConfig({
        "enabled": True,
        "check_interval": 5,
        "channels": {
            "console": {
                "enabled": True
            },
            "web": {
                "enabled": True,
                "max_alerts": 100
            }
        },
        "rules": [rule.to_dict() for rule in rules]
    })
    
    # Create alert manager
    alert_manager = AlertManager(config)
    
    # Create a console notification channel (for demo purposes)
    class ConsoleNotificationChannel(WebNotificationChannel):
        async def send_notification(self, alert: Alert) -> bool:
            logger.info(f"ALERT: {alert.severity.value.upper()} - {alert.message}")
            return await super().send_notification(alert)
    
    # Register console channel
    alert_manager.register_channel("console", ConsoleNotificationChannel({"enabled": True}))
    
    # Start the alert manager
    await alert_manager.start()
    
    try:
        # Run the metrics simulation
        await simulate_metrics(alert_manager, duration=60)
        
        # Get alert history
        alerts = await alert_manager.get_alert_history()
        logger.info(f"Alert history contains {len(alerts)} alerts")
        
        # Show the most recent alerts
        for i, alert in enumerate(alerts[:5]):
            logger.info(f"Alert {i+1}: {alert.severity.value.upper()} - {alert.message}")
        
    finally:
        # Stop the alert manager
        await alert_manager.stop()
    
    logger.info("Alerting system demo completed")


if __name__ == "__main__":
    asyncio.run(main())