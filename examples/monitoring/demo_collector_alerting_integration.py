"""
Demo of integrating the AlphaPulse Metrics Collector with the Alerting System.

This example demonstrates how to connect the metrics collector with the alerting system
to automatically process metrics and generate alerts when conditions are met.
"""
import asyncio
import logging
import os
import sys
import random
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.alpha_pulse.monitoring import (
    EnhancedMetricsCollector,
    AlertManager,
    AlertRule,
    AlertSeverity,
    MonitoringConfig,
    AlertingConfig,
    load_config,
    load_alerting_config
)
from src.alpha_pulse.portfolio.data_models import PortfolioData, Position


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("collector_alerting_demo")


class ConsoleNotificationChannel:
    """Simple console notification channel for demo purposes."""
    
    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or {}
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the channel."""
        self.initialized = True
        logger.info("Console notification channel initialized")
        return True
    
    async def send_notification(self, alert) -> bool:
        """Send notification for an alert."""
        logger.info(f"ALERT: {alert.severity.value.upper()} - {alert.message}")
        return True
    
    async def close(self) -> None:
        """Close the channel."""
        pass


async def generate_sample_portfolio() -> PortfolioData:
    """Generate a sample portfolio for demonstration."""
    # Create a portfolio with random performance metrics
    portfolio = PortfolioData(
        portfolio_id="demo_portfolio",
        timestamp=datetime.now(),
        total_value=1000000.0,
        cash=200000.0,
        positions=[
            Position(
                symbol="BTC",
                quantity=2.5,
                current_price=40000.0,
                cost_basis=35000.0
            ),
            Position(
                symbol="ETH",
                quantity=30.0,
                current_price=2500.0,
                cost_basis=2200.0
            ),
            Position(
                symbol="SOL",
                quantity=500.0,
                current_price=100.0,
                cost_basis=80.0
            )
        ]
    )
    
    # Add some performance metrics that might trigger alerts
    portfolio.sharpe_ratio = random.uniform(0.2, 1.2)
    portfolio.sortino_ratio = random.uniform(0.3, 1.5)
    portfolio.max_drawdown = random.uniform(0.05, 0.2)
    portfolio.volatility = random.uniform(0.1, 0.4)
    portfolio.var_95 = random.uniform(0.02, 0.08)
    portfolio.leverage = random.uniform(1.0, 1.8)
    
    return portfolio


async def main():
    """Main function to run the integration demo."""
    logger.info("Starting metrics collector and alerting system integration demo")
    
    # Load monitoring configuration
    monitoring_config = MonitoringConfig.from_dict({
        "storage": {
            "type": "memory",
            "memory_max_points": 1000
        },
        "collection_interval": 5,  # 5 seconds
        "enable_realtime": True
    })
    
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
            channels=["console"],
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
            channels=["console"],
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
            channels=["console"],
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
            channels=["console"],
            cooldown_period=10  # Short cooldown for demo
        )
    ]
    
    # Create alerting configuration
    alerting_config = AlertingConfig({
        "enabled": True,
        "check_interval": 5,  # 5 seconds
        "channels": {
            "console": {"enabled": True}
        },
        "rules": [rule.to_dict() for rule in rules]
    })
    
    # Create metrics collector
    collector = EnhancedMetricsCollector(config=monitoring_config)
    
    # Create alert manager
    alert_manager = AlertManager(alerting_config)
    
    # Register console notification channel
    alert_manager.register_channel("console", ConsoleNotificationChannel())
    
    # Define callback to process metrics with alert manager
    async def process_metrics_callback(metrics_dict):
        # Extract performance and risk metrics
        performance_metrics = metrics_dict.get("performance", {})
        risk_metrics = metrics_dict.get("risk", {})
        
        # Combine metrics for alerting
        combined_metrics = {**performance_metrics, **risk_metrics}
        
        if combined_metrics:
            # Process metrics with alert manager
            alerts = await alert_manager.process_metrics(combined_metrics)
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts")
    
    # Start both components
    await collector.start()
    await alert_manager.start()
    
    try:
        # Run for 60 seconds
        logger.info("Running integration demo for 60 seconds...")
        end_time = datetime.now() + timedelta(seconds=60)
        
        while datetime.now() < end_time:
            # Generate sample portfolio
            portfolio = await generate_sample_portfolio()
            
            # Log current metrics
            logger.info(f"Portfolio metrics - Sharpe: {portfolio.sharpe_ratio:.2f}, "
                       f"Drawdown: {portfolio.max_drawdown:.2%}, "
                       f"VaR: {portfolio.var_95:.2%}, "
                       f"Leverage: {portfolio.leverage:.2f}")
            
            # Collect metrics
            metrics = await collector.collect_and_store(portfolio_data=portfolio)
            
            # Process metrics with callback
            await process_metrics_callback(metrics)
            
            # Wait before next update
            await asyncio.sleep(5)
        
        # Get alert history
        alerts = await alert_manager.get_alert_history()
        logger.info(f"Alert history contains {len(alerts)} alerts")
        
        # Show the most recent alerts
        for i, alert in enumerate(alerts[:5]):
            logger.info(f"Alert {i+1}: {alert.severity.value.upper()} - {alert.message}")
        
    finally:
        # Stop both components
        await collector.stop()
        await alert_manager.stop()
    
    logger.info("Integration demo completed")


if __name__ == "__main__":
    asyncio.run(main())