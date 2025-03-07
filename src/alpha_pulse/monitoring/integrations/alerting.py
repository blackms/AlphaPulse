"""
Integration between the monitoring system and the alerting system.

This module provides the necessary components to connect the monitoring
system's metrics collection with the alerting system for rule evaluation
and alert generation.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List

from alpha_pulse.monitoring.alerting.config import load_alerting_config
from alpha_pulse.monitoring.alerting.manager import AlertManager
from alpha_pulse.monitoring.alerting.models import Alert


class AlertingIntegration:
    """Integration between monitoring and alerting systems."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the alerting integration.
        
        Args:
            config_path: Optional path to alerting configuration file
        """
        self.logger = logging.getLogger("alpha_pulse.monitoring.integrations.alerting")
        self.config_path = config_path
        self.alert_manager: Optional[AlertManager] = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the alerting integration.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Load alerting configuration
            config = load_alerting_config(self.config_path)
            
            # Create and start alert manager
            self.alert_manager = AlertManager(config)
            await self.alert_manager.start()
            
            self.initialized = True
            self.logger.info("Alerting integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize alerting integration: {str(e)}")
            return False
    
    async def process_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Process metrics and generate alerts if needed.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            List[Alert]: List of triggered alerts
        """
        if not self.initialized or not self.alert_manager:
            self.logger.warning("Alerting integration not initialized")
            return []
        
        try:
            # Process metrics through alert manager
            alerts = await self.alert_manager.process_metrics(metrics)
            
            if alerts:
                self.logger.info(f"Generated {len(alerts)} alerts from metrics")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error processing metrics for alerts: {str(e)}")
            return []
    
    async def shutdown(self) -> None:
        """Shut down the alerting integration."""
        if not self.initialized or not self.alert_manager:
            return
        
        try:
            # Stop alert manager
            await self.alert_manager.stop()
            self.alert_manager = None
            self.initialized = False
            
            self.logger.info("Alerting integration shut down")
            
        except Exception as e:
            self.logger.error(f"Error shutting down alerting integration: {str(e)}")


# Create a singleton instance for easy import
alerting_integration = AlertingIntegration()


async def initialize_alerting(config_path: Optional[str] = None) -> bool:
    """Initialize the alerting integration.
    
    Args:
        config_path: Optional path to alerting configuration file
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global alerting_integration
    
    # Create new instance if config path provided
    if config_path:
        alerting_integration = AlertingIntegration(config_path)
    
    # Initialize the integration
    return await alerting_integration.initialize()


async def process_metrics_for_alerts(metrics: Dict[str, Any]) -> List[Alert]:
    """Process metrics and generate alerts if needed.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        List[Alert]: List of triggered alerts
    """
    return await alerting_integration.process_metrics(metrics)


async def shutdown_alerting() -> None:
    """Shut down the alerting integration."""
    await alerting_integration.shutdown()