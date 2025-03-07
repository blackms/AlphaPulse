"""
Entry point for running the alerting system as a standalone service.

This module allows the alerting system to be run directly with:
    python -m alpha_pulse.monitoring.alerting

It loads the configuration, initializes the alerting system, and starts
processing metrics from the monitoring system.
"""
import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Any, Optional

from .config import load_alerting_config
from .manager import AlertManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alpha_pulse.alerting")


class AlertingService:
    """Service for running the alerting system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the alerting service.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.alert_manager: Optional[AlertManager] = None
        self.running = False
        self.metrics_queue = asyncio.Queue()
    
    async def start(self) -> None:
        """Start the alerting service."""
        logger.info("Starting alerting service")
        
        # Load configuration
        config = load_alerting_config(self.config_path)
        
        # Create and start alert manager
        self.alert_manager = AlertManager(config)
        await self.alert_manager.start()
        
        # Set running flag
        self.running = True
        
        # Start processing metrics
        asyncio.create_task(self._process_metrics())
        
        logger.info("Alerting service started")
    
    async def stop(self) -> None:
        """Stop the alerting service."""
        if not self.running:
            return
            
        logger.info("Stopping alerting service")
        
        # Clear running flag
        self.running = False
        
        # Stop alert manager
        if self.alert_manager:
            await self.alert_manager.stop()
            self.alert_manager = None
        
        logger.info("Alerting service stopped")
    
    async def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add metrics to the processing queue.
        
        Args:
            metrics: Dictionary of metrics
        """
        await self.metrics_queue.put(metrics)
    
    async def _process_metrics(self) -> None:
        """Process metrics from the queue."""
        while self.running:
            try:
                # Get metrics from queue with timeout
                try:
                    metrics = await asyncio.wait_for(self.metrics_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process metrics
                if self.alert_manager:
                    await self.alert_manager.process_metrics(metrics)
                
                # Mark task as done
                self.metrics_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing metrics: {str(e)}")
                await asyncio.sleep(1)  # Short sleep on error


async def main() -> None:
    """Main entry point."""
    # Get configuration path from environment or command line
    config_path = os.environ.get("AP_ALERTING_CONFIG")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and start service
    service = AlertingService(config_path)
    
    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))
    
    # Start service
    await service.start()
    
    # Keep running until stopped
    while service.running:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())