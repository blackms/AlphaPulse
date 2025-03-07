"""WebSocket subscription management."""
from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime

from .manager import ConnectionManager
from alpha_pulse.monitoring.metrics_calculations import calculate_derived_metrics


class SubscriptionManager:
    """Manage subscriptions and updates."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = SubscriptionManager()
        return cls._instance
    
    def __init__(self):
        """Initialize subscription manager."""
        self.logger = logging.getLogger("alpha_pulse.api.websockets.subscription")
        self.connection_manager = ConnectionManager.get_instance()
        self.running = False
        self.update_tasks = []
        
    async def start(self) -> None:
        """Start subscription manager."""
        if self.running:
            return
            
        self.running = True
        
        # Start update tasks
        self.update_tasks = [
            asyncio.create_task(self._update_metrics()),
            asyncio.create_task(self._update_portfolio()),
            asyncio.create_task(self._listen_for_alerts()),
            asyncio.create_task(self._listen_for_trades())
        ]
        
        self.logger.info("Subscription manager started")
        
    async def stop(self) -> None:
        """Stop subscription manager."""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel all tasks
        for task in self.update_tasks:
            task.cancel()
            
        self.update_tasks = []
        self.logger.info("Subscription manager stopped")
    
    async def _update_metrics(self) -> None:
        """Update metrics periodically."""
        from alpha_pulse.monitoring.collector import MetricsCollector
        
        collector = MetricsCollector.get_instance()
        
        try:
            while self.running:
                # Get latest metrics
                latest_metrics = await collector.collect_latest_metrics()
                
                # Add derived metrics
                derived = calculate_derived_metrics(latest_metrics)
                
                # Create update message
                message = {
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": {}
                }
                
                # Add metrics to message
                for metric in latest_metrics:
                    message["data"][metric.name] = {
                        "value": metric.value,
                        "timestamp": metric.timestamp.isoformat(),
                        "labels": metric.labels
                    }
                    
                # Add derived metrics
                for name, value in derived.items():
                    message["data"][name] = {
                        "value": value,
                        "timestamp": datetime.now().isoformat(),
                        "labels": {"derived": "true"}
                    }
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("metrics", message)
                
                # Wait for next update
                await asyncio.sleep(5)  # Update every 5 seconds
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            
    async def _update_portfolio(self) -> None:
        """Update portfolio periodically."""
        from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
        
        portfolio_manager = PortfolioManager.get_instance()
        
        try:
            while self.running:
                # Get current portfolio
                portfolio = portfolio_manager.get_portfolio_data()
                
                # Create update message
                message = {
                    "type": "portfolio",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "total_value": portfolio.total_value,
                        "cash": portfolio.cash,
                        "positions": []
                    }
                }
                
                # Add positions
                for position in portfolio.positions:
                    message["data"]["positions"].append({
                        "symbol": position.symbol,
                        "quantity": position.quantity,
                        "entry_price": position.entry_price,
                        "current_price": position.current_price,
                        "value": position.value,
                        "pnl": position.pnl,
                        "pnl_percentage": position.pnl_percentage
                    })
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("portfolio", message)
                
                # Wait for next update
                await asyncio.sleep(10)  # Update every 10 seconds
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {str(e)}")
            
    async def _listen_for_alerts(self) -> None:
        """Listen for alert events."""
        from alpha_pulse.monitoring.alerting.manager import AlertManager
        
        alert_manager = AlertManager.get_instance()
        
        try:
            # Register callback for new alerts
            async def handle_alert(alert):
                # Create update message
                message = {
                    "type": "alert",
                    "timestamp": datetime.now().isoformat(),
                    "data": alert.to_dict()
                }
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("alerts", message)
            
            # Register callback
            alert_manager.register_alert_callback(handle_alert)
            
            # Keep task alive
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Task was cancelled
            alert_manager.unregister_alert_callback(handle_alert)
        except Exception as e:
            self.logger.error(f"Error listening for alerts: {str(e)}")
            
    async def _listen_for_trades(self) -> None:
        """Listen for trade events."""
        from alpha_pulse.execution.broker_interface import BrokerInterface
        
        broker = BrokerInterface.get_instance()
        
        try:
            # Register callback for new trades
            async def handle_trade(trade):
                # Create update message
                message = {
                    "type": "trade",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "timestamp": trade.timestamp.isoformat(),
                        "status": trade.status,
                        "order_type": trade.order_type,
                        "fees": trade.fees
                    }
                }
                
                # Broadcast to subscribers
                await self.connection_manager.broadcast("trades", message)
            
            # Register callback
            broker.register_trade_callback(handle_trade)
            
            # Keep task alive
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Task was cancelled
            broker.unregister_trade_callback(handle_trade)
        except Exception as e:
            self.logger.error(f"Error listening for trades: {str(e)}")