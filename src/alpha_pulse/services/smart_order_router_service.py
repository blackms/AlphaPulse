"""
Smart Order Router Service for AlphaPulse.

Service wrapper for the SmartOrderRouter that integrates with the main application.
"""
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from alpha_pulse.execution.smart_order_router import (
    SmartOrderRouter, 
    SmartOrderConfig, 
    ExecutionStrategy, 
    OrderUrgency,
    SmartOrder
)
from alpha_pulse.execution.broker_interface import BrokerInterface, OrderSide
from alpha_pulse.services.regime_detection_service import RegimeDetectionService
from alpha_pulse.monitoring.alert_manager import AlertManager, Alert, AlertLevel


class SmartOrderRouterService:
    """
    Service wrapper for Smart Order Router with monitoring and alerting.
    
    Provides a high-level interface for smart order execution with:
    - Integration with regime detection
    - Performance monitoring
    - Alert management
    - Configuration management
    """
    
    def __init__(
        self,
        broker: BrokerInterface,
        regime_service: Optional[RegimeDetectionService] = None,
        alert_manager: Optional[AlertManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize smart order router service."""
        self.broker = broker
        self.regime_service = regime_service
        self.alert_manager = alert_manager
        self.config = config or {}
        
        # Initialize the router
        self.router = SmartOrderRouter(broker, regime_service)
        
        # Service state
        self.is_running = False
        self.total_orders_processed = 0
        self.performance_stats = {}
        
        # Default configurations for different order types
        self.default_configs = {
            'small_order': SmartOrderConfig(
                strategy=ExecutionStrategy.MARKET,
                urgency=OrderUrgency.MEDIUM,
                time_horizon_minutes=5
            ),
            'medium_order': SmartOrderConfig(
                strategy=ExecutionStrategy.ADAPTIVE,
                urgency=OrderUrgency.MEDIUM,
                time_horizon_minutes=15,
                max_slice_size=0.1
            ),
            'large_order': SmartOrderConfig(
                strategy=ExecutionStrategy.TWAP,
                urgency=OrderUrgency.LOW,
                time_horizon_minutes=30,
                max_slice_size=0.05
            ),
            'urgent_order': SmartOrderConfig(
                strategy=ExecutionStrategy.ICEBERG,
                urgency=OrderUrgency.HIGH,
                time_horizon_minutes=5,
                max_slice_size=0.2
            )
        }
        
        logger.info("Smart Order Router Service initialized")
    
    async def start(self):
        """Start the smart order router service."""
        if self.is_running:
            logger.warning("Smart Order Router Service is already running")
            return
        
        self.is_running = True
        logger.info("Smart Order Router Service started")
        
        # Send startup alert
        if self.alert_manager:
            await self._send_alert(
                "Smart Order Router Started",
                "Smart Order Router Service is now operational",
                AlertLevel.INFO
            )
    
    async def stop(self):
        """Stop the smart order router service."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Shutdown the router
        await self.router.shutdown()
        
        logger.info("Smart Order Router Service stopped")
        
        # Send shutdown alert
        if self.alert_manager:
            await self._send_alert(
                "Smart Order Router Stopped",
                f"Service stopped. Processed {self.total_orders_processed} orders",
                AlertLevel.INFO
            )
    
    async def submit_smart_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: str = "medium_order",
        custom_config: Optional[SmartOrderConfig] = None,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """
        Submit a smart order for execution.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Total quantity to execute
            order_type: Type of order ('small_order', 'medium_order', 'large_order', 'urgent_order')
            custom_config: Custom configuration (overrides order_type)
            target_price: Target price (optional)
            stop_price: Stop price (optional)
            
        Returns:
            Smart order ID
        """
        if not self.is_running:
            raise RuntimeError("Smart Order Router Service is not running")
        
        # Get configuration
        if custom_config:
            config = custom_config
        else:
            config = self.default_configs.get(order_type, self.default_configs['medium_order'])
        
        # Submit order to router
        order_id = await self.router.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            config=config,
            target_price=target_price,
            stop_price=stop_price
        )
        
        self.total_orders_processed += 1
        
        # Send order submitted alert for large orders
        if quantity > self.config.get('large_order_threshold', 10000):
            await self._send_alert(
                "Large Order Submitted",
                f"Smart order {order_id}: {side.value} {quantity:,.0f} {symbol}",
                AlertLevel.MEDIUM
            )
        
        logger.info(f"Submitted smart order {order_id}: {side.value} {quantity} {symbol}")
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a smart order."""
        if not self.is_running:
            return False
        
        success = await self.router.cancel_order(order_id)
        
        if success:
            logger.info(f"Cancelled smart order {order_id}")
            await self._send_alert(
                "Order Cancelled",
                f"Smart order {order_id} was cancelled",
                AlertLevel.MEDIUM
            )
        
        return success
    
    def get_order_status(self, order_id: str) -> Optional[SmartOrder]:
        """Get status of a smart order."""
        return self.router.get_order_status(order_id)
    
    def get_active_orders(self) -> Dict[str, SmartOrder]:
        """Get all active orders."""
        return self.router.active_orders.copy()
    
    def update_market_data(self, symbol: str, price: float, volume: float = 0.0, spread: float = 0.001):
        """Update market data for routing decisions."""
        self.router.update_market_data(symbol, price, volume, spread)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        router_stats = self.router.get_execution_stats()
        
        return {
            'service_stats': {
                'is_running': self.is_running,
                'total_orders_processed': self.total_orders_processed,
                'active_orders': len(self.router.active_orders)
            },
            'execution_stats': router_stats,
            'configuration': {
                'default_configs': {
                    name: {
                        'strategy': config.strategy.value,
                        'urgency': config.urgency.value,
                        'time_horizon_minutes': config.time_horizon_minutes,
                        'max_slice_size': config.max_slice_size
                    }
                    for name, config in self.default_configs.items()
                }
            }
        }
    
    async def _send_alert(self, title: str, message: str, level: AlertLevel):
        """Send alert through alert manager."""
        if not self.alert_manager:
            return
        
        alert = Alert(
            title=title,
            message=message,
            level=level,
            source="smart_order_router",
            timestamp=datetime.now()
        )
        
        await self.alert_manager.send_alert(alert)
    
    def configure_strategy(self, order_type: str, config: SmartOrderConfig):
        """Configure a default strategy."""
        if order_type in self.default_configs:
            self.default_configs[order_type] = config
            logger.info(f"Updated configuration for {order_type}")
        else:
            raise ValueError(f"Unknown order type: {order_type}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the service."""
        try:
            router_stats = self.router.get_execution_stats()
            
            health_status = {
                'status': 'healthy' if self.is_running else 'stopped',
                'active_orders': len(self.router.active_orders),
                'total_processed': self.total_orders_processed,
                'execution_stats': router_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check for any issues
            issues = []
            
            # Check execution performance
            if router_stats.get('completed_orders', 0) > 0:
                avg_slippage = router_stats.get('average_slippage', 0)
                if avg_slippage > 200:  # More than 2% slippage
                    issues.append(f"High average slippage: {avg_slippage:.1f} bps")
            
            # Check for stuck orders
            if len(self.router.active_orders) > 10:
                issues.append(f"High number of active orders: {len(self.router.active_orders)}")
            
            health_status['issues'] = issues
            health_status['status'] = 'warning' if issues else health_status['status']
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 