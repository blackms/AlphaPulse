"""
Smart Order Routing (SOR) for AlphaPulse.

Provides intelligent order execution with features like:
- TWAP (Time-Weighted Average Price) execution
- VWAP (Volume-Weighted Average Price) execution  
- Iceberg orders (hidden quantity)
- Liquidity detection and market impact minimization
- Multi-venue routing (when multiple exchanges are available)
- Dynamic slicing based on market conditions
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from loguru import logger

from .broker_interface import BrokerInterface, Order, OrderResult, OrderSide, OrderType, OrderStatus
from alpha_pulse.services.regime_detection_service import RegimeDetectionService
from alpha_pulse.ml.regime.regime_classifier import RegimeInfo, RegimeType


class ExecutionStrategy(Enum):
    """Smart order execution strategies."""
    MARKET = "market"           # Immediate execution
    TWAP = "twap"              # Time-Weighted Average Price
    VWAP = "vwap"              # Volume-Weighted Average Price  
    ICEBERG = "iceberg"        # Large orders with hidden quantity
    ADAPTIVE = "adaptive"       # Adapts strategy based on market conditions
    LIQUIDITY_SEEKING = "liquidity_seeking"  # Seeks liquidity pools


class OrderUrgency(Enum):
    """Order urgency levels."""
    LOW = "low"         # Accept higher slippage for better price
    MEDIUM = "medium"   # Balanced approach
    HIGH = "high"       # Minimize time, accept slippage
    CRITICAL = "critical"  # Immediate execution required


@dataclass
class SmartOrderConfig:
    """Configuration for smart order execution."""
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    max_participation_rate: float = 0.3  # Max % of volume to consume
    max_slice_size: float = 0.1          # Max % of order per slice
    time_horizon_minutes: int = 30       # Time to complete order
    min_slice_interval_seconds: int = 30  # Min time between slices
    max_market_impact_bps: int = 50      # Max acceptable market impact (basis points)
    enable_dark_pools: bool = True       # Use hidden liquidity
    enable_regime_adaptation: bool = True # Adapt to market regime
    slippage_tolerance_bps: int = 100    # Max acceptable slippage


@dataclass
class OrderSlice:
    """A slice of a larger order."""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    target_price: Optional[float] = None
    order_type: OrderType = OrderType.LIMIT
    scheduled_time: Optional[datetime] = None
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SmartOrder:
    """A smart order that can be executed using various strategies."""
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    config: SmartOrderConfig
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_filled_price: Optional[float] = None
    slices: List[OrderSlice] = field(default_factory=list)
    execution_stats: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class MarketDataAnalyzer:
    """Analyzes market data for smart routing decisions."""
    
    def __init__(self):
        self.volume_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.spread_history: Dict[str, List[float]] = {}
    
    def update_market_data(self, symbol: str, price: float, volume: float, spread: float):
        """Update market data for analysis."""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
            self.price_history[symbol] = []
            self.spread_history[symbol] = []
        
        # Keep last 100 data points
        self.volume_history[symbol].append(volume)
        self.price_history[symbol].append(price)
        self.spread_history[symbol].append(spread)
        
        for history in [self.volume_history[symbol], self.price_history[symbol], self.spread_history[symbol]]:
            if len(history) > 100:
                history.pop(0)
    
    def get_average_volume(self, symbol: str, periods: int = 20) -> float:
        """Get average volume over specified periods."""
        if symbol not in self.volume_history:
            return 0.0
        
        recent_volumes = self.volume_history[symbol][-periods:]
        return np.mean(recent_volumes) if recent_volumes else 0.0
    
    def get_volatility(self, symbol: str, periods: int = 20) -> float:
        """Get price volatility over specified periods."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return 0.02  # Default 2%
        
        prices = self.price_history[symbol][-periods:]
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(prices) / np.array(prices[:-1])
        return np.std(returns) if len(returns) > 0 else 0.02
    
    def get_average_spread(self, symbol: str, periods: int = 20) -> float:
        """Get average spread over specified periods."""
        if symbol not in self.spread_history:
            return 0.001  # Default 0.1%
        
        recent_spreads = self.spread_history[symbol][-periods:]
        return np.mean(recent_spreads) if recent_spreads else 0.001
    
    def estimate_market_impact(self, symbol: str, quantity: float, avg_volume: float) -> float:
        """Estimate market impact for a given quantity."""
        if avg_volume <= 0:
            return 0.01  # Default 1%
        
        participation_rate = quantity / avg_volume
        # Simple square-root impact model
        impact = 0.1 * np.sqrt(participation_rate)
        return min(impact, 0.05)  # Cap at 5%


class SmartOrderRouter:
    """
    Smart Order Router that intelligently executes large orders.
    
    Features:
    - Multiple execution strategies (TWAP, VWAP, Iceberg, Adaptive)
    - Market regime awareness
    - Liquidity detection and market impact minimization
    - Dynamic order slicing based on market conditions
    - Real-time adaptation to market changes
    """
    
    def __init__(self, broker: BrokerInterface, regime_service: Optional[RegimeDetectionService] = None):
        """Initialize smart order router."""
        self.broker = broker
        self.regime_service = regime_service
        self.market_analyzer = MarketDataAnalyzer()
        
        # Active orders
        self.active_orders: Dict[str, SmartOrder] = {}
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'completed_orders': 0,
            'average_slippage': 0.0,
            'average_market_impact': 0.0,
            'strategy_performance': {}
        }
        
        logger.info("Smart Order Router initialized")
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        config: SmartOrderConfig,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> str:
        """
        Submit a smart order for execution.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Total quantity to execute
            config: Smart order configuration
            target_price: Target price (optional)
            stop_price: Stop price (optional)
            
        Returns:
            Smart order ID
        """
        order_id = f"SOR_{symbol}_{side.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        smart_order = SmartOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            config=config,
            target_price=target_price,
            stop_price=stop_price
        )
        
        # Store order
        self.active_orders[order_id] = smart_order
        self.execution_stats['total_orders'] += 1
        
        # Start execution
        execution_task = asyncio.create_task(self._execute_smart_order(smart_order))
        self.execution_tasks[order_id] = execution_task
        
        logger.info(f"Submitted smart order {order_id}: {side.value} {quantity} {symbol} using {config.strategy.value}")
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a smart order."""
        if order_id not in self.active_orders:
            return False
        
        # Cancel execution task
        if order_id in self.execution_tasks:
            self.execution_tasks[order_id].cancel()
            del self.execution_tasks[order_id]
        
        # Cancel any pending slices
        smart_order = self.active_orders[order_id]
        for slice_order in smart_order.slices:
            if slice_order.broker_order_id and slice_order.status == OrderStatus.PENDING:
                await self.broker.cancel_order(slice_order.broker_order_id)
        
        smart_order.status = OrderStatus.CANCELLED
        logger.info(f"Cancelled smart order {order_id}")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[SmartOrder]:
        """Get status of a smart order."""
        return self.active_orders.get(order_id)
    
    def update_market_data(self, symbol: str, price: float, volume: float = 0.0, spread: float = 0.001):
        """Update market data for routing decisions."""
        self.market_analyzer.update_market_data(symbol, price, volume, spread)
    
    async def _execute_smart_order(self, smart_order: SmartOrder):
        """Execute a smart order using the specified strategy."""
        try:
            logger.info(f"Starting execution of {smart_order.order_id} using {smart_order.config.strategy.value}")
            
            # Get market regime if available
            regime_info = None
            if self.regime_service and smart_order.config.enable_regime_adaptation:
                regime_info = await self.regime_service.get_current_regime()
                if regime_info:
                    logger.info(f"Adapting execution to {regime_info.regime_type.value} regime")
            
            # Execute based on strategy
            if smart_order.config.strategy == ExecutionStrategy.MARKET:
                await self._execute_market_order(smart_order)
            elif smart_order.config.strategy == ExecutionStrategy.TWAP:
                await self._execute_twap_order(smart_order, regime_info)
            elif smart_order.config.strategy == ExecutionStrategy.VWAP:
                await self._execute_vwap_order(smart_order, regime_info)
            elif smart_order.config.strategy == ExecutionStrategy.ICEBERG:
                await self._execute_iceberg_order(smart_order, regime_info)
            elif smart_order.config.strategy == ExecutionStrategy.ADAPTIVE:
                await self._execute_adaptive_order(smart_order, regime_info)
            elif smart_order.config.strategy == ExecutionStrategy.LIQUIDITY_SEEKING:
                await self._execute_liquidity_seeking_order(smart_order, regime_info)
            
            # Mark as completed
            smart_order.status = OrderStatus.FILLED if smart_order.filled_quantity >= smart_order.total_quantity * 0.95 else OrderStatus.CANCELLED
            self.execution_stats['completed_orders'] += 1
            
            # Calculate execution statistics
            await self._calculate_execution_stats(smart_order)
            
            logger.info(f"Completed execution of {smart_order.order_id}")
            
        except asyncio.CancelledError:
            smart_order.status = OrderStatus.CANCELLED
            logger.info(f"Execution of {smart_order.order_id} was cancelled")
        except Exception as e:
            smart_order.status = OrderStatus.REJECTED
            logger.error(f"Error executing {smart_order.order_id}: {e}")
        finally:
            # Cleanup
            if smart_order.order_id in self.execution_tasks:
                del self.execution_tasks[smart_order.order_id]
    
    async def _execute_market_order(self, smart_order: SmartOrder):
        """Execute order immediately at market price."""
        order = Order(
            symbol=smart_order.symbol,
            side=smart_order.side,
            quantity=smart_order.total_quantity,
            order_type=OrderType.MARKET
        )
        
        result = await self.broker.place_order(order)
        
        if result.success and result.order_id:
            slice_order = OrderSlice(
                slice_id=f"{smart_order.order_id}_slice_1",
                parent_order_id=smart_order.order_id,
                symbol=smart_order.symbol,
                side=smart_order.side,
                quantity=smart_order.total_quantity,
                order_type=OrderType.MARKET,
                broker_order_id=result.order_id,
                status=OrderStatus.FILLED,
                filled_quantity=result.filled_quantity or smart_order.total_quantity,
                filled_price=result.filled_price
            )
            
            smart_order.slices.append(slice_order)
            smart_order.filled_quantity = slice_order.filled_quantity
            smart_order.avg_filled_price = slice_order.filled_price
    
    async def _execute_twap_order(self, smart_order: SmartOrder, regime_info: Optional[RegimeInfo]):
        """Execute order using Time-Weighted Average Price strategy."""
        # Adapt time horizon based on regime
        time_horizon = smart_order.config.time_horizon_minutes
        if regime_info:
            if regime_info.regime_type == RegimeType.VOLATILE:
                time_horizon = int(time_horizon * 0.5)  # Faster execution in volatile markets
            elif regime_info.regime_type == RegimeType.RANGING:
                time_horizon = int(time_horizon * 1.5)  # Slower execution in ranging markets
        
        # Calculate slicing parameters
        slice_interval = smart_order.config.min_slice_interval_seconds
        num_slices = max(1, time_horizon * 60 // slice_interval)
        slice_size = smart_order.total_quantity / num_slices
        
        logger.info(f"TWAP execution: {num_slices} slices of {slice_size:.4f} over {time_horizon} minutes")
        
        # Execute slices
        for i in range(num_slices):
            if smart_order.filled_quantity >= smart_order.total_quantity:
                break
            
            remaining_quantity = smart_order.total_quantity - smart_order.filled_quantity
            current_slice_size = min(slice_size, remaining_quantity)
            
            await self._execute_slice(smart_order, current_slice_size, i + 1)
            
            # Wait for next slice (except for last slice)
            if i < num_slices - 1:
                await asyncio.sleep(slice_interval)
    
    async def _execute_vwap_order(self, smart_order: SmartOrder, regime_info: Optional[RegimeInfo]):
        """Execute order using Volume-Weighted Average Price strategy."""
        # Get historical volume profile (simplified - would use actual VWAP data in production)
        avg_volume = self.market_analyzer.get_average_volume(smart_order.symbol)
        
        if avg_volume <= 0:
            # Fallback to TWAP if no volume data
            await self._execute_twap_order(smart_order, regime_info)
            return
        
        # Calculate participation rate
        max_participation = smart_order.config.max_participation_rate
        if regime_info and regime_info.regime_type == RegimeType.VOLATILE:
            max_participation *= 0.7  # More conservative in volatile markets
        
        # Execute based on volume profile
        target_volume_per_slice = avg_volume * max_participation
        slice_size = min(target_volume_per_slice, smart_order.total_quantity * smart_order.config.max_slice_size)
        
        num_slices = int(np.ceil(smart_order.total_quantity / slice_size))
        slice_interval = smart_order.config.min_slice_interval_seconds
        
        logger.info(f"VWAP execution: {num_slices} slices targeting {max_participation:.1%} participation")
        
        for i in range(num_slices):
            if smart_order.filled_quantity >= smart_order.total_quantity:
                break
            
            remaining_quantity = smart_order.total_quantity - smart_order.filled_quantity
            current_slice_size = min(slice_size, remaining_quantity)
            
            await self._execute_slice(smart_order, current_slice_size, i + 1)
            
            if i < num_slices - 1:
                await asyncio.sleep(slice_interval)
    
    async def _execute_iceberg_order(self, smart_order: SmartOrder, regime_info: Optional[RegimeInfo]):
        """Execute large order with hidden quantity (iceberg strategy)."""
        # Show only a small portion of the order
        visible_size = smart_order.total_quantity * smart_order.config.max_slice_size
        
        # Adapt visible size based on regime
        if regime_info:
            if regime_info.regime_type == RegimeType.VOLATILE:
                visible_size *= 0.5  # Smaller visible size in volatile markets
            elif regime_info.regime_type == RegimeType.BULL:
                visible_size *= 1.5  # Larger visible size in bull markets
        
        visible_size = max(visible_size, smart_order.total_quantity * 0.01)  # Minimum 1%
        
        logger.info(f"Iceberg execution: showing {visible_size:.4f} of {smart_order.total_quantity}")
        
        slice_count = 1
        while smart_order.filled_quantity < smart_order.total_quantity:
            remaining_quantity = smart_order.total_quantity - smart_order.filled_quantity
            current_slice_size = min(visible_size, remaining_quantity)
            
            await self._execute_slice(smart_order, current_slice_size, slice_count)
            slice_count += 1
            
            # Wait before next slice
            if smart_order.filled_quantity < smart_order.total_quantity:
                await asyncio.sleep(smart_order.config.min_slice_interval_seconds)
    
    async def _execute_adaptive_order(self, smart_order: SmartOrder, regime_info: Optional[RegimeInfo]):
        """Execute order using adaptive strategy based on market conditions."""
        # Choose strategy based on market conditions and regime
        volatility = self.market_analyzer.get_volatility(smart_order.symbol)
        avg_volume = self.market_analyzer.get_average_volume(smart_order.symbol)
        
        if regime_info:
            if regime_info.regime_type == RegimeType.VOLATILE:
                # Use conservative TWAP in volatile markets
                await self._execute_twap_order(smart_order, regime_info)
            elif regime_info.regime_type == RegimeType.BULL:
                # Use aggressive VWAP in bull markets
                await self._execute_vwap_order(smart_order, regime_info)
            elif regime_info.regime_type == RegimeType.BEAR:
                # Use iceberg in bear markets to hide intention
                await self._execute_iceberg_order(smart_order, regime_info)
            else:
                # Default to TWAP for ranging markets
                await self._execute_twap_order(smart_order, regime_info)
        else:
            # Fallback logic without regime info
            if volatility > 0.03:  # High volatility
                await self._execute_twap_order(smart_order, regime_info)
            elif avg_volume > smart_order.total_quantity * 10:  # High volume
                await self._execute_vwap_order(smart_order, regime_info)
            else:
                await self._execute_iceberg_order(smart_order, regime_info)
    
    async def _execute_liquidity_seeking_order(self, smart_order: SmartOrder, regime_info: Optional[RegimeInfo]):
        """Execute order seeking best liquidity and minimal market impact."""
        # Start with smaller slices and increase based on market response
        base_slice_size = smart_order.total_quantity * 0.02  # Start with 2%
        
        slice_count = 1
        while smart_order.filled_quantity < smart_order.total_quantity:
            remaining_quantity = smart_order.total_quantity - smart_order.filled_quantity
            
            # Estimate market impact
            avg_volume = self.market_analyzer.get_average_volume(smart_order.symbol)
            estimated_impact = self.market_analyzer.estimate_market_impact(
                smart_order.symbol, base_slice_size, avg_volume
            )
            
            # Adjust slice size based on estimated impact
            if estimated_impact > smart_order.config.max_market_impact_bps / 10000:
                base_slice_size *= 0.8  # Reduce slice size
            else:
                base_slice_size = min(base_slice_size * 1.2, remaining_quantity * 0.1)  # Increase slice size
            
            current_slice_size = min(base_slice_size, remaining_quantity)
            
            await self._execute_slice(smart_order, current_slice_size, slice_count)
            slice_count += 1
            
            # Dynamic wait time based on market conditions
            if smart_order.filled_quantity < smart_order.total_quantity:
                wait_time = max(
                    smart_order.config.min_slice_interval_seconds,
                    int(estimated_impact * 1000)  # Wait longer if high impact
                )
                await asyncio.sleep(wait_time)
    
    async def _execute_slice(self, smart_order: SmartOrder, slice_size: float, slice_number: int):
        """Execute a single slice of the smart order."""
        slice_id = f"{smart_order.order_id}_slice_{slice_number}"
        
        # Determine order type and price
        order_type = OrderType.LIMIT if smart_order.target_price else OrderType.MARKET
        price = smart_order.target_price
        
        # Create broker order
        order = Order(
            symbol=smart_order.symbol,
            side=smart_order.side,
            quantity=slice_size,
            order_type=order_type,
            price=price
        )
        
        # Execute order
        result = await self.broker.place_order(order)
        
        # Create slice record
        slice_order = OrderSlice(
            slice_id=slice_id,
            parent_order_id=smart_order.order_id,
            symbol=smart_order.symbol,
            side=smart_order.side,
            quantity=slice_size,
            target_price=price,
            order_type=order_type,
            broker_order_id=result.order_id,
            status=OrderStatus.FILLED if result.success else OrderStatus.REJECTED,
            filled_quantity=result.filled_quantity or 0.0,
            filled_price=result.filled_price
        )
        
        smart_order.slices.append(slice_order)
        
        # Update smart order
        if result.success:
            smart_order.filled_quantity += slice_order.filled_quantity
            
            # Update average filled price
            if smart_order.avg_filled_price is None:
                smart_order.avg_filled_price = slice_order.filled_price
            elif slice_order.filled_price:
                total_value = (smart_order.filled_quantity - slice_order.filled_quantity) * smart_order.avg_filled_price
                total_value += slice_order.filled_quantity * slice_order.filled_price
                smart_order.avg_filled_price = total_value / smart_order.filled_quantity
            
            logger.info(f"Executed slice {slice_number}: {slice_order.filled_quantity:.4f} at {slice_order.filled_price}")
        else:
            logger.warning(f"Failed to execute slice {slice_number}: {result.error}")
    
    async def _calculate_execution_stats(self, smart_order: SmartOrder):
        """Calculate execution statistics for the completed order."""
        if not smart_order.slices or smart_order.avg_filled_price is None:
            return
        
        # Calculate slippage (if target price was specified)
        slippage = 0.0
        if smart_order.target_price:
            slippage = abs(smart_order.avg_filled_price - smart_order.target_price) / smart_order.target_price
        
        # Calculate execution time
        start_time = smart_order.created_at
        end_time = max(slice_order.created_at for slice_order in smart_order.slices)
        execution_time = (end_time - start_time).total_seconds() / 60  # Minutes
        
        smart_order.execution_stats = {
            'slippage_bps': slippage * 10000,
            'execution_time_minutes': execution_time,
            'num_slices': len(smart_order.slices),
            'fill_rate': smart_order.filled_quantity / smart_order.total_quantity,
            'avg_slice_size': smart_order.total_quantity / len(smart_order.slices) if smart_order.slices else 0
        }
        
        # Update global stats
        self._update_global_stats(smart_order)
    
    def _update_global_stats(self, smart_order: SmartOrder):
        """Update global execution statistics."""
        strategy = smart_order.config.strategy.value
        
        if strategy not in self.execution_stats['strategy_performance']:
            self.execution_stats['strategy_performance'][strategy] = {
                'orders': 0,
                'avg_slippage': 0.0,
                'avg_execution_time': 0.0
            }
        
        strategy_stats = self.execution_stats['strategy_performance'][strategy]
        strategy_stats['orders'] += 1
        
        # Update running averages
        slippage = smart_order.execution_stats.get('slippage_bps', 0)
        execution_time = smart_order.execution_stats.get('execution_time_minutes', 0)
        
        n = strategy_stats['orders']
        strategy_stats['avg_slippage'] = ((n - 1) * strategy_stats['avg_slippage'] + slippage) / n
        strategy_stats['avg_execution_time'] = ((n - 1) * strategy_stats['avg_execution_time'] + execution_time) / n
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    async def shutdown(self):
        """Shutdown the smart order router."""
        # Cancel all active executions
        for order_id in list(self.execution_tasks.keys()):
            await self.cancel_order(order_id)
        
        # Wait for tasks to complete
        if self.execution_tasks:
            await asyncio.gather(*self.execution_tasks.values(), return_exceptions=True)
        
        logger.info("Smart Order Router shutdown complete") 