"""
Paper trading actuator for simulating trade execution.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from decimal import Decimal
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    value: float
    fees: float
    status: str
    metadata: Dict[str, Any]


class PaperActuator:
    """
    Paper trading actuator that simulates trade execution.
    Maintains a simulated portfolio without placing real trades.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize paper trading actuator."""
        self.config = config or {}
        self.paper_mode = True  # Always true for paper actuator
        self.slippage = self.config.get("slippage", 0.001)  # 0.1% default slippage
        self.fee_rate = self.config.get("fee_rate", 0.001)  # 0.1% default fee
        self.initial_balance = Decimal(str(self.config.get("initial_balance", 1000000)))
        
        # Initialize portfolio state
        self.cash_balance = self.initial_balance
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[PaperTrade] = []
        
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """
        Simulate trade execution.
        
        Args:
            symbol: Trading symbol
            side: Trade side ("buy" or "sell")
            quantity: Trade quantity
            price: Limit price (optional)
            order_type: Order type ("market" or "limit")
            
        Returns:
            Dictionary containing trade execution details
        """
        try:
            # Simulate network latency
            await asyncio.sleep(0.1)
            
            # Get current market price (simulated)
            current_price = price or await self._get_market_price(symbol)
            
            # Apply slippage to simulate market impact
            executed_price = self._apply_slippage(current_price, side)
            
            # Calculate trade value and fees
            trade_value = Decimal(str(quantity)) * Decimal(str(executed_price))
            fees = trade_value * Decimal(str(self.fee_rate))
            
            # Validate trade
            if not await self._validate_trade(side, trade_value, fees):
                return {
                    "status": "rejected",
                    "reason": "insufficient_funds",
                    "timestamp": datetime.now()
                }
                
            # Update portfolio state
            await self._update_portfolio(symbol, side, quantity, executed_price, trade_value, fees)
            
            # Record trade
            trade = PaperTrade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=executed_price,
                timestamp=datetime.now(),
                value=float(trade_value),
                fees=float(fees),
                status="executed",
                metadata={
                    "order_type": order_type,
                    "slippage": self.slippage,
                    "original_price": price or current_price
                }
            )
            self.trade_history.append(trade)
            
            return {
                "status": "executed",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": executed_price,
                "value": float(trade_value),
                "fees": float(fees),
                "timestamp": trade.timestamp
            }
            
        except Exception as e:
            logger.error(f"Paper trade execution error: {str(e)}")
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now()
            }
            
    async def get_portfolio_value(self) -> float:
        """
        Get current portfolio value.
        
        Returns:
            Total portfolio value including cash
        """
        total_value = self.cash_balance
        
        for symbol, position in self.positions.items():
            current_price = await self._get_market_price(symbol)
            position_value = Decimal(str(position['quantity'])) * Decimal(str(current_price))
            total_value += position_value
            
        return float(total_value)
        
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            Dictionary of current positions
        """
        positions = {}
        
        for symbol, position in self.positions.items():
            current_price = await self._get_market_price(symbol)
            position_value = Decimal(str(position['quantity'])) * Decimal(str(current_price))
            unrealized_pnl = position_value - Decimal(str(position['cost_basis']))
            
            positions[symbol] = {
                'quantity': position['quantity'],
                'avg_entry': position['avg_entry'],
                'cost_basis': position['cost_basis'],
                'current_price': current_price,
                'current_value': float(position_value),
                'unrealized_pnl': float(unrealized_pnl),
                'unrealized_pnl_pct': float(unrealized_pnl / Decimal(str(position['cost_basis'])))
            }
            
        return positions
        
    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PaperTrade]:
        """
        Get trade history with optional filters.
        
        Args:
            symbol: Filter by symbol (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            List of paper trades
        """
        filtered_trades = self.trade_history
        
        if symbol:
            filtered_trades = [t for t in filtered_trades if t.symbol == symbol]
            
        if start_time:
            filtered_trades = [t for t in filtered_trades if t.timestamp >= start_time]
            
        if end_time:
            filtered_trades = [t for t in filtered_trades if t.timestamp <= end_time]
            
        return filtered_trades
        
    async def _get_market_price(self, symbol: str) -> float:
        """Simulate getting current market price."""
        # In a real implementation, this would fetch the actual market price
        # For simulation, we'll use the last known price or a dummy price
        if symbol in self.positions:
            return self.positions[symbol]['last_price']
        return 100.0  # Dummy price for simulation
        
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply simulated slippage to price."""
        slippage_factor = 1 + (self.slippage if side == "buy" else -self.slippage)
        return price * slippage_factor
        
    async def _validate_trade(
        self,
        side: str,
        trade_value: Decimal,
        fees: Decimal
    ) -> bool:
        """Validate trade against available funds."""
        if side == "buy":
            total_cost = trade_value + fees
            return self.cash_balance >= total_cost
        return True
        
    async def _update_portfolio(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        trade_value: Decimal,
        fees: Decimal
    ) -> None:
        """Update portfolio state after trade."""
        if side == "buy":
            # Deduct cash
            self.cash_balance -= (trade_value + fees)
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_entry': price,
                    'cost_basis': float(trade_value),
                    'last_price': price
                }
            else:
                # Update existing position
                current = self.positions[symbol]
                total_quantity = current['quantity'] + quantity
                total_cost = Decimal(str(current['cost_basis'])) + trade_value
                
                self.positions[symbol] = {
                    'quantity': total_quantity,
                    'avg_entry': float(total_cost / Decimal(str(total_quantity))),
                    'cost_basis': float(total_cost),
                    'last_price': price
                }
        else:  # sell
            # Add cash
            self.cash_balance += (trade_value - fees)
            
            # Update position
            if symbol in self.positions:
                current = self.positions[symbol]
                new_quantity = current['quantity'] - quantity
                
                if abs(new_quantity) < 1e-8:  # Position closed
                    del self.positions[symbol]
                else:
                    self.positions[symbol]['quantity'] = new_quantity
                    self.positions[symbol]['last_price'] = price