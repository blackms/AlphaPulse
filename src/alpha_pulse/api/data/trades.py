"""Trade data access module."""
from typing import Dict, List, Optional
from datetime import datetime
import logging

from alpha_pulse.execution.broker_interface import BrokerInterface


class TradeDataAccessor:
    """Access trade data."""
    
    def __init__(self):
        """Initialize trade accessor."""
        self.logger = logging.getLogger("alpha_pulse.api.data.trades")
        self.broker = BrokerInterface.get_instance()
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of trade data
        """
        try:
            # Get trade history from broker
            trades = await self.broker.get_trade_history(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            # Transform to API format
            result = []
            for trade in trades:
                result.append({
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "timestamp": trade.timestamp.isoformat(),
                    "status": trade.status,
                    "order_type": trade.order_type,
                    "fees": trade.fees
                })
                
            return result
        except Exception as e:
            self.logger.error(f"Error retrieving trade history: {str(e)}")
            return []