from decimal import Decimal
from typing import List, Optional
from alpha_pulse.exchanges.base import BaseExchange
from alpha_pulse.hedging.common.interfaces import MarketDataProvider as IPositionFetcher
from alpha_pulse.hedging.common.types import SpotPosition, FuturesPosition

class ExchangePositionFetcher(IPositionFetcher):
    """Fetches position data from an exchange."""
    
    def __init__(self, exchange: BaseExchange):
        """
        Initialize the position fetcher.
        
        Args:
            exchange: Exchange connector instance
        """
        self.exchange = exchange
    
    async def get_spot_positions(self) -> List[SpotPosition]:
        """Fetch current spot positions from exchange."""
        raw_positions = await self.exchange.get_spot_positions()
        spot_positions = []
        
        for pos in raw_positions:
            # Handle None values when converting to Decimal
            avg_price = Decimal('0')
            if pos['avgPrice'] is not None:
                avg_price = Decimal(str(pos['avgPrice']))
            
            current_price = None
            if pos['currentPrice'] is not None:
                current_price = Decimal(str(pos['currentPrice']))
            
            spot_positions.append(
                SpotPosition(
                    symbol=pos['symbol'],
                    quantity=Decimal(str(pos['quantity'])),
                    avg_price=avg_price,
                    current_price=current_price
                )
            )
        
        return spot_positions
    
    async def get_futures_positions(self) -> List[FuturesPosition]:
        """Fetch current futures positions from exchange."""
        raw_positions = await self.exchange.get_futures_positions()
        futures_positions = []
        
        for pos in raw_positions:
            # Handle None values when converting to Decimal
            current_price = None
            if pos['currentPrice'] is not None:
                current_price = Decimal(str(pos['currentPrice']))
            
            futures_positions.append(
                FuturesPosition(
                    symbol=pos['symbol'],
                    quantity=Decimal(str(pos['quantity'])),
                    side=pos['side'],
                    entry_price=Decimal(str(pos['entryPrice'])),
                    leverage=Decimal(str(pos['leverage'])),
                    margin_used=Decimal(str(pos['marginUsed'])),
                    current_price=current_price
                )
            )
        
        return futures_positions
    
    async def get_current_price(
        self,
        symbol: str,
        is_futures: bool = False
    ) -> Optional[Decimal]:
        """Get current price for a symbol."""
        try:
            price = await self.exchange.get_current_price(
                symbol,
                is_futures=is_futures
            )
            return Decimal(str(price)) if price is not None else None
        except Exception:
            return None