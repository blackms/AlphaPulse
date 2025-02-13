"""
CCXT library adapter implementation.
"""
import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
import ccxt.async_support as ccxt
from loguru import logger

from ..interfaces import (
    BaseExchange,
    ExchangeConfiguration,
    ExchangeError,
    ConnectionError,
    OrderError,
    MarketDataError
)
from ..base import Balance, OHLCV


class CCXTAdapter(BaseExchange):
    """
    Adapter for CCXT library integration.
    
    This class adapts the CCXT library interface to our exchange interfaces,
    providing a consistent API while isolating the CCXT dependency.
    """
    
    def __init__(self, exchange_id: str, config: ExchangeConfiguration):
        """
        Initialize CCXT adapter.
        
        Args:
            exchange_id: CCXT exchange ID
            config: Exchange configuration
        """
        self.exchange_id = exchange_id
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self._markets = {}
        
    async def initialize(self) -> None:
        """Initialize exchange connection."""
        try:
            # Create CCXT exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    **self.config.options
                }
            })
            
            # Configure testnet if enabled
            if self.config.testnet:
                self.exchange.set_sandbox_mode(True)
            
            # Load markets
            self._markets = await self.exchange.load_markets()
            logger.info(f"Initialized {self.exchange_id} exchange")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize {self.exchange_id}: {str(e)}")
    
    async def close(self) -> None:
        """Close exchange connection."""
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.error(f"Error closing {self.exchange_id} connection: {str(e)}")
    
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return Decimal(str(ticker['last'])) if ticker['last'] else None
        except Exception as e:
            raise MarketDataError(f"Failed to fetch ticker for {symbol}: {str(e)}")
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV candles."""
        try:
            candles = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            return [OHLCV.from_list(candle) for candle in candles]
        except Exception as e:
            raise MarketDataError(f"Failed to fetch OHLCV data: {str(e)}")
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """Execute trade order."""
        try:
            params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': float(amount)
            }
            
            if price and order_type == 'limit':
                params['price'] = float(price)
            
            order = await self.exchange.create_order(**params)
            
            # Wait for market orders to complete
            if order_type == 'market':
                while True:
                    order = await self.exchange.fetch_order(order['id'], symbol)
                    if order['status'] in ['closed', 'canceled', 'expired', 'rejected']:
                        break
                    await asyncio.sleep(0.5)
            
            return order
            
        except Exception as e:
            raise OrderError(f"Failed to execute trade: {str(e)}")
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        try:
            params = {}
            if since:
                params['since'] = since
            if limit:
                params['limit'] = limit
            
            if symbol:
                return await self.exchange.fetch_orders(symbol, **params)
            
            # Fetch orders for all symbols
            orders = []
            for market in self._markets:
                try:
                    symbol_orders = await self.exchange.fetch_orders(market, **params)
                    orders.extend(symbol_orders)
                except Exception as e:
                    logger.warning(f"Error fetching orders for {market}: {e}")
            
            return orders
            
        except Exception as e:
            raise OrderError(f"Failed to fetch order history: {str(e)}")
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances."""
        try:
            response = await self.exchange.fetch_balance()
            balances = {}
            
            for asset, data in response['total'].items():
                if data > 0:
                    balances[asset] = Balance(
                        total=Decimal(str(data)),
                        available=Decimal(str(response['free'].get(asset, 0))),
                        locked=Decimal(str(response['used'].get(asset, 0)))
                    )
            
            return balances
            
        except Exception as e:
            raise ExchangeError(f"Failed to fetch balances: {str(e)}")
    
    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        try:
            balances = await self.get_balances()
            total_value = Decimal('0')
            
            for asset, balance in balances.items():
                if asset == 'USDT':  # Base currency
                    total_value += balance.total
                else:
                    price = await self.get_ticker_price(f"{asset}/USDT")
                    if price:
                        total_value += balance.total * price
            
            return total_value
            
        except Exception as e:
            raise ExchangeError(f"Failed to calculate portfolio value: {str(e)}")
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        try:
            positions = {}
            balances = await self.get_balances()
            
            for asset, balance in balances.items():
                if balance.total > 0:
                    price = await self.get_ticker_price(f"{asset}/USDT")
                    if price:
                        positions[asset] = {
                            'symbol': asset,
                            'quantity': float(balance.total),
                            'entry_price': None,  # Not available in spot
                            'current_price': float(price),
                            'unrealized_pnl': None,  # Not available in spot
                            'liquidation_price': None  # Not available in spot
                        }
            
            return positions
            
        except Exception as e:
            raise ExchangeError(f"Failed to fetch positions: {str(e)}")
    
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees."""
        try:
            trading_fees = await self.exchange.fetch_trading_fees()
            return {
                symbol: Decimal(str(fees.get('taker', 0)))
                for symbol, fees in trading_fees.items()
            }
        except Exception as e:
            logger.warning(f"Failed to fetch trading fees: {str(e)}")
            return {}
    
    async def get_average_entry_price(self, symbol: str) -> Optional[Decimal]:
        """Calculate average entry price for symbol."""
        try:
            orders = await self.get_order_history(symbol=symbol)
            if not orders:
                return None
            
            total_quantity = Decimal('0')
            total_value = Decimal('0')
            
            for order in orders:
                if order['side'] == 'buy' and order['status'] == 'filled':
                    quantity = Decimal(str(order['amount']))
                    price = Decimal(str(order['price']))
                    total_quantity += quantity
                    total_value += quantity * price
            
            if total_quantity == 0:
                return None
            
            return total_value / total_quantity
            
        except Exception as e:
            logger.error(f"Error calculating average entry price for {symbol}: {str(e)}")
            return None