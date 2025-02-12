"""
CCXT-based exchange base implementation.
"""
from abc import ABC
from typing import Dict, List, Optional, Any
from decimal import Decimal
import ccxt.async_support as ccxt
from loguru import logger

from .base import BaseExchange, Balance, OHLCV


class CCXTExchange(BaseExchange):
    """Base class for CCXT-based exchange implementations."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ):
        """
        Initialize CCXT exchange.

        Args:
            exchange_id: CCXT exchange ID
            api_key: API key
            api_secret: API secret
            testnet: Whether to use testnet
        """
        super().__init__(api_key, api_secret)
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.exchange: Optional[ccxt.Exchange] = None
        self._markets = {}

    async def initialize(self) -> None:
        """Initialize exchange connection."""
        try:
            # Create CCXT exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })

            # Configure testnet if enabled
            if self.testnet:
                self.exchange.set_sandbox_mode(True)

            # Load markets
            self._markets = await self.exchange.load_markets()
            logger.info(f"Initialized {self.exchange_id} exchange")

        except Exception as e:
            logger.error(f"Error initializing {self.exchange_id} exchange: {e}")
            raise

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
            logger.error(f"Error fetching balances: {e}")
            raise

    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return Decimal(str(ticker['last'])) if ticker['last'] else None

        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

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
            logger.error(f"Error calculating portfolio value: {e}")
            raise

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV candles."""
        try:
            # Fetch candles from exchange
            candles = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            # Convert to OHLCV objects
            return [OHLCV.from_list(candle) for candle in candles]

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """Execute trade."""
        try:
            # Prepare order parameters
            params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': float(amount)
            }

            if price and order_type == 'limit':
                params['price'] = float(price)

            # Create and execute order
            order = await self.exchange.create_order(**params)

            # Wait for order to complete if market order
            if order_type == 'market':
                while True:
                    order = await self.exchange.fetch_order(order['id'], symbol)
                    if order['status'] in ['closed', 'canceled', 'expired', 'rejected']:
                        break
                    await asyncio.sleep(0.5)

            return order

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise

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
            logger.error(f"Error fetching positions: {e}")
            raise

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
                orders = await self.exchange.fetch_orders(symbol, **params)
            else:
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
            logger.error(f"Error fetching order history: {e}")
            raise

    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees."""
        try:
            trading_fees = await self.exchange.fetch_trading_fees()
            return {
                symbol: Decimal(str(fees.get('taker', 0)))
                for symbol, fees in trading_fees.items()
            }

        except Exception as e:
            logger.error(f"Error fetching trading fees: {e}")
            return {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.exchange:
            await self.exchange.close()