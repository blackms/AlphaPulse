"""
Bybit exchange implementation.
"""
from decimal import Decimal
from typing import Dict, List, Optional

from loguru import logger

from .base import CCXTExchange, Balance, OHLCV
from .credentials.manager import credentials_manager


class BybitExchange(CCXTExchange):
    """Bybit exchange implementation."""
    
    def __init__(self, testnet: bool = False):
        """Initialize Bybit exchange.
        
        Args:
            testnet: Whether to use testnet
        """
        # Get credentials from manager
        creds = credentials_manager.get_credentials('bybit')
        if creds:
            api_key = creds.api_key
            api_secret = creds.api_secret
            # Override testnet from credentials if provided
            testnet = testnet or creds.testnet
        else:
            api_key = ""
            api_secret = ""
        
        super().__init__(
            exchange_id='bybit',
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
    
    async def initialize(self) -> None:
        """Initialize Bybit exchange connection."""
        await super().initialize()
        
        # Set additional Bybit-specific options
        if self.exchange:
            self.exchange.options.update({
                'defaultType': 'spot',
                'recvWindow': 60000,  # Extend receive window for high latency
            })
            
            if self.testnet:
                # Set testnet-specific endpoints
                self.exchange.urls.update({
                    'api': {
                        'public': 'https://api-testnet.bybit.com',
                        'private': 'https://api-testnet.bybit.com',
                    }
                })
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets."""
        try:
            # Get base implementation
            balances = await super().get_balances()
            
            # Add Bybit-specific processing if needed
            # For example, handle wallet types (SPOT, DERIVATIVES, etc.)
            account = await self.exchange.fetch_balance()
            if 'info' in account:
                for wallet_type in ['SPOT', 'DERIVATIVES']:
                    wallet_info = account['info'].get(wallet_type, {})
                    for coin in wallet_info.get('coins', []):
                        asset = coin['coin']
                        if asset in balances:
                            # Add any additional locked/staked balances
                            locked = Decimal(str(coin.get('locked', '0')))
                            balances[asset].locked += locked
                            balances[asset].total += locked
            
            return balances
            
        except Exception as e:
            logger.error(f"Error fetching Bybit balances: {e}")
            raise
    
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols."""
        try:
            # Get account trading fees
            response = await self.exchange.privateGetAccount()
            
            # Extract commission rates
            fees = {}
            markets = await self.exchange.load_markets()
            
            # Bybit has tiered fees based on trading volume
            # We'll use the base tier fees to be conservative
            base_maker_fee = Decimal('0.001')  # 0.1%
            base_taker_fee = Decimal('0.001')  # 0.1%
            
            for symbol in markets:
                # Check if symbol has specific fees in the response
                symbol_info = response.get('result', {}).get('symbols', {}).get(symbol, {})
                maker_fee = Decimal(str(symbol_info.get('makerFee', base_maker_fee)))
                taker_fee = Decimal(str(symbol_info.get('takerFee', base_taker_fee)))
                fees[symbol] = max(maker_fee, taker_fee)  # Use higher fee to be conservative
            
            return fees
            
        except Exception as e:
            logger.error(f"Error fetching Bybit trading fees: {e}")
            # Fall back to default implementation
            return await super().get_trading_fees()
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV data from Bybit."""
        try:
            # Validate timeframe
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']
            if timeframe not in valid_timeframes:
                raise ValueError(
                    f"Invalid timeframe '{timeframe}'. Must be one of: {valid_timeframes}"
                )
            
            # Bybit has a max limit of 1000 candles per request
            if limit and limit > 1000:
                logger.warning("Bybit has a maximum limit of 1000 candles per request")
                limit = 1000
            
            # Get OHLCV data
            return await super().fetch_ohlcv(symbol, timeframe, since, limit)
            
        except Exception as e:
            logger.error(f"Error fetching Bybit OHLCV data: {e}")
            raise

    async def validate_api_keys(self) -> bool:
        """Validate API keys by attempting to access private endpoints."""
        try:
            # Bybit validates keys by attempting to fetch wallet balance
            await self.exchange.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Bybit API key validation failed: {e}")
            return False

    async def close(self) -> None:
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()

    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        try:
            balances = await self.get_balances()
            total_value = Decimal('0')
            
            for asset, balance in balances.items():
                if balance.total > 0:
                    if asset == 'USDT':
                        total_value += balance.total
                    else:
                        price = await self.get_ticker_price(f"{asset}/USDT")
                        if price:
                            total_value += balance.total * price
            
            logger.debug(f"Total portfolio value: {total_value} USDT")
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            raise

    async def execute_trade(
        self,
        asset: str,
        amount: Decimal,
        side: str,
        order_type: str = "market"
    ) -> Dict:
        """Execute a trade on Bybit."""
        try:
            symbol = f"{asset}/USDT"
            
            # Convert amount to asset quantity using current price
            price = await self.get_ticker_price(symbol)
            if not price:
                raise ValueError(f"Could not get price for {symbol}")
                
            quantity = amount / price
            
            # Place order
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side.lower(),
                amount=float(quantity),
                params={}
            )
            
            logger.info(f"Executed {side} order for {quantity} {asset} at {price} USDT")
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise