"""
Binance exchange implementation.
"""
from decimal import Decimal
from typing import Dict, List, Optional
import asyncio
from loguru import logger

from .ccxt_base import CCXTExchange
from .base import Balance, OHLCV
from .credentials.manager import credentials_manager


class BinanceExchange(CCXTExchange):
    """Binance exchange implementation."""
    
    def __init__(self, testnet: bool = False):
        """Initialize Binance exchange.
        
        Args:
            testnet: Whether to use testnet
        """
        # Get credentials from manager
        creds = credentials_manager.get_credentials('binance')
        if creds:
            api_key = creds.api_key
            api_secret = creds.api_secret
            # Override testnet from credentials if provided
            testnet = testnet or creds.testnet
        else:
            api_key = ""
            api_secret = ""
        
        super().__init__(
            exchange_id='binance',
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
    
    async def initialize(self) -> None:
        """Initialize Binance exchange connection."""
        await super().initialize()
        
        # Set additional Binance-specific options
        if self.exchange:
            self.exchange.options.update({
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,  # Extend receive window for high latency
                'fetchMarkets': True,  # Load markets on initialization
            })
            
            if self.testnet:
                # Set testnet-specific endpoints
                self.exchange.urls.update({
                    'test': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    },
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                        'v1': 'https://testnet.binance.vision/api/v1',
                        'v3': 'https://testnet.binance.vision/api/v3',
                        'spot': {
                            'public': 'https://testnet.binance.vision/api/v3',
                            'private': 'https://testnet.binance.vision/api/v3'
                        },
                        'future': {
                            'public': 'https://testnet.binancefuture.com/fapi/v1',
                            'private': 'https://testnet.binancefuture.com/fapi/v1',
                            'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                            'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                            'dapiPublic': 'https://testnet.binancefuture.com/dapi/v1',
                            'dapiPrivate': 'https://testnet.binancefuture.com/dapi/v1'
                        }
                    }
                })
    
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol."""
        try:
            # Use v3 API endpoint directly for testnet
            if self.testnet:
                response = await self.exchange.publicGetTickerPrice({'symbol': symbol})
                if 'price' in response:
                    return Decimal(str(response['price']))
                return None
            
            # Use standard implementation for mainnet
            return await super().get_ticker_price(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets."""
        try:
            # Get base implementation
            balances = await super().get_balances()
            
            # Add Binance-specific processing if needed
            # For example, handle staking balances
            account = await self.exchange.fetch_balance()
            if 'info' in account:
                for asset in balances:
                    # Add staking balance if available
                    staking = Decimal('0')
                    for item in account['info'].get('userAssets', []):
                        if item['asset'] == asset:
                            staking = Decimal(str(item.get('locked', '0')))
                            balances[asset].locked += staking
                            balances[asset].total += staking
            
            return balances
            
        except Exception as e:
            logger.error(f"Error fetching Binance balances: {e}")
            raise
    
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols."""
        try:
            # Get account trading fees
            response = await self.exchange.fapiPrivateGetAccount()
            
            # Extract commission rates
            fees = {}
            for symbol in await self.exchange.load_markets():
                maker_fee = Decimal(str(response.get('makerCommission', 0))) / Decimal('10000')
                taker_fee = Decimal(str(response.get('takerCommission', 0))) / Decimal('10000')
                fees[symbol] = max(maker_fee, taker_fee)  # Use higher fee to be conservative
            
            return fees
            
        except Exception as e:
            logger.error(f"Error fetching Binance trading fees: {e}")
            # Fall back to default implementation
            return await super().get_trading_fees()
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV data from Binance."""
        try:
            # Validate timeframe
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            if timeframe not in valid_timeframes:
                raise ValueError(
                    f"Invalid timeframe '{timeframe}'. Must be one of: {valid_timeframes}"
                )
            
            # Get OHLCV data
            return await super().fetch_ohlcv(symbol, timeframe, since, limit)
            
        except Exception as e:
            logger.error(f"Error fetching Binance OHLCV data: {e}")
            raise
            
    async def close(self) -> None:
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
            
    def __del__(self):
        """Ensure resources are released."""
        if self.exchange:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception as e:
                logger.error(f"Error closing Binance exchange: {e}")