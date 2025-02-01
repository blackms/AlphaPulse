"""
Binance exchange connector implementation using CCXT.
"""
from decimal import Decimal
from typing import Dict, Optional
import ccxt.async_support as ccxt
from loguru import logger

from alpha_pulse.exchange_conn.interface import ExchangeConnector, Balance
from alpha_pulse.monitoring.metrics import track_latency, API_LATENCY


class BinanceConnector(ExchangeConnector):
    """Binance exchange connector implementation."""
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        """Initialize Binance connector.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet
        """
        super().__init__(api_key, api_secret, testnet)
        
        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'testnet': testnet,
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.exchange.close()
    
    @track_latency(API_LATENCY.labels(endpoint='get_balances'))
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets.
        
        Returns:
            Dict mapping asset symbols to their balances
        """
        try:
            account = await self.exchange.fetch_balance()
            balances = {}
            
            # Get all non-zero balances
            for currency, balance in account['total'].items():
                if balance > 0:
                    # Get price in USDT for value calculation
                    in_base = Decimal('0')
                    if currency != 'USDT':
                        try:
                            ticker = await self.get_ticker_price(f"{currency}/USDT")
                            if ticker:
                                in_base = Decimal(str(balance)) * ticker
                        except Exception as e:
                            logger.warning(f"Could not get price for {currency}/USDT: {e}")
                    else:
                        in_base = Decimal(str(balance))
                    
                    balances[currency] = Balance(
                        free=Decimal(str(account['free'].get(currency, 0))),
                        locked=Decimal(str(account['used'].get(currency, 0))),
                        total=Decimal(str(balance)),
                        in_base_currency=in_base
                    )
            
            return balances
        except Exception as e:
            logger.error(f"Error fetching balances: {e}")
            raise
    
    @track_latency(API_LATENCY.labels(endpoint='get_ticker_price'))
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Current price or None if not available
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return Decimal(str(ticker['last'])) if ticker['last'] else None
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    @track_latency(API_LATENCY.labels(endpoint='get_exchange_info'))
    async def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules.
        
        Returns:
            Dict containing exchange information
        """
        try:
            return await self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            raise
    
    @track_latency(API_LATENCY.labels(endpoint='get_trading_fees'))
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols.
        
        Returns:
            Dict mapping symbols to their trading fees
        """
        try:
            fees = await self.exchange.fetch_trading_fees()
            return {
                symbol: Decimal(str(fee.get('taker', 0)))
                for symbol, fee in fees.items()
            }
        except Exception as e:
            logger.error(f"Error fetching trading fees: {e}")
            raise
    
    @track_latency(API_LATENCY.labels(endpoint='validate_api_keys'))
    async def validate_api_keys(self) -> bool:
        """Validate API keys by attempting to access private endpoints.
        
        Returns:
            True if keys are valid, False otherwise
        """
        try:
            await self.exchange.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False