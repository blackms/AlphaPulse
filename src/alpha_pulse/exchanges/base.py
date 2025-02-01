"""
Base exchange interfaces and common functionality.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any

import ccxt.async_support as ccxt
from loguru import logger

from alpha_pulse.monitoring.metrics import track_latency, API_LATENCY


@dataclass
class Balance:
    """Represents an asset balance in the exchange."""
    free: Decimal  # Available balance
    locked: Decimal  # Balance in orders
    total: Decimal  # Total balance (free + locked)
    in_base_currency: Decimal  # Value in base currency (e.g., USDT)


@dataclass
class OHLCV:
    """Represents OHLCV data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class BaseExchange(ABC):
    """Base class for all exchange implementations."""
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        """Initialize exchange.
        
        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            testnet: Whether to use testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange: Optional[ccxt.Exchange] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.exchange:
            await self.exchange.close()
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize exchange connection."""
        pass
    
    @track_latency(API_LATENCY.labels(endpoint='get_balances'))
    @abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets.
        
        Returns:
            Dict mapping asset symbols to their balances
        """
        pass
    
    @track_latency(API_LATENCY.labels(endpoint='get_ticker_price'))
    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Current price or None if not available
        """
        pass
    
    @track_latency(API_LATENCY.labels(endpoint='get_exchange_info'))
    @abstractmethod
    async def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules.
        
        Returns:
            Dict containing exchange information
        """
        pass
    
    @track_latency(API_LATENCY.labels(endpoint='get_trading_fees'))
    @abstractmethod
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols.
        
        Returns:
            Dict mapping symbols to their trading fees
        """
        pass
    
    @track_latency(API_LATENCY.labels(endpoint='validate_api_keys'))
    @abstractmethod
    async def validate_api_keys(self) -> bool:
        """Validate API keys by attempting to access private endpoints.
        
        Returns:
            True if keys are valid, False otherwise
        """
        pass
    
    @track_latency(API_LATENCY.labels(endpoint='fetch_ohlcv'))
    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV data from exchange.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV objects
        """
        pass


class CCXTExchange(BaseExchange):
    """Base class for CCXT-based exchange implementations."""
    
    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False
    ):
        """Initialize CCXT exchange.
        
        Args:
            exchange_id: CCXT exchange ID
            api_key: Exchange API key
            api_secret: Exchange API secret
            testnet: Whether to use testnet
        """
        super().__init__(api_key, api_secret, testnet)
        self.exchange_id = exchange_id
    
    async def initialize(self) -> None:
        """Initialize CCXT exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'testnet': self.testnet,
                }
            })
            
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                
            logger.info(f"Initialized {self.exchange_id} exchange connection")
            
        except Exception as e:
            logger.error(f"Error initializing {self.exchange_id} exchange: {e}")
            raise
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances for all assets."""
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
    
    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return Decimal(str(ticker['last'])) if ticker['last'] else None
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules."""
        try:
            return await self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            raise
    
    async def get_trading_fees(self) -> Dict[str, Decimal]:
        """Get trading fees for all symbols."""
        try:
            fees = await self.exchange.fetch_trading_fees()
            return {
                symbol: Decimal(str(fee.get('taker', 0)))
                for symbol, fee in fees.items()
            }
        except Exception as e:
            logger.error(f"Error fetching trading fees: {e}")
            raise
    
    async def validate_api_keys(self) -> bool:
        """Validate API keys by attempting to access private endpoints."""
        try:
            await self.exchange.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OHLCV]:
        """Fetch OHLCV data from exchange."""
        try:
            if not self.exchange.has['fetchOHLCV']:
                raise NotImplementedError(
                    f"{self.exchange_id} does not support OHLCV data"
                )
            
            raw_data = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=limit
            )
            
            return [
                OHLCV(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                    open=Decimal(str(candle[1])),
                    high=Decimal(str(candle[2])),
                    low=Decimal(str(candle[3])),
                    close=Decimal(str(candle[4])),
                    volume=Decimal(str(candle[5]))
                )
                for candle in raw_data
            ]
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise