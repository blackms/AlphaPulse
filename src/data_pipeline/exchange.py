import ccxt
from datetime import datetime
from typing import Dict, Optional, List, Any
from loguru import logger

from config.settings import settings
from .interfaces import IExchange, IExchangeFactory

class CCXTExchange(IExchange):
    def __init__(self, exchange: ccxt.Exchange):
        self._exchange = exchange
        
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Any]:
        try:
            since_ts = int(since.timestamp() * 1000) if since else None
            return self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise
            
    def fetch_ticker(self, symbol: str) -> Dict:
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            logger.debug(f"Fetched ticker for {symbol}: {ticker}")
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {str(e)}")
            raise
            
    def load_markets(self) -> None:
        try:
            self._exchange.load_markets()
        except Exception as e:
            logger.error(f"Failed to load markets: {str(e)}")
            raise

class CCXTExchangeFactory(IExchangeFactory):
    def __init__(self):
        self._exchanges: Dict[str, IExchange] = {}
        
    def create_exchange(self, exchange_id: str) -> IExchange:
        if exchange_id in self._exchanges:
            return self._exchanges[exchange_id]
            
        try:
            # Get exchange configuration from settings
            exchange_config = settings.EXCHANGE_CONFIGS.get(exchange_id, {})
            
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class(exchange_config)
            
            # Create wrapped exchange
            wrapped_exchange = CCXTExchange(exchange)
            
            # Load markets to test connection
            wrapped_exchange.load_markets()
            
            self._exchanges[exchange_id] = wrapped_exchange
            logger.info(f"Successfully initialized exchange: {exchange_id}")
            return wrapped_exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {str(e)}")
            raise