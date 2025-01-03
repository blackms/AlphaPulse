import ccxt
from typing import Dict, Optional
from loguru import logger
from config.settings import settings

class ExchangeManager:
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        
    def initialize_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Initialize a new exchange connection"""
        try:
            # Get exchange configuration from settings
            exchange_config = settings.EXCHANGE_CONFIGS.get(exchange_id, {})
            
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class(exchange_config)
            
            # Load markets (this also tests the connection)
            exchange.load_markets()
            
            self.exchanges[exchange_id] = exchange
            logger.info(f"Successfully initialized exchange: {exchange_id}")
            return exchange
            
        except ccxt.ExchangeError as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {str(e)}")
            raise
    
    def get_exchange(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """Get an existing exchange connection or create a new one"""
        if exchange_id not in self.exchanges:
            return self.initialize_exchange(exchange_id)
        return self.exchanges[exchange_id]
    
    def get_ticker(self, exchange_id: str, symbol: str) -> Dict:
        """Get current ticker data for a symbol"""
        exchange = self.get_exchange(exchange_id)
        try:
            ticker = exchange.fetch_ticker(symbol)
            logger.debug(f"Fetched ticker for {symbol} from {exchange_id}: {ticker}")
            return ticker
        except ccxt.ExchangeError as e:
            logger.error(f"Failed to fetch ticker for {symbol} from {exchange_id}: {str(e)}")
            raise 