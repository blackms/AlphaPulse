from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

class IExchange(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[datetime] = None, limit: int = 1000) -> List[Any]:
        pass

    @abstractmethod
    def fetch_ticker(self, symbol: str) -> Dict:
        pass

    @abstractmethod
    def load_markets(self) -> None:
        pass

class IDataStorage(ABC):
    @abstractmethod
    def save_ohlcv(self, data: List[Any]) -> None:
        pass

    @abstractmethod
    def get_latest_ohlcv(self, exchange: str, symbol: str) -> Optional[datetime]:
        pass

class IExchangeFactory(ABC):
    @abstractmethod
    def create_exchange(self, exchange_id: str) -> IExchange:
        pass