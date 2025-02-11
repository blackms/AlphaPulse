"""
Data pipeline interfaces and data models.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Any, List


@dataclass
class MarketData:
    """Market data model."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    vwap: Optional[Decimal] = None
    trades: Optional[int] = None
    source: Optional[str] = None


@dataclass
class FundamentalData:
    """Fundamental data model."""
    symbol: str
    timestamp: datetime
    metadata: Dict[str, Any]  # Market cap, sector, etc.
    financial_ratios: Dict[str, float]  # P/E, P/B, etc.
    balance_sheet: Dict[str, float]  # Assets, liabilities, etc.
    income_statement: Dict[str, float]  # Revenue, net income, etc.
    cash_flow: Dict[str, float]  # Operating cash flow, free cash flow, etc.
    source: Optional[str] = None


@dataclass
class SentimentData:
    """Sentiment data model."""
    symbol: str
    timestamp: datetime
    news_sentiment: float  # -1 to 1
    social_sentiment: float  # -1 to 1
    analyst_sentiment: Optional[float] = None  # -1 to 1
    source_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    source: Optional[str] = None


@dataclass
class TechnicalIndicators:
    """Technical indicators model."""
    symbol: str
    timestamp: datetime
    trend: Dict[str, Any]  # SMA, EMA, MACD
    momentum: Dict[str, Any]  # RSI, Stochastic
    volatility: Dict[str, Any]  # Bollinger Bands, ATR
    volume: Dict[str, Any]  # OBV, AD
    source: Optional[str] = None
    
class DataProvider(Protocol):
    """Protocol for data providers."""
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        ...

    @property
    def provider_type(self) -> str:
        """Get provider type."""
        ...


class IMarketDataProvider(DataProvider):
    """Interface for market data providers."""
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """Get historical market data."""
        pass

    @abstractmethod
    async def get_real_time_data(self, symbol: str) -> MarketData:
        """Get real-time market data."""
        pass


class IFundamentalDataProvider(DataProvider):
    """Interface for fundamental data providers."""
    
    @abstractmethod
    async def get_financial_statements(
        self,
        symbol: str,
        start_time: Optional[datetime] = None
    ) -> FundamentalData:
        """Get financial statements data."""
        pass

    @abstractmethod
    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile data."""
        pass


class ISentimentDataProvider(DataProvider):
    """Interface for sentiment data providers."""
    
    @abstractmethod
    async def get_news_sentiment(
        self,
        symbol: str,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get news sentiment data."""
        pass

    @abstractmethod
    async def get_social_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get social media sentiment data."""
        pass


class ITechnicalAnalysisProvider(DataProvider):
    """Interface for technical analysis providers."""
    
    @abstractmethod
    def calculate_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str]
    ) -> Dict[str, pd.Series]:
        """Calculate technical indicators."""
        pass


class IDataManager(ABC):
    """Interface for data managers."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the data manager."""
        pass

    @abstractmethod
    async def get_market_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d"
    ) -> Dict[str, List[MarketData]]:
        """Get market data for multiple symbols."""
        pass

    @abstractmethod
    async def get_fundamental_data(
        self,
        symbols: List[str]
    ) -> Dict[str, FundamentalData]:
        """Get fundamental data for multiple symbols."""
        pass

    @abstractmethod
    async def get_sentiment_data(
        self,
        symbols: List[str]
    ) -> Dict[str, SentimentData]:
        """Get sentiment data for multiple symbols."""
        pass

    @abstractmethod
    def get_technical_indicators(
        self,
        market_data: Dict[str, List[MarketData]]
    ) -> Dict[str, TechnicalIndicators]:
        """Calculate technical indicators for market data."""
        pass


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class DataFetchError(Exception):
    """Exception raised for data fetching errors."""
    pass


class ProviderError(Exception):
    """Exception raised for provider-specific errors."""
    pass