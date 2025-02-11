"""
Core interfaces for the data pipeline system.
Following SOLID principles and Python best practices.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
import pandas as pd


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class FundamentalData:
    """Container for fundamental data."""
    symbol: str
    timestamp: datetime
    financial_ratios: Dict[str, float]
    balance_sheet: Dict[str, float]
    income_statement: Dict[str, float]
    cash_flow: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SentimentData:
    """Container for sentiment data."""
    symbol: str
    timestamp: datetime
    news_sentiment: float
    social_sentiment: float
    analyst_sentiment: float
    source_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TechnicalIndicators:
    """Container for technical indicators."""
    symbol: str
    timestamp: datetime
    trend_indicators: Dict[str, float]
    momentum_indicators: Dict[str, float]
    volatility_indicators: Dict[str, float]
    volume_indicators: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


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