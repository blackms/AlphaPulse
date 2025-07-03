"""
Enhanced market data models for real data feeds.

Provides:
- Comprehensive market data structures
- Data quality metadata
- Multi-source data aggregation
- Time series data handling
- Financial instrument definitions
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTOCURRENCY = "cryptocurrency"
    BOND = "bond"
    COMMODITY = "commodity"
    INDEX = "index"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"


class MarketSession(Enum):
    """Market session types."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    EXTENDED = "extended"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


class QuoteType(Enum):
    """Quote type indicators."""
    REAL_TIME = "real_time"
    DELAYED = "delayed"
    SNAPSHOT = "snapshot"
    CALCULATED = "calculated"


@dataclass
class DataSource:
    """Data source metadata."""
    provider: str
    feed_name: Optional[str] = None
    timestamp_received: Optional[datetime] = None
    latency_ms: Optional[int] = None
    quality: DataQuality = DataQuality.GOOD
    confidence_score: float = 1.0
    source_id: Optional[str] = None


@dataclass
class PricePoint:
    """Individual price point with metadata."""
    price: Decimal
    size: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    exchange: Optional[str] = None
    condition_codes: Optional[List[str]] = None
    source: Optional[DataSource] = None


@dataclass
class Quote:
    """Bid/Ask quote data."""
    bid: Optional[PricePoint] = None
    ask: Optional[PricePoint] = None
    spread: Optional[Decimal] = None
    mid_price: Optional[Decimal] = None
    quote_time: Optional[datetime] = None
    quote_type: QuoteType = QuoteType.REAL_TIME


@dataclass
class Trade:
    """Individual trade data."""
    price: Decimal
    size: Decimal
    timestamp: datetime
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
    condition_codes: Optional[List[str]] = None
    is_regular_hours: bool = True
    source: Optional[DataSource] = None


@dataclass
class OHLCV:
    """OHLC+Volume data with validation."""
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime
    vwap: Optional[Decimal] = None
    trade_count: Optional[int] = None
    
    def __post_init__(self):
        """Validate OHLC data consistency."""
        if self.high < self.low:
            raise ValueError("High price cannot be less than low price")
        
        if not (self.low <= self.open <= self.high):
            raise ValueError("Open price must be within high-low range")
        
        if not (self.low <= self.close <= self.high):
            raise ValueError("Close price must be within high-low range")
        
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")


@dataclass
class MarketDataPoint:
    """Comprehensive market data point."""
    symbol: str
    timestamp: datetime
    asset_class: AssetClass
    
    # OHLCV data
    ohlcv: Optional[OHLCV] = None
    
    # Quote data
    quote: Optional[Quote] = None
    
    # Recent trades
    trades: List[Trade] = field(default_factory=list)
    
    # Previous session data
    previous_close: Optional[Decimal] = None
    previous_volume: Optional[Decimal] = None
    
    # Market session info
    session: MarketSession = MarketSession.REGULAR
    is_market_open: bool = True
    
    # Calculated fields
    change: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None
    
    # Data quality metadata
    source: Optional[DataSource] = None
    data_quality: DataQuality = DataQuality.GOOD
    validation_errors: List[str] = field(default_factory=list)
    
    # Additional metadata
    exchange: Optional[str] = None
    currency: str = "USD"
    multiplier: float = 1.0
    tick_size: Optional[Decimal] = None
    
    def __post_init__(self):
        """Calculate derived fields and validate data."""
        self._calculate_change()
        self._validate_data_consistency()
    
    def _calculate_change(self):
        """Calculate price change and percentage."""
        if self.ohlcv and self.previous_close:
            self.change = self.ohlcv.close - self.previous_close
            if self.previous_close > 0:
                self.change_percent = (self.change / self.previous_close) * 100
    
    def _validate_data_consistency(self):
        """Validate data consistency and quality."""
        errors = []
        
        # Check timestamp
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Validate OHLCV if present
        if self.ohlcv:
            try:
                # OHLCV validation is done in __post_init__
                pass
            except ValueError as e:
                errors.append(f"OHLCV validation error: {str(e)}")
        
        # Validate quote data
        if self.quote and self.quote.bid and self.quote.ask:
            if self.quote.bid.price >= self.quote.ask.price:
                errors.append("Bid price is greater than or equal to ask price")
        
        # Check for extreme price movements
        if self.change_percent and abs(self.change_percent) > 50:
            errors.append(f"Extreme price movement: {self.change_percent:.2f}%")
        
        self.validation_errors = errors
        
        # Update data quality based on errors
        if len(errors) > 3:
            self.data_quality = DataQuality.POOR
        elif len(errors) > 1:
            self.data_quality = DataQuality.FAIR
        elif len(errors) > 0:
            self.data_quality = DataQuality.GOOD
        else:
            self.data_quality = DataQuality.EXCELLENT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'asset_class': self.asset_class.value,
            'ohlcv': {
                'open': float(self.ohlcv.open),
                'high': float(self.ohlcv.high),
                'low': float(self.ohlcv.low),
                'close': float(self.ohlcv.close),
                'volume': float(self.ohlcv.volume),
                'vwap': float(self.ohlcv.vwap) if self.ohlcv.vwap else None,
                'trade_count': self.ohlcv.trade_count
            } if self.ohlcv else None,
            'quote': {
                'bid': float(self.quote.bid.price) if self.quote and self.quote.bid else None,
                'ask': float(self.quote.ask.price) if self.quote and self.quote.ask else None,
                'spread': float(self.quote.spread) if self.quote and self.quote.spread else None,
                'mid_price': float(self.quote.mid_price) if self.quote and self.quote.mid_price else None
            } if self.quote else None,
            'previous_close': float(self.previous_close) if self.previous_close else None,
            'change': float(self.change) if self.change else None,
            'change_percent': float(self.change_percent) if self.change_percent else None,
            'session': self.session.value,
            'is_market_open': self.is_market_open,
            'exchange': self.exchange,
            'currency': self.currency,
            'data_quality': self.data_quality.value,
            'validation_errors': self.validation_errors,
            'source': {
                'provider': self.source.provider,
                'quality': self.source.quality.value,
                'confidence_score': self.source.confidence_score
            } if self.source else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketDataPoint':
        """Create instance from dictionary."""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        # Create OHLCV if present
        ohlcv = None
        if data.get('ohlcv'):
            ohlc_data = data['ohlcv']
            ohlcv = OHLCV(
                open=Decimal(str(ohlc_data['open'])),
                high=Decimal(str(ohlc_data['high'])),
                low=Decimal(str(ohlc_data['low'])),
                close=Decimal(str(ohlc_data['close'])),
                volume=Decimal(str(ohlc_data['volume'])),
                timestamp=timestamp,
                vwap=Decimal(str(ohlc_data['vwap'])) if ohlc_data.get('vwap') else None,
                trade_count=ohlc_data.get('trade_count')
            )
        
        # Create quote if present
        quote = None
        if data.get('quote'):
            quote_data = data['quote']
            bid = PricePoint(price=Decimal(str(quote_data['bid']))) if quote_data.get('bid') else None
            ask = PricePoint(price=Decimal(str(quote_data['ask']))) if quote_data.get('ask') else None
            quote = Quote(
                bid=bid,
                ask=ask,
                spread=Decimal(str(quote_data['spread'])) if quote_data.get('spread') else None,
                mid_price=Decimal(str(quote_data['mid_price'])) if quote_data.get('mid_price') else None
            )
        
        # Create data source if present
        source = None
        if data.get('source'):
            source_data = data['source']
            source = DataSource(
                provider=source_data['provider'],
                quality=DataQuality(source_data.get('quality', 'good')),
                confidence_score=source_data.get('confidence_score', 1.0)
            )
        
        return cls(
            symbol=data['symbol'],
            timestamp=timestamp,
            asset_class=AssetClass(data['asset_class']),
            ohlcv=ohlcv,
            quote=quote,
            previous_close=Decimal(str(data['previous_close'])) if data.get('previous_close') else None,
            session=MarketSession(data.get('session', 'regular')),
            is_market_open=data.get('is_market_open', True),
            exchange=data.get('exchange'),
            currency=data.get('currency', 'USD'),
            data_quality=DataQuality(data.get('data_quality', 'good')),
            validation_errors=data.get('validation_errors', []),
            source=source
        )


@dataclass
class TimeSeriesData:
    """Time series market data container."""
    symbol: str
    data_points: List[MarketDataPoint]
    start_time: datetime
    end_time: datetime
    interval: str
    asset_class: AssetClass
    
    # Metadata
    total_points: int = 0
    missing_points: int = 0
    data_quality_avg: float = 0.0
    sources: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate metadata after initialization."""
        self.total_points = len(self.data_points)
        
        if self.data_points:
            # Sort by timestamp
            self.data_points.sort(key=lambda x: x.timestamp)
            
            # Calculate quality metrics
            quality_scores = []
            sources_set = set()
            
            for point in self.data_points:
                # Map quality enum to numeric score
                quality_map = {
                    DataQuality.EXCELLENT: 5,
                    DataQuality.GOOD: 4,
                    DataQuality.FAIR: 3,
                    DataQuality.POOR: 2,
                    DataQuality.INVALID: 1
                }
                quality_scores.append(quality_map.get(point.data_quality, 3))
                
                if point.source:
                    sources_set.add(point.source.provider)
            
            self.data_quality_avg = sum(quality_scores) / len(quality_scores)
            self.sources = list(sources_set)
            
            # Calculate missing points (basic estimation)
            if len(self.data_points) > 1:
                time_diff = self.end_time - self.start_time
                expected_points = self._estimate_expected_points(time_diff, self.interval)
                self.missing_points = max(0, expected_points - self.total_points)
    
    def _estimate_expected_points(self, time_diff, interval: str) -> int:
        """Estimate expected number of data points."""
        total_seconds = time_diff.total_seconds()
        
        interval_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800
        }
        
        seconds = interval_seconds.get(interval, 86400)
        return int(total_seconds / seconds)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        data = []
        
        for point in self.data_points:
            row = {
                'timestamp': point.timestamp,
                'symbol': point.symbol
            }
            
            if point.ohlcv:
                row.update({
                    'open': float(point.ohlcv.open),
                    'high': float(point.ohlcv.high),
                    'low': float(point.ohlcv.low),
                    'close': float(point.ohlcv.close),
                    'volume': float(point.ohlcv.volume),
                    'vwap': float(point.ohlcv.vwap) if point.ohlcv.vwap else None,
                    'trade_count': point.ohlcv.trade_count
                })
            
            if point.quote:
                row.update({
                    'bid': float(point.quote.bid.price) if point.quote.bid else None,
                    'ask': float(point.quote.ask.price) if point.quote.ask else None,
                    'spread': float(point.quote.spread) if point.quote.spread else None
                })
            
            row.update({
                'change': float(point.change) if point.change else None,
                'change_percent': float(point.change_percent) if point.change_percent else None,
                'data_quality': point.data_quality.value,
                'source': point.source.provider if point.source else None
            })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate data quality report."""
        quality_counts = {}
        error_types = {}
        source_counts = {}
        
        for point in self.data_points:
            # Count quality levels
            quality = point.data_quality.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            # Count error types
            for error in point.validation_errors:
                error_types[error] = error_types.get(error, 0) + 1
            
            # Count sources
            if point.source:
                source = point.source.provider
                source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_points': self.total_points,
            'missing_points': self.missing_points,
            'data_completeness_pct': ((self.total_points - self.missing_points) / self.total_points * 100) if self.total_points > 0 else 0,
            'avg_quality_score': self.data_quality_avg,
            'quality_distribution': quality_counts,
            'error_types': error_types,
            'source_distribution': source_counts,
            'time_range': {
                'start': self.start_time.isoformat(),
                'end': self.end_time.isoformat(),
                'duration_hours': (self.end_time - self.start_time).total_seconds() / 3600
            }
        }


@dataclass
class InstrumentDefinition:
    """Financial instrument definition."""
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: str
    currency: str
    
    # Trading info
    tick_size: Optional[Decimal] = None
    lot_size: Optional[int] = None
    trading_hours: Optional[Dict[str, Any]] = None
    
    # Corporate info (for equities)
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[Decimal] = None
    
    # Option info (for options)
    underlying_symbol: Optional[str] = None
    expiration_date: Optional[datetime] = None
    strike_price: Optional[Decimal] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    
    # Metadata
    is_active: bool = True
    last_updated: Optional[datetime] = None
    data_sources: List[str] = field(default_factory=list)


# Utility functions for data manipulation
def merge_data_points(points: List[MarketDataPoint], strategy: str = "latest") -> MarketDataPoint:
    """
    Merge multiple data points for the same symbol and timestamp.
    
    Args:
        points: List of MarketDataPoint objects to merge
        strategy: Merge strategy ('latest', 'highest_quality', 'average')
    
    Returns:
        Merged MarketDataPoint
    """
    if not points:
        raise ValueError("Cannot merge empty list of data points")
    
    if len(points) == 1:
        return points[0]
    
    # Use the first point as base
    base_point = points[0]
    
    if strategy == "latest":
        # Return the most recent point
        return max(points, key=lambda p: p.timestamp)
    
    elif strategy == "highest_quality":
        # Return point with highest quality
        quality_order = {
            DataQuality.EXCELLENT: 5,
            DataQuality.GOOD: 4,
            DataQuality.FAIR: 3,
            DataQuality.POOR: 2,
            DataQuality.INVALID: 1
        }
        return max(points, key=lambda p: quality_order.get(p.data_quality, 0))
    
    elif strategy == "average":
        # Average numeric values (for OHLCV data)
        if all(p.ohlcv for p in points):
            avg_open = sum(p.ohlcv.open for p in points) / len(points)
            avg_high = max(p.ohlcv.high for p in points)
            avg_low = min(p.ohlcv.low for p in points)
            avg_close = sum(p.ohlcv.close for p in points) / len(points)
            avg_volume = sum(p.ohlcv.volume for p in points) / len(points)
            
            merged_ohlcv = OHLCV(
                open=avg_open,
                high=avg_high,
                low=avg_low,
                close=avg_close,
                volume=avg_volume,
                timestamp=base_point.timestamp
            )
            
            base_point.ohlcv = merged_ohlcv
        
        return base_point
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def validate_time_series_continuity(data: TimeSeriesData, tolerance_minutes: int = 60) -> List[str]:
    """
    Validate time series data for gaps and inconsistencies.
    
    Args:
        data: TimeSeriesData to validate
        tolerance_minutes: Allowed gap tolerance in minutes
    
    Returns:
        List of validation issues found
    """
    issues = []
    
    if len(data.data_points) < 2:
        return issues
    
    # Check for time gaps
    for i in range(1, len(data.data_points)):
        prev_point = data.data_points[i-1]
        curr_point = data.data_points[i]
        
        time_diff = (curr_point.timestamp - prev_point.timestamp).total_seconds() / 60
        
        if time_diff > tolerance_minutes:
            issues.append(f"Time gap of {time_diff:.1f} minutes between {prev_point.timestamp} and {curr_point.timestamp}")
    
    # Check for duplicate timestamps
    timestamps = [p.timestamp for p in data.data_points]
    if len(timestamps) != len(set(timestamps)):
        issues.append("Duplicate timestamps found in time series")
    
    # Check for price continuity (basic)
    for i in range(1, len(data.data_points)):
        prev_point = data.data_points[i-1]
        curr_point = data.data_points[i]
        
        if prev_point.ohlcv and curr_point.ohlcv:
            price_change = abs(curr_point.ohlcv.open - prev_point.ohlcv.close)
            if price_change > prev_point.ohlcv.close * Decimal('0.1'):  # 10% gap
                issues.append(f"Large price gap at {curr_point.timestamp}: {price_change}")
    
    return issues