"""
Comprehensive tests for real market data providers.

Tests:
- IEX Cloud provider integration and data format
- Polygon.io provider integration and rate limiting
- Provider factory failover mechanism
- Data normalization accuracy across sources
- Caching performance and consistency
- Data validation and quality scoring
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import aiohttp

from alpha_pulse.data_pipeline.providers.base_provider import (
    BaseRealDataProvider, ProviderHealth, ProviderStatus, RateLimitInfo
)
from alpha_pulse.data_pipeline.providers.iex_cloud import IEXCloudProvider
from alpha_pulse.data_pipeline.providers.polygon_io import PolygonIOProvider
from alpha_pulse.data_pipeline.providers.provider_factory import (
    DataProviderFactory, ProviderType, ProviderConfig, FailoverStrategy
)
from alpha_pulse.models.market_data import MarketDataPoint, TimeSeriesData, OHLCV, DataQuality
from alpha_pulse.utils.data_validation import MarketDataValidator, ValidationLevel
from alpha_pulse.services.data_aggregation import DataAggregationService, SubscriptionRequest


class TestBaseRealDataProvider:
    """Test base provider functionality."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        class MockProvider(BaseRealDataProvider):
            async def _add_auth_headers(self, headers):
                headers['Authorization'] = 'Bearer test-token'
            
            async def _test_connectivity(self):
                return True
            
            async def get_real_time_quote(self, symbol):
                pass
            
            async def get_historical_data(self, symbol, start_date, end_date, interval):
                pass
            
            async def get_company_info(self, symbol):
                pass
        
        rate_limits = RateLimitInfo(
            requests_per_second=10,
            requests_per_minute=600,
            requests_per_hour=36000,
            requests_per_day=864000
        )
        
        return MockProvider(
            provider_name="Test Provider",
            provider_type="test",
            base_url="https://api.test.com",
            api_key="test-key",
            rate_limits=rate_limits
        )

    @pytest.mark.asyncio
    async def test_provider_initialization(self, mock_provider):
        """Test provider initialization."""
        await mock_provider.initialize()
        
        health = await mock_provider.get_health()
        assert health.status == ProviderStatus.HEALTHY
        assert health.total_requests == 0
        assert health.error_count == 0
        
        await mock_provider.cleanup()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_provider):
        """Test rate limiting functionality."""
        await mock_provider.initialize()
        
        # Mock HTTP session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b'{"result": "success"}')
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        with patch.object(mock_provider, '_get_session', return_value=mock_session):
            # Make rapid requests to test rate limiting
            start_time = asyncio.get_event_loop().time()
            
            tasks = []
            for _ in range(5):
                task = mock_provider._rate_limited_request("test-endpoint")
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            elapsed_time = asyncio.get_event_loop().time() - start_time
            
            # Should take at least some time due to rate limiting
            assert elapsed_time > 0.1  # At least 100ms
        
        await mock_provider.cleanup()

    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_provider):
        """Test health monitoring functionality."""
        await mock_provider.initialize()
        
        # Simulate successful requests
        await mock_provider._record_request_metrics(0.1, True)
        await mock_provider._record_request_metrics(0.2, True)
        
        health = await mock_provider.get_health()
        assert health.success_rate == 1.0
        assert health.avg_latency > 0
        
        # Simulate failed request
        await mock_provider._record_request_metrics(0.3, False)
        
        health = await mock_provider.get_health()
        assert health.success_rate < 1.0
        assert health.error_count == 1
        
        await mock_provider.cleanup()

    def test_data_normalization(self, mock_provider):
        """Test data normalization functionality."""
        raw_data = {
            'timestamp': '2023-12-25T10:30:00',
            'open': '150.25',
            'high': '152.50',
            'low': '149.80',
            'close': '151.75',
            'volume': '1000000'
        }
        
        result = mock_provider._normalize_price_data(raw_data, 'AAPL')
        
        assert result.normalized_data is not None
        assert result.quality_score > 0.8
        assert len(result.normalization_errors) == 0
        assert result.normalized_data['symbol'] == 'AAPL'
        assert isinstance(result.normalized_data['open'], Decimal)


class TestIEXCloudProvider:
    """Test IEX Cloud provider integration."""

    @pytest.fixture
    def iex_provider(self):
        """Create IEX Cloud provider for testing."""
        return IEXCloudProvider(
            api_token="test-token",
            is_sandbox=True,
            cache_ttl=30
        )

    @pytest.mark.asyncio
    async def test_real_time_quote(self, iex_provider):
        """Test real-time quote retrieval."""
        mock_response_data = {
            'symbol': 'AAPL',
            'latestPrice': 150.25,
            'latestTime': int(datetime.utcnow().timestamp() * 1000),
            'previousClose': 149.50,
            'open': 150.00,
            'high': 151.00,
            'low': 149.75,
            'volume': 1000000
        }
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(iex_provider, '_get_session', return_value=mock_session):
            await iex_provider.initialize()
            
            quote = await iex_provider.get_real_time_quote('AAPL')
            
            assert quote.symbol == 'AAPL'
            assert quote.close == Decimal('150.25')
            assert quote.source == 'IEX Cloud'
            
            await iex_provider.cleanup()

    @pytest.mark.asyncio
    async def test_historical_data(self, iex_provider):
        """Test historical data retrieval."""
        mock_response_data = [
            {
                'date': '2023-12-25',
                'open': 150.00,
                'high': 151.00,
                'low': 149.50,
                'close': 150.75,
                'volume': 1000000
            },
            {
                'date': '2023-12-26',
                'open': 150.75,
                'high': 152.00,
                'low': 150.25,
                'close': 151.50,
                'volume': 1200000
            }
        ]
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(iex_provider, '_get_session', return_value=mock_session):
            await iex_provider.initialize()
            
            start_date = datetime(2023, 12, 25)
            end_date = datetime(2023, 12, 26)
            
            historical_data = await iex_provider.get_historical_data(
                'AAPL', start_date, end_date, '1d'
            )
            
            assert len(historical_data) == 2
            assert all(isinstance(point, type(historical_data[0])) for point in historical_data)
            assert historical_data[0].symbol == 'AAPL'
            assert historical_data[0].close == Decimal('150.75')
            
            await iex_provider.cleanup()

    @pytest.mark.asyncio
    async def test_batch_quotes(self, iex_provider):
        """Test batch quote functionality."""
        mock_response_data = {
            'AAPL': {
                'quote': {
                    'symbol': 'AAPL',
                    'latestPrice': 150.25,
                    'latestTime': int(datetime.utcnow().timestamp() * 1000),
                    'volume': 1000000
                }
            },
            'GOOGL': {
                'quote': {
                    'symbol': 'GOOGL',
                    'latestPrice': 2800.50,
                    'latestTime': int(datetime.utcnow().timestamp() * 1000),
                    'volume': 500000
                }
            }
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(iex_provider, '_get_session', return_value=mock_session):
            await iex_provider.initialize()
            
            quotes = await iex_provider.batch_quotes(['AAPL', 'GOOGL'])
            
            assert len(quotes) == 2
            assert 'AAPL' in quotes
            assert 'GOOGL' in quotes
            assert quotes['AAPL'].close == Decimal('150.25')
            assert quotes['GOOGL'].close == Decimal('2800.50')
            
            await iex_provider.cleanup()

    def test_cost_estimation(self, iex_provider):
        """Test cost estimation functionality."""
        cost_estimate = iex_provider.get_cost_estimate('quote', 5)
        
        assert 'request_type' in cost_estimate
        assert 'symbols_count' in cost_estimate
        assert 'total_credits' in cost_estimate
        assert 'estimated_cost_usd' in cost_estimate
        assert cost_estimate['symbols_count'] == 5


class TestPolygonIOProvider:
    """Test Polygon.io provider integration."""

    @pytest.fixture
    def polygon_provider(self):
        """Create Polygon.io provider for testing."""
        return PolygonIOProvider(
            api_key="test-key",
            tier="free",
            cache_ttl=60
        )

    @pytest.mark.asyncio
    async def test_real_time_quote(self, polygon_provider):
        """Test real-time quote retrieval."""
        mock_response_data = {
            'status': 'OK',
            'results': [
                {
                    't': int(datetime.utcnow().timestamp() * 1000),
                    'o': 150.00,
                    'h': 151.00,
                    'l': 149.50,
                    'c': 150.75,
                    'v': 1000000,
                    'vw': 150.40,
                    'n': 5000
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(polygon_provider, '_get_session', return_value=mock_session):
            await polygon_provider.initialize()
            
            quote = await polygon_provider.get_real_time_quote('AAPL')
            
            assert quote.symbol == 'AAPL'
            assert quote.close == Decimal('150.75')
            assert quote.vwap == Decimal('150.40')
            assert quote.trades == 5000
            assert quote.source == 'Polygon.io'
            
            await polygon_provider.cleanup()

    @pytest.mark.asyncio
    async def test_options_chain(self, polygon_provider):
        """Test options chain retrieval."""
        mock_response_data = {
            'status': 'OK',
            'results': [
                {
                    'ticker': 'O:AAPL231215C00150000',
                    'underlying_ticker': 'AAPL',
                    'contract_type': 'call',
                    'strike_price': 150.0,
                    'expiration_date': '2023-12-15'
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(polygon_provider, '_get_session', return_value=mock_session):
            await polygon_provider.initialize()
            
            options = await polygon_provider.get_options_chain('AAPL')
            
            assert len(options) == 1
            assert options[0]['underlying_ticker'] == 'AAPL'
            assert options[0]['contract_type'] == 'call'
            assert options[0]['source'] == 'Polygon.io'
            
            await polygon_provider.cleanup()

    def test_cost_estimation(self, polygon_provider):
        """Test cost estimation for subscription tiers."""
        cost_estimate = polygon_provider.get_cost_estimate('quote', 10)
        
        assert 'tier' in cost_estimate
        assert 'monthly_cost_usd' in cost_estimate
        assert 'daily_limit' in cost_estimate
        assert cost_estimate['symbols_count'] == 10
        assert cost_estimate['tier'] == 'free'


class TestProviderFactory:
    """Test provider factory functionality."""

    @pytest.fixture
    async def provider_factory(self):
        """Create provider factory for testing."""
        factory = DataProviderFactory()
        
        # Mock provider configurations for testing
        mock_configs = {
            ProviderType.IEX_CLOUD: ProviderConfig(
                provider_type=ProviderType.IEX_CLOUD,
                priority=1,
                enabled=True,
                config={"is_sandbox": True, "cache_ttl": 30}
            ),
            ProviderType.POLYGON_IO: ProviderConfig(
                provider_type=ProviderType.POLYGON_IO,
                priority=2,
                enabled=True,
                config={"tier": "free", "cache_ttl": 60}
            )
        }
        
        # Mock secret retrieval
        with patch('alpha_pulse.data_pipeline.providers.provider_factory.get_secret') as mock_secret:
            mock_secret.return_value = "test-api-key"
            await factory.initialize(mock_configs)
        
        yield factory
        await factory.cleanup()

    @pytest.mark.asyncio
    async def test_provider_failover(self, provider_factory):
        """Test failover between providers."""
        # Mock successful response from secondary provider
        mock_market_data = Mock()
        mock_market_data.symbol = 'AAPL'
        mock_market_data.close = Decimal('150.25')
        mock_market_data.timestamp = datetime.utcnow()
        mock_market_data.source = 'Polygon.io'
        
        # Mock primary provider failure
        with patch.object(provider_factory._providers[ProviderType.IEX_CLOUD], 'get_real_time_quote') as mock_primary:
            mock_primary.side_effect = Exception("Primary provider failed")
            
            with patch.object(provider_factory._providers[ProviderType.POLYGON_IO], 'get_real_time_quote') as mock_secondary:
                mock_secondary.return_value = mock_market_data
                
                result = await provider_factory.get_real_time_quote('AAPL')
                
                assert result == mock_market_data
                assert provider_factory._consecutive_failures[ProviderType.IEX_CLOUD] == 1
                assert provider_factory._last_successful_provider == ProviderType.POLYGON_IO

    @pytest.mark.asyncio
    async def test_health_based_failover(self, provider_factory):
        """Test health-based provider selection."""
        # Set failover strategy to health-based
        provider_factory._failover_strategy = FailoverStrategy.HEALTH_BASED
        
        # Mock provider performance metrics
        provider_factory._provider_performance[ProviderType.IEX_CLOUD] = Mock()
        provider_factory._provider_performance[ProviderType.IEX_CLOUD].success_rate = 0.5
        provider_factory._provider_performance[ProviderType.IEX_CLOUD].avg_latency = 0.5
        
        provider_factory._provider_performance[ProviderType.POLYGON_IO] = Mock()
        provider_factory._provider_performance[ProviderType.POLYGON_IO].success_rate = 0.9
        provider_factory._provider_performance[ProviderType.POLYGON_IO].avg_latency = 0.2
        
        # Get provider order (should prefer Polygon.io due to better health)
        provider_order = provider_factory._get_provider_priority_order()
        
        assert provider_order[0] == ProviderType.POLYGON_IO
        assert provider_order[1] == ProviderType.IEX_CLOUD

    def test_provider_status(self, provider_factory):
        """Test provider status reporting."""
        status = provider_factory.get_provider_status()
        
        assert 'total_providers' in status
        assert 'enabled_providers' in status
        assert 'failover_strategy' in status
        assert 'providers' in status
        
        assert status['total_providers'] == 2
        assert status['enabled_providers'] == 2
        assert status['failover_strategy'] == 'health_based'


class TestDataValidation:
    """Test data validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create data validator for testing."""
        return MarketDataValidator(ValidationLevel.STANDARD)

    @pytest.mark.asyncio
    async def test_data_point_validation(self, validator):
        """Test individual data point validation."""
        # Create valid data point
        ohlcv = OHLCV(
            open=Decimal('150.00'),
            high=Decimal('151.00'),
            low=Decimal('149.50'),
            close=Decimal('150.75'),
            volume=Decimal('1000000'),
            timestamp=datetime.utcnow()
        )
        
        data_point = MarketDataPoint(
            symbol='AAPL',
            timestamp=datetime.utcnow(),
            asset_class='equity',
            ohlcv=ohlcv
        )
        
        result = await validator.validate_data_point(data_point)
        
        assert result.is_valid
        assert result.quality_score > 0.8
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_invalid_data_detection(self, validator):
        """Test detection of invalid data."""
        # Create invalid data point (high < low)
        ohlcv = OHLCV(
            open=Decimal('150.00'),
            high=Decimal('149.00'),  # Invalid: high < low
            low=Decimal('149.50'),
            close=Decimal('150.75'),
            volume=Decimal('1000000'),
            timestamp=datetime.utcnow()
        )
        
        data_point = MarketDataPoint(
            symbol='AAPL',
            timestamp=datetime.utcnow(),
            asset_class='equity',
            ohlcv=ohlcv
        )
        
        result = await validator.validate_data_point(data_point)
        
        assert not result.is_valid
        assert result.quality_score < 0.5
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_time_series_validation(self, validator):
        """Test time series validation."""
        data_points = []
        base_time = datetime.utcnow()
        
        for i in range(5):
            ohlcv = OHLCV(
                open=Decimal(f'{150 + i}.00'),
                high=Decimal(f'{151 + i}.00'),
                low=Decimal(f'{149 + i}.50'),
                close=Decimal(f'{150 + i}.75'),
                volume=Decimal('1000000'),
                timestamp=base_time + timedelta(minutes=i)
            )
            
            data_point = MarketDataPoint(
                symbol='AAPL',
                timestamp=base_time + timedelta(minutes=i),
                asset_class='equity',
                ohlcv=ohlcv
            )
            data_points.append(data_point)
        
        time_series = TimeSeriesData(
            symbol='AAPL',
            data_points=data_points,
            start_time=base_time,
            end_time=base_time + timedelta(minutes=4),
            interval='1m',
            asset_class='equity'
        )
        
        result = await validator.validate_time_series(time_series)
        
        assert result.is_valid
        assert result.quality_score > 0.8


class TestDataAggregationService:
    """Test data aggregation service."""

    @pytest.fixture
    async def aggregation_service(self):
        """Create data aggregation service for testing."""
        service = DataAggregationService()
        
        # Mock provider factory
        mock_factory = AsyncMock()
        mock_factory.get_real_time_quote = AsyncMock()
        mock_factory.get_historical_data = AsyncMock()
        mock_factory.batch_quotes = AsyncMock()
        
        service._provider_factory = mock_factory
        
        # Mock Redis
        service._redis = AsyncMock()
        service._redis.get = AsyncMock(return_value=None)
        service._redis.setex = AsyncMock()
        
        yield service
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_real_time_data_caching(self, aggregation_service):
        """Test real-time data caching functionality."""
        # Mock market data
        mock_market_data = Mock()
        mock_market_data.symbol = 'AAPL'
        mock_market_data.close = Decimal('150.25')
        mock_market_data.timestamp = datetime.utcnow()
        mock_market_data.source = 'IEX Cloud'
        
        aggregation_service._provider_factory.get_real_time_quote.return_value = mock_market_data
        
        # First call should hit provider
        result1 = await aggregation_service.get_real_time_data('AAPL', use_cache=True)
        assert aggregation_service._request_stats['provider_requests'] == 1
        
        # Mock cache hit for second call
        cached_data = result1.to_dict()
        aggregation_service._memory_cache['realtime:AAPL'] = cached_data
        aggregation_service._cache_timestamps['realtime:AAPL'] = datetime.utcnow()
        
        result2 = await aggregation_service.get_real_time_data('AAPL', use_cache=True)
        
        # Should have used cache, no additional provider requests
        assert aggregation_service._request_stats['provider_requests'] == 1
        assert aggregation_service._request_stats['cache_hits'] == 1

    @pytest.mark.asyncio
    async def test_subscription_management(self, aggregation_service):
        """Test real-time data subscription management."""
        callback_data = []
        
        async def test_callback(data_point):
            callback_data.append(data_point)
        
        request = SubscriptionRequest(
            symbol='AAPL',
            data_types=['quotes'],
            frequency='real-time',
            callback=test_callback
        )
        
        subscription_id = await aggregation_service.subscribe_to_real_time_data(
            request, 'test-subscriber'
        )
        
        assert subscription_id == 'AAPL:test-subscriber'
        assert 'AAPL' in aggregation_service._subscriptions
        assert 'test-subscriber' in aggregation_service._subscriptions['AAPL']
        
        # Test unsubscription
        unsubscribed = await aggregation_service.unsubscribe(subscription_id)
        assert unsubscribed
        assert 'AAPL' not in aggregation_service._subscriptions

    def test_service_status(self, aggregation_service):
        """Test service status reporting."""
        status = aggregation_service.get_service_status()
        
        assert 'is_running' in status
        assert 'request_stats' in status
        assert 'cache_stats' in status
        assert 'subscription_stats' in status
        
        # Check cache hit rate calculation
        cache_stats = status['cache_stats']
        assert 'cache_hit_rate' in cache_stats
        assert isinstance(cache_stats['cache_hit_rate'], (int, float))


@pytest.mark.integration
class TestIntegrationDataFlow:
    """Integration tests for complete data flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self):
        """Test complete data flow from provider to validation."""
        # This would be a comprehensive integration test
        # that tests the entire pipeline with real or mocked data
        pass

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under concurrent load."""
        # Performance test with multiple concurrent requests
        pass

    @pytest.mark.asyncio
    async def test_failover_scenarios(self):
        """Test various failover scenarios."""
        # Test different failure modes and recovery
        pass


# Performance and stress tests
@pytest.mark.performance
class TestPerformance:
    """Performance tests for data providers."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling."""
        # Test handling of multiple simultaneous requests
        pass

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance under load."""
        # Test cache hit/miss ratios and response times
        pass

    @pytest.mark.asyncio
    async def test_rate_limit_compliance(self):
        """Test rate limit compliance."""
        # Ensure rate limits are properly enforced
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=strict"])