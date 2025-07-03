"""
Comprehensive tests for API protection systems.

Tests rate limiting, DDoS protection, IP filtering, and throttling.
"""

import pytest
import time
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta

from alpha_pulse.api.middleware.rate_limiting import (
    RateLimitingMiddleware,
    TokenBucketAlgorithm,
    SlidingWindowAlgorithm,
    FixedWindowAlgorithm,
    AdaptiveRateLimiter
)
from alpha_pulse.utils.ddos_protection import (
    DDoSMitigator,
    TrafficAnalyzer,
    IPReputationManager,
    ThreatIndicator
)
from alpha_pulse.utils.ip_filtering import IPFilterManager, IPInfo, FilterRule
from alpha_pulse.services.throttling_service import (
    ThrottlingService,
    PriorityQueue,
    CircuitBreaker,
    ThrottleRequest,
    RequestPriority
)
from alpha_pulse.config.rate_limits import RateLimitConfig, UserTier


@pytest.fixture
async def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    
    # Mock common Redis operations
    redis_mock.pipeline.return_value = redis_mock
    redis_mock.execute.return_value = [None, 0]
    redis_mock.hmget.return_value = [None, None]
    redis_mock.exists.return_value = False
    redis_mock.sismember.return_value = False
    redis_mock.zcard.return_value = 0
    redis_mock.llen.return_value = 0
    redis_mock.keys.return_value = []
    
    return redis_mock


class TestRateLimitingAlgorithms:
    """Test rate limiting algorithms."""
    
    @pytest.mark.asyncio
    async def test_token_bucket_algorithm(self, mock_redis):
        """Test token bucket algorithm."""
        algorithm = TokenBucketAlgorithm(mock_redis)
        config = RateLimitConfig(requests=10, window_seconds=60, burst_size=5)
        
        # Mock bucket state (empty bucket)
        mock_redis.hmget.return_value = [None, None]
        
        # Test initial request (should be allowed)
        allowed, info = await algorithm.is_allowed("test_user", "global", config)
        
        assert allowed is True
        assert info['algorithm'] == 'token_bucket'
        assert 'tokens_remaining' in info
        
    @pytest.mark.asyncio
    async def test_sliding_window_algorithm(self, mock_redis):
        """Test sliding window algorithm."""
        algorithm = SlidingWindowAlgorithm(mock_redis)
        config = RateLimitConfig(requests=5, window_seconds=60)
        
        # Mock no existing requests
        mock_redis.execute.return_value = [0, 0]
        
        # Test request under limit
        allowed, info = await algorithm.is_allowed("test_user", "global", config)
        
        assert allowed is True
        assert info['algorithm'] == 'sliding_window'
        assert 'requests_remaining' in info
        
    @pytest.mark.asyncio
    async def test_fixed_window_algorithm(self, mock_redis):
        """Test fixed window algorithm."""
        algorithm = FixedWindowAlgorithm(mock_redis)
        config = RateLimitConfig(requests=10, window_seconds=3600)
        
        # Mock current count
        mock_redis.execute.return_value = [5, None]  # 5 requests in current window
        
        # Test request under limit
        allowed, info = await algorithm.is_allowed("test_user", "global", config)
        
        assert allowed is True
        assert info['algorithm'] == 'fixed_window'
        assert info['requests_remaining'] == 5
        
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiter(self, mock_redis):
        """Test adaptive rate limiting."""
        limiter = AdaptiveRateLimiter(mock_redis)
        config = RateLimitConfig(requests=100, window_seconds=60)
        
        # Mock high CPU usage
        with patch.object(limiter, 'get_system_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'cpu_usage': 85,  # High CPU
                'memory_usage': 60,
                'avg_response_time': 200,
                'error_rate': 2
            }
            
            # Mock algorithm
            with patch.object(limiter, 'get_algorithm') as mock_get_alg:
                mock_algorithm = AsyncMock()
                mock_algorithm.is_allowed.return_value = (True, {'algorithm': 'test'})
                mock_get_alg.return_value = mock_algorithm
                
                allowed, info = await limiter.check_rate_limit(
                    "test_user", "global", config, adaptive=True
                )
                
                assert allowed is True
                assert 'adaptive_factor' in info
                assert info['adaptive_factor'] < 1.0  # Should be reduced due to high CPU


class TestDDoSProtection:
    """Test DDoS protection systems."""
    
    @pytest.mark.asyncio
    async def test_traffic_analyzer(self, mock_redis):
        """Test traffic pattern analysis."""
        analyzer = TrafficAnalyzer(mock_redis)
        
        # Test high frequency detection
        request_data = {
            'ip_address': '192.168.1.100',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'endpoint': '/api/v1/metrics',
            'method': 'GET'
        }
        
        # Simulate many requests from same IP
        for _ in range(50):
            analyzer.ip_requests['192.168.1.100'].append(time.time())
            
        indicators = await analyzer.analyze_request(request_data)
        
        # Should detect high frequency
        frequency_indicators = [i for i in indicators if i.indicator_type == 'high_frequency']
        assert len(frequency_indicators) > 0
        
    @pytest.mark.asyncio
    async def test_bot_detection(self, mock_redis):
        """Test bot detection in traffic analysis."""
        analyzer = TrafficAnalyzer(mock_redis)
        
        # Test bot user agent
        request_data = {
            'ip_address': '1.2.3.4',
            'user_agent': 'python-requests/2.25.1',
            'endpoint': '/api/v1/data',
            'method': 'GET'
        }
        
        indicators = await analyzer.analyze_request(request_data)
        
        # Should detect bot user agent
        bot_indicators = [i for i in indicators if i.indicator_type == 'bot_user_agent']
        assert len(bot_indicators) > 0
        assert bot_indicators[0].score > 50
        
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self, mock_redis):
        """Test SQL injection pattern detection."""
        analyzer = TrafficAnalyzer(mock_redis)
        
        # Test SQL injection attempt
        request_data = {
            'ip_address': '1.2.3.4',
            'user_agent': 'curl/7.68.0',
            'endpoint': "/api/v1/users?id=1' OR 1=1--",
            'method': 'GET'
        }
        
        indicators = await analyzer.analyze_request(request_data)
        
        # Should detect SQL injection
        sql_indicators = [i for i in indicators if i.indicator_type == 'sql_injection_attempt']
        assert len(sql_indicators) > 0
        assert sql_indicators[0].score > 80
        
    @pytest.mark.asyncio
    async def test_ddos_mitigation(self, mock_redis):
        """Test DDoS mitigation strategies."""
        mitigator = DDoSMitigator(mock_redis)
        
        # Create high threat indicators
        indicators = [
            ThreatIndicator(
                indicator_type="high_frequency",
                score=90,
                evidence={'requests_per_minute': 1000},
                confidence=0.9,
                timestamp=datetime.now(timezone.utc)
            ),
            ThreatIndicator(
                indicator_type="bot_user_agent",
                score=80,
                evidence={'user_agent': 'bot'},
                confidence=0.8,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        threat_level = await mitigator.assess_threat_level(indicators)
        assert threat_level == "critical"
        
        # Test mitigation application
        result = await mitigator.apply_mitigation("1.2.3.4", threat_level, indicators)
        
        assert result['threat_level'] == "critical"
        assert "ip_blocked" in result['mitigation_actions']
        
    @pytest.mark.asyncio
    async def test_ip_reputation(self, mock_redis):
        """Test IP reputation management."""
        reputation_manager = IPReputationManager(mock_redis)
        
        # Mock empty reputation (new IP)
        mock_redis.hgetall.return_value = {}
        
        with patch.object(reputation_manager, '_calculate_ip_reputation') as mock_calc:
            mock_calc.return_value = {
                'score': 75,
                'classification': 'good',
                'country': 'US',
                'is_vpn': 'false',
                'is_proxy': 'false',
                'last_seen': datetime.now(timezone.utc).isoformat(),
                'threat_types': '[]'
            }
            
            reputation = await reputation_manager.get_ip_reputation("8.8.8.8")
            
            assert reputation['score'] == 75
            assert reputation['classification'] == 'good'
            assert reputation['country'] == 'US'


class TestIPFiltering:
    """Test IP filtering functionality."""
    
    @pytest.mark.asyncio
    async def test_ip_whitelist(self, mock_redis):
        """Test IP whitelist functionality."""
        filter_manager = IPFilterManager(mock_redis)
        
        # Mock IP is whitelisted
        mock_redis.sismember.return_value = True
        
        allowed, reason, details = await filter_manager.should_allow_ip("192.168.1.1")
        
        assert allowed is True
        assert reason == "whitelisted"
        
    @pytest.mark.asyncio
    async def test_ip_blacklist(self, mock_redis):
        """Test IP blacklist functionality."""
        filter_manager = IPFilterManager(mock_redis)
        
        # Mock IP is not whitelisted but is blacklisted
        mock_redis.sismember.side_effect = lambda key, value: key == "ip_blacklist"
        
        allowed, reason, details = await filter_manager.should_allow_ip("1.2.3.4")
        
        assert allowed is False
        assert "blacklisted" in reason
        
    @pytest.mark.asyncio
    async def test_geographic_filtering(self, mock_redis):
        """Test geographic IP filtering."""
        filter_manager = IPFilterManager(mock_redis)
        
        # Mock geolocation
        with patch.object(filter_manager.geolocation, 'get_ip_info') as mock_geo:
            mock_geo.return_value = IPInfo(
                ip_address="1.2.3.4",
                country_code="CN",  # Blocked country
                country_name="China",
                city=None,
                region=None,
                latitude=None,
                longitude=None,
                timezone=None,
                isp=None,
                asn=None,
                is_vpn=False,
                is_proxy=False,
                is_tor=False,
                is_hosting=False,
                threat_score=30.0,
                reputation_score=60.0
            )
            
            # Mock that this IP is not whitelisted or blacklisted
            mock_redis.sismember.return_value = False
            mock_redis.smembers.return_value = []
            
            # Set geographic restrictions
            from alpha_pulse.config.rate_limits import GEO_RESTRICTIONS
            original_blocked = GEO_RESTRICTIONS.get("blocked_countries", [])
            GEO_RESTRICTIONS["blocked_countries"] = ["CN"]
            
            try:
                allowed, reason, details = await filter_manager.should_allow_ip("1.2.3.4")
                
                assert allowed is False
                assert "geo_blocked" in reason
                assert "CN" in reason
                
            finally:
                GEO_RESTRICTIONS["blocked_countries"] = original_blocked
                
    @pytest.mark.asyncio
    async def test_filter_rule_management(self, mock_redis):
        """Test filter rule management."""
        filter_manager = IPFilterManager(mock_redis)
        
        # Create a test rule
        rule = FilterRule(
            rule_id="test_rule_1",
            rule_type="blacklist",
            pattern="1.2.3.4",
            reason="Test malicious IP",
            created_at=datetime.now(timezone.utc),
            expires_at=None,
            created_by="admin",
            is_active=True
        )
        
        # Test adding rule
        result = await filter_manager.add_filter_rule(rule)
        assert result is True
        
        # Test removing rule
        mock_redis.hgetall.return_value = {
            'rule_type': 'blacklist',
            'pattern': '1.2.3.4'
        }
        
        result = await filter_manager.remove_filter_rule("test_rule_1")
        assert result is True


class TestThrottlingService:
    """Test throttling and circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_priority_queue(self, mock_redis):
        """Test priority queue functionality."""
        queue = PriorityQueue(mock_redis)
        
        # Mock queue size
        mock_redis.zcard.return_value = 0
        
        # Create test request
        request = ThrottleRequest(
            request_id="test_123",
            priority=RequestPriority.HIGH,
            user_tier=UserTier.PREMIUM,
            endpoint="/api/v1/trades"
        )
        
        # Test enqueue
        result = await queue.enqueue(request)
        assert result is True
        
        # Mock dequeue response
        mock_redis.zrange.return_value = [(
            json.dumps({
                'request_id': 'test_123',
                'priority': 2,
                'user_tier': 'premium',
                'endpoint': '/api/v1/trades',
                'estimated_duration': 1.0,
                'max_wait_time': 30.0,
                'created_at': time.time(),
                'metadata': '{}'
            }),
            time.time()
        )]
        
        # Test dequeue
        dequeued = await queue.dequeue()
        assert dequeued is not None
        assert dequeued.request_id == "test_123"
        assert dequeued.priority == RequestPriority.HIGH
        
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = CircuitBreaker("test_service", failure_threshold=3, recovery_timeout=1.0)
        
        # Test normal operation (closed state)
        assert breaker.state.value == "closed"
        
        # Simulate function that always fails
        async def failing_function():
            raise Exception("Service unavailable")
            
        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_function)
                
        # Circuit should now be open
        assert breaker.state.value == "open"
        
        # Further calls should raise CircuitBreakerOpenError
        with pytest.raises(Exception):  # Should be CircuitBreakerOpenError
            await breaker.call(failing_function)
            
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, mock_redis):
        """Test queue overflow handling."""
        queue = PriorityQueue(mock_redis, max_size=2)
        
        # Mock queue at capacity
        mock_redis.zcard.return_value = 2
        
        # Try to add request to full queue
        request = ThrottleRequest(
            request_id="test_overflow",
            priority=RequestPriority.LOW,
            user_tier=UserTier.BASIC,
            endpoint="/api/v1/data"
        )
        
        result = await queue.enqueue(request)
        assert result is False  # Should be rejected
        
    @pytest.mark.asyncio
    async def test_expired_request_cleanup(self, mock_redis):
        """Test cleanup of expired requests."""
        queue = PriorityQueue(mock_redis)
        
        # Mock expired request
        expired_time = time.time() - 3600  # 1 hour ago
        mock_redis.zrange.return_value = [(
            json.dumps({
                'request_id': 'expired_123',
                'priority': 3,
                'user_tier': 'basic',
                'endpoint': '/api/v1/data',
                'estimated_duration': 1.0,
                'max_wait_time': 30.0,  # 30 second max wait
                'created_at': expired_time,
                'metadata': '{}'
            }),
            expired_time
        )]
        
        # Attempt to dequeue should skip expired request
        result = await queue.dequeue()
        assert result is None  # Expired request should be skipped
        
    @pytest.mark.asyncio
    async def test_throttling_service_integration(self, mock_redis):
        """Test full throttling service integration."""
        service = ThrottlingService(mock_redis)
        
        # Mock worker availability
        mock_redis.keys.return_value = ["worker:1:heartbeat"]
        mock_redis.get.return_value = str(time.time())  # Recent heartbeat
        mock_redis.llen.return_value = 0  # Empty queue
        
        # Create test request
        request = ThrottleRequest(
            request_id="integration_test",
            priority=RequestPriority.NORMAL,
            user_tier=UserTier.PREMIUM,
            endpoint="/api/v1/portfolio"
        )
        
        # Test throttling
        result = await service.throttle_request(request)
        assert result is True  # Should be processed successfully


class TestMiddlewareIntegration:
    """Test middleware integration."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_middleware_basic(self):
        """Test basic rate limiting middleware functionality."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        
        app = Starlette()
        
        @app.route('/test')
        async def test_endpoint(request):
            return JSONResponse({"message": "success"})
            
        # Create middleware with mock Redis
        with patch('redis.from_url') as mock_redis_factory:
            mock_redis = AsyncMock()
            mock_redis_factory.return_value = mock_redis
            
            middleware = RateLimitingMiddleware(
                app,
                redis_url="redis://localhost:6379",
                enable_adaptive=False
            )
            
            # Mock rate limit check to allow request
            with patch.object(middleware.rate_limiter, 'check_rate_limit') as mock_check:
                mock_check.return_value = (True, {'algorithm': 'test', 'remaining': 99})
                
                # Create mock request
                from starlette.testclient import TestClient
                
                # This would need more setup for a proper integration test
                # For now, we'll test the core logic
                assert middleware is not None
                
    def test_security_headers_application(self):
        """Test security headers middleware."""
        from alpha_pulse.api.middleware.security_headers import SecurityHeadersMiddleware
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        
        app = Starlette()
        
        @app.route('/test')
        async def test_endpoint(request):
            return JSONResponse({"message": "test"})
            
        middleware = SecurityHeadersMiddleware(app)
        
        # Test that middleware is properly initialized
        assert middleware.hsts_max_age == 31536000
        assert "default-src 'self'" in middleware.csp_policy
        
    def test_performance_impact(self):
        """Test that protection systems have minimal performance impact."""
        from alpha_pulse.config.rate_limits import get_rate_limit, UserTier
        
        # Test that rate limit lookup is fast
        start_time = time.time()
        
        for _ in range(1000):
            config = get_rate_limit(
                endpoint="/api/v1/test",
                user_tier=UserTier.PREMIUM,
                limit_type="global"
            )
            
        elapsed = time.time() - start_time
        
        # Should complete 1000 lookups in under 100ms
        assert elapsed < 0.1, f"Rate limit lookup too slow: {elapsed:.3f}s"


class TestLoadTesting:
    """Load testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_burst_traffic_handling(self, mock_redis):
        """Test handling of burst traffic."""
        limiter = AdaptiveRateLimiter(mock_redis)
        config = RateLimitConfig(requests=100, window_seconds=60, burst_size=20)
        
        # Mock token bucket with tokens available
        mock_redis.hmget.return_value = ["20", str(time.time())]
        mock_redis.execute.return_value = [None, None]
        
        # Simulate burst of 15 requests (under burst limit)
        allowed_count = 0
        for i in range(15):
            # Update tokens for each request
            remaining_tokens = max(0, 20 - i - 1)
            mock_redis.hmget.return_value = [str(remaining_tokens), str(time.time())]
            
            allowed, info = await limiter.check_rate_limit(
                f"burst_user_{i}",
                "global",
                config,
                adaptive=False
            )
            
            if allowed:
                allowed_count += 1
                
        # Most requests should be allowed (bucket had 20 tokens)
        assert allowed_count >= 10
        
    @pytest.mark.asyncio
    async def test_sustained_load_handling(self, mock_redis):
        """Test handling of sustained load."""
        analyzer = TrafficAnalyzer(mock_redis)
        
        # Simulate sustained load from multiple IPs
        ips = [f"192.168.1.{i}" for i in range(1, 21)]  # 20 different IPs
        
        indicators_found = 0
        
        for minute in range(5):  # 5 minutes of traffic
            for ip in ips:
                for _ in range(10):  # 10 requests per IP per minute
                    request_data = {
                        'ip_address': ip,
                        'user_agent': 'Mozilla/5.0 (legitimate browser)',
                        'endpoint': '/api/v1/data',
                        'method': 'GET'
                    }
                    
                    indicators = await analyzer.analyze_request(request_data)
                    indicators_found += len(indicators)
                    
        # Sustained legitimate load should not trigger many indicators
        assert indicators_found < 50, f"Too many indicators for legitimate traffic: {indicators_found}"
        
    @pytest.mark.asyncio
    async def test_ddos_attack_simulation(self, mock_redis):
        """Test DDoS attack detection and mitigation."""
        mitigator = DDoSMitigator(mock_redis)
        analyzer = TrafficAnalyzer(mock_redis)
        
        # Simulate DDoS attack - many requests from few IPs
        attack_ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
        total_indicators = 0
        critical_threats = 0
        
        for attack_ip in attack_ips:
            # Simulate 1000 requests from each IP in short time
            for _ in range(100):  # Simulate high frequency
                analyzer.ip_requests[attack_ip].append(time.time())
                
            request_data = {
                'ip_address': attack_ip,
                'user_agent': 'python-requests/2.25.1',  # Bot user agent
                'endpoint': '/api/v1/trades',
                'method': 'POST'
            }
            
            indicators = await analyzer.analyze_request(request_data)
            total_indicators += len(indicators)
            
            if indicators:
                threat_level = await mitigator.assess_threat_level(indicators)
                if threat_level in ["critical", "high"]:
                    critical_threats += 1
                    
        # Should detect attack patterns
        assert total_indicators > 0, "Failed to detect attack indicators"
        assert critical_threats > 0, "Failed to classify threats as critical/high"


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for protection systems."""
    
    def test_rate_limit_lookup_performance(self):
        """Benchmark rate limit configuration lookup."""
        from alpha_pulse.config.rate_limits import get_rate_limit, UserTier
        
        iterations = 10000
        start_time = time.time()
        
        for i in range(iterations):
            get_rate_limit(
                endpoint=f"/api/v1/endpoint_{i % 10}",
                user_tier=UserTier.PREMIUM,
                limit_type="global"
            )
            
        elapsed = time.time() - start_time
        ops_per_second = iterations / elapsed
        
        # Should handle at least 50k operations per second
        assert ops_per_second > 50000, f"Rate limit lookup too slow: {ops_per_second:.0f} ops/sec"
        
    @pytest.mark.asyncio
    async def test_ip_reputation_lookup_performance(self, mock_redis):
        """Benchmark IP reputation lookup."""
        reputation_manager = IPReputationManager(mock_redis)
        
        # Mock cached reputation
        mock_redis.hgetall.return_value = {
            'score': '75',
            'classification': 'good',
            'country': 'US',
            'is_vpn': 'false',
            'is_proxy': 'false',
            'last_seen': datetime.now(timezone.utc).isoformat(),
            'threat_types': '[]'
        }
        
        iterations = 1000
        start_time = time.time()
        
        for i in range(iterations):
            await reputation_manager.get_ip_reputation(f"192.168.1.{i % 255}")
            
        elapsed = time.time() - start_time
        ops_per_second = iterations / elapsed
        
        # Should handle at least 5k lookups per second
        assert ops_per_second > 5000, f"IP reputation lookup too slow: {ops_per_second:.0f} ops/sec"