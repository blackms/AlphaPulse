"""
Rate limiting configuration for AlphaPulse API.

Defines rate limits per endpoint, user tier, and API key type.
Supports adaptive rate limiting based on system load.
"""

from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


class UserTier(Enum):
    """User subscription tiers."""
    ANONYMOUS = "anonymous"
    BASIC = "basic"
    PREMIUM = "premium"
    INSTITUTIONAL = "institutional"
    SYSTEM = "system"


class APIKeyType(Enum):
    """API key types with different permissions."""
    READ_ONLY = "read_only"
    TRADING_ENABLED = "trading_enabled"
    SYSTEM_INTEGRATION = "system_integration"
    INTERNAL = "internal"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests: int
    window_seconds: int
    burst_size: Optional[int] = None  # For token bucket algorithm
    
    def __str__(self):
        """Human-readable representation."""
        if self.window_seconds < 60:
            return f"{self.requests} requests/{self.window_seconds}s"
        elif self.window_seconds < 3600:
            return f"{self.requests} requests/{self.window_seconds//60}m"
        else:
            return f"{self.requests} requests/{self.window_seconds//3600}h"


# Per-IP rate limits for public endpoints
IP_RATE_LIMITS = {
    # Public endpoints - strict limits
    "public": RateLimitConfig(100, 60, burst_size=20),  # 100/min with 20 burst
    "auth": RateLimitConfig(10, 60, burst_size=5),      # 10/min for login attempts
    
    # Authenticated endpoints - more generous
    "authenticated": RateLimitConfig(500, 60, burst_size=100),  # 500/min
    "trading": RateLimitConfig(1000, 60, burst_size=200),       # 1000/min
    "data": RateLimitConfig(2000, 60, burst_size=500),          # 2000/min
    
    # WebSocket connections
    "websocket": RateLimitConfig(10, 3600),  # 10 connections/hour
}

# Per-user rate limits based on tier
USER_TIER_LIMITS = {
    UserTier.ANONYMOUS: {
        "global": RateLimitConfig(100, 3600),     # 100/hour
        "search": RateLimitConfig(20, 60),        # 20/min
        "data": RateLimitConfig(50, 3600),        # 50/hour
    },
    UserTier.BASIC: {
        "global": RateLimitConfig(1000, 3600),    # 1000/hour
        "trading": RateLimitConfig(100, 3600),    # 100 trades/hour
        "data": RateLimitConfig(5000, 3600),      # 5000 data requests/hour
        "alerts": RateLimitConfig(50, 3600),      # 50 alerts/hour
    },
    UserTier.PREMIUM: {
        "global": RateLimitConfig(10000, 3600),   # 10k/hour
        "trading": RateLimitConfig(1000, 3600),   # 1000 trades/hour
        "data": RateLimitConfig(50000, 3600),     # 50k data requests/hour
        "alerts": RateLimitConfig(500, 3600),     # 500 alerts/hour
    },
    UserTier.INSTITUTIONAL: {
        "global": RateLimitConfig(100000, 3600),  # 100k/hour
        "trading": RateLimitConfig(10000, 3600),  # 10k trades/hour
        "data": RateLimitConfig(500000, 3600),    # 500k data requests/hour
        "alerts": RateLimitConfig(5000, 3600),    # 5k alerts/hour
    },
    UserTier.SYSTEM: {
        # No limits for internal system users
        "global": RateLimitConfig(999999999, 1),
    }
}

# API key-based rate limits
API_KEY_LIMITS = {
    APIKeyType.READ_ONLY: {
        "global": RateLimitConfig(5000, 3600),     # 5k/hour
        "data": RateLimitConfig(5000, 3600),       # 5k data requests/hour
    },
    APIKeyType.TRADING_ENABLED: {
        "global": RateLimitConfig(50000, 3600),    # 50k/hour
        "trading": RateLimitConfig(5000, 3600),    # 5k trades/hour
        "data": RateLimitConfig(50000, 3600),      # 50k data requests/hour
    },
    APIKeyType.SYSTEM_INTEGRATION: {
        "global": RateLimitConfig(500000, 3600),   # 500k/hour
        "trading": RateLimitConfig(50000, 3600),   # 50k trades/hour
        "data": RateLimitConfig(500000, 3600),     # 500k data requests/hour
    },
    APIKeyType.INTERNAL: {
        # No limits for internal API keys
        "global": RateLimitConfig(999999999, 1),
    }
}

# Endpoint-specific rate limits (overrides global limits)
ENDPOINT_RATE_LIMITS = {
    # Authentication endpoints - very strict
    "/token": RateLimitConfig(5, 60, burst_size=2),           # 5/min login attempts
    "/api/v1/auth/register": RateLimitConfig(2, 3600),        # 2 registrations/hour
    "/api/v1/auth/password-reset": RateLimitConfig(3, 3600),  # 3/hour
    
    # Trading endpoints - moderate limits
    "/api/v1/trades/execute": RateLimitConfig(60, 60),        # 60 trades/min
    "/api/v1/trades/cancel": RateLimitConfig(120, 60),        # 120 cancels/min
    
    # Data endpoints - generous limits
    "/api/v1/metrics/": RateLimitConfig(600, 60),             # 600/min
    "/api/v1/portfolio/": RateLimitConfig(300, 60),           # 300/min
    
    # Expensive operations - strict limits
    "/api/v1/backtest/": RateLimitConfig(10, 3600),           # 10/hour
    "/api/v1/reports/generate": RateLimitConfig(20, 3600),    # 20/hour
    "/api/v1/audit/export": RateLimitConfig(5, 3600),         # 5/hour
}

# Adaptive rate limiting thresholds
ADAPTIVE_THRESHOLDS = {
    "cpu_high": 80,        # Reduce limits when CPU > 80%
    "memory_high": 85,     # Reduce limits when memory > 85%
    "response_slow": 500,  # Reduce limits when avg response > 500ms
    "error_rate": 5,       # Reduce limits when error rate > 5%
}

# Reduction factors for adaptive limiting
ADAPTIVE_REDUCTION_FACTORS = {
    "light": 0.9,    # 10% reduction
    "moderate": 0.7, # 30% reduction
    "heavy": 0.5,    # 50% reduction
    "critical": 0.2, # 80% reduction
}

# DDoS protection thresholds
DDOS_THRESHOLDS = {
    "requests_per_second": 1000,      # Max 1000 req/s from single IP
    "connections_per_second": 100,    # Max 100 new connections/s
    "unique_ips_per_minute": 10000,   # Max 10k unique IPs/min
    "error_rate_threshold": 50,       # Block if error rate > 50%
}

# IP reputation scoring weights
IP_REPUTATION_WEIGHTS = {
    "successful_requests": 1,
    "rate_limit_hits": -5,
    "auth_failures": -10,
    "malicious_patterns": -50,
    "whitelisted": 100,
}

# Geographic restrictions (if enabled)
GEO_RESTRICTIONS = {
    "allowed_countries": [],  # Empty = all allowed
    "blocked_countries": [],  # List of ISO country codes
    "require_verification": ["CN", "RU", "KP"],  # Countries requiring extra verification
}

# Circuit breaker configuration
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 50,      # Open circuit after 50 failures
    "recovery_timeout": 60,       # Try recovery after 60 seconds
    "expected_response_time": 1,  # Expected response time in seconds
}

# Priority queue configuration
PRIORITY_LEVELS = {
    UserTier.INSTITUTIONAL: 1,  # Highest priority
    UserTier.PREMIUM: 2,
    UserTier.BASIC: 3,
    UserTier.ANONYMOUS: 4,      # Lowest priority
}


def get_rate_limit(
    endpoint: str,
    user_tier: Optional[UserTier] = None,
    api_key_type: Optional[APIKeyType] = None,
    limit_type: str = "global"
) -> RateLimitConfig:
    """
    Get the appropriate rate limit for a request.
    
    Args:
        endpoint: API endpoint path
        user_tier: User's subscription tier
        api_key_type: Type of API key used
        limit_type: Type of limit (global, trading, data, etc.)
        
    Returns:
        RateLimitConfig with the most restrictive applicable limit
    """
    limits = []
    
    # Check endpoint-specific limits
    for pattern, config in ENDPOINT_RATE_LIMITS.items():
        if endpoint.startswith(pattern):
            limits.append(config)
            
    # Check user tier limits
    if user_tier and user_tier in USER_TIER_LIMITS:
        tier_limits = USER_TIER_LIMITS[user_tier]
        if limit_type in tier_limits:
            limits.append(tier_limits[limit_type])
            
    # Check API key limits
    if api_key_type and api_key_type in API_KEY_LIMITS:
        key_limits = API_KEY_LIMITS[api_key_type]
        if limit_type in key_limits:
            limits.append(key_limits[limit_type])
            
    # Return most restrictive limit
    if limits:
        return min(limits, key=lambda x: x.requests / x.window_seconds)
    
    # Default limit
    return RateLimitConfig(100, 60)


def get_adaptive_factor(system_metrics: Dict[str, float]) -> float:
    """
    Calculate adaptive rate limiting factor based on system metrics.
    
    Args:
        system_metrics: Current system metrics (cpu, memory, etc.)
        
    Returns:
        Multiplication factor for rate limits (0.2 to 1.0)
    """
    # Check various system health indicators
    if system_metrics.get("cpu_usage", 0) > ADAPTIVE_THRESHOLDS["cpu_high"]:
        return ADAPTIVE_REDUCTION_FACTORS["moderate"]
    
    if system_metrics.get("memory_usage", 0) > ADAPTIVE_THRESHOLDS["memory_high"]:
        return ADAPTIVE_REDUCTION_FACTORS["moderate"]
        
    if system_metrics.get("avg_response_time", 0) > ADAPTIVE_THRESHOLDS["response_slow"]:
        return ADAPTIVE_REDUCTION_FACTORS["light"]
        
    if system_metrics.get("error_rate", 0) > ADAPTIVE_THRESHOLDS["error_rate"]:
        return ADAPTIVE_REDUCTION_FACTORS["heavy"]
        
    # Check for critical conditions
    if (system_metrics.get("cpu_usage", 0) > 95 or 
        system_metrics.get("memory_usage", 0) > 95):
        return ADAPTIVE_REDUCTION_FACTORS["critical"]
        
    return 1.0  # No reduction


def should_enforce_geo_restriction(country_code: str) -> Tuple[bool, str]:
    """
    Check if geographic restrictions should be enforced.
    
    Args:
        country_code: ISO country code
        
    Returns:
        Tuple of (should_block, reason)
    """
    if country_code in GEO_RESTRICTIONS["blocked_countries"]:
        return True, f"Country {country_code} is blocked"
        
    if (GEO_RESTRICTIONS["allowed_countries"] and 
        country_code not in GEO_RESTRICTIONS["allowed_countries"]):
        return True, f"Country {country_code} is not in allowed list"
        
    return False, ""