"""
Configuration for market data sources and providers.

Defines:
- Data source configurations and credentials
- Provider priorities and failover settings
- Data quality and validation rules
- Cost management and rate limiting settings
"""

from datetime import timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass

from alpha_pulse.data_pipeline.providers.provider_factory import ProviderType, ProviderConfig, FailoverStrategy


class DataSourceTier(Enum):
    """Data source tier levels."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class DataQualityLevel(Enum):
    """Data quality requirement levels."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataSourceLimits:
    """Data source rate limits and quotas."""
    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    monthly_quota: Optional[int] = None
    cost_per_request: float = 0.0
    overage_cost: float = 0.0


@dataclass
class DataSourceFeatures:
    """Features supported by a data source."""
    real_time_quotes: bool = True
    historical_data: bool = True
    intraday_data: bool = True
    company_fundamentals: bool = False
    options_data: bool = False
    crypto_data: bool = False
    forex_data: bool = False
    technical_indicators: bool = False
    news_data: bool = False
    earnings_data: bool = False
    batch_requests: bool = False
    websocket_feeds: bool = False


class DataSourceConfig:
    """Configuration manager for data sources."""

    def __init__(self):
        """Initialize data source configurations."""
        self._provider_configs = self._setup_provider_configs()
        self._data_quality_rules = self._setup_quality_rules()
        self._cost_limits = self._setup_cost_limits()

    def _setup_provider_configs(self) -> Dict[ProviderType, Dict[str, Any]]:
        """Setup provider-specific configurations."""
        return {
            ProviderType.IEX_CLOUD: {
                "name": "IEX Cloud",
                "description": "Real-time and historical stock market data",
                "website": "https://iexcloud.io",
                "documentation": "https://iexcloud.io/docs/api/",
                "tiers": {
                    DataSourceTier.FREE: {
                        "limits": DataSourceLimits(
                            requests_per_second=100,
                            requests_per_minute=6000,
                            requests_per_hour=360000,
                            requests_per_day=8640000,
                            monthly_quota=500000,
                            cost_per_request=0.0001
                        ),
                        "features": DataSourceFeatures(
                            real_time_quotes=True,
                            historical_data=True,
                            intraday_data=True,
                            company_fundamentals=True,
                            batch_requests=True,
                            earnings_data=True
                        )
                    },
                    DataSourceTier.BASIC: {
                        "limits": DataSourceLimits(
                            requests_per_second=100,
                            requests_per_minute=6000,
                            requests_per_hour=360000,
                            requests_per_day=8640000,
                            monthly_quota=5000000,
                            cost_per_request=0.00005
                        ),
                        "features": DataSourceFeatures(
                            real_time_quotes=True,
                            historical_data=True,
                            intraday_data=True,
                            company_fundamentals=True,
                            batch_requests=True,
                            earnings_data=True,
                            news_data=True
                        )
                    }
                },
                "data_coverage": {
                    "exchanges": ["NYSE", "NASDAQ", "AMEX"],
                    "asset_classes": ["stocks", "etfs"],
                    "regions": ["US"],
                    "historical_depth": "15+ years",
                    "real_time_delay": "0ms",
                    "update_frequency": "real-time"
                },
                "data_quality": {
                    "accuracy": "exchange-grade",
                    "completeness": "99.9%",
                    "latency": "<100ms",
                    "corporate_actions": True,
                    "split_adjusted": True,
                    "dividend_adjusted": True
                }
            },
            
            ProviderType.POLYGON_IO: {
                "name": "Polygon.io",
                "description": "Real-time and historical financial market data",
                "website": "https://polygon.io",
                "documentation": "https://polygon.io/docs/",
                "tiers": {
                    DataSourceTier.FREE: {
                        "limits": DataSourceLimits(
                            requests_per_second=5,
                            requests_per_minute=300,
                            requests_per_hour=18000,
                            requests_per_day=432000,
                            cost_per_request=0.0
                        ),
                        "features": DataSourceFeatures(
                            real_time_quotes=True,
                            historical_data=True,
                            intraday_data=True,
                            company_fundamentals=True,
                            options_data=True,
                            crypto_data=True,
                            forex_data=True
                        )
                    },
                    DataSourceTier.BASIC: {
                        "limits": DataSourceLimits(
                            requests_per_second=10,
                            requests_per_minute=600,
                            requests_per_hour=36000,
                            requests_per_day=864000,
                            cost_per_request=0.0
                        ),
                        "features": DataSourceFeatures(
                            real_time_quotes=True,
                            historical_data=True,
                            intraday_data=True,
                            company_fundamentals=True,
                            options_data=True,
                            crypto_data=True,
                            forex_data=True,
                            technical_indicators=True
                        )
                    },
                    DataSourceTier.PROFESSIONAL: {
                        "limits": DataSourceLimits(
                            requests_per_second=50,
                            requests_per_minute=3000,
                            requests_per_hour=180000,
                            requests_per_day=4320000,
                            cost_per_request=0.0
                        ),
                        "features": DataSourceFeatures(
                            real_time_quotes=True,
                            historical_data=True,
                            intraday_data=True,
                            company_fundamentals=True,
                            options_data=True,
                            crypto_data=True,
                            forex_data=True,
                            technical_indicators=True,
                            websocket_feeds=True
                        )
                    }
                },
                "data_coverage": {
                    "exchanges": ["NYSE", "NASDAQ", "AMEX", "OTC", "Crypto Exchanges"],
                    "asset_classes": ["stocks", "options", "forex", "crypto"],
                    "regions": ["US", "Global"],
                    "historical_depth": "20+ years",
                    "real_time_delay": "0ms",
                    "update_frequency": "real-time"
                },
                "data_quality": {
                    "accuracy": "institutional-grade",
                    "completeness": "99.95%",
                    "latency": "<50ms",
                    "corporate_actions": True,
                    "split_adjusted": True,
                    "dividend_adjusted": True
                }
            }
        }

    def _setup_quality_rules(self) -> Dict[DataQualityLevel, Dict[str, Any]]:
        """Setup data quality validation rules."""
        return {
            DataQualityLevel.BASIC: {
                "max_latency_ms": 5000,
                "min_completeness_pct": 95.0,
                "max_price_deviation_pct": 10.0,
                "required_fields": ["symbol", "timestamp", "close", "volume"],
                "validation_rules": {
                    "price_sanity_check": True,
                    "volume_sanity_check": True,
                    "timestamp_validation": True
                }
            },
            DataQualityLevel.STANDARD: {
                "max_latency_ms": 1000,
                "min_completeness_pct": 99.0,
                "max_price_deviation_pct": 5.0,
                "required_fields": ["symbol", "timestamp", "open", "high", "low", "close", "volume"],
                "validation_rules": {
                    "price_sanity_check": True,
                    "volume_sanity_check": True,
                    "timestamp_validation": True,
                    "ohlc_consistency_check": True,
                    "historical_consistency_check": True
                }
            },
            DataQualityLevel.HIGH: {
                "max_latency_ms": 500,
                "min_completeness_pct": 99.9,
                "max_price_deviation_pct": 2.0,
                "required_fields": ["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"],
                "validation_rules": {
                    "price_sanity_check": True,
                    "volume_sanity_check": True,
                    "timestamp_validation": True,
                    "ohlc_consistency_check": True,
                    "historical_consistency_check": True,
                    "cross_provider_validation": True,
                    "corporate_action_adjustment": True
                }
            },
            DataQualityLevel.CRITICAL: {
                "max_latency_ms": 100,
                "min_completeness_pct": 99.99,
                "max_price_deviation_pct": 1.0,
                "required_fields": ["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap", "trades"],
                "validation_rules": {
                    "price_sanity_check": True,
                    "volume_sanity_check": True,
                    "timestamp_validation": True,
                    "ohlc_consistency_check": True,
                    "historical_consistency_check": True,
                    "cross_provider_validation": True,
                    "corporate_action_adjustment": True,
                    "real_time_validation": True,
                    "statistical_outlier_detection": True
                }
            }
        }

    def _setup_cost_limits(self) -> Dict[str, Any]:
        """Setup cost management limits."""
        return {
            "daily_budget_usd": 100.0,
            "monthly_budget_usd": 2000.0,
            "cost_alerts": {
                "daily_threshold_pct": 80.0,
                "monthly_threshold_pct": 90.0,
                "burst_cost_limit": 50.0  # Max cost in 1 hour
            },
            "cost_optimization": {
                "cache_aggressive": True,
                "batch_requests": True,
                "off_hours_reduced_polling": True,
                "weekend_minimal_polling": True
            }
        }

    def get_provider_config(
        self, 
        provider_type: ProviderType, 
        tier: DataSourceTier = DataSourceTier.FREE
    ) -> Dict[str, Any]:
        """Get configuration for a specific provider and tier."""
        provider_config = self._provider_configs.get(provider_type)
        if not provider_config:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        tier_config = provider_config["tiers"].get(tier)
        if not tier_config:
            raise ValueError(f"Unknown tier {tier} for provider {provider_type}")
        
        return {
            **provider_config,
            "current_tier": tier,
            **tier_config
        }

    def get_quality_rules(self, level: DataQualityLevel) -> Dict[str, Any]:
        """Get data quality rules for a specific level."""
        return self._quality_rules.get(level, self._quality_rules[DataQualityLevel.STANDARD])

    def get_cost_limits(self) -> Dict[str, Any]:
        """Get cost management limits."""
        return self._cost_limits

    def get_recommended_providers(
        self, 
        requirements: Dict[str, Any],
        budget_usd: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommended providers based on requirements.

        Args:
            requirements: Dictionary with feature requirements
            budget_usd: Monthly budget in USD

        Returns:
            List of recommended provider configurations
        """
        recommendations = []
        
        for provider_type, config in self._provider_configs.items():
            for tier, tier_config in config["tiers"].items():
                features = tier_config["features"]
                limits = tier_config["limits"]
                
                # Check feature requirements
                feature_score = 0
                total_features = 0
                
                for feature, required in requirements.get("features", {}).items():
                    if hasattr(features, feature):
                        total_features += 1
                        if getattr(features, feature) == required:
                            feature_score += 1
                
                feature_match_pct = (feature_score / total_features) * 100 if total_features > 0 else 0
                
                # Check budget constraints
                monthly_cost = 0
                if tier == DataSourceTier.FREE:
                    monthly_cost = 0
                elif provider_type == ProviderType.IEX_CLOUD:
                    monthly_cost = limits.monthly_quota * limits.cost_per_request if limits.monthly_quota else 0
                elif provider_type == ProviderType.POLYGON_IO:
                    tier_costs = {
                        DataSourceTier.BASIC: 99,
                        DataSourceTier.PROFESSIONAL: 199,
                        DataSourceTier.ENTERPRISE: 399
                    }
                    monthly_cost = tier_costs.get(tier, 0)
                
                within_budget = budget_usd is None or monthly_cost <= budget_usd
                
                recommendations.append({
                    "provider": provider_type,
                    "tier": tier,
                    "monthly_cost": monthly_cost,
                    "feature_match_pct": feature_match_pct,
                    "within_budget": within_budget,
                    "features": features,
                    "limits": limits,
                    "score": feature_match_pct * (1.0 if within_budget else 0.5)
                })
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations

    def create_provider_configs(
        self,
        primary_provider: ProviderType,
        secondary_provider: Optional[ProviderType] = None,
        failover_strategy: FailoverStrategy = FailoverStrategy.HEALTH_BASED,
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[ProviderType, ProviderConfig]:
        """
        Create provider configurations for the factory.

        Args:
            primary_provider: Primary data provider
            secondary_provider: Optional secondary provider for failover
            failover_strategy: Failover strategy to use
            custom_settings: Custom settings override

        Returns:
            Dictionary of provider configurations
        """
        configs = {}
        
        # Primary provider configuration
        primary_config = ProviderConfig(
            provider_type=primary_provider,
            priority=1,
            enabled=True,
            config=custom_settings.get(primary_provider.value, {}) if custom_settings else {}
        )
        configs[primary_provider] = primary_config
        
        # Secondary provider configuration
        if secondary_provider and secondary_provider != primary_provider:
            secondary_config = ProviderConfig(
                provider_type=secondary_provider,
                priority=2,
                enabled=True,
                config=custom_settings.get(secondary_provider.value, {}) if custom_settings else {}
            )
            configs[secondary_provider] = secondary_config
        
        return configs

    def validate_symbol_format(self, symbol: str, provider_type: ProviderType) -> str:
        """
        Validate and normalize symbol format for specific provider.

        Args:
            symbol: Stock symbol to validate
            provider_type: Target provider type

        Returns:
            Normalized symbol string
        """
        symbol = symbol.upper().strip()
        
        if provider_type == ProviderType.IEX_CLOUD:
            # IEX Cloud uses simple format: AAPL
            return symbol
        
        elif provider_type == ProviderType.POLYGON_IO:
            # Polygon.io supports different asset classes
            if symbol.startswith(('X:', 'C:')):
                return symbol  # Already formatted for crypto/forex
            else:
                return symbol  # Stocks use simple format
        
        return symbol

    def get_data_retention_policy(self) -> Dict[str, timedelta]:
        """Get data retention policy for different data types."""
        return {
            "real_time_quotes": timedelta(minutes=5),
            "intraday_1m": timedelta(days=7),
            "intraday_5m": timedelta(days=30),
            "daily_ohlc": timedelta(days=365*5),  # 5 years
            "company_info": timedelta(days=30),
            "earnings_data": timedelta(days=90),
            "news_data": timedelta(days=7)
        }

    def get_cache_settings(self) -> Dict[str, Any]:
        """Get cache configuration settings."""
        return {
            "redis_url": "redis://localhost:6379",
            "default_ttl": 300,  # 5 minutes
            "cache_compression": True,
            "cache_serialization": "json",
            "cache_key_prefix": "alphapulse:data:",
            "max_cache_size_mb": 1024,  # 1GB
            "cache_eviction_policy": "lru"
        }


# Global data source configuration instance
_data_source_config: Optional[DataSourceConfig] = None


def get_data_source_config() -> DataSourceConfig:
    """Get the global data source configuration instance."""
    global _data_source_config
    
    if _data_source_config is None:
        _data_source_config = DataSourceConfig()
    
    return _data_source_config


# Default provider configuration
DEFAULT_PROVIDER_SETUP = {
    "primary": ProviderType.IEX_CLOUD,
    "secondary": ProviderType.POLYGON_IO,
    "failover_strategy": FailoverStrategy.HEALTH_BASED,
    "quality_level": DataQualityLevel.STANDARD
}