"""
Quality rules and configuration management for data quality pipeline.

Provides:
- Centralized quality rule definitions
- Configurable thresholds and parameters
- Symbol-specific quality configurations
- Quality profile management
- Dynamic rule updating and validation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import yaml
from pathlib import Path
from loguru import logger

from alpha_pulse.models.market_data import AssetClass
from alpha_pulse.data.quality.data_validator import QualityDimension
from alpha_pulse.data.quality.anomaly_detector import AnomalyMethod, AnomalySeverity
from alpha_pulse.data.quality.quality_metrics import QualityMetricType, QualityThreshold, QualitySLA


class QualityProfile(Enum):
    """Predefined quality profiles."""
    STRICT = "strict"
    STANDARD = "standard"
    RELAXED = "relaxed"
    CUSTOM = "custom"


class RuleScope(Enum):
    """Scope of quality rules."""
    GLOBAL = "global"
    ASSET_CLASS = "asset_class"
    SYMBOL = "symbol"
    EXCHANGE = "exchange"


@dataclass
class ValidationRuleConfig:
    """Configuration for individual validation rules."""
    rule_name: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    scope: RuleScope = RuleScope.GLOBAL
    scope_value: Optional[str] = None  # Asset class, symbol, or exchange
    priority: int = 100  # Lower number = higher priority
    description: str = ""


@dataclass
class QualityDimensionConfig:
    """Configuration for quality dimensions."""
    dimension: QualityDimension
    weight: float = 1.0
    threshold: float = 0.8
    enabled: bool = True
    rules: List[ValidationRuleConfig] = field(default_factory=list)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    enabled: bool = True
    methods: List[AnomalyMethod] = field(default_factory=lambda: [
        AnomalyMethod.Z_SCORE,
        AnomalyMethod.IQR,
        AnomalyMethod.ISOLATION_FOREST
    ])
    
    # Method-specific parameters
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.1
    isolation_forest_n_estimators: int = 100
    svm_nu: float = 0.1
    
    # LSTM autoencoder parameters
    lstm_sequence_length: int = 50
    lstm_encoding_dim: int = 32
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'z_score': 0.2,
        'iqr': 0.2,
        'isolation_forest': 0.3,
        'one_class_svm': 0.3
    })
    
    # General parameters
    minimum_data_points: int = 30
    historical_window_size: int = 200
    retrain_interval_hours: int = 24
    
    # Severity mapping
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 1.0,
        'medium': 2.0,
        'high': 3.0,
        'critical': 4.0
    })


@dataclass
class QualityMetricsConfig:
    """Configuration for quality metrics."""
    enabled: bool = True
    collection_interval_seconds: int = 300
    retention_days: int = 90
    
    # Metric thresholds
    thresholds: List[QualityThreshold] = field(default_factory=lambda: [
        QualityThreshold(QualityMetricType.COMPLETENESS, 0.90, 0.80, 0.70),
        QualityThreshold(QualityMetricType.ACCURACY, 0.85, 0.75, 0.65),
        QualityThreshold(QualityMetricType.TIMELINESS, 0.85, 0.75, 0.65),
        QualityThreshold(QualityMetricType.CONSISTENCY, 0.80, 0.70, 0.60),
        QualityThreshold(QualityMetricType.RELIABILITY, 0.85, 0.75, 0.65),
        QualityThreshold(QualityMetricType.ANOMALY_RATE, 0.90, 0.80, 0.70)
    ])
    
    # SLA configurations
    slas: List[QualitySLA] = field(default_factory=list)
    
    # Alert settings
    alert_cooldown_minutes: int = 15
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    enable_webhook_alerts: bool = False


@dataclass
class SymbolQualityConfig:
    """Symbol-specific quality configuration."""
    symbol: str
    asset_class: AssetClass
    exchange: Optional[str] = None
    
    # Override global settings
    profile: QualityProfile = QualityProfile.STANDARD
    validation_config: Optional[Dict[str, Any]] = None
    anomaly_config: Optional[AnomalyDetectionConfig] = None
    metrics_config: Optional[QualityMetricsConfig] = None
    
    # Symbol-specific rules
    custom_rules: List[ValidationRuleConfig] = field(default_factory=list)
    
    # Processing settings
    enable_real_time_processing: bool = True
    batch_processing_interval_minutes: int = 60
    priority: int = 100  # Processing priority
    
    # Data retention
    historical_data_retention_days: int = 30
    quality_data_retention_days: int = 90


@dataclass
class QualityRulesConfig:
    """Master configuration for all quality rules."""
    profile: QualityProfile = QualityProfile.STANDARD
    version: str = "1.0.0"
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Global configurations
    validation_config: Dict[str, Any] = field(default_factory=dict)
    anomaly_config: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    metrics_config: QualityMetricsConfig = field(default_factory=QualityMetricsConfig)
    
    # Dimension configurations
    quality_dimensions: Dict[str, QualityDimensionConfig] = field(default_factory=dict)
    
    # Symbol-specific configurations
    symbol_configs: Dict[str, SymbolQualityConfig] = field(default_factory=dict)
    
    # Asset class defaults
    asset_class_defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Global rules
    global_rules: List[ValidationRuleConfig] = field(default_factory=list)


class QualityRulesManager:
    """Manager for quality rules and configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config/quality_rules.yaml")
        self.config: QualityRulesConfig = QualityRulesConfig()
        self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load default quality configuration."""
        # Set up default quality dimensions
        self.config.quality_dimensions = {
            QualityDimension.COMPLETENESS.value: QualityDimensionConfig(
                dimension=QualityDimension.COMPLETENESS,
                weight=0.25,
                threshold=0.95,
                rules=[
                    ValidationRuleConfig(
                        rule_name="required_fields_check",
                        parameters={"required_fields": ["symbol", "timestamp", "ohlcv"]},
                        description="Ensure all required fields are present"
                    ),
                    ValidationRuleConfig(
                        rule_name="ohlcv_completeness",
                        parameters={"required_ohlcv_fields": ["open", "high", "low", "close", "volume"]},
                        description="Ensure OHLCV data is complete"
                    )
                ]
            ),
            QualityDimension.ACCURACY.value: QualityDimensionConfig(
                dimension=QualityDimension.ACCURACY,
                weight=0.30,
                threshold=0.90,
                rules=[
                    ValidationRuleConfig(
                        rule_name="price_reasonableness",
                        parameters={"max_change_percent": 20.0},
                        description="Check for reasonable price movements"
                    ),
                    ValidationRuleConfig(
                        rule_name="ohlc_consistency",
                        parameters={},
                        description="Validate OHLC price relationships"
                    ),
                    ValidationRuleConfig(
                        rule_name="volume_validation",
                        parameters={"min_volume": 0, "volume_spike_threshold": 20.0},
                        description="Validate volume data"
                    )
                ]
            ),
            QualityDimension.CONSISTENCY.value: QualityDimensionConfig(
                dimension=QualityDimension.CONSISTENCY,
                weight=0.20,
                threshold=0.85,
                rules=[
                    ValidationRuleConfig(
                        rule_name="price_continuity",
                        parameters={"max_gap_percent": 10.0},
                        description="Check price continuity between data points"
                    ),
                    ValidationRuleConfig(
                        rule_name="timestamp_sequence",
                        parameters={},
                        description="Ensure timestamps are in sequence"
                    )
                ]
            ),
            QualityDimension.TIMELINESS.value: QualityDimensionConfig(
                dimension=QualityDimension.TIMELINESS,
                weight=0.15,
                threshold=0.90,
                rules=[
                    ValidationRuleConfig(
                        rule_name="data_freshness",
                        parameters={"max_age_minutes": 15},
                        description="Check data freshness"
                    ),
                    ValidationRuleConfig(
                        rule_name="processing_latency",
                        parameters={"max_latency_ms": 5000},
                        description="Monitor processing latency"
                    )
                ]
            ),
            QualityDimension.VALIDITY.value: QualityDimensionConfig(
                dimension=QualityDimension.VALIDITY,
                weight=0.08,
                threshold=0.95,
                rules=[
                    ValidationRuleConfig(
                        rule_name="symbol_format",
                        parameters={"max_length": 10},
                        description="Validate symbol format"
                    ),
                    ValidationRuleConfig(
                        rule_name="price_validity",
                        parameters={"min_price": 0.01, "max_price": 100000},
                        description="Validate price ranges"
                    )
                ]
            ),
            QualityDimension.UNIQUENESS.value: QualityDimensionConfig(
                dimension=QualityDimension.UNIQUENESS,
                weight=0.02,
                threshold=0.98,
                rules=[
                    ValidationRuleConfig(
                        rule_name="duplicate_detection",
                        parameters={"check_window": 10},
                        description="Detect duplicate data points"
                    )
                ]
            )
        }
        
        # Set up global validation rules
        self.config.global_rules = [
            ValidationRuleConfig(
                rule_name="market_hours_check",
                enabled=True,
                parameters={"check_trading_hours": True},
                description="Check if data aligns with market hours"
            ),
            ValidationRuleConfig(
                rule_name="corporate_actions_check",
                enabled=True,
                parameters={"check_splits": True, "check_dividends": True},
                description="Check for corporate actions affecting prices"
            )
        ]
        
        # Set up asset class defaults
        self.config.asset_class_defaults = {
            AssetClass.EQUITY.value: {
                "price_precision": 2,
                "volume_required": True,
                "trading_hours_check": True
            },
            AssetClass.OPTION.value: {
                "price_precision": 2,
                "volume_required": False,
                "trading_hours_check": True,
                "expiration_check": True
            },
            AssetClass.CRYPTOCURRENCY.value: {
                "price_precision": 8,
                "volume_required": True,
                "trading_hours_check": False
            },
            AssetClass.FOREX.value: {
                "price_precision": 5,
                "volume_required": False,
                "trading_hours_check": True
            }
        }
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from file."""
        path = Path(config_path) if config_path else self.config_path
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configuration with loaded data
            self._update_config_from_dict(config_data)
            logger.info(f"Loaded quality configuration from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            raise
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file."""
        path = Path(config_path) if config_path else self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = self._config_to_dict()
            
            with open(path, 'w') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Saved quality configuration to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            raise
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        # This would implement the actual configuration update logic
        # For now, we'll implement a basic version
        if 'profile' in config_data:
            self.config.profile = QualityProfile(config_data['profile'])
        
        if 'anomaly_config' in config_data:
            # Update anomaly configuration
            anomaly_data = config_data['anomaly_config']
            if 'z_score_threshold' in anomaly_data:
                self.config.anomaly_config.z_score_threshold = anomaly_data['z_score_threshold']
            # Add more field updates as needed
        
        # Update last modified timestamp
        self.config.last_updated = datetime.utcnow()
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)
    
    def get_symbol_config(self, symbol: str) -> SymbolQualityConfig:
        """Get quality configuration for a specific symbol."""
        if symbol in self.config.symbol_configs:
            return self.config.symbol_configs[symbol]
        
        # Create default config for symbol
        # This would typically involve looking up the symbol's asset class
        default_config = SymbolQualityConfig(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,  # Default
            profile=self.config.profile
        )
        
        return default_config
    
    def add_symbol_config(self, config: SymbolQualityConfig) -> None:
        """Add or update symbol-specific configuration."""
        self.config.symbol_configs[config.symbol] = config
        self.config.last_updated = datetime.utcnow()
        logger.info(f"Added/updated configuration for symbol {config.symbol}")
    
    def remove_symbol_config(self, symbol: str) -> bool:
        """Remove symbol-specific configuration."""
        if symbol in self.config.symbol_configs:
            del self.config.symbol_configs[symbol]
            self.config.last_updated = datetime.utcnow()
            logger.info(f"Removed configuration for symbol {symbol}")
            return True
        return False
    
    def get_validation_rules(self, symbol: Optional[str] = None) -> List[ValidationRuleConfig]:
        """Get applicable validation rules for a symbol."""
        rules = list(self.config.global_rules)
        
        if symbol and symbol in self.config.symbol_configs:
            symbol_config = self.config.symbol_configs[symbol]
            rules.extend(symbol_config.custom_rules)
        
        # Filter enabled rules and sort by priority
        enabled_rules = [rule for rule in rules if rule.enabled]
        return sorted(enabled_rules, key=lambda x: x.priority)
    
    def get_quality_thresholds(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """Get quality thresholds for a symbol."""
        thresholds = {}
        
        # Start with global dimension thresholds
        for dim_name, dim_config in self.config.quality_dimensions.items():
            thresholds[dim_name] = dim_config.threshold
        
        # Apply symbol-specific overrides if available
        if symbol and symbol in self.config.symbol_configs:
            symbol_config = self.config.symbol_configs[symbol]
            if symbol_config.validation_config:
                thresholds.update(symbol_config.validation_config.get('thresholds', {}))
        
        return thresholds
    
    def get_anomaly_config(self, symbol: Optional[str] = None) -> AnomalyDetectionConfig:
        """Get anomaly detection configuration for a symbol."""
        base_config = self.config.anomaly_config
        
        if symbol and symbol in self.config.symbol_configs:
            symbol_config = self.config.symbol_configs[symbol]
            if symbol_config.anomaly_config:
                return symbol_config.anomaly_config
        
        return base_config
    
    def get_metrics_config(self, symbol: Optional[str] = None) -> QualityMetricsConfig:
        """Get metrics configuration for a symbol."""
        base_config = self.config.metrics_config
        
        if symbol and symbol in self.config.symbol_configs:
            symbol_config = self.config.symbol_configs[symbol]
            if symbol_config.metrics_config:
                return symbol_config.metrics_config
        
        return base_config
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration."""
        errors = []
        
        # Validate dimension weights sum to reasonable total
        total_weight = sum(
            dim_config.weight 
            for dim_config in self.config.quality_dimensions.values()
            if dim_config.enabled
        )
        
        if abs(total_weight - 1.0) > 0.1:
            errors.append(f"Quality dimension weights sum to {total_weight}, should be close to 1.0")
        
        # Validate thresholds are in valid range
        for dim_name, dim_config in self.config.quality_dimensions.items():
            if not 0.0 <= dim_config.threshold <= 1.0:
                errors.append(f"Threshold for {dim_name} is {dim_config.threshold}, should be between 0.0 and 1.0")
        
        # Validate anomaly detection parameters
        anomaly_config = self.config.anomaly_config
        if anomaly_config.z_score_threshold <= 0:
            errors.append("Z-score threshold must be positive")
        
        if anomaly_config.iqr_multiplier <= 0:
            errors.append("IQR multiplier must be positive")
        
        # Validate ensemble weights
        ensemble_total = sum(anomaly_config.ensemble_weights.values())
        if abs(ensemble_total - 1.0) > 0.1:
            errors.append(f"Ensemble weights sum to {ensemble_total}, should be close to 1.0")
        
        return errors
    
    def apply_profile(self, profile: QualityProfile) -> None:
        """Apply a predefined quality profile."""
        self.config.profile = profile
        
        if profile == QualityProfile.STRICT:
            # Strict profile: Higher thresholds, more sensitive detection
            self._apply_strict_profile()
        elif profile == QualityProfile.STANDARD:
            # Standard profile: Balanced settings (already set as defaults)
            pass
        elif profile == QualityProfile.RELAXED:
            # Relaxed profile: Lower thresholds, less sensitive detection
            self._apply_relaxed_profile()
        
        self.config.last_updated = datetime.utcnow()
        logger.info(f"Applied quality profile: {profile.value}")
    
    def _apply_strict_profile(self) -> None:
        """Apply strict quality profile settings."""
        # Increase thresholds
        for dim_config in self.config.quality_dimensions.values():
            dim_config.threshold = min(1.0, dim_config.threshold + 0.05)
        
        # More sensitive anomaly detection
        self.config.anomaly_config.z_score_threshold = 2.5
        self.config.anomaly_config.iqr_multiplier = 1.2
        
        # Stricter metrics thresholds
        for threshold in self.config.metrics_config.thresholds:
            threshold.warning_threshold = min(1.0, threshold.warning_threshold + 0.05)
            threshold.critical_threshold = min(1.0, threshold.critical_threshold + 0.05)
    
    def _apply_relaxed_profile(self) -> None:
        """Apply relaxed quality profile settings."""
        # Lower thresholds
        for dim_config in self.config.quality_dimensions.values():
            dim_config.threshold = max(0.5, dim_config.threshold - 0.05)
        
        # Less sensitive anomaly detection
        self.config.anomaly_config.z_score_threshold = 3.5
        self.config.anomaly_config.iqr_multiplier = 2.0
        
        # More lenient metrics thresholds
        for threshold in self.config.metrics_config.thresholds:
            threshold.warning_threshold = max(0.5, threshold.warning_threshold - 0.05)
            threshold.critical_threshold = max(0.4, threshold.critical_threshold - 0.05)


# Global configuration manager instance
_quality_rules_manager: Optional[QualityRulesManager] = None


def get_quality_rules_manager(config_path: Optional[str] = None) -> QualityRulesManager:
    """Get the global quality rules manager instance."""
    global _quality_rules_manager
    
    if _quality_rules_manager is None:
        _quality_rules_manager = QualityRulesManager(config_path)
    
    return _quality_rules_manager