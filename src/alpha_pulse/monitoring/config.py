"""
Configuration for the monitoring system.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import timedelta
import os
import yaml


@dataclass
class StorageConfig:
    """Configuration for time series storage."""
    
    # Storage type: 'influxdb', 'timescaledb', or 'memory'
    type: str = "memory"
    
    # Common settings
    retention_days: int = 30
    
    # InfluxDB settings
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = "alpha_pulse"
    influxdb_bucket: str = "metrics"
    
    # TimescaleDB settings
    timescaledb_host: str = "localhost"
    timescaledb_port: int = 5432
    timescaledb_user: str = "postgres"
    timescaledb_password: str = ""
    timescaledb_database: str = "alpha_pulse"
    timescaledb_schema: str = "public"
    
    # Memory storage settings
    memory_max_points: int = 10000
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get storage-specific configuration.
        
        Returns:
            Dictionary of configuration parameters for the selected storage type
        """
        if self.type == "influxdb":
            return {
                "url": self.influxdb_url,
                "token": self.influxdb_token,
                "org": self.influxdb_org,
                "bucket": self.influxdb_bucket,
                "retention_duration": timedelta(days=self.retention_days)
            }
        elif self.type == "timescaledb":
            return {
                "host": self.timescaledb_host,
                "port": self.timescaledb_port,
                "user": self.timescaledb_user,
                "password": self.timescaledb_password,
                "database": self.timescaledb_database,
                "schema": self.timescaledb_schema,
                "retention_duration": timedelta(days=self.retention_days)
            }
        elif self.type == "memory":
            return {
                "max_points": self.memory_max_points,
                "retention_duration": timedelta(days=self.retention_days)
            }
        else:
            raise ValueError(f"Unsupported storage type: {self.type}")


@dataclass
class AlertConfig:
    """Configuration for an alert rule."""
    
    # Alert name
    name: str
    
    # Metric type this alert applies to
    metric_type: str
    
    # Field within the metric to check
    field: str
    
    # Condition type: 'threshold', 'change', 'anomaly'
    condition_type: str
    
    # Condition parameters
    condition_params: Dict[str, Any] = field(default_factory=dict)
    
    # Alert severity: 'info', 'warning', 'error', 'critical'
    severity: str = "warning"
    
    # Message template
    message: str = ""
    
    # Notification channels to use
    channels: List[str] = field(default_factory=list)
    
    def get_condition_function(self):
        """
        Get the condition function for this alert.
        
        Returns:
            Function that takes a value and returns True if the alert should trigger
        """
        if self.condition_type == "threshold":
            # Threshold condition
            operator = self.condition_params.get("operator", ">")
            threshold = self.condition_params.get("value", 0)
            
            if operator == ">":
                return lambda value: value > threshold
            elif operator == ">=":
                return lambda value: value >= threshold
            elif operator == "<":
                return lambda value: value < threshold
            elif operator == "<=":
                return lambda value: value <= threshold
            elif operator == "==":
                return lambda value: value == threshold
            elif operator == "!=":
                return lambda value: value != threshold
            else:
                raise ValueError(f"Unsupported operator: {operator}")
                
        elif self.condition_type == "change":
            # Change condition
            percent = self.condition_params.get("percent", 10)
            direction = self.condition_params.get("direction", "any")
            
            if direction == "increase":
                return lambda old, new: (new - old) / old * 100 > percent if old != 0 else False
            elif direction == "decrease":
                return lambda old, new: (old - new) / old * 100 > percent if old != 0 else False
            elif direction == "any":
                return lambda old, new: abs(new - old) / old * 100 > percent if old != 0 else False
            else:
                raise ValueError(f"Unsupported direction: {direction}")
                
        elif self.condition_type == "anomaly":
            # Anomaly condition (simple z-score)
            threshold = self.condition_params.get("threshold", 3.0)
            
            def is_anomaly(value, history):
                if not history or len(history) < 2:
                    return False
                    
                mean = sum(history) / len(history)
                variance = sum((x - mean) ** 2 for x in history) / len(history)
                std_dev = variance ** 0.5
                
                if std_dev == 0:
                    return False
                    
                z_score = abs(value - mean) / std_dev
                return z_score > threshold
                
            return is_anomaly
            
        else:
            raise ValueError(f"Unsupported condition type: {self.condition_type}")


@dataclass
class NotificationChannelConfig:
    """Configuration for a notification channel."""
    
    # Channel name
    name: str
    
    # Channel type: 'email', 'slack', 'webhook'
    type: str
    
    # Channel-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    
    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Alert configurations
    alerts: List[AlertConfig] = field(default_factory=list)
    
    # Notification channels
    notification_channels: List[NotificationChannelConfig] = field(default_factory=list)
    
    # Collection interval in seconds
    collection_interval: int = 60
    
    # Whether to enable real-time monitoring
    enable_realtime: bool = True
    
    # Whether to collect API latency metrics
    collect_api_latency: bool = True
    
    # Whether to collect trade metrics
    collect_trade_metrics: bool = True
    
    # Whether to collect agent performance metrics
    collect_agent_metrics: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MonitoringConfig':
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            MonitoringConfig instance
        """
        # Create storage config
        storage_dict = config_dict.get("storage", {})
        storage = StorageConfig(
            type=storage_dict.get("type", "memory"),
            retention_days=storage_dict.get("retention_days", 30),
            influxdb_url=storage_dict.get("influxdb_url", "http://localhost:8086"),
            influxdb_token=storage_dict.get("influxdb_token", ""),
            influxdb_org=storage_dict.get("influxdb_org", "alpha_pulse"),
            influxdb_bucket=storage_dict.get("influxdb_bucket", "metrics"),
            timescaledb_host=storage_dict.get("timescaledb_host", "localhost"),
            timescaledb_port=storage_dict.get("timescaledb_port", 5432),
            timescaledb_user=storage_dict.get("timescaledb_user", "postgres"),
            timescaledb_password=storage_dict.get("timescaledb_password", ""),
            timescaledb_database=storage_dict.get("timescaledb_database", "alpha_pulse"),
            timescaledb_schema=storage_dict.get("timescaledb_schema", "public"),
            memory_max_points=storage_dict.get("memory_max_points", 10000)
        )
        
        # Create alert configs
        alerts = []
        for alert_dict in config_dict.get("alerts", []):
            alerts.append(AlertConfig(
                name=alert_dict.get("name", ""),
                metric_type=alert_dict.get("metric_type", ""),
                field=alert_dict.get("field", ""),
                condition_type=alert_dict.get("condition_type", "threshold"),
                condition_params=alert_dict.get("condition_params", {}),
                severity=alert_dict.get("severity", "warning"),
                message=alert_dict.get("message", ""),
                channels=alert_dict.get("channels", [])
            ))
        
        # Create notification channel configs
        channels = []
        for channel_dict in config_dict.get("notification_channels", []):
            channels.append(NotificationChannelConfig(
                name=channel_dict.get("name", ""),
                type=channel_dict.get("type", ""),
                params=channel_dict.get("params", {})
            ))
        
        # Create monitoring config
        return cls(
            storage=storage,
            alerts=alerts,
            notification_channels=channels,
            collection_interval=config_dict.get("collection_interval", 60),
            enable_realtime=config_dict.get("enable_realtime", True),
            collect_api_latency=config_dict.get("collect_api_latency", True),
            collect_trade_metrics=config_dict.get("collect_trade_metrics", True),
            collect_agent_metrics=config_dict.get("collect_agent_metrics", True)
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MonitoringConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            MonitoringConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        """
        Load configuration from environment variables.
        
        Returns:
            MonitoringConfig instance
        """
        # Create storage config
        storage = StorageConfig(
            type=os.environ.get("AP_MONITORING_STORAGE_TYPE", "memory"),
            retention_days=int(os.environ.get("AP_MONITORING_RETENTION_DAYS", "30")),
            influxdb_url=os.environ.get("AP_INFLUXDB_URL", "http://localhost:8086"),
            influxdb_token=os.environ.get("AP_INFLUXDB_TOKEN", ""),
            influxdb_org=os.environ.get("AP_INFLUXDB_ORG", "alpha_pulse"),
            influxdb_bucket=os.environ.get("AP_INFLUXDB_BUCKET", "metrics"),
            timescaledb_host=os.environ.get("AP_TIMESCALEDB_HOST", "localhost"),
            timescaledb_port=int(os.environ.get("AP_TIMESCALEDB_PORT", "5432")),
            timescaledb_user=os.environ.get("AP_TIMESCALEDB_USER", "postgres"),
            timescaledb_password=os.environ.get("AP_TIMESCALEDB_PASSWORD", ""),
            timescaledb_database=os.environ.get("AP_TIMESCALEDB_DATABASE", "alpha_pulse"),
            timescaledb_schema=os.environ.get("AP_TIMESCALEDB_SCHEMA", "public"),
            memory_max_points=int(os.environ.get("AP_MEMORY_MAX_POINTS", "10000"))
        )
        
        # Create monitoring config
        return cls(
            storage=storage,
            collection_interval=int(os.environ.get("AP_MONITORING_INTERVAL", "60")),
            enable_realtime=os.environ.get("AP_MONITORING_REALTIME", "true").lower() == "true",
            collect_api_latency=os.environ.get("AP_COLLECT_API_LATENCY", "true").lower() == "true",
            collect_trade_metrics=os.environ.get("AP_COLLECT_TRADE_METRICS", "true").lower() == "true",
            collect_agent_metrics=os.environ.get("AP_COLLECT_AGENT_METRICS", "true").lower() == "true"
        )


def load_config(config_path: Optional[str] = None) -> MonitoringConfig:
    """
    Load monitoring configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        MonitoringConfig instance
    """
    # Check for config file path in environment
    if not config_path:
        config_path = os.environ.get("AP_MONITORING_CONFIG")
    
    # Load from file if available
    if config_path and os.path.exists(config_path):
        return MonitoringConfig.from_yaml(config_path)
    
    # Fall back to environment variables
    return MonitoringConfig.from_env()