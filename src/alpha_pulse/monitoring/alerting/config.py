"""
Configuration loading and validation for the alerting system.
"""
from typing import Any, Dict, List, Optional
import logging
import os
import yaml
import uuid

from .models import AlertRule, AlertSeverity


class AlertingConfig:
    """Configuration for the alerting system."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.enabled = config_dict.get("enabled", True)
        self.check_interval = config_dict.get("check_interval", 60)  # seconds
        
        # Parse channels configuration
        self.channels: Dict[str, Dict[str, Any]] = {}
        channels_config = config_dict.get("channels", {})
        for channel_name, channel_config in channels_config.items():
            if channel_config.get("enabled", True):
                self.channels[channel_name] = channel_config
        
        # Parse rules configuration
        self.rules: List[AlertRule] = []
        rules_config = config_dict.get("rules", [])
        for rule_config in rules_config:
            try:
                # Generate rule ID if not provided
                rule_id = rule_config.get("rule_id", str(uuid.uuid4()))
                
                rule = AlertRule(
                    rule_id=rule_id,
                    name=rule_config["name"],
                    description=rule_config.get("description", ""),
                    metric_name=rule_config["metric_name"],
                    condition=rule_config["condition"],
                    severity=rule_config.get("severity", "warning"),
                    message_template=rule_config["message_template"],
                    channels=rule_config["channels"],
                    cooldown_period=rule_config.get("cooldown_period", 3600),
                    enabled=rule_config.get("enabled", True)
                )
                self.rules.append(rule)
            except KeyError as e:
                logging.error(f"Missing required field in rule configuration: {e}")
                continue
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AlertingConfig":
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            AlertingConfig: Loaded configuration
            
        Raises:
            FileNotFoundError: If configuration file not found
            yaml.YAMLError: If YAML parsing fails
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        # Extract alerting section
        alerting_config = config_dict.get("alerting", {})
        return cls(alerting_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AlertingConfig":
        """Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            AlertingConfig: Loaded configuration
        """
        return cls(config_dict.get("alerting", {}))
    
    @classmethod
    def from_env(cls) -> "AlertingConfig":
        """Load configuration from environment variables.
        
        Returns:
            AlertingConfig: Loaded configuration
        """
        # Basic configuration
        config = {
            "enabled": os.environ.get("AP_ALERTING_ENABLED", "true").lower() == "true",
            "check_interval": int(os.environ.get("AP_ALERTING_CHECK_INTERVAL", "60")),
            "channels": {},
            "rules": []
        }
        
        # Email channel configuration
        if os.environ.get("AP_EMAIL_ENABLED", "false").lower() == "true":
            config["channels"]["email"] = {
                "enabled": True,
                "smtp_server": os.environ.get("AP_EMAIL_SMTP_SERVER", ""),
                "smtp_port": int(os.environ.get("AP_EMAIL_SMTP_PORT", "587")),
                "smtp_user": os.environ.get("AP_EMAIL_SMTP_USER", ""),
                "smtp_password": os.environ.get("AP_EMAIL_SMTP_PASSWORD", ""),
                "from_address": os.environ.get("AP_EMAIL_FROM", ""),
                "to_addresses": os.environ.get("AP_EMAIL_TO", "").split(","),
                "use_tls": os.environ.get("AP_EMAIL_USE_TLS", "true").lower() == "true"
            }
        
        # Slack channel configuration
        if os.environ.get("AP_SLACK_ENABLED", "false").lower() == "true":
            config["channels"]["slack"] = {
                "enabled": True,
                "webhook_url": os.environ.get("AP_SLACK_WEBHOOK", ""),
                "channel": os.environ.get("AP_SLACK_CHANNEL", "#alerts"),
                "username": os.environ.get("AP_SLACK_USERNAME", "AlphaPulse Alerting")
            }
        
        # Web channel configuration
        config["channels"]["web"] = {
            "enabled": True,
            "max_alerts": int(os.environ.get("AP_WEB_MAX_ALERTS", "100"))
        }
        
        return cls(config)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AlertingConfig(enabled={self.enabled}, "
            f"check_interval={self.check_interval}, "
            f"channels={list(self.channels.keys())}, "
            f"rules={len(self.rules)})"
        )


def load_alerting_config(config_path: Optional[str] = None) -> AlertingConfig:
    """
    Load alerting configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AlertingConfig: Loaded configuration
    """
    # Check for config file path in environment
    if not config_path:
        config_path = os.environ.get("AP_ALERTING_CONFIG")
    
    # Load from file if available
    if config_path and os.path.exists(config_path):
        return AlertingConfig.from_yaml(config_path)
    
    # Try to load from monitoring config
    monitoring_config_path = os.environ.get("AP_MONITORING_CONFIG")
    if monitoring_config_path and os.path.exists(monitoring_config_path):
        return AlertingConfig.from_yaml(monitoring_config_path)
    
    # Fall back to environment variables
    return AlertingConfig.from_env()