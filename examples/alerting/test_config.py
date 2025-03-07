#!/usr/bin/env python
"""
Script to test loading and validating the alerting configuration.

This script loads the alerting configuration from a YAML file and prints
the loaded configuration details for verification.

Usage:
    python test_config.py [config_path]
"""
import os
import sys
import argparse
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from alpha_pulse.monitoring.alerting.config import AlertingConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config_test")


def test_config(config_path):
    """Load and test the alerting configuration."""
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        # Load configuration from YAML file
        config = AlertingConfig.from_yaml(config_path)
        
        # Print configuration details
        print("\n=== Alerting Configuration ===")
        print(f"Enabled: {config.enabled}")
        print(f"Check Interval: {config.check_interval} seconds")
        
        # Print channels
        print("\n--- Notification Channels ---")
        for channel_name, channel_config in config.channels.items():
            print(f"Channel: {channel_name}")
            print(f"  Type: {channel_config.get('type', channel_name)}")
            print(f"  Enabled: {channel_config.get('enabled', True)}")
            
            # Print channel-specific details
            if channel_name == "email":
                print(f"  SMTP Server: {channel_config.get('smtp_server')}")
                print(f"  From Address: {channel_config.get('from_address')}")
                print(f"  To Addresses: {channel_config.get('to_addresses')}")
            elif channel_name == "slack":
                print(f"  Channel: {channel_config.get('channel')}")
                print(f"  Username: {channel_config.get('username')}")
            elif channel_name == "sms":
                print(f"  From Number: {channel_config.get('from_number')}")
                print(f"  To Numbers: {channel_config.get('to_numbers')}")
            elif channel_name == "web":
                print(f"  Max Alerts: {channel_config.get('max_alerts')}")
        
        # Print rules
        print("\n--- Alert Rules ---")
        for rule in config.rules:
            print(f"Rule: {rule.name} ({rule.rule_id})")
            print(f"  Description: {rule.description}")
            print(f"  Metric: {rule.metric_name}")
            print(f"  Condition: {rule.condition}")
            print(f"  Severity: {rule.severity.value}")
            print(f"  Message: {rule.message_template}")
            print(f"  Channels: {rule.channels}")
            print(f"  Cooldown: {rule.cooldown_period} seconds")
            print(f"  Enabled: {rule.enabled}")
            print()
        
        logger.info("Configuration loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test alerting configuration")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="../../config/alerting_config.yaml",
        help="Path to the alerting configuration YAML file"
    )
    args = parser.parse_args()
    
    # Resolve relative path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config_path))
    
    # Test the configuration
    success = test_config(config_path)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()