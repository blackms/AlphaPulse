"""
Demo script for the alerting system integration with the API.

This script demonstrates how to:
1. Start the API server
2. Generate alerts through the alerting system
3. Retrieve alerts through the API
4. Acknowledge alerts through the API
"""
import asyncio
import json
import logging
import sys
import os
import uuid
from datetime import datetime
import requests
import time
import subprocess
import signal

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from alpha_pulse.monitoring.alerting import AlertManager, AlertRule, AlertSeverity, Alert
from alpha_pulse.monitoring.alerting.config import load_alerting_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API base URL
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "mock_token"


def start_api_server():
    """Start the API server in a separate process."""
    logger.info("Starting API server...")
    
    # Start the API server using uvicorn
    process = subprocess.Popen(
        ["uvicorn", "src.alpha_pulse.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Wait for the server to start
    time.sleep(3)
    
    # Check if the server is running
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            logger.info("API server started successfully")
            return process
        else:
            logger.error(f"API server returned unexpected status code: {response.status_code}")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return None
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to API server")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return None


def stop_api_server(process):
    """Stop the API server."""
    if process:
        logger.info("Stopping API server...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        logger.info("API server stopped")


def get_auth_headers():
    """Get authentication headers for API requests."""
    return {
        "Authorization": f"Bearer {API_TOKEN}"
    }


def get_alerts():
    """Get alerts from the API."""
    response = requests.get(
        f"{API_BASE_URL}/api/v1/alerts",
        headers=get_auth_headers()
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to get alerts: {response.status_code} {response.text}")
        return []


def acknowledge_alert(alert_id):
    """Acknowledge an alert through the API."""
    response = requests.post(
        f"{API_BASE_URL}/api/v1/alerts/{alert_id}/acknowledge",
        headers=get_auth_headers()
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to acknowledge alert: {response.status_code} {response.text}")
        return {"success": False}


async def generate_test_alerts(alert_manager):
    """Generate test alerts through the alerting system."""
    # Create test alerts with different severities
    alerts = [
        Alert(
            rule_id="test_rule",
            metric_name="cpu_usage",
            metric_value=85.0,
            severity=AlertSeverity.WARNING,
            message="CPU usage is high (85%)",
            alert_id=f"cpu_alert_{uuid.uuid4()}",
            timestamp=datetime.now()
        ),
        Alert(
            rule_id="test_rule",
            metric_name="memory_usage",
            metric_value=95.0,
            severity=AlertSeverity.CRITICAL,
            message="Memory usage is critical (95%)",
            alert_id=f"memory_alert_{uuid.uuid4()}",
            timestamp=datetime.now()
        ),
        Alert(
            rule_id="test_rule",
            metric_name="disk_space",
            metric_value=10.0,
            severity=AlertSeverity.INFO,
            message="Disk space is low (10% free)",
            alert_id=f"disk_alert_{uuid.uuid4()}",
            timestamp=datetime.now()
        )
    ]
    
    # Store the alerts in the alert history
    for alert in alerts:
        await alert_manager.alert_history.store_alert(alert)
        logger.info(f"Generated alert: {alert.alert_id} - {alert.message}")
    
    return alerts


async def main():
    """Main function."""
    # Start the API server
    api_process = start_api_server()
    if not api_process:
        logger.error("Failed to start API server")
        return
    
    try:
        # Load alerting configuration
        alerting_config = load_alerting_config()
        
        # Create alert manager
        alert_manager = AlertManager(alerting_config)
        
        # Start the alert manager
        await alert_manager.start()
        
        # Generate test alerts
        alerts = await generate_test_alerts(alert_manager)
        
        # Wait for the alerts to be processed
        logger.info("Waiting for alerts to be processed...")
        await asyncio.sleep(2)
        
        # Get alerts from the API
        api_alerts = get_alerts()
        logger.info(f"Retrieved {len(api_alerts)} alerts from the API")
        
        # Print the alerts
        for alert in api_alerts:
            logger.info(f"Alert: {alert['id']} - {alert['message']} ({alert['severity']})")
        
        # Acknowledge one of the alerts
        if api_alerts:
            alert_to_acknowledge = api_alerts[0]
            logger.info(f"Acknowledging alert: {alert_to_acknowledge['id']}")
            result = acknowledge_alert(alert_to_acknowledge['id'])
            
            if result["success"]:
                logger.info(f"Alert acknowledged successfully")
            else:
                logger.error(f"Failed to acknowledge alert: {result}")
        
        # Get alerts again to see the acknowledged alert
        api_alerts = get_alerts()
        logger.info(f"Retrieved {len(api_alerts)} alerts from the API after acknowledgment")
        
        # Print the alerts
        for alert in api_alerts:
            acknowledged_status = "Acknowledged" if alert["acknowledged"] else "Not acknowledged"
            logger.info(f"Alert: {alert['id']} - {alert['message']} ({alert['severity']}) - {acknowledged_status}")
        
        # Stop the alert manager
        await alert_manager.stop()
        
    finally:
        # Stop the API server
        stop_api_server(api_process)


if __name__ == "__main__":
    asyncio.run(main())