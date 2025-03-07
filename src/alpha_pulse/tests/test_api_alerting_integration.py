"""
Integration test for the API and alerting system.

This test verifies that the alerting system is properly integrated with the API.
"""
import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient

from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_alert_manager
from alpha_pulse.monitoring.alerting.models import Alert, AlertSeverity


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_alert_api_integration():
    """Test that alerts from the alerting system appear in the API."""
    # Get the alert manager
    alert_manager = get_alert_manager()
    
    # Create a test alert
    test_alert = Alert(
        rule_id="test_rule",
        metric_name="test_metric",
        metric_value=100.0,
        severity=AlertSeverity.WARNING,
        message="Test alert for API integration",
        alert_id="test_alert_123",
        timestamp=datetime.now()
    )
    
    # Store the alert in the alert history
    await alert_manager.alert_history.store_alert(test_alert)
    
    # Create a test client
    client = TestClient(app)
    
    # Get alerts from the API
    response = client.get(
        "/api/v1/alerts",
        headers={"Authorization": "Bearer mock_token"}
    )
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the test alert is in the response
    alerts = response.json()
    assert len(alerts) > 0
    
    # Find our test alert
    test_alert_found = False
    for alert in alerts:
        if alert.get("id") == "test_alert_123":
            test_alert_found = True
            assert alert["message"] == "Test alert for API integration"
            assert alert["severity"] == "warning"
            assert alert["source"] == "test_metric"
            break
    
    assert test_alert_found, "Test alert not found in API response"


@pytest.mark.asyncio
async def test_alert_acknowledgment():
    """Test that alert acknowledgment works through the API."""
    # Get the alert manager
    alert_manager = get_alert_manager()
    
    # Create a test alert
    test_alert = Alert(
        rule_id="test_rule",
        metric_name="test_metric",
        metric_value=100.0,
        severity=AlertSeverity.WARNING,
        message="Test alert for acknowledgment",
        alert_id="test_alert_456",
        timestamp=datetime.now()
    )
    
    # Store the alert in the alert history
    await alert_manager.alert_history.store_alert(test_alert)
    
    # Create a test client
    client = TestClient(app)
    
    # Acknowledge the alert through the API
    response = client.post(
        f"/api/v1/alerts/test_alert_456/acknowledge",
        headers={"Authorization": "Bearer mock_token"}
    )
    
    # Check that the response is successful
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    
    # Verify that the alert is acknowledged in the alerting system
    alerts = await alert_manager.get_alert_history(
        filters={"alert_id": "test_alert_456"}
    )
    assert len(alerts) == 1
    assert alerts[0].acknowledged is True
    assert alerts[0].acknowledged_by == "admin"  # From the mock user


@pytest.mark.asyncio
async def test_alert_filtering():
    """Test that alert filtering works through the API."""
    # Get the alert manager
    alert_manager = get_alert_manager()
    
    # Create test alerts with different severities
    info_alert = Alert(
        rule_id="test_rule",
        metric_name="test_metric",
        metric_value=50.0,
        severity=AlertSeverity.INFO,
        message="Info test alert",
        alert_id="test_alert_info",
        timestamp=datetime.now()
    )
    
    warning_alert = Alert(
        rule_id="test_rule",
        metric_name="test_metric",
        metric_value=75.0,
        severity=AlertSeverity.WARNING,
        message="Warning test alert",
        alert_id="test_alert_warning",
        timestamp=datetime.now()
    )
    
    critical_alert = Alert(
        rule_id="test_rule",
        metric_name="test_metric",
        metric_value=95.0,
        severity=AlertSeverity.CRITICAL,
        message="Critical test alert",
        alert_id="test_alert_critical",
        timestamp=datetime.now()
    )
    
    # Store the alerts in the alert history
    await alert_manager.alert_history.store_alert(info_alert)
    await alert_manager.alert_history.store_alert(warning_alert)
    await alert_manager.alert_history.store_alert(critical_alert)
    
    # Create a test client
    client = TestClient(app)
    
    # Test filtering by severity
    response = client.get(
        "/api/v1/alerts?severity=critical",
        headers={"Authorization": "Bearer mock_token"}
    )
    
    # Check that the response is successful
    assert response.status_code == 200
    alerts = response.json()
    
    # Verify that only critical alerts are returned
    assert all(alert["severity"] == "critical" for alert in alerts)
    assert any(alert["id"] == "test_alert_critical" for alert in alerts)