"""
Unit tests for the database alert history storage.
"""
import unittest
import asyncio
import os
import tempfile
from datetime import datetime, timedelta

from alpha_pulse.monitoring.alerting.models import Alert, AlertSeverity
from alpha_pulse.monitoring.alerting.storage import DatabaseAlertHistory


class TestDatabaseAlertHistory(unittest.TestCase):
    """Test cases for the database alert history storage."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_alerts.db")
        
        # Create the storage instance
        self.storage = DatabaseAlertHistory(self.db_path)
        
        # Create test alerts
        self.alerts = [
            Alert(
                rule_id="test_rule_1",
                metric_name="test_metric_1",
                metric_value=0.75,
                severity=AlertSeverity.WARNING,
                message="Test warning alert",
                alert_id="alert-001",
                timestamp=datetime.now() - timedelta(hours=2)
            ),
            Alert(
                rule_id="test_rule_2",
                metric_name="test_metric_2",
                metric_value=0.25,
                severity=AlertSeverity.ERROR,
                message="Test error alert",
                alert_id="alert-002",
                timestamp=datetime.now() - timedelta(hours=1)
            ),
            Alert(
                rule_id="test_rule_3",
                metric_name="test_metric_1",
                metric_value=0.95,
                severity=AlertSeverity.CRITICAL,
                message="Test critical alert",
                alert_id="alert-003",
                timestamp=datetime.now()
            )
        ]
        
        # Initialize the database
        asyncio.run(self.storage.initialize())
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialize(self):
        """Test database initialization."""
        # Check that the database file was created
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check that initialization flag is set
        self.assertTrue(self.storage.initialized)
    
    def test_store_and_retrieve_alerts(self):
        """Test storing and retrieving alerts."""
        # Store test alerts
        for alert in self.alerts:
            asyncio.run(self.storage.store_alert(alert))
        
        # Retrieve all alerts
        retrieved_alerts = asyncio.run(self.storage.get_alerts())
        
        # Check that all alerts were retrieved
        self.assertEqual(len(retrieved_alerts), len(self.alerts))
        
        # Check that alerts are sorted by timestamp (newest first)
        self.assertEqual(retrieved_alerts[0].alert_id, "alert-003")
        self.assertEqual(retrieved_alerts[1].alert_id, "alert-002")
        self.assertEqual(retrieved_alerts[2].alert_id, "alert-001")
    
    def test_filter_by_severity(self):
        """Test filtering alerts by severity."""
        # Store test alerts
        for alert in self.alerts:
            asyncio.run(self.storage.store_alert(alert))
        
        # Filter by severity
        warning_alerts = asyncio.run(self.storage.get_alerts(
            filters={"severity": "warning"}
        ))
        error_alerts = asyncio.run(self.storage.get_alerts(
            filters={"severity": "error"}
        ))
        critical_alerts = asyncio.run(self.storage.get_alerts(
            filters={"severity": "critical"}
        ))
        
        # Check filter results
        self.assertEqual(len(warning_alerts), 1)
        self.assertEqual(warning_alerts[0].alert_id, "alert-001")
        
        self.assertEqual(len(error_alerts), 1)
        self.assertEqual(error_alerts[0].alert_id, "alert-002")
        
        self.assertEqual(len(critical_alerts), 1)
        self.assertEqual(critical_alerts[0].alert_id, "alert-003")
    
    def test_filter_by_metric_name(self):
        """Test filtering alerts by metric name."""
        # Store test alerts
        for alert in self.alerts:
            asyncio.run(self.storage.store_alert(alert))
        
        # Filter by metric name
        metric1_alerts = asyncio.run(self.storage.get_alerts(
            filters={"metric_name": "test_metric_1"}
        ))
        
        # Check filter results
        self.assertEqual(len(metric1_alerts), 2)
        self.assertEqual(metric1_alerts[0].alert_id, "alert-003")
        self.assertEqual(metric1_alerts[1].alert_id, "alert-001")
    
    def test_filter_by_time_range(self):
        """Test filtering alerts by time range."""
        # Store test alerts
        for alert in self.alerts:
            asyncio.run(self.storage.store_alert(alert))
        
        # Filter by time range
        now = datetime.now()
        start_time = now - timedelta(hours=1, minutes=30)
        end_time = now - timedelta(minutes=30)
        
        time_range_alerts = asyncio.run(self.storage.get_alerts(
            start_time=start_time,
            end_time=end_time
        ))
        
        # Check filter results
        self.assertEqual(len(time_range_alerts), 1)
        self.assertEqual(time_range_alerts[0].alert_id, "alert-002")
    
    def test_update_alert(self):
        """Test updating an alert."""
        # Store test alerts
        for alert in self.alerts:
            asyncio.run(self.storage.store_alert(alert))
        
        # Update an alert
        update_result = asyncio.run(self.storage.update_alert(
            "alert-002",
            {
                "acknowledged": True,
                "acknowledged_by": "test_user",
                "acknowledged_at": datetime.now()
            }
        ))
        
        # Check update result
        self.assertTrue(update_result)
        
        # Retrieve the updated alert
        retrieved_alerts = asyncio.run(self.storage.get_alerts(
            filters={"acknowledged": True}
        ))
        
        # Check that the alert was updated
        self.assertEqual(len(retrieved_alerts), 1)
        self.assertEqual(retrieved_alerts[0].alert_id, "alert-002")
        self.assertTrue(retrieved_alerts[0].acknowledged)
        self.assertEqual(retrieved_alerts[0].acknowledged_by, "test_user")
        self.assertIsNotNone(retrieved_alerts[0].acknowledged_at)
    
    def test_update_nonexistent_alert(self):
        """Test updating a non-existent alert."""
        # Try to update a non-existent alert
        update_result = asyncio.run(self.storage.update_alert(
            "nonexistent-alert",
            {"acknowledged": True}
        ))
        
        # Check that update failed
        self.assertFalse(update_result)


if __name__ == '__main__':
    unittest.main()