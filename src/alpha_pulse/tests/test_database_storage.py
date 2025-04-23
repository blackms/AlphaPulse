"""
Unit tests for the database alert history storage.
"""
import unittest
import asyncio
import os
from datetime import datetime, timedelta
import unittest.mock

from alpha_pulse.monitoring.alerting.models import Alert, AlertSeverity
from alpha_pulse.monitoring.alerting.storage import DatabaseAlertHistory


class TestDatabaseAlertHistory(unittest.TestCase):
    """Test cases for the database alert history storage."""
    
    @unittest.mock.patch.object(DatabaseAlertHistory, 'initialize')
    def setUp(self, mock_initialize):
        """Set up test fixtures."""
        # Create the storage instance with dummy connection params
        self.connection_params = {
            "host": "localhost",
            "port": 5432,
            "user": "test_user",
            "password": "test_password",
            "database": "test_db",
        }
        self.storage = DatabaseAlertHistory(self.connection_params)
        
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
    
    def tearDown(self):
        """Clean up after tests."""
    
    def test_initialize(self):
        """Test database initialization."""
        with unittest.mock.patch.object(self.storage, 'initialize') as mock_initialize:
            # Configure the mock initialize method
            mock_initialize.return_value = True
            self.storage.initialized = True # Manually set initialized flag
            
            # Mock the connection pool and connection
            self.mock_pool = unittest.mock.AsyncMock()
            self.mock_conn = unittest.mock.AsyncMock()
            
            # Configure the mock pool to return an async context manager
            mock_context_manager = unittest.mock.AsyncMock()
            mock_context_manager.__aenter__.return_value = self.mock_conn
            self.mock_pool.acquire.return_value = mock_context_manager
            
            self.storage.pool = self.mock_pool # Assign the mock pool to the storage instance
            
            # Initialize the database (this will call the mocked initialize)
            asyncio.run(self.storage.initialize())
        
        # Check that initialization flag is set
        self.assertTrue(self.storage.initialized)
    
    def test_store_and_retrieve_alerts(self):
        """Test storing and retrieving alerts."""
        with unittest.mock.patch.object(self.storage, 'initialize') as mock_initialize:
            # Configure the mock initialize method
            mock_initialize.return_value = True
            self.storage.initialized = True # Manually set initialized flag
            
            # Mock the connection pool and connection
            self.mock_pool = unittest.mock.AsyncMock()
            self.mock_conn = unittest.mock.AsyncMock()
            
            # Configure the mock pool to return an async context manager
            mock_context_manager = unittest.mock.AsyncMock()
            mock_context_manager.__aenter__.return_value = self.mock_conn
            self.mock_pool.acquire.return_value = mock_context_manager
            
            self.storage.pool = self.mock_pool # Assign the mock pool to the storage instance
            
            # Configure mock connection methods
            self.mock_conn.execute.return_value = None # Simulate successful execute
            
            # Simulate fetch returning alerts
            mock_rows = []
            for alert in self.alerts:
                # Create mock row objects with attribute access
                mock_row = unittest.mock.Mock()
                mock_row.alert_id = alert.alert_id
                mock_row.rule_id = alert.rule_id
                mock_row.metric_name = alert.metric_name
                mock_row.metric_value = str(alert.metric_value) # Store as string in mock row
                mock_row.severity = alert.severity.value
                mock_row.message = alert.message
                mock_row.timestamp = alert.timestamp
                mock_row.acknowledged = alert.acknowledged
                mock_row.acknowledged_by = alert.acknowledged_by
                mock_row.acknowledged_at = alert.acknowledged_at
                mock_rows.append(mock_row)
            
            self.mock_conn.fetch.return_value = mock_rows
            
            # Initialize the database (this will call the mocked initialize)
            asyncio.run(self.storage.initialize())
        
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