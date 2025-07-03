"""
Tests for audit logging system.
"""

import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
import threading
import queue

from alpha_pulse.utils.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditSeverity,
    AuditContext,
    AuditLog,
    get_audit_logger,
    audit_call
)
from alpha_pulse.utils.audit_queries import (
    AuditQueryBuilder,
    AuditReporter,
    generate_audit_summary
)


class TestAuditLogger:
    """Test cases for AuditLogger."""
    
    @pytest.fixture
    def audit_logger(self):
        """Create a test audit logger."""
        logger = AuditLogger(batch_size=5, flush_interval=1.0)
        yield logger
        logger.shutdown(timeout=5.0)
        
    def test_basic_logging(self, audit_logger):
        """Test basic audit logging functionality."""
        # Log an event
        audit_logger.log(
            event_type=AuditEventType.AUTH_LOGIN,
            event_data={'user_id': 'test_user'},
            severity=AuditSeverity.INFO,
            success=True
        )
        
        # Verify event was queued
        assert audit_logger._queue.qsize() > 0
        
    def test_context_management(self, audit_logger):
        """Test audit context management."""
        # Set context
        with audit_logger.context(user_id="test_user", ip_address="127.0.0.1"):
            context = audit_logger.get_context()
            assert context.user_id == "test_user"
            assert context.ip_address == "127.0.0.1"
            
            # Log event with context
            audit_logger.log(
                event_type=AuditEventType.API_REQUEST,
                event_data={'method': 'GET', 'path': '/api/test'}
            )
            
        # Context should be cleared
        context = audit_logger.get_context()
        assert context.user_id is None
        assert context.ip_address is None
        
    def test_nested_contexts(self, audit_logger):
        """Test nested audit contexts."""
        with audit_logger.context(user_id="user1"):
            assert audit_logger.get_context().user_id == "user1"
            
            with audit_logger.context(ip_address="192.168.1.1"):
                context = audit_logger.get_context()
                assert context.user_id == "user1"  # Inherited
                assert context.ip_address == "192.168.1.1"
                
            # Inner context cleared, outer remains
            context = audit_logger.get_context()
            assert context.user_id == "user1"
            assert context.ip_address is None
            
    def test_convenience_methods(self, audit_logger):
        """Test convenience logging methods."""
        # Test login logging
        audit_logger.log_login("test_user", success=True, method="password")
        audit_logger.log_login("bad_user", success=False, method="password", error="Invalid password")
        
        # Test API logging
        audit_logger.log_api_request("GET", "/api/users", status_code=200, duration_ms=50.5)
        
        # Test trade logging
        audit_logger.log_trade_decision(
            agent="TechnicalAgent",
            symbol="BTC/USD",
            action="buy",
            quantity=0.5,
            reasoning={'rsi': 30, 'macd': 'bullish'},
            confidence=0.85
        )
        
        audit_logger.log_trade_execution(
            order_id="12345",
            symbol="BTC/USD",
            action="buy",
            quantity=0.5,
            price=50000,
            success=True
        )
        
        # Test risk logging
        audit_logger.log_risk_event(
            risk_type="position_size",
            threshold=0.1,
            actual_value=0.15,
            action_taken="reduce_position"
        )
        
        # Test secret access logging
        audit_logger.log_secret_access("api_key", "trading_execution")
        
        # Verify events were queued
        assert audit_logger._queue.qsize() >= 7
        
    @patch('alpha_pulse.utils.audit_logger.get_db_session')
    def test_batch_flushing(self, mock_get_session, audit_logger):
        """Test batch flushing behavior."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        # Log events up to batch size
        for i in range(5):  # batch_size=5
            audit_logger.log(
                event_type=AuditEventType.API_REQUEST,
                event_data={'request_id': i}
            )
            
        # Wait for flush
        time.sleep(0.5)
        
        # Verify bulk save was called
        mock_session.bulk_save_objects.assert_called()
        mock_session.commit.assert_called()
        
    @patch('alpha_pulse.utils.audit_logger.get_db_session')
    def test_flush_on_interval(self, mock_get_session, audit_logger):
        """Test flushing based on time interval."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        # Log a single event
        audit_logger.log(
            event_type=AuditEventType.API_REQUEST,
            event_data={'test': 'interval_flush'}
        )
        
        # Wait for flush interval
        time.sleep(1.5)  # flush_interval=1.0
        
        # Verify flush occurred
        mock_session.bulk_save_objects.assert_called()
        
    def test_queue_overflow_handling(self):
        """Test behavior when queue is full."""
        # Create logger with small queue
        logger = AuditLogger(batch_size=1, max_queue_size=2)
        
        try:
            # Fill the queue
            for i in range(5):
                logger.log(
                    event_type=AuditEventType.API_REQUEST,
                    event_data={'request_id': i}
                )
                
            # Queue should be at max size, additional events dropped
            assert logger._queue.qsize() <= 2
            
        finally:
            logger.shutdown(timeout=1.0)
            
    def test_thread_safety(self, audit_logger):
        """Test thread-safe logging."""
        events_logged = []
        
        def log_events(thread_id):
            """Log events from a thread."""
            with audit_logger.context(user_id=f"thread_{thread_id}"):
                for i in range(10):
                    audit_logger.log(
                        event_type=AuditEventType.API_REQUEST,
                        event_data={'thread': thread_id, 'request': i}
                    )
                    events_logged.append((thread_id, i))
                    
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_events, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify all events were logged
        assert len(events_logged) == 50  # 5 threads * 10 events
        
    @patch('alpha_pulse.utils.audit_logger.get_db_session')
    def test_error_resilience(self, mock_get_session, audit_logger):
        """Test resilience to database errors."""
        # Mock database error
        mock_session = MagicMock()
        mock_session.bulk_save_objects.side_effect = Exception("DB Error")
        mock_get_session.return_value = mock_session
        
        # Log events
        for i in range(10):
            audit_logger.log(
                event_type=AuditEventType.API_REQUEST,
                event_data={'request_id': i}
            )
            
        # Wait for flush attempt
        time.sleep(0.5)
        
        # Logger should continue despite errors
        # (In production, might implement fallback to file logging)
        
    def test_shutdown(self, audit_logger):
        """Test graceful shutdown."""
        # Log some events
        for i in range(3):
            audit_logger.log(
                event_type=AuditEventType.API_REQUEST,
                event_data={'request_id': i}
            )
            
        # Shutdown
        audit_logger.shutdown(timeout=2.0)
        
        # Worker thread should be stopped
        assert audit_logger._stop_event.is_set()
        assert not audit_logger._flush_thread.is_alive()


class TestAuditDecorator:
    """Test cases for audit decorators."""
    
    def test_audit_call_decorator(self):
        """Test the audit_call decorator."""
        
        @audit_call(
            AuditEventType.TRADE_DECISION,
            extract_data=lambda *args, **kwargs: {'symbol': args[0]}
        )
        def make_decision(symbol, quantity):
            return {'action': 'buy', 'quantity': quantity}
            
        with patch('alpha_pulse.utils.audit_logger.get_audit_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Call decorated function
            result = make_decision("BTC/USD", 0.5)
            
            # Verify audit log was created
            mock_logger.log.assert_called_once()
            call_args = mock_logger.log.call_args
            
            assert call_args.kwargs['event_type'] == AuditEventType.TRADE_DECISION
            assert call_args.kwargs['event_data']['symbol'] == "BTC/USD"
            assert call_args.kwargs['success'] is True
            assert 'duration_ms' in call_args.kwargs
            
    def test_audit_call_with_exception(self):
        """Test audit_call decorator with exception."""
        
        @audit_call(AuditEventType.TRADE_DECISION)
        def failing_function():
            raise ValueError("Test error")
            
        with patch('alpha_pulse.utils.audit_logger.get_audit_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Call should raise exception
            with pytest.raises(ValueError):
                failing_function()
                
            # Verify audit log was created with failure
            mock_logger.log.assert_called_once()
            call_args = mock_logger.log.call_args
            
            assert call_args.kwargs['success'] is False
            assert call_args.kwargs['error_message'] == "Test error"


class TestAuditQueries:
    """Test cases for audit query functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        with patch('alpha_pulse.utils.audit_queries.get_db_session') as mock:
            session = MagicMock()
            mock.return_value = session
            yield session
            
    def test_query_builder_basic(self, mock_session):
        """Test basic query building."""
        builder = AuditQueryBuilder(mock_session)
        
        # Build a query
        builder.time_range(
            datetime.now() - timedelta(days=1),
            datetime.now()
        ).event_types(
            AuditEventType.AUTH_LOGIN,
            AuditEventType.AUTH_FAILED
        ).user("test_user").limit(10)
        
        # Verify query methods were called
        query = builder.query
        query.filter.assert_called()
        query.limit.assert_called_with(10)
        
    def test_query_builder_severity_filter(self, mock_session):
        """Test severity filtering."""
        builder = AuditQueryBuilder(mock_session)
        
        builder.severity(AuditSeverity.WARNING)
        
        # Should filter for WARNING and above
        query = builder.query
        query.filter.assert_called()
        
    def test_reporter_security_summary(self, mock_session):
        """Test security summary report generation."""
        # Mock query results
        mock_logs = [
            Mock(
                event_type=AuditEventType.AUTH_LOGIN.value,
                timestamp=datetime.now(),
                user_id="user1",
                ip_address="192.168.1.1",
                success=True
            ),
            Mock(
                event_type=AuditEventType.AUTH_FAILED.value,
                timestamp=datetime.now(),
                user_id="user2",
                ip_address="192.168.1.2",
                success=False
            )
        ]
        
        mock_session.query.return_value.filter.return_value.all.return_value = mock_logs
        
        reporter = AuditReporter(mock_session)
        summary = reporter.security_summary(
            datetime.now() - timedelta(days=1),
            datetime.now()
        )
        
        assert 'authentication' in summary
        assert 'period' in summary
        
    def test_anomaly_detection(self, mock_session):
        """Test anomaly detection."""
        # Mock data with anomaly
        normal_count = 5
        anomaly_count = 50
        
        # Create mock failed login data
        mock_logs = []
        base_time = datetime.now(timezone.utc)
        
        # Normal hours
        for hour in range(20):
            for i in range(normal_count):
                mock_logs.append(Mock(
                    event_type=AuditEventType.AUTH_FAILED.value,
                    timestamp=base_time - timedelta(hours=hour)
                ))
                
        # Anomalous hour
        for i in range(anomaly_count):
            mock_logs.append(Mock(
                event_type=AuditEventType.AUTH_FAILED.value,
                timestamp=base_time - timedelta(hours=21)
            ))
            
        mock_session.query.return_value.filter.return_value.all.return_value = mock_logs
        
        reporter = AuditReporter(mock_session)
        anomalies = reporter.detect_anomalies(lookback_days=1, threshold_multiplier=2.0)
        
        # Should detect the anomaly
        assert len(anomalies) > 0
        assert any(a['type'] == 'excessive_failed_logins' for a in anomalies)
        
    def test_audit_summary_generation(self, mock_session):
        """Test human-readable summary generation."""
        # Mock various report data
        with patch.object(AuditReporter, 'security_summary') as mock_security:
            mock_security.return_value = {
                'authentication': {
                    'successful_logins': 100,
                    'failed_logins': 5,
                    'unique_users': 20
                },
                'secret_access_count': 50
            }
            
            with patch.object(AuditReporter, 'trading_activity') as mock_trading:
                mock_trading.return_value = {
                    'summary': {
                        'total_decisions': 200,
                        'total_executions': 180,
                        'total_failures': 20,
                        'success_rate': 0.9
                    }
                }
                
                with patch.object(AuditReporter, 'risk_events') as mock_risk:
                    mock_risk.return_value = {
                        'summary': {
                            'total_events': 10,
                            'total_overrides': 2
                        }
                    }
                    
                    with patch.object(AuditReporter, 'detect_anomalies') as mock_anomalies:
                        mock_anomalies.return_value = [
                            {'type': 'test_anomaly', 'severity': 'high'}
                        ]
                        
                        summary = generate_audit_summary(days=30)
                        
                        # Verify summary contains expected sections
                        assert "Audit Summary" in summary
                        assert "Security Summary" in summary
                        assert "Trading Activity" in summary
                        assert "Risk Events" in summary
                        assert "Detected Anomalies" in summary
                        assert "100" in summary  # successful logins
                        assert "90.0%" in summary  # success rate