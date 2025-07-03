"""
Comprehensive input validation framework for AlphaPulse API.

Provides robust validation mechanisms for all user inputs including:
- Data type validation
- Format validation
- Range validation
- Business logic validation
- Security-focused validation (XSS, SQL injection prevention)
"""

import re
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Callable
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException

from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class ValidationError(Exception):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None, sanitized_value: Any = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.sanitized_value = sanitized_value
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False


class InputValidator:
    """Comprehensive input validation framework."""
    
    def __init__(self):
        """Initialize validator with audit logging."""
        self.audit_logger = get_audit_logger()
        
        # Common regex patterns
        self.patterns = {
            'stock_symbol': re.compile(r'^[A-Z]{1,5}$'),
            'alpha_numeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'alpha_numeric_spaces': re.compile(r'^[a-zA-Z0-9\s\-\'\.]+$'),
            'iso_date': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            'iso_datetime': re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
            'safe_filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'safe_path': re.compile(r'^[a-zA-Z0-9._/-]+$'),
            'hex_color': re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'),
            'password_strong': re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
        }
        
        # XSS patterns to detect
        self.xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<object[^>]*>.*?</object>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<embed[^>]*>', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'data:text/html', re.IGNORECASE)
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)', re.IGNORECASE),
            re.compile(r'(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b)', re.IGNORECASE),
            re.compile(r'(\'|\").*(\;|--)', re.IGNORECASE),
            re.compile(r'(\b(xp_|sp_)\w+\b)', re.IGNORECASE),
            re.compile(r'(\b(INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS)\b)', re.IGNORECASE)
        ]
    
    def validate_string(
        self, 
        value: Any, 
        field_name: str = "field",
        min_length: int = 0,
        max_length: int = 10000,
        pattern: str = None,
        allow_empty: bool = False,
        sanitize_html: bool = True,
        check_xss: bool = True,
        check_sql: bool = True
    ) -> ValidationResult:
        """Validate string input with comprehensive security checks."""
        result = ValidationResult()
        
        # Type check
        if not isinstance(value, str):
            if value is None and allow_empty:
                result.sanitized_value = ""
                return result
            result.add_error(f"{field_name} must be a string")
            return result
        
        original_value = value
        
        # Length validation
        if len(value) < min_length:
            result.add_error(f"{field_name} must be at least {min_length} characters")
        
        if len(value) > max_length:
            result.add_error(f"{field_name} must be at most {max_length} characters")
            
        # Empty check
        if not allow_empty and not value.strip():
            result.add_error(f"{field_name} cannot be empty")
        
        # XSS detection
        if check_xss and self._detect_xss(value):
            result.add_error(f"{field_name} contains potentially malicious content")
            self._log_security_violation("xss_attempt", field_name, value)
        
        # SQL injection detection
        if check_sql and self._detect_sql_injection(value):
            result.add_error(f"{field_name} contains potentially malicious SQL content")
            self._log_security_violation("sql_injection_attempt", field_name, value)
        
        # Pattern validation
        if pattern and pattern in self.patterns:
            if not self.patterns[pattern].match(value):
                result.add_error(f"{field_name} format is invalid")
        
        # Sanitization
        if sanitize_html:
            value = html.escape(value)
        
        result.sanitized_value = value
        return result
    
    def validate_email(self, value: Any, field_name: str = "email") -> ValidationResult:
        """Validate email address."""
        result = ValidationResult()
        
        if not isinstance(value, str):
            result.add_error(f"{field_name} must be a string")
            return result
        
        try:
            validated_email = validate_email(value)
            result.sanitized_value = validated_email.email
        except EmailNotValidError as e:
            result.add_error(f"{field_name} is not a valid email address: {str(e)}")
        
        return result
    
    def validate_phone(self, value: Any, field_name: str = "phone", region: str = "US") -> ValidationResult:
        """Validate phone number in E.164 format."""
        result = ValidationResult()
        
        if not isinstance(value, str):
            result.add_error(f"{field_name} must be a string")
            return result
        
        try:
            parsed_number = phonenumbers.parse(value, region)
            if phonenumbers.is_valid_number(parsed_number):
                result.sanitized_value = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
            else:
                result.add_error(f"{field_name} is not a valid phone number")
        except NumberParseException as e:
            result.add_error(f"{field_name} is not a valid phone number: {str(e)}")
        
        return result
    
    def validate_decimal(
        self, 
        value: Any, 
        field_name: str = "value",
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None,
        max_decimal_places: int = 8
    ) -> ValidationResult:
        """Validate decimal/financial values."""
        result = ValidationResult()
        
        try:
            if isinstance(value, str):
                decimal_value = Decimal(value)
            elif isinstance(value, (int, float)):
                decimal_value = Decimal(str(value))
            else:
                result.add_error(f"{field_name} must be a number")
                return result
            
            # Check decimal places
            if decimal_value.as_tuple().exponent < -max_decimal_places:
                result.add_error(f"{field_name} cannot have more than {max_decimal_places} decimal places")
            
            # Range validation
            if min_value is not None and decimal_value < min_value:
                result.add_error(f"{field_name} must be at least {min_value}")
            
            if max_value is not None and decimal_value > max_value:
                result.add_error(f"{field_name} must be at most {max_value}")
            
            result.sanitized_value = decimal_value
            
        except InvalidOperation:
            result.add_error(f"{field_name} is not a valid number")
        
        return result
    
    def validate_integer(
        self, 
        value: Any, 
        field_name: str = "value",
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> ValidationResult:
        """Validate integer values."""
        result = ValidationResult()
        
        try:
            if isinstance(value, str):
                int_value = int(value)
            elif isinstance(value, float) and value.is_integer():
                int_value = int(value)
            elif isinstance(value, int):
                int_value = value
            else:
                result.add_error(f"{field_name} must be an integer")
                return result
            
            # Range validation
            if min_value is not None and int_value < min_value:
                result.add_error(f"{field_name} must be at least {min_value}")
            
            if max_value is not None and int_value > max_value:
                result.add_error(f"{field_name} must be at most {max_value}")
            
            result.sanitized_value = int_value
            
        except (ValueError, TypeError):
            result.add_error(f"{field_name} is not a valid integer")
        
        return result
    
    def validate_percentage(self, value: Any, field_name: str = "percentage") -> ValidationResult:
        """Validate percentage values (-100 to 100)."""
        return self.validate_decimal(
            value, 
            field_name, 
            min_value=Decimal('-100'), 
            max_value=Decimal('100'),
            max_decimal_places=4
        )
    
    def validate_stock_symbol(self, value: Any, field_name: str = "symbol") -> ValidationResult:
        """Validate stock symbol format."""
        result = self.validate_string(
            value, 
            field_name, 
            min_length=1, 
            max_length=5, 
            pattern='stock_symbol'
        )
        
        if result.is_valid:
            result.sanitized_value = result.sanitized_value.upper()
        
        return result
    
    def validate_datetime(self, value: Any, field_name: str = "datetime") -> ValidationResult:
        """Validate datetime input."""
        result = ValidationResult()
        
        if isinstance(value, datetime):
            result.sanitized_value = value
            return result
        
        if isinstance(value, str):
            try:
                # Try ISO format first
                if 'T' in value:
                    parsed_datetime = datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                    # Try date only
                    parsed_date = datetime.strptime(value, '%Y-%m-%d')
                    parsed_datetime = parsed_date
                
                result.sanitized_value = parsed_datetime
                return result
                
            except ValueError:
                result.add_error(f"{field_name} is not a valid datetime format")
        else:
            result.add_error(f"{field_name} must be a datetime string or datetime object")
        
        return result
    
    def validate_file_upload(
        self, 
        filename: str, 
        content_type: str, 
        file_size: int,
        allowed_extensions: List[str] = None,
        max_size: int = 10 * 1024 * 1024  # 10MB default
    ) -> ValidationResult:
        """Validate file upload parameters."""
        result = ValidationResult()
        
        # Filename validation
        if not self.patterns['safe_filename'].match(filename):
            result.add_error("Filename contains invalid characters")
        
        # Extension validation
        if allowed_extensions:
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if file_ext not in allowed_extensions:
                result.add_error(f"File extension .{file_ext} is not allowed")
        
        # Size validation
        if file_size > max_size:
            result.add_error(f"File size {file_size} exceeds maximum allowed size {max_size}")
        
        # Content type validation
        dangerous_types = [
            'application/x-executable',
            'application/x-msdownload',
            'application/x-msdos-program',
            'text/javascript'
        ]
        
        if content_type in dangerous_types:
            result.add_error("File type is not allowed for security reasons")
        
        return result
    
    def validate_password(self, value: Any, field_name: str = "password") -> ValidationResult:
        """Validate password strength."""
        result = ValidationResult()
        
        if not isinstance(value, str):
            result.add_error(f"{field_name} must be a string")
            return result
        
        # Length check
        if len(value) < 8:
            result.add_error(f"{field_name} must be at least 8 characters long")
        
        if len(value) > 128:
            result.add_error(f"{field_name} must be at most 128 characters long")
        
        # Complexity checks
        if not re.search(r'[a-z]', value):
            result.add_error(f"{field_name} must contain at least one lowercase letter")
        
        if not re.search(r'[A-Z]', value):
            result.add_error(f"{field_name} must contain at least one uppercase letter")
        
        if not re.search(r'\d', value):
            result.add_error(f"{field_name} must contain at least one digit")
        
        if not re.search(r'[@$!%*?&]', value):
            result.add_error(f"{field_name} must contain at least one special character (@$!%*?&)")
        
        # Common password checks
        common_passwords = ['password', '123456', 'qwerty', 'abc123', 'password123']
        if value.lower() in common_passwords:
            result.add_error(f"{field_name} is too common")
        
        result.sanitized_value = value
        return result
    
    def validate_pagination(
        self, 
        page: Any = 1, 
        page_size: Any = 20,
        max_page_size: int = 100
    ) -> ValidationResult:
        """Validate pagination parameters."""
        result = ValidationResult()
        
        # Validate page
        page_result = self.validate_integer(page, "page", min_value=1)
        if not page_result.is_valid:
            result.errors.extend(page_result.errors)
        
        # Validate page size
        size_result = self.validate_integer(page_size, "page_size", min_value=1, max_value=max_page_size)
        if not size_result.is_valid:
            result.errors.extend(size_result.errors)
        
        if result.is_valid:
            result.sanitized_value = {
                'page': page_result.sanitized_value,
                'page_size': size_result.sanitized_value
            }
        
        return result
    
    def _detect_xss(self, value: str) -> bool:
        """Detect potential XSS attacks."""
        for pattern in self.xss_patterns:
            if pattern.search(value):
                return True
        return False
    
    def _detect_sql_injection(self, value: str) -> bool:
        """Detect potential SQL injection attacks."""
        for pattern in self.sql_patterns:
            if pattern.search(value):
                return True
        return False
    
    def _log_security_violation(self, violation_type: str, field_name: str, value: str):
        """Log security violations for monitoring."""
        self.audit_logger.log(
            event_type=AuditEventType.SECURITY_VIOLATION,
            event_data={
                'violation_type': violation_type,
                'field_name': field_name,
                'attempted_value': value[:100],  # Truncate for logging
                'detection_method': 'input_validation'
            },
            severity=AuditSeverity.WARNING,
            data_classification="restricted"
        )
    
    def sanitize_url(self, url: str) -> str:
        """Sanitize URL to prevent various attacks."""
        # URL encode the URL
        return urllib.parse.quote(url, safe=':/?#[]@!$&\'()*+,;=')
    
    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        return html.escape(html_content)
    
    def validate_json_schema(self, data: Dict, schema: Dict) -> ValidationResult:
        """Validate data against JSON schema."""
        result = ValidationResult()
        
        try:
            import jsonschema
            jsonschema.validate(data, schema)
            result.sanitized_value = data
        except ImportError:
            result.add_error("JSON schema validation requires jsonschema package")
        except jsonschema.ValidationError as e:
            result.add_error(f"Schema validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            result.add_error(f"Invalid schema: {e.message}")
        
        return result


# Global validator instance
validator = InputValidator()


def validate_input(validation_func: Callable) -> Callable:
    """Decorator for input validation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = validation_func(*args, **kwargs)
            if not result.is_valid:
                raise ValidationError("; ".join(result.errors))
            return func(*args, **kwargs)
        return wrapper
    return decorator