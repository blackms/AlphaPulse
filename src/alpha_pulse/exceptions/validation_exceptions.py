"""
Custom exceptions for validation errors in AlphaPulse.

Provides specific exception types for different validation scenarios:
- Input validation errors
- Security validation errors
- Business logic validation errors
- Schema validation errors
"""

from typing import List, Dict, Any, Optional
from fastapi import HTTPException


class BaseValidationError(Exception):
    """Base class for all validation errors."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None, 
        value: Any = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize base validation error.
        
        Args:
            message: Error message
            field: Field name that caused the error
            value: Value that caused the error
            error_code: Specific error code for categorization
        """
        self.message = message
        self.field = field
        self.value = value
        self.error_code = error_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            'message': self.message,
            'field': self.field,
            'value': str(self.value) if self.value is not None else None,
            'error_code': self.error_code,
            'error_type': self.__class__.__name__
        }


class ValidationError(BaseValidationError):
    """General validation error."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None, 
        value: Any = None,
        error_code: str = "VALIDATION_ERROR"
    ):
        super().__init__(message, field, value, error_code)


class FieldValidationError(BaseValidationError):
    """Field-specific validation error."""
    
    def __init__(
        self, 
        field: str, 
        message: str, 
        value: Any = None,
        error_code: str = "FIELD_VALIDATION_ERROR"
    ):
        super().__init__(message, field, value, error_code)


class SecurityValidationError(BaseValidationError):
    """Security-related validation error."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None, 
        value: Any = None,
        violation_type: Optional[str] = None,
        error_code: str = "SECURITY_VALIDATION_ERROR"
    ):
        self.violation_type = violation_type
        super().__init__(message, field, value, error_code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security error to dictionary format."""
        result = super().to_dict()
        result['violation_type'] = self.violation_type
        return result


class SQLInjectionError(SecurityValidationError):
    """SQL injection attempt detected."""
    
    def __init__(
        self, 
        message: str = "Potential SQL injection detected", 
        field: Optional[str] = None, 
        value: Any = None,
        pattern_matched: Optional[str] = None
    ):
        self.pattern_matched = pattern_matched
        super().__init__(
            message, 
            field, 
            value, 
            violation_type="sql_injection",
            error_code="SQL_INJECTION_ERROR"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SQL injection error to dictionary format."""
        result = super().to_dict()
        result['pattern_matched'] = self.pattern_matched
        return result


class XSSValidationError(SecurityValidationError):
    """Cross-site scripting (XSS) attempt detected."""
    
    def __init__(
        self, 
        message: str = "Potential XSS attack detected", 
        field: Optional[str] = None, 
        value: Any = None,
        pattern_matched: Optional[str] = None
    ):
        self.pattern_matched = pattern_matched
        super().__init__(
            message, 
            field, 
            value, 
            violation_type="xss_attack",
            error_code="XSS_VALIDATION_ERROR"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert XSS error to dictionary format."""
        result = super().to_dict()
        result['pattern_matched'] = self.pattern_matched
        return result


class CSRFValidationError(SecurityValidationError):
    """CSRF token validation error."""
    
    def __init__(
        self, 
        message: str = "CSRF token missing or invalid",
        error_code: str = "CSRF_VALIDATION_ERROR"
    ):
        super().__init__(
            message, 
            violation_type="csrf_violation",
            error_code=error_code
        )


class RateLimitValidationError(SecurityValidationError):
    """Rate limit exceeded error."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None,
        error_code: str = "RATE_LIMIT_ERROR"
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        super().__init__(
            message,
            violation_type="rate_limit_exceeded",
            error_code=error_code
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rate limit error to dictionary format."""
        result = super().to_dict()
        result.update({
            'limit': self.limit,
            'window_seconds': self.window_seconds,
            'retry_after': self.retry_after
        })
        return result


class BusinessLogicValidationError(BaseValidationError):
    """Business logic validation error."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None, 
        value: Any = None,
        business_rule: Optional[str] = None,
        error_code: str = "BUSINESS_LOGIC_ERROR"
    ):
        self.business_rule = business_rule
        super().__init__(message, field, value, error_code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert business logic error to dictionary format."""
        result = super().to_dict()
        result['business_rule'] = self.business_rule
        return result


class FinancialValidationError(BusinessLogicValidationError):
    """Financial data validation error."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None, 
        value: Any = None,
        financial_rule: Optional[str] = None,
        error_code: str = "FINANCIAL_VALIDATION_ERROR"
    ):
        super().__init__(
            message, 
            field, 
            value, 
            business_rule=financial_rule,
            error_code=error_code
        )


class PortfolioValidationError(FinancialValidationError):
    """Portfolio-specific validation error."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None, 
        value: Any = None,
        portfolio_rule: Optional[str] = None,
        error_code: str = "PORTFOLIO_VALIDATION_ERROR"
    ):
        super().__init__(
            message, 
            field, 
            value, 
            financial_rule=portfolio_rule,
            error_code=error_code
        )


class RiskManagementValidationError(FinancialValidationError):
    """Risk management validation error."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None, 
        value: Any = None,
        risk_rule: Optional[str] = None,
        current_risk_level: Optional[float] = None,
        max_risk_level: Optional[float] = None,
        error_code: str = "RISK_VALIDATION_ERROR"
    ):
        self.current_risk_level = current_risk_level
        self.max_risk_level = max_risk_level
        super().__init__(
            message, 
            field, 
            value, 
            financial_rule=risk_rule,
            error_code=error_code
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert risk management error to dictionary format."""
        result = super().to_dict()
        result.update({
            'current_risk_level': self.current_risk_level,
            'max_risk_level': self.max_risk_level
        })
        return result


class SchemaValidationError(BaseValidationError):
    """JSON schema validation error."""
    
    def __init__(
        self, 
        message: str, 
        schema_path: Optional[str] = None,
        invalid_value: Any = None,
        error_code: str = "SCHEMA_VALIDATION_ERROR"
    ):
        self.schema_path = schema_path
        super().__init__(message, schema_path, invalid_value, error_code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema error to dictionary format."""
        result = super().to_dict()
        result['schema_path'] = self.schema_path
        return result


class FileValidationError(BaseValidationError):
    """File upload validation error."""
    
    def __init__(
        self, 
        message: str, 
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
        error_code: str = "FILE_VALIDATION_ERROR"
    ):
        self.filename = filename
        self.file_size = file_size
        self.file_type = file_type
        super().__init__(message, error_code=error_code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file error to dictionary format."""
        result = super().to_dict()
        result.update({
            'filename': self.filename,
            'file_size': self.file_size,
            'file_type': self.file_type
        })
        return result


class ValidationErrorCollection:
    """Collection of validation errors."""
    
    def __init__(self, errors: List[BaseValidationError] = None):
        """Initialize error collection."""
        self.errors = errors or []
    
    def add_error(self, error: BaseValidationError):
        """Add error to collection."""
        self.errors.append(error)
    
    def add_field_error(self, field: str, message: str, value: Any = None):
        """Add field validation error."""
        self.add_error(FieldValidationError(field, message, value))
    
    def add_security_error(self, message: str, violation_type: str, field: str = None, value: Any = None):
        """Add security validation error."""
        self.add_error(SecurityValidationError(message, field, value, violation_type))
    
    def add_business_error(self, message: str, business_rule: str, field: str = None, value: Any = None):
        """Add business logic validation error."""
        self.add_error(BusinessLogicValidationError(message, field, value, business_rule))
    
    def has_errors(self) -> bool:
        """Check if collection has any errors."""
        return len(self.errors) > 0
    
    def get_error_count(self) -> int:
        """Get total number of errors."""
        return len(self.errors)
    
    def get_errors_by_type(self, error_type: type) -> List[BaseValidationError]:
        """Get errors of specific type."""
        return [error for error in self.errors if isinstance(error, error_type)]
    
    def get_security_errors(self) -> List[SecurityValidationError]:
        """Get all security-related errors."""
        return self.get_errors_by_type(SecurityValidationError)
    
    def get_field_errors(self) -> Dict[str, List[str]]:
        """Get field errors grouped by field name."""
        field_errors = {}
        for error in self.errors:
            if error.field:
                if error.field not in field_errors:
                    field_errors[error.field] = []
                field_errors[error.field].append(error.message)
        return field_errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error collection to dictionary format."""
        return {
            'error_count': self.get_error_count(),
            'errors': [error.to_dict() for error in self.errors],
            'field_errors': self.get_field_errors(),
            'security_errors': [error.to_dict() for error in self.get_security_errors()]
        }
    
    def to_http_exception(self, status_code: int = 400) -> HTTPException:
        """Convert error collection to HTTP exception."""
        if not self.has_errors():
            raise ValueError("Cannot create HTTP exception without errors")
        
        # Determine appropriate status code based on error types
        if self.get_security_errors():
            status_code = 403  # Forbidden for security violations
        elif any(isinstance(error, RateLimitValidationError) for error in self.errors):
            status_code = 429  # Too Many Requests
        
        return HTTPException(
            status_code=status_code,
            detail=self.to_dict()
        )


def create_validation_error_response(
    errors: List[BaseValidationError], 
    status_code: int = 400
) -> HTTPException:
    """
    Create standardized validation error response.
    
    Args:
        errors: List of validation errors
        status_code: HTTP status code
        
    Returns:
        HTTPException with standardized error format
    """
    error_collection = ValidationErrorCollection(errors)
    return error_collection.to_http_exception(status_code)


def handle_pydantic_validation_error(pydantic_error) -> ValidationErrorCollection:
    """
    Convert Pydantic validation error to our error format.
    
    Args:
        pydantic_error: Pydantic ValidationError
        
    Returns:
        ValidationErrorCollection with converted errors
    """
    error_collection = ValidationErrorCollection()
    
    for error in pydantic_error.errors():
        field_path = '.'.join(str(loc) for loc in error['loc'])
        message = error['msg']
        error_type = error['type']
        
        if error_type.startswith('value_error'):
            # Custom validator error
            error_collection.add_field_error(field_path, message, error.get('input'))
        elif error_type == 'missing':
            # Required field missing
            error_collection.add_field_error(field_path, f"Field '{field_path}' is required")
        elif error_type.startswith('type_error'):
            # Type validation error
            error_collection.add_field_error(field_path, f"Invalid type for field '{field_path}': {message}")
        else:
            # Generic validation error
            error_collection.add_field_error(field_path, message, error.get('input'))
    
    return error_collection


# Exception mapping for different validation scenarios
VALIDATION_EXCEPTION_MAP = {
    'sql_injection': SQLInjectionError,
    'xss_attack': XSSValidationError,
    'csrf_violation': CSRFValidationError,
    'rate_limit': RateLimitValidationError,
    'business_logic': BusinessLogicValidationError,
    'financial_data': FinancialValidationError,
    'portfolio_data': PortfolioValidationError,
    'risk_management': RiskManagementValidationError,
    'schema_validation': SchemaValidationError,
    'file_upload': FileValidationError,
    'field_validation': FieldValidationError,
    'generic': ValidationError
}


def create_validation_exception(
    error_type: str, 
    message: str, 
    **kwargs
) -> BaseValidationError:
    """
    Create appropriate validation exception based on error type.
    
    Args:
        error_type: Type of validation error
        message: Error message
        **kwargs: Additional arguments for specific exception types
        
    Returns:
        Appropriate validation exception instance
    """
    exception_class = VALIDATION_EXCEPTION_MAP.get(error_type, ValidationError)
    return exception_class(message, **kwargs)