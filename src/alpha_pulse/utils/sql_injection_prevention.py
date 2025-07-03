"""
SQL injection prevention utilities for AlphaPulse.

Provides comprehensive protection against SQL injection attacks:
- Query analysis and validation
- Parameter binding enforcement
- Raw SQL monitoring and blocking
- ORM query validation
"""

import re
import ast
import inspect
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from contextlib import contextmanager

from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.sql import ClauseElement
from sqlalchemy.orm import Query

from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class SQLInjectionError(Exception):
    """Exception raised when SQL injection is detected."""
    pass


class SQLSecurityViolation(Exception):
    """Exception raised when SQL security rules are violated."""
    pass


class SQLInjectionPrevention:
    """Comprehensive SQL injection prevention system."""
    
    def __init__(self):
        """Initialize SQL injection prevention."""
        self.audit_logger = get_audit_logger()
        self.blocked_patterns = self._initialize_blocked_patterns()
        self.allowed_raw_sql_functions = set()
        self.strict_mode = True
        
        # Statistics
        self.stats = {
            'queries_analyzed': 0,
            'injections_blocked': 0,
            'raw_queries_blocked': 0,
            'parameter_validations': 0
        }
    
    def _initialize_blocked_patterns(self) -> List[re.Pattern]:
        """Initialize SQL injection detection patterns."""
        patterns = [
            # Basic SQL injection patterns
            re.compile(r'\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b', re.IGNORECASE),
            
            # Comment-based injections
            re.compile(r'(--|#|/\*|\*/)', re.IGNORECASE),
            
            # Quote escaping attempts
            re.compile(r"(\\')|(\\\")|('')|(\"\")", re.IGNORECASE),
            
            # Stacked queries
            re.compile(r';\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)', re.IGNORECASE),
            
            # Boolean-based blind SQL injection
            re.compile(r'\b(AND|OR)\b\s+\d+\s*[=<>]\s*\d+', re.IGNORECASE),
            re.compile(r'\b(AND|OR)\b\s+[\'"][\w\s]*[\'"]', re.IGNORECASE),
            
            # Time-based blind SQL injection
            re.compile(r'\b(SLEEP|DELAY|WAITFOR|BENCHMARK)\s*\(', re.IGNORECASE),
            
            # Error-based SQL injection
            re.compile(r'\b(CAST|CONVERT|EXTRACT)\s*\(.*\bAS\b', re.IGNORECASE),
            
            # Information schema attacks
            re.compile(r'\b(INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS|MSysObjects)\b', re.IGNORECASE),
            
            # System function attacks
            re.compile(r'\b(xp_|sp_|fn_)\w+', re.IGNORECASE),
            
            # Database-specific patterns
            re.compile(r'\b(@@version|@@servername|user\(\)|database\(\)|version\(\))', re.IGNORECASE),
            
            # File operations
            re.compile(r'\b(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b', re.IGNORECASE),
            
            # Privilege escalation
            re.compile(r'\b(GRANT|REVOKE|ALTER\s+USER|CREATE\s+USER)\b', re.IGNORECASE),
        ]
        
        return patterns
    
    def analyze_query(self, query: str, parameters: Dict[str, Any] = None) -> bool:
        """
        Analyze SQL query for injection attempts.
        
        Returns True if query is safe, raises SQLInjectionError if malicious.
        """
        self.stats['queries_analyzed'] += 1
        
        # Check for SQL injection patterns
        for pattern in self.blocked_patterns:
            if pattern.search(query):
                self.stats['injections_blocked'] += 1
                violation_data = {
                    'query': query[:500],  # Truncate for logging
                    'parameters': str(parameters)[:200] if parameters else None,
                    'pattern_matched': pattern.pattern,
                    'detection_method': 'pattern_matching'
                }
                
                self._log_sql_injection_attempt(violation_data)
                raise SQLInjectionError(f"Potential SQL injection detected in query")
        
        # Validate parameters
        if parameters:
            self._validate_parameters(parameters)
        
        return True
    
    def _validate_parameters(self, parameters: Dict[str, Any]):
        """Validate query parameters for injection attempts."""
        self.stats['parameter_validations'] += 1
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # Check for SQL injection in parameter values
                for pattern in self.blocked_patterns:
                    if pattern.search(value):
                        violation_data = {
                            'parameter_name': key,
                            'parameter_value': value[:200],
                            'pattern_matched': pattern.pattern,
                            'detection_method': 'parameter_validation'
                        }
                        
                        self._log_sql_injection_attempt(violation_data)
                        raise SQLInjectionError(f"Potential SQL injection in parameter '{key}'")
    
    def validate_raw_sql(self, sql: str, allow_unsafe: bool = False) -> bool:
        """Validate raw SQL queries with strict security checks."""
        if self.strict_mode and not allow_unsafe:
            # In strict mode, only allow specific pre-approved queries
            caller_function = self._get_caller_function()
            if caller_function not in self.allowed_raw_sql_functions:
                self.stats['raw_queries_blocked'] += 1
                self._log_raw_sql_attempt(sql, caller_function)
                raise SQLSecurityViolation(
                    f"Raw SQL not allowed from function '{caller_function}'. "
                    f"Use parameterized queries instead."
                )
        
        # Analyze the SQL for injection patterns
        return self.analyze_query(sql)
    
    def _get_caller_function(self) -> str:
        """Get the name of the calling function."""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller
            caller_frame = frame.f_back.f_back.f_back
            return caller_frame.f_code.co_name
        finally:
            del frame
    
    def _log_sql_injection_attempt(self, violation_data: Dict[str, Any]):
        """Log SQL injection attempts for security monitoring."""
        self.audit_logger.log(
            event_type=AuditEventType.SECURITY_VIOLATION,
            event_data={
                'violation_type': 'sql_injection_attempt',
                **violation_data
            },
            severity=AuditSeverity.CRITICAL,
            data_classification="restricted"
        )
    
    def _log_raw_sql_attempt(self, sql: str, caller_function: str):
        """Log unauthorized raw SQL attempts."""
        self.audit_logger.log(
            event_type=AuditEventType.SECURITY_VIOLATION,
            event_data={
                'violation_type': 'unauthorized_raw_sql',
                'sql_query': sql[:500],
                'caller_function': caller_function
            },
            severity=AuditSeverity.WARNING,
            data_classification="internal"
        )
    
    def register_safe_function(self, function_name: str):
        """Register a function as safe for raw SQL usage."""
        self.allowed_raw_sql_functions.add(function_name)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get SQL injection prevention statistics."""
        return self.stats.copy()


# Global SQL injection prevention instance
sql_guard = SQLInjectionPrevention()


class SafeQuery:
    """Wrapper for safe SQL query execution."""
    
    def __init__(self, query: Union[str, ClauseElement], parameters: Dict[str, Any] = None):
        """Initialize safe query with validation."""
        self.original_query = query
        self.parameters = parameters or {}
        
        # Validate the query
        if isinstance(query, str):
            sql_guard.analyze_query(query, self.parameters)
        
        self.query = query
    
    def execute(self, connection):
        """Execute the validated query safely."""
        if isinstance(self.query, str):
            return connection.execute(text(self.query), self.parameters)
        else:
            return connection.execute(self.query)


def safe_execute(query: Union[str, ClauseElement], parameters: Dict[str, Any] = None):
    """Decorator for safe SQL execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate query before execution
            if isinstance(query, str):
                sql_guard.analyze_query(query, parameters)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def block_raw_sql(func):
    """Decorator to block raw SQL usage in a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Temporarily enable strict mode
        original_strict = sql_guard.strict_mode
        sql_guard.strict_mode = True
        try:
            return func(*args, **kwargs)
        finally:
            sql_guard.strict_mode = original_strict
    return wrapper


def allow_raw_sql(func):
    """Decorator to allow raw SQL usage in a specific function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Register this function as safe
        sql_guard.register_safe_function(func.__name__)
        return func(*args, **kwargs)
    return wrapper


@contextmanager
def safe_db_context(strict_mode: bool = True):
    """Context manager for safe database operations."""
    original_strict = sql_guard.strict_mode
    sql_guard.strict_mode = strict_mode
    try:
        yield sql_guard
    finally:
        sql_guard.strict_mode = original_strict


class ParameterizedQueryBuilder:
    """Builder for creating safe parameterized queries."""
    
    def __init__(self):
        """Initialize query builder."""
        self.query_parts = []
        self.parameters = {}
        self.parameter_counter = 0
    
    def select(self, columns: List[str], table: str) -> 'ParameterizedQueryBuilder':
        """Add SELECT clause."""
        # Validate column names to prevent injection
        safe_columns = [self._validate_identifier(col) for col in columns]
        safe_table = self._validate_identifier(table)
        
        self.query_parts.append(f"SELECT {', '.join(safe_columns)} FROM {safe_table}")
        return self
    
    def where(self, condition: str, **params) -> 'ParameterizedQueryBuilder':
        """Add WHERE clause with parameters."""
        # Replace parameters with placeholders
        parameterized_condition = condition
        for key, value in params.items():
            param_name = f"param_{self.parameter_counter}"
            self.parameters[param_name] = value
            parameterized_condition = parameterized_condition.replace(f":{key}", f":{param_name}")
            self.parameter_counter += 1
        
        if self.query_parts:
            self.query_parts.append(f"WHERE {parameterized_condition}")
        else:
            raise ValueError("WHERE clause requires a SELECT clause first")
        
        return self
    
    def order_by(self, columns: List[str], direction: str = "ASC") -> 'ParameterizedQueryBuilder':
        """Add ORDER BY clause."""
        safe_columns = [self._validate_identifier(col) for col in columns]
        safe_direction = direction.upper() if direction.upper() in ["ASC", "DESC"] else "ASC"
        
        self.query_parts.append(f"ORDER BY {', '.join(safe_columns)} {safe_direction}")
        return self
    
    def limit(self, count: int, offset: int = 0) -> 'ParameterizedQueryBuilder':
        """Add LIMIT clause."""
        if not isinstance(count, int) or count < 0:
            raise ValueError("LIMIT count must be a non-negative integer")
        
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("LIMIT offset must be a non-negative integer")
        
        if offset > 0:
            self.query_parts.append(f"LIMIT {count} OFFSET {offset}")
        else:
            self.query_parts.append(f"LIMIT {count}")
        
        return self
    
    def build(self) -> SafeQuery:
        """Build the final safe query."""
        if not self.query_parts:
            raise ValueError("Query cannot be empty")
        
        query_string = " ".join(self.query_parts)
        return SafeQuery(query_string, self.parameters)
    
    def _validate_identifier(self, identifier: str) -> str:
        """Validate SQL identifiers (table/column names)."""
        # Only allow alphanumeric characters and underscores
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        
        # Check for SQL keywords
        sql_keywords = {
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'UNION', 'WHERE', 'ORDER', 'GROUP', 'HAVING', 'FROM', 'INTO',
            'VALUES', 'SET', 'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE'
        }
        
        if identifier.upper() in sql_keywords:
            raise ValueError(f"SQL keyword not allowed as identifier: {identifier}")
        
        return identifier


# SQLAlchemy event listeners for automatic protection
@event.listens_for(Engine, "before_execute")
def analyze_sql_before_execute(conn, clauseelement, multiparams, params, execution_options):
    """Automatically analyze all SQL queries before execution."""
    try:
        if isinstance(clauseelement, str):
            # This is raw SQL - validate it
            sql_guard.validate_raw_sql(clauseelement, allow_unsafe=False)
        
        # Validate parameters
        if params:
            sql_guard._validate_parameters(params)
            
    except (SQLInjectionError, SQLSecurityViolation) as e:
        # Log the error and re-raise
        sql_guard.audit_logger.log(
            event_type=AuditEventType.SECURITY_VIOLATION,
            event_data={
                'error': str(e),
                'query': str(clauseelement)[:500],
                'parameters': str(params)[:200] if params else None
            },
            severity=AuditSeverity.CRITICAL
        )
        raise


def create_safe_query_builder() -> ParameterizedQueryBuilder:
    """Create a new safe query builder instance."""
    return ParameterizedQueryBuilder()


# Pre-register some common safe functions
sql_guard.register_safe_function('get_system_health')
sql_guard.register_safe_function('get_database_version')
sql_guard.register_safe_function('analyze_performance_metrics')