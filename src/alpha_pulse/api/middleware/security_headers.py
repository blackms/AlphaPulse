"""
Security headers middleware for AlphaPulse API.

Implements comprehensive security headers and additional protection mechanisms:
- OWASP recommended security headers
- Content Security Policy (CSP)
- Cross-Origin Resource Sharing (CORS) controls
- Additional security enhancements
"""

import time
import hashlib
import secrets
from typing import Dict, List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds comprehensive security headers to all responses.
    
    Features:
    - OWASP Top 10 protection
    - Content Security Policy
    - Clickjacking prevention
    - XSS protection
    - MIME type sniffing prevention
    - Referrer policy control
    """
    
    def __init__(
        self,
        app: ASGIApp,
        hsts_max_age: int = 31536000,  # 1 year
        csp_policy: Optional[str] = None,
        enable_permissions_policy: bool = True,
        enable_coep: bool = True,
        enable_coop: bool = True
    ):
        """
        Initialize security headers middleware.
        
        Args:
            app: ASGI application
            hsts_max_age: HSTS max age in seconds
            csp_policy: Custom Content Security Policy
            enable_permissions_policy: Enable Permissions Policy header
            enable_coep: Enable Cross-Origin-Embedder-Policy
            enable_coop: Enable Cross-Origin-Opener-Policy
        """
        super().__init__(app)
        self.hsts_max_age = hsts_max_age
        self.csp_policy = csp_policy or self._get_default_csp()
        self.enable_permissions_policy = enable_permissions_policy
        self.enable_coep = enable_coep
        self.enable_coop = enable_coop
        self.audit_logger = get_audit_logger()
        
    def _get_default_csp(self) -> str:
        """Get default Content Security Policy."""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "object-src 'none'; "
            "upgrade-insecure-requests"
        )
        
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Apply security headers
        self._add_security_headers(request, response)
        
        # Check for security violations
        await self._check_security_violations(request, response)
        
        return response
        
    def _add_security_headers(self, request: Request, response: Response):
        """Add comprehensive security headers."""
        
        # HTTP Strict Transport Security (HSTS)
        if request.url.scheme == 'https':
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains; preload"
            )
            
        # Content Security Policy
        response.headers["Content-Security-Policy"] = self.csp_policy
        
        # X-Frame-Options (clickjacking prevention)
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options (MIME sniffing prevention)
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # X-XSS-Protection (legacy, but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Server header removal/obfuscation
        response.headers["Server"] = "AlphaPulse"
        
        # Cross-Origin Embedder Policy
        if self.enable_coep:
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
            
        # Cross-Origin Opener Policy
        if self.enable_coop:
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            
        # Permissions Policy (Feature Policy successor)
        if self.enable_permissions_policy:
            response.headers["Permissions-Policy"] = (
                "camera=(), microphone=(), geolocation=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=(), "
                "accelerometer=(), ambient-light-sensor=(), autoplay=()"
            )
            
        # Cache control for sensitive endpoints
        if self._is_sensitive_endpoint(request.url.path):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            
        # Security-related custom headers
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Request ID for correlation
        request_id = request.headers.get("X-Request-ID", self._generate_request_id())
        response.headers["X-Request-ID"] = request_id
        
        # Rate limiting information (if available)
        if hasattr(request.state, 'rate_limit_info'):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info.get('limit', 'unknown'))
            response.headers["X-RateLimit-Remaining"] = str(info.get('remaining', 'unknown'))
            
    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if endpoint contains sensitive data."""
        sensitive_patterns = [
            '/api/v1/auth/',
            '/api/v1/trades/',
            '/api/v1/portfolio/',
            '/api/v1/audit/',
            '/token',
            '/admin'
        ]
        
        return any(pattern in path for pattern in sensitive_patterns)
        
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return secrets.token_hex(16)
        
    async def _check_security_violations(self, request: Request, response: Response):
        """Check for potential security violations."""
        violations = []
        
        # Check for missing security headers in response
        required_headers = [
            "Content-Security-Policy",
            "X-Frame-Options", 
            "X-Content-Type-Options"
        ]
        
        for header in required_headers:
            if header not in response.headers:
                violations.append(f"missing_{header.lower().replace('-', '_')}")
                
        # Check for insecure protocols
        if request.url.scheme != 'https' and not self._is_development():
            violations.append("insecure_protocol")
            
        # Check for suspicious request patterns
        user_agent = request.headers.get("User-Agent", "")
        if self._is_suspicious_user_agent(user_agent):
            violations.append("suspicious_user_agent")
            
        # Check for potential XSS in query parameters
        if self._contains_xss_patterns(str(request.query_params)):
            violations.append("xss_in_query_params")
            
        # Log violations
        if violations:
            await self._log_security_violations(request, violations)
            
    def _is_development(self) -> bool:
        """Check if running in development mode."""
        import os
        return os.getenv("ALPHAPULSE_ENV", "production") == "development"
        
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check for suspicious user agent patterns."""
        suspicious_patterns = [
            'sqlmap', 'nikto', 'scanner', 'exploit', 'hack',
            'vulnerability', 'injection', 'payload'
        ]
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
        
    def _contains_xss_patterns(self, query_string: str) -> bool:
        """Check for XSS patterns in query string."""
        xss_patterns = [
            '<script', 'javascript:', 'onload=', 'onerror=',
            'alert(', 'document.cookie', 'eval('
        ]
        
        query_lower = query_string.lower()
        return any(pattern in query_lower for pattern in xss_patterns)
        
    async def _log_security_violations(self, request: Request, violations: List[str]):
        """Log security violations for monitoring."""
        client_ip = request.client.host if request.client else "unknown"
        
        self.audit_logger.log(
            event_type=AuditEventType.API_ERROR,
            event_data={
                'security_violations': violations,
                'endpoint': request.url.path,
                'method': request.method,
                'user_agent': request.headers.get("User-Agent", ""),
                'ip_address': client_ip,
                'query_params': str(request.query_params)
            },
            severity=AuditSeverity.WARNING,
            data_classification="restricted"
        )


class ContentSecurityPolicyReportMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle CSP violation reports.
    
    Processes CSP violation reports sent by browsers when CSP policies
    are violated, enabling real-time security monitoring.
    """
    
    def __init__(self, app: ASGIApp, report_endpoint: str = "/api/v1/security/csp-report"):
        """Initialize CSP report middleware."""
        super().__init__(app)
        self.report_endpoint = report_endpoint
        self.audit_logger = get_audit_logger()
        
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle CSP violation reports."""
        if request.url.path == self.report_endpoint and request.method == "POST":
            return await self._handle_csp_report(request)
            
        return await call_next(request)
        
    async def _handle_csp_report(self, request: Request) -> Response:
        """Process CSP violation report."""
        try:
            report_data = await request.json()
            
            # Extract violation details
            csp_report = report_data.get("csp-report", {})
            
            violation_data = {
                'blocked_uri': csp_report.get('blocked-uri'),
                'document_uri': csp_report.get('document-uri'),
                'effective_directive': csp_report.get('effective-directive'),
                'original_policy': csp_report.get('original-policy'),
                'referrer': csp_report.get('referrer'),
                'violated_directive': csp_report.get('violated-directive'),
                'source_file': csp_report.get('source-file'),
                'line_number': csp_report.get('line-number'),
                'column_number': csp_report.get('column-number')
            }
            
            # Log the violation
            self.audit_logger.log(
                event_type=AuditEventType.API_ERROR,
                event_data={
                    'csp_violation': True,
                    'violation_details': violation_data,
                    'client_ip': request.client.host if request.client else "unknown",
                    'user_agent': request.headers.get("User-Agent", "")
                },
                severity=AuditSeverity.WARNING,
                data_classification="internal"
            )
            
            # Store for analysis
            await self._store_violation_for_analysis(violation_data)
            
            return Response(status_code=204)  # No content
            
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.API_ERROR,
                event_data={
                    'csp_report_error': str(e),
                    'client_ip': request.client.host if request.client else "unknown"
                },
                severity=AuditSeverity.ERROR
            )
            
            return Response(status_code=400)
            
    async def _store_violation_for_analysis(self, violation_data: Dict):
        """Store CSP violation for further analysis."""
        # This would typically store in a database or send to a monitoring system
        # For now, we can use Redis for temporary storage
        try:
            import redis
            import json
            
            redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            
            violation_entry = {
                'timestamp': time.time(),
                'violation': violation_data
            }
            
            await redis_client.lpush(
                "csp_violations",
                json.dumps(violation_entry)
            )
            
            # Keep only last 1000 violations
            await redis_client.ltrim("csp_violations", 0, 999)
            
        except Exception:
            # Fail silently if Redis is not available
            pass


class SecurityResponseHeadersFilter:
    """
    Additional security response filtering.
    
    Removes sensitive information from error responses and
    applies additional security transformations.
    """
    
    @staticmethod
    def filter_error_response(response: Response) -> Response:
        """Filter sensitive information from error responses."""
        if response.status_code >= 400:
            # Remove server version information
            if "Server" in response.headers:
                response.headers["Server"] = "AlphaPulse"
                
            # Remove any stack trace information from body
            if hasattr(response, 'body'):
                # This would need to be implemented based on response type
                pass
                
            # Add security headers specific to error responses
            response.headers["X-Error-Handled"] = "true"
            response.headers["Cache-Control"] = "no-store"
            
        return response
        
    @staticmethod
    def add_api_security_headers(response: Response, endpoint_type: str) -> Response:
        """Add security headers specific to API endpoints."""
        
        # API-specific headers
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Endpoint-specific security
        if endpoint_type == "trading":
            response.headers["X-Trading-API"] = "true"
            response.headers["Cache-Control"] = "no-store, private"
            
        elif endpoint_type == "auth":
            response.headers["X-Auth-API"] = "true"
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            
        elif endpoint_type == "data":
            response.headers["X-Data-API"] = "true"
            # Allow caching for data endpoints
            response.headers["Cache-Control"] = "public, max-age=60"
            
        return response


def create_security_headers_config() -> Dict:
    """Create default security headers configuration."""
    return {
        "hsts_max_age": 31536000,  # 1 year
        "csp_policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "object-src 'none'"
        ),
        "enable_permissions_policy": True,
        "enable_coep": False,  # May break some integrations
        "enable_coop": True,
        "csp_report_endpoint": "/api/v1/security/csp-report"
    }