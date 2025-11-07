"""
Quota enforcement middleware for FastAPI.

Intercepts requests to enforce cache quota limits before processing.
"""

import time
import logging
from typing import Callable, Optional, Awaitable, List
from uuid import UUID
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from redis.asyncio import Redis

from alpha_pulse.models.quota import QuotaDecision
from alpha_pulse.services.quota_checker import QuotaChecker
from alpha_pulse.services.quota_cache_service import QuotaCacheService
from alpha_pulse.services.usage_tracker import UsageTracker
from alpha_pulse.middleware.quota_metrics import (
    quota_checks_total,
    quota_rejections_total,
    quota_warnings_total,
    quota_check_latency_ms,
    quota_enforcement_enabled,
    quota_excluded_paths_total,
)

logger = logging.getLogger(__name__)


class QuotaEnforcementMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for quota enforcement.

    Enforces cache quota limits on incoming requests with:
    - Two-tier caching (Redis -> PostgreSQL)
    - Atomic usage tracking
    - Three-level decision logic (ALLOW/WARN/REJECT)
    - Prometheus metrics
    - Feature flag control

    Configuration:
        enabled: Master feature flag (default: True)
        cache_ttl_seconds: Redis cache TTL (default: 300)
        exclude_paths: Paths to skip quota check (default: [])
        redis_client: Redis client instance
        db_session_factory: Database session factory
    """

    def __init__(
        self,
        app,
        enabled: bool = True,
        cache_ttl_seconds: int = 300,
        exclude_paths: Optional[List[str]] = None,
        redis_client: Optional[Redis] = None,
        db_session_factory: Optional[Callable[[], Awaitable]] = None
    ):
        """
        Initialize quota enforcement middleware.

        Args:
            app: FastAPI application
            enabled: Master feature flag
            cache_ttl_seconds: Redis cache TTL
            exclude_paths: Paths to skip quota check
            redis_client: Redis client instance
            db_session_factory: Database session factory
        """
        super().__init__(app)
        self.enabled = enabled
        self.exclude_paths = exclude_paths or []

        # Initialize services
        if redis_client and db_session_factory:
            self.cache_service = QuotaCacheService(
                redis_client=redis_client,
                db_session_factory=db_session_factory,
                cache_ttl_seconds=cache_ttl_seconds
            )
            self.usage_tracker = UsageTracker(redis_client=redis_client)
            self.quota_checker = QuotaChecker(
                cache_service=self.cache_service,
                usage_tracker=self.usage_tracker
            )
        else:
            logger.warning("Quota enforcement disabled: no Redis/DB clients provided")
            self.enabled = False

        # Update metrics
        quota_enforcement_enabled.set(1 if self.enabled else 0)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process request with quota enforcement.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response (200, 429, or error response)
        """
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip excluded paths
        if self._is_excluded_path(request.url.path):
            quota_excluded_paths_total.labels(path=request.url.path).inc()
            return await call_next(request)

        # Extract tenant context
        tenant_id = self._get_tenant_id(request)
        if tenant_id is None:
            # No tenant context - skip quota check
            logger.debug(
                f"Quota check skipped (no tenant): path={request.url.path}"
            )
            return await call_next(request)

        # Extract write size from request
        write_size_mb = await self._extract_write_size(request)
        if write_size_mb is None or write_size_mb == 0:
            # Not a cache write - skip quota check
            return await call_next(request)

        # Perform quota check
        start_time = time.perf_counter()

        try:
            result = await self.quota_checker.check_quota(
                tenant_id=tenant_id,
                write_size_mb=write_size_mb
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record metrics
            quota_checks_total.labels(
                tenant_id=str(tenant_id),
                decision=result.decision.value
            ).inc()

            quota_check_latency_ms.labels(
                operation="quota_check"
            ).observe(latency_ms)

            if result.decision == QuotaDecision.REJECT:
                quota_rejections_total.labels(tenant_id=str(tenant_id)).inc()
            elif result.decision == QuotaDecision.WARN:
                quota_warnings_total.labels(tenant_id=str(tenant_id)).inc()

            logger.info(
                f"Quota check completed: tenant_id={tenant_id}, decision={result.decision.value}, "
                f"write_size_mb={write_size_mb}, latency_ms={latency_ms:.2f}"
            )

            # Handle REJECT decision
            if result.decision == QuotaDecision.REJECT:
                return self._create_429_response(result)

            # Process request (ALLOW or WARN)
            response = await call_next(request)

            # Add quota headers to response
            self._add_quota_headers(response, result)

            return response

        except Exception as e:
            logger.error(
                f"Quota check failed: tenant_id={tenant_id}, error={e}"
            )

            # On error, allow request (fail open)
            return await call_next(request)

    def _is_excluded_path(self, path: str) -> bool:
        """
        Check if path is excluded from quota enforcement.

        Args:
            path: Request path

        Returns:
            True if path should be excluded
        """
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    def _get_tenant_id(self, request: Request) -> Optional[UUID]:
        """
        Extract tenant ID from request context.

        Expects TenantContextMiddleware to have set tenant_id in request.state.

        Args:
            request: HTTP request

        Returns:
            Tenant UUID or None
        """
        # Check request state (set by TenantContextMiddleware)
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            return tenant_id

        # Fallback: check X-Tenant-ID header (for testing)
        tenant_header = request.headers.get("X-Tenant-ID")
        if tenant_header:
            try:
                return UUID(tenant_header)
            except ValueError:
                logger.warning(
                    f"Invalid tenant ID header: value={tenant_header}"
                )

        return None

    async def _extract_write_size(self, request: Request) -> Optional[float]:
        """
        Extract write size from request body.

        Expects cache write requests to have 'size_mb' field in JSON body.

        Args:
            request: HTTP request

        Returns:
            Write size in MB, or None if not a cache write
        """
        # Only check POST requests to cache endpoints
        if request.method != "POST":
            return None

        if not request.url.path.startswith("/cache/"):
            return None

        try:
            # Read body (must be careful not to consume it)
            body = await request.body()

            # Parse JSON
            import json
            data = json.loads(body)

            # Extract size_mb field
            size_mb = data.get("size_mb")
            if size_mb is not None:
                return float(size_mb)

        except Exception as e:
            logger.warning(
                f"Write size extraction failed: error={e}"
            )

        return None

    def _create_429_response(self, result) -> JSONResponse:
        """
        Create 429 Too Many Requests response.

        Args:
            result: QuotaCheckResult with rejection details

        Returns:
            JSONResponse with 429 status
        """
        headers = result.to_headers()

        return JSONResponse(
            status_code=429,
            content={
                "error": "quota_exceeded",
                "message": result.message,
                "quota_mb": result.quota_config.quota_mb,
                "current_usage_mb": result.quota_config.current_usage_mb,
                "requested_mb": result.requested_mb
            },
            headers=headers
        )

    def _add_quota_headers(self, response: Response, result) -> None:
        """
        Add quota headers to response.

        Args:
            response: HTTP response
            result: QuotaCheckResult
        """
        headers = result.to_headers()
        for key, value in headers.items():
            response.headers[key] = value
