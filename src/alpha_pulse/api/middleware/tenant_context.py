"""
Tenant Context Middleware for Multi-Tenant SaaS

This middleware:
1. Extracts tenant_id from JWT token
2. Sets PostgreSQL session variable for RLS
3. Stores tenant context in request state
4. Respects RLS_ENABLED feature flag

EPIC-001: Database Multi-Tenancy
User Story: US-005 (5 SP)
"""
import logging
from typing import Callable
from fastapi import Request, Response, HTTPException
from jose import jwt, JWTError
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract tenant_id from JWT and set PostgreSQL RLS context.
    """

    # Paths that don't require tenant context
    EXEMPT_PATHS = {
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/token",  # Login endpoint must be public
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and inject tenant context.
        """
        # Skip tenant context for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        try:
            # Extract and validate JWT
            tenant_id, user_id = await self._extract_tenant_from_jwt(request)

            # Store in request state (available to all route handlers)
            request.state.tenant_id = tenant_id
            request.state.user_id = user_id

            # Set PostgreSQL session variable (if RLS enabled)
            if request.app.state.settings.RLS_ENABLED:
                await self._set_postgres_tenant_context(request, tenant_id)

            # Log tenant context (debug only)
            logger.debug(
                f"Tenant context set: tenant_id={tenant_id}, "
                f"user_id={user_id}, "
                f"path={request.url.path}"
            )

            # Continue processing request
            response = await call_next(request)
            return response

        except HTTPException:
            # Re-raise HTTP exceptions (already formatted)
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Tenant context middleware error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error processing tenant context"
            )

    async def _extract_tenant_from_jwt(self, request: Request) -> tuple[str, str]:
        """
        Extract tenant_id and user_id from JWT token.

        Returns:
            tuple[str, str]: (tenant_id, user_id)

        Raises:
            HTTPException: If token is missing, invalid, or missing tenant_id
        """
        # Get Authorization header
        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization header"
            )

        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format (expected: Bearer <token>)"
            )

        # Extract token
        token = auth_header.replace("Bearer ", "")

        try:
            # Decode and validate JWT
            payload = jwt.decode(
                token,
                request.app.state.settings.JWT_SECRET,
                algorithms=[request.app.state.settings.JWT_ALGORITHM]
            )

            # Extract claims
            tenant_id = payload.get("tenant_id")
            user_id = payload.get("sub")  # Standard JWT "subject" claim

            if not tenant_id:
                raise HTTPException(
                    status_code=401,
                    detail="Missing tenant_id claim in JWT token"
                )

            if not user_id:
                raise HTTPException(
                    status_code=401,
                    detail="Missing sub (user_id) claim in JWT token"
                )

            return tenant_id, user_id

        except JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid JWT token: {str(e)}"
            )

    async def _set_postgres_tenant_context(self, request: Request, tenant_id: str) -> None:
        """
        Set PostgreSQL session variable for RLS.

        Args:
            request: FastAPI request object
            tenant_id: Tenant UUID

        Raises:
            HTTPException: If database operation fails
        """
        try:
            # Acquire database connection from pool
            async with request.app.state.db_pool.acquire() as conn:
                # Set session variable (LOCAL = transaction-scoped)
                await conn.execute(
                    "SET LOCAL app.current_tenant_id = $1",
                    tenant_id
                )

            logger.debug(f"PostgreSQL tenant context set: {tenant_id}")

        except Exception as e:
            logger.error(f"Failed to set PostgreSQL tenant context: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to set database tenant context"
            )


# Dependency injection for route handlers
async def get_current_tenant_id(request: Request) -> str:
    """
    Get tenant_id from request state.

    Usage in route handlers:
        @app.get("/api/users")
        async def get_users(tenant_id: str = Depends(get_current_tenant_id)):
            # tenant_id is automatically injected
            pass
    """
    if not hasattr(request.state, "tenant_id"):
        raise HTTPException(
            status_code=500,
            detail="Tenant context not set (middleware not configured?)"
        )

    return request.state.tenant_id


async def get_current_user_id(request: Request) -> str:
    """
    Get user_id from request state.

    Usage in route handlers:
        @app.get("/api/profile")
        async def get_profile(user_id: str = Depends(get_current_user_id)):
            # user_id is automatically injected
            pass
    """
    if not hasattr(request.state, "user_id"):
        raise HTTPException(
            status_code=500,
            detail="User context not set (middleware not configured?)"
        )

    return request.state.user_id


def verify_tenant_access(resource_tenant_id: str, request_tenant_id: str) -> None:
    """
    Verify that the current tenant has access to a resource.

    Args:
        resource_tenant_id: Tenant ID of the resource
        request_tenant_id: Tenant ID from request context

    Raises:
        HTTPException: If tenant IDs don't match (403 Forbidden)
    """
    if resource_tenant_id != request_tenant_id:
        logger.warning(
            f"Tenant access violation: "
            f"tenant {request_tenant_id} tried to access "
            f"resource owned by tenant {resource_tenant_id}"
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied: resource belongs to another tenant"
        )
