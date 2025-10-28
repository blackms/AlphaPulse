#!/usr/bin/env python3
"""
Minimal API server to test multi-tenant authentication.

This script starts a simple FastAPI server with just the core authentication
and tenant context middleware to verify the multi-tenant integration works.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

# Import authentication
from alpha_pulse.api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    User
)

# Import tenant middleware
from alpha_pulse.api.middleware.tenant_context import TenantContextMiddleware

# Create FastAPI app
app = FastAPI(
    title="AlphaPulse Multi-Tenant Test API",
    description="Minimal API to test multi-tenant authentication",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add tenant context middleware
app.add_middleware(TenantContextMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "alphapulse-test-api"}


@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None
):
    """
    Login endpoint - returns JWT token with tenant_id.

    Test users:
    - admin / admin123!@#
    - trader / trader123!@#
    - viewer / viewer123!@#
    """
    user = authenticate_user(form_data.username, form_data.password, request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token with tenant_id
    access_token = create_access_token(
        data={
            "sub": user.username,
            "tenant_id": user.tenant_id  # Multi-tenant: Include tenant_id in JWT
        },
        expires_delta=timedelta(minutes=30)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "tenant_id": user.tenant_id
        }
    }


@app.get("/me")
async def read_users_me(
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """
    Get current user info.

    This endpoint demonstrates that:
    1. JWT authentication works
    2. Tenant context is extracted from JWT
    3. Request state contains tenant_id (set by middleware)
    """
    # Get tenant_id from request state (set by middleware)
    tenant_id_from_state = getattr(request.state, 'tenant_id', None) if request else None

    return {
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "permissions": current_user.permissions,
        "tenant_id": current_user.tenant_id,
        "tenant_id_from_middleware": tenant_id_from_state,
        "middleware_working": tenant_id_from_state == current_user.tenant_id
    }


@app.get("/protected")
async def protected_endpoint(
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """
    Protected endpoint requiring authentication.

    Demonstrates multi-tenant context propagation.
    """
    tenant_id = getattr(request.state, 'tenant_id', None) if request else None

    return {
        "message": f"Hello {current_user.username}!",
        "tenant_id": tenant_id,
        "role": current_user.role,
        "permissions": current_user.permissions
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting AlphaPulse Multi-Tenant Test API")
    print("📍 Server: http://0.0.0.0:8000")
    print("📖 Docs: http://0.0.0.0:8000/docs")
    print("\n👥 Test Users:")
    print("  - admin / admin123!@#")
    print("  - trader / trader123!@#")
    print("  - viewer / viewer123!@#")
    print("\n🧪 Test the multi-tenant integration:")
    print("  1. POST /token with username/password")
    print("  2. GET /me with Bearer token")
    print("  3. Check that tenant_id is in JWT and middleware extracts it")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
