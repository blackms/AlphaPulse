"""
Credentials Management API Routes.

This module provides REST API endpoints for managing exchange credentials
with secure Vault storage, caching, and audit logging.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import uuid

from alpha_pulse.api.dependencies import get_current_user, get_current_tenant_id
from alpha_pulse.utils.secrets_manager import create_secrets_manager
from alpha_pulse.cache.cache_invalidation import invalidate_pattern
from alpha_pulse.exchanges.credentials.manager import credentials_manager

router = APIRouter(prefix="/credentials", tags=["credentials"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CredentialBase(BaseModel):
    """Base model for credential data."""
    exchange: str = Field(..., description="Exchange name (binance, bybit, etc.)")
    api_key: str = Field(..., description="Exchange API key", min_length=10)
    api_secret: str = Field(..., description="Exchange API secret", min_length=10)
    testnet: bool = Field(default=False, description="Use testnet environment")
    passphrase: Optional[str] = Field(None, description="Optional passphrase (for OKX, etc.)")

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        """Validate exchange name."""
        allowed_exchanges = ["binance", "bybit", "coinbase", "kraken", "okx", "huobi"]
        if v.lower() not in allowed_exchanges:
            raise ValueError(f"Exchange must be one of: {', '.join(allowed_exchanges)}")
        return v.lower()


class CredentialCreate(CredentialBase):
    """Model for creating new credentials."""
    pass


class CredentialUpdate(BaseModel):
    """Model for updating credentials (rotation)."""
    api_key: str = Field(..., description="New API key", min_length=10)
    api_secret: str = Field(..., description="New API secret", min_length=10)
    passphrase: Optional[str] = Field(None, description="New passphrase (if applicable)")


class CredentialResponse(BaseModel):
    """Model for credential response (without sensitive data)."""
    exchange: str
    testnet: bool
    created_at: datetime
    updated_at: datetime
    has_passphrase: bool
    api_key_prefix: str = Field(..., description="First 8 characters of API key")


class CredentialRotationResponse(BaseModel):
    """Model for credential rotation response."""
    exchange: str
    rotated_at: datetime
    old_credentials_valid_until: datetime
    api_key_prefix: str
    message: str


class CredentialListResponse(BaseModel):
    """Model for list of credentials."""
    credentials: List[CredentialResponse]
    total: int


# ============================================================================
# Helper Functions
# ============================================================================

def get_vault_path(tenant_id: str, exchange: str) -> str:
    """
    Generate Vault path for tenant credentials.

    Args:
        tenant_id: Tenant UUID
        exchange: Exchange name

    Returns:
        Vault KV path (e.g., tenants/{tenant_id}/{exchange}/api_key)
    """
    return f"tenants/{tenant_id}/{exchange}/api_key"


async def invalidate_credential_cache(tenant_id: str, exchange: str) -> None:
    """
    Invalidate credential cache for a tenant/exchange combination.

    Args:
        tenant_id: Tenant UUID
        exchange: Exchange name
    """
    try:
        # Invalidate cachetools LRU cache (in-memory)
        credentials_manager._credentials.pop(exchange.lower(), None)

        # Invalidate Redis cache (distributed)
        cache_key = f"credentials:{tenant_id}:{exchange}"
        await invalidate_pattern(cache_key)

        logger.info(
            f"[Tenant: {tenant_id}] Invalidated credential cache for {exchange}"
        )
    except Exception as e:
        logger.error(
            f"[Tenant: {tenant_id}] Failed to invalidate cache for {exchange}: {e}"
        )


def mask_api_key(api_key: str) -> str:
    """
    Mask API key for safe logging/display.

    Args:
        api_key: Full API key

    Returns:
        Masked key (first 8 chars + ***)
    """
    return f"{api_key[:8]}***" if len(api_key) > 8 else "***"


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/",
    response_model=CredentialResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create exchange credentials",
    description="Store new exchange API credentials in Vault for the authenticated tenant"
)
async def create_credentials(
    credentials: CredentialCreate,
    user: Dict[str, Any] = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant_id)
) -> CredentialResponse:
    """
    Create new exchange credentials for tenant.

    **Authentication Required**: JWT token with valid tenant_id

    **Acceptance Criteria**:
    - Credentials stored in Vault at tenants/{tenant_id}/{exchange}/api_key
    - Credentials encrypted at rest (Vault handles this)
    - Audit log entry created
    """
    try:
        logger.info(
            f"[Tenant: {tenant_id}] [User: {user.get('sub')}] "
            f"Creating credentials for {credentials.exchange}"
        )

        # Initialize secrets manager (Vault)
        secrets_mgr = create_secrets_manager()

        # Store credentials in Vault
        vault_path = get_vault_path(tenant_id, credentials.exchange)

        credential_data = {
            "api_key": credentials.api_key,
            "api_secret": credentials.api_secret,
            "exchange": credentials.exchange,
            "testnet": credentials.testnet,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "created_by": user.get("sub"),
            "tenant_id": tenant_id
        }

        if credentials.passphrase:
            credential_data["passphrase"] = credentials.passphrase

        success = secrets_mgr.set_secret(vault_path, credential_data)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store credentials in Vault"
            )

        # Audit log
        logger.info(
            f"[AUDIT] [Tenant: {tenant_id}] [User: {user.get('sub')}] "
            f"Created credentials for {credentials.exchange} "
            f"(API key: {mask_api_key(credentials.api_key)})"
        )

        # Return response (without sensitive data)
        return CredentialResponse(
            exchange=credentials.exchange,
            testnet=credentials.testnet,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            has_passphrase=bool(credentials.passphrase),
            api_key_prefix=mask_api_key(credentials.api_key)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[Tenant: {tenant_id}] Failed to create credentials: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create credentials: {str(e)}"
        )


@router.put(
    "/{exchange}/rotate",
    response_model=CredentialRotationResponse,
    status_code=status.HTTP_200_OK,
    summary="Rotate exchange credentials",
    description="Rotate API credentials for a specific exchange with grace period support"
)
async def rotate_credentials(
    exchange: str,
    new_credentials: CredentialUpdate,
    user: Dict[str, Any] = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant_id)
) -> CredentialRotationResponse:
    """
    Rotate exchange credentials with 5-minute grace period.

    **Story 3.6 Acceptance Criteria**:
    - ✅ PUT /credentials/{exchange}/rotate endpoint updates Vault
    - ✅ Cache invalidated immediately
    - ✅ Old credentials work for 5-min grace period (stored in Vault metadata)
    - ✅ Audit log entry created

    **Grace Period Implementation**:
    Old credentials are stored in Vault with TTL metadata. The CredentialsManager
    checks both current and grace-period credentials during authentication.

    **Authentication Required**: JWT token with valid tenant_id
    """
    try:
        logger.info(
            f"[Tenant: {tenant_id}] [User: {user.get('sub')}] "
            f"Rotating credentials for {exchange}"
        )

        # Initialize secrets manager (Vault)
        secrets_mgr = create_secrets_manager()

        # Get existing credentials for grace period
        vault_path = get_vault_path(tenant_id, exchange)
        old_credentials = secrets_mgr.get_secret(vault_path)

        if not old_credentials:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No existing credentials found for {exchange}"
            )

        # Calculate grace period expiry
        rotation_time = datetime.utcnow()
        grace_period_end = rotation_time + timedelta(minutes=5)

        # Store old credentials with grace period metadata
        grace_path = f"{vault_path}_grace_{rotation_time.strftime('%Y%m%d%H%M%S')}"
        grace_data = {
            "api_key": old_credentials.get("api_key"),
            "api_secret": old_credentials.get("api_secret"),
            "valid_until": grace_period_end.isoformat(),
            "rotated_at": rotation_time.isoformat(),
            "rotated_by": user.get("sub")
        }

        if old_credentials.get("passphrase"):
            grace_data["passphrase"] = old_credentials.get("passphrase")

        secrets_mgr.set_secret(grace_path, grace_data)

        # Store new credentials
        new_credential_data = {
            "api_key": new_credentials.api_key,
            "api_secret": new_credentials.api_secret,
            "exchange": exchange,
            "testnet": old_credentials.get("testnet", False),
            "created_at": old_credentials.get("created_at", rotation_time.isoformat()),
            "updated_at": rotation_time.isoformat(),
            "rotated_at": rotation_time.isoformat(),
            "rotated_by": user.get("sub"),
            "tenant_id": tenant_id,
            "grace_period_path": grace_path,
            "grace_period_until": grace_period_end.isoformat()
        }

        if new_credentials.passphrase:
            new_credential_data["passphrase"] = new_credentials.passphrase

        success = secrets_mgr.set_secret(vault_path, new_credential_data)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to rotate credentials in Vault"
            )

        # Invalidate cache immediately
        await invalidate_credential_cache(tenant_id, exchange)

        # Audit log
        logger.info(
            f"[AUDIT] [Tenant: {tenant_id}] [User: {user.get('sub')}] "
            f"Rotated credentials for {exchange} "
            f"(New API key: {mask_api_key(new_credentials.api_key)}) "
            f"(Grace period until: {grace_period_end.isoformat()})"
        )

        return CredentialRotationResponse(
            exchange=exchange,
            rotated_at=rotation_time,
            old_credentials_valid_until=grace_period_end,
            api_key_prefix=mask_api_key(new_credentials.api_key),
            message=(
                f"Credentials rotated successfully. "
                f"Old credentials will remain valid until {grace_period_end.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                f"(5-minute grace period)"
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[Tenant: {tenant_id}] Failed to rotate credentials for {exchange}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rotate credentials: {str(e)}"
        )


@router.get(
    "/",
    response_model=CredentialListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all credentials",
    description="Get list of all exchange credentials for the authenticated tenant (without sensitive data)"
)
async def list_credentials(
    user: Dict[str, Any] = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant_id)
) -> CredentialListResponse:
    """
    List all credentials for tenant (without sensitive data).

    **Authentication Required**: JWT token with valid tenant_id
    """
    try:
        logger.info(
            f"[Tenant: {tenant_id}] [User: {user.get('sub')}] "
            f"Listing credentials"
        )

        # Initialize secrets manager (Vault)
        secrets_mgr = create_secrets_manager()

        # List all credentials for tenant
        tenant_path = f"tenants/{tenant_id}"
        exchanges = secrets_mgr.list_secrets(tenant_path)

        credentials_list = []
        for exchange in exchanges:
            try:
                vault_path = get_vault_path(tenant_id, exchange)
                cred_data = secrets_mgr.get_secret(vault_path)

                if cred_data:
                    credentials_list.append(CredentialResponse(
                        exchange=exchange,
                        testnet=cred_data.get("testnet", False),
                        created_at=datetime.fromisoformat(cred_data.get("created_at", datetime.utcnow().isoformat())),
                        updated_at=datetime.fromisoformat(cred_data.get("updated_at", datetime.utcnow().isoformat())),
                        has_passphrase=bool(cred_data.get("passphrase")),
                        api_key_prefix=mask_api_key(cred_data.get("api_key", ""))
                    ))
            except Exception as e:
                logger.warning(
                    f"[Tenant: {tenant_id}] Failed to load credentials for {exchange}: {e}"
                )

        logger.info(
            f"[Tenant: {tenant_id}] Listed {len(credentials_list)} credentials"
        )

        return CredentialListResponse(
            credentials=credentials_list,
            total=len(credentials_list)
        )

    except Exception as e:
        logger.error(
            f"[Tenant: {tenant_id}] Failed to list credentials: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list credentials: {str(e)}"
        )


@router.delete(
    "/{exchange}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete exchange credentials",
    description="Delete API credentials for a specific exchange"
)
async def delete_credentials(
    exchange: str,
    user: Dict[str, Any] = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant_id)
) -> None:
    """
    Delete exchange credentials for tenant.

    **Authentication Required**: JWT token with valid tenant_id
    """
    try:
        logger.info(
            f"[Tenant: {tenant_id}] [User: {user.get('sub')}] "
            f"Deleting credentials for {exchange}"
        )

        # Initialize secrets manager (Vault)
        secrets_mgr = create_secrets_manager()

        # Delete credentials from Vault
        vault_path = get_vault_path(tenant_id, exchange)
        success = secrets_mgr.delete_secret(vault_path)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No credentials found for {exchange}"
            )

        # Invalidate cache
        await invalidate_credential_cache(tenant_id, exchange)

        # Audit log
        logger.info(
            f"[AUDIT] [Tenant: {tenant_id}] [User: {user.get('sub')}] "
            f"Deleted credentials for {exchange}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[Tenant: {tenant_id}] Failed to delete credentials for {exchange}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete credentials: {str(e)}"
        )
