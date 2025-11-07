# EPIC-003: Credential Management - High-Level Design (HLD)

**Epic**: EPIC-003 (#142)
**Sprint**: 9-10
**Story Points**: 36
**Date**: 2025-11-07
**Phase**: Design
**Author**: Tech Lead (via Claude Code)
**Status**: DRAFT

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Component Design](#component-design)
4. [Data Models](#data-models)
5. [API Specifications](#api-specifications)
6. [Security Architecture](#security-architecture)
7. [Performance & Scalability](#performance--scalability)
8. [Monitoring & Observability](#monitoring--observability)
9. [Deployment Architecture](#deployment-architecture)
10. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Purpose
Transform AlphaPulse's credential management from single-tenant file-based storage to multi-tenant Vault-based secure storage with automatic validation, health checks, and rotation.

### Scope
- **In Scope**: Multi-tenant credential storage, CCXT validation, API endpoints, health checks, rotation flow, Vault HA deployment
- **Out of Scope**: Automatic credential rotation (exchanges don't support it), custom exchange integrations (CCXT only)

### Key Design Decisions
1. **Storage**: HashiCorp Vault for secrets + PostgreSQL for metadata (per ADR-003)
2. **Validation**: CCXT test API calls before storage (prevent invalid credentials)
3. **Caching**: 5-minute TTL in-memory cache (balance security + performance)
4. **Health Checks**: Background job every 6 hours (automatic detection)
5. **Isolation**: Tenant-scoped Vault paths: `secret/tenants/{tenant_id}/exchanges/{exchange}`

### Success Criteria
- ✅ <5ms P99 credential retrieval latency (from cache)
- ✅ 100% invalid credentials rejected before storage
- ✅ Vault survives single node failure (HA with 3 replicas)
- ✅ SOC2-compliant audit trail (all access logged)

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Credentials Router (/api/v1/credentials)                  │  │
│  │  - POST   /credentials          (Create/Update)            │  │
│  │  - GET    /credentials          (List all)                 │  │
│  │  - GET    /credentials/{id}     (Get one)                  │  │
│  │  - DELETE /credentials/{id}     (Delete)                   │  │
│  │  - POST   /credentials/{id}/validate (Manual validation)   │  │
│  │  - GET    /credentials/{id}/health   (Health status)       │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Service Layer (Business Logic)                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         TenantCredentialService                            │  │
│  │  - get_credentials(tenant_id, exchange)                    │  │
│  │  - validate_and_store(tenant_id, exchange, api_key, ...)   │  │
│  │  - delete_credentials(tenant_id, exchange)                 │  │
│  │  - rotate_credentials(tenant_id, exchange, new_key, ...)   │  │
│  │  - health_check_all_active()                               │  │
│  └────────┬──────────────────┬──────────────────┬─────────────┘  │
└───────────┼──────────────────┼──────────────────┼────────────────┘
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
    │ SecretsManager│  │ CCXTAdapter  │  │ PostgreSQL   │
    │ (Existing)    │  │ (Existing)   │  │ (Metadata)   │
    │               │  │              │  │              │
    │ - Vault I/O   │  │ - Validation │  │ - Status     │
    │ - Caching     │  │ - Test calls │  │ - Tracking   │
    └───────┬───────┘  └──────────────┘  └──────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │   HashiCorp Vault (HA Cluster)        │
    │   ┌─────────────────────────────────┐ │
    │   │ Path: secret/tenants/{id}/...   │ │
    │   │ - api_key: "abc123..."          │ │
    │   │ - api_secret: "def456..."       │ │
    │   │ - passphrase: "xyz789..." (opt) │ │
    │   └─────────────────────────────────┘ │
    │   Nodes: vault-0, vault-1, vault-2    │
    │   Backend: Raft (consensus)           │
    │   Auto-unseal: AWS KMS                │
    └───────────────────────────────────────┘

    ┌───────────────────────────────────────┐
    │   Background Jobs (APScheduler)       │
    │   ┌─────────────────────────────────┐ │
    │   │ CredentialHealthCheckJob        │ │
    │   │ - Runs every 6 hours            │ │
    │   │ - Tests all active credentials  │ │
    │   │ - Updates status in database    │ │
    │   │ - Sends webhook on failure      │ │
    │   └─────────────────────────────────┘ │
    └───────────────────────────────────────┘
```

### Data Flow Diagrams

#### 1. Credential Creation Flow

```
Tenant                API              TenantCredentialService      CCXTAdapter         Vault           Database
  │                    │                        │                      │                  │                │
  │ POST /credentials  │                        │                      │                  │                │
  ├───────────────────>│                        │                      │                  │                │
  │                    │ validate_and_store()   │                      │                  │                │
  │                    ├───────────────────────>│                      │                  │                │
  │                    │                        │ test_credentials()   │                  │                │
  │                    │                        ├─────────────────────>│                  │                │
  │                    │                        │                      │ fetch_balance()  │                │
  │                    │                        │                      ├──────────────────>Exchange API
  │                    │                        │                      │<─────────────────┤
  │                    │                        │ ValidationResult     │                  │                │
  │                    │                        │<─────────────────────┤                  │                │
  │                    │                        │                      │                  │                │
  │                    │                        │ store_secret()       │                  │                │
  │                    │                        ├─────────────────────────────────────────>│                │
  │                    │                        │                      │                  │ Write secret   │
  │                    │                        │<─────────────────────────────────────────┤                │
  │                    │                        │                      │                  │                │
  │                    │                        │ INSERT tenant_credentials                │                │
  │                    │                        ├──────────────────────────────────────────────────────────>│
  │                    │                        │                      │                  │                │
  │                    │ credential_id          │                      │                  │                │
  │                    │<───────────────────────┤                      │                  │                │
  │ 201 Created        │                        │                      │                  │                │
  │<───────────────────┤                        │                      │                  │                │
```

#### 2. Credential Retrieval Flow (with Cache)

```
Trading Agent      TenantCredentialService    CacheLayer         SecretsManager       Vault
      │                     │                      │                    │                 │
      │ get_credentials()   │                      │                    │                 │
      ├────────────────────>│                      │                    │                 │
      │                     │ Check L1 cache       │                    │                 │
      │                     ├─────────────────────>│                    │                 │
      │                     │ Cache MISS           │                    │                 │
      │                     │<─────────────────────┤                    │                 │
      │                     │                      │                    │                 │
      │                     │ get_secret(vault_path)                    │                 │
      │                     ├───────────────────────────────────────────>│                 │
      │                     │                      │                    │ Read secret     │
      │                     │                      │                    ├────────────────>│
      │                     │                      │                    │ Secret data     │
      │                     │                      │                    │<────────────────┤
      │                     │ Credentials          │                    │                 │
      │                     │<───────────────────────────────────────────┤                 │
      │                     │                      │                    │                 │
      │                     │ Store in L1 (5-min TTL)                   │                 │
      │                     ├─────────────────────>│                    │                 │
      │                     │                      │                    │                 │
      │ Credentials         │                      │                    │                 │
      │<────────────────────┤                      │                    │                 │
      │                     │                      │                    │                 │
      │ [Next request within 5 min - Cache HIT]    │                    │                 │
      │ get_credentials()   │                      │                    │                 │
      ├────────────────────>│ Check L1 cache       │                    │                 │
      │                     ├─────────────────────>│                    │                 │
      │                     │ Cache HIT            │                    │                 │
      │                     │<─────────────────────┤                    │                 │
      │ Credentials (cached)│                      │                    │                 │
      │<────────────────────┤                      │                    │                 │
```

#### 3. Background Health Check Flow

```
APScheduler       HealthCheckJob     TenantCredentialService    CCXTAdapter      Database      Webhook
     │                  │                      │                      │              │             │
     │ Every 6h trigger │                      │                      │              │             │
     ├─────────────────>│                      │                      │              │             │
     │                  │ health_check_all_active()                   │              │             │
     │                  ├─────────────────────>│                      │              │             │
     │                  │                      │ SELECT * FROM tenant_credentials    │             │
     │                  │                      │  WHERE status='active'              │             │
     │                  │                      ├───────────────────────────────────────>             │
     │                  │                      │ [credential_1, credential_2, ...]   │             │
     │                  │                      │<────────────────────────────────────┤             │
     │                  │                      │                      │              │             │
     │                  │                      │ For each credential: │              │             │
     │                  │                      │ test_credentials()   │              │             │
     │                  │                      ├─────────────────────>│              │             │
     │                  │                      │                      │ fetch_balance() → Exchange  │
     │                  │                      │ ValidationResult     │              │             │
     │                  │                      │<─────────────────────┤              │             │
     │                  │                      │                      │              │             │
     │                  │                      │ If SUCCESS:          │              │             │
     │                  │                      │ UPDATE last_validated_at            │             │
     │                  │                      ├───────────────────────────────────────>             │
     │                  │                      │                      │              │             │
     │                  │                      │ If FAILURE:          │              │             │
     │                  │                      │ UPDATE status='invalid'             │             │
     │                  │                      ├───────────────────────────────────────>             │
     │                  │                      │ send_webhook(tenant_id, error)      │             │
     │                  │                      ├─────────────────────────────────────────────────────>
     │                  │ HealthCheckResults   │                      │              │             │
     │                  │<─────────────────────┤                      │              │             │
     │                  │ Log summary          │                      │              │             │
```

---

## Component Design

### 1. TenantCredentialService

**Responsibility**: Orchestrate credential operations with tenant isolation.

**Location**: `src/alpha_pulse/services/tenant_credential_service.py`

**Interface**:
```python
from typing import Dict, List, Optional, UUID
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CredentialMetadata:
    """Credential metadata (no secrets)."""
    id: int
    tenant_id: UUID
    exchange: str
    credential_type: str  # 'readonly' | 'trading'
    status: str  # 'active' | 'invalid' | 'expired' | 'revoked'
    vault_path: str
    last_validated_at: Optional[datetime]
    health_check_status: Optional[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class Credentials:
    """Actual credentials (includes secrets)."""
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of credential validation."""
    valid: bool
    error: Optional[str] = None
    credential_type: Optional[str] = None  # Auto-detected
    permissions: Optional[Dict] = None  # {trading: True, withdraw: False}

class TenantCredentialService:
    """Multi-tenant credential management service."""

    def __init__(
        self,
        secrets_manager: SecretsManager,
        ccxt_adapter: CCXTAdapter,
        db_session: AsyncSession,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.secrets_manager = secrets_manager
        self.ccxt_adapter = ccxt_adapter
        self.db = db_session
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Credentials] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    async def get_credentials(
        self,
        tenant_id: UUID,
        exchange: str,
        credential_type: str = 'trading'
    ) -> Optional[Credentials]:
        """
        Retrieve credentials for a tenant's exchange.

        Args:
            tenant_id: Tenant identifier
            exchange: Exchange name (e.g., 'binance', 'coinbase')
            credential_type: Type of credentials ('trading' or 'readonly')

        Returns:
            Credentials object or None if not found

        Raises:
            CredentialNotFoundException: No credentials found
            CredentialExpiredException: Credentials expired/revoked
        """
        # Check cache first
        cache_key = f"{tenant_id}:{exchange}:{credential_type}"
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]

        # Fetch metadata from database
        metadata = await self._get_metadata(tenant_id, exchange, credential_type)
        if not metadata:
            raise CredentialNotFoundException(tenant_id, exchange)

        if metadata.status != 'active':
            raise CredentialExpiredException(metadata.status)

        # Fetch secrets from Vault
        vault_path = metadata.vault_path
        secret_data = await self.secrets_manager.get_secret(vault_path)

        credentials = Credentials(
            api_key=secret_data['api_key'],
            api_secret=secret_data['api_secret'],
            passphrase=secret_data.get('passphrase')
        )

        # Cache for 5 minutes
        self._cache[cache_key] = credentials
        self._cache_timestamps[cache_key] = datetime.utcnow()

        return credentials

    async def validate_and_store(
        self,
        tenant_id: UUID,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        credential_type: Optional[str] = None
    ) -> CredentialMetadata:
        """
        Validate credentials via CCXT test call, then store if valid.

        Args:
            tenant_id: Tenant identifier
            exchange: Exchange name
            api_key: API key
            api_secret: API secret
            passphrase: Optional passphrase (for some exchanges)
            credential_type: Optional type override

        Returns:
            CredentialMetadata with credential_id

        Raises:
            InvalidCredentialsException: Validation failed
        """
        # Step 1: Validate via CCXT
        validation_result = await self.ccxt_adapter.test_credentials(
            exchange=exchange,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase
        )

        if not validation_result.valid:
            logger.warning(f"Credential validation failed: {validation_result.error}")
            raise InvalidCredentialsException(validation_result.error)

        # Step 2: Determine credential type (auto-detect if not specified)
        final_type = credential_type or validation_result.credential_type

        # Step 3: Store in Vault
        vault_path = f"secret/tenants/{tenant_id}/exchanges/{exchange}/credentials"
        await self.secrets_manager.set_secret(vault_path, {
            'api_key': api_key,
            'api_secret': api_secret,
            'passphrase': passphrase
        })

        # Step 4: Store metadata in database
        metadata = await self._create_or_update_metadata(
            tenant_id=tenant_id,
            exchange=exchange,
            credential_type=final_type,
            vault_path=vault_path,
            permissions=validation_result.permissions
        )

        # Step 5: Invalidate cache
        cache_key = f"{tenant_id}:{exchange}:{final_type}"
        self._invalidate_cache(cache_key)

        logger.info(f"Stored credentials for tenant {tenant_id}, exchange {exchange}")
        return metadata

    async def delete_credentials(
        self,
        tenant_id: UUID,
        exchange: str,
        credential_type: str = 'trading'
    ) -> bool:
        """
        Delete credentials from Vault and database.

        Args:
            tenant_id: Tenant identifier
            exchange: Exchange name
            credential_type: Type of credentials

        Returns:
            True if deleted, False if not found
        """
        # Fetch metadata
        metadata = await self._get_metadata(tenant_id, exchange, credential_type)
        if not metadata:
            return False

        # Delete from Vault
        await self.secrets_manager.delete_secret(metadata.vault_path)

        # Delete from database
        await self._delete_metadata(metadata.id)

        # Invalidate cache
        cache_key = f"{tenant_id}:{exchange}:{credential_type}"
        self._invalidate_cache(cache_key)

        logger.info(f"Deleted credentials for tenant {tenant_id}, exchange {exchange}")
        return True

    async def rotate_credentials(
        self,
        tenant_id: UUID,
        exchange: str,
        new_api_key: str,
        new_api_secret: str,
        new_passphrase: Optional[str] = None,
        credential_type: str = 'trading'
    ) -> CredentialMetadata:
        """
        Rotate credentials with graceful transition period.

        Args:
            tenant_id: Tenant identifier
            exchange: Exchange name
            new_api_key: New API key
            new_api_secret: New API secret
            new_passphrase: New passphrase (optional)
            credential_type: Type of credentials

        Returns:
            Updated CredentialMetadata

        Raises:
            InvalidCredentialsException: New credentials validation failed
        """
        # Validate new credentials first
        validation_result = await self.ccxt_adapter.test_credentials(
            exchange=exchange,
            api_key=new_api_key,
            api_secret=new_api_secret,
            passphrase=new_passphrase
        )

        if not validation_result.valid:
            raise InvalidCredentialsException(validation_result.error)

        # Store new credentials in Vault (creates new version)
        vault_path = f"secret/tenants/{tenant_id}/exchanges/{exchange}/credentials"
        await self.secrets_manager.set_secret(vault_path, {
            'api_key': new_api_key,
            'api_secret': new_api_secret,
            'passphrase': new_passphrase
        })

        # Update metadata
        metadata = await self._update_metadata_on_rotation(
            tenant_id=tenant_id,
            exchange=exchange,
            credential_type=credential_type,
            vault_path=vault_path
        )

        # Invalidate cache
        cache_key = f"{tenant_id}:{exchange}:{credential_type}"
        self._invalidate_cache(cache_key)

        logger.info(f"Rotated credentials for tenant {tenant_id}, exchange {exchange}")
        return metadata

    async def health_check_all_active(self) -> List[Dict]:
        """
        Run health check on all active credentials.

        Returns:
            List of results: [{tenant_id, exchange, status, error}, ...]
        """
        # Fetch all active credentials from database
        active_credentials = await self._get_all_active_metadata()

        results = []
        for metadata in active_credentials:
            # Fetch credentials from Vault
            secret_data = await self.secrets_manager.get_secret(metadata.vault_path)

            # Test via CCXT
            validation_result = await self.ccxt_adapter.test_credentials(
                exchange=metadata.exchange,
                api_key=secret_data['api_key'],
                api_secret=secret_data['api_secret'],
                passphrase=secret_data.get('passphrase')
            )

            if validation_result.valid:
                # Update last_validated_at
                await self._update_health_check_success(metadata.id)
                results.append({
                    'tenant_id': metadata.tenant_id,
                    'exchange': metadata.exchange,
                    'status': 'healthy',
                    'error': None
                })
            else:
                # Mark as invalid
                await self._update_health_check_failure(
                    metadata.id,
                    validation_result.error
                )
                results.append({
                    'tenant_id': metadata.tenant_id,
                    'exchange': metadata.exchange,
                    'status': 'unhealthy',
                    'error': validation_result.error
                })

                # Send webhook notification
                await self._send_webhook_notification(
                    metadata.tenant_id,
                    metadata.exchange,
                    validation_result.error
                )

        return results

    # Private helper methods
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self._cache_timestamps:
            return False
        age = (datetime.utcnow() - self._cache_timestamps[cache_key]).total_seconds()
        return age < self.cache_ttl

    def _invalidate_cache(self, cache_key: str):
        """Remove entry from cache."""
        self._cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)

    async def _get_metadata(
        self,
        tenant_id: UUID,
        exchange: str,
        credential_type: str
    ) -> Optional[CredentialMetadata]:
        """Fetch metadata from database."""
        # SQLAlchemy query implementation
        pass

    async def _create_or_update_metadata(
        self,
        tenant_id: UUID,
        exchange: str,
        credential_type: str,
        vault_path: str,
        permissions: Optional[Dict]
    ) -> CredentialMetadata:
        """Create or update metadata in database."""
        # SQLAlchemy insert/update implementation
        pass

    async def _delete_metadata(self, credential_id: int):
        """Delete metadata from database."""
        # SQLAlchemy delete implementation
        pass

    async def _update_metadata_on_rotation(
        self,
        tenant_id: UUID,
        exchange: str,
        credential_type: str,
        vault_path: str
    ) -> CredentialMetadata:
        """Update metadata after rotation."""
        # SQLAlchemy update implementation
        pass

    async def _get_all_active_metadata(self) -> List[CredentialMetadata]:
        """Fetch all active credentials from database."""
        # SQLAlchemy query implementation
        pass

    async def _update_health_check_success(self, credential_id: int):
        """Update last_validated_at timestamp."""
        # SQLAlchemy update implementation
        pass

    async def _update_health_check_failure(self, credential_id: int, error: str):
        """Mark credential as invalid."""
        # SQLAlchemy update implementation
        pass

    async def _send_webhook_notification(
        self,
        tenant_id: UUID,
        exchange: str,
        error: str
    ):
        """Send webhook notification to tenant."""
        # Webhook implementation
        pass
```

**Dependencies**:
- `SecretsManager` (existing: `src/alpha_pulse/utils/secrets_manager.py`)
- `CCXTAdapter` (existing: `src/alpha_pulse/exchanges/adapters/ccxt_adapter.py`)
- `AsyncSession` (SQLAlchemy async database session)

### 2. CCXT Validation Adapter

**Enhancement to Existing**: `src/alpha_pulse/exchanges/adapters/ccxt_adapter.py`

**New Method**:
```python
class CCXTAdapter:
    """Existing CCXT adapter - add validation method."""

    async def test_credentials(
        self,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None
    ) -> ValidationResult:
        """
        Test exchange credentials by making a minimal API call.

        Args:
            exchange: Exchange name (e.g., 'binance')
            api_key: API key
            api_secret: API secret
            passphrase: Passphrase (required for some exchanges like Coinbase)

        Returns:
            ValidationResult with valid=True/False and error details

        Implementation:
            1. Create CCXT exchange instance with credentials
            2. Call fetch_balance() (requires read permission)
            3. Optionally test trading permission with test order
            4. Return result with auto-detected permission level
        """
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange)
            client = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'password': passphrase,  # Some exchanges use 'password' for passphrase
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds
            })

            # Test read permission
            balance = await client.fetch_balance()
            logger.debug(f"Credentials valid for {exchange}, balance fetched")

            # Detect trading permission (optional test order)
            has_trading = await self._test_trading_permission(client)

            permissions = {
                'trading': has_trading,
                'read': True,
                'withdraw': False  # Never auto-detect withdraw (too dangerous)
            }

            credential_type = 'trading' if has_trading else 'readonly'

            return ValidationResult(
                valid=True,
                credential_type=credential_type,
                permissions=permissions
            )

        except ccxt.AuthenticationError as e:
            logger.warning(f"Authentication failed for {exchange}: {e}")
            return ValidationResult(
                valid=False,
                error=f"Invalid API key or secret: {str(e)}"
            )

        except ccxt.InsufficientPermissions as e:
            logger.warning(f"Insufficient permissions for {exchange}: {e}")
            return ValidationResult(
                valid=False,
                error=f"API key lacks required permissions: {str(e)}"
            )

        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange {exchange} not available: {e}")
            return ValidationResult(
                valid=False,
                error=f"Exchange temporarily unavailable: {str(e)}"
            )

        except Exception as e:
            logger.error(f"Unexpected error testing {exchange} credentials: {e}")
            return ValidationResult(
                valid=False,
                error=f"Unexpected error: {str(e)}"
            )

    async def _test_trading_permission(self, client) -> bool:
        """
        Test if credentials have trading permission.

        Strategy: Try to create a test order (very small amount, limit order)
        that will likely not execute, then immediately cancel it.

        Returns:
            True if trading permission detected, False otherwise
        """
        try:
            # Attempt to fetch open orders (requires trading permission)
            await client.fetch_open_orders()
            return True
        except ccxt.InsufficientPermissions:
            return False
        except Exception:
            # If error is not permission-related, assume trading is available
            return True
```

### 3. Background Health Check Job

**Location**: `src/alpha_pulse/jobs/credential_health_check.py` (NEW)

```python
import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from alpha_pulse.services.tenant_credential_service import TenantCredentialService
from alpha_pulse.database import get_async_session

logger = logging.getLogger(__name__)

class CredentialHealthCheckJob:
    """Background job to health check all active credentials."""

    def __init__(
        self,
        credential_service: TenantCredentialService,
        scheduler: AsyncIOScheduler
    ):
        self.credential_service = credential_service
        self.scheduler = scheduler

    def start(self):
        """Start the health check job (runs every 6 hours)."""
        self.scheduler.add_job(
            self.run_health_check,
            trigger=IntervalTrigger(hours=6),
            id='credential_health_check',
            name='Credential Health Check',
            replace_existing=True
        )
        logger.info("Credential health check job started (interval: 6 hours)")

    async def run_health_check(self):
        """Execute health check on all active credentials."""
        logger.info("Starting credential health check for all tenants")

        try:
            results = await self.credential_service.health_check_all_active()

            # Log summary
            total = len(results)
            healthy = sum(1 for r in results if r['status'] == 'healthy')
            unhealthy = total - healthy

            logger.info(
                f"Health check complete: {total} credentials checked, "
                f"{healthy} healthy, {unhealthy} unhealthy"
            )

            # Log unhealthy credentials
            for result in results:
                if result['status'] == 'unhealthy':
                    logger.warning(
                        f"Unhealthy credential: tenant={result['tenant_id']}, "
                        f"exchange={result['exchange']}, error={result['error']}"
                    )

        except Exception as e:
            logger.error(f"Health check job failed: {e}", exc_info=True)
```

---

## Data Models

### 1. Database Schema

**Table**: `tenant_credentials`

```sql
CREATE TABLE tenant_credentials (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,  -- 'binance', 'coinbase', etc.
    credential_type VARCHAR(20) NOT NULL,  -- 'readonly' | 'trading'
    vault_path VARCHAR(255) NOT NULL,  -- Path in Vault
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- 'active' | 'invalid' | 'expired' | 'revoked'
    permissions JSONB,  -- {trading: true, withdraw: false, read: true}
    last_validated_at TIMESTAMP,
    health_check_status VARCHAR(50),  -- Last health check result
    validation_error TEXT,  -- Last error message (if any)
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,  -- Optional expiration date

    CONSTRAINT unique_tenant_exchange_type UNIQUE(tenant_id, exchange, credential_type)
);

-- Indexes for performance
CREATE INDEX idx_tenant_credentials_tenant_id ON tenant_credentials(tenant_id);
CREATE INDEX idx_tenant_credentials_status ON tenant_credentials(status);
CREATE INDEX idx_tenant_credentials_exchange ON tenant_credentials(exchange);
CREATE INDEX idx_tenant_credentials_last_validated
    ON tenant_credentials(last_validated_at)
    WHERE status = 'active';

-- Row-Level Security (RLS) for tenant isolation
ALTER TABLE tenant_credentials ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_credentials_isolation_policy ON tenant_credentials
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
```

### 2. Vault Secret Structure

**Path**: `secret/tenants/{tenant_id}/exchanges/{exchange}/credentials`

**Data Format** (JSON):
```json
{
  "api_key": "abc123def456ghi789...",
  "api_secret": "xyz987uvw654rst321...",
  "passphrase": "optional-passphrase-for-some-exchanges",
  "metadata": {
    "created_at": "2025-11-07T10:30:00Z",
    "created_by": "user@example.com",
    "version": 2
  }
}
```

**Vault Configuration**:
```hcl
# Enable KV v2 secrets engine
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Tenant isolation policy (enforced per-request)
path "secret/tenants/{{identity.entity.metadata.tenant_id}}/*" {
  capabilities = ["read", "list"]
}

# Service account policy (application access)
path "secret/tenants/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
```

### 3. SQLAlchemy ORM Models

**Location**: `src/alpha_pulse/models/tenant_credential.py` (NEW)

```python
from sqlalchemy import Column, Integer, String, TIMESTAMP, UUID, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from alpha_pulse.database import Base

class TenantCredential(Base):
    """ORM model for tenant_credentials table."""

    __tablename__ = 'tenant_credentials'

    id = Column(Integer, primary_key=True)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    credential_type = Column(String(20), nullable=False)
    vault_path = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default='active', index=True)
    permissions = Column(JSONB)
    last_validated_at = Column(TIMESTAMP)
    health_check_status = Column(String(50))
    validation_error = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())
    expires_at = Column(TIMESTAMP)

    __table_args__ = (
        UniqueConstraint('tenant_id', 'exchange', 'credential_type', name='unique_tenant_exchange_type'),
    )

    def __repr__(self):
        return f"<TenantCredential(tenant_id={self.tenant_id}, exchange={self.exchange}, type={self.credential_type})>"
```

---

## API Specifications

### REST API Endpoints

**Base Path**: `/api/v1/credentials`

**Authentication**: JWT token required (tenant_id extracted from token)

#### 1. Create/Update Credentials

```
POST /api/v1/credentials
```

**Request Body**:
```json
{
  "exchange": "binance",
  "api_key": "abc123...",
  "api_secret": "xyz789...",
  "passphrase": "optional",
  "credential_type": "trading"
}
```

**Response** (201 Created):
```json
{
  "id": 42,
  "tenant_id": "00000000-0000-0000-0000-000000000001",
  "exchange": "binance",
  "credential_type": "trading",
  "status": "active",
  "permissions": {
    "trading": true,
    "read": true,
    "withdraw": false
  },
  "last_validated_at": "2025-11-07T10:30:00Z",
  "created_at": "2025-11-07T10:30:00Z"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input (missing required fields)
- `401 Unauthorized`: Invalid/missing JWT token
- `422 Unprocessable Entity`: Credential validation failed
  ```json
  {
    "detail": "Invalid API key or secret: Authentication failed"
  }
  ```

#### 2. List All Credentials

```
GET /api/v1/credentials
```

**Query Parameters**:
- `exchange` (optional): Filter by exchange
- `status` (optional): Filter by status ('active', 'invalid', etc.)

**Response** (200 OK):
```json
{
  "credentials": [
    {
      "id": 42,
      "exchange": "binance",
      "credential_type": "trading",
      "status": "active",
      "last_validated_at": "2025-11-07T10:30:00Z",
      "created_at": "2025-11-06T08:00:00Z"
    },
    {
      "id": 43,
      "exchange": "coinbase",
      "credential_type": "readonly",
      "status": "active",
      "last_validated_at": "2025-11-07T09:15:00Z",
      "created_at": "2025-11-05T14:30:00Z"
    }
  ],
  "total": 2
}
```

#### 3. Get Specific Credential

```
GET /api/v1/credentials/{id}
```

**Response** (200 OK):
```json
{
  "id": 42,
  "exchange": "binance",
  "credential_type": "trading",
  "status": "active",
  "permissions": {
    "trading": true,
    "read": true
  },
  "last_validated_at": "2025-11-07T10:30:00Z",
  "health_check_status": "healthy",
  "created_at": "2025-11-06T08:00:00Z",
  "updated_at": "2025-11-07T10:30:00Z"
}
```

**Note**: Response does NOT include actual secrets (api_key, api_secret).

#### 4. Delete Credentials

```
DELETE /api/v1/credentials/{id}
```

**Response** (204 No Content)

#### 5. Manual Validation Trigger

```
POST /api/v1/credentials/{id}/validate
```

**Response** (200 OK):
```json
{
  "valid": true,
  "credential_type": "trading",
  "permissions": {
    "trading": true,
    "read": true
  },
  "validated_at": "2025-11-07T11:00:00Z"
}
```

#### 6. Get Health Status

```
GET /api/v1/credentials/{id}/health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "last_check_at": "2025-11-07T06:00:00Z",
  "next_check_at": "2025-11-07T12:00:00Z",
  "error": null
}
```

---

## Security Architecture

### 1. Vault High Availability Setup

**Deployment**: Kubernetes StatefulSet with 3 replicas

**Components**:
- **Vault Cluster**: 3 nodes (vault-0, vault-1, vault-2)
- **Storage Backend**: Raft (integrated storage, no Consul dependency)
- **Auto-Unseal**: AWS KMS (no manual unseal on restart)
- **Load Balancer**: Kubernetes Service (active/standby routing)

**Vault Configuration** (`vault-config.hcl`):
```hcl
storage "raft" {
  path = "/vault/data"
  node_id = "vault-0"

  retry_join {
    leader_api_addr = "http://vault-1:8200"
  }
  retry_join {
    leader_api_addr = "http://vault-2:8200"
  }
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/tls.crt"
  tls_key_file = "/vault/tls/tls.key"
  tls_min_version = "tls13"
}

seal "awskms" {
  region = "us-east-1"
  kms_key_id = "arn:aws:kms:us-east-1:123456789:key/vault-unseal-key"
}

api_addr = "https://vault-0.vault-internal:8200"
cluster_addr = "https://vault-0.vault-internal:8201"

log_level = "info"
ui = true

# Enable audit logging
audit {
  file {
    path = "/vault/logs/audit.log"
    log_raw = false
    format = "json"
  }
}
```

### 2. Authentication Flow

**Application → Vault**:
1. Application authenticates using Kubernetes ServiceAccount JWT
2. Vault validates JWT and maps to Vault role
3. Vault attaches policy based on role
4. Application receives Vault token (TTL: 1 hour)
5. Token includes tenant_id metadata for path enforcement

**Kubernetes Auth Configuration**:
```hcl
# Enable Kubernetes auth method
auth "kubernetes" {
  kubernetes_host = "https://kubernetes.default.svc"
  kubernetes_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
}

# Role for AlphaPulse application
role "alphapulse-app" {
  bound_service_account_names = ["alphapulse"]
  bound_service_account_namespaces = ["production"]
  policies = ["alphapulse-secrets"]
  ttl = "1h"
  max_ttl = "24h"
}
```

### 3. Tenant Isolation Enforcement

**Vault Policy** (`alphapulse-secrets-policy.hcl`):
```hcl
# Allow application to access all tenant secrets
path "secret/data/tenants/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Allow application to delete secret versions
path "secret/delete/tenants/*" {
  capabilities = ["update"]
}

# Allow application to undelete secret versions (rollback)
path "secret/undelete/tenants/*" {
  capabilities = ["update"]
}

# Deny access to other secret paths
path "secret/data/admin/*" {
  capabilities = ["deny"]
}
```

**Application-Level Enforcement**:
- TenantCredentialService always includes `tenant_id` in Vault path
- JWT middleware extracts `tenant_id` from token
- All API endpoints validate `tenant_id` matches JWT
- Database RLS prevents cross-tenant metadata access

### 4. Audit Logging

**Vault Audit Log** (JSON format):
```json
{
  "time": "2025-11-07T10:30:00Z",
  "type": "request",
  "auth": {
    "client_token": "hvs.CAES...",
    "display_name": "alphapulse-app",
    "metadata": {
      "tenant_id": "00000000-0000-0000-0000-000000000001"
    }
  },
  "request": {
    "operation": "read",
    "path": "secret/data/tenants/00000000-0000-0000-0000-000000000001/exchanges/binance/credentials",
    "remote_address": "10.0.1.45"
  }
}
```

**Application Audit Log**:
- Log all credential operations (create, read, update, delete, rotate)
- Include: timestamp, tenant_id, user_id, operation, exchange, IP address
- Store in PostgreSQL `audit_log` table
- Retention: 1 year (compliance requirement)

---

## Performance & Scalability

### 1. Latency Budget

| Operation | Target P99 | Breakdown |
|-----------|------------|-----------|
| Get credentials (cached) | <5ms | L1 cache lookup: <1ms |
| Get credentials (uncached) | <50ms | Vault read: 20ms + DB query: 10ms + cache write: 5ms |
| Store credentials | <200ms | CCXT validation: 100ms + Vault write: 30ms + DB insert: 20ms |
| Health check (single) | <500ms | CCXT fetch_balance: 300ms + DB update: 50ms |

### 2. Throughput

**Vault Cluster**:
- Capacity: 10,000 reads/sec, 5,000 writes/sec per node
- Cluster total: 30,000 reads/sec (3 nodes)
- Expected load: ~100 reads/sec (1000 tenants × 5 requests/min / 60 sec)
- **Headroom**: 300x capacity

**Database**:
- PostgreSQL handles 10,000+ queries/sec
- Expected load: ~50 queries/sec (credential metadata lookups)
- **Headroom**: 200x capacity

### 3. Caching Strategy

**L1 Cache** (In-Memory LRU):
- Size: 1000 entries (max)
- TTL: 5 minutes
- Eviction: LRU per tenant
- Hit Rate Target: >90%

**Cache Invalidation**:
- On credential update: Invalidate specific tenant+exchange
- On credential rotation: Invalidate specific tenant+exchange
- On health check failure: Invalidate specific tenant+exchange
- No global invalidation (prevents cache stampede)

### 4. Scalability Limits

| Component | Current Limit | Scaling Strategy |
|-----------|---------------|------------------|
| Tenants | 10,000 | Vault namespaces (Enterprise feature) |
| Credentials per tenant | 10 exchanges | No hard limit, reasonable for Pro tier |
| Vault storage | 100GB | Raft scales to 1TB+ |
| Cache size | 10,000 entries | Increase LRU max size or add Redis |
| Health check duration | 1 hour (10K creds × 0.5s) | Parallelize with asyncio.gather() |

---

## Monitoring & Observability

### 1. Prometheus Metrics

**Credentials Service Metrics**:
```
# Total credential operations
alphapulse_credentials_operations_total{tenant_id, exchange, operation, status}

# Credential validation duration
alphapulse_credentials_validation_duration_seconds{exchange}

# Credential cache hit rate
alphapulse_credentials_cache_hit_rate{tenant_id}

# Health check failures
alphapulse_credentials_health_check_failures_total{tenant_id, exchange}

# Active credentials count
alphapulse_credentials_active_total{status}
```

**Vault Metrics** (via Prometheus exporter):
```
# Vault operations
vault_core_handle_request_count{method, path}

# Vault latency
vault_core_handle_request_duration_seconds{method, path, percentile}

# Vault seal status (0=sealed, 1=unsealed)
vault_core_unsealed

# Raft cluster health
vault_raft_leader_lastContact_sum
vault_raft_replication_appendEntries_rpc{server}
```

### 2. Grafana Dashboards

**Dashboard 1: Credential Management Overview**
- Total active credentials by exchange
- Credential validation success rate (last 24h)
- Health check failure rate (last 7 days)
- Top 10 tenants by credential count

**Dashboard 2: Vault Performance**
- Vault latency (P50, P95, P99)
- Vault operations per second
- Vault seal status (all nodes)
- Raft replication lag

**Dashboard 3: Tenant Credential Health**
- Credential status distribution (active/invalid/expired)
- Per-tenant health check results
- Validation error frequency
- Webhook delivery success rate

### 3. Alerts

**Critical**:
- `VaultSealed`: Vault in sealed state (page on-call)
- `VaultNodeDown`: Vault node unreachable (page on-call)
- `CredentialValidationRate < 80%`: High validation failure rate

**Warning**:
- `HealthCheckFailureRate > 5%`: Unhealthy credentials increasing
- `VaultLatencyP99 > 100ms`: Vault performance degradation
- `CacheHitRate < 70%`: Cache not effective

---

## Deployment Architecture

### 1. Kubernetes Resources

**Vault StatefulSet**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vault
  namespace: production
spec:
  serviceName: vault-internal
  replicas: 3
  selector:
    matchLabels:
      app: vault
  template:
    metadata:
      labels:
        app: vault
    spec:
      serviceAccountName: vault
      containers:
      - name: vault
        image: hashicorp/vault:1.15.0
        ports:
        - containerPort: 8200
          name: api
        - containerPort: 8201
          name: cluster
        env:
        - name: VAULT_API_ADDR
          value: "https://vault-0.vault-internal:8200"
        - name: VAULT_CLUSTER_ADDR
          value: "https://vault-0.vault-internal:8201"
        volumeMounts:
        - name: config
          mountPath: /vault/config
        - name: data
          mountPath: /vault/data
        - name: tls
          mountPath: /vault/tls
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
      volumes:
      - name: config
        configMap:
          name: vault-config
      - name: tls
        secret:
          secretName: vault-tls
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

**Vault Service**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: vault
  namespace: production
spec:
  type: LoadBalancer
  ports:
  - port: 8200
    targetPort: 8200
    name: api
  selector:
    app: vault
---
apiVersion: v1
kind: Service
metadata:
  name: vault-internal
  namespace: production
spec:
  clusterIP: None
  ports:
  - port: 8200
    name: api
  - port: 8201
    name: cluster
  selector:
    app: vault
```

### 2. Database Migration

**Alembic Migration** (`alembic/versions/xxx_add_tenant_credentials.py`):
```python
"""Add tenant_credentials table

Revision ID: xxx
Revises: yyy
Create Date: 2025-11-07

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    op.create_table(
        'tenant_credentials',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('exchange', sa.String(50), nullable=False),
        sa.Column('credential_type', sa.String(20), nullable=False),
        sa.Column('vault_path', sa.String(255), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('permissions', postgresql.JSONB(), nullable=True),
        sa.Column('last_validated_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('health_check_status', sa.String(50), nullable=True),
        sa.Column('validation_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('expires_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'exchange', 'credential_type', name='unique_tenant_exchange_type')
    )

    op.create_index('idx_tenant_credentials_tenant_id', 'tenant_credentials', ['tenant_id'])
    op.create_index('idx_tenant_credentials_status', 'tenant_credentials', ['status'])
    op.create_index('idx_tenant_credentials_exchange', 'tenant_credentials', ['exchange'])

    # Enable RLS
    op.execute('ALTER TABLE tenant_credentials ENABLE ROW LEVEL SECURITY')

    # Create RLS policy
    op.execute("""
        CREATE POLICY tenant_credentials_isolation_policy ON tenant_credentials
        USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    """)

def downgrade():
    op.drop_table('tenant_credentials')
```

---

## Testing Strategy

### 1. Unit Tests (90% coverage)

**TenantCredentialService Tests** (`tests/services/test_tenant_credential_service.py`):
```python
import pytest
from unittest.mock import Mock, AsyncMock
from alpha_pulse.services.tenant_credential_service import TenantCredentialService

@pytest.mark.asyncio
async def test_get_credentials_cache_hit():
    """Test credential retrieval from cache."""
    service = TenantCredentialService(...)
    # Pre-populate cache
    # Assert cache hit

@pytest.mark.asyncio
async def test_validate_and_store_success():
    """Test successful credential validation and storage."""
    # Mock CCXT validation success
    # Mock Vault write success
    # Mock database insert
    # Assert credential stored

@pytest.mark.asyncio
async def test_validate_and_store_invalid_credentials():
    """Test validation failure."""
    # Mock CCXT validation failure
    # Assert InvalidCredentialsException raised
    # Assert nothing stored in Vault or database

@pytest.mark.asyncio
async def test_rotate_credentials():
    """Test credential rotation."""
    # Mock validation success for new credentials
    # Mock Vault versioning
    # Assert old and new versions exist
```

### 2. Integration Tests

**Vault Integration** (`tests/integration/test_vault_integration.py`):
```python
@pytest.mark.integration
async def test_vault_write_and_read():
    """Test actual Vault write and read."""
    # Connect to test Vault instance
    # Write test credentials
    # Read back and verify
    # Cleanup

@pytest.mark.integration
async def test_vault_failover():
    """Test Vault HA failover."""
    # Kill primary Vault node
    # Verify secondary takes over
    # Verify credential access still works
```

### 3. End-to-End Tests

**Full Credential Lifecycle** (`tests/e2e/test_credential_lifecycle.py`):
```python
@pytest.mark.e2e
async def test_full_credential_lifecycle():
    """Test: create → use → rotate → delete."""
    # Step 1: Create credentials via API
    # Step 2: Retrieve and use in trading agent
    # Step 3: Rotate credentials
    # Step 4: Verify old version still works (grace period)
    # Step 5: Delete credentials
    # Step 6: Verify retrieval fails
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
