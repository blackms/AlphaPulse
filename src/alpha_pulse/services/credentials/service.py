"""
Multi-tenant credential service with Vault integration and caching.

Provides tenant-scoped credential management with validation and caching.
"""
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from cachetools import TTLCache
from loguru import logger

from alpha_pulse.utils.secrets_manager import HashiCorpVaultProvider
from .validator import CredentialValidator, ValidationResult


# Global/default tenant for backward compatibility
DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000000")


@dataclass
class TenantCredentials:
    """Tenant-scoped exchange credentials."""

    tenant_id: UUID
    exchange: str
    api_key: str
    api_secret: str
    credential_type: str  # 'trading' or 'readonly'
    testnet: bool = False
    passphrase: Optional[str] = None
    exchange_account_id: Optional[str] = None
    last_validated_at: Optional[datetime] = None


class TenantCredentialService:
    """Multi-tenant credential management service with Vault and caching."""

    def __init__(
        self,
        vault_provider: HashiCorpVaultProvider,
        validator: Optional[CredentialValidator] = None,
        cache_ttl: int = 300,  # 5 minutes
        cache_maxsize: int = 1000,
    ):
        """
        Initialize tenant credential service.

        Args:
            vault_provider: HashiCorp Vault provider instance
            validator: Credential validator (optional)
            cache_ttl: Cache TTL in seconds (default: 300 = 5 minutes)
            cache_maxsize: Maximum cache size (default: 1000)
        """
        self.vault = vault_provider
        self.validator = validator or CredentialValidator()

        # Thread-safe TTL cache
        self._cache: TTLCache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self._cache_lock = threading.Lock()

        logger.info(
            f"Initialized TenantCredentialService (cache_ttl={cache_ttl}s, maxsize={cache_maxsize})"
        )

    def _build_vault_path(
        self, tenant_id: UUID, exchange: str, credential_type: str = "trading"
    ) -> str:
        """
        Build Vault path for tenant credentials.

        Args:
            tenant_id: Tenant UUID
            exchange: Exchange name (e.g., 'binance')
            credential_type: Credential type ('trading' or 'readonly')

        Returns:
            Vault path like 'tenants/{tenant_id}/exchanges/{exchange}/{type}'
        """
        return f"tenants/{tenant_id}/exchanges/{exchange}/{credential_type}"

    def _build_cache_key(
        self, tenant_id: UUID, exchange: str, credential_type: str = "trading"
    ) -> str:
        """
        Build cache key for credentials.

        Args:
            tenant_id: Tenant UUID
            exchange: Exchange name
            credential_type: Credential type

        Returns:
            Cache key like 'creds:{tenant_id}:{exchange}:{type}'
        """
        return f"creds:{tenant_id}:{exchange}:{credential_type}"

    async def get_credentials(
        self,
        tenant_id: UUID,
        exchange: str,
        credential_type: str = "trading",
    ) -> Optional[TenantCredentials]:
        """
        Get credentials for tenant and exchange.

        Args:
            tenant_id: Tenant UUID
            exchange: Exchange name
            credential_type: Credential type ('trading' or 'readonly')

        Returns:
            TenantCredentials if found, None otherwise
        """
        cache_key = self._build_cache_key(tenant_id, exchange, credential_type)

        # Check cache first
        with self._cache_lock:
            if cache_key in self._cache:
                logger.debug(
                    f"Cache HIT: {cache_key} (tenant={tenant_id}, exchange={exchange})"
                )
                return self._cache[cache_key]

        logger.debug(
            f"Cache MISS: {cache_key} (tenant={tenant_id}, exchange={exchange})"
        )

        # Fetch from Vault
        vault_path = self._build_vault_path(tenant_id, exchange, credential_type)
        secret_data = self.vault.get_secret(vault_path)

        if not secret_data:
            logger.warning(
                f"No credentials found in Vault for tenant={tenant_id}, exchange={exchange}, type={credential_type}"
            )
            return None

        # Parse credentials
        try:
            credentials = TenantCredentials(
                tenant_id=tenant_id,
                exchange=exchange,
                api_key=secret_data["api_key"],
                api_secret=secret_data["secret"],
                credential_type=secret_data.get("metadata", {}).get(
                    "credential_type", credential_type
                ),
                testnet=secret_data.get("testnet", False),
                passphrase=secret_data.get("passphrase"),
                exchange_account_id=secret_data.get("metadata", {}).get(
                    "exchange_account_id"
                ),
            )

            # Cache the credentials
            with self._cache_lock:
                self._cache[cache_key] = credentials

            logger.info(
                f"Retrieved credentials from Vault for tenant={tenant_id}, exchange={exchange}"
            )
            return credentials

        except KeyError as e:
            logger.error(
                f"Invalid credential format in Vault for tenant={tenant_id}, exchange={exchange}: missing {e}"
            )
            return None

    async def store_credentials(
        self,
        tenant_id: UUID,
        exchange: str,
        api_key: str,
        secret: str,
        passphrase: Optional[str] = None,
        testnet: bool = False,
        validate: bool = True,
        created_by: Optional[str] = None,
    ) -> ValidationResult:
        """
        Store credentials for tenant after validation.

        Args:
            tenant_id: Tenant UUID
            exchange: Exchange name
            api_key: API key
            secret: API secret
            passphrase: Optional passphrase
            testnet: Whether testnet credentials
            validate: Whether to validate before storing (default: True)
            created_by: User who created credentials (for audit)

        Returns:
            ValidationResult with success/error status
        """
        # Validate credentials first (if enabled)
        if validate:
            logger.info(
                f"Validating credentials for tenant={tenant_id}, exchange={exchange}..."
            )
            validation_result = await self.validator.validate(
                exchange=exchange,
                api_key=api_key,
                secret=secret,
                passphrase=passphrase,
                testnet=testnet,
            )

            if not validation_result.valid:
                logger.warning(
                    f"Credential validation failed for tenant={tenant_id}, exchange={exchange}: {validation_result.error}"
                )
                return validation_result
        else:
            # Skip validation - create default result
            validation_result = ValidationResult(
                valid=True, credential_type="trading", error=None
            )

        # Build Vault secret data
        secret_data = {
            "api_key": api_key,
            "secret": secret,
            "testnet": testnet,
            "metadata": {
                "exchange": exchange,
                "credential_type": validation_result.credential_type,
                "created_at": datetime.utcnow().isoformat(),
                "exchange_account_id": validation_result.exchange_account_id,
            },
        }

        if passphrase:
            secret_data["passphrase"] = passphrase

        if created_by:
            secret_data["metadata"]["created_by"] = created_by

        # Store in Vault
        vault_path = self._build_vault_path(
            tenant_id, exchange, validation_result.credential_type or "trading"
        )
        success = self.vault.set_secret(vault_path, secret_data)

        if not success:
            logger.error(
                f"Failed to store credentials in Vault for tenant={tenant_id}, exchange={exchange}"
            )
            return ValidationResult(valid=False, error="Failed to store in Vault")

        # Invalidate cache
        cache_key = self._build_cache_key(
            tenant_id, exchange, validation_result.credential_type or "trading"
        )
        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"Invalidated cache: {cache_key}")

        logger.info(
            f"Successfully stored credentials for tenant={tenant_id}, exchange={exchange}, type={validation_result.credential_type}"
        )
        return validation_result

    async def delete_credentials(
        self,
        tenant_id: UUID,
        exchange: str,
        credential_type: str = "trading",
    ) -> bool:
        """
        Delete credentials for tenant and exchange.

        Args:
            tenant_id: Tenant UUID
            exchange: Exchange name
            credential_type: Credential type

        Returns:
            True if deleted successfully, False otherwise
        """
        vault_path = self._build_vault_path(tenant_id, exchange, credential_type)
        success = self.vault.delete_secret(vault_path)

        if success:
            # Invalidate cache
            cache_key = self._build_cache_key(tenant_id, exchange, credential_type)
            with self._cache_lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.debug(f"Invalidated cache: {cache_key}")

            logger.info(
                f"Deleted credentials for tenant={tenant_id}, exchange={exchange}, type={credential_type}"
            )

        return success

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache size, hits, misses, etc.
        """
        with self._cache_lock:
            return {
                "size": len(self._cache),
                "maxsize": self._cache.maxsize,
                "ttl": self._cache.ttl,
                "currsize": self._cache.currsize,
            }

    def clear_cache(self, tenant_id: Optional[UUID] = None) -> int:
        """
        Clear cache (optionally for specific tenant).

        Args:
            tenant_id: Tenant UUID (if None, clear all)

        Returns:
            Number of entries cleared
        """
        with self._cache_lock:
            if tenant_id is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared all cache entries ({count} total)")
                return count
            else:
                # Clear only tenant-specific entries
                prefix = f"creds:{tenant_id}:"
                keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
                for key in keys_to_delete:
                    del self._cache[key]
                logger.info(
                    f"Cleared {len(keys_to_delete)} cache entries for tenant={tenant_id}"
                )
                return len(keys_to_delete)
