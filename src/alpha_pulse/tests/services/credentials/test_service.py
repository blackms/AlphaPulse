"""
Unit tests for TenantCredentialService.

Tests the multi-tenant credential service with Vault integration and caching.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import UUID

from alpha_pulse.services.credentials.service import (
    TenantCredentialService,
    TenantCredentials,
    DEFAULT_TENANT_ID,
)
from alpha_pulse.services.credentials.validator import (
    CredentialValidator,
    ValidationResult,
)


class TestTenantCredentials:
    """Tests for TenantCredentials dataclass."""

    def test_tenant_credentials_creation(self):
        """Test creating TenantCredentials instance."""
        tenant_id = UUID("12345678-1234-1234-1234-123456789abc")
        creds = TenantCredentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            credential_type="trading",
            testnet=False,
        )

        assert creds.tenant_id == tenant_id
        assert creds.exchange == "binance"
        assert creds.api_key == "test_key"
        assert creds.api_secret == "test_secret"
        assert creds.credential_type == "trading"
        assert creds.testnet is False
        assert creds.passphrase is None
        assert creds.exchange_account_id is None
        assert creds.last_validated_at is None

    def test_tenant_credentials_with_optional_fields(self):
        """Test TenantCredentials with all optional fields."""
        tenant_id = UUID("12345678-1234-1234-1234-123456789abc")
        now = datetime.utcnow()

        creds = TenantCredentials(
            tenant_id=tenant_id,
            exchange="coinbase",
            api_key="test_key",
            api_secret="test_secret",
            credential_type="readonly",
            testnet=True,
            passphrase="test_passphrase",
            exchange_account_id="account-123",
            last_validated_at=now,
        )

        assert creds.passphrase == "test_passphrase"
        assert creds.exchange_account_id == "account-123"
        assert creds.last_validated_at == now


class TestTenantCredentialService:
    """Tests for TenantCredentialService class."""

    @pytest.fixture
    def mock_vault(self):
        """Create a mock Vault provider."""
        vault = Mock()
        vault.get_secret = Mock(return_value=None)
        vault.set_secret = Mock(return_value=True)
        vault.delete_secret = Mock(return_value=True)
        return vault

    @pytest.fixture
    def mock_validator(self):
        """Create a mock CredentialValidator."""
        validator = Mock(spec=CredentialValidator)
        # Make validate an async function
        validator.validate = AsyncMock(
            return_value=ValidationResult(
                valid=True,
                credential_type="trading",
                exchange_account_id="test-account",
            )
        )
        return validator

    @pytest.fixture
    def service(self, mock_vault, mock_validator):
        """Create a TenantCredentialService instance."""
        return TenantCredentialService(
            vault_provider=mock_vault,
            validator=mock_validator,
            cache_ttl=300,
            cache_maxsize=1000,
        )

    @pytest.fixture
    def tenant_id(self):
        """Create a test tenant ID."""
        return UUID("12345678-1234-1234-1234-123456789abc")

    def test_build_vault_path(self, service, tenant_id):
        """Test Vault path building."""
        path = service._build_vault_path(tenant_id, "binance", "trading")
        assert path == f"tenants/{tenant_id}/exchanges/binance/trading"

    def test_build_cache_key(self, service, tenant_id):
        """Test cache key building."""
        key = service._build_cache_key(tenant_id, "binance", "trading")
        assert key == f"creds:{tenant_id}:binance:trading"

    @pytest.mark.asyncio
    async def test_get_credentials_cache_hit(self, service, mock_vault, tenant_id):
        """Test getting credentials from cache (cache hit)."""
        # Pre-populate cache
        cached_creds = TenantCredentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="cached_key",
            api_secret="cached_secret",
            credential_type="trading",
        )
        cache_key = service._build_cache_key(tenant_id, "binance", "trading")
        service._cache[cache_key] = cached_creds

        # Get credentials
        result = await service.get_credentials(tenant_id, "binance", "trading")

        assert result == cached_creds
        # Vault should not be called on cache hit
        mock_vault.get_secret.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_credentials_cache_miss_vault_hit(
        self, service, mock_vault, tenant_id
    ):
        """Test getting credentials from Vault (cache miss)."""
        # Mock Vault response
        mock_vault.get_secret.return_value = {
            "api_key": "vault_key",
            "secret": "vault_secret",
            "testnet": False,
            "metadata": {
                "credential_type": "trading",
                "exchange_account_id": "account-123",
            },
        }

        # Get credentials
        result = await service.get_credentials(tenant_id, "binance", "trading")

        assert result is not None
        assert result.tenant_id == tenant_id
        assert result.exchange == "binance"
        assert result.api_key == "vault_key"
        assert result.api_secret == "vault_secret"
        assert result.credential_type == "trading"
        assert result.exchange_account_id == "account-123"

        # Verify Vault was called with correct path
        expected_path = f"tenants/{tenant_id}/exchanges/binance/trading"
        mock_vault.get_secret.assert_called_once_with(expected_path)

        # Verify credentials were cached
        cache_key = service._build_cache_key(tenant_id, "binance", "trading")
        assert cache_key in service._cache

    @pytest.mark.asyncio
    async def test_get_credentials_not_found(self, service, mock_vault, tenant_id):
        """Test getting credentials when not found in Vault."""
        # Mock Vault returning None
        mock_vault.get_secret.return_value = None

        # Get credentials
        result = await service.get_credentials(tenant_id, "binance", "trading")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_credentials_invalid_format(
        self, service, mock_vault, tenant_id
    ):
        """Test getting credentials with invalid Vault data format."""
        # Mock Vault response missing required fields
        mock_vault.get_secret.return_value = {
            "api_key": "vault_key",
            # Missing "secret" field
        }

        # Get credentials
        result = await service.get_credentials(tenant_id, "binance", "trading")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_credentials_success(
        self, service, mock_vault, mock_validator, tenant_id
    ):
        """Test storing credentials successfully."""
        result = await service.store_credentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="new_key",
            secret="new_secret",
            testnet=False,
            validate=True,
            created_by="admin@example.com",
        )

        assert result.valid is True
        assert result.credential_type == "trading"

        # Verify validator was called
        mock_validator.validate.assert_called_once_with(
            exchange="binance",
            api_key="new_key",
            secret="new_secret",
            passphrase=None,
            testnet=False,
        )

        # Verify Vault was called
        assert mock_vault.set_secret.called
        call_args = mock_vault.set_secret.call_args
        vault_path = call_args[0][0]
        secret_data = call_args[0][1]

        assert f"tenants/{tenant_id}/exchanges/binance/trading" == vault_path
        assert secret_data["api_key"] == "new_key"
        assert secret_data["secret"] == "new_secret"
        assert secret_data["metadata"]["created_by"] == "admin@example.com"

    @pytest.mark.asyncio
    async def test_store_credentials_validation_failure(
        self, service, mock_vault, mock_validator, tenant_id
    ):
        """Test storing credentials with validation failure."""
        # Mock validation failure
        mock_validator.validate.return_value = ValidationResult(
            valid=False, error="Invalid API key"
        )

        result = await service.store_credentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="invalid_key",
            secret="invalid_secret",
        )

        assert result.valid is False
        assert result.error == "Invalid API key"

        # Vault should not be called on validation failure
        mock_vault.set_secret.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_credentials_without_validation(
        self, service, mock_vault, mock_validator, tenant_id
    ):
        """Test storing credentials without validation."""
        result = await service.store_credentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="new_key",
            secret="new_secret",
            validate=False,
        )

        assert result.valid is True
        # Validator should not be called
        mock_validator.validate.assert_not_called()
        # Vault should still be called
        assert mock_vault.set_secret.called

    @pytest.mark.asyncio
    async def test_store_credentials_vault_failure(
        self, service, mock_vault, mock_validator, tenant_id
    ):
        """Test storing credentials with Vault write failure."""
        # Mock Vault failure
        mock_vault.set_secret.return_value = False

        result = await service.store_credentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="new_key",
            secret="new_secret",
        )

        assert result.valid is False
        assert "Failed to store in Vault" in result.error

    @pytest.mark.asyncio
    async def test_store_credentials_with_passphrase(
        self, service, mock_vault, mock_validator, tenant_id
    ):
        """Test storing credentials with passphrase (Coinbase style)."""
        result = await service.store_credentials(
            tenant_id=tenant_id,
            exchange="coinbase",
            api_key="new_key",
            secret="new_secret",
            passphrase="test_passphrase",
        )

        assert result.valid is True

        # Verify passphrase was included in Vault data
        call_args = mock_vault.set_secret.call_args
        secret_data = call_args[0][1]
        assert secret_data["passphrase"] == "test_passphrase"

    @pytest.mark.asyncio
    async def test_store_credentials_invalidates_cache(
        self, service, mock_vault, mock_validator, tenant_id
    ):
        """Test that storing credentials invalidates the cache."""
        # Pre-populate cache
        cache_key = service._build_cache_key(tenant_id, "binance", "trading")
        old_creds = TenantCredentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="old_key",
            api_secret="old_secret",
            credential_type="trading",
        )
        service._cache[cache_key] = old_creds

        # Store new credentials
        await service.store_credentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="new_key",
            secret="new_secret",
        )

        # Cache should be invalidated
        assert cache_key not in service._cache

    @pytest.mark.asyncio
    async def test_delete_credentials_success(self, service, mock_vault, tenant_id):
        """Test deleting credentials successfully."""
        result = await service.delete_credentials(tenant_id, "binance", "trading")

        assert result is True
        # Verify Vault was called
        expected_path = f"tenants/{tenant_id}/exchanges/binance/trading"
        mock_vault.delete_secret.assert_called_once_with(expected_path)

    @pytest.mark.asyncio
    async def test_delete_credentials_invalidates_cache(
        self, service, mock_vault, tenant_id
    ):
        """Test that deleting credentials invalidates the cache."""
        # Pre-populate cache
        cache_key = service._build_cache_key(tenant_id, "binance", "trading")
        creds = TenantCredentials(
            tenant_id=tenant_id,
            exchange="binance",
            api_key="key",
            api_secret="secret",
            credential_type="trading",
        )
        service._cache[cache_key] = creds

        # Delete credentials
        await service.delete_credentials(tenant_id, "binance", "trading")

        # Cache should be invalidated
        assert cache_key not in service._cache

    def test_get_cache_stats(self, service, tenant_id):
        """Test getting cache statistics."""
        # Populate cache with some entries
        for i in range(5):
            cache_key = service._build_cache_key(tenant_id, f"exchange{i}", "trading")
            service._cache[cache_key] = Mock()

        stats = service.get_cache_stats()

        assert stats["size"] == 5
        assert stats["maxsize"] == 1000
        assert stats["ttl"] == 300
        assert stats["currsize"] == 5

    def test_clear_cache_all(self, service, tenant_id):
        """Test clearing all cache entries."""
        # Populate cache
        for i in range(10):
            cache_key = service._build_cache_key(tenant_id, f"exchange{i}", "trading")
            service._cache[cache_key] = Mock()

        # Clear all
        count = service.clear_cache()

        assert count == 10
        assert len(service._cache) == 0

    def test_clear_cache_tenant_specific(self, service):
        """Test clearing cache for specific tenant."""
        tenant1 = UUID("11111111-1111-1111-1111-111111111111")
        tenant2 = UUID("22222222-2222-2222-2222-222222222222")

        # Populate cache with entries for two tenants
        for i in range(5):
            key1 = service._build_cache_key(tenant1, f"exchange{i}", "trading")
            key2 = service._build_cache_key(tenant2, f"exchange{i}", "trading")
            service._cache[key1] = Mock()
            service._cache[key2] = Mock()

        # Clear only tenant1
        count = service.clear_cache(tenant_id=tenant1)

        assert count == 5
        assert len(service._cache) == 5
        # Verify tenant2 entries still exist
        for i in range(5):
            key2 = service._build_cache_key(tenant2, f"exchange{i}", "trading")
            assert key2 in service._cache

    def test_default_tenant_id(self):
        """Test DEFAULT_TENANT_ID constant."""
        assert DEFAULT_TENANT_ID == UUID("00000000-0000-0000-0000-000000000000")

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, service, mock_vault, tenant_id):
        """Test thread-safe cache access."""
        import asyncio

        # Mock Vault to return different data
        call_count = [0]

        def get_secret_side_effect(path):
            call_count[0] += 1
            return {
                "api_key": f"key_{call_count[0]}",
                "secret": f"secret_{call_count[0]}",
                "testnet": False,
                "metadata": {"credential_type": "trading"},
            }

        mock_vault.get_secret.side_effect = get_secret_side_effect

        # Make concurrent requests
        tasks = [
            service.get_credentials(tenant_id, "binance", "trading") for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # All results should be the same (from cache after first fetch)
        # Vault should only be called once
        assert mock_vault.get_secret.call_count == 1

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.api_key == first_result.api_key
            assert result.api_secret == first_result.api_secret
