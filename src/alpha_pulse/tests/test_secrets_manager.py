"""
Unit tests for the secure secrets management system.
"""
import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from alpha_pulse.utils.secrets_manager import (
    SecretProvider,
    EnvironmentSecretProvider,
    AWSSecretsManagerProvider,
    HashiCorpVaultProvider,
    LocalEncryptedFileProvider,
    SecretsManager,
    create_secrets_manager
)


class TestEnvironmentSecretProvider:
    """Test cases for environment variable secret provider."""
    
    def test_get_secret_exists(self):
        """Test retrieving existing environment variable."""
        with patch.dict(os.environ, {"ALPHAPULSE_TEST_SECRET": "test_value"}):
            provider = EnvironmentSecretProvider()
            assert provider.get_secret("test_secret") == "test_value"
    
    def test_get_secret_not_exists(self):
        """Test retrieving non-existent environment variable."""
        provider = EnvironmentSecretProvider()
        assert provider.get_secret("nonexistent") is None
    
    def test_custom_prefix(self):
        """Test custom environment variable prefix."""
        with patch.dict(os.environ, {"CUSTOM_TEST": "value"}):
            provider = EnvironmentSecretProvider(prefix="CUSTOM_")
            assert provider.get_secret("test") == "value"
    
    def test_list_secrets(self):
        """Test listing environment secrets."""
        env_vars = {
            "ALPHAPULSE_SECRET1": "value1",
            "ALPHAPULSE_SECRET2": "value2",
            "OTHER_VAR": "value3"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            provider = EnvironmentSecretProvider()
            secrets = provider.list_secrets()
            assert "secret1" in secrets
            assert "secret2" in secrets
            assert "other_var" not in secrets
    
    def test_set_delete_not_supported(self):
        """Test that set and delete operations are not supported."""
        provider = EnvironmentSecretProvider()
        assert provider.set_secret("test", "value") is False
        assert provider.delete_secret("test") is False


class TestAWSSecretsManagerProvider:
    """Test cases for AWS Secrets Manager provider."""
    
    @patch("boto3.client")
    def test_get_secret_success(self, mock_boto_client):
        """Test successful secret retrieval."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        mock_client.get_secret_value.return_value = {
            "SecretString": "test_secret_value"
        }
        
        provider = AWSSecretsManagerProvider()
        result = provider.get_secret("test_secret")
        
        assert result == "test_secret_value"
        mock_client.get_secret_value.assert_called_with(SecretId="test_secret")
    
    @patch("boto3.client")
    def test_get_secret_json(self, mock_boto_client):
        """Test retrieving JSON secret."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        secret_data = {"key": "value", "nested": {"data": 123}}
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps(secret_data)
        }
        
        provider = AWSSecretsManagerProvider()
        result = provider.get_secret("test_json")
        
        assert result == secret_data
    
    @patch("boto3.client")
    def test_caching(self, mock_boto_client):
        """Test secret caching functionality."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        mock_client.get_secret_value.return_value = {
            "SecretString": "cached_value"
        }
        
        provider = AWSSecretsManagerProvider(cache_ttl=60)
        
        # First call
        result1 = provider.get_secret("cached_secret")
        assert result1 == "cached_value"
        assert mock_client.get_secret_value.call_count == 1
        
        # Second call (should use cache)
        result2 = provider.get_secret("cached_secret")
        assert result2 == "cached_value"
        assert mock_client.get_secret_value.call_count == 1
    
    @patch("boto3.client")
    def test_set_secret_new(self, mock_boto_client):
        """Test creating a new secret."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Simulate secret not existing
        mock_client.put_secret_value.side_effect = Exception("ResourceNotFoundException")
        
        provider = AWSSecretsManagerProvider()
        result = provider.set_secret("new_secret", "new_value")
        
        assert result is True
        mock_client.create_secret.assert_called_once()
    
    @patch("boto3.client")
    def test_list_secrets(self, mock_boto_client):
        """Test listing secrets."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"SecretList": [
                {"Name": "secret1"},
                {"Name": "secret2"}
            ]}
        ]
        
        provider = AWSSecretsManagerProvider()
        secrets = provider.list_secrets()
        
        assert secrets == ["secret1", "secret2"]


class TestHashiCorpVaultProvider:
    """Test cases for HashiCorp Vault provider."""
    
    @patch("hvac.Client")
    def test_initialization(self, mock_hvac_client):
        """Test Vault provider initialization."""
        mock_client = Mock()
        mock_hvac_client.return_value = mock_client
        mock_client.is_authenticated.return_value = True
        
        provider = HashiCorpVaultProvider(
            vault_url="http://localhost:8200",
            vault_token="test_token"
        )
        
        assert provider.client == mock_client
        mock_hvac_client.assert_called_with(
            url="http://localhost:8200",
            token="test_token"
        )
    
    @patch("hvac.Client")
    def test_get_secret(self, mock_hvac_client):
        """Test retrieving secret from Vault."""
        mock_client = Mock()
        mock_hvac_client.return_value = mock_client
        mock_client.is_authenticated.return_value = True
        
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"key": "value"}}
        }
        
        provider = HashiCorpVaultProvider(
            vault_url="http://localhost:8200",
            vault_token="test_token"
        )
        
        result = provider.get_secret("test_secret")
        assert result == {"key": "value"}
    
    @patch("hvac.Client")
    def test_set_secret(self, mock_hvac_client):
        """Test storing secret in Vault."""
        mock_client = Mock()
        mock_hvac_client.return_value = mock_client
        mock_client.is_authenticated.return_value = True
        
        provider = HashiCorpVaultProvider(
            vault_url="http://localhost:8200",
            vault_token="test_token"
        )
        
        result = provider.set_secret("test_secret", {"key": "value"})
        assert result is True
        
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once()


class TestLocalEncryptedFileProvider:
    """Test cases for local encrypted file provider."""
    
    def test_encrypt_decrypt_string(self):
        """Test encrypting and decrypting string secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalEncryptedFileProvider(secrets_dir=tmpdir)
            
            # Store secret
            assert provider.set_secret("test_secret", "secret_value") is True
            
            # Retrieve secret
            result = provider.get_secret("test_secret")
            assert result == "secret_value"
    
    def test_encrypt_decrypt_dict(self):
        """Test encrypting and decrypting dictionary secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalEncryptedFileProvider(secrets_dir=tmpdir)
            
            secret_data = {"api_key": "key123", "api_secret": "secret456"}
            
            # Store secret
            assert provider.set_secret("api_creds", secret_data) is True
            
            # Retrieve secret
            result = provider.get_secret("api_creds")
            assert result == secret_data
    
    def test_list_secrets(self):
        """Test listing encrypted secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalEncryptedFileProvider(secrets_dir=tmpdir)
            
            # Store multiple secrets
            provider.set_secret("secret1", "value1")
            provider.set_secret("secret2", "value2")
            
            secrets = provider.list_secrets()
            assert "secret1" in secrets
            assert "secret2" in secrets
    
    def test_delete_secret(self):
        """Test deleting encrypted secret."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalEncryptedFileProvider(secrets_dir=tmpdir)
            
            # Store and delete
            provider.set_secret("temp_secret", "temp_value")
            assert provider.delete_secret("temp_secret") is True
            
            # Verify deleted
            assert provider.get_secret("temp_secret") is None


class TestSecretsManager:
    """Test cases for the main SecretsManager class."""
    
    def test_primary_provider_success(self):
        """Test successful secret retrieval from primary provider."""
        primary = Mock(spec=SecretProvider)
        primary.get_secret.return_value = "primary_value"
        
        manager = SecretsManager(primary)
        result = manager.get_secret("test_secret")
        
        assert result == "primary_value"
        primary.get_secret.assert_called_with("test_secret")
    
    def test_fallback_provider(self):
        """Test fallback to secondary provider."""
        primary = Mock(spec=SecretProvider)
        primary.get_secret.return_value = None
        
        fallback = Mock(spec=SecretProvider)
        fallback.get_secret.return_value = "fallback_value"
        
        manager = SecretsManager(primary, [fallback])
        result = manager.get_secret("test_secret")
        
        assert result == "fallback_value"
        primary.get_secret.assert_called_with("test_secret")
        fallback.get_secret.assert_called_with("test_secret")
    
    def test_audit_logging(self):
        """Test audit log functionality."""
        primary = Mock(spec=SecretProvider)
        primary.get_secret.return_value = "value"
        
        manager = SecretsManager(primary)
        manager.get_secret("test_secret")
        
        audit_log = manager.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0]["action"] == "get"
        assert audit_log[0]["secret_name"] == "test_secret"
        assert audit_log[0]["success"] is True
    
    def test_get_database_credentials(self):
        """Test getting database credentials."""
        primary = Mock(spec=SecretProvider)
        primary.get_secret.return_value = {
            "host": "db.example.com",
            "port": "5432",
            "user": "dbuser",
            "password": "dbpass",
            "database": "mydb"
        }
        
        manager = SecretsManager(primary)
        creds = manager.get_database_credentials()
        
        assert creds["host"] == "db.example.com"
        assert creds["user"] == "dbuser"
        assert creds["password"] == "dbpass"
    
    def test_get_exchange_credentials(self):
        """Test getting exchange credentials."""
        primary = Mock(spec=SecretProvider)
        primary.get_secret.return_value = {
            "api_key": "exchange_key",
            "api_secret": "exchange_secret"
        }
        
        manager = SecretsManager(primary)
        creds = manager.get_exchange_credentials("binance")
        
        assert creds["api_key"] == "exchange_key"
        assert creds["api_secret"] == "exchange_secret"
    
    def test_cache_clearing(self):
        """Test LRU cache clearing."""
        primary = Mock(spec=SecretProvider)
        primary.get_secret.return_value = "cached_value"
        
        manager = SecretsManager(primary)
        
        # First call
        manager.get_secret("cached")
        assert primary.get_secret.call_count == 1
        
        # Second call (cached)
        manager.get_secret("cached")
        assert primary.get_secret.call_count == 1
        
        # Clear cache
        manager.clear_cache()
        
        # Third call (not cached)
        manager.get_secret("cached")
        assert primary.get_secret.call_count == 2


class TestCreateSecretsManager:
    """Test cases for the factory function."""
    
    @patch.dict(os.environ, {"ALPHAPULSE_ENV": "development"})
    def test_development_environment(self):
        """Test creating manager for development environment."""
        manager = create_secrets_manager("development")
        
        assert isinstance(manager.primary_provider, EnvironmentSecretProvider)
        assert len(manager.fallback_providers) == 1
        assert isinstance(manager.fallback_providers[0], LocalEncryptedFileProvider)
    
    @patch("boto3.client")
    @patch.dict(os.environ, {"ALPHAPULSE_ENV": "production", "AWS_REGION": "us-west-2"})
    def test_production_environment(self, mock_boto_client):
        """Test creating manager for production environment."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        manager = create_secrets_manager("production")
        
        assert isinstance(manager.primary_provider, AWSSecretsManagerProvider)
        assert len(manager.fallback_providers) == 1
        assert isinstance(manager.fallback_providers[0], EnvironmentSecretProvider)
    
    @patch("hvac.Client")
    @patch.dict(os.environ, {
        "ALPHAPULSE_ENV": "staging",
        "VAULT_URL": "http://vault:8200",
        "VAULT_TOKEN": "test_token"
    })
    def test_staging_environment(self, mock_hvac_client):
        """Test creating manager for staging environment."""
        mock_client = Mock()
        mock_hvac_client.return_value = mock_client
        mock_client.is_authenticated.return_value = True
        
        manager = create_secrets_manager("staging")
        
        assert isinstance(manager.primary_provider, HashiCorpVaultProvider)
        assert len(manager.fallback_providers) == 1
        assert isinstance(manager.fallback_providers[0], EnvironmentSecretProvider)