"""
Secure secrets management for AlphaPulse trading system.

This module provides a unified interface for managing secrets across different
environments (development, staging, production) with support for multiple
secret management backends.
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from functools import lru_cache
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
import hvac
from cryptography.fernet import Fernet
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretProvider(ABC):
    """Abstract base class for secret providers."""
    
    @abstractmethod
    def get_secret(self, secret_name: str) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve a secret by name."""
        pass
    
    @abstractmethod
    def set_secret(self, secret_name: str, secret_value: Union[str, Dict[str, Any]]) -> bool:
        """Store a secret."""
        pass
    
    @abstractmethod
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret."""
        pass
    
    @abstractmethod
    def list_secrets(self) -> list:
        """List all available secrets."""
        pass


class EnvironmentSecretProvider(SecretProvider):
    """Secret provider that uses environment variables."""
    
    def __init__(self, prefix: str = "ALPHAPULSE_"):
        self.prefix = prefix
        logger.info(f"Initialized EnvironmentSecretProvider with prefix: {prefix}")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from environment variables."""
        env_var = f"{self.prefix}{secret_name.upper()}"
        value = os.environ.get(env_var)
        if value:
            logger.debug(f"Retrieved secret {secret_name} from environment")
        else:
            logger.warning(f"Secret {secret_name} not found in environment")
        return value
    
    def set_secret(self, secret_name: str, secret_value: Union[str, Dict[str, Any]]) -> bool:
        """Environment variables cannot be set at runtime."""
        logger.warning("Cannot set environment variables at runtime")
        return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Environment variables cannot be deleted at runtime."""
        logger.warning("Cannot delete environment variables at runtime")
        return False
    
    def list_secrets(self) -> list:
        """List all secrets with the configured prefix."""
        return [
            key.replace(self.prefix, "").lower()
            for key in os.environ.keys()
            if key.startswith(self.prefix)
        ]


class AWSSecretsManagerProvider(SecretProvider):
    """Secret provider using AWS Secrets Manager."""
    
    def __init__(self, region_name: str = "us-east-1", cache_ttl: int = 300):
        self.client = boto3.client("secretsmanager", region_name=region_name)
        self.region_name = region_name
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_timestamps = {}
        logger.info(f"Initialized AWSSecretsManagerProvider in region: {region_name}")
    
    def _is_cache_valid(self, secret_name: str) -> bool:
        """Check if cached secret is still valid."""
        if secret_name not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[secret_name]
        return datetime.now() - cache_time < timedelta(seconds=self.cache_ttl)
    
    def get_secret(self, secret_name: str) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve secret from AWS Secrets Manager with caching."""
        # Check cache first
        if self._is_cache_valid(secret_name):
            logger.debug(f"Returning cached secret: {secret_name}")
            return self._cache[secret_name]
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            
            if "SecretString" in response:
                secret = response["SecretString"]
                # Try to parse as JSON
                try:
                    secret = json.loads(secret)
                except json.JSONDecodeError:
                    pass
            else:
                # Binary secret
                secret = response["SecretBinary"]
            
            # Update cache
            self._cache[secret_name] = secret
            self._cache_timestamps[secret_name] = datetime.now()
            
            logger.info(f"Successfully retrieved secret: {secret_name}")
            return secret
            
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.error(f"Secret {secret_name} not found")
            else:
                logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
            return None
    
    def set_secret(self, secret_name: str, secret_value: Union[str, Dict[str, Any]]) -> bool:
        """Create or update a secret in AWS Secrets Manager."""
        try:
            # Convert dict to JSON string if needed
            if isinstance(secret_value, dict):
                secret_string = json.dumps(secret_value)
            else:
                secret_string = str(secret_value)
            
            # Try to update existing secret
            try:
                self.client.put_secret_value(
                    SecretId=secret_name,
                    SecretString=secret_string
                )
                logger.info(f"Updated existing secret: {secret_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    # Create new secret
                    self.client.create_secret(
                        Name=secret_name,
                        SecretString=secret_string
                    )
                    logger.info(f"Created new secret: {secret_name}")
                else:
                    raise
            
            # Invalidate cache
            if secret_name in self._cache:
                del self._cache[secret_name]
                del self._cache_timestamps[secret_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting secret {secret_name}: {str(e)}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from AWS Secrets Manager."""
        try:
            self.client.delete_secret(
                SecretId=secret_name,
                ForceDeleteWithoutRecovery=False
            )
            
            # Remove from cache
            if secret_name in self._cache:
                del self._cache[secret_name]
                del self._cache_timestamps[secret_name]
            
            logger.info(f"Deleted secret: {secret_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting secret {secret_name}: {str(e)}")
            return False
    
    def list_secrets(self) -> list:
        """List all secrets in AWS Secrets Manager."""
        try:
            secrets = []
            paginator = self.client.get_paginator("list_secrets")
            
            for page in paginator.paginate():
                for secret in page["SecretList"]:
                    secrets.append(secret["Name"])
            
            return secrets
            
        except Exception as e:
            logger.error(f"Error listing secrets: {str(e)}")
            return []


class HashiCorpVaultProvider(SecretProvider):
    """Secret provider using HashiCorp Vault."""
    
    def __init__(self, vault_url: str, vault_token: str, mount_point: str = "secret"):
        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.mount_point = mount_point
        
        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed")
        
        logger.info(f"Initialized HashiCorpVaultProvider with URL: {vault_url}")
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_name,
                mount_point=self.mount_point
            )
            logger.info(f"Successfully retrieved secret: {secret_name}")
            return response["data"]["data"]
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
            return None
    
    def set_secret(self, secret_name: str, secret_value: Union[str, Dict[str, Any]]) -> bool:
        """Store secret in Vault."""
        try:
            # Ensure value is a dict for Vault KV v2
            if isinstance(secret_value, str):
                secret_data = {"value": secret_value}
            else:
                secret_data = secret_value
            
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_name,
                secret=secret_data,
                mount_point=self.mount_point
            )
            logger.info(f"Successfully stored secret: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Error storing secret {secret_name}: {str(e)}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete secret from Vault."""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=secret_name,
                mount_point=self.mount_point
            )
            logger.info(f"Successfully deleted secret: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting secret {secret_name}: {str(e)}")
            return False
    
    def list_secrets(self) -> list:
        """List all secrets in Vault."""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                mount_point=self.mount_point
            )
            return response["data"]["keys"]
        except Exception as e:
            logger.error(f"Error listing secrets: {str(e)}")
            return []


class LocalEncryptedFileProvider(SecretProvider):
    """Secret provider using locally encrypted files (for development only)."""
    
    def __init__(self, secrets_dir: str = ".secrets", encryption_key: Optional[str] = None):
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(exist_ok=True)
        
        # Generate or use provided encryption key
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        else:
            # Generate a new key if not provided
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logger.warning("Generated new encryption key. Store this safely!")
            logger.warning(f"Encryption key: {key.decode()}")
        
        logger.info(f"Initialized LocalEncryptedFileProvider with directory: {secrets_dir}")
    
    def _get_secret_path(self, secret_name: str) -> Path:
        """Get the file path for a secret."""
        return self.secrets_dir / f"{secret_name}.enc"
    
    def get_secret(self, secret_name: str) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve and decrypt secret from file."""
        secret_path = self._get_secret_path(secret_name)
        
        if not secret_path.exists():
            logger.warning(f"Secret file not found: {secret_name}")
            return None
        
        try:
            with open(secret_path, "rb") as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data).decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted_data)
            except json.JSONDecodeError:
                return decrypted_data
                
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
            return None
    
    def set_secret(self, secret_name: str, secret_value: Union[str, Dict[str, Any]]) -> bool:
        """Encrypt and store secret to file."""
        try:
            # Convert to string if needed
            if isinstance(secret_value, dict):
                data = json.dumps(secret_value)
            else:
                data = str(secret_value)
            
            # Encrypt the data
            encrypted_data = self.cipher.encrypt(data.encode())
            
            # Write to file
            secret_path = self._get_secret_path(secret_name)
            with open(secret_path, "wb") as f:
                f.write(encrypted_data)
            
            logger.info(f"Successfully stored encrypted secret: {secret_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing secret {secret_name}: {str(e)}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete secret file."""
        try:
            secret_path = self._get_secret_path(secret_name)
            if secret_path.exists():
                secret_path.unlink()
                logger.info(f"Successfully deleted secret: {secret_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting secret {secret_name}: {str(e)}")
            return False
    
    def list_secrets(self) -> list:
        """List all encrypted secret files."""
        return [
            f.stem for f in self.secrets_dir.glob("*.enc")
        ]


class SecretsManager:
    """
    Main secrets manager that provides a unified interface and handles
    fallback mechanisms across different secret providers.
    """
    
    def __init__(self, primary_provider: SecretProvider, fallback_providers: Optional[list] = None):
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self._audit_log = []
        logger.info("Initialized SecretsManager")
    
    def _log_access(self, action: str, secret_name: str, success: bool, provider: str):
        """Log secret access for auditing."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "secret_name": secret_name,
            "success": success,
            "provider": provider
        }
        self._audit_log.append(log_entry)
        logger.info(f"Secret access: {log_entry}")
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Get a secret with fallback mechanism.
        
        Args:
            secret_name: Name of the secret to retrieve
            
        Returns:
            Secret value or None if not found
        """
        # Try primary provider first
        try:
            secret = self.primary_provider.get_secret(secret_name)
            if secret is not None:
                self._log_access("get", secret_name, True, type(self.primary_provider).__name__)
                return secret
        except Exception as e:
            logger.error(f"Primary provider failed: {str(e)}")
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                secret = provider.get_secret(secret_name)
                if secret is not None:
                    self._log_access("get", secret_name, True, type(provider).__name__)
                    logger.warning(f"Used fallback provider {type(provider).__name__} for {secret_name}")
                    return secret
            except Exception as e:
                logger.error(f"Fallback provider {type(provider).__name__} failed: {str(e)}")
        
        self._log_access("get", secret_name, False, "none")
        logger.error(f"Failed to retrieve secret {secret_name} from any provider")
        return None
    
    def get_database_credentials(self) -> Dict[str, str]:
        """Get database connection credentials."""
        creds = self.get_secret("database_credentials")
        if isinstance(creds, dict):
            return creds
        
        # Fallback to individual secrets
        return {
            "host": self.get_secret("db_host") or "localhost",
            "port": self.get_secret("db_port") or "5432",
            "user": self.get_secret("db_user") or "",
            "password": self.get_secret("db_password") or "",
            "database": self.get_secret("db_name") or "alphapulse"
        }
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service."""
        # Try service-specific key first
        key = self.get_secret(f"{service}_api_key")
        if key:
            return key
        
        # Try generic format
        return self.get_secret(f"api_key_{service}")
    
    def get_exchange_credentials(self, exchange: str) -> Dict[str, str]:
        """Get exchange API credentials."""
        creds = self.get_secret(f"{exchange}_credentials")
        if isinstance(creds, dict):
            return creds
        
        # Fallback to individual secrets
        return {
            "api_key": self.get_secret(f"{exchange}_api_key") or "",
            "api_secret": self.get_secret(f"{exchange}_api_secret") or "",
            "passphrase": self.get_secret(f"{exchange}_passphrase") or ""
        }
    
    def get_jwt_secret(self) -> str:
        """Get JWT signing secret."""
        secret = self.get_secret("jwt_secret")
        if not secret:
            raise ValueError("JWT secret not found - this is a critical security issue!")
        return str(secret)
    
    def get_encryption_key(self) -> bytes:
        """Get AES encryption key."""
        key = self.get_secret("encryption_key")
        if not key:
            raise ValueError("Encryption key not found - this is a critical security issue!")
        
        if isinstance(key, str):
            return key.encode()
        return key
    
    def get_audit_log(self) -> list:
        """Get the audit log of secret accesses."""
        return self._audit_log.copy()
    
    def clear_cache(self):
        """Clear the LRU cache for secrets."""
        self.get_secret.cache_clear()
        logger.info("Cleared secrets cache")


def create_secrets_manager(environment: str = None) -> SecretsManager:
    """
    Factory function to create appropriate secrets manager based on environment.
    
    Args:
        environment: Environment name (development, staging, production)
        
    Returns:
        Configured SecretsManager instance
    """
    if environment is None:
        environment = os.environ.get("ALPHAPULSE_ENV", "development").lower()
    
    logger.info(f"Creating secrets manager for environment: {environment}")
    
    if environment == "production":
        # Production uses AWS Secrets Manager with environment variable fallback
        primary = AWSSecretsManagerProvider(
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            cache_ttl=300
        )
        fallback = [EnvironmentSecretProvider()]
        
    elif environment == "staging":
        # Staging uses HashiCorp Vault with environment variable fallback
        vault_url = os.environ.get("VAULT_URL", "http://localhost:8200")
        vault_token = os.environ.get("VAULT_TOKEN", "")
        
        if vault_token:
            primary = HashiCorpVaultProvider(vault_url, vault_token)
        else:
            logger.warning("Vault token not found, using environment variables")
            primary = EnvironmentSecretProvider()
        
        fallback = [EnvironmentSecretProvider()]
        
    else:
        # Development uses environment variables with local encrypted file fallback
        primary = EnvironmentSecretProvider()
        
        # Use local encrypted files as fallback
        encryption_key = os.environ.get("ALPHAPULSE_ENCRYPTION_KEY")
        fallback = [LocalEncryptedFileProvider(encryption_key=encryption_key)]
    
    return SecretsManager(primary, fallback)