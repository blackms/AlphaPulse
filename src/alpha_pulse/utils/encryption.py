"""
Enterprise-grade encryption utilities for AlphaPulse.

This module provides AES-256-GCM encryption for sensitive data with
key management, rotation, and performance optimization.
"""
import os
import base64
import json
import hashlib
from typing import Any, Union, Optional, Tuple, Dict
from datetime import datetime, timedelta
from functools import lru_cache
import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import logging

from ..config.secure_settings import get_secrets_manager

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Base exception for encryption-related errors."""
    pass


class KeyRotationError(EncryptionError):
    """Exception raised during key rotation operations."""
    pass


class DecryptionError(EncryptionError):
    """Exception raised when decryption fails."""
    pass


class EncryptionKeyManager:
    """
    Manages encryption keys with support for rotation and versioning.
    """
    
    def __init__(self, key_storage_backend: str = "secrets_manager"):
        self.backend = key_storage_backend
        self.secrets_manager = get_secrets_manager()
        self._key_cache = {}
        self._active_key_version = None
        
    def get_or_create_master_key(self) -> bytes:
        """Get or create the master encryption key."""
        master_key = self.secrets_manager.get_secret("encryption_master_key")
        
        if not master_key:
            # Generate a new master key
            master_key = os.urandom(32)  # 256 bits
            master_key_b64 = base64.b64encode(master_key).decode('utf-8')
            
            # Store in secrets manager
            self.secrets_manager.primary_provider.set_secret(
                "encryption_master_key",
                master_key_b64
            )
            
            logger.info("Generated new master encryption key")
        else:
            if isinstance(master_key, str):
                master_key = base64.b64decode(master_key)
        
        return master_key
    
    def derive_data_key(self, context: str, key_version: int = None) -> Tuple[bytes, int]:
        """
        Derive a data encryption key from the master key.
        
        Args:
            context: Context string for key derivation (e.g., "trading_data")
            key_version: Specific key version to use
            
        Returns:
            Tuple of (derived_key, key_version)
        """
        if key_version is None:
            key_version = self.get_current_key_version()
        
        cache_key = f"{context}:{key_version}"
        if cache_key in self._key_cache:
            return self._key_cache[cache_key], key_version
        
        master_key = self.get_or_create_master_key()
        
        # Use PBKDF2 to derive a key for the specific context
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=f"{context}:v{key_version}".encode(),
            iterations=100000,
            backend=default_backend()
        )
        
        derived_key = kdf.derive(master_key)
        self._key_cache[cache_key] = derived_key
        
        return derived_key, key_version
    
    def get_current_key_version(self) -> int:
        """Get the current active key version."""
        if self._active_key_version is not None:
            return self._active_key_version
        
        version = self.secrets_manager.get_secret("encryption_key_version")
        if version is None:
            version = 1
            self.secrets_manager.primary_provider.set_secret(
                "encryption_key_version", 
                version
            )
        
        self._active_key_version = int(version)
        return self._active_key_version
    
    def rotate_keys(self) -> int:
        """
        Rotate to a new key version.
        
        Returns:
            New key version number
        """
        current_version = self.get_current_key_version()
        new_version = current_version + 1
        
        # Store the new version
        self.secrets_manager.primary_provider.set_secret(
            "encryption_key_version",
            new_version
        )
        
        # Record rotation timestamp
        self.secrets_manager.primary_provider.set_secret(
            f"key_rotation_timestamp_v{new_version}",
            datetime.utcnow().isoformat()
        )
        
        # Clear cache
        self._key_cache.clear()
        self._active_key_version = new_version
        
        logger.info(f"Rotated encryption keys from version {current_version} to {new_version}")
        return new_version
    
    def get_key_metadata(self, key_version: int) -> Dict[str, Any]:
        """Get metadata about a specific key version."""
        rotation_timestamp = self.secrets_manager.get_secret(
            f"key_rotation_timestamp_v{key_version}"
        )
        
        return {
            "version": key_version,
            "rotation_timestamp": rotation_timestamp,
            "is_active": key_version == self.get_current_key_version()
        }


class AESCipher:
    """
    AES-256-GCM cipher for encrypting/decrypting data.
    """
    
    def __init__(self, key_manager: EncryptionKeyManager = None):
        self.key_manager = key_manager or EncryptionKeyManager()
        
    def encrypt(self, plaintext: Union[str, bytes], context: str = "default") -> Dict[str, Any]:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            plaintext: Data to encrypt (string or bytes)
            context: Encryption context for key derivation
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        # Get encryption key
        key, key_version = self.key_manager.derive_data_key(context)
        
        # Generate random IV (96 bits for GCM)
        iv = os.urandom(12)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Get authentication tag
        tag = encryptor.tag
        
        # Combine IV + tag + ciphertext
        encrypted_data = iv + tag + ciphertext
        
        # Encode for storage
        encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
        
        return {
            "ciphertext": encrypted_b64,
            "key_version": key_version,
            "context": context,
            "algorithm": "AES-256-GCM"
        }
    
    def decrypt(self, encrypted_data: Dict[str, Any]) -> bytes:
        """
        Decrypt data encrypted with AES-256-GCM.
        
        Args:
            encrypted_data: Dictionary containing encrypted data and metadata
            
        Returns:
            Decrypted data as bytes
        """
        try:
            # Extract components
            ciphertext_b64 = encrypted_data["ciphertext"]
            key_version = encrypted_data["key_version"]
            context = encrypted_data.get("context", "default")
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(ciphertext_b64)
            
            # Extract IV, tag, and ciphertext
            iv = encrypted_bytes[:12]
            tag = encrypted_bytes[12:28]
            ciphertext = encrypted_bytes[28:]
            
            # Get decryption key
            key, _ = self.key_manager.derive_data_key(context, key_version)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise DecryptionError(f"Failed to decrypt data: {str(e)}")
    
    def encrypt_json(self, data: Any, context: str = "default") -> Dict[str, Any]:
        """Encrypt JSON-serializable data."""
        json_str = json.dumps(data, separators=(',', ':'))
        return self.encrypt(json_str, context)
    
    def decrypt_json(self, encrypted_data: Dict[str, Any]) -> Any:
        """Decrypt JSON data."""
        plaintext = self.decrypt(encrypted_data)
        return json.loads(plaintext.decode('utf-8'))


class SearchableEncryption:
    """
    Provides searchable encryption using deterministic encryption for indexing.
    """
    
    def __init__(self, key_manager: EncryptionKeyManager = None):
        self.key_manager = key_manager or EncryptionKeyManager()
        
    def generate_search_token(self, plaintext: str, context: str = "search") -> str:
        """
        Generate a deterministic search token for exact matches.
        
        Args:
            plaintext: Value to create search token for
            context: Context for key derivation
            
        Returns:
            Base64-encoded search token
        """
        key, _ = self.key_manager.derive_data_key(context)
        
        # Use HMAC for deterministic token generation
        h = hashlib.blake2b(key=key, digest_size=32)
        h.update(plaintext.encode('utf-8'))
        
        token = h.digest()
        return base64.b64encode(token).decode('utf-8')
    
    def encrypt_searchable(self, plaintext: str, context: str = "default") -> Dict[str, Any]:
        """
        Encrypt data with searchable capabilities.
        
        Returns dictionary with both encrypted data and search token.
        """
        cipher = AESCipher(self.key_manager)
        
        # Regular encryption
        encrypted = cipher.encrypt(plaintext, context)
        
        # Add search token
        encrypted["search_token"] = self.generate_search_token(plaintext, context)
        
        return encrypted


class PerformanceOptimizedCipher:
    """
    Optimized cipher with caching and batch operations.
    """
    
    def __init__(self, key_manager: EncryptionKeyManager = None):
        self.key_manager = key_manager or EncryptionKeyManager()
        self.cipher = AESCipher(key_manager)
        self._key_cache = {}
        
    @lru_cache(maxsize=1000)
    def _get_cached_key(self, context: str, version: int) -> bytes:
        """Get cached encryption key."""
        key, _ = self.key_manager.derive_data_key(context, version)
        return key
    
    def batch_encrypt(self, items: list, context: str = "default") -> list:
        """
        Encrypt multiple items efficiently.
        
        Args:
            items: List of items to encrypt
            context: Encryption context
            
        Returns:
            List of encrypted items
        """
        key, key_version = self.key_manager.derive_data_key(context)
        encrypted_items = []
        
        for item in items:
            if isinstance(item, str):
                item = item.encode('utf-8')
            
            # Generate IV
            iv = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Encrypt
            ciphertext = encryptor.update(item) + encryptor.finalize()
            tag = encryptor.tag
            
            # Combine and encode
            encrypted_data = iv + tag + ciphertext
            encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
            
            encrypted_items.append({
                "ciphertext": encrypted_b64,
                "key_version": key_version,
                "context": context,
                "algorithm": "AES-256-GCM"
            })
        
        return encrypted_items
    
    def batch_decrypt(self, encrypted_items: list) -> list:
        """
        Decrypt multiple items efficiently.
        
        Args:
            encrypted_items: List of encrypted items
            
        Returns:
            List of decrypted items
        """
        decrypted_items = []
        
        # Group by key version and context for efficiency
        grouped = {}
        for i, item in enumerate(encrypted_items):
            key = (item["key_version"], item.get("context", "default"))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((i, item))
        
        # Process each group
        results = [None] * len(encrypted_items)
        
        for (key_version, context), items in grouped.items():
            key = self._get_cached_key(context, key_version)
            
            for idx, encrypted_data in items:
                try:
                    # Decode
                    encrypted_bytes = base64.b64decode(encrypted_data["ciphertext"])
                    
                    # Extract components
                    iv = encrypted_bytes[:12]
                    tag = encrypted_bytes[12:28]
                    ciphertext = encrypted_bytes[28:]
                    
                    # Decrypt
                    cipher = Cipher(
                        algorithms.AES(key),
                        modes.GCM(iv, tag),
                        backend=default_backend()
                    )
                    decryptor = cipher.decryptor()
                    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
                    
                    results[idx] = plaintext
                    
                except Exception as e:
                    logger.error(f"Failed to decrypt item {idx}: {str(e)}")
                    results[idx] = None
        
        return results


# Global cipher instance
_cipher: Optional[AESCipher] = None
_searchable_cipher: Optional[SearchableEncryption] = None
_optimized_cipher: Optional[PerformanceOptimizedCipher] = None


def get_cipher() -> AESCipher:
    """Get or create global cipher instance."""
    global _cipher
    if _cipher is None:
        _cipher = AESCipher()
    return _cipher


def get_searchable_cipher() -> SearchableEncryption:
    """Get or create global searchable cipher instance."""
    global _searchable_cipher
    if _searchable_cipher is None:
        _searchable_cipher = SearchableEncryption()
    return _searchable_cipher


def get_optimized_cipher() -> PerformanceOptimizedCipher:
    """Get or create global optimized cipher instance."""
    global _optimized_cipher
    if _optimized_cipher is None:
        _optimized_cipher = PerformanceOptimizedCipher()
    return _optimized_cipher


def encrypt_field(value: Any, searchable: bool = False, context: str = "default") -> Dict[str, Any]:
    """
    Convenience function to encrypt a field value.
    
    Args:
        value: Value to encrypt
        searchable: Whether to make the field searchable
        context: Encryption context
        
    Returns:
        Encrypted data dictionary
    """
    if searchable:
        cipher = get_searchable_cipher()
        return cipher.encrypt_searchable(str(value), context)
    else:
        cipher = get_cipher()
        if isinstance(value, (dict, list)):
            return cipher.encrypt_json(value, context)
        else:
            return cipher.encrypt(str(value), context)


def decrypt_field(encrypted_data: Dict[str, Any]) -> Any:
    """
    Convenience function to decrypt a field value.
    
    Args:
        encrypted_data: Encrypted data dictionary
        
    Returns:
        Decrypted value
    """
    cipher = get_cipher()
    
    # Check if it's JSON data
    try:
        return cipher.decrypt_json(encrypted_data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Not JSON, return as string
        plaintext = cipher.decrypt(encrypted_data)
        return plaintext.decode('utf-8')


# Export main classes and functions
__all__ = [
    "EncryptionError",
    "KeyRotationError", 
    "DecryptionError",
    "EncryptionKeyManager",
    "AESCipher",
    "SearchableEncryption",
    "PerformanceOptimizedCipher",
    "get_cipher",
    "get_searchable_cipher",
    "get_optimized_cipher",
    "encrypt_field",
    "decrypt_field"
]