"""
Comprehensive tests for the encryption system.
"""
import pytest
import json
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

from alpha_pulse.utils.encryption import (
    EncryptionKeyManager,
    AESCipher,
    SearchableEncryption,
    PerformanceOptimizedCipher,
    encrypt_field,
    decrypt_field,
    EncryptionError,
    DecryptionError
)
from alpha_pulse.models.encrypted_fields import (
    EncryptedString,
    EncryptedInteger,
    EncryptedFloat,
    EncryptedJSON,
    EncryptedSearchableString,
    SearchTokenIndex
)

# Test base for models
TestBase = declarative_base()


class TestEncryptionKeyManager:
    """Test cases for encryption key manager."""
    
    @patch('alpha_pulse.utils.encryption.get_secrets_manager')
    def test_get_or_create_master_key_existing(self, mock_secrets_manager):
        """Test retrieving existing master key."""
        mock_sm = Mock()
        mock_secrets_manager.return_value = mock_sm
        mock_sm.get_secret.return_value = "dGVzdF9tYXN0ZXJfa2V5X2Jhc2U2NA=="  # base64 encoded
        
        manager = EncryptionKeyManager()
        key = manager.get_or_create_master_key()
        
        assert key == b"test_master_key_base64"
        mock_sm.get_secret.assert_called_with("encryption_master_key")
    
    @patch('alpha_pulse.utils.encryption.get_secrets_manager')
    @patch('os.urandom')
    def test_get_or_create_master_key_new(self, mock_urandom, mock_secrets_manager):
        """Test creating new master key."""
        mock_sm = Mock()
        mock_secrets_manager.return_value = mock_sm
        mock_sm.get_secret.return_value = None
        mock_urandom.return_value = b"new_random_master_key_32_bytes!!"
        
        manager = EncryptionKeyManager()
        key = manager.get_or_create_master_key()
        
        assert key == b"new_random_master_key_32_bytes!!"
        mock_sm.primary_provider.set_secret.assert_called_once()
    
    @patch('alpha_pulse.utils.encryption.get_secrets_manager')
    def test_derive_data_key(self, mock_secrets_manager):
        """Test deriving data encryption key."""
        mock_sm = Mock()
        mock_secrets_manager.return_value = mock_sm
        mock_sm.get_secret.side_effect = lambda k: {
            "encryption_master_key": "dGVzdF9tYXN0ZXJfa2V5X2Jhc2U2NA==",
            "encryption_key_version": 1
        }.get(k)
        
        manager = EncryptionKeyManager()
        key, version = manager.derive_data_key("test_context")
        
        assert isinstance(key, bytes)
        assert len(key) == 32  # 256 bits
        assert version == 1
    
    @patch('alpha_pulse.utils.encryption.get_secrets_manager')
    def test_key_rotation(self, mock_secrets_manager):
        """Test key rotation functionality."""
        mock_sm = Mock()
        mock_secrets_manager.return_value = mock_sm
        mock_sm.get_secret.return_value = 1
        
        manager = EncryptionKeyManager()
        new_version = manager.rotate_keys()
        
        assert new_version == 2
        assert mock_sm.primary_provider.set_secret.call_count >= 2  # version and timestamp


class TestAESCipher:
    """Test cases for AES cipher."""
    
    @pytest.fixture
    def cipher(self):
        """Create cipher instance with mocked key manager."""
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            return AESCipher()
    
    def test_encrypt_decrypt_string(self, cipher):
        """Test encrypting and decrypting strings."""
        plaintext = "sensitive data"
        
        encrypted = cipher.encrypt(plaintext)
        
        assert isinstance(encrypted, dict)
        assert "ciphertext" in encrypted
        assert "key_version" in encrypted
        assert "algorithm" in encrypted
        assert encrypted["algorithm"] == "AES-256-GCM"
        
        decrypted = cipher.decrypt(encrypted)
        assert decrypted.decode('utf-8') == plaintext
    
    def test_encrypt_decrypt_bytes(self, cipher):
        """Test encrypting and decrypting bytes."""
        plaintext = b"binary data \x00\x01\x02"
        
        encrypted = cipher.encrypt(plaintext)
        decrypted = cipher.decrypt(encrypted)
        
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_json(self, cipher):
        """Test encrypting and decrypting JSON data."""
        data = {
            "account": "12345",
            "balance": 1000.50,
            "trades": [1, 2, 3]
        }
        
        encrypted = cipher.encrypt_json(data)
        decrypted = cipher.decrypt_json(encrypted)
        
        assert decrypted == data
    
    def test_decrypt_tampered_data(self, cipher):
        """Test decryption fails for tampered data."""
        plaintext = "test data"
        encrypted = cipher.encrypt(plaintext)
        
        # Tamper with ciphertext
        tampered = encrypted.copy()
        tampered["ciphertext"] = tampered["ciphertext"][:-4] + "XXXX"
        
        with pytest.raises(DecryptionError):
            cipher.decrypt(tampered)
    
    def test_encrypt_with_context(self, cipher):
        """Test encryption with different contexts."""
        plaintext = "context test"
        
        encrypted1 = cipher.encrypt(plaintext, context="context1")
        encrypted2 = cipher.encrypt(plaintext, context="context2")
        
        # Different contexts should produce different ciphertexts
        assert encrypted1["ciphertext"] != encrypted2["ciphertext"]
        assert encrypted1["context"] == "context1"
        assert encrypted2["context"] == "context2"


class TestSearchableEncryption:
    """Test cases for searchable encryption."""
    
    @pytest.fixture
    def searchable_cipher(self):
        """Create searchable cipher instance."""
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            return SearchableEncryption()
    
    def test_generate_search_token(self, searchable_cipher):
        """Test search token generation."""
        value = "test@example.com"
        
        token1 = searchable_cipher.generate_search_token(value)
        token2 = searchable_cipher.generate_search_token(value)
        
        # Same value should produce same token (deterministic)
        assert token1 == token2
        assert isinstance(token1, str)
        assert len(token1) > 0
    
    def test_search_token_uniqueness(self, searchable_cipher):
        """Test different values produce different tokens."""
        token1 = searchable_cipher.generate_search_token("value1")
        token2 = searchable_cipher.generate_search_token("value2")
        
        assert token1 != token2
    
    def test_encrypt_searchable(self, searchable_cipher):
        """Test searchable encryption includes search token."""
        value = "searchable@example.com"
        
        encrypted = searchable_cipher.encrypt_searchable(value)
        
        assert "search_token" in encrypted
        assert "ciphertext" in encrypted
        
        # Verify search token matches
        expected_token = searchable_cipher.generate_search_token(value)
        assert encrypted["search_token"] == expected_token


class TestPerformanceOptimizedCipher:
    """Test cases for performance optimized cipher."""
    
    @pytest.fixture
    def optimized_cipher(self):
        """Create optimized cipher instance."""
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            return PerformanceOptimizedCipher()
    
    def test_batch_encrypt(self, optimized_cipher):
        """Test batch encryption."""
        items = ["item1", "item2", "item3", b"binary_item"]
        
        encrypted_items = optimized_cipher.batch_encrypt(items)
        
        assert len(encrypted_items) == len(items)
        for encrypted in encrypted_items:
            assert isinstance(encrypted, dict)
            assert "ciphertext" in encrypted
    
    def test_batch_decrypt(self, optimized_cipher):
        """Test batch decryption."""
        items = ["item1", "item2", "item3"]
        
        # Encrypt first
        encrypted_items = optimized_cipher.batch_encrypt(items)
        
        # Decrypt
        decrypted_items = optimized_cipher.batch_decrypt(encrypted_items)
        
        assert len(decrypted_items) == len(items)
        for i, decrypted in enumerate(decrypted_items):
            assert decrypted.decode('utf-8') == items[i]
    
    def test_batch_decrypt_mixed_versions(self, optimized_cipher):
        """Test batch decryption with different key versions."""
        # Simulate encrypted items with different key versions
        encrypted_items = [
            {"ciphertext": "test1", "key_version": 1, "context": "default"},
            {"ciphertext": "test2", "key_version": 2, "context": "default"},
            {"ciphertext": "test3", "key_version": 1, "context": "other"}
        ]
        
        # Mock the decryption to avoid actual crypto operations
        with patch.object(optimized_cipher, '_get_cached_key') as mock_key:
            mock_key.return_value = b"test_key" * 4  # 32 bytes
            
            # This will fail in real decryption but tests the grouping logic
            try:
                optimized_cipher.batch_decrypt(encrypted_items)
            except:
                pass
            
            # Verify keys were requested for each unique (version, context) pair
            assert mock_key.call_count >= 3


class TestEncryptedFields:
    """Test cases for SQLAlchemy encrypted fields."""
    
    @pytest.fixture
    def test_db(self):
        """Create in-memory test database."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        
        # Define test model
        class TestModel(TestBase):
            __tablename__ = "test_model"
            
            id = Column(Integer, primary_key=True)
            encrypted_string = Column(EncryptedString())
            encrypted_int = Column(EncryptedInteger())
            encrypted_float = Column(EncryptedFloat())
            encrypted_json = Column(EncryptedJSON())
            searchable_email = Column(EncryptedSearchableString())
            email_search = Column(SearchTokenIndex("searchable_email"))
        
        TestBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        
        return Session(), TestModel
    
    def test_encrypted_string_field(self, test_db):
        """Test encrypted string field operations."""
        session, TestModel = test_db
        
        # Create record
        record = TestModel(
            encrypted_string="sensitive string data"
        )
        session.add(record)
        session.commit()
        
        # Retrieve and verify
        retrieved = session.query(TestModel).first()
        assert retrieved.encrypted_string == "sensitive string data"
        
        # Verify data is encrypted in database
        raw_data = session.execute(
            "SELECT encrypted_string FROM test_model"
        ).first()
        assert raw_data[0] != "sensitive string data"
        assert raw_data[0].startswith('{"')  # JSON encrypted data
    
    def test_encrypted_numeric_fields(self, test_db):
        """Test encrypted numeric fields."""
        session, TestModel = test_db
        
        record = TestModel(
            encrypted_int=42,
            encrypted_float=3.14159
        )
        session.add(record)
        session.commit()
        
        retrieved = session.query(TestModel).first()
        assert retrieved.encrypted_int == 42
        assert abs(retrieved.encrypted_float - 3.14159) < 0.00001
    
    def test_encrypted_json_field(self, test_db):
        """Test encrypted JSON field."""
        session, TestModel = test_db
        
        data = {
            "settings": {"theme": "dark", "notifications": True},
            "preferences": ["option1", "option2"]
        }
        
        record = TestModel(encrypted_json=data)
        session.add(record)
        session.commit()
        
        retrieved = session.query(TestModel).first()
        assert retrieved.encrypted_json == data
    
    def test_searchable_field(self, test_db):
        """Test searchable encrypted field."""
        session, TestModel = test_db
        
        email = "test@example.com"
        record = TestModel(searchable_email=email)
        
        # Search token should be set automatically
        session.add(record)
        session.commit()
        
        # Generate expected search token
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            from alpha_pulse.utils.encryption import get_searchable_cipher
            cipher = get_searchable_cipher()
            expected_token = cipher.generate_search_token(email, "search")
        
        # Query by search token
        found = session.query(TestModel).filter(
            TestModel.email_search == expected_token
        ).first()
        
        assert found is not None
        assert found.searchable_email == email


class TestEncryptionHelpers:
    """Test encryption helper functions."""
    
    def test_encrypt_field_string(self):
        """Test encrypt_field with string."""
        with patch('alpha_pulse.utils.encryption.get_cipher') as mock_cipher:
            mock_cipher.return_value.encrypt.return_value = {"encrypted": "data"}
            
            result = encrypt_field("test string")
            
            assert result == {"encrypted": "data"}
            mock_cipher.return_value.encrypt.assert_called_with("test string", "default")
    
    def test_encrypt_field_searchable(self):
        """Test encrypt_field with searchable option."""
        with patch('alpha_pulse.utils.encryption.get_searchable_cipher') as mock_cipher:
            mock_cipher.return_value.encrypt_searchable.return_value = {
                "encrypted": "data",
                "search_token": "token"
            }
            
            result = encrypt_field("searchable", searchable=True)
            
            assert result["search_token"] == "token"
    
    def test_decrypt_field_string(self):
        """Test decrypt_field with string data."""
        with patch('alpha_pulse.utils.encryption.get_cipher') as mock_cipher:
            mock_cipher.return_value.decrypt.return_value = b"decrypted string"
            
            encrypted_data = {"ciphertext": "encrypted"}
            result = decrypt_field(encrypted_data)
            
            assert result == "decrypted string"
    
    def test_decrypt_field_json(self):
        """Test decrypt_field with JSON data."""
        with patch('alpha_pulse.utils.encryption.get_cipher') as mock_cipher:
            mock_cipher.return_value.decrypt_json.return_value = {"key": "value"}
            
            encrypted_data = {"ciphertext": "encrypted"}
            result = decrypt_field(encrypted_data)
            
            assert result == {"key": "value"}


class TestEncryptionPerformance:
    """Performance tests for encryption operations."""
    
    @pytest.mark.performance
    def test_encryption_performance(self):
        """Test encryption performance metrics."""
        import time
        
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            cipher = AESCipher()
            
            # Test single encryption performance
            data = "x" * 1000  # 1KB of data
            
            start = time.time()
            for _ in range(100):
                encrypted = cipher.encrypt(data)
            single_time = time.time() - start
            
            # Should encrypt 100 x 1KB in under 1 second
            assert single_time < 1.0
            
            # Test batch encryption performance
            optimized = PerformanceOptimizedCipher()
            items = [f"item_{i}" for i in range(1000)]
            
            start = time.time()
            encrypted_items = optimized.batch_encrypt(items)
            batch_time = time.time() - start
            
            # Batch should be more efficient
            assert batch_time < 2.0  # 1000 items in under 2 seconds
    
    @pytest.mark.performance
    def test_key_caching_performance(self):
        """Test key derivation caching improves performance."""
        import time
        
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            manager = EncryptionKeyManager()
            
            # First derivation (no cache)
            start = time.time()
            key1, _ = manager.derive_data_key("test_context")
            first_time = time.time() - start
            
            # Subsequent derivations (cached)
            start = time.time()
            for _ in range(100):
                key2, _ = manager.derive_data_key("test_context")
            cached_time = time.time() - start
            
            # Cached calls should be significantly faster
            assert cached_time < first_time * 10  # At least 10x improvement


class TestEncryptionSecurity:
    """Security-focused tests for encryption."""
    
    def test_unique_ivs(self):
        """Test that each encryption uses unique IV."""
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            cipher = AESCipher()
            
            # Encrypt same data multiple times
            plaintext = "same data"
            encrypted1 = cipher.encrypt(plaintext)
            encrypted2 = cipher.encrypt(plaintext)
            
            # Ciphertexts should be different due to different IVs
            assert encrypted1["ciphertext"] != encrypted2["ciphertext"]
    
    def test_authenticated_encryption(self):
        """Test that authentication tag prevents tampering."""
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            cipher = AESCipher()
            
            encrypted = cipher.encrypt("test data")
            
            # Decode and tamper with the tag
            import base64
            encrypted_bytes = base64.b64decode(encrypted["ciphertext"])
            
            # Tamper with authentication tag (bytes 12-28)
            tampered_bytes = encrypted_bytes[:12] + b'x' * 16 + encrypted_bytes[28:]
            tampered_b64 = base64.b64encode(tampered_bytes).decode('utf-8')
            
            tampered_data = encrypted.copy()
            tampered_data["ciphertext"] = tampered_b64
            
            # Should fail authentication
            with pytest.raises(DecryptionError):
                cipher.decrypt(tampered_data)
    
    def test_key_rotation_isolation(self):
        """Test that data encrypted with old keys can still be decrypted."""
        with patch('alpha_pulse.utils.encryption.get_secrets_manager'):
            manager = EncryptionKeyManager()
            cipher = AESCipher(manager)
            
            # Encrypt with version 1
            data_v1 = cipher.encrypt("version 1 data")
            assert data_v1["key_version"] == 1
            
            # Rotate keys
            manager.rotate_keys()
            
            # Encrypt with version 2
            data_v2 = cipher.encrypt("version 2 data")
            assert data_v2["key_version"] == 2
            
            # Both should still decrypt correctly
            assert cipher.decrypt(data_v1).decode('utf-8') == "version 1 data"
            assert cipher.decrypt(data_v2).decode('utf-8') == "version 2 data"