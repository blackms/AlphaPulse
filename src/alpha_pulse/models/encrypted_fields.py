"""
SQLAlchemy encrypted field types for transparent encryption/decryption.

This module provides custom SQLAlchemy types that automatically encrypt
data before storing and decrypt when retrieving from the database.
"""
import json
from typing import Any, Optional, Dict, Type
from sqlalchemy import TypeDecorator, String, Text, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
import logging

from ..utils.encryption import (
    encrypt_field,
    decrypt_field,
    get_searchable_cipher,
    EncryptionError,
    DecryptionError
)

logger = logging.getLogger(__name__)


class EncryptedType(TypeDecorator):
    """
    Base class for encrypted SQLAlchemy types.
    
    This provides transparent encryption/decryption for database fields.
    """
    
    impl = Text
    cache_ok = True
    
    def __init__(self, encryption_context: str = "default", searchable: bool = False, *args, **kwargs):
        self.encryption_context = encryption_context
        self.searchable = searchable
        super().__init__(*args, **kwargs)
    
    def process_bind_param(self, value, dialect):
        """Encrypt value before storing in database."""
        if value is None:
            return None
        
        try:
            encrypted_data = encrypt_field(
                value,
                searchable=self.searchable,
                context=self.encryption_context
            )
            # Store as JSON string in database
            return json.dumps(encrypted_data)
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise EncryptionError(f"Failed to encrypt value: {str(e)}")
    
    def process_result_value(self, value, dialect):
        """Decrypt value when retrieving from database."""
        if value is None:
            return None
        
        try:
            # Parse JSON string from database
            encrypted_data = json.loads(value)
            return decrypt_field(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise DecryptionError(f"Failed to decrypt value: {str(e)}")


class EncryptedString(EncryptedType):
    """Encrypted string field with optional length limit."""
    
    def __init__(self, length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.length = length
    
    def process_result_value(self, value, dialect):
        """Ensure decrypted value is a string."""
        decrypted = super().process_result_value(value, dialect)
        if decrypted is not None and not isinstance(decrypted, str):
            decrypted = str(decrypted)
        return decrypted


class EncryptedText(EncryptedType):
    """Encrypted text field for longer content."""
    
    impl = Text
    
    def process_result_value(self, value, dialect):
        """Ensure decrypted value is a string."""
        decrypted = super().process_result_value(value, dialect)
        if decrypted is not None and not isinstance(decrypted, str):
            decrypted = str(decrypted)
        return decrypted


class EncryptedInteger(EncryptedType):
    """Encrypted integer field."""
    
    def process_bind_param(self, value, dialect):
        """Convert to string before encryption."""
        if value is not None:
            value = str(value)
        return super().process_bind_param(value, dialect)
    
    def process_result_value(self, value, dialect):
        """Convert back to integer after decryption."""
        decrypted = super().process_result_value(value, dialect)
        if decrypted is not None:
            try:
                return int(decrypted)
            except (ValueError, TypeError):
                logger.error(f"Failed to convert decrypted value to integer: {decrypted}")
                return None
        return None


class EncryptedFloat(EncryptedType):
    """Encrypted float field."""
    
    def process_bind_param(self, value, dialect):
        """Convert to string before encryption."""
        if value is not None:
            value = str(value)
        return super().process_bind_param(value, dialect)
    
    def process_result_value(self, value, dialect):
        """Convert back to float after decryption."""
        decrypted = super().process_result_value(value, dialect)
        if decrypted is not None:
            try:
                return float(decrypted)
            except (ValueError, TypeError):
                logger.error(f"Failed to convert decrypted value to float: {decrypted}")
                return None
        return None


class EncryptedJSON(EncryptedType):
    """Encrypted JSON field with automatic serialization."""
    
    impl = Text
    
    def process_bind_param(self, value, dialect):
        """Serialize to JSON before encryption."""
        if value is not None:
            # The encryption function handles JSON serialization
            pass
        return super().process_bind_param(value, dialect)
    
    def process_result_value(self, value, dialect):
        """Deserialize from JSON after decryption."""
        decrypted = super().process_result_value(value, dialect)
        return decrypted  # Already deserialized by decrypt_field


class EncryptedSearchableString(EncryptedString):
    """
    Encrypted string field with search capabilities.
    
    This creates an additional search token for exact match queries.
    """
    
    def __init__(self, **kwargs):
        kwargs['searchable'] = True
        super().__init__(**kwargs)


class SearchTokenIndex(TypeDecorator):
    """
    Search token index field for searchable encrypted fields.
    
    This is used alongside searchable encrypted fields to enable queries.
    """
    
    impl = String(64)  # Blake2b hash is 256 bits = 64 hex chars
    cache_ok = True
    
    def __init__(self, source_field: str, encryption_context: str = "search"):
        self.source_field = source_field
        self.encryption_context = encryption_context
        super().__init__()
    
    def process_bind_param(self, value, dialect):
        """Generate search token from value."""
        if value is None:
            return None
        
        cipher = get_searchable_cipher()
        return cipher.generate_search_token(str(value), self.encryption_context)
    
    def process_result_value(self, value, dialect):
        """Return search token as-is."""
        return value


# Mutable encrypted types for automatic change detection
class MutableEncryptedDict(MutableDict):
    """Mutable dictionary that tracks changes for encrypted JSON fields."""
    
    @classmethod
    def coerce(cls, key, value):
        """Convert plain dict to MutableEncryptedDict."""
        if not isinstance(value, MutableEncryptedDict):
            if isinstance(value, dict):
                return MutableEncryptedDict(value)
            return MutableDict.coerce(key, value)
        return value


class MutableEncryptedList(MutableList):
    """Mutable list that tracks changes for encrypted JSON fields."""
    
    @classmethod  
    def coerce(cls, key, value):
        """Convert plain list to MutableEncryptedList."""
        if not isinstance(value, MutableEncryptedList):
            if isinstance(value, list):
                return MutableEncryptedList(value)
            return MutableList.coerce(key, value)
        return value


# Associate mutable types with encrypted JSON
MutableEncryptedDict.associate_with(EncryptedJSON)
MutableEncryptedList.associate_with(EncryptedJSON)


def create_encrypted_column(
    column_type: Type[TypeDecorator],
    encryption_context: str,
    nullable: bool = True,
    searchable: bool = False,
    **kwargs
) -> TypeDecorator:
    """
    Factory function to create encrypted columns with consistent settings.
    
    Args:
        column_type: The encrypted type class to use
        encryption_context: Context for key derivation
        nullable: Whether the column can be null
        searchable: Whether to enable search capabilities
        **kwargs: Additional arguments for the column type
        
    Returns:
        Configured encrypted column type
    """
    if searchable and column_type == EncryptedString:
        column_type = EncryptedSearchableString
    
    return column_type(
        encryption_context=encryption_context,
        searchable=searchable,
        **kwargs
    )


# Utility functions for migration
def migrate_to_encrypted(model_class, field_name: str, encryption_context: str, batch_size: int = 1000):
    """
    Helper to migrate existing unencrypted data to encrypted format.
    
    Args:
        model_class: SQLAlchemy model class
        field_name: Name of the field to encrypt
        encryption_context: Encryption context to use
        batch_size: Number of records to process at once
    """
    from sqlalchemy.orm import Session
    from ..utils.encryption import get_optimized_cipher
    
    cipher = get_optimized_cipher()
    
    def migrate_batch(session: Session, offset: int):
        """Migrate a batch of records."""
        records = session.query(model_class).offset(offset).limit(batch_size).all()
        
        if not records:
            return False
        
        # Prepare values for batch encryption
        values = []
        for record in records:
            value = getattr(record, field_name)
            if value is not None:
                values.append(str(value))
            else:
                values.append(None)
        
        # Batch encrypt
        encrypted_values = []
        for value in values:
            if value is not None:
                encrypted = encrypt_field(value, context=encryption_context)
                encrypted_values.append(json.dumps(encrypted))
            else:
                encrypted_values.append(None)
        
        # Update records
        for record, encrypted_value in zip(records, encrypted_values):
            setattr(record, f"{field_name}_encrypted", encrypted_value)
        
        session.commit()
        return True
    
    return migrate_batch


# Export main classes
__all__ = [
    "EncryptedType",
    "EncryptedString",
    "EncryptedText",
    "EncryptedInteger",
    "EncryptedFloat",
    "EncryptedJSON",
    "EncryptedSearchableString",
    "SearchTokenIndex",
    "MutableEncryptedDict",
    "MutableEncryptedList",
    "create_encrypted_column",
    "migrate_to_encrypted"
]