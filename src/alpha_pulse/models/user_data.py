"""
Encrypted user data models for AlphaPulse.

This module contains SQLAlchemy models for user data with
field-level encryption for PII and sensitive information.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean,
    ForeignKey, Index, Table, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func, text

from .encrypted_fields import (
    EncryptedString,
    EncryptedSearchableString,
    EncryptedJSON,
    EncryptedText,
    SearchTokenIndex,
    create_encrypted_column
)

Base = declarative_base()


class Tenant(Base):
    """Tenant metadata for multi-tenant SaaS."""

    __tablename__ = "tenants"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False, unique=True)
    subscription_tier = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, server_default="active")
    max_users = Column(Integer, server_default="5")
    max_api_calls_per_day = Column(Integer, server_default="10000")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    metadata_json = Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))

    # Relationships
    users = relationship("User", back_populates="tenant")

    __table_args__ = (
        Index("idx_tenants_slug", "slug"),
        Index("idx_tenants_status", "status"),
        Index("idx_tenants_tier", "subscription_tier"),
    )

# Association table for user roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)


class User(Base):
    """Model for users with encrypted PII."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    username = Column(String(50), unique=True, nullable=False)
    
    # Encrypted PII fields
    email = Column(
        EncryptedSearchableString(
            encryption_context="user_pii"
        ),
        nullable=False
    )
    email_search = Column(
        SearchTokenIndex(
            source_field="email",
            encryption_context="user_pii"
        ),
        index=True
    )
    
    phone_number = Column(
        EncryptedString(encryption_context="user_pii"),
        nullable=True
    )
    
    # Encrypted personal information
    first_name = Column(
        EncryptedString(encryption_context="user_personal"),
        nullable=True
    )
    last_name = Column(
        EncryptedString(encryption_context="user_personal"),
        nullable=True
    )
    date_of_birth = Column(
        EncryptedString(encryption_context="user_personal"),
        nullable=True
    )
    
    # Encrypted address information
    address_data = Column(
        EncryptedJSON(encryption_context="user_address"),
        nullable=True
    )
    
    # Authentication (password hash is already encrypted by bcrypt)
    password_hash = Column(String(255), nullable=False)
    
    # Encrypted security settings
    two_factor_secret = Column(
        EncryptedString(encryption_context="user_security"),
        nullable=True
    )
    backup_codes = Column(
        EncryptedJSON(encryption_context="user_security"),
        nullable=True
    )
    
    # Account settings
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    
    # Encrypted preferences
    preferences = Column(
        EncryptedJSON(encryption_context="user_preferences"),
        nullable=True,
        default={}
    )
    
    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("UserAuditLog", back_populates="user")
    notification_settings = relationship("NotificationSettings", back_populates="user", uselist=False)
    
    __table_args__ = (
        Index("idx_user_email_search", "email_search"),
        Index("idx_user_username", "username"),
        Index("idx_user_active", "is_active"),
        Index("idx_user_tenant_id", "tenant_id"),
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
    )
    
    @validates("email")
    def validate_email(self, key, value):
        """Validate and set search token for email."""
        if value:
            # The search token will be automatically generated
            self.email_search = value
        return value
    
    @property
    def full_name(self):
        """Get decrypted full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    def mask_email(self) -> str:
        """Return masked email for display."""
        if not self.email:
            return ""
        
        parts = self.email.split("@")
        if len(parts) != 2:
            return "***@***"
        
        local = parts[0]
        domain = parts[1]
        
        if len(local) <= 3:
            masked_local = "*" * len(local)
        else:
            masked_local = local[:2] + "*" * (len(local) - 3) + local[-1]
        
        return f"{masked_local}@{domain}"


class Role(Base):
    """Model for user roles."""
    
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    
    # Encrypted permissions
    permissions = Column(
        EncryptedJSON(encryption_context="role_permissions"),
        nullable=False,
        default=[]
    )
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")


class APIKey(Base):
    """Model for API keys with encrypted values."""
    
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Public identifier
    key_id = Column(String(50), unique=True, nullable=False)
    
    # Encrypted API key
    key_hash = Column(
        EncryptedString(encryption_context="api_key"),
        nullable=False
    )
    
    # Key metadata
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Encrypted permissions
    permissions = Column(
        EncryptedJSON(encryption_context="api_permissions"),
        nullable=False,
        default=[]
    )
    
    # Rate limiting
    rate_limit = Column(Integer, default=1000, nullable=False)
    
    # Validity
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Encrypted usage statistics
    usage_stats = Column(
        EncryptedJSON(encryption_context="api_usage"),
        nullable=True,
        default={}
    )
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index("idx_apikey_user", "user_id"),
        Index("idx_apikey_keyid", "key_id"),
        Index("idx_apikey_active", "is_active"),
    )


class NotificationSettings(Base):
    """Model for user notification settings with encrypted contact info."""
    
    __tablename__ = "notification_settings"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Encrypted notification channels
    email_notifications = Column(Boolean, default=True, nullable=False)
    sms_notifications = Column(Boolean, default=False, nullable=False)
    push_notifications = Column(Boolean, default=True, nullable=False)
    
    # Encrypted contact information
    notification_email = Column(
        EncryptedString(encryption_context="notification_contact"),
        nullable=True
    )
    notification_phone = Column(
        EncryptedString(encryption_context="notification_contact"),
        nullable=True
    )
    
    # Encrypted webhook settings
    webhook_url = Column(
        EncryptedString(encryption_context="notification_webhook"),
        nullable=True
    )
    webhook_secret = Column(
        EncryptedString(encryption_context="notification_webhook"),
        nullable=True
    )
    
    # Notification preferences (encrypted)
    preferences = Column(
        EncryptedJSON(encryption_context="notification_preferences"),
        nullable=False,
        default={
            "trade_executions": True,
            "risk_alerts": True,
            "system_updates": True,
            "performance_reports": True,
            "security_alerts": True
        }
    )
    
    # Quiet hours
    quiet_hours_start = Column(String(5), nullable=True)  # HH:MM format
    quiet_hours_end = Column(String(5), nullable=True)
    
    # Timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="notification_settings")


class UserAuditLog(Base):
    """Model for user audit logs with encrypted sensitive actions."""
    
    __tablename__ = "user_audit_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Action details
    action = Column(String(100), nullable=False)
    resource = Column(String(100), nullable=True)
    
    # Encrypted action details
    details = Column(
        EncryptedJSON(encryption_context="audit_details"),
        nullable=True
    )
    
    # Request information (encrypted)
    ip_address = Column(
        EncryptedString(encryption_context="audit_request"),
        nullable=True
    )
    user_agent = Column(
        EncryptedString(encryption_context="audit_request"),
        nullable=True
    )
    
    # Result
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_action", "action"),
        Index("idx_audit_created", "created_at"),
    )


class KYCInformation(Base):
    """Model for KYC information with encrypted sensitive documents."""
    
    __tablename__ = "kyc_information"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Encrypted identity information
    document_type = Column(String(50), nullable=False)  # passport, driver_license, etc.
    document_number = Column(
        EncryptedString(encryption_context="kyc_document"),
        nullable=False
    )
    document_country = Column(String(2), nullable=False)  # ISO country code
    
    # Encrypted document data
    document_data = Column(
        EncryptedJSON(encryption_context="kyc_document_data"),
        nullable=False
    )
    
    # Encrypted verification data
    verification_data = Column(
        EncryptedJSON(encryption_context="kyc_verification"),
        nullable=True
    )
    
    # Status
    status = Column(String(20), default="pending", nullable=False)
    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(String(100), nullable=True)
    
    # Encrypted notes
    notes = Column(
        EncryptedText(encryption_context="kyc_notes"),
        nullable=True
    )
    
    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_kyc_user", "user_id"),
        Index("idx_kyc_status", "status"),
    )


# Utility functions for user data
def create_user_tables(engine):
    """Create all user tables with encryption."""
    Base.metadata.create_all(engine)


def drop_user_tables(engine):
    """Drop all user tables."""
    Base.metadata.drop_all(engine)


def anonymize_user_data(user: User) -> Dict[str, Any]:
    """
    Return anonymized user data for analytics.
    
    Args:
        user: User instance
        
    Returns:
        Dictionary with anonymized data
    """
    return {
        "id": f"user_{user.id}",
        "created_month": user.created_at.strftime("%Y-%m"),
        "is_active": user.is_active,
        "has_2fa": bool(user.two_factor_secret),
        "role_count": len(user.roles),
        "api_key_count": len([k for k in user.api_keys if k.is_active])
    }


# Export models
__all__ = [
    "Tenant",
    "User",
    "Role",
    "APIKey",
    "NotificationSettings",
    "UserAuditLog",
    "KYCInformation",
    "create_user_tables",
    "drop_user_tables",
    "anonymize_user_data"
]
