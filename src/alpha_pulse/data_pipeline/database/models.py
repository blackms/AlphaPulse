"""
Base database models for the data pipeline.

This module provides the base SQLAlchemy model class for
all database models in the data pipeline.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base


# Create declarative base
Base = declarative_base()


class BaseModel(Base):
    """Base model for all database models."""
    
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            
            # Handle UUID
            if isinstance(value, uuid.UUID):
                value = str(value)
                
            # Handle datetime
            if isinstance(value, datetime):
                value = value.isoformat()
                
            result[column.name] = value
            
        return result