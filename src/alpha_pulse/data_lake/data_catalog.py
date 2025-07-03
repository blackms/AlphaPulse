"""
Data catalog for the data lake.

Provides metadata management, data discovery, and lineage tracking
for all datasets in the data lake.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import logging

from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from alpha_pulse.data.lineage.lineage_tracker import LineageTracker


logger = logging.getLogger(__name__)

Base = declarative_base()


class DatasetType(Enum):
    """Types of datasets in the catalog."""
    RAW = "raw"              # Bronze layer raw data
    PROCESSED = "processed"   # Silver layer processed data
    BUSINESS = "business"     # Gold layer business datasets
    FEATURE = "feature"       # ML feature datasets
    MODEL = "model"          # Model training datasets
    REPORT = "report"        # Reporting datasets


class DatasetStatus(Enum):
    """Dataset lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class DatasetSchema:
    """Schema definition for a dataset."""
    columns: Dict[str, str]  # column_name -> data_type
    primary_keys: List[str] = field(default_factory=list)
    partition_keys: List[str] = field(default_factory=list)
    description: Optional[str] = None
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "columns": self.columns,
            "primary_keys": self.primary_keys,
            "partition_keys": self.partition_keys,
            "description": self.description,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSchema':
        """Create from dictionary."""
        return cls(
            columns=data["columns"],
            primary_keys=data.get("primary_keys", []),
            partition_keys=data.get("partition_keys", []),
            description=data.get("description"),
            version=data.get("version", "1.0")
        )


@dataclass
class DatasetMetadata:
    """Comprehensive metadata for a dataset."""
    dataset_id: str
    name: str
    type: DatasetType
    layer: str  # bronze, silver, gold
    path: str
    schema: DatasetSchema
    owner: str
    created_at: datetime
    updated_at: datetime
    status: DatasetStatus = DatasetStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "type": self.type.value,
            "layer": self.layer,
            "path": self.path,
            "schema": self.schema.to_dict(),
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "tags": self.tags,
            "properties": self.properties,
            "quality_score": self.quality_score,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count
        }


class CatalogEntry(Base):
    """SQLAlchemy model for catalog entries."""
    __tablename__ = 'data_catalog'
    
    dataset_id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False, index=True)
    layer = Column(String, nullable=False, index=True)
    path = Column(String, nullable=False)
    schema_json = Column(JSON, nullable=False)
    owner = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    tags = Column(JSON, default=[])
    properties = Column(JSON, default={})
    quality_score = Column(Float)
    size_bytes = Column(Integer)
    row_count = Column(Integer)
    lineage_json = Column(JSON)
    search_text = Column(Text)  # For full-text search


class DataCatalog:
    """Main data catalog for the data lake."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize data catalog."""
        if db_url is None:
            # Use SQLite by default
            db_path = Path.home() / ".alphapulse" / "data_catalog.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{db_path}"
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.lineage_tracker = LineageTracker()
        
        logger.info(f"Data catalog initialized with database: {db_url}")
    
    def register_dataset(
        self,
        name: str,
        type: DatasetType,
        layer: str,
        path: str,
        schema: DatasetSchema,
        owner: str,
        tags: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new dataset in the catalog."""
        dataset_id = str(uuid.uuid4())
        
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            type=type,
            layer=layer,
            path=path,
            schema=schema,
            owner=owner,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=tags or [],
            properties=properties or {}
        )
        
        # Create catalog entry
        entry = CatalogEntry(
            dataset_id=dataset_id,
            name=name,
            type=type.value,
            layer=layer,
            path=path,
            schema_json=schema.to_dict(),
            owner=owner,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            status=DatasetStatus.ACTIVE.value,
            tags=metadata.tags,
            properties=metadata.properties,
            search_text=self._create_search_text(metadata)
        )
        
        # Save to database
        session = self.Session()
        try:
            session.add(entry)
            session.commit()
            logger.info(f"Registered dataset: {name} (ID: {dataset_id})")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to register dataset: {e}")
            raise
        finally:
            session.close()
        
        # Track lineage
        self._track_dataset_lineage(dataset_id, name, layer)
        
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by ID."""
        session = self.Session()
        try:
            entry = session.query(CatalogEntry).filter_by(dataset_id=dataset_id).first()
            if entry:
                return self._entry_to_metadata(entry)
            return None
        finally:
            session.close()
    
    def find_dataset_by_name(self, name: str) -> Optional[DatasetMetadata]:
        """Find dataset by name."""
        session = self.Session()
        try:
            entry = session.query(CatalogEntry).filter_by(
                name=name, 
                status=DatasetStatus.ACTIVE.value
            ).first()
            if entry:
                return self._entry_to_metadata(entry)
            return None
        finally:
            session.close()
    
    def search_datasets(
        self,
        query: Optional[str] = None,
        layer: Optional[str] = None,
        type: Optional[DatasetType] = None,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[DatasetStatus] = None
    ) -> List[DatasetMetadata]:
        """Search datasets with various filters."""
        session = self.Session()
        try:
            q = session.query(CatalogEntry)
            
            # Apply filters
            if query:
                q = q.filter(CatalogEntry.search_text.contains(query.lower()))
            if layer:
                q = q.filter_by(layer=layer)
            if type:
                q = q.filter_by(type=type.value)
            if owner:
                q = q.filter_by(owner=owner)
            if status:
                q = q.filter_by(status=status.value)
            else:
                # Default to active datasets
                q = q.filter_by(status=DatasetStatus.ACTIVE.value)
            
            # Tag filtering
            if tags:
                for tag in tags:
                    q = q.filter(CatalogEntry.tags.contains([tag]))
            
            # Order by updated time
            q = q.order_by(CatalogEntry.updated_at.desc())
            
            entries = q.all()
            return [self._entry_to_metadata(entry) for entry in entries]
        finally:
            session.close()
    
    def update_dataset(
        self,
        dataset_id: str,
        schema: Optional[DatasetSchema] = None,
        tags: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None,
        size_bytes: Optional[int] = None,
        row_count: Optional[int] = None
    ):
        """Update dataset metadata."""
        session = self.Session()
        try:
            entry = session.query(CatalogEntry).filter_by(dataset_id=dataset_id).first()
            if not entry:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Update fields
            if schema:
                entry.schema_json = schema.to_dict()
            if tags is not None:
                entry.tags = tags
            if properties is not None:
                entry.properties = properties
            if quality_score is not None:
                entry.quality_score = quality_score
            if size_bytes is not None:
                entry.size_bytes = size_bytes
            if row_count is not None:
                entry.row_count = row_count
            
            entry.updated_at = datetime.utcnow()
            entry.search_text = self._create_search_text(self._entry_to_metadata(entry))
            
            session.commit()
            logger.info(f"Updated dataset: {dataset_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update dataset: {e}")
            raise
        finally:
            session.close()
    
    def deprecate_dataset(self, dataset_id: str, reason: Optional[str] = None):
        """Mark dataset as deprecated."""
        session = self.Session()
        try:
            entry = session.query(CatalogEntry).filter_by(dataset_id=dataset_id).first()
            if not entry:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            entry.status = DatasetStatus.DEPRECATED.value
            entry.updated_at = datetime.utcnow()
            
            if reason:
                if not entry.properties:
                    entry.properties = {}
                entry.properties["deprecation_reason"] = reason
                entry.properties["deprecation_date"] = datetime.utcnow().isoformat()
            
            session.commit()
            logger.info(f"Deprecated dataset: {dataset_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to deprecate dataset: {e}")
            raise
        finally:
            session.close()
    
    def get_dataset_lineage(self, dataset_id: str) -> Dict[str, Any]:
        """Get lineage information for a dataset."""
        metadata = self.get_dataset(dataset_id)
        if not metadata:
            return {}
        
        # Get lineage from lineage tracker
        lineage = self.lineage_tracker.get_lineage(
            entity_type="dataset",
            entity_id=dataset_id
        )
        
        return {
            "dataset": metadata.to_dict(),
            "lineage": lineage
        }
    
    def get_dataset_dependencies(self, dataset_id: str) -> Dict[str, List[str]]:
        """Get upstream and downstream dependencies."""
        return {
            "upstream": self.lineage_tracker.get_upstream_dependencies(
                "dataset", dataset_id
            ),
            "downstream": self.lineage_tracker.get_downstream_dependencies(
                "dataset", dataset_id
            )
        }
    
    def get_catalog_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        session = self.Session()
        try:
            total_count = session.query(CatalogEntry).count()
            active_count = session.query(CatalogEntry).filter_by(
                status=DatasetStatus.ACTIVE.value
            ).count()
            
            # Count by layer
            layer_counts = {}
            for layer in ["bronze", "silver", "gold"]:
                count = session.query(CatalogEntry).filter_by(
                    layer=layer,
                    status=DatasetStatus.ACTIVE.value
                ).count()
                layer_counts[layer] = count
            
            # Count by type
            type_counts = {}
            for dtype in DatasetType:
                count = session.query(CatalogEntry).filter_by(
                    type=dtype.value,
                    status=DatasetStatus.ACTIVE.value
                ).count()
                type_counts[dtype.value] = count
            
            # Calculate total size
            total_size = session.query(CatalogEntry).filter(
                CatalogEntry.size_bytes.isnot(None)
            ).with_entities(
                CatalogEntry.size_bytes
            ).all()
            total_size_bytes = sum(s[0] for s in total_size if s[0])
            
            return {
                "total_datasets": total_count,
                "active_datasets": active_count,
                "datasets_by_layer": layer_counts,
                "datasets_by_type": type_counts,
                "total_size_bytes": total_size_bytes,
                "total_size_gb": total_size_bytes / (1024**3) if total_size_bytes else 0
            }
        finally:
            session.close()
    
    def _entry_to_metadata(self, entry: CatalogEntry) -> DatasetMetadata:
        """Convert catalog entry to metadata object."""
        return DatasetMetadata(
            dataset_id=entry.dataset_id,
            name=entry.name,
            type=DatasetType(entry.type),
            layer=entry.layer,
            path=entry.path,
            schema=DatasetSchema.from_dict(entry.schema_json),
            owner=entry.owner,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            status=DatasetStatus(entry.status),
            tags=entry.tags or [],
            properties=entry.properties or {},
            quality_score=entry.quality_score,
            size_bytes=entry.size_bytes,
            row_count=entry.row_count
        )
    
    def _create_search_text(self, metadata: DatasetMetadata) -> str:
        """Create searchable text from metadata."""
        parts = [
            metadata.name.lower(),
            metadata.type.value,
            metadata.layer,
            metadata.owner.lower(),
            " ".join(metadata.tags).lower(),
            " ".join(metadata.schema.columns.keys()).lower()
        ]
        
        if metadata.schema.description:
            parts.append(metadata.schema.description.lower())
        
        return " ".join(parts)
    
    def _track_dataset_lineage(self, dataset_id: str, name: str, layer: str):
        """Track dataset in lineage system."""
        self.lineage_tracker.track_transformation(
            input_datasets=[],
            output_dataset=dataset_id,
            transformation_type="create",
            transformation_id=f"create_{dataset_id}",
            metadata={
                "name": name,
                "layer": layer,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def export_catalog(self, output_path: str, format: str = "json"):
        """Export entire catalog."""
        datasets = self.search_datasets()
        
        if format == "json":
            catalog_data = {
                "export_date": datetime.utcnow().isoformat(),
                "dataset_count": len(datasets),
                "datasets": [ds.to_dict() for ds in datasets]
            }
            
            with open(output_path, 'w') as f:
                json.dump(catalog_data, f, indent=2)
        
        elif format == "csv":
            import pandas as pd
            
            # Flatten dataset information
            rows = []
            for ds in datasets:
                row = {
                    "dataset_id": ds.dataset_id,
                    "name": ds.name,
                    "type": ds.type.value,
                    "layer": ds.layer,
                    "path": ds.path,
                    "owner": ds.owner,
                    "created_at": ds.created_at,
                    "updated_at": ds.updated_at,
                    "status": ds.status.value,
                    "quality_score": ds.quality_score,
                    "size_bytes": ds.size_bytes,
                    "row_count": ds.row_count,
                    "column_count": len(ds.schema.columns),
                    "tags": ", ".join(ds.tags)
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Exported catalog to {output_path}")
    
    def close(self):
        """Close catalog connections."""
        self.engine.dispose()