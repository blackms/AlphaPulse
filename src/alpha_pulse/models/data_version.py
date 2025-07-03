"""
Data Version models for versioning system.

Provides:
- Version metadata structures
- Version status tracking
- Version configuration models
- Rollback request models
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from alpha_pulse.models.base import BaseModel


class VersionType(Enum):
    """Types of version changes."""
    MAJOR = "major"  # Breaking changes, schema modifications
    MINOR = "minor"  # New features, additions
    PATCH = "patch"  # Bug fixes, small changes
    TIMESTAMP = "timestamp"  # Time-based versioning
    CONTENT = "content"  # Content hash-based
    BRANCH = "branch"  # Branch-based experimental


class VersionStatus(Enum):
    """Status of a data version."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"
    DRAFT = "draft"


@dataclass
class VersionMetadata:
    """Metadata for a data version."""
    trigger: str  # What triggered the version creation
    description: str
    author: str
    parent_version: Optional[str] = None
    branch: Optional[str] = None
    data_stats: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    change_summary: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trigger": self.trigger,
            "description": self.description,
            "author": self.author,
            "parent_version": self.parent_version,
            "branch": self.branch,
            "data_stats": self.data_stats,
            "quality_metrics": self.quality_metrics,
            "change_summary": self.change_summary,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata
        }


@dataclass
class DataVersion(BaseModel):
    """Represents a version of data."""
    version_id: str
    dataset_id: str
    fingerprint: str  # Content fingerprint
    created_at: datetime
    version_type: VersionType
    status: VersionStatus
    metadata: VersionMetadata
    
    # Optional fields
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema_version: Optional[str] = None
    
    # Relationships
    downstream_versions: List[str] = field(default_factory=list)
    upstream_versions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "dataset_id": self.dataset_id,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at.isoformat(),
            "version_type": self.version_type.value,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "schema_version": self.schema_version,
            "downstream_versions": self.downstream_versions,
            "upstream_versions": self.upstream_versions
        }
    
    def is_active(self) -> bool:
        """Check if version is active."""
        return self.status == VersionStatus.ACTIVE
    
    def get_version_string(self) -> str:
        """Get human-readable version string."""
        if self.version_type == VersionType.TIMESTAMP:
            return self.created_at.strftime("%Y%m%d-%H%M%S")
        else:
            return self.version_id


@dataclass
class VersionDiff:
    """Represents differences between two versions."""
    version1_id: str
    version2_id: str
    content_changes: Dict[str, Any]
    schema_changes: Dict[str, Any]
    stats_changes: Dict[str, Any]
    quality_changes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version1_id": self.version1_id,
            "version2_id": self.version2_id,
            "content_changes": self.content_changes,
            "schema_changes": self.schema_changes,
            "stats_changes": self.stats_changes,
            "quality_changes": self.quality_changes
        }
    
    def has_breaking_changes(self) -> bool:
        """Check if diff contains breaking changes."""
        # Schema changes are typically breaking
        if self.schema_changes:
            return True
        
        # Check for significant quality degradation
        for metric, change in self.quality_changes.items():
            if isinstance(change, dict) and 'old' in change and 'new' in change:
                if change['new'] < change['old'] * 0.8:  # 20% degradation
                    return True
        
        return False


@dataclass
class RollbackRequest:
    """Request to rollback to a previous version."""
    dataset_id: str
    from_version: str
    to_version: str
    reason: str
    requested_at: datetime
    requested_by: str
    
    # Rollback metadata
    impact_analysis: Optional[Dict[str, Any]] = None
    approval_required: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Execution details
    executed: bool = False
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
            "requested_at": self.requested_at.isoformat(),
            "requested_by": self.requested_by,
            "impact_analysis": self.impact_analysis,
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "executed": self.executed,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_result": self.execution_result
        }


@dataclass
class VersioningConfig:
    """Configuration for versioning system."""
    # Storage settings
    storage_backend: str = "filesystem"  # filesystem, s3, database
    storage_path: str = "./data/versions"
    compression_enabled: bool = True
    
    # Versioning behavior
    auto_version_on_change: bool = True
    allow_duplicates: bool = False
    require_description: bool = True
    
    # Retention policy
    retention_days: Optional[int] = 90
    retention_versions: int = 100
    auto_cleanup: bool = True
    
    # Snapshot settings
    enable_snapshots: bool = True
    snapshot_format: str = "pickle"  # pickle, parquet, json
    max_snapshot_size_mb: int = 1000
    
    # Performance settings
    cache_enabled: bool = True
    cache_size: int = 100
    parallel_operations: bool = True
    
    # Quality integration
    require_quality_check: bool = True
    min_quality_score: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_backend": self.storage_backend,
            "storage_path": self.storage_path,
            "compression_enabled": self.compression_enabled,
            "auto_version_on_change": self.auto_version_on_change,
            "allow_duplicates": self.allow_duplicates,
            "require_description": self.require_description,
            "retention_days": self.retention_days,
            "retention_versions": self.retention_versions,
            "auto_cleanup": self.auto_cleanup,
            "enable_snapshots": self.enable_snapshots,
            "snapshot_format": self.snapshot_format,
            "max_snapshot_size_mb": self.max_snapshot_size_mb,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "parallel_operations": self.parallel_operations,
            "require_quality_check": self.require_quality_check,
            "min_quality_score": self.min_quality_score
        }


@dataclass
class VersionHistory:
    """Version history for a dataset."""
    dataset_id: str
    versions: List[DataVersion]
    current_version: str
    total_versions: int
    
    # Statistics
    first_version_date: datetime
    last_version_date: datetime
    average_version_size: Optional[float] = None
    total_storage_used: Optional[int] = None
    
    # Version frequency
    versions_per_day: Optional[float] = None
    versions_per_week: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "versions": [v.to_dict() for v in self.versions],
            "current_version": self.current_version,
            "total_versions": self.total_versions,
            "first_version_date": self.first_version_date.isoformat(),
            "last_version_date": self.last_version_date.isoformat(),
            "average_version_size": self.average_version_size,
            "total_storage_used": self.total_storage_used,
            "versions_per_day": self.versions_per_day,
            "versions_per_week": self.versions_per_week
        }
    
    def get_version_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of version changes."""
        timeline = []
        
        for version in sorted(self.versions, key=lambda v: v.created_at):
            timeline.append({
                "timestamp": version.created_at.isoformat(),
                "version_id": version.version_id,
                "type": version.version_type.value,
                "trigger": version.metadata.trigger,
                "description": version.metadata.description
            })
        
        return timeline