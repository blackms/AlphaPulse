"""
Data Version Manager for comprehensive versioning of market data.

Provides:
- Semantic versioning for data assets
- Timestamp-based versioning for time series
- Hash-based versioning for content changes
- Version rollback mechanisms
- Version metadata tracking
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
from collections import defaultdict
import pickle
from loguru import logger

from alpha_pulse.models.data_version import (
    DataVersion, VersionType, VersionStatus, VersionMetadata,
    VersionDiff, RollbackRequest, VersioningConfig
)
from alpha_pulse.utils.data_fingerprinting import DataFingerprinter
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class VersioningStrategy(Enum):
    """Strategies for data versioning."""
    SEMANTIC = "semantic"  # major.minor.patch
    TIMESTAMP = "timestamp"  # timestamp-based
    HASH = "hash"  # content hash-based
    BRANCH = "branch"  # branch-based for experiments


class VersionTrigger(Enum):
    """Triggers for creating new versions."""
    DATA_SOURCE_CHANGE = "data_source_change"
    SCHEMA_MODIFICATION = "schema_modification"
    TRANSFORMATION_CHANGE = "transformation_change"
    MANUAL = "manual"
    QUALITY_THRESHOLD = "quality_threshold"
    TIME_BASED = "time_based"


@dataclass
class VersioningContext:
    """Context for version creation."""
    trigger: VersionTrigger
    description: str
    author: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    branch: Optional[str] = None


@dataclass
class VersionSnapshot:
    """Snapshot of versioned data."""
    version_id: str
    data: Any
    fingerprint: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'version_id': self.version_id,
            'fingerprint': self.fingerprint,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class DataVersionManager:
    """Manager for comprehensive data versioning."""
    
    def __init__(self, config: Optional[VersioningConfig] = None):
        self.config = config or VersioningConfig()
        self.fingerprinter = DataFingerprinter()
        self.audit_logger = get_audit_logger()
        
        # Version storage
        self.versions: Dict[str, Dict[str, DataVersion]] = defaultdict(dict)
        self.snapshots: Dict[str, Dict[str, VersionSnapshot]] = defaultdict(dict)
        self.version_graph: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        
        # Current versions tracking
        self.current_versions: Dict[str, str] = {}  # dataset_id -> version_id
        self.version_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Version history
        self.version_history: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        
        # Initialize storage if configured
        if self.config.storage_backend:
            self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize persistent storage backend."""
        storage_path = Path(self.config.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing versions if available
        self._load_versions_from_storage()
    
    async def create_version(
        self,
        dataset_id: str,
        data: Any,
        context: VersioningContext,
        strategy: Optional[VersioningStrategy] = None
    ) -> DataVersion:
        """
        Create a new version of data.
        
        Args:
            dataset_id: Unique identifier for the dataset
            data: The data to version
            context: Context for version creation
            strategy: Versioning strategy to use
        
        Returns:
            Created DataVersion instance
        """
        async with self.version_locks[dataset_id]:
            try:
                # Determine versioning strategy
                if strategy is None:
                    strategy = self._determine_strategy(dataset_id, data, context)
                
                # Generate version ID based on strategy
                version_id = await self._generate_version_id(
                    dataset_id, data, strategy, context
                )
                
                # Calculate data fingerprint
                fingerprint = await self.fingerprinter.calculate_fingerprint(data)
                
                # Check if content already exists (deduplication)
                existing_version = self._find_version_by_fingerprint(dataset_id, fingerprint)
                if existing_version and not self.config.allow_duplicates:
                    logger.info(f"Content already exists in version {existing_version.version_id}")
                    return existing_version
                
                # Create version metadata
                metadata = VersionMetadata(
                    trigger=context.trigger.value,
                    description=context.description,
                    author=context.author or "system",
                    parent_version=context.parent_version or self.current_versions.get(dataset_id),
                    branch=context.branch,
                    data_stats=self._calculate_data_stats(data),
                    quality_metrics=context.metadata.get('quality_metrics', {}),
                    change_summary=await self._generate_change_summary(
                        dataset_id, data, context.parent_version
                    )
                )
                
                # Create version instance
                version = DataVersion(
                    version_id=version_id,
                    dataset_id=dataset_id,
                    fingerprint=fingerprint,
                    created_at=datetime.utcnow(),
                    version_type=self._strategy_to_type(strategy),
                    status=VersionStatus.ACTIVE,
                    metadata=metadata
                )
                
                # Store version
                self.versions[dataset_id][version_id] = version
                
                # Create and store snapshot if enabled
                if self.config.enable_snapshots:
                    snapshot = VersionSnapshot(
                        version_id=version_id,
                        data=data,
                        fingerprint=fingerprint,
                        timestamp=datetime.utcnow(),
                        metadata=metadata.to_dict()
                    )
                    self.snapshots[dataset_id][version_id] = snapshot
                    await self._save_snapshot(dataset_id, snapshot)
                
                # Update version graph
                if context.parent_version:
                    self.version_graph[context.parent_version].append(version_id)
                
                # Update current version
                self.current_versions[dataset_id] = version_id
                
                # Update version history
                self.version_history[dataset_id].append((datetime.utcnow(), version_id))
                
                # Log version creation
                self.audit_logger.log(
                    event_type=AuditEventType.DATA_VERSION_CREATED,
                    event_data={
                        "dataset_id": dataset_id,
                        "version_id": version_id,
                        "trigger": context.trigger.value,
                        "fingerprint": fingerprint,
                        "parent_version": context.parent_version
                    },
                    severity=AuditSeverity.INFO
                )
                
                # Trigger cleanup if needed
                await self._cleanup_old_versions(dataset_id)
                
                return version
                
            except Exception as e:
                logger.error(f"Failed to create version for {dataset_id}: {e}")
                raise
    
    async def get_version(
        self,
        dataset_id: str,
        version_id: Optional[str] = None
    ) -> Optional[DataVersion]:
        """
        Get a specific version or the current version.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Specific version ID (optional)
        
        Returns:
            DataVersion instance or None
        """
        if version_id is None:
            version_id = self.current_versions.get(dataset_id)
        
        if version_id and dataset_id in self.versions:
            return self.versions[dataset_id].get(version_id)
        
        return None
    
    async def get_version_data(
        self,
        dataset_id: str,
        version_id: str
    ) -> Optional[Any]:
        """
        Retrieve the actual data for a version.
        
        Args:
            dataset_id: Dataset identifier
            version_id: Version identifier
        
        Returns:
            The versioned data or None
        """
        # Check in-memory snapshots first
        if dataset_id in self.snapshots and version_id in self.snapshots[dataset_id]:
            return self.snapshots[dataset_id][version_id].data
        
        # Load from storage if available
        if self.config.storage_backend:
            snapshot = await self._load_snapshot(dataset_id, version_id)
            if snapshot:
                return snapshot.data
        
        return None
    
    async def rollback(
        self,
        dataset_id: str,
        target_version: str,
        reason: str,
        validate: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Rollback to a specific version.
        
        Args:
            dataset_id: Dataset identifier
            target_version: Target version to rollback to
            reason: Reason for rollback
            validate: Whether to validate before rollback
        
        Returns:
            Tuple of (success, error_message)
        """
        async with self.version_locks[dataset_id]:
            try:
                # Validate target version exists
                target = await self.get_version(dataset_id, target_version)
                if not target:
                    return False, f"Target version {target_version} not found"
                
                # Get current version
                current_version_id = self.current_versions.get(dataset_id)
                if not current_version_id:
                    return False, "No current version found"
                
                if current_version_id == target_version:
                    return False, "Already at target version"
                
                # Perform validation if requested
                if validate:
                    validation_result = await self._validate_rollback(
                        dataset_id, current_version_id, target_version
                    )
                    if not validation_result.is_valid:
                        return False, f"Rollback validation failed: {validation_result.reason}"
                
                # Create rollback request
                rollback_request = RollbackRequest(
                    dataset_id=dataset_id,
                    from_version=current_version_id,
                    to_version=target_version,
                    reason=reason,
                    requested_at=datetime.utcnow(),
                    requested_by="system"
                )
                
                # Perform rollback
                self.current_versions[dataset_id] = target_version
                
                # Update version history
                self.version_history[dataset_id].append((datetime.utcnow(), target_version))
                
                # Mark intermediate versions as rolled back
                await self._mark_versions_as_rolled_back(
                    dataset_id, current_version_id, target_version
                )
                
                # Log rollback
                self.audit_logger.log(
                    event_type=AuditEventType.DATA_VERSION_ROLLBACK,
                    event_data={
                        "dataset_id": dataset_id,
                        "from_version": current_version_id,
                        "to_version": target_version,
                        "reason": reason
                    },
                    severity=AuditSeverity.WARNING
                )
                
                return True, None
                
            except Exception as e:
                logger.error(f"Rollback failed for {dataset_id}: {e}")
                return False, str(e)
    
    async def get_version_diff(
        self,
        dataset_id: str,
        version1: str,
        version2: str
    ) -> Optional[VersionDiff]:
        """
        Calculate differences between two versions.
        
        Args:
            dataset_id: Dataset identifier
            version1: First version ID
            version2: Second version ID
        
        Returns:
            VersionDiff instance or None
        """
        v1 = await self.get_version(dataset_id, version1)
        v2 = await self.get_version(dataset_id, version2)
        
        if not v1 or not v2:
            return None
        
        # Get data for both versions
        data1 = await self.get_version_data(dataset_id, version1)
        data2 = await self.get_version_data(dataset_id, version2)
        
        if data1 is None or data2 is None:
            return None
        
        # Calculate differences
        content_diff = await self.fingerprinter.calculate_content_diff(data1, data2)
        schema_diff = self._calculate_schema_diff(data1, data2)
        stats_diff = self._calculate_stats_diff(
            v1.metadata.data_stats,
            v2.metadata.data_stats
        )
        
        return VersionDiff(
            version1_id=version1,
            version2_id=version2,
            content_changes=content_diff,
            schema_changes=schema_diff,
            stats_changes=stats_diff,
            quality_changes=self._calculate_quality_diff(
                v1.metadata.quality_metrics,
                v2.metadata.quality_metrics
            )
        )
    
    async def get_version_history(
        self,
        dataset_id: str,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[DataVersion]:
        """
        Get version history for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            limit: Maximum number of versions to return
            start_date: Filter versions after this date
            end_date: Filter versions before this date
        
        Returns:
            List of DataVersion instances
        """
        if dataset_id not in self.versions:
            return []
        
        versions = list(self.versions[dataset_id].values())
        
        # Apply date filters
        if start_date:
            versions = [v for v in versions if v.created_at >= start_date]
        if end_date:
            versions = [v for v in versions if v.created_at <= end_date]
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        # Apply limit
        if limit:
            versions = versions[:limit]
        
        return versions
    
    async def prune_versions(
        self,
        dataset_id: str,
        keep_last: int = 10,
        keep_days: Optional[int] = None
    ) -> int:
        """
        Prune old versions based on retention policy.
        
        Args:
            dataset_id: Dataset identifier
            keep_last: Number of recent versions to keep
            keep_days: Keep versions from last N days
        
        Returns:
            Number of versions pruned
        """
        if dataset_id not in self.versions:
            return 0
        
        versions = list(self.versions[dataset_id].values())
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        # Determine versions to keep
        versions_to_keep = set()
        
        # Keep last N versions
        for v in versions[:keep_last]:
            versions_to_keep.add(v.version_id)
        
        # Keep versions within date range
        if keep_days:
            cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
            for v in versions:
                if v.created_at >= cutoff_date:
                    versions_to_keep.add(v.version_id)
        
        # Keep current version
        current_version = self.current_versions.get(dataset_id)
        if current_version:
            versions_to_keep.add(current_version)
        
        # Prune versions
        pruned_count = 0
        for version in versions:
            if version.version_id not in versions_to_keep:
                await self._delete_version(dataset_id, version.version_id)
                pruned_count += 1
        
        logger.info(f"Pruned {pruned_count} versions for dataset {dataset_id}")
        return pruned_count
    
    def _determine_strategy(
        self,
        dataset_id: str,
        data: Any,
        context: VersioningContext
    ) -> VersioningStrategy:
        """Determine appropriate versioning strategy."""
        # Use branch strategy for experimental versions
        if context.branch and context.branch != "main":
            return VersioningStrategy.BRANCH
        
        # Use timestamp for time series data
        if self._is_time_series_data(data):
            return VersioningStrategy.TIMESTAMP
        
        # Use hash for content-based versioning
        if context.trigger == VersionTrigger.DATA_SOURCE_CHANGE:
            return VersioningStrategy.HASH
        
        # Default to semantic versioning
        return VersioningStrategy.SEMANTIC
    
    async def _generate_version_id(
        self,
        dataset_id: str,
        data: Any,
        strategy: VersioningStrategy,
        context: VersioningContext
    ) -> str:
        """Generate version ID based on strategy."""
        if strategy == VersioningStrategy.SEMANTIC:
            return self._generate_semantic_version(dataset_id, context)
        elif strategy == VersioningStrategy.TIMESTAMP:
            return f"{dataset_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}"
        elif strategy == VersioningStrategy.HASH:
            fingerprint = await self.fingerprinter.calculate_fingerprint(data)
            return f"{dataset_id}-{fingerprint[:12]}"
        elif strategy == VersioningStrategy.BRANCH:
            timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            return f"{dataset_id}-{context.branch}-{timestamp}"
        else:
            raise ValueError(f"Unknown versioning strategy: {strategy}")
    
    def _generate_semantic_version(
        self,
        dataset_id: str,
        context: VersioningContext
    ) -> str:
        """Generate semantic version number."""
        # Get current version
        current_version = self.current_versions.get(dataset_id)
        
        if not current_version or not current_version.startswith(f"{dataset_id}-"):
            # First version
            return f"{dataset_id}-1.0.0"
        
        # Parse current version
        try:
            version_part = current_version.replace(f"{dataset_id}-", "")
            parts = version_part.split(".")
            
            if len(parts) != 3:
                return f"{dataset_id}-1.0.0"
            
            major, minor, patch = map(int, parts)
            
            # Determine version increment based on trigger
            if context.trigger == VersionTrigger.SCHEMA_MODIFICATION:
                major += 1
                minor = 0
                patch = 0
            elif context.trigger in [VersionTrigger.TRANSFORMATION_CHANGE, 
                                   VersionTrigger.DATA_SOURCE_CHANGE]:
                minor += 1
                patch = 0
            else:
                patch += 1
            
            return f"{dataset_id}-{major}.{minor}.{patch}"
            
        except Exception:
            # Fallback to new version
            return f"{dataset_id}-1.0.0"
    
    def _calculate_data_stats(self, data: Any) -> Dict[str, Any]:
        """Calculate statistics about the data."""
        stats = {
            "type": type(data).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if hasattr(data, '__len__'):
            stats["size"] = len(data)
        
        if hasattr(data, 'memory_usage'):
            stats["memory_bytes"] = data.memory_usage()
        
        # Add type-specific stats
        if isinstance(data, list):
            stats["element_count"] = len(data)
        elif hasattr(data, 'shape'):  # DataFrame-like
            stats["shape"] = data.shape
        elif isinstance(data, dict):
            stats["key_count"] = len(data)
        
        return stats
    
    async def _generate_change_summary(
        self,
        dataset_id: str,
        data: Any,
        parent_version: Optional[str]
    ) -> Dict[str, Any]:
        """Generate summary of changes from parent version."""
        if not parent_version:
            return {"type": "initial_version"}
        
        parent_data = await self.get_version_data(dataset_id, parent_version)
        if not parent_data:
            return {"type": "parent_not_found"}
        
        # Calculate differences
        content_diff = await self.fingerprinter.calculate_content_diff(data, parent_data)
        
        return {
            "type": "incremental",
            "content_changes": content_diff,
            "size_change": self._calculate_size_change(data, parent_data)
        }
    
    def _calculate_size_change(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """Calculate size changes between data versions."""
        size1 = len(data1) if hasattr(data1, '__len__') else 0
        size2 = len(data2) if hasattr(data2, '__len__') else 0
        
        return {
            "old_size": size2,
            "new_size": size1,
            "change": size1 - size2,
            "change_percent": ((size1 - size2) / size2 * 100) if size2 > 0 else 0
        }
    
    def _find_version_by_fingerprint(
        self,
        dataset_id: str,
        fingerprint: str
    ) -> Optional[DataVersion]:
        """Find existing version with same fingerprint."""
        if dataset_id not in self.versions:
            return None
        
        for version in self.versions[dataset_id].values():
            if version.fingerprint == fingerprint:
                return version
        
        return None
    
    def _strategy_to_type(self, strategy: VersioningStrategy) -> VersionType:
        """Convert versioning strategy to version type."""
        mapping = {
            VersioningStrategy.SEMANTIC: VersionType.MAJOR,
            VersioningStrategy.TIMESTAMP: VersionType.TIMESTAMP,
            VersioningStrategy.HASH: VersionType.CONTENT,
            VersioningStrategy.BRANCH: VersionType.BRANCH
        }
        return mapping.get(strategy, VersionType.MINOR)
    
    def _is_time_series_data(self, data: Any) -> bool:
        """Check if data is time series."""
        # Simple heuristic - can be extended
        if hasattr(data, 'index') and hasattr(data.index, 'is_monotonic_increasing'):
            return True
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if hasattr(first_item, 'timestamp'):
                return True
        return False
    
    async def _cleanup_old_versions(self, dataset_id: str) -> None:
        """Clean up old versions based on retention policy."""
        if not self.config.auto_cleanup:
            return
        
        # Apply retention policy
        await self.prune_versions(
            dataset_id,
            keep_last=self.config.retention_versions,
            keep_days=self.config.retention_days
        )
    
    async def _validate_rollback(
        self,
        dataset_id: str,
        from_version: str,
        to_version: str
    ) -> Any:
        """Validate rollback operation."""
        # Placeholder for rollback validation logic
        # Would check dependencies, downstream impacts, etc.
        class ValidationResult:
            is_valid = True
            reason = None
        
        return ValidationResult()
    
    async def _mark_versions_as_rolled_back(
        self,
        dataset_id: str,
        from_version: str,
        to_version: str
    ) -> None:
        """Mark intermediate versions as rolled back."""
        # Get version chain between from and to
        versions = self.versions.get(dataset_id, {})
        
        for version_id, version in versions.items():
            # Simple check - can be enhanced with proper graph traversal
            if version.created_at > versions[to_version].created_at:
                version.status = VersionStatus.ROLLED_BACK
    
    def _calculate_schema_diff(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """Calculate schema differences between data versions."""
        # Placeholder - would implement actual schema comparison
        return {"changes": []}
    
    def _calculate_stats_diff(
        self,
        stats1: Dict[str, Any],
        stats2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate statistics differences."""
        diff = {}
        
        all_keys = set(stats1.keys()) | set(stats2.keys())
        for key in all_keys:
            if key not in stats1:
                diff[key] = {"added": stats2[key]}
            elif key not in stats2:
                diff[key] = {"removed": stats1[key]}
            elif stats1[key] != stats2[key]:
                diff[key] = {"old": stats1[key], "new": stats2[key]}
        
        return diff
    
    def _calculate_quality_diff(
        self,
        quality1: Dict[str, Any],
        quality2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate quality metric differences."""
        return self._calculate_stats_diff(quality1, quality2)
    
    async def _save_snapshot(self, dataset_id: str, snapshot: VersionSnapshot) -> None:
        """Save snapshot to persistent storage."""
        if not self.config.storage_backend:
            return
        
        storage_path = Path(self.config.storage_path) / dataset_id
        storage_path.mkdir(parents=True, exist_ok=True)
        
        snapshot_file = storage_path / f"{snapshot.version_id}.pkl"
        with open(snapshot_file, 'wb') as f:
            pickle.dump(snapshot, f)
    
    async def _load_snapshot(
        self,
        dataset_id: str,
        version_id: str
    ) -> Optional[VersionSnapshot]:
        """Load snapshot from persistent storage."""
        if not self.config.storage_backend:
            return None
        
        snapshot_file = Path(self.config.storage_path) / dataset_id / f"{version_id}.pkl"
        if not snapshot_file.exists():
            return None
        
        try:
            with open(snapshot_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None
    
    def _load_versions_from_storage(self) -> None:
        """Load existing versions from storage."""
        # Placeholder - would implement actual loading logic
        pass
    
    async def _delete_version(self, dataset_id: str, version_id: str) -> None:
        """Delete a version and its snapshot."""
        # Remove from memory
        if dataset_id in self.versions and version_id in self.versions[dataset_id]:
            del self.versions[dataset_id][version_id]
        
        if dataset_id in self.snapshots and version_id in self.snapshots[dataset_id]:
            del self.snapshots[dataset_id][version_id]
        
        # Remove from storage
        if self.config.storage_backend:
            snapshot_file = Path(self.config.storage_path) / dataset_id / f"{version_id}.pkl"
            if snapshot_file.exists():
                snapshot_file.unlink()
    
    def get_version_statistics(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """Get versioning statistics."""
        if dataset_id:
            versions = self.versions.get(dataset_id, {})
            return {
                "dataset_id": dataset_id,
                "total_versions": len(versions),
                "current_version": self.current_versions.get(dataset_id),
                "active_versions": sum(1 for v in versions.values() 
                                     if v.status == VersionStatus.ACTIVE),
                "storage_used": sum(
                    len(str(s.data)) for s in self.snapshots.get(dataset_id, {}).values()
                )
            }
        else:
            # Global statistics
            total_versions = sum(len(v) for v in self.versions.values())
            total_storage = sum(
                len(str(s.data)) 
                for snapshots in self.snapshots.values() 
                for s in snapshots.values()
            )
            
            return {
                "total_datasets": len(self.versions),
                "total_versions": total_versions,
                "total_storage_bytes": total_storage,
                "datasets": list(self.versions.keys())
            }


# Global instance
_version_manager: Optional[DataVersionManager] = None


def get_version_manager(config: Optional[VersioningConfig] = None) -> DataVersionManager:
    """Get the global version manager instance."""
    global _version_manager
    
    if _version_manager is None:
        _version_manager = DataVersionManager(config)
    
    return _version_manager