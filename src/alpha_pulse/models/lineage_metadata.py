"""
Lineage metadata models for data lineage tracking.

Provides:
- Lineage graph structures
- Node and edge definitions
- Transformation tracking models
- Impact analysis models
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from alpha_pulse.models.base import BaseModel


class DependencyType(Enum):
    """Types of data dependencies."""
    DIRECT = "direct"  # Direct data flow
    TRANSFORMATION = "transformation"  # Data transformation
    DERIVATION = "derivation"  # Derived dataset
    AGGREGATION = "aggregation"  # Aggregated from multiple sources
    REFERENCE = "reference"  # Reference/lookup data
    VALIDATION = "validation"  # Validation dependency


class TransformationType(Enum):
    """Types of data transformations."""
    FILTER = "filter"
    MAP = "map"
    AGGREGATE = "aggregate"
    JOIN = "join"
    UNION = "union"
    PIVOT = "pivot"
    NORMALIZE = "normalize"
    ENRICH = "enrich"
    VALIDATE = "validate"
    CUSTOM = "custom"


@dataclass
class LineageMetadata:
    """Metadata for lineage tracking."""
    created_at: datetime
    created_by: str
    version: str
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
            "tags": self.tags,
            "properties": self.properties
        }


@dataclass
class LineageNode(BaseModel):
    """Represents a node in the lineage graph."""
    node_id: str
    node_type: str  # dataset, process, model, report, etc.
    name: str
    system: str  # System that owns this node
    created_at: datetime
    metadata: LineageMetadata
    
    # Optional attributes
    description: Optional[str] = None
    schema_version: Optional[str] = None
    data_version: Optional[str] = None
    location: Optional[str] = None
    
    # Relationships
    upstream_nodes: List[str] = field(default_factory=list)
    downstream_nodes: List[str] = field(default_factory=list)
    
    # Quality and compliance
    quality_score: Optional[float] = None
    compliance_status: Optional[str] = None
    sensitivity_level: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "system": self.system,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata.to_dict(),
            "description": self.description,
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "location": self.location,
            "upstream_nodes": self.upstream_nodes,
            "downstream_nodes": self.downstream_nodes,
            "quality_score": self.quality_score,
            "compliance_status": self.compliance_status,
            "sensitivity_level": self.sensitivity_level
        }


@dataclass
class LineageEdge(BaseModel):
    """Represents an edge in the lineage graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: DependencyType
    created_at: datetime
    
    # Optional attributes
    transformation_id: Optional[str] = None
    transformation_type: Optional[TransformationType] = None
    
    # Edge properties
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "created_at": self.created_at.isoformat(),
            "transformation_id": self.transformation_id,
            "transformation_type": self.transformation_type.value if self.transformation_type else None,
            "metadata": self.metadata
        }


@dataclass
class LineageGraph:
    """Represents a lineage graph."""
    graph_id: str
    nodes: Dict[str, LineageNode]
    edges: Dict[str, LineageEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph_id": self.graph_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "metadata": self.metadata
        }
    
    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return len(self.edges)
    
    def get_systems(self) -> Set[str]:
        """Get all systems in the graph."""
        return {node.system for node in self.nodes.values()}


@dataclass
class DataFlow:
    """Represents data flow between nodes."""
    flow_id: str
    source_node: str
    target_node: str
    paths: List[List[str]]  # Multiple possible paths
    transformations: List[List[str]]  # Transformations for each path
    flow_metrics: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flow_id": self.flow_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "paths": self.paths,
            "transformations": self.transformations,
            "flow_metrics": self.flow_metrics,
            "created_at": self.created_at.isoformat()
        }
    
    def get_shortest_path(self) -> Optional[List[str]]:
        """Get the shortest path."""
        if not self.paths:
            return None
        return min(self.paths, key=len)
    
    def get_transformation_count(self) -> int:
        """Get total number of transformations."""
        return sum(len(t) for t in self.transformations)


@dataclass
class ImpactAnalysis:
    """Result of impact analysis for changes."""
    source_node: str
    change_type: str  # update, delete, schema_change, etc.
    direct_impacts: List[str]
    indirect_impacts: List[str]
    affected_systems: List[str]
    critical_paths: List[List[str]]
    risk_score: float  # 0-10 scale
    recommendations: List[str]
    analysis_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_node": self.source_node,
            "change_type": self.change_type,
            "direct_impacts": self.direct_impacts,
            "indirect_impacts": self.indirect_impacts,
            "affected_systems": self.affected_systems,
            "critical_paths": self.critical_paths,
            "risk_score": self.risk_score,
            "recommendations": self.recommendations,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }
    
    def get_total_impact_count(self) -> int:
        """Get total number of impacted nodes."""
        return len(set(self.direct_impacts + self.indirect_impacts))
    
    def is_high_risk(self) -> bool:
        """Check if change is high risk."""
        return self.risk_score >= 7.0


@dataclass
class LineageQuery:
    """Query parameters for lineage search."""
    # Node filters
    node_types: Optional[List[str]] = None
    systems: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    
    # Time filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Relationship filters
    has_upstream: Optional[bool] = None
    has_downstream: Optional[bool] = None
    is_isolated: Optional[bool] = None
    
    # Quality filters
    min_quality_score: Optional[float] = None
    compliance_status: Optional[str] = None
    
    # Search parameters
    name_pattern: Optional[str] = None
    max_results: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_types": self.node_types,
            "systems": self.systems,
            "tags": self.tags,
            "created_after": self.created_after.isoformat() if self.created_after else None,
            "created_before": self.created_before.isoformat() if self.created_before else None,
            "has_upstream": self.has_upstream,
            "has_downstream": self.has_downstream,
            "is_isolated": self.is_isolated,
            "min_quality_score": self.min_quality_score,
            "compliance_status": self.compliance_status,
            "name_pattern": self.name_pattern,
            "max_results": self.max_results
        }


@dataclass
class LineageReport:
    """Comprehensive lineage report."""
    report_id: str
    generated_at: datetime
    
    # Graph statistics
    total_nodes: int
    total_edges: int
    systems_count: int
    
    # Node breakdown
    nodes_by_type: Dict[str, int]
    nodes_by_system: Dict[str, int]
    
    # Quality metrics
    average_quality_score: Optional[float]
    nodes_below_quality_threshold: List[str]
    
    # Connectivity analysis
    isolated_nodes: List[str]
    highly_connected_nodes: List[Tuple[str, int]]  # (node_id, connection_count)
    critical_paths: List[List[str]]
    
    # Compliance
    non_compliant_nodes: List[str]
    sensitive_data_nodes: List[str]
    
    # Recommendations
    optimization_opportunities: List[str]
    risk_areas: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "systems_count": self.systems_count,
            "nodes_by_type": self.nodes_by_type,
            "nodes_by_system": self.nodes_by_system,
            "average_quality_score": self.average_quality_score,
            "nodes_below_quality_threshold": self.nodes_below_quality_threshold,
            "isolated_nodes": self.isolated_nodes,
            "highly_connected_nodes": [
                {"node_id": node, "connections": count} 
                for node, count in self.highly_connected_nodes
            ],
            "critical_paths": self.critical_paths,
            "non_compliant_nodes": self.non_compliant_nodes,
            "sensitive_data_nodes": self.sensitive_data_nodes,
            "optimization_opportunities": self.optimization_opportunities,
            "risk_areas": self.risk_areas
        }