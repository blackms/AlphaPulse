"""
Data Lineage Tracker for comprehensive tracking of data transformations and dependencies.

Provides:
- End-to-end data lineage tracking
- Transformation step documentation
- Dependency mapping and impact analysis
- Cross-system data flow tracking
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx
from loguru import logger

from alpha_pulse.models.lineage_metadata import (
    LineageNode, LineageEdge, TransformationType, LineageGraph,
    DataFlow, ImpactAnalysis, LineageMetadata, DependencyType
)
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class LineageEventType(Enum):
    """Types of lineage events."""
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    JOIN = "join"
    SPLIT = "split"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"


@dataclass
class LineageContext:
    """Context for lineage tracking."""
    operation_id: str
    operation_type: LineageEventType
    timestamp: datetime
    operator: str  # System or user performing operation
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TransformationRecord:
    """Record of a data transformation."""
    transformation_id: str
    transformation_type: TransformationType
    input_nodes: List[str]
    output_nodes: List[str]
    operation_details: Dict[str, Any]
    timestamp: datetime
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'transformation_id': self.transformation_id,
            'transformation_type': self.transformation_type.value,
            'input_nodes': self.input_nodes,
            'output_nodes': self.output_nodes,
            'operation_details': self.operation_details,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms
        }


class LineageTracker:
    """Comprehensive data lineage tracking system."""
    
    def __init__(self):
        self.audit_logger = get_audit_logger()
        
        # Lineage graph using NetworkX
        self.lineage_graph = nx.DiGraph()
        
        # Node and edge registries
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, LineageEdge] = {}
        
        # Transformation history
        self.transformations: Dict[str, TransformationRecord] = {}
        self.transformation_index: Dict[str, List[str]] = defaultdict(list)  # node -> transformations
        
        # Active operations tracking
        self.active_operations: Dict[str, LineageContext] = {}
        
        # Cross-system mappings
        self.system_mappings: Dict[str, Set[str]] = defaultdict(set)  # system -> nodes
        
        # Performance metrics
        self.operation_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Locks for concurrent access
        self.graph_lock = asyncio.Lock()
    
    async def create_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        system: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LineageNode:
        """
        Create a new lineage node.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (dataset, process, etc.)
            name: Human-readable name
            system: System owning the node
            metadata: Additional metadata
        
        Returns:
            Created LineageNode instance
        """
        async with self.graph_lock:
            # Check if node already exists
            if node_id in self.nodes:
                logger.warning(f"Node {node_id} already exists")
                return self.nodes[node_id]
            
            # Create lineage metadata
            lineage_metadata = LineageMetadata(
                created_at=datetime.utcnow(),
                created_by=system,
                version="1.0",
                tags=metadata.get('tags', []) if metadata else [],
                properties=metadata or {}
            )
            
            # Create node
            node = LineageNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                system=system,
                created_at=datetime.utcnow(),
                metadata=lineage_metadata
            )
            
            # Add to registries
            self.nodes[node_id] = node
            self.lineage_graph.add_node(node_id, data=node)
            self.system_mappings[system].add(node_id)
            
            # Log creation
            self.audit_logger.log(
                event_type=AuditEventType.LINEAGE_NODE_CREATED,
                event_data={
                    "node_id": node_id,
                    "node_type": node_type,
                    "system": system
                },
                severity=AuditSeverity.INFO
            )
            
            return node
    
    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: DependencyType,
        transformation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LineageEdge:
        """
        Create a lineage edge between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of dependency
            transformation_id: Associated transformation
            metadata: Additional metadata
        
        Returns:
            Created LineageEdge instance
        """
        async with self.graph_lock:
            # Validate nodes exist
            if source_id not in self.nodes:
                raise ValueError(f"Source node {source_id} not found")
            if target_id not in self.nodes:
                raise ValueError(f"Target node {target_id} not found")
            
            # Generate edge ID
            edge_id = f"{source_id}-{target_id}-{datetime.utcnow().timestamp()}"
            
            # Create edge
            edge = LineageEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                created_at=datetime.utcnow(),
                transformation_id=transformation_id,
                metadata=metadata or {}
            )
            
            # Add to graph and registry
            self.edges[edge_id] = edge
            self.lineage_graph.add_edge(
                source_id, target_id,
                edge_id=edge_id,
                data=edge
            )
            
            # Update node relationships
            self.nodes[source_id].downstream_nodes.append(target_id)
            self.nodes[target_id].upstream_nodes.append(source_id)
            
            return edge
    
    async def track_transformation(
        self,
        context: LineageContext,
        input_nodes: List[str],
        output_nodes: List[str],
        transformation_type: TransformationType,
        operation_details: Optional[Dict[str, Any]] = None
    ) -> TransformationRecord:
        """
        Track a data transformation operation.
        
        Args:
            context: Lineage context
            input_nodes: Input node IDs
            output_nodes: Output node IDs
            transformation_type: Type of transformation
            operation_details: Details about the operation
        
        Returns:
            TransformationRecord instance
        """
        # Create transformation record
        transformation = TransformationRecord(
            transformation_id=context.operation_id,
            transformation_type=transformation_type,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            operation_details=operation_details or {},
            timestamp=context.timestamp
        )
        
        # Store transformation
        self.transformations[transformation.transformation_id] = transformation
        
        # Index transformation by nodes
        for node_id in input_nodes + output_nodes:
            self.transformation_index[node_id].append(transformation.transformation_id)
        
        # Create edges for the transformation
        for input_node in input_nodes:
            for output_node in output_nodes:
                await self.create_edge(
                    source_id=input_node,
                    target_id=output_node,
                    edge_type=DependencyType.TRANSFORMATION,
                    transformation_id=transformation.transformation_id,
                    metadata={
                        "transformation_type": transformation_type.value,
                        "operator": context.operator
                    }
                )
        
        # Log transformation
        self.audit_logger.log(
            event_type=AuditEventType.LINEAGE_TRANSFORMATION,
            event_data={
                "transformation_id": transformation.transformation_id,
                "type": transformation_type.value,
                "input_count": len(input_nodes),
                "output_count": len(output_nodes)
            },
            severity=AuditSeverity.INFO
        )
        
        return transformation
    
    async def get_upstream_lineage(
        self,
        node_id: str,
        max_depth: Optional[int] = None,
        include_indirect: bool = True
    ) -> LineageGraph:
        """
        Get upstream lineage for a node.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
            include_indirect: Include indirect dependencies
        
        Returns:
            LineageGraph containing upstream lineage
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        # Use BFS to traverse upstream
        visited = set()
        nodes_to_include = set()
        edges_to_include = set()
        
        queue = deque([(node_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            nodes_to_include.add(current_id)
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            # Get upstream nodes
            if include_indirect:
                predecessors = list(self.lineage_graph.predecessors(current_id))
            else:
                predecessors = self.nodes[current_id].upstream_nodes
            
            for pred_id in predecessors:
                # Add edge
                for edge_data in self.lineage_graph.get_edge_data(pred_id, current_id, default={}).values():
                    if 'edge_id' in edge_data:
                        edges_to_include.add(edge_data['edge_id'])
                
                # Add to queue
                queue.append((pred_id, depth + 1))
        
        # Build lineage graph
        return self._build_lineage_graph(nodes_to_include, edges_to_include)
    
    async def get_downstream_lineage(
        self,
        node_id: str,
        max_depth: Optional[int] = None,
        include_indirect: bool = True
    ) -> LineageGraph:
        """
        Get downstream lineage for a node.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth
            include_indirect: Include indirect dependencies
        
        Returns:
            LineageGraph containing downstream lineage
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        # Use BFS to traverse downstream
        visited = set()
        nodes_to_include = set()
        edges_to_include = set()
        
        queue = deque([(node_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            nodes_to_include.add(current_id)
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            # Get downstream nodes
            if include_indirect:
                successors = list(self.lineage_graph.successors(current_id))
            else:
                successors = self.nodes[current_id].downstream_nodes
            
            for succ_id in successors:
                # Add edge
                for edge_data in self.lineage_graph.get_edge_data(current_id, succ_id, default={}).values():
                    if 'edge_id' in edge_data:
                        edges_to_include.add(edge_data['edge_id'])
                
                # Add to queue
                queue.append((succ_id, depth + 1))
        
        # Build lineage graph
        return self._build_lineage_graph(nodes_to_include, edges_to_include)
    
    async def analyze_impact(
        self,
        node_id: str,
        change_type: str = "update"
    ) -> ImpactAnalysis:
        """
        Analyze the impact of changes to a node.
        
        Args:
            node_id: Node to analyze
            change_type: Type of change (update, delete, etc.)
        
        Returns:
            ImpactAnalysis instance
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        # Get downstream lineage
        downstream_graph = await self.get_downstream_lineage(node_id)
        
        # Analyze direct impacts
        direct_impacts = self.nodes[node_id].downstream_nodes
        
        # Analyze indirect impacts
        all_downstream = set(downstream_graph.nodes.keys())
        all_downstream.discard(node_id)
        indirect_impacts = list(all_downstream - set(direct_impacts))
        
        # Identify critical paths
        critical_paths = []
        critical_nodes = self._identify_critical_nodes()
        
        for critical_node in critical_nodes:
            if critical_node in all_downstream:
                try:
                    paths = list(nx.all_simple_paths(
                        self.lineage_graph, node_id, critical_node, cutoff=5
                    ))
                    critical_paths.extend(paths)
                except nx.NetworkXNoPath:
                    pass
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            node_id, direct_impacts, indirect_impacts, change_type
        )
        
        # Generate recommendations
        recommendations = self._generate_impact_recommendations(
            node_id, direct_impacts, indirect_impacts, risk_score
        )
        
        return ImpactAnalysis(
            source_node=node_id,
            change_type=change_type,
            direct_impacts=direct_impacts,
            indirect_impacts=indirect_impacts,
            affected_systems=self._get_affected_systems(all_downstream),
            critical_paths=critical_paths,
            risk_score=risk_score,
            recommendations=recommendations,
            analysis_timestamp=datetime.utcnow()
        )
    
    async def get_data_flow(
        self,
        start_node: str,
        end_node: str,
        max_paths: int = 10
    ) -> DataFlow:
        """
        Get data flow paths between two nodes.
        
        Args:
            start_node: Starting node ID
            end_node: Ending node ID
            max_paths: Maximum number of paths to return
        
        Returns:
            DataFlow instance
        """
        if start_node not in self.nodes:
            raise ValueError(f"Start node {start_node} not found")
        if end_node not in self.nodes:
            raise ValueError(f"End node {end_node} not found")
        
        # Find all paths
        try:
            all_paths = list(nx.all_simple_paths(
                self.lineage_graph, start_node, end_node
            ))
        except nx.NetworkXNoPath:
            all_paths = []
        
        # Limit paths
        paths = all_paths[:max_paths]
        
        # Get transformations for each path
        path_transformations = []
        for path in paths:
            transformations = []
            for i in range(len(path) - 1):
                edge_data = self.lineage_graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    for edge_info in edge_data.values():
                        if 'data' in edge_info and edge_info['data'].transformation_id:
                            transformations.append(edge_info['data'].transformation_id)
            path_transformations.append(transformations)
        
        # Calculate flow metrics
        flow_metrics = {
            "total_paths": len(all_paths),
            "shortest_path_length": min(len(p) for p in paths) if paths else 0,
            "longest_path_length": max(len(p) for p in paths) if paths else 0,
            "average_path_length": sum(len(p) for p in paths) / len(paths) if paths else 0
        }
        
        return DataFlow(
            flow_id=f"{start_node}-to-{end_node}",
            source_node=start_node,
            target_node=end_node,
            paths=paths,
            transformations=path_transformations,
            flow_metrics=flow_metrics,
            created_at=datetime.utcnow()
        )
    
    async def search_lineage(
        self,
        criteria: Dict[str, Any]
    ) -> List[LineageNode]:
        """
        Search for nodes matching criteria.
        
        Args:
            criteria: Search criteria
        
        Returns:
            List of matching LineageNode instances
        """
        matching_nodes = []
        
        for node_id, node in self.nodes.items():
            if self._node_matches_criteria(node, criteria):
                matching_nodes.append(node)
        
        return matching_nodes
    
    def _build_lineage_graph(
        self,
        node_ids: Set[str],
        edge_ids: Set[str]
    ) -> LineageGraph:
        """Build a LineageGraph from node and edge IDs."""
        nodes = {nid: self.nodes[nid] for nid in node_ids if nid in self.nodes}
        edges = {eid: self.edges[eid] for eid in edge_ids if eid in self.edges}
        
        return LineageGraph(
            graph_id=f"lineage-{datetime.utcnow().timestamp()}",
            nodes=nodes,
            edges=edges,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "created_at": datetime.utcnow().isoformat()
            }
        )
    
    def _identify_critical_nodes(self) -> List[str]:
        """Identify critical nodes in the lineage graph."""
        critical_nodes = []
        
        # Nodes with high degree centrality
        centrality = nx.degree_centrality(self.lineage_graph)
        threshold = 0.1  # Top 10% most connected
        
        for node_id, score in centrality.items():
            if score >= threshold:
                critical_nodes.append(node_id)
        
        # Add nodes marked as critical in metadata
        for node_id, node in self.nodes.items():
            if node.metadata.properties.get('critical', False):
                if node_id not in critical_nodes:
                    critical_nodes.append(node_id)
        
        return critical_nodes
    
    def _calculate_risk_score(
        self,
        node_id: str,
        direct_impacts: List[str],
        indirect_impacts: List[str],
        change_type: str
    ) -> float:
        """Calculate risk score for a change."""
        base_score = 0.0
        
        # Factor in number of impacts
        base_score += len(direct_impacts) * 0.2
        base_score += len(indirect_impacts) * 0.1
        
        # Factor in node criticality
        if node_id in self._identify_critical_nodes():
            base_score += 0.3
        
        # Factor in change type
        change_multipliers = {
            "delete": 2.0,
            "schema_change": 1.5,
            "update": 1.0,
            "read": 0.5
        }
        multiplier = change_multipliers.get(change_type, 1.0)
        
        # Calculate final score (0-10 scale)
        risk_score = min(base_score * multiplier, 10.0)
        
        return risk_score
    
    def _get_affected_systems(self, node_ids: Union[Set[str], List[str]]) -> List[str]:
        """Get list of systems affected by nodes."""
        affected_systems = set()
        
        for node_id in node_ids:
            if node_id in self.nodes:
                affected_systems.add(self.nodes[node_id].system)
        
        return list(affected_systems)
    
    def _generate_impact_recommendations(
        self,
        node_id: str,
        direct_impacts: List[str],
        indirect_impacts: List[str],
        risk_score: float
    ) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        if risk_score >= 7.0:
            recommendations.append("High risk change - comprehensive testing required")
            recommendations.append("Consider phased rollout with monitoring")
        elif risk_score >= 4.0:
            recommendations.append("Medium risk - test direct downstream dependencies")
            recommendations.append("Monitor for 24 hours after deployment")
        
        if len(direct_impacts) > 10:
            recommendations.append("Many direct dependencies - consider refactoring")
        
        if len(indirect_impacts) > 50:
            recommendations.append("Extensive indirect impact - document changes thoroughly")
        
        critical_nodes = self._identify_critical_nodes()
        affected_critical = [n for n in direct_impacts + indirect_impacts if n in critical_nodes]
        if affected_critical:
            recommendations.append(f"Affects {len(affected_critical)} critical nodes - extra caution advised")
        
        return recommendations
    
    def _node_matches_criteria(
        self,
        node: LineageNode,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if node matches search criteria."""
        # Match by node type
        if 'node_type' in criteria and node.node_type != criteria['node_type']:
            return False
        
        # Match by system
        if 'system' in criteria and node.system != criteria['system']:
            return False
        
        # Match by tags
        if 'tags' in criteria:
            required_tags = set(criteria['tags'])
            node_tags = set(node.metadata.tags)
            if not required_tags.issubset(node_tags):
                return False
        
        # Match by date range
        if 'created_after' in criteria and node.created_at < criteria['created_after']:
            return False
        if 'created_before' in criteria and node.created_at > criteria['created_before']:
            return False
        
        # Match by name pattern
        if 'name_pattern' in criteria:
            import re
            if not re.search(criteria['name_pattern'], node.name):
                return False
        
        return True
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """Get statistics about the lineage graph."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_transformations": len(self.transformations),
            "systems": list(self.system_mappings.keys()),
            "graph_density": nx.density(self.lineage_graph),
            "connected_components": nx.number_weakly_connected_components(self.lineage_graph),
            "average_degree": sum(dict(self.lineage_graph.degree()).values()) / len(self.nodes) if self.nodes else 0,
            "critical_nodes": len(self._identify_critical_nodes())
        }


# Global instance
_lineage_tracker: Optional[LineageTracker] = None


def get_lineage_tracker() -> LineageTracker:
    """Get the global lineage tracker instance."""
    global _lineage_tracker
    
    if _lineage_tracker is None:
        _lineage_tracker = LineageTracker()
    
    return _lineage_tracker