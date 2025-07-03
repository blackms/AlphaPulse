"""
Lineage Service for comprehensive data lineage management.

Provides:
- High-level lineage operations
- Integration with versioning system
- Lineage visualization and reporting
- Governance and compliance tracking
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from alpha_pulse.data.lineage.lineage_tracker import LineageTracker, LineageContext, LineageEventType
from alpha_pulse.data.lineage.dependency_graph import (
    DependencyGraphVisualizer, DependencyGraphAnalyzer,
    GraphLayout, GraphStyle
)
from alpha_pulse.data.versioning.data_version_manager import (
    DataVersionManager, VersioningContext, VersionTrigger
)
from alpha_pulse.models.lineage_metadata import (
    LineageNode, LineageGraph, ImpactAnalysis, LineageReport,
    TransformationType, DependencyType
)
from alpha_pulse.models.data_version import DataVersion
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


@dataclass
class LineageOperationResult:
    """Result of a lineage operation."""
    success: bool
    operation_type: str
    details: Dict[str, Any]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "operation_type": self.operation_type,
            "details": self.details,
            "error": self.error
        }


class LineageService:
    """Service for managing data lineage and dependencies."""
    
    def __init__(self):
        self.lineage_tracker = LineageTracker()
        self.version_manager = DataVersionManager()
        self.audit_logger = get_audit_logger()
        
        # Visualization components
        self.graph_visualizer = DependencyGraphVisualizer()
        
        # Governance tracking
        self.data_catalog: Dict[str, Dict[str, Any]] = {}
        self.compliance_registry: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.operation_metrics: Dict[str, List[float]] = defaultdict(list)
    
    async def track_data_operation(
        self,
        operation_type: LineageEventType,
        input_datasets: List[str],
        output_datasets: List[str],
        operator: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LineageOperationResult:
        """
        Track a data operation in the lineage system.
        
        Args:
            operation_type: Type of operation
            input_datasets: Input dataset IDs
            output_datasets: Output dataset IDs
            operator: System or user performing operation
            metadata: Additional operation metadata
        
        Returns:
            LineageOperationResult
        """
        start_time = datetime.utcnow()
        
        try:
            # Create lineage context
            context = LineageContext(
                operation_id=f"op_{start_time.timestamp()}",
                operation_type=operation_type,
                timestamp=start_time,
                operator=operator,
                metadata=metadata or {}
            )
            
            # Ensure all datasets have nodes
            all_datasets = set(input_datasets + output_datasets)
            for dataset_id in all_datasets:
                await self._ensure_lineage_node(dataset_id, operator)
            
            # Determine transformation type
            transformation_type = self._map_operation_to_transformation(operation_type)
            
            # Track transformation
            transformation = await self.lineage_tracker.track_transformation(
                context=context,
                input_nodes=input_datasets,
                output_nodes=output_datasets,
                transformation_type=transformation_type,
                operation_details=metadata
            )
            
            # Create versions for output datasets
            for output_dataset in output_datasets:
                version_context = VersioningContext(
                    trigger=VersionTrigger.TRANSFORMATION_CHANGE,
                    description=f"{operation_type.value} operation",
                    author=operator,
                    metadata={
                        "transformation_id": transformation.transformation_id,
                        "input_datasets": input_datasets
                    }
                )
                
                # Get output data (would be passed in real implementation)
                output_data = metadata.get("output_data", {})
                
                await self.version_manager.create_version(
                    dataset_id=output_dataset,
                    data=output_data,
                    context=version_context
                )
            
            # Update data catalog
            self._update_catalog(all_datasets, operation_type, context)
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.operation_metrics[operation_type.value].append(duration)
            
            return LineageOperationResult(
                success=True,
                operation_type=operation_type.value,
                details={
                    "transformation_id": transformation.transformation_id,
                    "input_count": len(input_datasets),
                    "output_count": len(output_datasets),
                    "duration_seconds": duration
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to track data operation: {e}")
            return LineageOperationResult(
                success=False,
                operation_type=operation_type.value,
                details={},
                error=str(e)
            )
    
    async def analyze_data_impact(
        self,
        dataset_id: str,
        change_type: str = "update"
    ) -> ImpactAnalysis:
        """
        Analyze the impact of changes to a dataset.
        
        Args:
            dataset_id: Dataset to analyze
            change_type: Type of change being considered
        
        Returns:
            ImpactAnalysis instance
        """
        # Get impact analysis from lineage tracker
        impact = await self.lineage_tracker.analyze_impact(dataset_id, change_type)
        
        # Enhance with version information
        affected_versions = await self._get_affected_versions(
            dataset_id,
            impact.direct_impacts + impact.indirect_impacts
        )
        
        impact.details = impact.details or {}
        impact.details["affected_versions"] = affected_versions
        
        # Add governance considerations
        compliance_impacts = self._analyze_compliance_impact(
            dataset_id,
            impact.affected_systems
        )
        
        if compliance_impacts:
            impact.recommendations.append(
                f"Compliance review required for: {', '.join(compliance_impacts)}"
            )
        
        # Log impact analysis
        self.audit_logger.log(
            event_type=AuditEventType.IMPACT_ANALYSIS,
            event_data={
                "dataset_id": dataset_id,
                "change_type": change_type,
                "total_impacts": impact.get_total_impact_count(),
                "risk_score": impact.risk_score
            },
            severity=AuditSeverity.INFO
        )
        
        return impact
    
    async def get_data_lineage_graph(
        self,
        dataset_id: str,
        direction: str = "both",
        max_depth: int = 5
    ) -> LineageGraph:
        """
        Get lineage graph for a dataset.
        
        Args:
            dataset_id: Dataset ID
            direction: 'upstream', 'downstream', or 'both'
            max_depth: Maximum traversal depth
        
        Returns:
            LineageGraph instance
        """
        graphs = []
        
        if direction in ["upstream", "both"]:
            upstream_graph = await self.lineage_tracker.get_upstream_lineage(
                dataset_id, max_depth
            )
            graphs.append(upstream_graph)
        
        if direction in ["downstream", "both"]:
            downstream_graph = await self.lineage_tracker.get_downstream_lineage(
                dataset_id, max_depth
            )
            graphs.append(downstream_graph)
        
        # Merge graphs if both directions
        if len(graphs) == 2:
            merged_nodes = {**graphs[0].nodes, **graphs[1].nodes}
            merged_edges = {**graphs[0].edges, **graphs[1].edges}
            
            return LineageGraph(
                graph_id=f"lineage_{dataset_id}_{datetime.utcnow().timestamp()}",
                nodes=merged_nodes,
                edges=merged_edges,
                metadata={
                    "root_node": dataset_id,
                    "direction": direction,
                    "max_depth": max_depth
                }
            )
        else:
            return graphs[0]
    
    async def visualize_lineage(
        self,
        dataset_id: str,
        output_format: str = "plotly",
        output_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Visualize data lineage.
        
        Args:
            dataset_id: Dataset to visualize
            output_format: 'plotly', 'matplotlib', 'graphviz', 'cytoscape'
            output_path: Path to save visualization
            **kwargs: Additional visualization parameters
        
        Returns:
            Visualization object or None
        """
        # Get lineage graph
        lineage_graph = await self.get_data_lineage_graph(
            dataset_id,
            direction=kwargs.get("direction", "both"),
            max_depth=kwargs.get("max_depth", 5)
        )
        
        if output_format == "plotly":
            return self.graph_visualizer.visualize_plotly(
                lineage_graph,
                title=f"Data Lineage: {dataset_id}"
            )
        
        elif output_format == "matplotlib":
            return self.graph_visualizer.visualize_matplotlib(
                lineage_graph,
                output_path=output_path,
                title=f"Data Lineage: {dataset_id}"
            )
        
        elif output_format == "graphviz":
            self.graph_visualizer.export_to_graphviz(
                lineage_graph,
                output_path=output_path or f"{dataset_id}_lineage.dot"
            )
        
        elif output_format == "cytoscape":
            self.graph_visualizer.export_to_cytoscape(
                lineage_graph,
                output_path=output_path or f"{dataset_id}_lineage.json"
            )
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    async def generate_lineage_report(
        self,
        scope: Optional[List[str]] = None
    ) -> LineageReport:
        """
        Generate comprehensive lineage report.
        
        Args:
            scope: List of systems/datasets to include (None for all)
        
        Returns:
            LineageReport instance
        """
        # Get all nodes and edges
        all_nodes = self.lineage_tracker.nodes
        all_edges = self.lineage_tracker.edges
        
        # Filter by scope if provided
        if scope:
            all_nodes = {
                nid: node for nid, node in all_nodes.items()
                if node.system in scope or nid in scope
            }
        
        # Calculate statistics
        nodes_by_type = defaultdict(int)
        nodes_by_system = defaultdict(int)
        quality_scores = []
        
        for node in all_nodes.values():
            nodes_by_type[node.node_type] += 1
            nodes_by_system[node.system] += 1
            
            if node.quality_score is not None:
                quality_scores.append(node.quality_score)
        
        # Analyze graph
        graph = LineageGraph(
            graph_id="report_graph",
            nodes=all_nodes,
            edges=all_edges
        )
        
        analyzer = DependencyGraphAnalyzer(graph)
        metrics = analyzer.analyze_graph_metrics()
        
        # Identify issues
        isolated_nodes = metrics["connectivity_metrics"]["isolated_nodes"]
        critical_paths = analyzer.find_critical_paths()
        
        # Quality analysis
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
        low_quality_nodes = [
            nid for nid, node in all_nodes.items()
            if node.quality_score is not None and node.quality_score < 0.7
        ]
        
        # Compliance analysis
        non_compliant = [
            nid for nid, node in all_nodes.items()
            if node.compliance_status == "non_compliant"
        ]
        
        sensitive_data = [
            nid for nid, node in all_nodes.items()
            if node.sensitivity_level in ["high", "critical"]
        ]
        
        # Generate recommendations
        recommendations = self._generate_lineage_recommendations(
            metrics, isolated_nodes, critical_paths, low_quality_nodes
        )
        
        return LineageReport(
            report_id=f"lineage_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow(),
            total_nodes=len(all_nodes),
            total_edges=len(all_edges),
            systems_count=len(set(node.system for node in all_nodes.values())),
            nodes_by_type=dict(nodes_by_type),
            nodes_by_system=dict(nodes_by_system),
            average_quality_score=avg_quality,
            nodes_below_quality_threshold=low_quality_nodes,
            isolated_nodes=isolated_nodes,
            highly_connected_nodes=metrics["centrality_metrics"]["most_depended_upon"][:10],
            critical_paths=critical_paths[:5],
            non_compliant_nodes=non_compliant,
            sensitive_data_nodes=sensitive_data,
            optimization_opportunities=recommendations["optimizations"],
            risk_areas=recommendations["risks"]
        )
    
    async def perform_data_rollback(
        self,
        dataset_id: str,
        target_version: str,
        cascade: bool = False
    ) -> Tuple[bool, str]:
        """
        Perform data rollback with lineage awareness.
        
        Args:
            dataset_id: Dataset to rollback
            target_version: Target version ID
            cascade: Whether to cascade rollback to dependent datasets
        
        Returns:
            Tuple of (success, message)
        """
        # Analyze impact first
        impact = await self.analyze_data_impact(dataset_id, "rollback")
        
        if impact.is_high_risk() and not cascade:
            return False, "High-risk rollback requires cascade=True"
        
        # Perform primary rollback
        success, error = await self.version_manager.rollback(
            dataset_id=dataset_id,
            target_version=target_version,
            reason="Lineage-aware rollback",
            validate=True
        )
        
        if not success:
            return False, f"Rollback failed: {error}"
        
        # Handle cascade if requested
        if cascade and impact.direct_impacts:
            cascade_results = await self._cascade_rollback(
                impact.direct_impacts,
                dataset_id,
                target_version
            )
            
            if not all(r[0] for r in cascade_results):
                failed = [r[1] for r in cascade_results if not r[0]]
                return False, f"Cascade rollback partially failed: {failed}"
        
        # Update lineage
        await self._update_lineage_after_rollback(dataset_id, target_version)
        
        return True, "Rollback completed successfully"
    
    async def _ensure_lineage_node(self, dataset_id: str, system: str) -> None:
        """Ensure a lineage node exists for a dataset."""
        if dataset_id not in self.lineage_tracker.nodes:
            await self.lineage_tracker.create_node(
                node_id=dataset_id,
                node_type="dataset",
                name=dataset_id,
                system=system,
                metadata={
                    "created_by": "lineage_service",
                    "auto_created": True
                }
            )
    
    def _map_operation_to_transformation(
        self,
        operation_type: LineageEventType
    ) -> TransformationType:
        """Map lineage event type to transformation type."""
        mapping = {
            LineageEventType.TRANSFORMATION: TransformationType.CUSTOM,
            LineageEventType.AGGREGATION: TransformationType.AGGREGATE,
            LineageEventType.FILTERING: TransformationType.FILTER,
            LineageEventType.JOIN: TransformationType.JOIN,
            LineageEventType.SPLIT: TransformationType.PIVOT,
            LineageEventType.VALIDATION: TransformationType.VALIDATE,
            LineageEventType.ENRICHMENT: TransformationType.ENRICH
        }
        
        return mapping.get(operation_type, TransformationType.CUSTOM)
    
    def _update_catalog(
        self,
        datasets: set,
        operation_type: LineageEventType,
        context: LineageContext
    ) -> None:
        """Update data catalog with operation information."""
        for dataset_id in datasets:
            if dataset_id not in self.data_catalog:
                self.data_catalog[dataset_id] = {
                    "first_seen": datetime.utcnow(),
                    "operations": [],
                    "usage_count": 0,
                    "last_accessed": None
                }
            
            catalog_entry = self.data_catalog[dataset_id]
            catalog_entry["operations"].append({
                "type": operation_type.value,
                "timestamp": context.timestamp,
                "operator": context.operator
            })
            catalog_entry["usage_count"] += 1
            catalog_entry["last_accessed"] = datetime.utcnow()
    
    async def _get_affected_versions(
        self,
        source_dataset: str,
        affected_datasets: List[str]
    ) -> Dict[str, str]:
        """Get current versions of affected datasets."""
        versions = {}
        
        for dataset_id in affected_datasets:
            version = await self.version_manager.get_version(dataset_id)
            if version:
                versions[dataset_id] = version.version_id
        
        return versions
    
    def _analyze_compliance_impact(
        self,
        dataset_id: str,
        affected_systems: List[str]
    ) -> List[str]:
        """Analyze compliance impact of changes."""
        compliance_issues = []
        
        # Check if dataset has compliance requirements
        if dataset_id in self.compliance_registry:
            requirements = self.compliance_registry[dataset_id]
            
            if "retention_required" in requirements:
                compliance_issues.append("Data retention policy")
            
            if "audit_required" in requirements:
                compliance_issues.append("Audit trail maintenance")
        
        # Check affected systems
        regulated_systems = ["trading", "risk", "compliance"]
        affected_regulated = [s for s in affected_systems if s in regulated_systems]
        
        if affected_regulated:
            compliance_issues.append(f"Regulated systems: {', '.join(affected_regulated)}")
        
        return compliance_issues
    
    def _generate_lineage_recommendations(
        self,
        metrics: Dict[str, Any],
        isolated_nodes: List[str],
        critical_paths: List[List[str]],
        low_quality_nodes: List[str]
    ) -> Dict[str, List[str]]:
        """Generate recommendations based on lineage analysis."""
        recommendations = {
            "optimizations": [],
            "risks": []
        }
        
        # Optimization opportunities
        if isolated_nodes:
            recommendations["optimizations"].append(
                f"Remove or connect {len(isolated_nodes)} isolated nodes"
            )
        
        if metrics["basic_stats"]["density"] < 0.1:
            recommendations["optimizations"].append(
                "Consider consolidating sparse data dependencies"
            )
        
        # Risk areas
        if len(critical_paths) > 10:
            recommendations["risks"].append(
                "High number of critical paths indicates fragile architecture"
            )
        
        if low_quality_nodes:
            recommendations["risks"].append(
                f"{len(low_quality_nodes)} nodes have low quality scores"
            )
        
        if not metrics["basic_stats"]["is_dag"]:
            recommendations["risks"].append(
                "Circular dependencies detected - review data flow"
            )
        
        return recommendations
    
    async def _cascade_rollback(
        self,
        affected_datasets: List[str],
        source_dataset: str,
        source_version: str
    ) -> List[Tuple[bool, str]]:
        """Perform cascading rollback on affected datasets."""
        results = []
        
        for dataset_id in affected_datasets:
            # Determine appropriate rollback version
            # (In practice, this would be more sophisticated)
            success, error = await self.version_manager.rollback(
                dataset_id=dataset_id,
                target_version=source_version,  # Simplified
                reason=f"Cascade from {source_dataset}",
                validate=True
            )
            
            results.append((success, dataset_id if not success else ""))
        
        return results
    
    async def _update_lineage_after_rollback(
        self,
        dataset_id: str,
        target_version: str
    ) -> None:
        """Update lineage information after rollback."""
        # Update node metadata
        if dataset_id in self.lineage_tracker.nodes:
            node = self.lineage_tracker.nodes[dataset_id]
            node.data_version = target_version
            node.metadata.properties["last_rollback"] = datetime.utcnow().isoformat()
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "lineage_stats": self.lineage_tracker.get_lineage_statistics(),
            "version_stats": self.version_manager.get_version_statistics(),
            "catalog_size": len(self.data_catalog),
            "operation_metrics": {
                op: {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times) if times else 0
                }
                for op, times in self.operation_metrics.items()
            }
        }


# Global instance
_lineage_service: Optional[LineageService] = None


def get_lineage_service() -> LineageService:
    """Get the global lineage service instance."""
    global _lineage_service
    
    if _lineage_service is None:
        _lineage_service = LineageService()
    
    return _lineage_service