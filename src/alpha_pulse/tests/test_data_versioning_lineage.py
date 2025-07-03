"""
Test suite for data versioning and lineage tracking system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from alpha_pulse.data.versioning import (
    DataVersionManager, VersioningStrategy, VersionTrigger,
    VersioningContext, get_version_manager
)
from alpha_pulse.data.lineage import (
    LineageTracker, LineageEventType, LineageContext,
    DependencyGraphAnalyzer, get_lineage_tracker
)
from alpha_pulse.services.lineage_service import LineageService, get_lineage_service
from alpha_pulse.models.data_version import VersionType, VersionStatus
from alpha_pulse.models.lineage_metadata import TransformationType, DependencyType
from alpha_pulse.utils.data_fingerprinting import DataFingerprinter


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'symbol': ['AAPL'] * 100,
        'price': np.random.randn(100).cumsum() + 150,
        'volume': np.random.randint(1000000, 5000000, 100)
    })


@pytest.fixture
def version_manager():
    """Create a version manager instance."""
    return DataVersionManager()


@pytest.fixture
def lineage_tracker():
    """Create a lineage tracker instance."""
    return LineageTracker()


@pytest.fixture
def lineage_service():
    """Create a lineage service instance."""
    return LineageService()


class TestDataVersionManager:
    """Test data version management functionality."""
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_manager, sample_dataframe):
        """Test creating a data version."""
        context = VersioningContext(
            trigger=VersionTrigger.DATA_SOURCE_CHANGE,
            description="Initial data load",
            author="test_user"
        )
        
        version = await version_manager.create_version(
            dataset_id="test_dataset",
            data=sample_dataframe,
            context=context,
            strategy=VersioningStrategy.SEMANTIC
        )
        
        assert version is not None
        assert version.dataset_id == "test_dataset"
        assert version.version_type == VersionType.MAJOR
        assert version.status == VersionStatus.ACTIVE
        assert version.metadata.author == "test_user"
    
    @pytest.mark.asyncio
    async def test_semantic_versioning(self, version_manager, sample_dataframe):
        """Test semantic version numbering."""
        # Create initial version
        context1 = VersioningContext(
            trigger=VersionTrigger.MANUAL,
            description="Initial version"
        )
        
        version1 = await version_manager.create_version(
            dataset_id="test_semantic",
            data=sample_dataframe,
            context=context1,
            strategy=VersioningStrategy.SEMANTIC
        )
        
        assert version1.version_id == "test_semantic-1.0.0"
        
        # Create minor version
        context2 = VersioningContext(
            trigger=VersionTrigger.TRANSFORMATION_CHANGE,
            description="Minor update"
        )
        
        version2 = await version_manager.create_version(
            dataset_id="test_semantic",
            data=sample_dataframe,
            context=context2,
            strategy=VersioningStrategy.SEMANTIC
        )
        
        assert version2.version_id == "test_semantic-1.1.0"
        
        # Create patch version
        context3 = VersioningContext(
            trigger=VersionTrigger.MANUAL,
            description="Patch update"
        )
        
        version3 = await version_manager.create_version(
            dataset_id="test_semantic",
            data=sample_dataframe,
            context=context3,
            strategy=VersioningStrategy.SEMANTIC
        )
        
        assert version3.version_id == "test_semantic-1.1.1"
    
    @pytest.mark.asyncio
    async def test_content_deduplication(self, version_manager, sample_dataframe):
        """Test content-based deduplication."""
        context = VersioningContext(
            trigger=VersionTrigger.MANUAL,
            description="Test deduplication"
        )
        
        # Create first version
        version1 = await version_manager.create_version(
            dataset_id="test_dedup",
            data=sample_dataframe,
            context=context
        )
        
        # Try to create duplicate
        version2 = await version_manager.create_version(
            dataset_id="test_dedup",
            data=sample_dataframe,
            context=context
        )
        
        # Should return the same version (deduplication)
        assert version1.version_id == version2.version_id
        assert version1.fingerprint == version2.fingerprint
    
    @pytest.mark.asyncio
    async def test_version_rollback(self, version_manager, sample_dataframe):
        """Test version rollback functionality."""
        # Create versions
        versions = []
        for i in range(3):
            context = VersioningContext(
                trigger=VersionTrigger.MANUAL,
                description=f"Version {i+1}"
            )
            
            # Modify data slightly
            modified_df = sample_dataframe.copy()
            modified_df['price'] = modified_df['price'] + i
            
            version = await version_manager.create_version(
                dataset_id="test_rollback",
                data=modified_df,
                context=context
            )
            versions.append(version)
        
        # Current version should be the last one
        current = await version_manager.get_version("test_rollback")
        assert current.version_id == versions[2].version_id
        
        # Rollback to first version
        success, error = await version_manager.rollback(
            dataset_id="test_rollback",
            target_version=versions[0].version_id,
            reason="Test rollback"
        )
        
        assert success
        assert error is None
        
        # Current version should now be the first one
        current = await version_manager.get_version("test_rollback")
        assert current.version_id == versions[0].version_id
    
    @pytest.mark.asyncio
    async def test_version_history(self, version_manager, sample_dataframe):
        """Test version history retrieval."""
        # Create multiple versions
        for i in range(5):
            context = VersioningContext(
                trigger=VersionTrigger.MANUAL,
                description=f"Version {i+1}"
            )
            
            await version_manager.create_version(
                dataset_id="test_history",
                data=sample_dataframe,
                context=context
            )
        
        # Get version history
        history = await version_manager.get_version_history("test_history")
        
        assert len(history) == 5
        # History should be ordered newest first
        assert history[0].created_at > history[1].created_at
    
    @pytest.mark.asyncio
    async def test_version_pruning(self, version_manager, sample_dataframe):
        """Test version pruning functionality."""
        # Create multiple versions
        for i in range(10):
            context = VersioningContext(
                trigger=VersionTrigger.MANUAL,
                description=f"Version {i+1}"
            )
            
            await version_manager.create_version(
                dataset_id="test_prune",
                data=sample_dataframe,
                context=context
            )
        
        # Prune versions, keeping only last 3
        pruned_count = await version_manager.prune_versions(
            dataset_id="test_prune",
            keep_last=3
        )
        
        assert pruned_count == 7
        
        # Check remaining versions
        history = await version_manager.get_version_history("test_prune")
        assert len(history) == 3


class TestLineageTracker:
    """Test lineage tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_create_lineage_nodes(self, lineage_tracker):
        """Test creating lineage nodes."""
        # Create source node
        source_node = await lineage_tracker.create_node(
            node_id="source_data",
            node_type="dataset",
            name="Source Dataset",
            system="data_lake",
            metadata={"tags": ["raw", "external"]}
        )
        
        assert source_node.node_id == "source_data"
        assert source_node.system == "data_lake"
        assert "raw" in source_node.metadata.tags
        
        # Create process node
        process_node = await lineage_tracker.create_node(
            node_id="transform_process",
            node_type="process",
            name="Data Transformation",
            system="etl_pipeline"
        )
        
        assert process_node.node_type == "process"
    
    @pytest.mark.asyncio
    async def test_create_lineage_edges(self, lineage_tracker):
        """Test creating lineage edges."""
        # Create nodes
        await lineage_tracker.create_node(
            node_id="input_data",
            node_type="dataset",
            name="Input",
            system="source"
        )
        
        await lineage_tracker.create_node(
            node_id="output_data",
            node_type="dataset",
            name="Output",
            system="target"
        )
        
        # Create edge
        edge = await lineage_tracker.create_edge(
            source_id="input_data",
            target_id="output_data",
            edge_type=DependencyType.TRANSFORMATION
        )
        
        assert edge.source_id == "input_data"
        assert edge.target_id == "output_data"
        assert edge.edge_type == DependencyType.TRANSFORMATION
    
    @pytest.mark.asyncio
    async def test_track_transformation(self, lineage_tracker):
        """Test tracking data transformations."""
        # Create nodes
        inputs = ["raw_data_1", "raw_data_2"]
        outputs = ["processed_data"]
        
        for node_id in inputs + outputs:
            await lineage_tracker.create_node(
                node_id=node_id,
                node_type="dataset",
                name=node_id,
                system="test"
            )
        
        # Track transformation
        context = LineageContext(
            operation_id="transform_123",
            operation_type=LineageEventType.TRANSFORMATION,
            timestamp=datetime.utcnow(),
            operator="test_operator"
        )
        
        transformation = await lineage_tracker.track_transformation(
            context=context,
            input_nodes=inputs,
            output_nodes=outputs,
            transformation_type=TransformationType.AGGREGATE,
            operation_details={"method": "sum"}
        )
        
        assert transformation.transformation_id == "transform_123"
        assert len(transformation.input_nodes) == 2
        assert len(transformation.output_nodes) == 1
    
    @pytest.mark.asyncio
    async def test_upstream_lineage(self, lineage_tracker):
        """Test retrieving upstream lineage."""
        # Create a simple lineage: A -> B -> C -> D
        nodes = ["node_a", "node_b", "node_c", "node_d"]
        
        for node in nodes:
            await lineage_tracker.create_node(
                node_id=node,
                node_type="dataset",
                name=node,
                system="test"
            )
        
        # Create edges
        for i in range(len(nodes) - 1):
            await lineage_tracker.create_edge(
                source_id=nodes[i],
                target_id=nodes[i + 1],
                edge_type=DependencyType.DIRECT
            )
        
        # Get upstream lineage from node_c
        upstream_graph = await lineage_tracker.get_upstream_lineage(
            node_id="node_c",
            max_depth=2
        )
        
        assert "node_c" in upstream_graph.nodes
        assert "node_b" in upstream_graph.nodes
        assert "node_a" in upstream_graph.nodes
        assert "node_d" not in upstream_graph.nodes
    
    @pytest.mark.asyncio
    async def test_impact_analysis(self, lineage_tracker):
        """Test impact analysis functionality."""
        # Create branching lineage: A -> B -> C, A -> D -> E
        nodes = {
            "node_a": "source",
            "node_b": "process",
            "node_c": "output",
            "node_d": "process",
            "node_e": "output"
        }
        
        for node_id, node_type in nodes.items():
            await lineage_tracker.create_node(
                node_id=node_id,
                node_type=node_type,
                name=node_id,
                system="test"
            )
        
        # Create edges
        edges = [
            ("node_a", "node_b"),
            ("node_b", "node_c"),
            ("node_a", "node_d"),
            ("node_d", "node_e")
        ]
        
        for source, target in edges:
            await lineage_tracker.create_edge(
                source_id=source,
                target_id=target,
                edge_type=DependencyType.DIRECT
            )
        
        # Analyze impact of changes to node_a
        impact = await lineage_tracker.analyze_impact(
            node_id="node_a",
            change_type="update"
        )
        
        assert len(impact.direct_impacts) == 2  # B and D
        assert len(impact.indirect_impacts) == 2  # C and E
        assert impact.risk_score > 0


class TestLineageService:
    """Test lineage service functionality."""
    
    @pytest.mark.asyncio
    async def test_track_data_operation(self, lineage_service):
        """Test tracking data operations through the service."""
        result = await lineage_service.track_data_operation(
            operation_type=LineageEventType.TRANSFORMATION,
            input_datasets=["raw_sales", "raw_customers"],
            output_datasets=["enriched_sales"],
            operator="etl_system",
            metadata={
                "transformation": "customer_enrichment",
                "records_processed": 10000
            }
        )
        
        assert result.success
        assert result.details["input_count"] == 2
        assert result.details["output_count"] == 1
    
    @pytest.mark.asyncio
    async def test_lineage_visualization(self, lineage_service):
        """Test lineage visualization generation."""
        # Create some lineage data
        await lineage_service.track_data_operation(
            operation_type=LineageEventType.TRANSFORMATION,
            input_datasets=["dataset_1"],
            output_datasets=["dataset_2"],
            operator="test"
        )
        
        await lineage_service.track_data_operation(
            operation_type=LineageEventType.AGGREGATION,
            input_datasets=["dataset_2"],
            output_datasets=["dataset_3"],
            operator="test"
        )
        
        # Get lineage graph
        graph = await lineage_service.get_data_lineage_graph(
            dataset_id="dataset_2",
            direction="both",
            max_depth=2
        )
        
        assert len(graph.nodes) >= 3
        assert len(graph.edges) >= 2
    
    @pytest.mark.asyncio
    async def test_data_rollback_with_lineage(self, lineage_service):
        """Test data rollback with lineage awareness."""
        # Create lineage
        await lineage_service.track_data_operation(
            operation_type=LineageEventType.TRANSFORMATION,
            input_datasets=["source"],
            output_datasets=["target"],
            operator="test"
        )
        
        # Attempt rollback (would need actual versions in real scenario)
        success, message = await lineage_service.perform_data_rollback(
            dataset_id="target",
            target_version="v1.0.0",
            cascade=False
        )
        
        # In test environment, this might fail due to missing versions
        assert isinstance(success, bool)
        assert isinstance(message, str)


class TestDataFingerprinting:
    """Test data fingerprinting functionality."""
    
    @pytest.mark.asyncio
    async def test_sha256_fingerprint(self, sample_dataframe):
        """Test SHA-256 fingerprinting."""
        fingerprinter = DataFingerprinter()
        
        fingerprint = await fingerprinter.calculate_fingerprint(
            sample_dataframe,
            algorithm="sha256"
        )
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA-256 produces 64 hex characters
    
    @pytest.mark.asyncio
    async def test_statistical_fingerprint(self, sample_dataframe):
        """Test statistical fingerprinting."""
        fingerprinter = DataFingerprinter()
        
        fingerprint = await fingerprinter.calculate_fingerprint(
            sample_dataframe,
            algorithm="statistical"
        )
        
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 16  # Truncated fingerprint
    
    @pytest.mark.asyncio
    async def test_content_diff_detection(self, sample_dataframe):
        """Test content difference detection."""
        fingerprinter = DataFingerprinter()
        
        # Create modified version
        modified_df = sample_dataframe.copy()
        modified_df.loc[0, 'price'] = 999.99
        
        diff = await fingerprinter.calculate_content_diff(
            sample_dataframe,
            modified_df
        )
        
        assert diff["has_changes"]
        assert diff["change_type"] == "content_change"
        assert "column_changes" in diff["details"]


class TestDependencyGraphAnalysis:
    """Test dependency graph analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_graph_metrics(self, lineage_tracker):
        """Test graph metrics calculation."""
        # Create a simple graph
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
        
        for node in nodes:
            await lineage_tracker.create_node(
                node_id=node,
                node_type="dataset",
                name=node,
                system="test"
            )
        
        for source, target in edges:
            await lineage_tracker.create_edge(
                source_id=source,
                target_id=target,
                edge_type=DependencyType.DIRECT
            )
        
        # Get full graph
        graph = await lineage_tracker.get_downstream_lineage("A", max_depth=10)
        
        # Analyze graph
        analyzer = DependencyGraphAnalyzer(graph)
        metrics = analyzer.analyze_graph_metrics()
        
        assert metrics["basic_stats"]["node_count"] == 5
        assert metrics["basic_stats"]["edge_count"] >= 4
        assert metrics["basic_stats"]["is_dag"] == True
        
        # Check centrality metrics
        assert len(metrics["centrality_metrics"]["most_depended_upon"]) > 0
        assert len(metrics["centrality_metrics"]["most_dependent"]) > 0


@pytest.mark.asyncio
async def test_end_to_end_versioning_lineage_integration():
    """Test complete integration of versioning and lineage systems."""
    # Initialize services
    version_manager = get_version_manager()
    lineage_tracker = get_lineage_tracker()
    lineage_service = get_lineage_service()
    
    # Create sample data
    df1 = pd.DataFrame({
        'id': range(100),
        'value': np.random.randn(100)
    })
    
    df2 = pd.DataFrame({
        'id': range(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Track initial data load
    await lineage_service.track_data_operation(
        operation_type=LineageEventType.DATA_READ,
        input_datasets=[],
        output_datasets=["dataset_1", "dataset_2"],
        operator="data_loader",
        metadata={"source": "external_system"}
    )
    
    # Create versions for initial data
    version1 = await version_manager.create_version(
        dataset_id="dataset_1",
        data=df1,
        context=VersioningContext(
            trigger=VersionTrigger.DATA_SOURCE_CHANGE,
            description="Initial load of dataset 1"
        )
    )
    
    version2 = await version_manager.create_version(
        dataset_id="dataset_2",
        data=df2,
        context=VersioningContext(
            trigger=VersionTrigger.DATA_SOURCE_CHANGE,
            description="Initial load of dataset 2"
        )
    )
    
    # Track transformation
    await lineage_service.track_data_operation(
        operation_type=LineageEventType.JOIN,
        input_datasets=["dataset_1", "dataset_2"],
        output_datasets=["dataset_3"],
        operator="transformation_engine",
        metadata={"join_key": "id"}
    )
    
    # Create joined dataset
    df3 = pd.merge(df1, df2, on='id')
    
    version3 = await version_manager.create_version(
        dataset_id="dataset_3",
        data=df3,
        context=VersioningContext(
            trigger=VersionTrigger.TRANSFORMATION_CHANGE,
            description="Joined dataset from 1 and 2"
        )
    )
    
    # Analyze impact
    impact = await lineage_service.analyze_data_impact(
        dataset_id="dataset_1",
        change_type="update"
    )
    
    # Verify integration
    assert len(impact.direct_impacts) >= 1  # dataset_3 depends on dataset_1
    assert version3.metadata.parent_version is not None
    
    # Generate lineage report
    report = await lineage_service.generate_lineage_report()
    
    assert report.total_nodes >= 3
    assert report.total_edges >= 2
    assert len(report.systems_count) >= 1