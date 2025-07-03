"""
Dependency Graph visualization and analysis for data lineage.

Provides:
- Interactive dependency graph visualization
- Graph analysis and metrics
- Path finding and traversal
- Graph export formats
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
from loguru import logger

from alpha_pulse.models.lineage_metadata import (
    LineageNode, LineageEdge, LineageGraph, DependencyType
)


@dataclass
class GraphLayout:
    """Layout configuration for graph visualization."""
    layout_type: str = "hierarchical"  # hierarchical, circular, spring, kamada_kawai
    node_spacing: float = 2.0
    level_spacing: float = 3.0
    edge_bundling: bool = True
    show_labels: bool = True
    node_size_attribute: Optional[str] = None  # Attribute to scale node size
    edge_weight_attribute: Optional[str] = None  # Attribute for edge thickness


@dataclass
class GraphStyle:
    """Styling configuration for graph visualization."""
    # Node styles by type
    node_colors: Dict[str, str] = None
    node_shapes: Dict[str, str] = None
    
    # Edge styles by type
    edge_colors: Dict[str, str] = None
    edge_styles: Dict[str, str] = None  # solid, dashed, dotted
    
    # General style
    background_color: str = "white"
    font_family: str = "Arial"
    font_size: int = 10
    
    def __post_init__(self):
        if self.node_colors is None:
            self.node_colors = {
                "dataset": "#4CAF50",
                "process": "#2196F3",
                "model": "#FF9800",
                "report": "#9C27B0",
                "external": "#607D8B"
            }
        
        if self.node_shapes is None:
            self.node_shapes = {
                "dataset": "rectangle",
                "process": "ellipse",
                "model": "diamond",
                "report": "hexagon",
                "external": "triangle"
            }
        
        if self.edge_colors is None:
            self.edge_colors = {
                DependencyType.DIRECT.value: "#333333",
                DependencyType.TRANSFORMATION.value: "#2196F3",
                DependencyType.DERIVATION.value: "#4CAF50",
                DependencyType.AGGREGATION.value: "#FF9800",
                DependencyType.REFERENCE.value: "#9E9E9E"
            }
        
        if self.edge_styles is None:
            self.edge_styles = {
                DependencyType.DIRECT.value: "solid",
                DependencyType.TRANSFORMATION.value: "solid",
                DependencyType.DERIVATION.value: "dashed",
                DependencyType.AGGREGATION.value: "dashdot",
                DependencyType.REFERENCE.value: "dotted"
            }


class DependencyGraphVisualizer:
    """Visualizer for dependency graphs."""
    
    def __init__(
        self,
        layout: Optional[GraphLayout] = None,
        style: Optional[GraphStyle] = None
    ):
        self.layout = layout or GraphLayout()
        self.style = style or GraphStyle()
    
    def visualize_matplotlib(
        self,
        lineage_graph: LineageGraph,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a static visualization using matplotlib.
        
        Args:
            lineage_graph: The lineage graph to visualize
            output_path: Path to save the figure
            figsize: Figure size
            title: Graph title
        
        Returns:
            Matplotlib figure
        """
        # Create NetworkX graph
        G = self._create_networkx_graph(lineage_graph)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout
        pos = self._calculate_layout(G, lineage_graph)
        
        # Draw nodes
        for node_id, (x, y) in pos.items():
            node = lineage_graph.nodes[node_id]
            color = self.style.node_colors.get(node.node_type, "#888888")
            
            # Draw node shape
            if self.style.node_shapes.get(node.node_type) == "rectangle":
                rect = FancyBboxPatch(
                    (x - 0.4, y - 0.2), 0.8, 0.4,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.8
                )
                ax.add_patch(rect)
            else:
                ax.scatter(x, y, s=500, c=color, alpha=0.8, edgecolors='black')
            
            # Draw label
            if self.layout.show_labels:
                ax.text(x, y, node.name, ha='center', va='center',
                       fontsize=self.style.font_size, fontfamily=self.style.font_family)
        
        # Draw edges
        for edge_id, edge in lineage_graph.edges.items():
            if edge.source_id in pos and edge.target_id in pos:
                x1, y1 = pos[edge.source_id]
                x2, y2 = pos[edge.target_id]
                
                color = self.style.edge_colors.get(edge.edge_type.value, "#666666")
                style = self.style.edge_styles.get(edge.edge_type.value, "solid")
                
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color,
                                         linestyle=style, alpha=0.6))
        
        # Add title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Remove axes
        ax.set_xlim(min(x for x, y in pos.values()) - 1,
                   max(x for x, y in pos.values()) + 1)
        ax.set_ylim(min(y for x, y in pos.values()) - 1,
                   max(y for x, y in pos.values()) + 1)
        ax.axis('off')
        
        # Add legend
        self._add_legend(ax)
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_plotly(
        self,
        lineage_graph: LineageGraph,
        title: Optional[str] = None,
        height: int = 800,
        width: int = 1200
    ) -> go.Figure:
        """
        Create an interactive visualization using Plotly.
        
        Args:
            lineage_graph: The lineage graph to visualize
            title: Graph title
            height: Figure height
            width: Figure width
        
        Returns:
            Plotly figure
        """
        # Create NetworkX graph
        G = self._create_networkx_graph(lineage_graph)
        
        # Calculate layout
        pos = self._calculate_layout(G, lineage_graph)
        
        # Create edge traces
        edge_traces = self._create_edge_traces(lineage_graph, pos)
        
        # Create node trace
        node_trace = self._create_node_trace(lineage_graph, pos)
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        fig.update_layout(
            title=title or "Data Lineage Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=height,
            width=width,
            paper_bgcolor=self.style.background_color,
            plot_bgcolor=self.style.background_color
        )
        
        return fig
    
    def export_to_graphviz(
        self,
        lineage_graph: LineageGraph,
        output_path: str,
        format: str = "dot"
    ) -> None:
        """
        Export graph to Graphviz format.
        
        Args:
            lineage_graph: The lineage graph to export
            output_path: Output file path
            format: Output format (dot, svg, png, pdf)
        """
        dot_content = self._generate_dot_content(lineage_graph)
        
        if format == "dot":
            with open(output_path, 'w') as f:
                f.write(dot_content)
        else:
            # Use graphviz to render
            import graphviz
            graph = graphviz.Source(dot_content)
            graph.render(output_path, format=format, cleanup=True)
    
    def export_to_cytoscape(
        self,
        lineage_graph: LineageGraph,
        output_path: str
    ) -> None:
        """
        Export graph to Cytoscape.js format.
        
        Args:
            lineage_graph: The lineage graph to export
            output_path: Output file path
        """
        elements = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node in lineage_graph.nodes.items():
            elements["nodes"].append({
                "data": {
                    "id": node_id,
                    "label": node.name,
                    "type": node.node_type,
                    "system": node.system
                },
                "classes": node.node_type
            })
        
        # Add edges
        for edge_id, edge in lineage_graph.edges.items():
            elements["edges"].append({
                "data": {
                    "id": edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type.value
                },
                "classes": edge.edge_type.value
            })
        
        # Add style
        style = self._generate_cytoscape_style()
        
        output = {
            "elements": elements,
            "style": style,
            "layout": {"name": self.layout.layout_type}
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    
    def _create_networkx_graph(self, lineage_graph: LineageGraph) -> nx.DiGraph:
        """Create NetworkX graph from lineage graph."""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in lineage_graph.nodes.items():
            G.add_node(node_id, data=node)
        
        # Add edges
        for edge_id, edge in lineage_graph.edges.items():
            G.add_edge(edge.source_id, edge.target_id,
                      edge_id=edge_id, data=edge)
        
        return G
    
    def _calculate_layout(
        self,
        G: nx.DiGraph,
        lineage_graph: LineageGraph
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions for layout."""
        if self.layout.layout_type == "hierarchical":
            # Use topological sort for hierarchical layout
            try:
                layers = list(nx.topological_generations(G))
                pos = {}
                
                for i, layer in enumerate(layers):
                    layer_y = -i * self.layout.level_spacing
                    layer_width = len(layer) * self.layout.node_spacing
                    
                    for j, node in enumerate(layer):
                        x = (j - len(layer) / 2) * self.layout.node_spacing
                        pos[node] = (x, layer_y)
                
                return pos
            except nx.NetworkXError:
                # Fall back to spring layout if graph has cycles
                return nx.spring_layout(G, k=self.layout.node_spacing)
        
        elif self.layout.layout_type == "circular":
            return nx.circular_layout(G)
        elif self.layout.layout_type == "spring":
            return nx.spring_layout(G, k=self.layout.node_spacing)
        elif self.layout.layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        else:
            return nx.spring_layout(G)
    
    def _create_edge_traces(
        self,
        lineage_graph: LineageGraph,
        pos: Dict[str, Tuple[float, float]]
    ) -> List[go.Scatter]:
        """Create Plotly edge traces."""
        edge_traces = []
        
        # Group edges by type for different styling
        edges_by_type = defaultdict(list)
        for edge_id, edge in lineage_graph.edges.items():
            edges_by_type[edge.edge_type.value].append(edge)
        
        for edge_type, edges in edges_by_type.items():
            edge_x = []
            edge_y = []
            
            for edge in edges:
                if edge.source_id in pos and edge.target_id in pos:
                    x0, y0 = pos[edge.source_id]
                    x1, y1 = pos[edge.target_id]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            color = self.style.edge_colors.get(edge_type, "#666666")
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color=color),
                hoverinfo='none',
                mode='lines',
                name=edge_type
            )
            edge_traces.append(edge_trace)
        
        return edge_traces
    
    def _create_node_trace(
        self,
        lineage_graph: LineageGraph,
        pos: Dict[str, Tuple[float, float]]
    ) -> go.Scatter:
        """Create Plotly node trace."""
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node_id, (x, y) in pos.items():
            node = lineage_graph.nodes[node_id]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text
            hover_text = f"<b>{node.name}</b><br>"
            hover_text += f"Type: {node.node_type}<br>"
            hover_text += f"System: {node.system}<br>"
            hover_text += f"Created: {node.created_at.strftime('%Y-%m-%d %H:%M')}<br>"
            
            if node.metadata.tags:
                hover_text += f"Tags: {', '.join(node.metadata.tags)}<br>"
            
            node_text.append(hover_text)
            node_color.append(self.style.node_colors.get(node.node_type, "#888888"))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[lineage_graph.nodes[nid].name for nid in pos.keys()],
            hovertext=node_text,
            textposition="top center",
            marker=dict(
                size=10,
                color=node_color,
                line=dict(color='black', width=1)
            )
        )
        
        return node_trace
    
    def _add_legend(self, ax: plt.Axes) -> None:
        """Add legend to matplotlib plot."""
        # Node type legend
        node_patches = []
        for node_type, color in self.style.node_colors.items():
            patch = mpatches.Patch(color=color, label=node_type.capitalize())
            node_patches.append(patch)
        
        # Edge type legend
        edge_lines = []
        for edge_type, color in self.style.edge_colors.items():
            style = self.style.edge_styles.get(edge_type, "solid")
            line = plt.Line2D([0], [0], color=color, linestyle=style,
                            label=edge_type.replace('_', ' ').capitalize())
            edge_lines.append(line)
        
        # Add legends
        legend1 = ax.legend(handles=node_patches, loc='upper left',
                          title="Node Types", bbox_to_anchor=(1.02, 1))
        ax.add_artist(legend1)
        
        ax.legend(handles=edge_lines, loc='upper left',
                 title="Edge Types", bbox_to_anchor=(1.02, 0.5))
    
    def _generate_dot_content(self, lineage_graph: LineageGraph) -> str:
        """Generate Graphviz DOT content."""
        lines = ['digraph DataLineage {']
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box, style=filled];')
        lines.append('')
        
        # Add nodes
        for node_id, node in lineage_graph.nodes.items():
            color = self.style.node_colors.get(node.node_type, "#888888")
            shape = self.style.node_shapes.get(node.node_type, "box")
            
            lines.append(f'  "{node_id}" [label="{node.name}", '
                        f'fillcolor="{color}", shape={shape}];')
        
        lines.append('')
        
        # Add edges
        for edge_id, edge in lineage_graph.edges.items():
            style = self.style.edge_styles.get(edge.edge_type.value, "solid")
            color = self.style.edge_colors.get(edge.edge_type.value, "#666666")
            
            lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" '
                        f'[style={style}, color="{color}"];')
        
        lines.append('}')
        
        return '\n'.join(lines)
    
    def _generate_cytoscape_style(self) -> List[Dict[str, Any]]:
        """Generate Cytoscape.js style configuration."""
        styles = []
        
        # Node styles
        for node_type, color in self.style.node_colors.items():
            styles.append({
                "selector": f"node.{node_type}",
                "style": {
                    "background-color": color,
                    "label": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": self.style.font_size,
                    "font-family": self.style.font_family
                }
            })
        
        # Edge styles
        for edge_type, color in self.style.edge_colors.items():
            style = self.style.edge_styles.get(edge_type, "solid")
            styles.append({
                "selector": f"edge.{edge_type}",
                "style": {
                    "line-color": color,
                    "line-style": style,
                    "target-arrow-color": color,
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier"
                }
            })
        
        return styles


class DependencyGraphAnalyzer:
    """Analyzer for dependency graph metrics and patterns."""
    
    def __init__(self, lineage_graph: LineageGraph):
        self.lineage_graph = lineage_graph
        self.G = self._create_networkx_graph()
    
    def _create_networkx_graph(self) -> nx.DiGraph:
        """Create NetworkX graph from lineage graph."""
        G = nx.DiGraph()
        
        for node_id in self.lineage_graph.nodes:
            G.add_node(node_id)
        
        for edge in self.lineage_graph.edges.values():
            G.add_edge(edge.source_id, edge.target_id)
        
        return G
    
    def analyze_graph_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive graph metrics."""
        metrics = {
            "basic_stats": self._calculate_basic_stats(),
            "centrality_metrics": self._calculate_centrality_metrics(),
            "connectivity_metrics": self._calculate_connectivity_metrics(),
            "path_metrics": self._calculate_path_metrics(),
            "clustering_metrics": self._calculate_clustering_metrics()
        }
        
        return metrics
    
    def _calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic graph statistics."""
        return {
            "node_count": self.G.number_of_nodes(),
            "edge_count": self.G.number_of_edges(),
            "density": nx.density(self.G),
            "is_dag": nx.is_directed_acyclic_graph(self.G),
            "number_of_cycles": len(list(nx.simple_cycles(self.G))) if not nx.is_directed_acyclic_graph(self.G) else 0
        }
    
    def _calculate_centrality_metrics(self) -> Dict[str, Any]:
        """Calculate centrality metrics."""
        metrics = {}
        
        # Degree centrality
        in_degree = nx.in_degree_centrality(self.G)
        out_degree = nx.out_degree_centrality(self.G)
        
        metrics["most_dependent"] = sorted(in_degree.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
        metrics["most_depended_upon"] = sorted(out_degree.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]
        
        # Betweenness centrality (nodes that are on many paths)
        if len(self.G) > 1:
            betweenness = nx.betweenness_centrality(self.G)
            metrics["critical_nodes"] = sorted(betweenness.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]
        
        return metrics
    
    def _calculate_connectivity_metrics(self) -> Dict[str, Any]:
        """Calculate connectivity metrics."""
        metrics = {}
        
        # Weakly connected components
        components = list(nx.weakly_connected_components(self.G))
        metrics["num_components"] = len(components)
        metrics["largest_component_size"] = len(max(components, key=len)) if components else 0
        metrics["isolated_nodes"] = [n for n in self.G.nodes() if self.G.degree(n) == 0]
        
        # Find source and sink nodes
        metrics["source_nodes"] = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
        metrics["sink_nodes"] = [n for n in self.G.nodes() if self.G.out_degree(n) == 0]
        
        return metrics
    
    def _calculate_path_metrics(self) -> Dict[str, Any]:
        """Calculate path-related metrics."""
        metrics = {}
        
        # Average shortest path length for connected components
        path_lengths = []
        for component in nx.weakly_connected_components(self.G):
            subgraph = self.G.subgraph(component)
            if len(subgraph) > 1:
                for source in subgraph:
                    lengths = nx.single_source_shortest_path_length(subgraph, source)
                    path_lengths.extend(lengths.values())
        
        if path_lengths:
            metrics["avg_path_length"] = sum(path_lengths) / len(path_lengths)
            metrics["max_path_length"] = max(path_lengths)
        
        # Longest path in DAG
        if nx.is_directed_acyclic_graph(self.G):
            longest_path = nx.dag_longest_path(self.G)
            metrics["longest_path_length"] = len(longest_path)
            metrics["longest_path_nodes"] = longest_path
        
        return metrics
    
    def _calculate_clustering_metrics(self) -> Dict[str, Any]:
        """Calculate clustering metrics."""
        metrics = {}
        
        # Node type distribution
        node_types = Counter()
        for node_id, node in self.lineage_graph.nodes.items():
            node_types[node.node_type] += 1
        metrics["node_type_distribution"] = dict(node_types)
        
        # System distribution
        system_dist = Counter()
        for node in self.lineage_graph.nodes.values():
            system_dist[node.system] += 1
        metrics["system_distribution"] = dict(system_dist)
        
        # Edge type distribution
        edge_types = Counter()
        for edge in self.lineage_graph.edges.values():
            edge_types[edge.edge_type.value] += 1
        metrics["edge_type_distribution"] = dict(edge_types)
        
        return metrics
    
    def find_critical_paths(self) -> List[List[str]]:
        """Find critical paths in the graph."""
        critical_paths = []
        
        # Find paths between source and sink nodes
        sources = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
        sinks = [n for n in self.G.nodes() if self.G.out_degree(n) == 0]
        
        for source in sources:
            for sink in sinks:
                try:
                    paths = list(nx.all_simple_paths(self.G, source, sink, cutoff=10))
                    critical_paths.extend(paths[:3])  # Top 3 paths per source-sink pair
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by length (longer paths are often more critical)
        critical_paths.sort(key=len, reverse=True)
        
        return critical_paths[:10]  # Return top 10 critical paths
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the graph."""
        try:
            cycles = list(nx.simple_cycles(self.G))
            return cycles
        except:
            return []
    
    def analyze_impact_radius(self, node_id: str, max_hops: int = 3) -> Dict[str, Set[str]]:
        """Analyze the impact radius of a node."""
        impact = {
            "upstream": set(),
            "downstream": set()
        }
        
        # Upstream impact (ancestors)
        for hop in range(1, max_hops + 1):
            nodes_at_hop = set()
            for node in impact["upstream"] if hop > 1 else [node_id]:
                nodes_at_hop.update(self.G.predecessors(node))
            impact[f"upstream_hop_{hop}"] = nodes_at_hop - impact["upstream"]
            impact["upstream"].update(nodes_at_hop)
        
        # Downstream impact (descendants)
        for hop in range(1, max_hops + 1):
            nodes_at_hop = set()
            for node in impact["downstream"] if hop > 1 else [node_id]:
                nodes_at_hop.update(self.G.successors(node))
            impact[f"downstream_hop_{hop}"] = nodes_at_hop - impact["downstream"]
            impact["downstream"].update(nodes_at_hop)
        
        return impact