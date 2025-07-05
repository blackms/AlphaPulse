"""Visualization utilities for explainable AI."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

from ..models.explanation_result import (
    ExplanationResult, FeatureContribution, GlobalExplanation,
    CounterfactualExplanation, ExplanationComparison
)


class ExplainabilityVisualizer:
    """Visualizer for explainability results."""
    
    def __init__(self, style: str = "whitegrid"):
        """Initialize visualizer.
        
        Args:
            style: Seaborn style to use
        """
        sns.set_style(style)
        self.colors = sns.color_palette("husl", 10)
    
    def plot_feature_importance(
        self,
        explanation: Union[ExplanationResult, GlobalExplanation],
        top_k: int = 10,
        plot_type: str = "bar",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot feature importance.
        
        Args:
            explanation: Explanation result
            top_k: Number of top features to show
            plot_type: Type of plot ("bar", "horizontal_bar", "lollipop")
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract feature importance
        if isinstance(explanation, ExplanationResult):
            features = explanation.get_top_features(top_k)
            importance_data = [
                (fc.feature_name, fc.contribution) for fc in features
            ]
        else:  # GlobalExplanation
            importance_data = sorted(
                explanation.global_feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_k]
        
        # Create DataFrame
        df = pd.DataFrame(importance_data, columns=["Feature", "Importance"])
        df["Abs_Importance"] = df["Importance"].abs()
        df = df.sort_values("Abs_Importance", ascending=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == "bar":
            bars = ax.barh(df["Feature"], df["Importance"])
            # Color bars based on positive/negative
            for bar, value in zip(bars, df["Importance"]):
                bar.set_color("steelblue" if value > 0 else "coral")
            ax.set_xlabel("Feature Importance")
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
        elif plot_type == "horizontal_bar":
            ax.barh(df["Feature"], df["Abs_Importance"], color="skyblue")
            ax.set_xlabel("Absolute Feature Importance")
            
        elif plot_type == "lollipop":
            ax.hlines(y=df["Feature"], xmin=0, xmax=df["Importance"],
                     color="skyblue", alpha=0.7, linewidth=2)
            ax.plot(df["Importance"], df["Feature"], "o", 
                   markersize=8, color="darkblue", alpha=0.8)
            ax.set_xlabel("Feature Importance")
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance Analysis")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_shap_waterfall(
        self,
        explanation: ExplanationResult,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create SHAP waterfall plot using Plotly.
        
        Args:
            explanation: SHAP explanation result
            figsize: Figure size (width, height)
            save_path: Path to save figure
            
        Returns:
            Plotly figure
        """
        # Sort features by absolute contribution
        features = sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        
        # Prepare data for waterfall
        feature_names = []
        contributions = []
        
        # Add base value
        feature_names.append("Base Value")
        contributions.append(explanation.base_value)
        
        # Add features
        for fc in features[:10]:  # Top 10 features
            feature_names.append(f"{fc.feature_name} = {fc.value:.3f}")
            contributions.append(fc.contribution)
        
        # Add remaining features if any
        remaining = sum(fc.contribution for fc in features[10:])
        if abs(remaining) > 0.001:
            feature_names.append("Other Features")
            contributions.append(remaining)
        
        # Create waterfall plot
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(contributions) - 1),
            x=feature_names,
            y=contributions,
            text=[f"{v:.3f}" for v in contributions],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="SHAP Waterfall Plot - Feature Contributions",
            xaxis_title="Features",
            yaxis_title="Model Output Value",
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_lime_explanation(
        self,
        explanation: ExplanationResult,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot LIME explanation.
        
        Args:
            explanation: LIME explanation result
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get feature contributions
        features = explanation.get_top_features(10)
        
        # Create DataFrame
        data = [(fc.feature_name, fc.contribution) for fc in features]
        df = pd.DataFrame(data, columns=["Feature", "Weight"])
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by weight
        df = df.sort_values("Weight")
        
        # Color based on positive/negative
        colors = ["green" if w > 0 else "red" for w in df["Weight"]]
        
        bars = ax.barh(df["Feature"], df["Weight"], color=colors, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left' if width > 0 else 'right',
                   va='center', fontsize=9)
        
        ax.set_xlabel("Feature Weight")
        ax.set_ylabel("Features")
        ax.set_title("LIME Local Explanation")
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_counterfactual_comparison(
        self,
        counterfactual: CounterfactualExplanation,
        top_k: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot counterfactual explanation comparison.
        
        Args:
            counterfactual: Counterfactual explanation
            top_k: Number of features to show
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get changed features
        changes = counterfactual.get_minimal_changes()
        
        # Prepare data
        feature_names = list(changes.keys())[:top_k]
        original_values = [changes[f][0] for f in feature_names]
        counterfactual_values = [changes[f][1] for f in feature_names]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Feature value comparison
        x = np.arange(len(feature_names))
        width = 0.35
        
        ax1.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
        ax1.bar(x + width/2, counterfactual_values, width, label='Counterfactual', alpha=0.8)
        
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Feature Values')
        ax1.set_title('Original vs Counterfactual Feature Values')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Change magnitude
        change_magnitudes = [abs(changes[f][1] - changes[f][0]) for f in feature_names]
        
        bars = ax2.barh(feature_names, change_magnitudes, color='skyblue')
        
        # Add value labels
        for bar, mag in zip(bars, change_magnitudes):
            ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{mag:.3f}', ha='left', va='center')
        
        ax2.set_xlabel('Change Magnitude')
        ax2.set_ylabel('Features')
        ax2.set_title('Feature Change Magnitudes')
        ax2.grid(True, alpha=0.3)
        
        # Add text summary
        fig.text(0.5, 0.02, 
                f'Original Prediction: {counterfactual.original_prediction:.3f} → '
                f'Counterfactual Prediction: {counterfactual.counterfactual_prediction:.3f}',
                ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_explanation_comparison(
        self,
        comparison: ExplanationComparison,
        feature_subset: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of multiple explanations.
        
        Args:
            comparison: Explanation comparison result
            feature_subset: Subset of features to show
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Plot 1: Feature contribution differences
        if feature_subset is None:
            # Get top disputed features
            feature_subset = comparison.disputed_features[:10]
        
        contribution_data = []
        for feature in feature_subset:
            if feature in comparison.contribution_differences:
                contributions = comparison.contribution_differences[feature]
                for i, contrib in enumerate(contributions):
                    contribution_data.append({
                        'Feature': feature,
                        'Method': f'Method {i+1}',
                        'Contribution': contrib
                    })
        
        if contribution_data:
            df = pd.DataFrame(contribution_data)
            df_pivot = df.pivot(index='Feature', columns='Method', values='Contribution')
            
            # Create grouped bar plot
            df_pivot.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Contribution')
            ax1.set_title('Feature Contributions by Method')
            ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 2: Method agreement heatmap
        sns.heatmap(comparison.method_agreement_matrix, 
                   annot=True, fmt='.2f', cmap='coolwarm',
                   center=0.5, vmin=0, vmax=1,
                   xticklabels=[f'M{i+1}' for i in range(len(comparison.method_agreement_matrix))],
                   yticklabels=[f'M{i+1}' for i in range(len(comparison.method_agreement_matrix))],
                   ax=ax2)
        ax2.set_title('Method Agreement Matrix')
        
        # Plot 3: Consistency scores
        consistency_items = sorted(
            comparison.consistency_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        if consistency_items:
            features, scores = zip(*consistency_items)
            
            bars = ax3.barh(features, scores, color='lightgreen')
            
            # Add threshold line
            ax3.axvline(x=0.7, color='red', linestyle='--', 
                       label=f'Consensus Threshold')
            
            # Color bars based on consensus
            for bar, score in zip(bars, scores):
                if score < 0.7:
                    bar.set_color('salmon')
            
            ax3.set_xlabel('Consistency Score')
            ax3.set_ylabel('Features')
            ax3.set_title('Feature Consistency Across Methods')
            ax3.set_xlim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Explanation Method Comparison', fontsize=14, weight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(
        self,
        explanations: List[ExplanationResult],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive dashboard for exploring explanations.
        
        Args:
            explanations: List of explanation results
            save_path: Path to save HTML dashboard
            
        Returns:
            Plotly figure with dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Feature Importance Over Time", 
                          "Top Features Comparison",
                          "Prediction Values Distribution",
                          "Explanation Confidence"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                  [{"type": "histogram"}, {"type": "box"}]]
        )
        
        # Plot 1: Feature importance over time
        if len(explanations) > 1:
            # Track top features over time
            feature_traces = {}
            for i, exp in enumerate(explanations):
                for fc in exp.get_top_features(5):
                    if fc.feature_name not in feature_traces:
                        feature_traces[fc.feature_name] = {'x': [], 'y': []}
                    feature_traces[fc.feature_name]['x'].append(i)
                    feature_traces[fc.feature_name]['y'].append(fc.contribution)
            
            for feature, data in feature_traces.items():
                fig.add_trace(
                    go.Scatter(x=data['x'], y=data['y'], name=feature,
                             mode='lines+markers'),
                    row=1, col=1
                )
        
        # Plot 2: Top features comparison (latest explanation)
        latest_exp = explanations[-1]
        top_features = latest_exp.get_top_features(10)
        
        fig.add_trace(
            go.Bar(
                x=[fc.feature_name for fc in top_features],
                y=[fc.contribution for fc in top_features],
                marker_color=['green' if fc.contribution > 0 else 'red' 
                             for fc in top_features]
            ),
            row=1, col=2
        )
        
        # Plot 3: Prediction values distribution
        pred_values = [exp.prediction_value for exp in explanations]
        fig.add_trace(
            go.Histogram(x=pred_values, nbinsx=20),
            row=2, col=1
        )
        
        # Plot 4: Confidence scores
        confidence_scores = [exp.confidence_score for exp in explanations 
                           if exp.confidence_score is not None]
        if confidence_scores:
            fig.add_trace(
                go.Box(y=confidence_scores, name="Confidence"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Explainability Dashboard"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Explanation Index", row=1, col=1)
        fig.update_yaxes(title_text="Contribution", row=1, col=1)
        
        fig.update_xaxes(title_text="Features", row=1, col=2)
        fig.update_yaxes(title_text="Contribution", row=1, col=2)
        
        fig.update_xaxes(title_text="Prediction Value", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_yaxes(title_text="Confidence Score", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def export_explanation_report(
        self,
        explanation: ExplanationResult,
        output_path: str,
        format: str = "html",
        include_visualizations: bool = True
    ):
        """Export explanation as a report.
        
        Args:
            explanation: Explanation to export
            output_path: Path to save report
            format: Output format ("html", "pdf", "json")
            include_visualizations: Whether to include plots
        """
        if format == "json":
            # Export as JSON
            with open(output_path, 'w') as f:
                json.dump(explanation.to_dict(), f, indent=2)
        
        elif format == "html":
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>Explanation Report - {explanation.explanation_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Explanation Report</h1>
                
                <h2>Summary</h2>
                <ul>
                    <li><strong>Model ID:</strong> {explanation.model_id}</li>
                    <li><strong>Prediction ID:</strong> {explanation.prediction_id}</li>
                    <li><strong>Type:</strong> {explanation.explanation_type.value}</li>
                    <li><strong>Level:</strong> {explanation.explanation_level.value}</li>
                    <li><strong>Timestamp:</strong> {explanation.timestamp}</li>
                    <li><strong>Base Value:</strong> {explanation.base_value:.4f}</li>
                    <li><strong>Prediction Value:</strong> {explanation.prediction_value:.4f}</li>
                </ul>
                
                <h2>Feature Contributions</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Value</th>
                        <th>Contribution</th>
                    </tr>
            """
            
            for fc in explanation.get_top_features(20):
                contribution_class = "positive" if fc.contribution > 0 else "negative"
                html_content += f"""
                    <tr>
                        <td>{fc.feature_name}</td>
                        <td>{fc.value:.4f}</td>
                        <td class="{contribution_class}">{fc.contribution:.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Metadata</h2>
                <ul>
            """
            
            if explanation.confidence_score:
                html_content += f"<li><strong>Confidence:</strong> {explanation.confidence_score:.4f}</li>"
            
            if explanation.computation_time:
                html_content += f"<li><strong>Computation Time:</strong> {explanation.computation_time:.2f}s</li>"
            
            html_content += """
                </ul>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")