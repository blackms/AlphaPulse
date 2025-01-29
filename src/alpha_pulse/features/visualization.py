"""
Visualization module for feature analysis and model evaluation.

This module provides plotting utilities for analyzing features
and evaluating model performance.
"""
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from loguru import logger


class FeatureVisualizer:
    """Class for creating and managing feature visualization plots."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize visualizer with output directory.

        Args:
            output_dir: Directory to save plots (defaults to 'plots')
        """
        self.output_dir = Path(output_dir) if output_dir else Path('plots')
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized FeatureVisualizer with output dir: {self.output_dir}")

    def _save_plot(self, name: str) -> Path:
        """Save current plot with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"{name}_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved plot to {plot_path}")
        return plot_path

    def plot_feature_importance(
        self,
        importance: pd.Series,
        title: str = "Feature Importance Scores",
        figsize: tuple = (12, 6)
    ) -> Path:
        """
        Plot feature importance scores.

        Args:
            importance: Series of feature importance scores
            title: Plot title
            figsize: Figure size (width, height)

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=figsize)
        importance.sort_values().plot(kind='barh')
        plt.title(title)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        return self._save_plot('feature_importance')

    def plot_predictions_vs_actual(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        title: str = "Predicted vs Actual Returns",
        figsize: tuple = (12, 6)
    ) -> Path:
        """
        Plot predicted vs actual values.

        Args:
            y_true: Series of actual values
            y_pred: Array of predicted values
            title: Plot title
            figsize: Figure size (width, height)

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'r--', lw=2)
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title(title)
        plt.tight_layout()
        return self._save_plot('predictions_vs_actual')

    def plot_feature_distributions(
        self,
        features: pd.DataFrame,
        n_cols: int = 3,
        figsize: tuple = (15, 10)
    ) -> Path:
        """
        Plot distributions of all features.

        Args:
            features: DataFrame of features
            n_cols: Number of columns in subplot grid
            figsize: Figure size (width, height)

        Returns:
            Path to saved plot
        """
        n_features = len(features.columns)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=figsize)
        for i, col in enumerate(features.columns, 1):
            plt.subplot(n_rows, n_cols, i)
            features[col].hist(bins=50)
            plt.title(col)
        plt.tight_layout()
        return self._save_plot('feature_distributions')

    def plot_correlation_matrix(
        self,
        features: pd.DataFrame,
        figsize: tuple = (12, 10)
    ) -> Path:
        """
        Plot correlation matrix heatmap.

        Args:
            features: DataFrame of features
            figsize: Figure size (width, height)

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=figsize)
        corr = features.corr()
        plt.imshow(corr, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        return self._save_plot('correlation_matrix')

    def plot_time_series_features(
        self,
        features: pd.DataFrame,
        columns: Optional[list] = None,
        n_cols: int = 2,
        figsize: tuple = (15, 10)
    ) -> Path:
        """
        Plot time series of selected features.

        Args:
            features: DataFrame of features with datetime index
            columns: List of columns to plot (defaults to all)
            n_cols: Number of columns in subplot grid
            figsize: Figure size (width, height)

        Returns:
            Path to saved plot
        """
        columns = columns or features.columns
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        plt.figure(figsize=figsize)
        for i, col in enumerate(columns, 1):
            plt.subplot(n_rows, n_cols, i)
            features[col].plot()
            plt.title(col)
        plt.tight_layout()
        return self._save_plot('time_series_features')