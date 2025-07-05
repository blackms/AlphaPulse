"""Result aggregation and merging system for distributed backtesting."""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import json
from collections import defaultdict
import statistics
from scipy import stats

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating results."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    CUSTOM = "custom"


class MergeStrategy(Enum):
    """Strategy for merging distributed results."""
    CONCATENATE = "concatenate"  # Combine time series
    AVERAGE = "average"  # Average metrics
    PORTFOLIO = "portfolio"  # Portfolio-level aggregation
    HIERARCHICAL = "hierarchical"  # Multi-level aggregation


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    merge_strategy: MergeStrategy = MergeStrategy.PORTFOLIO
    aggregation_methods: Dict[str, AggregationMethod] = field(default_factory=lambda: {
        "total_return": AggregationMethod.WEIGHTED_MEAN,
        "sharpe_ratio": AggregationMethod.MEAN,
        "max_drawdown": AggregationMethod.MIN,
        "win_rate": AggregationMethod.MEAN,
    })
    weights: Optional[Dict[str, float]] = None
    confidence_level: float = 0.95
    calculate_statistics: bool = True
    handle_missing: str = "skip"  # skip, fill, or raise


@dataclass
class AggregatedResult:
    """Container for aggregated results."""
    aggregate_metrics: Dict[str, float]
    statistics: Dict[str, Dict[str, float]]
    component_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    correlations: Optional[pd.DataFrame] = None


class ResultAggregator:
    """Aggregates and merges distributed backtesting results."""
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        """Initialize result aggregator.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        self._custom_aggregators: Dict[str, Callable] = {}
        
    def aggregate_results(self, 
                         results: List[Dict[str, Any]],
                         weights: Optional[List[float]] = None) -> AggregatedResult:
        """Aggregate multiple backtesting results.
        
        Args:
            results: List of result dictionaries
            weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated result object
        """
        if not results:
            raise ValueError("No results to aggregate")
            
        # Filter valid results
        valid_results = self._filter_valid_results(results)
        if not valid_results:
            raise ValueError("No valid results after filtering")
            
        # Use provided weights or config weights
        if weights is None and self.config.weights:
            weights = [self.config.weights.get(str(i), 1.0) for i in range(len(valid_results))]
        elif weights is None:
            weights = [1.0] * len(valid_results)
            
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Aggregate based on strategy
        if self.config.merge_strategy == MergeStrategy.CONCATENATE:
            return self._concatenate_results(valid_results, weights)
        elif self.config.merge_strategy == MergeStrategy.AVERAGE:
            return self._average_results(valid_results, weights)
        elif self.config.merge_strategy == MergeStrategy.PORTFOLIO:
            return self._portfolio_aggregate(valid_results, weights)
        else:  # HIERARCHICAL
            return self._hierarchical_aggregate(valid_results, weights)
    
    def merge_time_series(self,
                         series_list: List[pd.Series],
                         method: str = "outer") -> pd.Series:
        """Merge multiple time series.
        
        Args:
            series_list: List of pandas Series
            method: Join method (outer, inner, left, right)
            
        Returns:
            Merged series
        """
        if not series_list:
            return pd.Series()
            
        # Align all series
        merged = series_list[0]
        for series in series_list[1:]:
            merged = pd.concat([merged, series], axis=1, join=method).mean(axis=1)
            
        return merged
    
    def aggregate_metrics(self,
                         metrics_list: List[Dict[str, float]],
                         weights: Optional[List[float]] = None) -> Dict[str, float]:
        """Aggregate performance metrics.
        
        Args:
            metrics_list: List of metric dictionaries
            weights: Optional weights
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
            
        aggregated = {}
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
            
        for key in all_keys:
            values = [m.get(key, np.nan) for m in metrics_list]
            method = self.config.aggregation_methods.get(key, AggregationMethod.MEAN)
            
            aggregated[key] = self._apply_aggregation_method(values, method, weights)
            
        return aggregated
    
    def calculate_portfolio_metrics(self,
                                  component_returns: List[np.ndarray],
                                  weights: List[float]) -> Dict[str, float]:
        """Calculate portfolio-level metrics from component returns.
        
        Args:
            component_returns: List of return arrays
            weights: Component weights
            
        Returns:
            Portfolio metrics
        """
        # Calculate weighted portfolio returns
        portfolio_returns = np.zeros(max(len(r) for r in component_returns))
        
        for returns, weight in zip(component_returns, weights):
            # Pad shorter series
            if len(returns) < len(portfolio_returns):
                padded = np.pad(returns, (0, len(portfolio_returns) - len(returns)), 
                              mode='constant', constant_values=0)
            else:
                padded = returns[:len(portfolio_returns)]
            portfolio_returns += weight * padded
            
        # Calculate metrics
        total_return = np.prod(1 + portfolio_returns) - 1
        sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        
        # Drawdown calculation
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Additional metrics
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": np.std(portfolio_returns) * np.sqrt(252),
            "win_rate": len(positive_returns) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0,
            "profit_factor": np.sum(positive_returns) / -np.sum(negative_returns) if len(negative_returns) > 0 else np.inf,
            "avg_win": np.mean(positive_returns) if len(positive_returns) > 0 else 0,
            "avg_loss": np.mean(negative_returns) if len(negative_returns) > 0 else 0,
        }
    
    def register_custom_aggregator(self, metric: str, aggregator: Callable) -> None:
        """Register a custom aggregation function.
        
        Args:
            metric: Metric name
            aggregator: Aggregation function
        """
        self._custom_aggregators[metric] = aggregator
        
    def _filter_valid_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out invalid or error results."""
        valid_results = []
        
        for result in results:
            if isinstance(result, dict) and "error" not in result:
                valid_results.append(result)
            elif self.config.handle_missing == "raise":
                raise ValueError(f"Invalid result: {result}")
                
        return valid_results
    
    def _concatenate_results(self, 
                           results: List[Dict[str, Any]],
                           weights: np.ndarray) -> AggregatedResult:
        """Concatenate time-series results."""
        # Extract time series data
        all_returns = []
        all_dates = []
        
        for result in results:
            if "returns" in result:
                all_returns.append(np.array(result["returns"]))
            if "dates" in result:
                all_dates.extend(result["dates"])
                
        # Concatenate returns
        if all_returns:
            concatenated_returns = np.concatenate(all_returns)
        else:
            concatenated_returns = np.array([])
            
        # Calculate metrics on concatenated data
        metrics = self.calculate_portfolio_metrics([concatenated_returns], [1.0])
        
        # Calculate statistics
        statistics = self._calculate_statistics(results)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(results)
        
        return AggregatedResult(
            aggregate_metrics=metrics,
            statistics=statistics,
            component_results=results,
            metadata={
                "aggregation_method": "concatenate",
                "n_components": len(results),
                "total_periods": len(concatenated_returns),
            },
            confidence_intervals=confidence_intervals
        )
    
    def _average_results(self,
                        results: List[Dict[str, Any]],
                        weights: np.ndarray) -> AggregatedResult:
        """Average metrics across results."""
        # Extract metrics
        metrics_list = []
        for result in results:
            metrics = {k: v for k, v in result.items() 
                      if isinstance(v, (int, float)) and not np.isnan(v)}
            metrics_list.append(metrics)
            
        # Average metrics
        aggregated_metrics = self.aggregate_metrics(metrics_list, weights)
        
        # Calculate statistics
        statistics = self._calculate_statistics(results)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(results)
        
        return AggregatedResult(
            aggregate_metrics=aggregated_metrics,
            statistics=statistics,
            component_results=results,
            metadata={
                "aggregation_method": "average",
                "n_components": len(results),
                "weights": weights.tolist(),
            },
            confidence_intervals=confidence_intervals
        )
    
    def _portfolio_aggregate(self,
                           results: List[Dict[str, Any]],
                           weights: np.ndarray) -> AggregatedResult:
        """Aggregate as portfolio components."""
        # Extract component returns
        component_returns = []
        for result in results:
            if "returns" in result:
                component_returns.append(np.array(result["returns"]))
            elif "total_return" in result:
                # Synthesize returns from total return
                n_periods = result.get("n_periods", 252)
                avg_return = result["total_return"] / n_periods
                component_returns.append(np.full(n_periods, avg_return))
                
        # Calculate portfolio metrics
        if component_returns:
            portfolio_metrics = self.calculate_portfolio_metrics(component_returns, weights)
        else:
            # Fallback to averaging
            portfolio_metrics = self.aggregate_metrics(results, weights)
            
        # Calculate component statistics
        statistics = self._calculate_statistics(results)
        
        # Calculate correlations if possible
        correlations = None
        if len(component_returns) > 1 and all(len(r) > 1 for r in component_returns):
            # Align series lengths
            min_length = min(len(r) for r in component_returns)
            aligned_returns = [r[:min_length] for r in component_returns]
            
            # Calculate correlation matrix
            returns_df = pd.DataFrame(aligned_returns).T
            correlations = returns_df.corr()
            
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(results)
        
        return AggregatedResult(
            aggregate_metrics=portfolio_metrics,
            statistics=statistics,
            component_results=results,
            metadata={
                "aggregation_method": "portfolio",
                "n_components": len(results),
                "weights": weights.tolist(),
                "diversification_ratio": self._calculate_diversification_ratio(
                    component_returns, weights, correlations
                ),
            },
            confidence_intervals=confidence_intervals,
            correlations=correlations
        )
    
    def _hierarchical_aggregate(self,
                              results: List[Dict[str, Any]],
                              weights: np.ndarray) -> AggregatedResult:
        """Perform hierarchical aggregation."""
        # Group results by strategy type or other criteria
        grouped_results = defaultdict(list)
        
        for i, result in enumerate(results):
            group = result.get("strategy_type", "default")
            grouped_results[group].append((result, weights[i]))
            
        # First level: aggregate within groups
        group_aggregates = {}
        for group, group_data in grouped_results.items():
            group_results = [r for r, _ in group_data]
            group_weights = np.array([w for _, w in group_data])
            group_weights = group_weights / group_weights.sum()
            
            group_aggregates[group] = self._portfolio_aggregate(
                group_results, group_weights
            )
            
        # Second level: aggregate across groups
        final_results = []
        final_weights = []
        
        for group, aggregate in group_aggregates.items():
            final_results.append(aggregate.aggregate_metrics)
            final_weights.append(len(grouped_results[group]))
            
        final_weights = np.array(final_weights) / sum(final_weights)
        
        # Final aggregation
        final_metrics = self.aggregate_metrics(final_results, final_weights)
        
        # Compile statistics
        all_statistics = {}
        for group, aggregate in group_aggregates.items():
            for metric, stats in aggregate.statistics.items():
                if metric not in all_statistics:
                    all_statistics[metric] = {}
                all_statistics[metric][f"{group}_mean"] = stats.get("mean", 0)
                all_statistics[metric][f"{group}_std"] = stats.get("std", 0)
                
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(results)
        
        return AggregatedResult(
            aggregate_metrics=final_metrics,
            statistics=all_statistics,
            component_results=results,
            metadata={
                "aggregation_method": "hierarchical",
                "n_components": len(results),
                "n_groups": len(group_aggregates),
                "groups": list(group_aggregates.keys()),
            },
            confidence_intervals=confidence_intervals
        )
    
    def _apply_aggregation_method(self,
                                values: List[float],
                                method: AggregationMethod,
                                weights: Optional[np.ndarray] = None) -> float:
        """Apply specific aggregation method to values."""
        # Filter out NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        if not valid_indices:
            return np.nan
            
        valid_values = [values[i] for i in valid_indices]
        
        if weights is not None:
            valid_weights = [weights[i] for i in valid_indices]
            valid_weights = np.array(valid_weights) / np.sum(valid_weights)
        else:
            valid_weights = None
            
        if method == AggregationMethod.MEAN:
            return np.mean(valid_values)
        elif method == AggregationMethod.WEIGHTED_MEAN:
            if valid_weights is not None:
                return np.average(valid_values, weights=valid_weights)
            return np.mean(valid_values)
        elif method == AggregationMethod.MEDIAN:
            return np.median(valid_values)
        elif method == AggregationMethod.SUM:
            return np.sum(valid_values)
        elif method == AggregationMethod.MIN:
            return np.min(valid_values)
        elif method == AggregationMethod.MAX:
            return np.max(valid_values)
        elif method == AggregationMethod.CUSTOM:
            # Look for custom aggregator
            custom_func = self._custom_aggregators.get(method.value)
            if custom_func:
                return custom_func(valid_values, valid_weights)
            return np.mean(valid_values)
        else:
            return np.mean(valid_values)
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each metric."""
        if not self.config.calculate_statistics:
            return {}
            
        statistics = {}
        
        # Collect all metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(k for k, v in result.items() 
                             if isinstance(v, (int, float)))
            
        # Calculate statistics for each metric
        for metric in all_metrics:
            values = [r.get(metric, np.nan) for r in results]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                statistics[metric] = {
                    "mean": np.mean(valid_values),
                    "std": np.std(valid_values),
                    "min": np.min(valid_values),
                    "max": np.max(valid_values),
                    "median": np.median(valid_values),
                    "skew": stats.skew(valid_values),
                    "kurtosis": stats.kurtosis(valid_values),
                    "count": len(valid_values),
                }
            else:
                statistics[metric] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                    "skew": np.nan,
                    "kurtosis": np.nan,
                    "count": 0,
                }
                
        return statistics
    
    def _calculate_confidence_intervals(self, 
                                      results: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics."""
        confidence_intervals = {}
        
        # Collect all metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(k for k, v in result.items() 
                             if isinstance(v, (int, float)))
            
        # Calculate CI for each metric
        for metric in all_metrics:
            values = [r.get(metric, np.nan) for r in results]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if len(valid_values) >= 2:
                mean = np.mean(valid_values)
                sem = stats.sem(valid_values)
                ci = stats.t.interval(
                    self.config.confidence_level,
                    len(valid_values) - 1,
                    loc=mean,
                    scale=sem
                )
                confidence_intervals[metric] = ci
            else:
                confidence_intervals[metric] = (np.nan, np.nan)
                
        return confidence_intervals
    
    def _calculate_diversification_ratio(self,
                                       component_returns: List[np.ndarray],
                                       weights: np.ndarray,
                                       correlations: Optional[pd.DataFrame]) -> float:
        """Calculate portfolio diversification ratio."""
        if not component_returns or correlations is None:
            return 1.0
            
        # Calculate individual volatilities
        volatilities = [np.std(returns) * np.sqrt(252) for returns in component_returns]
        
        # Weighted average volatility
        weighted_vol = np.dot(weights[:len(volatilities)], volatilities)
        
        # Portfolio volatility
        cov_matrix = correlations.values * np.outer(volatilities, volatilities)
        portfolio_vol = np.sqrt(np.dot(weights[:len(volatilities)], 
                                     np.dot(cov_matrix, weights[:len(volatilities)])))
        
        # Diversification ratio
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0