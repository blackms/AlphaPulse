"""
Correlation analysis for portfolio risk management.

Provides comprehensive correlation analysis including dynamic correlations,
regime detection, and tail dependency analysis using copulas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.stats import kendalltau, spearmanr
from statsmodels.tsa.stattools import acf
from sklearn.covariance import LedoitWolf
import warnings

from alpha_pulse.models.correlation_matrix import (
    CorrelationMatrix, CorrelationType, CorrelationRegime
)
from alpha_pulse.utils.statistical_analysis import (
    calculate_rolling_statistics, detect_structural_breaks
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """Correlation calculation methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    DISTANCE = "distance"
    TAIL = "tail"


@dataclass
class CorrelationAnalysisConfig:
    """Configuration for correlation analysis."""
    lookback_period: int = 252  # Trading days
    rolling_window: int = 63    # ~3 months
    min_observations: int = 30
    correlation_methods: List[CorrelationMethod] = field(
        default_factory=lambda: [CorrelationMethod.PEARSON, CorrelationMethod.SPEARMAN]
    )
    detect_regimes: bool = True
    calculate_tail_dependencies: bool = True
    shrinkage_target: str = "constant"  # constant, diagonal, empirical
    confidence_level: float = 0.95


@dataclass
class TailDependency:
    """Tail dependency metrics between assets."""
    asset1: str
    asset2: str
    lower_tail: float  # Lower tail dependence coefficient
    upper_tail: float  # Upper tail dependence coefficient
    asymmetry: float   # Difference between upper and lower
    confidence_interval: Tuple[float, float]


@dataclass
class CorrelationBreakdown:
    """Detailed correlation breakdown."""
    systematic_correlation: float
    idiosyncratic_correlation: float
    factor_contributions: Dict[str, float]
    time_varying_component: float
    structural_component: float


class CorrelationAnalyzer:
    """Advanced correlation analysis for portfolios."""
    
    def __init__(self, config: CorrelationAnalysisConfig = None):
        """Initialize correlation analyzer."""
        self.config = config or CorrelationAnalysisConfig()
        self.correlation_cache = {}
        self.regime_history = []
        
    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        apply_shrinkage: bool = True
    ) -> CorrelationMatrix:
        """Calculate correlation matrix with specified method."""
        if len(returns) < self.config.min_observations:
            raise ValueError(f"Insufficient data: {len(returns)} < {self.config.min_observations}")
        
        # Calculate base correlation
        if method == CorrelationMethod.PEARSON:
            corr_matrix = returns.corr(method='pearson')
        elif method == CorrelationMethod.SPEARMAN:
            corr_matrix = returns.corr(method='spearman')
        elif method == CorrelationMethod.KENDALL:
            corr_matrix = returns.corr(method='kendall')
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Apply shrinkage if requested
        if apply_shrinkage and method == CorrelationMethod.PEARSON:
            corr_matrix = self._apply_shrinkage(returns)
        
        # Create correlation matrix object
        return CorrelationMatrix(
            matrix=corr_matrix.values,
            assets=corr_matrix.columns.tolist(),
            correlation_type=CorrelationType.FULL,
            timestamp=datetime.utcnow(),
            metadata={
                "method": method.value,
                "observations": len(returns),
                "shrinkage_applied": apply_shrinkage
            }
        )
    
    def calculate_rolling_correlations(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
        method: CorrelationMethod = CorrelationMethod.PEARSON
    ) -> Dict[str, pd.DataFrame]:
        """Calculate rolling correlations between all asset pairs."""
        window = window or self.config.rolling_window
        
        # Initialize storage
        rolling_corrs = {}
        assets = returns.columns.tolist()
        
        # Calculate for each pair
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                pair_key = f"{asset1}_{asset2}"
                
                if method == CorrelationMethod.PEARSON:
                    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
                elif method == CorrelationMethod.SPEARMAN:
                    # Custom rolling Spearman
                    rolling_corr = self._rolling_spearman(
                        returns[asset1], returns[asset2], window
                    )
                else:
                    # Custom rolling Kendall
                    rolling_corr = self._rolling_kendall(
                        returns[asset1], returns[asset2], window
                    )
                
                rolling_corrs[pair_key] = rolling_corr
        
        return rolling_corrs
    
    def detect_correlation_regimes(
        self,
        returns: pd.DataFrame,
        n_regimes: int = 3
    ) -> List[CorrelationRegime]:
        """Detect correlation regimes using statistical methods."""
        # Calculate rolling average correlation
        rolling_corrs = self.calculate_rolling_correlations(returns)
        avg_corr = pd.DataFrame(rolling_corrs).mean(axis=1)
        
        # Detect regime changes
        breakpoints = detect_structural_breaks(avg_corr.dropna(), n_regimes - 1)
        
        # Define regimes
        regimes = []
        start_idx = 0
        
        for i, end_idx in enumerate(breakpoints + [len(avg_corr)]):
            if end_idx <= start_idx:
                continue
                
            # Calculate regime statistics
            regime_data = returns.iloc[start_idx:end_idx]
            regime_corr = self.calculate_correlation_matrix(regime_data)
            
            # Determine regime type
            avg_correlation = np.mean(regime_corr.get_average_correlation())
            if avg_correlation > 0.7:
                regime_type = "high_correlation"
            elif avg_correlation < 0.3:
                regime_type = "low_correlation"
            else:
                regime_type = "normal_correlation"
            
            regime = CorrelationRegime(
                regime_id=f"regime_{i}",
                start_date=returns.index[start_idx],
                end_date=returns.index[min(end_idx - 1, len(returns) - 1)],
                regime_type=regime_type,
                average_correlation=avg_correlation,
                volatility_regime=self._classify_volatility_regime(regime_data)
            )
            
            regimes.append(regime)
            start_idx = end_idx
        
        self.regime_history = regimes
        return regimes
    
    def calculate_conditional_correlations(
        self,
        returns: pd.DataFrame,
        conditioning_variable: Optional[pd.Series] = None
    ) -> Dict[str, CorrelationMatrix]:
        """Calculate correlations conditional on market conditions."""
        if conditioning_variable is None:
            # Use market volatility as default conditioning variable
            conditioning_variable = returns.std(axis=1).rolling(20).mean()
        
        # Define quantiles for conditioning
        quantiles = [0.2, 0.5, 0.8]
        thresholds = conditioning_variable.quantile(quantiles)
        
        conditional_corrs = {}
        
        # Calculate correlations for each condition
        conditions = ["low", "medium", "high", "very_high"]
        for i, condition in enumerate(conditions):
            if i == 0:
                mask = conditioning_variable <= thresholds.iloc[0]
            elif i == len(conditions) - 1:
                mask = conditioning_variable > thresholds.iloc[-1]
            else:
                mask = (conditioning_variable > thresholds.iloc[i-1]) & \
                       (conditioning_variable <= thresholds.iloc[i])
            
            if mask.sum() >= self.config.min_observations:
                conditional_returns = returns[mask]
                conditional_corrs[condition] = self.calculate_correlation_matrix(
                    conditional_returns
                )
            else:
                conditional_corrs[condition] = None
        
        return conditional_corrs
    
    def calculate_tail_dependencies(
        self,
        returns: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[TailDependency]:
        """Calculate tail dependencies using empirical methods."""
        tail_deps = []
        assets = returns.columns.tolist()
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                # Convert to uniform margins using empirical CDF
                u1 = returns[asset1].rank() / (len(returns) + 1)
                u2 = returns[asset2].rank() / (len(returns) + 1)
                
                # Calculate tail dependence coefficients
                lower_tail = self._empirical_tail_dependence(u1, u2, 1 - threshold, "lower")
                upper_tail = self._empirical_tail_dependence(u1, u2, threshold, "upper")
                
                # Calculate confidence intervals
                ci_lower, ci_upper = self._tail_dependence_ci(
                    u1, u2, threshold, n_bootstrap=100
                )
                
                tail_dep = TailDependency(
                    asset1=asset1,
                    asset2=asset2,
                    lower_tail=lower_tail,
                    upper_tail=upper_tail,
                    asymmetry=upper_tail - lower_tail,
                    confidence_interval=(ci_lower, ci_upper)
                )
                
                tail_deps.append(tail_dep)
        
        return tail_deps
    
    def decompose_correlation(
        self,
        returns: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None
    ) -> Dict[str, CorrelationBreakdown]:
        """Decompose correlations into systematic and idiosyncratic components."""
        breakdowns = {}
        
        if factors is None:
            # Use PCA to extract factors
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(3, len(returns.columns) - 1))
            factors = pd.DataFrame(
                pca.fit_transform(returns),
                index=returns.index,
                columns=[f"PC{i+1}" for i in range(pca.n_components_)]
            )
        
        # For each asset pair
        assets = returns.columns.tolist()
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                # Factor model regression
                from sklearn.linear_model import LinearRegression
                
                # Regress returns on factors
                model1 = LinearRegression()
                model2 = LinearRegression()
                
                model1.fit(factors, returns[asset1])
                model2.fit(factors, returns[asset2])
                
                # Get systematic and idiosyncratic returns
                systematic1 = model1.predict(factors)
                systematic2 = model2.predict(factors)
                
                idiosyncratic1 = returns[asset1] - systematic1
                idiosyncratic2 = returns[asset2] - systematic2
                
                # Calculate correlations
                total_corr = returns[asset1].corr(returns[asset2])
                systematic_corr = np.corrcoef(systematic1, systematic2)[0, 1]
                idiosyncratic_corr = np.corrcoef(idiosyncratic1, idiosyncratic2)[0, 1]
                
                # Factor contributions
                factor_contribs = {}
                for k, factor in enumerate(factors.columns):
                    contrib = model1.coef_[k] * model2.coef_[k] * factors[factor].var()
                    factor_contribs[factor] = contrib / total_corr if total_corr != 0 else 0
                
                breakdown = CorrelationBreakdown(
                    systematic_correlation=systematic_corr,
                    idiosyncratic_correlation=idiosyncratic_corr,
                    factor_contributions=factor_contribs,
                    time_varying_component=self._calculate_time_varying_component(
                        returns[asset1], returns[asset2]
                    ),
                    structural_component=total_corr - systematic_corr
                )
                
                breakdowns[f"{asset1}_{asset2}"] = breakdown
        
        return breakdowns
    
    def calculate_distance_correlation(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate distance correlation matrix (captures non-linear dependencies)."""
        from scipy.spatial.distance import pdist, squareform
        
        n_assets = len(returns.columns)
        dcorr_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(i, n_assets):
                if i == j:
                    dcorr_matrix[i, j] = 1.0
                else:
                    dcorr = self._distance_correlation(
                        returns.iloc[:, i].values,
                        returns.iloc[:, j].values
                    )
                    dcorr_matrix[i, j] = dcorr
                    dcorr_matrix[j, i] = dcorr
        
        return pd.DataFrame(
            dcorr_matrix,
            index=returns.columns,
            columns=returns.columns
        )
    
    def _apply_shrinkage(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Apply Ledoit-Wolf shrinkage to correlation matrix."""
        # Standardize returns
        standardized = (returns - returns.mean()) / returns.std()
        
        # Apply Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        shrunk_cov, _ = lw.fit(standardized).covariance_, lw.shrinkage_
        
        # Convert to correlation
        D = np.sqrt(np.diag(shrunk_cov))
        shrunk_corr = shrunk_cov / np.outer(D, D)
        
        return pd.DataFrame(
            shrunk_corr,
            index=returns.columns,
            columns=returns.columns
        )
    
    def _rolling_spearman(
        self,
        series1: pd.Series,
        series2: pd.Series,
        window: int
    ) -> pd.Series:
        """Calculate rolling Spearman correlation."""
        def spearman_corr(x, y):
            if len(x) < 3:
                return np.nan
            return spearmanr(x, y)[0]
        
        return series1.rolling(window).apply(
            lambda x: spearman_corr(x, series2.iloc[-len(x):])
        )
    
    def _rolling_kendall(
        self,
        series1: pd.Series,
        series2: pd.Series,
        window: int
    ) -> pd.Series:
        """Calculate rolling Kendall tau correlation."""
        def kendall_corr(x, y):
            if len(x) < 3:
                return np.nan
            return kendalltau(x, y)[0]
        
        return series1.rolling(window).apply(
            lambda x: kendall_corr(x, series2.iloc[-len(x):])
        )
    
    def _empirical_tail_dependence(
        self,
        u1: pd.Series,
        u2: pd.Series,
        threshold: float,
        tail: str = "upper"
    ) -> float:
        """Calculate empirical tail dependence coefficient."""
        if tail == "upper":
            condition = (u1 > threshold) & (u2 > threshold)
            base = u1 > threshold
        else:
            condition = (u1 < threshold) & (u2 < threshold)
            base = u1 < threshold
        
        if base.sum() == 0:
            return 0.0
        
        return condition.sum() / base.sum()
    
    def _tail_dependence_ci(
        self,
        u1: pd.Series,
        u2: pd.Series,
        threshold: float,
        n_bootstrap: int = 100
    ) -> Tuple[float, float]:
        """Calculate confidence interval for tail dependence using bootstrap."""
        bootstrap_estimates = []
        n = len(u1)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            u1_boot = u1.iloc[indices]
            u2_boot = u2.iloc[indices]
            
            # Calculate tail dependence
            tail_dep = self._empirical_tail_dependence(
                u1_boot, u2_boot, threshold, "upper"
            )
            bootstrap_estimates.append(tail_dep)
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(bootstrap_estimates, alpha/2 * 100)
        upper = np.percentile(bootstrap_estimates, (1 - alpha/2) * 100)
        
        return lower, upper
    
    def _classify_volatility_regime(self, returns: pd.DataFrame) -> str:
        """Classify volatility regime."""
        avg_vol = returns.std().mean() * np.sqrt(252)  # Annualized
        
        if avg_vol < 0.10:
            return "low_volatility"
        elif avg_vol < 0.20:
            return "normal_volatility"
        elif avg_vol < 0.30:
            return "high_volatility"
        else:
            return "extreme_volatility"
    
    def _calculate_time_varying_component(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> float:
        """Calculate time-varying component of correlation."""
        # Use rolling correlation variance as proxy
        rolling_corr = series1.rolling(63).corr(series2)
        return rolling_corr.std() if not rolling_corr.isna().all() else 0.0
    
    def _distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance correlation between two variables."""
        n = len(x)
        
        # Calculate distance matrices
        a = np.abs(np.subtract.outer(x, x))
        b = np.abs(np.subtract.outer(y, y))
        
        # Double-center the distance matrices
        a_centered = a - a.mean(axis=0, keepdims=True) - \
                     a.mean(axis=1, keepdims=True) + a.mean()
        b_centered = b - b.mean(axis=0, keepdims=True) - \
                     b.mean(axis=1, keepdims=True) + b.mean()
        
        # Calculate distance covariance and variances
        dcov_xy = np.sqrt(np.sum(a_centered * b_centered) / (n * n))
        dcov_xx = np.sqrt(np.sum(a_centered * a_centered) / (n * n))
        dcov_yy = np.sqrt(np.sum(b_centered * b_centered) / (n * n))
        
        # Calculate distance correlation
        if dcov_xx * dcov_yy > 0:
            dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        else:
            dcorr = 0.0
        
        return dcorr
    
    def get_correlation_summary(
        self,
        returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get comprehensive correlation analysis summary."""
        # Calculate various correlation measures
        pearson_corr = self.calculate_correlation_matrix(
            returns, CorrelationMethod.PEARSON
        )
        spearman_corr = self.calculate_correlation_matrix(
            returns, CorrelationMethod.SPEARMAN
        )
        
        # Average correlations
        avg_pearson = pearson_corr.get_average_correlation()
        avg_spearman = spearman_corr.get_average_correlation()
        
        # Correlation concentration
        corr_values = pearson_corr.matrix[np.triu_indices_from(
            pearson_corr.matrix, k=1
        )]
        
        summary = {
            "average_correlation": {
                "pearson": avg_pearson,
                "spearman": avg_spearman,
                "difference": avg_pearson - avg_spearman
            },
            "correlation_distribution": {
                "min": np.min(corr_values),
                "25%": np.percentile(corr_values, 25),
                "median": np.median(corr_values),
                "75%": np.percentile(corr_values, 75),
                "max": np.max(corr_values),
                "std": np.std(corr_values)
            },
            "high_correlation_pairs": self._find_high_correlation_pairs(
                pearson_corr, threshold=0.8
            ),
            "negative_correlation_pairs": self._find_negative_correlation_pairs(
                pearson_corr, threshold=-0.3
            ),
            "correlation_stability": self._assess_correlation_stability(returns)
        }
        
        return summary
    
    def _find_high_correlation_pairs(
        self,
        corr_matrix: CorrelationMatrix,
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find highly correlated asset pairs."""
        high_corr_pairs = []
        n_assets = len(corr_matrix.assets)
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = corr_matrix.matrix[i, j]
                if corr > threshold:
                    high_corr_pairs.append({
                        "asset1": corr_matrix.assets[i],
                        "asset2": corr_matrix.assets[j],
                        "correlation": corr
                    })
        
        return sorted(high_corr_pairs, key=lambda x: x["correlation"], reverse=True)
    
    def _find_negative_correlation_pairs(
        self,
        corr_matrix: CorrelationMatrix,
        threshold: float = -0.3
    ) -> List[Dict[str, Any]]:
        """Find negatively correlated asset pairs."""
        neg_corr_pairs = []
        n_assets = len(corr_matrix.assets)
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = corr_matrix.matrix[i, j]
                if corr < threshold:
                    neg_corr_pairs.append({
                        "asset1": corr_matrix.assets[i],
                        "asset2": corr_matrix.assets[j],
                        "correlation": corr
                    })
        
        return sorted(neg_corr_pairs, key=lambda x: x["correlation"])
    
    def _assess_correlation_stability(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Assess stability of correlations over time."""
        rolling_corrs = self.calculate_rolling_correlations(returns)
        
        stability_metrics = {}
        for pair, corr_series in rolling_corrs.items():
            clean_series = corr_series.dropna()
            if len(clean_series) > 0:
                stability_metrics[pair] = {
                    "mean": clean_series.mean(),
                    "std": clean_series.std(),
                    "coefficient_of_variation": clean_series.std() / abs(clean_series.mean())
                    if clean_series.mean() != 0 else np.inf
                }
        
        # Overall stability
        all_stds = [m["std"] for m in stability_metrics.values()]
        overall_stability = {
            "average_std": np.mean(all_stds) if all_stds else 0,
            "max_std": np.max(all_stds) if all_stds else 0,
            "stability_score": 1 - np.mean(all_stds) if all_stds else 1
        }
        
        return overall_stability