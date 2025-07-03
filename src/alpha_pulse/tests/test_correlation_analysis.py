"""
Tests for correlation analysis functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from alpha_pulse.risk.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationMethod,
    CorrelationAnalysisConfig,
    TailDependency,
    CorrelationBreakdown
)
from alpha_pulse.models.correlation_matrix import (
    CorrelationMatrix,
    CorrelationType,
    CorrelationRegime
)


class TestCorrelationAnalyzer:
    """Test correlation analyzer functionality."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        n_days = 252
        n_assets = 5
        
        # Generate correlated returns
        mean = np.zeros(n_assets)
        
        # Create correlation structure
        corr = np.array([
            [1.0, 0.7, 0.3, -0.2, 0.1],
            [0.7, 1.0, 0.5, -0.1, 0.2],
            [0.3, 0.5, 1.0, 0.0, 0.4],
            [-0.2, -0.1, 0.0, 1.0, 0.6],
            [0.1, 0.2, 0.4, 0.6, 1.0]
        ])
        
        # Generate covariance matrix
        vol = np.array([0.01, 0.015, 0.02, 0.012, 0.018])
        cov = np.outer(vol, vol) * corr
        
        # Generate returns
        returns = np.random.multivariate_normal(mean, cov, n_days)
        
        return pd.DataFrame(
            returns,
            index=pd.date_range(start='2023-01-01', periods=n_days, freq='D'),
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
        )
    
    @pytest.fixture
    def analyzer(self):
        """Create correlation analyzer instance."""
        config = CorrelationAnalysisConfig(
            lookback_period=252,
            rolling_window=63,
            min_observations=30
        )
        return CorrelationAnalyzer(config)
    
    def test_calculate_correlation_matrix_pearson(self, analyzer, sample_returns):
        """Test Pearson correlation calculation."""
        corr_matrix = analyzer.calculate_correlation_matrix(
            sample_returns,
            method=CorrelationMethod.PEARSON
        )
        
        assert isinstance(corr_matrix, CorrelationMatrix)
        assert corr_matrix.matrix.shape == (5, 5)
        assert np.allclose(np.diag(corr_matrix.matrix), 1.0)
        assert corr_matrix.correlation_type == CorrelationType.FULL
        
        # Check symmetry
        assert np.allclose(corr_matrix.matrix, corr_matrix.matrix.T)
        
        # Check correlation bounds
        assert np.all(corr_matrix.matrix >= -1)
        assert np.all(corr_matrix.matrix <= 1)
    
    def test_calculate_correlation_matrix_spearman(self, analyzer, sample_returns):
        """Test Spearman correlation calculation."""
        corr_matrix = analyzer.calculate_correlation_matrix(
            sample_returns,
            method=CorrelationMethod.SPEARMAN
        )
        
        assert isinstance(corr_matrix, CorrelationMatrix)
        assert corr_matrix.matrix.shape == (5, 5)
        assert corr_matrix.metadata['method'] == 'spearman'
    
    def test_calculate_rolling_correlations(self, analyzer, sample_returns):
        """Test rolling correlation calculation."""
        rolling_corrs = analyzer.calculate_rolling_correlations(
            sample_returns,
            window=63
        )
        
        # Check we have correlations for all pairs
        n_pairs = 5 * 4 // 2  # n*(n-1)/2
        assert len(rolling_corrs) == n_pairs
        
        # Check each series
        for pair_key, corr_series in rolling_corrs.items():
            assert isinstance(corr_series, pd.Series)
            assert len(corr_series) == len(sample_returns)
            # First window-1 values should be NaN
            assert corr_series.iloc[:62].isna().all()
            # Remaining values should be valid correlations
            valid_corrs = corr_series.dropna()
            assert np.all(valid_corrs >= -1)
            assert np.all(valid_corrs <= 1)
    
    def test_detect_correlation_regimes(self, analyzer, sample_returns):
        """Test correlation regime detection."""
        # Create data with regime change
        regime1_returns = sample_returns.iloc[:126]
        regime2_returns = sample_returns.iloc[126:].copy()
        
        # Increase correlations in second regime
        for col in regime2_returns.columns[1:]:
            regime2_returns[col] = regime2_returns[col] * 0.5 + \
                                  regime2_returns.iloc[:, 0] * 0.5
        
        combined_returns = pd.concat([regime1_returns, regime2_returns])
        
        regimes = analyzer.detect_correlation_regimes(combined_returns, n_regimes=2)
        
        assert len(regimes) >= 1
        assert all(isinstance(r, CorrelationRegime) for r in regimes)
        
        # Check regime properties
        for regime in regimes:
            assert regime.duration_days > 0
            assert 0 <= regime.average_correlation <= 1
            assert regime.regime_type in ['high_correlation', 'low_correlation', 
                                         'normal_correlation']
    
    def test_calculate_conditional_correlations(self, analyzer, sample_returns):
        """Test conditional correlation calculation."""
        # Use volatility as conditioning variable
        volatility = sample_returns.std(axis=1).rolling(20).mean()
        
        conditional_corrs = analyzer.calculate_conditional_correlations(
            sample_returns,
            conditioning_variable=volatility
        )
        
        assert 'low' in conditional_corrs
        assert 'medium' in conditional_corrs
        assert 'high' in conditional_corrs
        assert 'very_high' in conditional_corrs
        
        # Check each conditional correlation matrix
        for condition, corr_matrix in conditional_corrs.items():
            if corr_matrix is not None:
                assert isinstance(corr_matrix, CorrelationMatrix)
                assert corr_matrix.matrix.shape == (5, 5)
    
    def test_calculate_tail_dependencies(self, analyzer, sample_returns):
        """Test tail dependency calculation."""
        tail_deps = analyzer.calculate_tail_dependencies(
            sample_returns,
            threshold=0.95
        )
        
        # Check we have tail dependencies for all pairs
        n_pairs = 5 * 4 // 2
        assert len(tail_deps) == n_pairs
        
        for tail_dep in tail_deps:
            assert isinstance(tail_dep, TailDependency)
            assert 0 <= tail_dep.lower_tail <= 1
            assert 0 <= tail_dep.upper_tail <= 1
            assert isinstance(tail_dep.confidence_interval, tuple)
            assert len(tail_dep.confidence_interval) == 2
    
    def test_decompose_correlation(self, analyzer, sample_returns):
        """Test correlation decomposition."""
        breakdowns = analyzer.decompose_correlation(sample_returns)
        
        # Check we have breakdowns for all pairs
        n_pairs = 5 * 4 // 2
        assert len(breakdowns) == n_pairs
        
        for pair_key, breakdown in breakdowns.items():
            assert isinstance(breakdown, CorrelationBreakdown)
            assert -1 <= breakdown.systematic_correlation <= 1
            assert -1 <= breakdown.idiosyncratic_correlation <= 1
            assert isinstance(breakdown.factor_contributions, dict)
            assert breakdown.time_varying_component >= 0
    
    def test_calculate_distance_correlation(self, analyzer, sample_returns):
        """Test distance correlation calculation."""
        dcorr_matrix = analyzer.calculate_distance_correlation(sample_returns)
        
        assert isinstance(dcorr_matrix, pd.DataFrame)
        assert dcorr_matrix.shape == (5, 5)
        
        # Check properties
        assert np.allclose(np.diag(dcorr_matrix), 1.0)
        assert np.all(dcorr_matrix >= 0)  # Distance correlation is non-negative
        assert np.all(dcorr_matrix <= 1)
        
        # Check symmetry
        assert np.allclose(dcorr_matrix.values, dcorr_matrix.values.T)
    
    def test_get_correlation_summary(self, analyzer, sample_returns):
        """Test correlation summary generation."""
        summary = analyzer.get_correlation_summary(sample_returns)
        
        assert 'average_correlation' in summary
        assert 'correlation_distribution' in summary
        assert 'high_correlation_pairs' in summary
        assert 'negative_correlation_pairs' in summary
        assert 'correlation_stability' in summary
        
        # Check average correlation
        avg_corr = summary['average_correlation']
        assert 'pearson' in avg_corr
        assert 'spearman' in avg_corr
        assert -1 <= avg_corr['pearson'] <= 1
        
        # Check distribution stats
        dist = summary['correlation_distribution']
        assert dist['min'] <= dist['25%'] <= dist['median'] <= dist['75%'] <= dist['max']
    
    def test_shrinkage_estimation(self, analyzer, sample_returns):
        """Test correlation matrix shrinkage."""
        # Test with small sample to see shrinkage effect
        small_sample = sample_returns.iloc[:50]
        
        corr_raw = analyzer.calculate_correlation_matrix(
            small_sample,
            method=CorrelationMethod.PEARSON,
            apply_shrinkage=False
        )
        
        corr_shrunk = analyzer.calculate_correlation_matrix(
            small_sample,
            method=CorrelationMethod.PEARSON,
            apply_shrinkage=True
        )
        
        # Shrunk correlations should be pulled toward the mean
        off_diag_raw = corr_raw.matrix[np.triu_indices_from(corr_raw.matrix, k=1)]
        off_diag_shrunk = corr_shrunk.matrix[np.triu_indices_from(corr_shrunk.matrix, k=1)]
        
        # Variance of shrunk correlations should be lower
        assert np.var(off_diag_shrunk) <= np.var(off_diag_raw)
    
    def test_correlation_with_missing_data(self, analyzer, sample_returns):
        """Test correlation calculation with missing data."""
        # Add some NaN values
        returns_with_nan = sample_returns.copy()
        returns_with_nan.iloc[10:20, 0] = np.nan
        returns_with_nan.iloc[30:40, 2] = np.nan
        
        corr_matrix = analyzer.calculate_correlation_matrix(
            returns_with_nan,
            method=CorrelationMethod.PEARSON
        )
        
        assert isinstance(corr_matrix, CorrelationMatrix)
        assert not np.any(np.isnan(corr_matrix.matrix))
    
    def test_regime_detection_edge_cases(self, analyzer):
        """Test regime detection with edge cases."""
        # Test with constant returns
        constant_returns = pd.DataFrame(
            np.ones((100, 3)) * 0.01,
            columns=['A', 'B', 'C']
        )
        
        regimes = analyzer.detect_correlation_regimes(constant_returns, n_regimes=2)
        assert len(regimes) >= 1
        
        # Test with too few observations
        small_returns = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['A', 'B', 'C']
        )
        
        regimes = analyzer.detect_correlation_regimes(small_returns, n_regimes=2)
        assert len(regimes) >= 1
    
    def test_tail_dependency_bootstrap_ci(self, analyzer, sample_returns):
        """Test bootstrap confidence intervals for tail dependencies."""
        # Use smaller bootstrap samples for speed
        analyzer.config.confidence_level = 0.90
        
        tail_deps = analyzer.calculate_tail_dependencies(
            sample_returns.iloc[:, :2],  # Just two assets for speed
            threshold=0.90
        )
        
        assert len(tail_deps) == 1
        tail_dep = tail_deps[0]
        
        # Check confidence interval
        ci_lower, ci_upper = tail_dep.confidence_interval
        assert 0 <= ci_lower <= ci_upper <= 1
        assert ci_lower <= tail_dep.upper_tail <= ci_upper


class TestCorrelationMatrix:
    """Test CorrelationMatrix model."""
    
    def test_correlation_matrix_creation(self):
        """Test creating correlation matrix."""
        matrix = np.array([
            [1.0, 0.5, -0.3],
            [0.5, 1.0, 0.2],
            [-0.3, 0.2, 1.0]
        ])
        
        corr_matrix = CorrelationMatrix(
            matrix=matrix,
            assets=['A', 'B', 'C'],
            correlation_type=CorrelationType.FULL,
            timestamp=datetime.now()
        )
        
        assert corr_matrix.get_correlation('A', 'B') == 0.5
        assert corr_matrix.get_correlation('A', 'C') == -0.3
        assert corr_matrix.get_correlation('B', 'C') == 0.2
    
    def test_correlation_matrix_statistics(self):
        """Test correlation matrix statistics."""
        matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.4],
            [0.6, 0.4, 1.0]
        ])
        
        corr_matrix = CorrelationMatrix(
            matrix=matrix,
            assets=['A', 'B', 'C'],
            correlation_type=CorrelationType.FULL,
            timestamp=datetime.now()
        )
        
        # Test average correlation
        avg_corr = corr_matrix.get_average_correlation()
        expected_avg = (0.8 + 0.6 + 0.4) / 3
        assert abs(avg_corr - expected_avg) < 1e-10
        
        # Test max correlation
        asset1, asset2, max_corr = corr_matrix.get_max_correlation()
        assert max_corr == 0.8
        assert (asset1, asset2) == ('A', 'B') or (asset1, asset2) == ('B', 'A')
        
        # Test min correlation
        asset1, asset2, min_corr = corr_matrix.get_min_correlation()
        assert min_corr == 0.4
        assert (asset1, asset2) == ('B', 'C') or (asset1, asset2) == ('C', 'B')
    
    def test_correlation_matrix_validation(self):
        """Test correlation matrix validation."""
        # Test non-square matrix
        with pytest.raises(ValueError, match="must be square"):
            CorrelationMatrix(
                matrix=np.array([[1.0, 0.5], [0.5, 1.0, 0.3]]),
                assets=['A', 'B'],
                correlation_type=CorrelationType.FULL,
                timestamp=datetime.now()
            )
        
        # Test dimension mismatch
        with pytest.raises(ValueError, match="must match number of assets"):
            CorrelationMatrix(
                matrix=np.eye(3),
                assets=['A', 'B'],
                correlation_type=CorrelationType.FULL,
                timestamp=datetime.now()
            )