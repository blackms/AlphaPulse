"""
Statistical analysis utilities for financial data.

Provides functions for statistical tests, time series analysis,
and structural break detection.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from scipy import stats
from scipy.stats import jarque_bera, anderson, kstest
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')


def calculate_rolling_statistics(
    series: pd.Series,
    window: int = 20,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """Calculate comprehensive rolling statistics."""
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    rolling = series.rolling(window=window, min_periods=min_periods)
    
    stats_df = pd.DataFrame(index=series.index)
    stats_df['mean'] = rolling.mean()
    stats_df['std'] = rolling.std()
    stats_df['skew'] = rolling.skew()
    stats_df['kurt'] = rolling.kurt()
    stats_df['min'] = rolling.min()
    stats_df['max'] = rolling.max()
    stats_df['range'] = stats_df['max'] - stats_df['min']
    
    # Coefficient of variation
    stats_df['cv'] = stats_df['std'] / stats_df['mean'].abs()
    
    # Z-score
    stats_df['z_score'] = (series - stats_df['mean']) / stats_df['std']
    
    return stats_df


def detect_structural_breaks(
    series: pd.Series,
    max_breaks: int = 5,
    min_segment_size: int = 30
) -> List[int]:
    """
    Detect structural breaks in time series using Bai-Perron method.
    
    Returns list of break point indices.
    """
    n = len(series)
    values = series.values
    
    if n < min_segment_size * 2:
        return []
    
    # Simple implementation using recursive residual sum of squares
    break_points = []
    
    def find_single_break(data: np.ndarray, start_idx: int = 0) -> Optional[int]:
        """Find single break point in data segment."""
        n_segment = len(data)
        if n_segment < min_segment_size * 2:
            return None
        
        min_rss = np.inf
        best_break = None
        
        for i in range(min_segment_size, n_segment - min_segment_size):
            # Fit separate means before and after break
            rss1 = np.sum((data[:i] - np.mean(data[:i]))**2)
            rss2 = np.sum((data[i:] - np.mean(data[i:]))**2)
            total_rss = rss1 + rss2
            
            if total_rss < min_rss:
                min_rss = total_rss
                best_break = i
        
        # Test if break is significant using Chow test
        if best_break:
            f_stat = chow_test(data, best_break)
            if f_stat['p_value'] < 0.05:
                return start_idx + best_break
        
        return None
    
    # Recursively find breaks
    segments = [(0, n)]
    
    for _ in range(max_breaks):
        best_break = None
        best_segment_idx = None
        max_improvement = 0
        
        # Check each segment for breaks
        for idx, (start, end) in enumerate(segments):
            segment_data = values[start:end]
            
            if len(segment_data) >= min_segment_size * 2:
                break_point = find_single_break(segment_data, start)
                
                if break_point:
                    # Calculate improvement in fit
                    orig_rss = np.sum((segment_data - np.mean(segment_data))**2)
                    new_rss1 = np.sum((values[start:break_point] - 
                                     np.mean(values[start:break_point]))**2)
                    new_rss2 = np.sum((values[break_point:end] - 
                                     np.mean(values[break_point:end]))**2)
                    improvement = orig_rss - (new_rss1 + new_rss2)
                    
                    if improvement > max_improvement:
                        max_improvement = improvement
                        best_break = break_point
                        best_segment_idx = idx
        
        if best_break:
            break_points.append(best_break)
            # Update segments
            start, end = segments[best_segment_idx]
            segments[best_segment_idx] = (start, best_break)
            segments.insert(best_segment_idx + 1, (best_break, end))
        else:
            break
    
    return sorted(break_points)


def chow_test(data: np.ndarray, break_point: int) -> Dict[str, float]:
    """
    Perform Chow test for structural break.
    
    Tests if coefficients are equal before and after break point.
    """
    n = len(data)
    
    # Calculate RSS for full sample
    mean_full = np.mean(data)
    rss_full = np.sum((data - mean_full)**2)
    
    # Calculate RSS for subsamples
    mean1 = np.mean(data[:break_point])
    mean2 = np.mean(data[break_point:])
    rss1 = np.sum((data[:break_point] - mean1)**2)
    rss2 = np.sum((data[break_point:] - mean2)**2)
    rss_restricted = rss1 + rss2
    
    # Calculate F-statistic
    k = 1  # Number of parameters (just mean in this case)
    f_stat = ((rss_full - rss_restricted) / k) / (rss_restricted / (n - 2*k))
    
    # Calculate p-value
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'break_significant': p_value < 0.05
    }


def test_stationarity(
    series: pd.Series,
    test_type: str = 'both'
) -> Dict[str, Any]:
    """
    Test time series stationarity using ADF and KPSS tests.
    
    Parameters:
    - test_type: 'adf', 'kpss', or 'both'
    """
    results = {}
    
    if test_type in ['adf', 'both']:
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna(), autolag='AIC')
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
    
    if test_type in ['kpss', 'both']:
        # KPSS test
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
    
    # Combined result
    if test_type == 'both':
        results['combined_stationary'] = (
            results['adf']['is_stationary'] and 
            results['kpss']['is_stationary']
        )
    
    return results


def test_normality(series: pd.Series) -> Dict[str, Any]:
    """
    Test if series follows normal distribution.
    
    Uses multiple tests for robustness.
    """
    clean_series = series.dropna()
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = jarque_bera(clean_series)
    
    # Anderson-Darling test
    ad_result = anderson(clean_series, dist='norm')
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = kstest(clean_series, 'norm', 
                                args=(clean_series.mean(), clean_series.std()))
    
    # Shapiro-Wilk test (for smaller samples)
    if len(clean_series) <= 5000:
        sw_stat, sw_pvalue = stats.shapiro(clean_series)
    else:
        sw_stat, sw_pvalue = np.nan, np.nan
    
    return {
        'jarque_bera': {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        },
        'anderson_darling': {
            'statistic': ad_result.statistic,
            'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], 
                                      ad_result.critical_values)),
            'is_normal': ad_result.statistic < ad_result.critical_values[2]  # 5% level
        },
        'kolmogorov_smirnov': {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'is_normal': ks_pvalue > 0.05
        },
        'shapiro_wilk': {
            'statistic': sw_stat,
            'p_value': sw_pvalue,
            'is_normal': sw_pvalue > 0.05 if not np.isnan(sw_pvalue) else None
        },
        'combined_normal': all([
            jb_pvalue > 0.05,
            ad_result.statistic < ad_result.critical_values[2],
            ks_pvalue > 0.05
        ])
    }


def test_autocorrelation(
    series: pd.Series,
    lags: int = 20
) -> Dict[str, Any]:
    """Test for autocorrelation in time series."""
    clean_series = series.dropna()
    
    # Ljung-Box test
    lb_result = acorr_ljungbox(clean_series, lags=lags, return_df=True)
    
    # ACF and PACF
    acf_values = acf(clean_series, nlags=lags, fft=True)
    pacf_values = pacf(clean_series, nlags=lags)
    
    # Find significant lags
    n = len(clean_series)
    confidence_interval = 1.96 / np.sqrt(n)
    
    significant_acf_lags = [
        i for i in range(1, len(acf_values))
        if abs(acf_values[i]) > confidence_interval
    ]
    
    significant_pacf_lags = [
        i for i in range(1, len(pacf_values))
        if abs(pacf_values[i]) > confidence_interval
    ]
    
    return {
        'ljung_box': {
            'statistics': lb_result['lb_stat'].values,
            'p_values': lb_result['lb_pvalue'].values,
            'has_autocorrelation': any(lb_result['lb_pvalue'] < 0.05)
        },
        'acf': {
            'values': acf_values,
            'significant_lags': significant_acf_lags,
            'confidence_interval': confidence_interval
        },
        'pacf': {
            'values': pacf_values,
            'significant_lags': significant_pacf_lags,
            'confidence_interval': confidence_interval
        }
    }


def calculate_information_criteria(
    residuals: np.ndarray,
    n_params: int
) -> Dict[str, float]:
    """Calculate information criteria for model selection."""
    n = len(residuals)
    rss = np.sum(residuals**2)
    log_likelihood = -n/2 * (np.log(2*np.pi) + np.log(rss/n) + 1)
    
    # Akaike Information Criterion
    aic = -2 * log_likelihood + 2 * n_params
    
    # Bayesian Information Criterion
    bic = -2 * log_likelihood + n_params * np.log(n)
    
    # Hannan-Quinn Criterion
    hqc = -2 * log_likelihood + 2 * n_params * np.log(np.log(n))
    
    return {
        'aic': aic,
        'bic': bic,
        'hqc': hqc,
        'log_likelihood': log_likelihood
    }


def detect_outliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect outliers using various methods.
    
    Methods:
    - 'iqr': Interquartile range
    - 'zscore': Z-score method
    - 'mad': Median absolute deviation
    - 'isolation': Isolation forest (requires sklearn)
    """
    clean_series = series.dropna()
    
    if method == 'iqr':
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (clean_series < lower_bound) | (clean_series > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(clean_series))
        outliers = z_scores > threshold
        
    elif method == 'mad':
        median = clean_series.median()
        mad = np.median(np.abs(clean_series - median))
        modified_z_scores = 0.6745 * (clean_series - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        
    elif method == 'isolation':
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(clean_series.values.reshape(-1, 1)) == -1
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    outlier_indices = clean_series.index[outliers].tolist()
    
    return {
        'method': method,
        'n_outliers': int(outliers.sum()),
        'outlier_percentage': outliers.sum() / len(clean_series) * 100,
        'outlier_indices': outlier_indices,
        'outlier_values': clean_series[outliers].tolist() if outliers.any() else [],
        'lower_bound': lower_bound if method == 'iqr' else None,
        'upper_bound': upper_bound if method == 'iqr' else None
    }


def calculate_tail_statistics(
    series: pd.Series,
    tail_probability: float = 0.05
) -> Dict[str, float]:
    """Calculate statistics for distribution tails."""
    clean_series = series.dropna()
    
    # Calculate VaR and CVaR
    var_lower = clean_series.quantile(tail_probability)
    var_upper = clean_series.quantile(1 - tail_probability)
    
    cvar_lower = clean_series[clean_series <= var_lower].mean()
    cvar_upper = clean_series[clean_series >= var_upper].mean()
    
    # Calculate tail indices (Hill estimator)
    sorted_series = clean_series.sort_values()
    n_tail = int(len(clean_series) * tail_probability)
    
    # Lower tail index
    lower_tail_values = sorted_series.iloc[:n_tail].values
    if len(lower_tail_values) > 1 and all(lower_tail_values < 0):
        lower_tail_index = len(lower_tail_values) / np.sum(
            np.log(-lower_tail_values / lower_tail_values[-1])
        )
    else:
        lower_tail_index = np.nan
    
    # Upper tail index
    upper_tail_values = sorted_series.iloc[-n_tail:].values
    if len(upper_tail_values) > 1 and all(upper_tail_values > 0):
        upper_tail_index = len(upper_tail_values) / np.sum(
            np.log(upper_tail_values / upper_tail_values[0])
        )
    else:
        upper_tail_index = np.nan
    
    return {
        'var_lower': var_lower,
        'var_upper': var_upper,
        'cvar_lower': cvar_lower,
        'cvar_upper': cvar_upper,
        'lower_tail_index': lower_tail_index,
        'upper_tail_index': upper_tail_index,
        'tail_asymmetry': abs(cvar_lower) - cvar_upper if not np.isnan(cvar_lower) else np.nan
    }


def perform_granger_causality_test(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 10
) -> Dict[str, Any]:
    """
    Test for Granger causality between two time series.
    
    Tests if series1 Granger-causes series2.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Prepare data
    data = pd.DataFrame({
        'series1': series1,
        'series2': series2
    }).dropna()
    
    # Run test for different lags
    try:
        results = grangercausalitytests(
            data[['series2', 'series1']], 
            maxlag=max_lag,
            verbose=False
        )
        
        # Extract p-values for each lag
        p_values = {}
        for lag in range(1, max_lag + 1):
            # Get p-value from F-test
            p_values[lag] = results[lag][0]['ssr_ftest'][1]
        
        # Find optimal lag
        optimal_lag = min(p_values, key=p_values.get)
        
        return {
            'p_values': p_values,
            'optimal_lag': optimal_lag,
            'is_causal': p_values[optimal_lag] < 0.05,
            'test_results': results
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'is_causal': False
        }