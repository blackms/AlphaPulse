"""
Market Regime Feature Engineering for HMM Detection.

This module provides comprehensive feature engineering for market regime detection,
including volatility measures, return characteristics, liquidity indicators, and
sentiment features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import talib
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegimeFeatureConfig:
    """Configuration for regime feature extraction."""
    volatility_windows: List[int] = None
    return_windows: List[int] = None
    volume_windows: List[int] = None
    use_vix: bool = True
    use_sentiment: bool = True
    use_liquidity: bool = True
    normalize_features: bool = True
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 60]
        if self.return_windows is None:
            self.return_windows = [1, 5, 20, 60]
        if self.volume_windows is None:
            self.volume_windows = [5, 10, 20]


class RegimeFeatureEngineer:
    """Engineer features for market regime detection."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        self.config = config or RegimeFeatureConfig()
        self.feature_names: List[str] = []
        self.scaler_params: Dict[str, Dict[str, float]] = {}
        
    def extract_features(self, 
                        data: pd.DataFrame,
                        additional_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Extract all regime features from market data.
        
        Args:
            data: DataFrame with columns: open, high, low, close, volume
            additional_data: Optional dict with VIX, sentiment, etc.
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Extract volatility features
        volatility_features = self._extract_volatility_features(data)
        features = pd.concat([features, volatility_features], axis=1)
        
        # Extract return features
        return_features = self._extract_return_features(data)
        features = pd.concat([features, return_features], axis=1)
        
        # Extract volume/liquidity features
        if self.config.use_liquidity:
            liquidity_features = self._extract_liquidity_features(data)
            features = pd.concat([features, liquidity_features], axis=1)
        
        # Extract sentiment features if available
        if self.config.use_sentiment and additional_data and 'sentiment' in additional_data:
            sentiment_features = self._extract_sentiment_features(
                data, additional_data['sentiment']
            )
            features = pd.concat([features, sentiment_features], axis=1)
        
        # Add VIX features if available
        if self.config.use_vix and additional_data and 'vix' in additional_data:
            vix_features = self._extract_vix_features(additional_data['vix'])
            features = pd.concat([features, vix_features], axis=1)
        
        # Add technical indicators
        technical_features = self._extract_technical_features(data)
        features = pd.concat([features, technical_features], axis=1)
        
        # Add credit spread features if available
        if additional_data and 'credit_spreads' in additional_data:
            credit_features = self._extract_credit_spread_features(additional_data['credit_spreads'])
            features = pd.concat([features, credit_features], axis=1)
        
        # Add yield curve features if available
        if additional_data and 'yield_curve' in additional_data:
            yield_features = self._extract_yield_curve_features(additional_data['yield_curve'])
            features = pd.concat([features, yield_features], axis=1)
        
        # Forward fill then drop remaining NaN rows
        features = features.ffill().dropna()
        
        # Normalize features if configured
        if self.config.normalize_features:
            features = self._normalize_features(features)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility-based features."""
        features = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change()
        
        for window in self.config.volatility_windows:
            # Realized volatility
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
            
            # GARCH-style volatility (simplified)
            squared_returns = returns ** 2
            features[f'garch_vol_{window}d'] = np.sqrt(
                squared_returns.rolling(window).mean() * 252
            )
            
            # EWMA volatility
            features[f'ewma_vol_{window}d'] = returns.ewm(span=window).std() * np.sqrt(252)
            
            # Volatility of volatility
            vol_series = returns.rolling(window).std()
            features[f'vol_of_vol_{window}d'] = vol_series.rolling(window).std()
            
            # Parkinson volatility (using high-low)
            hl_ratio = np.log(data['high'] / data['low'])
            features[f'parkinson_vol_{window}d'] = (
                hl_ratio.rolling(window).apply(lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))))
                * np.sqrt(252)
            )
        
        # Volatility term structure
        if len(self.config.volatility_windows) >= 2:
            short_window = self.config.volatility_windows[0]
            long_window = self.config.volatility_windows[-1]
            features['vol_term_structure'] = (
                features[f'volatility_{short_window}d'] / 
                features[f'volatility_{long_window}d']
            )
        
        return features
    
    def _extract_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract return-based features."""
        features = pd.DataFrame(index=data.index)
        
        for window in self.config.return_windows:
            # Simple returns
            features[f'return_{window}d'] = data['close'].pct_change(window)
            
            # Log returns
            features[f'log_return_{window}d'] = np.log(data['close'] / data['close'].shift(window))
            
            # Return momentum
            if window > 1:
                features[f'momentum_{window}d'] = (
                    data['close'] / data['close'].shift(window) - 1
                )
            
            # Return skewness and kurtosis
            returns = data['close'].pct_change()
            features[f'skewness_{window}d'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
            
            # Maximum drawdown
            rolling_max = data['close'].rolling(window).max()
            drawdown = (data['close'] - rolling_max) / rolling_max
            features[f'max_drawdown_{window}d'] = drawdown.rolling(window).min()
        
        # Trend strength (ADX)
        features['adx_14'] = talib.ADX(
            data['high'].values,
            data['low'].values, 
            data['close'].values,
            timeperiod=14
        )
        
        return features
    
    def _extract_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract liquidity and volume features."""
        features = pd.DataFrame(index=data.index)
        
        for window in self.config.volume_windows:
            # Volume metrics
            features[f'volume_ma_{window}d'] = data['volume'].rolling(window).mean()
            features[f'volume_std_{window}d'] = data['volume'].rolling(window).std()
            
            # Relative volume
            features[f'relative_volume_{window}d'] = (
                data['volume'] / data['volume'].rolling(window).mean()
            )
            
            # Dollar volume
            dollar_volume = data['close'] * data['volume']
            features[f'dollar_volume_{window}d'] = dollar_volume.rolling(window).mean()
        
        # Amihud illiquidity measure (simplified)
        returns = data['close'].pct_change().abs()
        dollar_volume = data['close'] * data['volume']
        features['amihud_illiquidity'] = (returns / dollar_volume).rolling(20).mean()
        
        # Volume-price correlation
        features['volume_price_corr'] = (
            data['close'].pct_change().rolling(20).corr(data['volume'].pct_change())
        )
        
        return features
    
    def _extract_sentiment_features(self, 
                                  data: pd.DataFrame,
                                  sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Extract sentiment-based features."""
        features = pd.DataFrame(index=data.index)
        
        if 'put_call_ratio' in sentiment_data.columns:
            features['put_call_ratio'] = sentiment_data['put_call_ratio']
            features['put_call_ma_20'] = features['put_call_ratio'].rolling(20).mean()
        
        if 'sentiment_score' in sentiment_data.columns:
            features['sentiment_score'] = sentiment_data['sentiment_score']
            features['sentiment_ma_5'] = features['sentiment_score'].rolling(5).mean()
            features['sentiment_std_20'] = features['sentiment_score'].rolling(20).std()
        
        if 'fear_greed_index' in sentiment_data.columns:
            features['fear_greed_index'] = sentiment_data['fear_greed_index']
            features['fear_greed_change'] = features['fear_greed_index'].diff()
        
        return features
    
    def _extract_vix_features(self, vix_data: pd.DataFrame) -> pd.DataFrame:
        """Extract VIX-based features."""
        features = pd.DataFrame(index=vix_data.index)
        
        # VIX level
        features['vix_level'] = vix_data['close']
        
        # VIX changes
        features['vix_change_1d'] = features['vix_level'].pct_change()
        features['vix_change_5d'] = features['vix_level'].pct_change(5)
        
        # VIX term structure (if available)
        if 'vix9d' in vix_data.columns and 'vix30d' in vix_data.columns:
            features['vix_term_structure'] = vix_data['vix9d'] / vix_data['vix30d']
        
        # VIX percentile
        features['vix_percentile_252d'] = (
            features['vix_level'].rolling(252).rank(pct=True)
        )
        
        # VIX regime
        features['vix_high_regime'] = (features['vix_level'] > 20).astype(int)
        features['vix_extreme_regime'] = (features['vix_level'] > 30).astype(int)
        
        return features
    
    def _extract_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicator features."""
        features = pd.DataFrame(index=data.index)
        
        # RSI
        features['rsi_14'] = talib.RSI(data['close'].values, timeperiod=14)
        features['rsi_oversold'] = (features['rsi_14'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi_14'] > 70).astype(int)
        
        # MACD
        macd, signal, hist = talib.MACD(
            data['close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        features['macd_hist'] = hist
        features['macd_signal'] = (macd > signal).astype(int)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )
        features['bb_position'] = (data['close'] - lower) / (upper - lower)
        features['bb_width'] = (upper - lower) / middle
        
        # ATR (Average True Range)
        features['atr_14'] = talib.ATR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=14
        )
        features['atr_ratio'] = features['atr_14'] / data['close']
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using rolling statistics."""
        normalized = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            # Use rolling z-score normalization
            rolling_mean = features[col].rolling(252, min_periods=60).mean()
            rolling_std = features[col].rolling(252, min_periods=60).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1e-8)
            
            normalized[col] = (features[col] - rolling_mean) / rolling_std
            
            # Clip extreme values
            normalized[col] = normalized[col].clip(-3, 3)
            
            # Store normalization parameters
            self.scaler_params[col] = {
                'mean': rolling_mean.iloc[-1],
                'std': rolling_std.iloc[-1]
            }
        
        return normalized
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using stored normalization parameters."""
        if not self.config.normalize_features:
            return features
        
        normalized = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            if col in self.scaler_params:
                mean = self.scaler_params[col]['mean']
                std = self.scaler_params[col]['std']
                normalized[col] = (features[col] - mean) / std
                normalized[col] = normalized[col].clip(-3, 3)
            else:
                logger.warning(f"No normalization parameters for feature: {col}")
                normalized[col] = features[col]
        
        return normalized
    
    def get_feature_importance(self, 
                             features: pd.DataFrame,
                             regimes: np.ndarray) -> pd.DataFrame:
        """Calculate feature importance for regime detection."""
        importance_scores = []
        
        for feature in features.columns:
            # ANOVA F-statistic for each feature across regimes
            groups = [features[feature][regimes == r] for r in np.unique(regimes)]
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Mutual information (simplified)
            mi_score = self._mutual_information(features[feature].values, regimes)
            
            importance_scores.append({
                'feature': feature,
                'f_statistic': f_stat,
                'p_value': p_value,
                'mutual_information': mi_score
            })
        
        return pd.DataFrame(importance_scores).sort_values(
            'f_statistic', ascending=False
        )
    
    def _mutual_information(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between feature and regimes."""
        # Simplified mutual information calculation
        # In practice, use sklearn.feature_selection.mutual_info_classif
        
        # Discretize continuous feature
        X_discrete = pd.qcut(X, q=10, labels=False, duplicates='drop')
        
        # Calculate MI
        contingency = pd.crosstab(X_discrete, y)
        mi = 0.0
        n = len(X)
        
        for i in contingency.index:
            for j in contingency.columns:
                pij = contingency.loc[i, j] / n
                pi = contingency.loc[i].sum() / n
                pj = contingency[j].sum() / n
                
                if pij > 0:
                    mi += pij * np.log(pij / (pi * pj))
        
        return mi