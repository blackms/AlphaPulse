"""
Slippage prediction models for market impact estimation.

Implements various slippage models including linear, square-root,
Almgren-Chriss, and machine learning based approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from alpha_pulse.models.slippage_estimates import (
    SlippageModel, SlippageParameters, SlippageEstimate,
    MarketImpactEstimate, ImpactType, ExecutionStrategy,
    OptimalExecutionParams
)
from alpha_pulse.models.liquidity_metrics import LiquidityMetrics

logger = logging.getLogger(__name__)


class BaseSlippageModel:
    """Base class for slippage models."""
    
    def __init__(self, model_type: SlippageModel):
        """Initialize base slippage model."""
        self.model_type = model_type
        self.parameters = SlippageParameters()
        self.is_calibrated = False
        
    def estimate_slippage(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        execution_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate slippage in basis points."""
        raise NotImplementedError
    
    def calibrate(
        self,
        historical_trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calibrate model parameters using historical data."""
        raise NotImplementedError


class LinearImpactModel(BaseSlippageModel):
    """Linear market impact model."""
    
    def __init__(self):
        """Initialize linear impact model."""
        super().__init__(SlippageModel.LINEAR)
        
    def estimate_slippage(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        execution_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate slippage using linear model."""
        # Extract market parameters
        adv = market_data.get('average_daily_volume', 1e6)
        spread = market_data.get('spread_bps', 10)
        volatility = market_data.get('volatility', 0.02)
        
        # Calculate participation rate
        participation_rate = order_size / adv
        
        # Linear impact
        temporary_impact = (
            self.parameters.temporary_impact_coefficient * 
            participation_rate * 10000  # Convert to bps
        )
        
        permanent_impact = (
            self.parameters.permanent_impact_coefficient * 
            participation_rate * 10000
        )
        
        # Add spread cost
        spread_cost = spread / 2  # Half spread
        
        # Total slippage
        total_slippage = spread_cost + temporary_impact + permanent_impact
        
        return total_slippage
    
    def calibrate(
        self,
        historical_trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calibrate linear model parameters."""
        logger.info("Calibrating linear impact model")
        
        # Prepare features
        features = []
        targets = []
        
        for _, trade in historical_trades.iterrows():
            # Get market data for trade
            symbol = trade['symbol']
            trade_date = trade['execution_date']
            
            # Calculate participation rate
            adv = market_data.loc[trade_date, f'{symbol}_adv']
            participation = trade['order_size'] / adv
            
            # Features: participation rate
            features.append([participation])
            
            # Target: realized slippage
            targets.append(trade['realized_slippage_bps'])
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Simple linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(features, targets)
        
        # Update parameters
        self.parameters.temporary_impact_coefficient = model.coef_[0] / 10000
        self.parameters.permanent_impact_coefficient = model.coef_[0] / 20000  # Half for permanent
        
        # Calculate R-squared
        r_squared = model.score(features, targets)
        
        self.is_calibrated = True
        self.parameters.calibration_date = datetime.now()
        self.parameters.calibration_r_squared = r_squared
        
        return {
            'r_squared': r_squared,
            'temp_impact_coef': self.parameters.temporary_impact_coefficient,
            'perm_impact_coef': self.parameters.permanent_impact_coefficient
        }


class SquareRootModel(BaseSlippageModel):
    """Square-root market impact model (Barra-style)."""
    
    def __init__(self):
        """Initialize square-root model."""
        super().__init__(SlippageModel.SQUARE_ROOT)
        
    def estimate_slippage(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        execution_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate slippage using square-root model."""
        # Extract market parameters
        adv = market_data.get('average_daily_volume', 1e6)
        spread = market_data.get('spread_bps', 10)
        volatility = market_data.get('volatility', 0.02)
        
        # Execution parameters
        if execution_params:
            duration = execution_params.get('duration_minutes', 60)
            urgency = execution_params.get('urgency', 1.0)
        else:
            duration = 60
            urgency = 1.0
        
        # Calculate participation rate
        participation_rate = order_size / adv
        
        # Square-root impact formula
        # Impact = spread/2 + sigma * sqrt(order_size/ADV) * liquidity_factor
        temporary_impact = (
            volatility * 
            np.sqrt(participation_rate) * 
            self.parameters.liquidity_factor * 
            urgency * 
            10000  # Convert to bps
        )
        
        # Permanent impact (smaller than temporary)
        permanent_impact = temporary_impact * self.parameters.permanent_impact_coefficient
        
        # Timing risk (decreases with longer duration)
        timing_risk = volatility * np.sqrt(duration / 390) * 10000 * 0.5  # Half day adjustment
        
        # Total slippage
        total_slippage = spread / 2 + temporary_impact + permanent_impact + timing_risk
        
        return total_slippage
    
    def calibrate(
        self,
        historical_trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calibrate square-root model parameters."""
        logger.info("Calibrating square-root impact model")
        
        def objective(params):
            """Objective function for calibration."""
            self.parameters.liquidity_factor = params[0]
            self.parameters.permanent_impact_coefficient = params[1]
            
            errors = []
            for _, trade in historical_trades.iterrows():
                # Prepare market data
                market_info = {
                    'average_daily_volume': trade['adv'],
                    'spread_bps': trade['spread_bps'],
                    'volatility': trade['volatility']
                }
                
                # Estimate slippage
                estimated = self.estimate_slippage(
                    trade['order_size'],
                    market_info,
                    {'duration_minutes': trade.get('duration_minutes', 60)}
                )
                
                # Calculate error
                error = (estimated - trade['realized_slippage_bps']) ** 2
                errors.append(error)
            
            return np.mean(errors)
        
        # Optimize parameters
        result = minimize(
            objective,
            x0=[0.5, 0.1],  # Initial guess
            bounds=[(0.1, 2.0), (0.01, 0.5)],
            method='L-BFGS-B'
        )
        
        # Calculate R-squared
        predictions = []
        actuals = []
        
        for _, trade in historical_trades.iterrows():
            market_info = {
                'average_daily_volume': trade['adv'],
                'spread_bps': trade['spread_bps'],
                'volatility': trade['volatility']
            }
            
            estimated = self.estimate_slippage(
                trade['order_size'],
                market_info,
                {'duration_minutes': trade.get('duration_minutes', 60)}
            )
            
            predictions.append(estimated)
            actuals.append(trade['realized_slippage_bps'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        self.is_calibrated = True
        self.parameters.calibration_date = datetime.now()
        self.parameters.calibration_r_squared = r_squared
        
        return {
            'r_squared': r_squared,
            'liquidity_factor': self.parameters.liquidity_factor,
            'perm_impact_coef': self.parameters.permanent_impact_coefficient
        }


class AlmgrenChrissModel(BaseSlippageModel):
    """Almgren-Chriss optimal execution model."""
    
    def __init__(self):
        """Initialize Almgren-Chriss model."""
        super().__init__(SlippageModel.ALMGREN_CHRISS)
        self.optimal_trajectory = None
        
    def estimate_slippage(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        execution_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate slippage using Almgren-Chriss model."""
        # Extract parameters
        volatility = market_data.get('volatility', 0.02)
        adv = market_data.get('average_daily_volume', 1e6)
        spread = market_data.get('spread_bps', 10)
        
        # Execution parameters
        if execution_params:
            risk_aversion = execution_params.get('risk_aversion', 1.0)
            duration_minutes = execution_params.get('duration_minutes', 60)
        else:
            risk_aversion = 1.0
            duration_minutes = 60
        
        # Calculate optimal execution
        optimal_params = self._calculate_optimal_execution(
            order_size, volatility, adv, spread, risk_aversion, duration_minutes
        )
        
        # Expected cost (temporary + permanent impact)
        expected_cost = optimal_params['expected_cost']
        
        return expected_cost
    
    def _calculate_optimal_execution(
        self,
        order_size: float,
        volatility: float,
        adv: float,
        spread: float,
        risk_aversion: float,
        duration_minutes: float
    ) -> Dict[str, Any]:
        """Calculate optimal execution trajectory."""
        # Time parameters
        T = duration_minutes / 390  # Fraction of trading day
        n_slices = max(1, int(duration_minutes / 5))  # 5-minute slices
        tau = T / n_slices
        
        # Market impact parameters
        eta = self.parameters.temporary_impact_coefficient  # Temporary impact
        gamma = self.parameters.permanent_impact_coefficient  # Permanent impact
        
        # Optimal trade schedule (simplified)
        # In full implementation, would solve optimization problem
        kappa = np.sqrt(risk_aversion * volatility**2 / (eta * tau))
        
        # Generate trading trajectory
        times = np.linspace(0, T, n_slices + 1)
        
        # Optimal trajectory (linear for simplified version)
        # Full version would use sinh/cosh functions
        remaining = order_size * (1 - times / T)
        trades = -np.diff(remaining)
        
        # Calculate expected cost
        # Temporary impact cost
        temp_cost = eta * np.sum(trades**2 / (adv * tau)) * 10000
        
        # Permanent impact cost
        perm_cost = gamma * order_size / adv * 10000
        
        # Spread cost
        spread_cost = spread / 2
        
        # Timing risk (variance of cost)
        variance = volatility**2 * order_size**2 * T / 3
        
        # Total expected cost
        total_cost = spread_cost + temp_cost + perm_cost
        
        # Risk-adjusted cost
        risk_adjusted_cost = total_cost + risk_aversion * np.sqrt(variance) * 10000
        
        self.optimal_trajectory = trades
        
        return {
            'expected_cost': total_cost,
            'risk_adjusted_cost': risk_adjusted_cost,
            'variance': variance,
            'trajectory': trades,
            'temporary_impact': temp_cost,
            'permanent_impact': perm_cost
        }
    
    def get_optimal_trajectory(
        self,
        order_size: float,
        duration_minutes: float,
        n_slices: int
    ) -> np.ndarray:
        """Get optimal execution trajectory."""
        if self.optimal_trajectory is not None:
            return self.optimal_trajectory
        
        # Simple linear trajectory as fallback
        return np.ones(n_slices) * order_size / n_slices
    
    def calibrate(
        self,
        historical_trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calibrate Almgren-Chriss model parameters."""
        logger.info("Calibrating Almgren-Chriss model")
        
        # This would require more sophisticated calibration
        # For now, use heuristic values
        self.parameters.temporary_impact_coefficient = 0.1
        self.parameters.permanent_impact_coefficient = 0.05
        
        self.is_calibrated = True
        self.parameters.calibration_date = datetime.now()
        
        return {
            'status': 'calibrated',
            'method': 'heuristic'
        }


class MachineLearningSlippageModel(BaseSlippageModel):
    """Machine learning based slippage prediction model."""
    
    def __init__(self):
        """Initialize ML slippage model."""
        super().__init__(SlippageModel.MACHINE_LEARNING)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def estimate_slippage(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        execution_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate slippage using ML model."""
        if not self.is_calibrated:
            # Fallback to simple estimate
            return 10.0  # Default 10 bps
        
        # Prepare features
        features = self._prepare_features(order_size, market_data, execution_params)
        features_scaled = self.scaler.transform([features])
        
        # Predict slippage
        slippage_bps = self.model.predict(features_scaled)[0]
        
        return max(0, slippage_bps)  # Ensure non-negative
    
    def _prepare_features(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        execution_params: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Prepare features for ML model."""
        features = []
        
        # Order features
        adv = market_data.get('average_daily_volume', 1e6)
        features.append(order_size / adv)  # Participation rate
        features.append(np.log(order_size))  # Log order size
        
        # Market features
        features.append(market_data.get('spread_bps', 10))
        features.append(market_data.get('volatility', 0.02))
        features.append(market_data.get('volume_volatility', 0.5))
        
        # Liquidity features
        features.append(market_data.get('amihud_illiquidity', 0.1))
        features.append(market_data.get('depth_imbalance', 0.0))
        features.append(market_data.get('liquidity_score', 50))
        
        # Execution features
        if execution_params:
            features.append(execution_params.get('duration_minutes', 60))
            features.append(execution_params.get('urgency', 1.0))
            features.append(execution_params.get('aggressiveness', 0.5))
        else:
            features.extend([60, 1.0, 0.5])
        
        # Time features
        hour = market_data.get('hour', 12)
        features.append(hour)
        features.append(int(hour in [9, 10, 15, 16]))  # Opening/closing hour
        
        # Market regime features
        features.append(market_data.get('market_return', 0.0))
        features.append(market_data.get('vix_level', 20))
        
        return features
    
    def calibrate(
        self,
        historical_trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Train ML model on historical data."""
        logger.info("Training machine learning slippage model")
        
        # Prepare training data
        X = []
        y = []
        
        for _, trade in historical_trades.iterrows():
            # Get market data for trade
            trade_market_data = self._get_trade_market_data(trade, market_data)
            
            # Prepare features
            features = self._prepare_features(
                trade['order_size'],
                trade_market_data,
                {
                    'duration_minutes': trade.get('duration_minutes', 60),
                    'urgency': trade.get('urgency', 1.0),
                    'aggressiveness': trade.get('aggressiveness', 0.5)
                }
            )
            
            X.append(features)
            y.append(trade['realized_slippage_bps'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        
        self.is_calibrated = True
        self.parameters.calibration_date = datetime.now()
        self.parameters.calibration_r_squared = val_score
        
        return {
            'train_r_squared': train_score,
            'val_r_squared': val_score,
            'n_features': len(feature_importance),
            'top_features': self._get_top_features(feature_importance)
        }
    
    def _get_trade_market_data(
        self,
        trade: pd.Series,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract market data for a specific trade."""
        # This would extract relevant market data
        # For now, return dummy data
        return {
            'average_daily_volume': trade.get('adv', 1e6),
            'spread_bps': trade.get('spread_bps', 10),
            'volatility': trade.get('volatility', 0.02),
            'volume_volatility': 0.5,
            'amihud_illiquidity': 0.1,
            'depth_imbalance': 0.0,
            'liquidity_score': 50,
            'hour': trade.get('hour', 12),
            'market_return': 0.0,
            'vix_level': 20
        }
    
    def _get_top_features(self, importance: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Get top n important features."""
        feature_names = [
            'participation_rate', 'log_order_size', 'spread_bps',
            'volatility', 'volume_volatility', 'amihud_illiquidity',
            'depth_imbalance', 'liquidity_score', 'duration_minutes',
            'urgency', 'aggressiveness', 'hour', 'is_open_close',
            'market_return', 'vix_level'
        ]
        
        # Get indices of top features
        top_indices = np.argsort(importance)[-n:][::-1]
        
        top_features = [
            (feature_names[i], importance[i])
            for i in top_indices
            if i < len(feature_names)
        ]
        
        return top_features
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if self.is_calibrated:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'parameters': self.parameters,
                'feature_names': self.feature_names
            }, filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.parameters = data['parameters']
        self.feature_names = data['feature_names']
        self.is_calibrated = True
        logger.info(f"Model loaded from {filepath}")


class SlippageModelEnsemble:
    """Ensemble of slippage models for robust predictions."""
    
    def __init__(self):
        """Initialize model ensemble."""
        self.models = {
            SlippageModel.LINEAR: LinearImpactModel(),
            SlippageModel.SQUARE_ROOT: SquareRootModel(),
            SlippageModel.ALMGREN_CHRISS: AlmgrenChrissModel(),
            SlippageModel.MACHINE_LEARNING: MachineLearningSlippageModel()
        }
        
        # Model weights (can be optimized)
        self.weights = {
            SlippageModel.LINEAR: 0.15,
            SlippageModel.SQUARE_ROOT: 0.35,
            SlippageModel.ALMGREN_CHRISS: 0.30,
            SlippageModel.MACHINE_LEARNING: 0.20
        }
        
    def estimate_slippage(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        execution_params: Optional[Dict[str, Any]] = None,
        confidence_level: float = 0.95
    ) -> SlippageEstimate:
        """Estimate slippage using ensemble of models."""
        logger.info(f"Estimating slippage for order size {order_size}")
        
        # Get estimates from each model
        model_estimates = {}
        
        for model_type, model in self.models.items():
            try:
                estimate = model.estimate_slippage(
                    order_size, market_data, execution_params
                )
                model_estimates[model_type] = estimate
            except Exception as e:
                logger.warning(f"Model {model_type} failed: {e}")
                model_estimates[model_type] = np.nan
        
        # Calculate weighted average
        valid_estimates = [
            (est, self.weights[model_type])
            for model_type, est in model_estimates.items()
            if not np.isnan(est)
        ]
        
        if not valid_estimates:
            # Fallback estimate
            ensemble_estimate = 10.0  # Default 10 bps
        else:
            total_weight = sum(w for _, w in valid_estimates)
            ensemble_estimate = sum(
                est * w / total_weight for est, w in valid_estimates
            )
        
        # Calculate confidence interval
        if len(valid_estimates) > 1:
            estimates_array = np.array([est for est, _ in valid_estimates])
            std_dev = np.std(estimates_array)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            confidence_interval = (
                ensemble_estimate - z_score * std_dev,
                ensemble_estimate + z_score * std_dev
            )
        else:
            confidence_interval = (ensemble_estimate * 0.8, ensemble_estimate * 1.2)
        
        # Determine recommended execution strategy
        recommended_strategy = self._recommend_execution_strategy(
            order_size, market_data, ensemble_estimate
        )
        
        # Create SlippageEstimate object
        return SlippageEstimate(
            order_id=f"est_{datetime.now().timestamp()}",
            symbol=market_data.get('symbol', 'UNKNOWN'),
            timestamp=datetime.now(),
            order_size=order_size,
            order_value=order_size * market_data.get('price', 100),
            side=market_data.get('side', 'buy'),
            urgency=execution_params.get('urgency', 'medium') if execution_params else 'medium',
            current_spread=market_data.get('spread_bps', 10),
            current_volatility=market_data.get('volatility', 0.02),
            average_daily_volume=market_data.get('average_daily_volume', 1e6),
            current_volume=market_data.get('current_volume', 0),
            model_estimates=model_estimates,
            ensemble_estimate=ensemble_estimate,
            recommended_strategy=recommended_strategy,
            recommended_duration=execution_params.get('duration_minutes', 60) if execution_params else 60,
            recommended_participation=min(0.3, order_size / market_data.get('average_daily_volume', 1e6)),
            expected_slippage_bps=ensemble_estimate,
            expected_slippage_dollars=order_size * market_data.get('price', 100) * ensemble_estimate / 10000,
            worst_case_slippage_bps=confidence_interval[1],
            models_used=list(model_estimates.keys()),
            model_weights=self.weights,
            estimation_confidence=1 - std_dev / ensemble_estimate if len(valid_estimates) > 1 else 0.8
        )
    
    def _recommend_execution_strategy(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        expected_slippage: float
    ) -> ExecutionStrategy:
        """Recommend optimal execution strategy."""
        # Simple heuristic-based recommendation
        participation_rate = order_size / market_data.get('average_daily_volume', 1e6)
        volatility = market_data.get('volatility', 0.02)
        liquidity_score = market_data.get('liquidity_score', 50)
        
        if participation_rate > 0.2:
            # Large order - use VWAP or IS
            if volatility > 0.03:
                return ExecutionStrategy.IS  # Minimize risk
            else:
                return ExecutionStrategy.VWAP
        elif participation_rate > 0.1:
            # Medium order
            if liquidity_score < 40:
                return ExecutionStrategy.ADAPTIVE  # Adapt to liquidity
            else:
                return ExecutionStrategy.TWAP
        else:
            # Small order
            if expected_slippage < 5:
                return ExecutionStrategy.AGGRESSIVE  # Quick execution
            else:
                return ExecutionStrategy.POV  # Percentage of volume
    
    def calibrate_all_models(
        self,
        historical_trades: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calibrate all models in the ensemble."""
        calibration_results = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Calibrating {model_type.value} model")
            try:
                results = model.calibrate(historical_trades, market_data)
                calibration_results[model_type.value] = results
            except Exception as e:
                logger.error(f"Failed to calibrate {model_type.value}: {e}")
                calibration_results[model_type.value] = {'error': str(e)}
        
        return calibration_results
    
    def update_weights(self, performance_metrics: Dict[SlippageModel, float]):
        """Update model weights based on performance."""
        # Normalize performance metrics (higher is better)
        total_performance = sum(performance_metrics.values())
        
        if total_performance > 0:
            for model_type, performance in performance_metrics.items():
                self.weights[model_type] = performance / total_performance
        
        logger.info(f"Updated model weights: {self.weights}")