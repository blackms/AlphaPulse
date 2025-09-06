"""
Market data simulator for empirical validation.

Generates synthetic market data with controllable regimes and scenarios
to test the hierarchical RL trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"


@dataclass
class RegimeParameters:
    """Parameters for each market regime"""
    drift: float  # Annual drift rate
    volatility: float  # Annual volatility
    mean_reversion: float  # Mean reversion strength
    jump_intensity: float  # Jump frequency per year
    jump_size_mean: float  # Mean jump size
    jump_size_std: float  # Jump size standard deviation


class MarketDataSimulator:
    """
    Synthetic market data generator with regime switching.
    
    Features:
    - Multiple market regimes (bull, bear, sideways, high/low vol)
    - Jump diffusion process
    - Microstructure noise
    - Volume simulation
    - News event simulation
    """
    
    def __init__(self, 
                 initial_price: float = 50000.0,
                 dt: float = 1/365/24,  # Hourly data
                 random_seed: Optional[int] = None):
        """
        Initialize market simulator.
        
        Args:
            initial_price: Starting price
            dt: Time step (fraction of year)
            random_seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.dt = dt
        if random_seed:
            np.random.seed(random_seed)
        
        # Define regime parameters
        self.regime_params = {
            MarketRegime.BULL: RegimeParameters(
                drift=0.15, volatility=0.4, mean_reversion=0.1,
                jump_intensity=10, jump_size_mean=0.02, jump_size_std=0.01
            ),
            MarketRegime.BEAR: RegimeParameters(
                drift=-0.20, volatility=0.6, mean_reversion=0.15,
                jump_intensity=15, jump_size_mean=-0.03, jump_size_std=0.015
            ),
            MarketRegime.SIDEWAYS: RegimeParameters(
                drift=0.0, volatility=0.3, mean_reversion=0.3,
                jump_intensity=5, jump_size_mean=0.0, jump_size_std=0.008
            ),
            MarketRegime.HIGH_VOLATILITY: RegimeParameters(
                drift=0.05, volatility=0.8, mean_reversion=0.05,
                jump_intensity=25, jump_size_mean=0.0, jump_size_std=0.025
            ),
            MarketRegime.LOW_VOLATILITY: RegimeParameters(
                drift=0.08, volatility=0.2, mean_reversion=0.2,
                jump_intensity=2, jump_size_mean=0.01, jump_size_std=0.005
            )
        }
        
        # Regime transition matrix (rows: from, cols: to)
        self.transition_matrix = np.array([
            [0.85, 0.05, 0.05, 0.03, 0.02],  # BULL
            [0.03, 0.85, 0.07, 0.03, 0.02],  # BEAR  
            [0.10, 0.10, 0.70, 0.05, 0.05],  # SIDEWAYS
            [0.05, 0.10, 0.10, 0.70, 0.05],  # HIGH_VOL
            [0.15, 0.05, 0.15, 0.05, 0.60]   # LOW_VOL
        ])
        
        self.regime_names = list(MarketRegime)
        
    def simulate_regime_path(self, n_steps: int, 
                           initial_regime: MarketRegime = MarketRegime.SIDEWAYS) -> List[MarketRegime]:
        """
        Simulate regime switching path using Markov chain.
        
        Args:
            n_steps: Number of time steps
            initial_regime: Starting regime
            
        Returns:
            List of regimes for each time step
        """
        regimes = [initial_regime]
        current_idx = self.regime_names.index(initial_regime)
        
        for _ in range(n_steps - 1):
            # Sample next regime based on transition probabilities
            next_idx = np.random.choice(
                len(self.regime_names),
                p=self.transition_matrix[current_idx]
            )
            regimes.append(self.regime_names[next_idx])
            current_idx = next_idx
            
        return regimes
    
    def simulate_price_path(self, n_steps: int, 
                          regimes: Optional[List[MarketRegime]] = None) -> pd.DataFrame:
        """
        Simulate price path using jump diffusion with regime switching.
        
        Args:
            n_steps: Number of time steps to simulate
            regimes: Regime sequence (if None, will simulate)
            
        Returns:
            DataFrame with OHLCV data
        """
        if regimes is None:
            regimes = self.simulate_regime_path(n_steps)
        
        # Initialize arrays
        timestamps = pd.date_range(start='2024-01-01', periods=n_steps, freq='H')
        prices = np.zeros(n_steps)
        volumes = np.zeros(n_steps)
        
        prices[0] = self.initial_price
        log_price = np.log(prices[0])
        
        for i in range(1, n_steps):
            regime = regimes[i]
            params = self.regime_params[regime]
            
            # Jump component
            if np.random.poisson(params.jump_intensity * self.dt) > 0:
                jump = np.random.normal(params.jump_size_mean, params.jump_size_std)
                log_price += jump
            
            # Diffusion component with mean reversion
            drift = params.drift - params.mean_reversion * (log_price - np.log(self.initial_price))
            diffusion = params.volatility * np.random.normal(0, np.sqrt(self.dt))
            
            log_price += drift * self.dt + diffusion
            prices[i] = np.exp(log_price)
            
            # Simulate volume (correlated with volatility and price changes)
            price_change = abs(np.log(prices[i] / prices[i-1]))
            base_volume = 1000000 * (1 + params.volatility)
            volume_noise = np.random.lognormal(0, 0.5)
            volumes[i] = base_volume * (1 + 10 * price_change) * volume_noise
        
        # Create OHLCV data (simplified: using price as all OHLC)
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.normal(0, 0.001, n_steps)),
            'high': prices * (1 + abs(np.random.normal(0, 0.002, n_steps))),
            'low': prices * (1 - abs(np.random.normal(0, 0.002, n_steps))),
            'close': prices,
            'volume': volumes,
            'regime': regimes
        })
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    def add_microstructure_noise(self, data: pd.DataFrame, 
                               bid_ask_spread: float = 0.001) -> pd.DataFrame:
        """
        Add bid-ask spread and microstructure noise to prices.
        
        Args:
            data: OHLCV DataFrame
            bid_ask_spread: Relative bid-ask spread
            
        Returns:
            DataFrame with bid/ask prices
        """
        spread = data['close'] * bid_ask_spread
        data['bid'] = data['close'] - spread / 2
        data['ask'] = data['close'] + spread / 2
        
        # Add tick-by-tick noise
        tick_noise = np.random.normal(0, spread * 0.1, len(data))
        data['mid_price'] = (data['bid'] + data['ask']) / 2 + tick_noise
        
        return data
    
    def simulate_news_events(self, data: pd.DataFrame, 
                           event_frequency: float = 0.1) -> pd.DataFrame:
        """
        Simulate news events and their market impact.
        
        Args:
            data: Market data DataFrame
            event_frequency: Events per time step
            
        Returns:
            DataFrame with news events
        """
        n_steps = len(data)
        
        # Generate events
        events = np.random.poisson(event_frequency, n_steps)
        event_impacts = np.zeros(n_steps)
        event_types = [''] * n_steps
        
        for i in range(n_steps):
            if events[i] > 0:
                # Event impact on price
                impact_size = np.random.normal(0, 0.02)  # ±2% average impact
                event_impacts[i] = impact_size
                
                # Event type based on impact
                if impact_size > 0.01:
                    event_types[i] = 'positive_news'
                elif impact_size < -0.01:
                    event_types[i] = 'negative_news'
                else:
                    event_types[i] = 'neutral_news'
        
        data['news_impact'] = event_impacts
        data['news_type'] = event_types
        
        return data
    
    def generate_dataset(self, n_days: int = 30, 
                        include_microstructure: bool = True,
                        include_news: bool = True) -> pd.DataFrame:
        """
        Generate complete synthetic dataset for backtesting.
        
        Args:
            n_days: Number of days to simulate
            include_microstructure: Add bid-ask spreads and noise
            include_news: Add news events
            
        Returns:
            Complete market dataset
        """
        n_steps = n_days * 24  # Hourly data
        
        logger.info(f"Generating {n_days} days of synthetic market data...")
        
        # Simulate price path
        data = self.simulate_price_path(n_steps)
        
        # Add microstructure
        if include_microstructure:
            data = self.add_microstructure_noise(data)
        
        # Add news events
        if include_news:
            data = self.simulate_news_events(data)
        
        # Calculate technical indicators
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(24).std()  # 24h rolling vol
        
        # Remove first row (NaN values)
        data = data.dropna().reset_index(drop=True)
        
        logger.info(f"Generated {len(data)} data points with regimes: {data['regime'].value_counts().to_dict()}")
        
        return data


if __name__ == "__main__":
    # Example usage
    simulator = MarketDataSimulator(random_seed=42)
    data = simulator.generate_dataset(n_days=7)
    
    print("Sample data:")
    print(data.head())
    print(f"\nRegime distribution:")
    print(data['regime'].value_counts())