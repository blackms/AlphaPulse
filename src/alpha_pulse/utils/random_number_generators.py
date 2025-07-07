"""
Random number generators for Monte Carlo simulation.

Provides various random number generation methods including pseudo-random
(Mersenne Twister) and quasi-random (Sobol, Halton) sequences.
"""

import numpy as np
from typing import Tuple, Optional, List, Union, Any, Dict
from abc import ABC, abstractmethod
import logging
from scipy import stats
from scipy.stats import qmc

logger = logging.getLogger(__name__)


class RandomNumberGenerator(ABC):
    """Abstract base class for random number generators."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random number generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.counter = 0
        
    @abstractmethod
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate uniform random numbers in [0, 1)."""
        pass
    
    def generate_standard_normal(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate standard normal random numbers."""
        uniforms = self.generate_uniform(shape)
        return self._uniform_to_normal(uniforms)
    
    def _uniform_to_normal(self, uniforms: np.ndarray) -> np.ndarray:
        """Convert uniform to normal using inverse CDF."""
        # Use inverse normal CDF (more stable than Box-Muller for quasi-random)
        return stats.norm.ppf(uniforms)
    
    def reset(self):
        """Reset generator to initial state."""
        self.counter = 0
    
    def skip(self, n: int):
        """Skip n random numbers."""
        self.counter += n


class MersenneTwisterGenerator(RandomNumberGenerator):
    """Mersenne Twister pseudo-random number generator."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize Mersenne Twister generator."""
        super().__init__(seed)
        self.rng = np.random.RandomState(seed)
        
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate uniform random numbers."""
        self.counter += np.prod(shape)
        return self.rng.rand(*self._normalize_shape(shape))
    
    def generate_standard_normal(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate standard normal random numbers directly."""
        self.counter += np.prod(shape)
        return self.rng.randn(*self._normalize_shape(shape))
    
    def _normalize_shape(self, shape: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """Normalize shape to tuple."""
        if isinstance(shape, int):
            return (shape,)
        return shape
    
    def set_state(self, state: Tuple[str, np.ndarray, int, int, float]):
        """Set internal state of generator."""
        self.rng.set_state(state)
    
    def get_state(self) -> Tuple[str, np.ndarray, int, int, float]:
        """Get internal state of generator."""
        return self.rng.get_state()


class SobolGenerator(RandomNumberGenerator):
    """Sobol quasi-random sequence generator."""
    
    def __init__(
        self,
        dimension: int,
        seed: Optional[int] = None,
        scramble: bool = True
    ):
        """
        Initialize Sobol sequence generator.
        
        Args:
            dimension: Number of dimensions
            seed: Random seed for scrambling
            scramble: Whether to use scrambled Sobol
        """
        super().__init__(seed)
        self.dimension = dimension
        self.scramble = scramble
        
        # Initialize Sobol engine
        self.engine = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
        
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate Sobol sequence."""
        if isinstance(shape, int):
            n_samples = shape
            output_shape = (shape, self.dimension)
        else:
            n_samples = shape[0]
            if len(shape) > 1 and shape[-1] != self.dimension:
                raise ValueError(f"Last dimension must be {self.dimension}")
            output_shape = shape
        
        # Generate Sobol points
        points = self.engine.random(n_samples)
        self.counter += n_samples
        
        # Reshape if needed
        if len(output_shape) > 2:
            points = points.reshape(output_shape)
        
        return points
    
    def reset(self):
        """Reset Sobol generator."""
        super().reset()
        self.engine.reset()
    
    def fast_forward(self, n: int):
        """Fast forward the sequence by n points."""
        self.engine.fast_forward(n)
        self.counter += n


class HaltonGenerator(RandomNumberGenerator):
    """Halton quasi-random sequence generator."""
    
    def __init__(
        self,
        dimension: int,
        seed: Optional[int] = None,
        scramble: bool = True
    ):
        """
        Initialize Halton sequence generator.
        
        Args:
            dimension: Number of dimensions
            seed: Random seed for scrambling
            scramble: Whether to use scrambled Halton
        """
        super().__init__(seed)
        self.dimension = dimension
        self.scramble = scramble
        
        # Initialize Halton engine
        self.engine = qmc.Halton(d=dimension, scramble=scramble, seed=seed)
        
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate Halton sequence."""
        if isinstance(shape, int):
            n_samples = shape
            output_shape = (shape, self.dimension)
        else:
            n_samples = shape[0]
            if len(shape) > 1 and shape[-1] != self.dimension:
                raise ValueError(f"Last dimension must be {self.dimension}")
            output_shape = shape
        
        # Generate Halton points
        points = self.engine.random(n_samples)
        self.counter += n_samples
        
        # Reshape if needed
        if len(output_shape) > 2:
            points = points.reshape(output_shape)
        
        return points
    
    def reset(self):
        """Reset Halton generator."""
        super().reset()
        self.engine.reset()


class LatinHypercubeGenerator(RandomNumberGenerator):
    """Latin Hypercube sampling generator."""
    
    def __init__(
        self,
        dimension: int,
        seed: Optional[int] = None,
        optimization: Optional[str] = 'random-cd'
    ):
        """
        Initialize Latin Hypercube generator.
        
        Args:
            dimension: Number of dimensions
            seed: Random seed
            optimization: Optimization criterion ('random-cd', 'correlation', etc.)
        """
        super().__init__(seed)
        self.dimension = dimension
        self.optimization = optimization
        self.rng = np.random.RandomState(seed)
        
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate Latin Hypercube sample."""
        if isinstance(shape, int):
            n_samples = shape
        else:
            n_samples = shape[0]
        
        # Generate LHS
        engine = qmc.LatinHypercube(d=self.dimension, seed=self.seed)
        
        if self.optimization:
            # Optimize the sample
            sample = engine.random(n_samples)
            if self.optimization == 'random-cd':
                # Optimize for low correlation
                sample = self._optimize_correlation(sample)
        else:
            sample = engine.random(n_samples)
        
        self.counter += n_samples
        
        return sample
    
    def _optimize_correlation(
        self,
        sample: np.ndarray,
        n_iterations: int = 100
    ) -> np.ndarray:
        """Optimize sample to minimize correlation."""
        best_sample = sample.copy()
        best_score = self._correlation_score(best_sample)
        
        for _ in range(n_iterations):
            # Random column permutations
            new_sample = sample.copy()
            for col in range(self.dimension):
                if self.rng.rand() < 0.5:
                    new_sample[:, col] = self.rng.permutation(new_sample[:, col])
            
            score = self._correlation_score(new_sample)
            if score < best_score:
                best_sample = new_sample
                best_score = score
        
        return best_sample
    
    def _correlation_score(self, sample: np.ndarray) -> float:
        """Calculate correlation score (lower is better)."""
        corr_matrix = np.corrcoef(sample.T)
        # Sum of absolute off-diagonal elements
        score = np.sum(np.abs(corr_matrix)) - self.dimension
        return score


class AntitheticGenerator(RandomNumberGenerator):
    """Wrapper for antithetic variates generation."""
    
    def __init__(
        self,
        base_generator: RandomNumberGenerator,
        paired: bool = True
    ):
        """
        Initialize antithetic generator.
        
        Args:
            base_generator: Underlying random number generator
            paired: If True, generate antithetic pairs consecutively
        """
        super().__init__(base_generator.seed)
        self.base_generator = base_generator
        self.paired = paired
        self.generating_antithetic = False
        
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate uniform with antithetic variates."""
        if self.paired:
            # Generate half from base, half antithetic
            if isinstance(shape, int):
                n_samples = shape
                half_n = n_samples // 2
                
                # Generate base samples
                base_samples = self.base_generator.generate_uniform(half_n)
                
                # Create antithetic samples
                anti_samples = 1 - base_samples
                
                # Combine
                if n_samples % 2 == 0:
                    samples = np.concatenate([base_samples, anti_samples])
                else:
                    # Add one more base sample if odd
                    extra = self.base_generator.generate_uniform(1)
                    samples = np.concatenate([base_samples, anti_samples, extra])
            else:
                # Multi-dimensional case
                shape_list = list(shape)
                shape_list[0] = shape_list[0] // 2
                
                base_samples = self.base_generator.generate_uniform(tuple(shape_list))
                anti_samples = 1 - base_samples
                
                if shape[0] % 2 == 0:
                    samples = np.concatenate([base_samples, anti_samples], axis=0)
                else:
                    shape_list[0] = 1
                    extra = self.base_generator.generate_uniform(tuple(shape_list))
                    samples = np.concatenate([base_samples, anti_samples, extra], axis=0)
        else:
            # Alternate between base and antithetic
            if self.generating_antithetic:
                samples = 1 - self.base_generator.generate_uniform(shape)
            else:
                samples = self.base_generator.generate_uniform(shape)
            
            self.generating_antithetic = not self.generating_antithetic
        
        self.counter += np.prod(shape)
        return samples
    
    def generate_standard_normal(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate standard normal with antithetic variates."""
        if hasattr(self.base_generator, 'generate_standard_normal') and not self.paired:
            # Use direct normal generation if available
            if self.generating_antithetic:
                samples = -self.base_generator.generate_standard_normal(shape)
            else:
                samples = self.base_generator.generate_standard_normal(shape)
            
            self.generating_antithetic = not self.generating_antithetic
            return samples
        else:
            # Use uniform to normal transformation
            return super().generate_standard_normal(shape)


class StratifiedGenerator(RandomNumberGenerator):
    """Stratified sampling generator."""
    
    def __init__(
        self,
        n_strata: int,
        dimension: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize stratified sampling generator.
        
        Args:
            n_strata: Number of strata per dimension
            dimension: Number of dimensions
            seed: Random seed
        """
        super().__init__(seed)
        self.n_strata = n_strata
        self.dimension = dimension
        self.rng = np.random.RandomState(seed)
        
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate stratified uniform samples."""
        if isinstance(shape, int):
            n_samples = shape
            output_shape = (shape,) if self.dimension == 1 else (shape, self.dimension)
        else:
            n_samples = shape[0]
            output_shape = shape
        
        samples_per_stratum = n_samples // (self.n_strata ** self.dimension)
        remaining = n_samples - samples_per_stratum * (self.n_strata ** self.dimension)
        
        all_samples = []
        
        if self.dimension == 1:
            # 1D stratification
            for i in range(self.n_strata):
                n_in_stratum = samples_per_stratum
                if i < remaining:
                    n_in_stratum += 1
                
                # Generate uniform in stratum
                stratum_samples = self.rng.uniform(
                    i / self.n_strata,
                    (i + 1) / self.n_strata,
                    n_in_stratum
                )
                all_samples.append(stratum_samples)
            
            samples = np.concatenate(all_samples)
        else:
            # Multi-dimensional stratification
            strata_indices = np.indices(tuple([self.n_strata] * self.dimension))
            strata_indices = strata_indices.reshape(self.dimension, -1).T
            
            for stratum_idx in strata_indices:
                n_in_stratum = samples_per_stratum
                
                # Generate samples in this stratum
                stratum_samples = np.zeros((n_in_stratum, self.dimension))
                
                for d in range(self.dimension):
                    stratum_samples[:, d] = self.rng.uniform(
                        stratum_idx[d] / self.n_strata,
                        (stratum_idx[d] + 1) / self.n_strata,
                        n_in_stratum
                    )
                
                all_samples.append(stratum_samples)
            
            # Add remaining samples randomly
            if remaining > 0:
                extra_samples = self.rng.uniform(0, 1, (remaining, self.dimension))
                all_samples.append(extra_samples)
            
            samples = np.vstack(all_samples)
        
        # Shuffle to remove ordering
        shuffle_idx = self.rng.permutation(n_samples)
        samples = samples[shuffle_idx]
        
        self.counter += n_samples
        
        if samples.shape != output_shape:
            samples = samples.reshape(output_shape)
        
        return samples


class MomentMatchingGenerator(RandomNumberGenerator):
    """Generator with moment matching for variance reduction."""
    
    def __init__(
        self,
        base_generator: RandomNumberGenerator,
        match_moments: List[int] = [1, 2]
    ):
        """
        Initialize moment matching generator.
        
        Args:
            base_generator: Underlying generator
            match_moments: Which moments to match (1=mean, 2=variance, etc.)
        """
        super().__init__(base_generator.seed)
        self.base_generator = base_generator
        self.match_moments = match_moments
        
    def generate_standard_normal(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate standard normal with moment matching."""
        # Generate base samples
        samples = self.base_generator.generate_standard_normal(shape)
        
        # Match moments
        if 1 in self.match_moments:
            # Match mean
            samples = samples - np.mean(samples, axis=0)
        
        if 2 in self.match_moments:
            # Match variance
            current_std = np.std(samples, axis=0)
            samples = samples / current_std
        
        if 3 in self.match_moments:
            # Match skewness (simplified)
            # This is a basic adjustment, more sophisticated methods exist
            skew = stats.skew(samples, axis=0)
            samples = samples - skew * samples**2 / 6
        
        if 4 in self.match_moments:
            # Match kurtosis (simplified)
            kurt = stats.kurtosis(samples, axis=0)
            target_kurt = 0  # Excess kurtosis for normal
            samples = samples * np.sqrt((target_kurt + 3) / (kurt + 3))
        
        self.counter += np.prod(shape)
        return samples
    
    def generate_uniform(
        self,
        shape: Union[int, Tuple[int, ...]]
    ) -> np.ndarray:
        """Generate uniform (no moment matching applied)."""
        return self.base_generator.generate_uniform(shape)


def create_generator(
    generator_type: str,
    dimension: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> RandomNumberGenerator:
    """
    Factory function to create random number generators.
    
    Args:
        generator_type: Type of generator ('mersenne', 'sobol', 'halton', etc.)
        dimension: Number of dimensions for quasi-random
        seed: Random seed
        **kwargs: Additional arguments for specific generators
        
    Returns:
        Random number generator instance
    """
    generator_type = generator_type.lower()
    
    if generator_type in ['mersenne', 'mersenne_twister', 'mt19937']:
        return MersenneTwisterGenerator(seed)
    
    elif generator_type == 'sobol':
        scramble = kwargs.get('scramble', True)
        return SobolGenerator(dimension, seed, scramble)
    
    elif generator_type == 'halton':
        scramble = kwargs.get('scramble', True)
        return HaltonGenerator(dimension, seed, scramble)
    
    elif generator_type in ['lhs', 'latin_hypercube']:
        optimization = kwargs.get('optimization', 'random-cd')
        return LatinHypercubeGenerator(dimension, seed, optimization)
    
    elif generator_type == 'stratified':
        n_strata = kwargs.get('n_strata', 10)
        return StratifiedGenerator(n_strata, dimension, seed)
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def test_generator_quality(
    generator: RandomNumberGenerator,
    n_samples: int = 10000,
    dimension: int = 1
) -> Dict[str, Any]:
    """
    Test quality of random number generator.
    
    Args:
        generator: Random number generator to test
        n_samples: Number of samples to generate
        dimension: Dimension of samples
        
    Returns:
        Dictionary of test results
    """
    # Generate samples
    if dimension == 1:
        uniform_samples = generator.generate_uniform(n_samples)
        normal_samples = generator.generate_standard_normal(n_samples)
    else:
        uniform_samples = generator.generate_uniform((n_samples, dimension))
        normal_samples = generator.generate_standard_normal((n_samples, dimension))
    
    results = {}
    
    # Uniformity tests
    if dimension == 1:
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(uniform_samples, 'uniform')
        results['uniform_ks_test'] = {'statistic': ks_stat, 'p_value': ks_pval}
        
        # Chi-square test
        hist, _ = np.histogram(uniform_samples, bins=20)
        expected = n_samples / 20
        chi2_stat = np.sum((hist - expected)**2 / expected)
        chi2_pval = 1 - stats.chi2.cdf(chi2_stat, df=19)
        results['uniform_chi2_test'] = {'statistic': chi2_stat, 'p_value': chi2_pval}
    
    # Normality tests
    if dimension == 1:
        # Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(normal_samples)
        results['normal_jb_test'] = {'statistic': jb_stat, 'p_value': jb_pval}
        
        # Anderson-Darling test
        ad_result = stats.anderson(normal_samples, dist='norm')
        results['normal_ad_test'] = {
            'statistic': ad_result.statistic,
            'critical_values': ad_result.critical_values.tolist(),
            'significance_levels': ad_result.significance_level.tolist()
        }
    
    # Correlation tests for multi-dimensional
    if dimension > 1:
        # Check correlation matrix
        if uniform_samples.ndim == 2:
            corr_matrix = np.corrcoef(uniform_samples.T)
            max_corr = np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            results['max_correlation'] = max_corr
            results['correlation_test_passed'] = max_corr < 0.1
    
    # Discrepancy for quasi-random
    if hasattr(generator, 'engine') and hasattr(generator.engine, 'discrepancy'):
        results['discrepancy'] = generator.engine.discrepancy(uniform_samples)
    
    return results