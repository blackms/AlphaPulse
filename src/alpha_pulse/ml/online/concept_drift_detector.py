"""Concept drift detection algorithms for online learning."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy import stats
from collections import deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DriftPoint:
    """Information about detected drift."""
    timestamp: datetime
    sample_index: int
    drift_level: float
    drift_type: str  # 'sudden', 'gradual', 'incremental', 'recurring'
    confidence: float
    metadata: Dict[str, Any]


class BaseDriftDetector(ABC):
    """Base class for drift detection algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_samples = 0
        self.drift_detected = False
        self.warning_detected = False
        self.drift_history: List[DriftPoint] = []
        
    @abstractmethod
    def add_element(self, error: float) -> None:
        """Add new error value for drift detection."""
        pass
        
    @abstractmethod
    def detected_change(self) -> bool:
        """Check if drift has been detected."""
        pass
        
    @abstractmethod
    def detected_warning(self) -> bool:
        """Check if warning level has been reached."""
        pass
        
    def reset(self) -> None:
        """Reset detector after drift handling."""
        self.n_samples = 0
        self.drift_detected = False
        self.warning_detected = False
        
    def get_drift_info(self) -> Optional[DriftPoint]:
        """Get information about last detected drift."""
        return self.drift_history[-1] if self.drift_history else None


class ADWIN(BaseDriftDetector):
    """Adaptive Windowing method for concept drift detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.delta = config.get('delta', 0.002)  # Confidence parameter
        self.max_buckets = config.get('max_buckets', 5)
        
        self.width = 0
        self.total = 0.0
        self.variance = 0.0
        self.bucket_list = []
        
    def add_element(self, value: float) -> None:
        """Add element and check for drift."""
        self.n_samples += 1
        self._insert_element(value)
        
        if self._check_drift():
            self.drift_detected = True
            self._record_drift('sudden')
            
    def _insert_element(self, value: float) -> None:
        """Insert element into ADWIN structure."""
        self.width += 1
        self.total += value
        
        # Update variance incrementally
        if self.width > 1:
            self.variance += (self.width - 1) * (value - self.total / self.width) ** 2 / self.width
            
        # Add to bucket structure
        if not self.bucket_list:
            self.bucket_list.append(_Bucket())
            
        self.bucket_list[0].add(value)
        self._compress_buckets()
        
    def _check_drift(self) -> bool:
        """Check if there is a drift between two windows."""
        if self.width < 2:
            return False
            
        for i in range(len(self.bucket_list) - 1):
            # Check all possible split points
            n0 = sum(b.count for b in self.bucket_list[:i+1])
            n1 = self.width - n0
            
            if n0 < 5 or n1 < 5:  # Minimum window size
                continue
                
            sum0 = sum(b.total for b in self.bucket_list[:i+1])
            sum1 = self.total - sum0
            
            mean0 = sum0 / n0
            mean1 = sum1 / n1
            
            # ADWIN bound
            epsilon_cut = self._calculate_epsilon_cut(n0, n1)
            
            if abs(mean0 - mean1) > epsilon_cut:
                # Remove old buckets
                self.bucket_list = self.bucket_list[i+1:]
                self.width = n1
                self.total = sum1
                return True
                
        return False
        
    def _calculate_epsilon_cut(self, n0: int, n1: int) -> float:
        """Calculate ADWIN cut threshold."""
        n = n0 + n1
        delta_prime = self.delta / n
        
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        epsilon = np.sqrt(2.0 * m * np.log(2.0 / delta_prime) / n)
        
        return epsilon
        
    def _compress_buckets(self) -> None:
        """Compress buckets to maintain memory efficiency."""
        # Implement exponential histogram compression
        cursor = 0
        
        while cursor < len(self.bucket_list) - 1:
            bucket1 = self.bucket_list[cursor]
            bucket2 = self.bucket_list[cursor + 1]
            
            if bucket1.count == bucket2.count:
                # Merge buckets
                bucket2.total += bucket1.total
                bucket2.count += bucket1.count
                self.bucket_list.pop(cursor)
            else:
                cursor += 1
                
    def detected_change(self) -> bool:
        """Check if drift has been detected."""
        return self.drift_detected
        
    def detected_warning(self) -> bool:
        """ADWIN doesn't have explicit warning level."""
        return False
        
    def _record_drift(self, drift_type: str) -> None:
        """Record drift detection."""
        drift_point = DriftPoint(
            timestamp=datetime.now(),
            sample_index=self.n_samples,
            drift_level=1.0,  # Binary detection
            drift_type=drift_type,
            confidence=1.0 - self.delta,
            metadata={'width': self.width}
        )
        self.drift_history.append(drift_point)


class _Bucket:
    """Bucket for ADWIN algorithm."""
    
    def __init__(self):
        self.count = 0
        self.total = 0.0
        
    def add(self, value: float) -> None:
        """Add value to bucket."""
        self.count += 1
        self.total += value


class DDM(BaseDriftDetector):
    """Drift Detection Method based on error rate monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.warning_level = config.get('warning_level', 2.0)
        self.drift_level = config.get('drift_level', 3.0)
        self.min_samples = config.get('min_samples', 30)
        
        self.error_rate = 0.0
        self.std_dev = 0.0
        self.m_p = 1.0
        self.s_p = 0.0
        self.n_errors = 0
        
    def add_element(self, error: float) -> None:
        """Add prediction error (0 or 1)."""
        self.n_samples += 1
        
        # Update error rate
        self.n_errors += error
        self.error_rate = self.n_errors / self.n_samples
        
        # Update standard deviation
        self.std_dev = np.sqrt(self.error_rate * (1 - self.error_rate) / self.n_samples)
        
        if self.n_samples < self.min_samples:
            return
            
        # Update minimum values
        if self.error_rate + self.std_dev < self.m_p + self.s_p:
            self.m_p = self.error_rate
            self.s_p = self.std_dev
            
        # Check for drift
        if self.error_rate + self.std_dev >= self.m_p + self.drift_level * self.s_p:
            self.drift_detected = True
            self._record_drift('sudden')
        elif self.error_rate + self.std_dev >= self.m_p + self.warning_level * self.s_p:
            self.warning_detected = True
            
    def detected_change(self) -> bool:
        """Check if drift has been detected."""
        return self.drift_detected
        
    def detected_warning(self) -> bool:
        """Check if warning has been detected."""
        return self.warning_detected
        
    def _record_drift(self, drift_type: str) -> None:
        """Record drift detection."""
        drift_level = (self.error_rate + self.std_dev - self.m_p) / self.s_p if self.s_p > 0 else 0
        
        drift_point = DriftPoint(
            timestamp=datetime.now(),
            sample_index=self.n_samples,
            drift_level=drift_level,
            drift_type=drift_type,
            confidence=0.99,  # Based on 3-sigma rule
            metadata={
                'error_rate': self.error_rate,
                'min_error_rate': self.m_p
            }
        )
        self.drift_history.append(drift_point)


class PageHinkley(BaseDriftDetector):
    """Page-Hinkley test for drift detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.delta = config.get('delta', 0.005)
        self.threshold = config.get('threshold', 50.0)
        self.alpha = config.get('alpha', 0.999)  # Forgetting factor
        
        self.sum = 0.0
        self.x_mean = 0.0
        self.m_p = float('inf')
        self.m_n = float('-inf')
        
    def add_element(self, value: float) -> None:
        """Add new value for monitoring."""
        self.n_samples += 1
        
        # Update mean with forgetting factor
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * value
        
        # Update cumulative sum
        self.sum += value - self.x_mean - self.delta
        
        # Update min/max
        self.m_p = min(self.m_p, self.sum)
        self.m_n = max(self.m_n, self.sum)
        
        # Check for positive drift
        if self.sum - self.m_p > self.threshold:
            self.drift_detected = True
            self._record_drift('incremental', 'positive')
            self.reset_after_drift()
            
        # Check for negative drift  
        if self.m_n - self.sum > self.threshold:
            self.drift_detected = True
            self._record_drift('incremental', 'negative')
            self.reset_after_drift()
            
    def reset_after_drift(self) -> None:
        """Reset statistics after drift detection."""
        self.sum = 0.0
        self.m_p = float('inf')
        self.m_n = float('-inf')
        
    def detected_change(self) -> bool:
        """Check if drift has been detected."""
        return self.drift_detected
        
    def detected_warning(self) -> bool:
        """Page-Hinkley doesn't have explicit warning."""
        return False
        
    def _record_drift(self, drift_type: str, direction: str) -> None:
        """Record drift detection."""
        drift_level = max(self.sum - self.m_p, self.m_n - self.sum)
        
        drift_point = DriftPoint(
            timestamp=datetime.now(),
            sample_index=self.n_samples,
            drift_level=drift_level,
            drift_type=drift_type,
            confidence=0.95,
            metadata={
                'direction': direction,
                'threshold': self.threshold
            }
        )
        self.drift_history.append(drift_point)


class KSWIN(BaseDriftDetector):
    """Kolmogorov-Smirnov Windowing method for drift detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get('window_size', 100)
        self.alpha = config.get('alpha', 0.05)  # Significance level
        self.stat_threshold = config.get('stat_threshold', 0.3)
        
        self.window = deque(maxlen=self.window_size)
        self.reference_window = None
        
    def add_element(self, value: float) -> None:
        """Add element and check for drift."""
        self.n_samples += 1
        self.window.append(value)
        
        # Need full window to start
        if len(self.window) < self.window_size:
            return
            
        # Initialize reference window
        if self.reference_window is None:
            self.reference_window = list(self.window)
            return
            
        # Perform KS test
        if self._check_drift():
            self.drift_detected = True
            self._record_drift('distributional')
            # Update reference window
            self.reference_window = list(self.window)
            
    def _check_drift(self) -> bool:
        """Check for distributional drift using KS test."""
        try:
            statistic, p_value = stats.ks_2samp(
                self.reference_window,
                list(self.window)
            )
            
            # Check both p-value and statistic magnitude
            if p_value < self.alpha and statistic > self.stat_threshold:
                return True
                
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            
        return False
        
    def detected_change(self) -> bool:
        """Check if drift has been detected."""
        return self.drift_detected
        
    def detected_warning(self) -> bool:
        """Check for warning level."""
        if not self.reference_window or len(self.window) < self.window_size:
            return False
            
        try:
            statistic, p_value = stats.ks_2samp(
                self.reference_window,
                list(self.window)
            )
            
            # Warning if p-value is low but statistic is moderate
            if p_value < self.alpha * 2 and statistic > self.stat_threshold * 0.7:
                return True
                
        except:
            pass
            
        return False
        
    def _record_drift(self, drift_type: str) -> None:
        """Record drift detection."""
        statistic, p_value = stats.ks_2samp(
            self.reference_window,
            list(self.window)
        )
        
        drift_point = DriftPoint(
            timestamp=datetime.now(),
            sample_index=self.n_samples,
            drift_level=statistic,
            drift_type=drift_type,
            confidence=1.0 - p_value,
            metadata={
                'ks_statistic': statistic,
                'p_value': p_value
            }
        )
        self.drift_history.append(drift_point)


class ConceptDriftDetector:
    """Main interface for concept drift detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get('method', 'adwin')
        self.check_frequency = config.get('check_frequency', 100)
        
        # Initialize detector
        if self.method == 'adwin':
            self.detector = ADWIN(config)
        elif self.method == 'ddm':
            self.detector = DDM(config)
        elif self.method == 'page_hinkley':
            self.detector = PageHinkley(config)
        elif self.method == 'kswin':
            self.detector = KSWIN(config)
        else:
            raise ValueError(f"Unknown drift detection method: {self.method}")
            
        # Multi-detector ensemble
        self.ensemble_mode = config.get('ensemble_mode', False)
        self.ensemble_detectors = {}
        
        if self.ensemble_mode:
            self._initialize_ensemble()
            
    def _initialize_ensemble(self) -> None:
        """Initialize ensemble of detectors."""
        methods = ['adwin', 'ddm', 'page_hinkley', 'kswin']
        
        for method in methods:
            if method != self.method:
                config_copy = self.config.copy()
                config_copy['method'] = method
                
                if method == 'adwin':
                    self.ensemble_detectors[method] = ADWIN(config_copy)
                elif method == 'ddm':
                    self.ensemble_detectors[method] = DDM(config_copy)
                elif method == 'page_hinkley':
                    self.ensemble_detectors[method] = PageHinkley(config_copy)
                elif method == 'kswin':
                    self.ensemble_detectors[method] = KSWIN(config_copy)
                    
    def add_element(self, error: float) -> None:
        """Add error value to detector(s)."""
        self.detector.add_element(error)
        
        if self.ensemble_mode:
            for detector in self.ensemble_detectors.values():
                detector.add_element(error)
                
    def detected_change(self) -> bool:
        """Check if drift has been detected."""
        if self.ensemble_mode:
            # Majority voting
            votes = [self.detector.detected_change()]
            votes.extend([d.detected_change() for d in self.ensemble_detectors.values()])
            
            return sum(votes) > len(votes) / 2
        else:
            return self.detector.detected_change()
            
    def detected_warning(self) -> bool:
        """Check if warning has been detected."""
        if self.ensemble_mode:
            # Any detector warning
            warnings = [self.detector.detected_warning()]
            warnings.extend([d.detected_warning() for d in self.ensemble_detectors.values()])
            
            return any(warnings)
        else:
            return self.detector.detected_warning()
            
    def get_drift_info(self) -> Dict[str, Any]:
        """Get comprehensive drift information."""
        info = {
            'method': self.method,
            'drift_detected': self.detected_change(),
            'warning_detected': self.detected_warning(),
            'n_samples': self.detector.n_samples
        }
        
        # Add last drift info
        last_drift = self.detector.get_drift_info()
        if last_drift:
            info['last_drift'] = {
                'timestamp': last_drift.timestamp.isoformat(),
                'sample_index': last_drift.sample_index,
                'drift_level': last_drift.drift_level,
                'drift_type': last_drift.drift_type,
                'confidence': last_drift.confidence
            }
            
        # Add ensemble info
        if self.ensemble_mode:
            ensemble_status = {}
            for method, detector in self.ensemble_detectors.items():
                ensemble_status[method] = {
                    'drift': detector.detected_change(),
                    'warning': detector.detected_warning()
                }
            info['ensemble_status'] = ensemble_status
            
        return info
        
    def reset(self) -> None:
        """Reset all detectors."""
        self.detector.reset()
        
        if self.ensemble_mode:
            for detector in self.ensemble_detectors.values():
                detector.reset()