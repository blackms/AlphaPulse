"""
Data migration utilities for transitioning from mock to real data providers.

Provides:
- Gradual migration from mock to real data
- Data validation and comparison tools
- Performance impact assessment
- Rollback capabilities
- Migration monitoring and reporting
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from loguru import logger

from alpha_pulse.data_pipeline.providers.mock_provider import MockMarketDataProvider
from alpha_pulse.data_pipeline.providers.provider_factory import DataProviderFactory
from alpha_pulse.models.market_data import MarketDataPoint, TimeSeriesData
from alpha_pulse.utils.data_validation import MarketDataValidator, ValidationLevel
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class MigrationPhase(Enum):
    """Migration phase stages."""
    PREPARATION = "preparation"
    PARALLEL_TESTING = "parallel_testing"
    GRADUAL_TRANSITION = "gradual_transition"
    FULL_MIGRATION = "full_migration"
    VALIDATION = "validation"
    COMPLETED = "completed"


class DataSource(Enum):
    """Data source types."""
    MOCK = "mock"
    REAL = "real"
    HYBRID = "hybrid"


@dataclass
class MigrationConfig:
    """Configuration for data migration."""
    target_symbols: List[str]
    test_duration_hours: int = 24
    parallel_test_duration_hours: int = 72
    gradual_transition_duration_hours: int = 168  # 1 week
    performance_threshold_ms: int = 1000
    quality_threshold: float = 0.8
    rollback_enabled: bool = True
    validation_level: ValidationLevel = ValidationLevel.STANDARD


@dataclass
class MigrationMetrics:
    """Metrics for migration monitoring."""
    phase: MigrationPhase
    symbols_migrated: int
    total_symbols: int
    mock_requests: int
    real_requests: int
    errors_mock: int
    errors_real: int
    avg_latency_mock: float
    avg_latency_real: float
    data_quality_mock: float
    data_quality_real: float
    migration_start_time: datetime
    current_time: datetime


@dataclass
class ComparisonResult:
    """Result of comparing mock vs real data."""
    symbol: str
    timestamp: datetime
    mock_data: Optional[MarketDataPoint]
    real_data: Optional[MarketDataPoint]
    price_deviation_percent: float
    volume_deviation_percent: float
    quality_score_mock: float
    quality_score_real: float
    issues: List[str]


class DataMigrationManager:
    """Manager for data source migration."""

    def __init__(
        self,
        config: MigrationConfig,
        provider_factory: DataProviderFactory
    ):
        """
        Initialize data migration manager.

        Args:
            config: Migration configuration
            provider_factory: Real data provider factory
        """
        self.config = config
        self.provider_factory = provider_factory
        
        # Migration state
        self.current_phase = MigrationPhase.PREPARATION
        self.migration_start_time = datetime.utcnow()
        self.migrated_symbols: set = set()
        
        # Data providers
        self.mock_provider = MockMarketDataProvider()
        self.validator = MarketDataValidator(config.validation_level)
        
        # Metrics tracking
        self.metrics = {
            'mock_requests': 0,
            'real_requests': 0,
            'mock_errors': 0,
            'real_errors': 0,
            'mock_latencies': [],
            'real_latencies': [],
            'mock_quality_scores': [],
            'real_quality_scores': [],
            'comparison_results': []
        }
        
        # Migration rules
        self.symbol_routing: Dict[str, DataSource] = {}
        for symbol in config.target_symbols:
            self.symbol_routing[symbol] = DataSource.MOCK
        
        # Audit logging
        self.audit_logger = get_audit_logger()

    async def start_migration(self) -> None:
        """Start the data migration process."""
        logger.info("Starting data migration process")
        
        self.audit_logger.log(
            event_type=AuditEventType.SYSTEM_CHANGE,
            event_data={
                "migration_phase": "start",
                "target_symbols": self.config.target_symbols,
                "config": {
                    "test_duration_hours": self.config.test_duration_hours,
                    "performance_threshold_ms": self.config.performance_threshold_ms,
                    "quality_threshold": self.config.quality_threshold
                }
            },
            severity=AuditSeverity.INFO
        )
        
        try:
            await self._phase_preparation()
            await self._phase_parallel_testing()
            await self._phase_gradual_transition()
            await self._phase_full_migration()
            await self._phase_validation()
            
            self.current_phase = MigrationPhase.COMPLETED
            logger.info("Data migration completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if self.config.rollback_enabled:
                await self._rollback_migration()
            raise

    async def _phase_preparation(self) -> None:
        """Preparation phase - validate setup and connectivity."""
        logger.info("Phase 1: Preparation")
        self.current_phase = MigrationPhase.PREPARATION
        
        # Test provider connectivity
        try:
            health_status = self.provider_factory.get_provider_status()
            if health_status['enabled_providers'] == 0:
                raise RuntimeError("No real data providers available")
            
            # Test sample data retrieval
            test_symbol = self.config.target_symbols[0] if self.config.target_symbols else 'AAPL'
            test_data = await self.provider_factory.get_real_time_quote(test_symbol)
            
            if not test_data:
                raise RuntimeError("Failed to retrieve test data from real providers")
            
            logger.info("Preparation phase completed successfully")
            
        except Exception as e:
            logger.error(f"Preparation phase failed: {e}")
            raise

    async def _phase_parallel_testing(self) -> None:
        """Parallel testing phase - run both sources simultaneously."""
        logger.info("Phase 2: Parallel Testing")
        self.current_phase = MigrationPhase.PARALLEL_TESTING
        
        end_time = datetime.utcnow() + timedelta(hours=self.config.parallel_test_duration_hours)
        comparison_results = []
        
        while datetime.utcnow() < end_time:
            for symbol in self.config.target_symbols:
                try:
                    # Get data from both sources
                    mock_data, real_data = await asyncio.gather(
                        self._get_mock_data(symbol),
                        self._get_real_data(symbol),
                        return_exceptions=True
                    )
                    
                    # Compare results
                    if not isinstance(mock_data, Exception) and not isinstance(real_data, Exception):
                        comparison = await self._compare_data(symbol, mock_data, real_data)
                        comparison_results.append(comparison)
                        self.metrics['comparison_results'].append(comparison)
                    
                except Exception as e:
                    logger.warning(f"Parallel test error for {symbol}: {e}")
            
            # Wait before next comparison
            await asyncio.sleep(60)  # Test every minute
        
        # Analyze parallel test results
        await self._analyze_parallel_results(comparison_results)
        logger.info("Parallel testing phase completed")

    async def _phase_gradual_transition(self) -> None:
        """Gradual transition phase - migrate symbols incrementally."""
        logger.info("Phase 3: Gradual Transition")
        self.current_phase = MigrationPhase.GRADUAL_TRANSITION
        
        total_duration = timedelta(hours=self.config.gradual_transition_duration_hours)
        symbols_per_batch = max(1, len(self.config.target_symbols) // 10)  # 10 batches
        
        for i in range(0, len(self.config.target_symbols), symbols_per_batch):
            batch_symbols = self.config.target_symbols[i:i + symbols_per_batch]
            
            # Migrate batch to real data
            for symbol in batch_symbols:
                await self._migrate_symbol_to_real(symbol)
            
            # Monitor performance for this batch
            await self._monitor_batch_performance(batch_symbols)
            
            # Wait before next batch
            batch_delay = total_duration.total_seconds() / (len(self.config.target_symbols) // symbols_per_batch)
            await asyncio.sleep(batch_delay)
        
        logger.info("Gradual transition phase completed")

    async def _phase_full_migration(self) -> None:
        """Full migration phase - ensure all symbols use real data."""
        logger.info("Phase 4: Full Migration")
        self.current_phase = MigrationPhase.FULL_MIGRATION
        
        # Migrate any remaining symbols
        for symbol in self.config.target_symbols:
            if self.symbol_routing[symbol] != DataSource.REAL:
                await self._migrate_symbol_to_real(symbol)
        
        # Verify all symbols are using real data
        assert all(
            self.symbol_routing[symbol] == DataSource.REAL 
            for symbol in self.config.target_symbols
        ), "Not all symbols migrated to real data"
        
        logger.info("Full migration phase completed")

    async def _phase_validation(self) -> None:
        """Validation phase - final checks and performance validation."""
        logger.info("Phase 5: Validation")
        self.current_phase = MigrationPhase.VALIDATION
        
        # Run validation tests
        validation_duration = timedelta(hours=self.config.test_duration_hours)
        end_time = datetime.utcnow() + validation_duration
        
        validation_results = []
        
        while datetime.utcnow() < end_time:
            for symbol in self.config.target_symbols:
                try:
                    # Get real data and validate
                    start_time = datetime.utcnow()
                    real_data = await self._get_real_data(symbol)
                    latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    # Validate data quality
                    validation_result = await self.validator.validate_data_point(real_data)
                    
                    validation_results.append({
                        'symbol': symbol,
                        'latency_ms': latency_ms,
                        'quality_score': validation_result.quality_score,
                        'is_valid': validation_result.is_valid,
                        'timestamp': datetime.utcnow()
                    })
                    
                except Exception as e:
                    logger.warning(f"Validation error for {symbol}: {e}")
            
            await asyncio.sleep(300)  # Validate every 5 minutes
        
        # Analyze validation results
        await self._analyze_validation_results(validation_results)
        logger.info("Validation phase completed")

    async def _get_mock_data(self, symbol: str) -> MarketDataPoint:
        """Get data from mock provider."""
        start_time = datetime.utcnow()
        
        try:
            # Get mock data (simplified - you'd need to convert format)
            mock_data = await self.mock_provider.get_ticker_price(symbol)
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.metrics['mock_requests'] += 1
            self.metrics['mock_latencies'].append(latency_ms)
            
            # Convert to MarketDataPoint (simplified)
            # You'd need proper conversion logic here
            return None  # Placeholder
            
        except Exception as e:
            self.metrics['mock_errors'] += 1
            raise

    async def _get_real_data(self, symbol: str) -> MarketDataPoint:
        """Get data from real provider."""
        start_time = datetime.utcnow()
        
        try:
            real_data = await self.provider_factory.get_real_time_quote(symbol)
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.metrics['real_requests'] += 1
            self.metrics['real_latencies'].append(latency_ms)
            
            return real_data
            
        except Exception as e:
            self.metrics['real_errors'] += 1
            raise

    async def _compare_data(
        self, 
        symbol: str, 
        mock_data: MarketDataPoint, 
        real_data: MarketDataPoint
    ) -> ComparisonResult:
        """Compare mock and real data for analysis."""
        issues = []
        
        # Calculate price deviation
        price_deviation = 0.0
        if mock_data and real_data and mock_data.ohlcv and real_data.ohlcv:
            mock_price = float(mock_data.ohlcv.close)
            real_price = float(real_data.ohlcv.close)
            price_deviation = abs(mock_price - real_price) / real_price * 100
            
            if price_deviation > 10.0:  # 10% threshold
                issues.append(f"Large price deviation: {price_deviation:.2f}%")
        
        # Calculate volume deviation
        volume_deviation = 0.0
        if mock_data and real_data and mock_data.ohlcv and real_data.ohlcv:
            mock_volume = float(mock_data.ohlcv.volume)
            real_volume = float(real_data.ohlcv.volume)
            if real_volume > 0:
                volume_deviation = abs(mock_volume - real_volume) / real_volume * 100
        
        # Validate both data points
        mock_validation = await self.validator.validate_data_point(mock_data)
        real_validation = await self.validator.validate_data_point(real_data)
        
        return ComparisonResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            mock_data=mock_data,
            real_data=real_data,
            price_deviation_percent=price_deviation,
            volume_deviation_percent=volume_deviation,
            quality_score_mock=mock_validation.quality_score,
            quality_score_real=real_validation.quality_score,
            issues=issues
        )

    async def _analyze_parallel_results(self, results: List[ComparisonResult]) -> None:
        """Analyze results from parallel testing phase."""
        if not results:
            logger.warning("No parallel test results to analyze")
            return
        
        # Calculate metrics
        price_deviations = [r.price_deviation_percent for r in results]
        mock_quality_scores = [r.quality_score_mock for r in results]
        real_quality_scores = [r.quality_score_real for r in results]
        
        analysis = {
            'total_comparisons': len(results),
            'avg_price_deviation': statistics.mean(price_deviations),
            'max_price_deviation': max(price_deviations),
            'avg_mock_quality': statistics.mean(mock_quality_scores),
            'avg_real_quality': statistics.mean(real_quality_scores),
            'quality_improvement': statistics.mean(real_quality_scores) - statistics.mean(mock_quality_scores)
        }
        
        # Log analysis
        self.audit_logger.log(
            event_type=AuditEventType.DATA_ANALYSIS,
            event_data={
                "migration_phase": "parallel_testing_analysis",
                "analysis": analysis
            },
            severity=AuditSeverity.INFO
        )
        
        # Check if real data meets quality thresholds
        if analysis['avg_real_quality'] < self.config.quality_threshold:
            raise RuntimeError(f"Real data quality {analysis['avg_real_quality']:.3f} below threshold {self.config.quality_threshold}")

    async def _migrate_symbol_to_real(self, symbol: str) -> None:
        """Migrate a specific symbol to real data."""
        logger.info(f"Migrating {symbol} to real data")
        
        # Test real data for this symbol
        try:
            test_data = await self._get_real_data(symbol)
            validation_result = await self.validator.validate_data_point(test_data)
            
            if not validation_result.is_valid:
                logger.warning(f"Real data validation failed for {symbol}: {validation_result.errors}")
                if self.config.validation_level == ValidationLevel.CRITICAL:
                    raise RuntimeError(f"Critical validation failure for {symbol}")
            
            # Update routing
            self.symbol_routing[symbol] = DataSource.REAL
            self.migrated_symbols.add(symbol)
            
            self.audit_logger.log(
                event_type=AuditEventType.SYSTEM_CHANGE,
                event_data={
                    "migration_action": "symbol_migrated",
                    "symbol": symbol,
                    "data_quality": validation_result.quality_score
                },
                severity=AuditSeverity.INFO
            )
            
        except Exception as e:
            logger.error(f"Failed to migrate {symbol} to real data: {e}")
            raise

    async def _monitor_batch_performance(self, symbols: List[str]) -> None:
        """Monitor performance after migrating a batch of symbols."""
        logger.info(f"Monitoring performance for batch: {symbols}")
        
        # Monitor for 30 minutes
        end_time = datetime.utcnow() + timedelta(minutes=30)
        latencies = []
        
        while datetime.utcnow() < end_time:
            for symbol in symbols:
                try:
                    start_time = datetime.utcnow()
                    await self._get_real_data(symbol)
                    latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    latencies.append(latency_ms)
                    
                except Exception as e:
                    logger.warning(f"Performance monitoring error for {symbol}: {e}")
            
            await asyncio.sleep(60)  # Check every minute
        
        # Analyze performance
        if latencies:
            avg_latency = statistics.mean(latencies)
            if avg_latency > self.config.performance_threshold_ms:
                logger.warning(f"Average latency {avg_latency:.1f}ms exceeds threshold {self.config.performance_threshold_ms}ms")

    async def _analyze_validation_results(self, results: List[Dict[str, Any]]) -> None:
        """Analyze final validation results."""
        if not results:
            raise RuntimeError("No validation results available")
        
        # Calculate final metrics
        latencies = [r['latency_ms'] for r in results]
        quality_scores = [r['quality_score'] for r in results]
        valid_count = sum(1 for r in results if r['is_valid'])
        
        final_analysis = {
            'total_validations': len(results),
            'valid_percentage': valid_count / len(results) * 100,
            'avg_latency_ms': statistics.mean(latencies),
            'max_latency_ms': max(latencies),
            'avg_quality_score': statistics.mean(quality_scores),
            'min_quality_score': min(quality_scores)
        }
        
        # Check if migration meets requirements
        if final_analysis['avg_latency_ms'] > self.config.performance_threshold_ms:
            raise RuntimeError(f"Average latency {final_analysis['avg_latency_ms']:.1f}ms exceeds threshold")
        
        if final_analysis['avg_quality_score'] < self.config.quality_threshold:
            raise RuntimeError(f"Average quality {final_analysis['avg_quality_score']:.3f} below threshold")
        
        # Log final results
        self.audit_logger.log(
            event_type=AuditEventType.SYSTEM_CHANGE,
            event_data={
                "migration_phase": "validation_complete",
                "final_analysis": final_analysis,
                "migration_duration_hours": (datetime.utcnow() - self.migration_start_time).total_seconds() / 3600
            },
            severity=AuditSeverity.INFO
        )

    async def _rollback_migration(self) -> None:
        """Rollback migration to mock data."""
        logger.warning("Rolling back migration to mock data")
        
        # Reset all symbols to mock
        for symbol in self.config.target_symbols:
            self.symbol_routing[symbol] = DataSource.MOCK
        
        self.migrated_symbols.clear()
        self.current_phase = MigrationPhase.PREPARATION
        
        self.audit_logger.log(
            event_type=AuditEventType.SYSTEM_CHANGE,
            event_data={
                "migration_action": "rollback",
                "reason": "migration_failure"
            },
            severity=AuditSeverity.WARNING
        )

    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        return {
            'current_phase': self.current_phase.value,
            'migration_progress': len(self.migrated_symbols) / len(self.config.target_symbols) * 100,
            'symbols_migrated': len(self.migrated_symbols),
            'total_symbols': len(self.config.target_symbols),
            'migration_duration_hours': (datetime.utcnow() - self.migration_start_time).total_seconds() / 3600,
            'metrics': self._get_current_metrics()
        }

    def _get_current_metrics(self) -> MigrationMetrics:
        """Get current migration metrics."""
        return MigrationMetrics(
            phase=self.current_phase,
            symbols_migrated=len(self.migrated_symbols),
            total_symbols=len(self.config.target_symbols),
            mock_requests=self.metrics['mock_requests'],
            real_requests=self.metrics['real_requests'],
            errors_mock=self.metrics['mock_errors'],
            errors_real=self.metrics['real_errors'],
            avg_latency_mock=statistics.mean(self.metrics['mock_latencies']) if self.metrics['mock_latencies'] else 0,
            avg_latency_real=statistics.mean(self.metrics['real_latencies']) if self.metrics['real_latencies'] else 0,
            data_quality_mock=statistics.mean(self.metrics['mock_quality_scores']) if self.metrics['mock_quality_scores'] else 0,
            data_quality_real=statistics.mean(self.metrics['real_quality_scores']) if self.metrics['real_quality_scores'] else 0,
            migration_start_time=self.migration_start_time,
            current_time=datetime.utcnow()
        )

    async def get_data(self, symbol: str) -> MarketDataPoint:
        """Get data based on current migration routing."""
        data_source = self.symbol_routing.get(symbol, DataSource.MOCK)
        
        if data_source == DataSource.MOCK:
            return await self._get_mock_data(symbol)
        elif data_source == DataSource.REAL:
            return await self._get_real_data(symbol)
        else:
            # Hybrid mode - use real data with mock fallback
            try:
                return await self._get_real_data(symbol)
            except Exception as e:
                logger.warning(f"Real data failed for {symbol}, falling back to mock: {e}")
                return await self._get_mock_data(symbol)