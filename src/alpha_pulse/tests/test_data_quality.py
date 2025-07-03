"""
Test suite for data quality validation pipeline.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from alpha_pulse.models.market_data import MarketDataPoint, OHLCV, AssetClass
from alpha_pulse.data.quality import (
    DataQualityValidator,
    AnomalyDetector,
    QualityMetricsService,
    MarketDataQualityChecks,
    QualityDimension,
    AnomalyMethod
)
from alpha_pulse.pipelines import DataQualityPipeline, PipelineConfig, PipelineMode
from alpha_pulse.config.quality_rules import QualityRulesManager, QualityProfile


@pytest.fixture
def sample_market_data():
    """Create sample market data points for testing."""
    now = datetime.utcnow()
    return [
        MarketDataPoint(
            symbol="AAPL",
            timestamp=now - timedelta(minutes=i),
            ohlcv=OHLCV(
                open=Decimal("150.00") + Decimal(str(i * 0.1)),
                high=Decimal("151.00") + Decimal(str(i * 0.1)),
                low=Decimal("149.00") + Decimal(str(i * 0.1)),
                close=Decimal("150.50") + Decimal(str(i * 0.1)),
                volume=Decimal("1000000") + Decimal(str(i * 10000))
            ),
            metadata={"asset_class": AssetClass.EQUITY.value}
        )
        for i in range(10)
    ]


@pytest.fixture
def anomalous_data_point():
    """Create an anomalous data point for testing."""
    return MarketDataPoint(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        ohlcv=OHLCV(
            open=Decimal("200.00"),  # 33% jump from previous ~150
            high=Decimal("210.00"),
            low=Decimal("195.00"),
            close=Decimal("205.00"),
            volume=Decimal("50000000")  # 50x normal volume
        ),
        metadata={"asset_class": AssetClass.EQUITY.value}
    )


@pytest.fixture
def invalid_data_point():
    """Create an invalid data point for testing."""
    return MarketDataPoint(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        ohlcv=OHLCV(
            open=Decimal("150.00"),
            high=Decimal("140.00"),  # High less than open (invalid)
            low=Decimal("155.00"),   # Low greater than high (invalid)
            close=Decimal("145.00"),
            volume=Decimal("-1000")   # Negative volume (invalid)
        ),
        metadata={"asset_class": AssetClass.EQUITY.value}
    )


class TestDataQualityValidator:
    """Test data quality validator."""
    
    @pytest.mark.asyncio
    async def test_validate_valid_data(self, sample_market_data):
        """Test validation of valid market data."""
        validator = DataQualityValidator()
        
        data_point = sample_market_data[0]
        historical_context = sample_market_data[1:]
        
        result = await validator.validate_data_point(data_point, historical_context)
        
        assert result.is_valid
        assert not result.is_quarantined
        assert result.quality_score.overall_score > 0.8
        assert len(result.validation_errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_invalid_data(self, invalid_data_point, sample_market_data):
        """Test validation of invalid market data."""
        validator = DataQualityValidator()
        
        result = await validator.validate_data_point(
            invalid_data_point, 
            sample_market_data
        )
        
        assert not result.is_valid
        assert len(result.validation_errors) > 0
        assert result.quality_score.accuracy < 0.5
    
    @pytest.mark.asyncio
    async def test_quarantine_bad_data(self, invalid_data_point):
        """Test quarantine mechanism for bad data."""
        validator = DataQualityValidator()
        
        # Set aggressive quarantine thresholds
        validator.config.quarantine_thresholds['overall_score'] = 0.9
        
        result = await validator.validate_data_point(invalid_data_point, [])
        
        assert result.is_quarantined
        assert result.quarantine_reason is not None


class TestAnomalyDetector:
    """Test anomaly detector."""
    
    @pytest.mark.asyncio
    async def test_detect_price_anomaly(self, anomalous_data_point, sample_market_data):
        """Test detection of price anomalies."""
        detector = AnomalyDetector()
        
        results = await detector.detect_anomalies(
            anomalous_data_point,
            sample_market_data,
            methods=[AnomalyMethod.Z_SCORE, AnomalyMethod.IQR]
        )
        
        assert len(results) == 2
        
        # Check z-score detection
        z_score_result = next(r for r in results if r.method == AnomalyMethod.Z_SCORE)
        assert z_score_result.is_anomaly
        assert z_score_result.anomaly_score > 0
        assert 'close_price' in z_score_result.affected_fields
        
        # Check IQR detection
        iqr_result = next(r for r in results if r.method == AnomalyMethod.IQR)
        assert iqr_result.is_anomaly
    
    @pytest.mark.asyncio
    async def test_no_anomaly_normal_data(self, sample_market_data):
        """Test that normal data doesn't trigger anomalies."""
        detector = AnomalyDetector()
        
        # Test middle data point against others
        data_point = sample_market_data[5]
        historical_context = sample_market_data[:5] + sample_market_data[6:]
        
        results = await detector.detect_anomalies(
            data_point,
            historical_context,
            methods=[AnomalyMethod.Z_SCORE]
        )
        
        assert len(results) == 1
        assert not results[0].is_anomaly


class TestQualityMetricsService:
    """Test quality metrics service."""
    
    @pytest.mark.asyncio
    async def test_calculate_quality_metrics(self, sample_market_data):
        """Test quality metrics calculation."""
        service = QualityMetricsService()
        validator = DataQualityValidator()
        
        # Generate validation results
        validation_results = []
        for i in range(len(sample_market_data) - 1):
            result = await validator.validate_data_point(
                sample_market_data[i],
                sample_market_data[i+1:]
            )
            validation_results.append(result)
        
        # Calculate metrics
        metrics = await service.calculate_quality_metrics(
            "AAPL",
            validation_results,
            [],  # No anomalies
            timedelta(hours=1)
        )
        
        assert len(metrics) > 0
        
        # Check completeness metric
        completeness_metric = next(
            m for m in metrics 
            if m.metric_type.value == "completeness"
        )
        assert completeness_metric.value > 0.9
    
    @pytest.mark.asyncio
    async def test_generate_quality_report(self, sample_market_data):
        """Test quality report generation."""
        service = QualityMetricsService()
        
        # Generate report for last hour
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        report = await service.generate_quality_report(
            "AAPL",
            start_time,
            end_time
        )
        
        assert report.symbol == "AAPL"
        assert report.report_period == (start_time, end_time)


class TestQualityChecks:
    """Test individual quality checks."""
    
    def test_ohlc_consistency_check(self, invalid_data_point):
        """Test OHLC consistency validation."""
        checks = MarketDataQualityChecks()
        
        result = checks.check_ohlc_consistency(
            invalid_data_point,
            {},
            None
        )
        
        assert result.result.value == "fail"
        assert result.score < 1.0
        assert len(result.details['violations']) > 0
    
    def test_price_reasonableness_check(self, anomalous_data_point, sample_market_data):
        """Test price reasonableness check."""
        checks = MarketDataQualityChecks()
        
        result = checks.check_price_reasonableness(
            anomalous_data_point,
            {"max_change_percent": 20.0},
            sample_market_data
        )
        
        assert result.result.value == "warning"
        assert 'unreasonable_changes' in result.details
    
    def test_volume_validation(self, sample_market_data):
        """Test volume validation check."""
        checks = MarketDataQualityChecks()
        
        result = checks.check_volume_validation(
            sample_market_data[0],
            {"min_volume": 0, "volume_spike_threshold": 20.0},
            sample_market_data[1:]
        )
        
        assert result.result.value == "pass"
        assert result.score == 1.0


class TestDataQualityPipeline:
    """Test data quality pipeline orchestration."""
    
    @pytest.mark.asyncio
    async def test_pipeline_real_time_mode(self, sample_market_data):
        """Test pipeline in real-time mode."""
        config = PipelineConfig(
            mode=PipelineMode.REAL_TIME,
            enable_validation=True,
            enable_anomaly_detection=True,
            enable_metrics_collection=True
        )
        
        pipeline = DataQualityPipeline(config)
        
        # Start pipeline
        await pipeline.start()
        
        try:
            # Process data points
            for data_point in sample_market_data[:3]:
                result = await pipeline.process_data_point(data_point)
                assert result.status.value in ["pending", "processing"]
            
            # Give pipeline time to process
            await asyncio.sleep(1)
            
            # Check pipeline status
            status = pipeline.get_pipeline_status()
            assert status['is_running']
            assert status['processing_stats']['total_processed'] > 0
            
        finally:
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_mode(self, sample_market_data):
        """Test pipeline in batch mode."""
        config = PipelineConfig(
            mode=PipelineMode.BATCH,
            enable_validation=True,
            enable_anomaly_detection=True
        )
        
        pipeline = DataQualityPipeline(config)
        
        # Process batch
        results = []
        for data_point in sample_market_data[:3]:
            result = await pipeline.process_data_point(data_point)
            results.append(result)
        
        assert len(results) == 3
        assert all(r.status.value == "validated" for r in results)
        assert all(r.validation_result is not None for r in results)


class TestQualityRulesManager:
    """Test quality rules configuration manager."""
    
    def test_default_configuration(self):
        """Test default quality rules configuration."""
        manager = QualityRulesManager()
        
        assert manager.config.profile == QualityProfile.STANDARD
        assert len(manager.config.quality_dimensions) > 0
        assert len(manager.config.global_rules) > 0
    
    def test_apply_strict_profile(self):
        """Test applying strict quality profile."""
        manager = QualityRulesManager()
        original_threshold = manager.config.quality_dimensions['completeness'].threshold
        
        manager.apply_profile(QualityProfile.STRICT)
        
        new_threshold = manager.config.quality_dimensions['completeness'].threshold
        assert new_threshold > original_threshold
        assert manager.config.anomaly_config.z_score_threshold < 3.0
    
    def test_get_symbol_config(self):
        """Test getting symbol-specific configuration."""
        manager = QualityRulesManager()
        
        # Get config for unknown symbol
        config = manager.get_symbol_config("AAPL")
        assert config.symbol == "AAPL"
        assert config.profile == QualityProfile.STANDARD
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        manager = QualityRulesManager()
        
        errors = manager.validate_config()
        assert len(errors) == 0  # Default config should be valid
        
        # Break configuration
        manager.config.anomaly_config.z_score_threshold = -1.0
        errors = manager.validate_config()
        assert len(errors) > 0


@pytest.mark.asyncio
async def test_end_to_end_quality_pipeline():
    """Test complete end-to-end quality pipeline flow."""
    # Create pipeline with all features enabled
    config = PipelineConfig(
        mode=PipelineMode.BATCH,
        enable_validation=True,
        enable_anomaly_detection=True,
        enable_metrics_collection=True
    )
    
    pipeline = DataQualityPipeline(config)
    
    # Create test data
    now = datetime.utcnow()
    test_data = [
        MarketDataPoint(
            symbol="TEST",
            timestamp=now - timedelta(minutes=i),
            ohlcv=OHLCV(
                open=Decimal("100.00"),
                high=Decimal("101.00"),
                low=Decimal("99.00"),
                close=Decimal("100.50"),
                volume=Decimal("1000000")
            ),
            metadata={"asset_class": AssetClass.EQUITY.value}
        )
        for i in range(5)
    ]
    
    # Add one anomalous point
    test_data.append(
        MarketDataPoint(
            symbol="TEST",
            timestamp=now,
            ohlcv=OHLCV(
                open=Decimal("150.00"),  # 50% jump
                high=Decimal("155.00"),
                low=Decimal("145.00"),
                close=Decimal("150.00"),
                volume=Decimal("50000000")  # 50x volume
            ),
            metadata={"asset_class": AssetClass.EQUITY.value}
        )
    )
    
    # Process all data
    results = []
    for data_point in test_data:
        result = await pipeline.process_data_point(data_point)
        results.append(result)
    
    # Verify results
    assert len(results) == 6
    
    # Normal data should pass validation
    for result in results[:-1]:
        assert result.validation_result.is_valid
        assert not any(r.is_anomaly for r in result.anomaly_results)
    
    # Anomalous data should be detected
    anomalous_result = results[-1]
    assert any(r.is_anomaly for r in anomalous_result.anomaly_results)
    
    # Generate quality report
    report = await pipeline.generate_quality_report("TEST", hours=1)
    assert report is not None