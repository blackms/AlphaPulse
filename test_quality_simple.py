#!/usr/bin/env python3
"""Simple test of data quality pipeline without pytest."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from alpha_pulse.models.market_data import MarketDataPoint, OHLCV, AssetClass
from alpha_pulse.data.quality.data_validator import DataQualityValidator
from alpha_pulse.data.quality.anomaly_detector import AnomalyDetector, AnomalyMethod
from alpha_pulse.data.quality.quality_metrics import QualityMetricsService
from alpha_pulse.pipelines.data_quality_pipeline import DataQualityPipeline, PipelineConfig, PipelineMode


async def test_basic_validation():
    """Test basic data validation."""
    print("Testing basic data validation...")
    
    # Create validator
    validator = DataQualityValidator()
    
    # Create valid data point
    valid_point = MarketDataPoint(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        ohlcv=OHLCV(
            open=Decimal("150.00"),
            high=Decimal("151.00"),
            low=Decimal("149.00"),
            close=Decimal("150.50"),
            volume=Decimal("1000000")
        ),
        metadata={"asset_class": AssetClass.EQUITY.value}
    )
    
    # Validate
    result = await validator.validate_data_point(valid_point, [])
    
    print(f"✓ Valid data point - Is Valid: {result.is_valid}")
    print(f"  Overall Score: {result.quality_score.overall_score:.2f}")
    print(f"  Completeness: {result.quality_score.completeness:.2f}")
    print(f"  Accuracy: {result.quality_score.accuracy:.2f}")
    
    # Create invalid data point
    invalid_point = MarketDataPoint(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        ohlcv=OHLCV(
            open=Decimal("150.00"),
            high=Decimal("140.00"),  # High < Open (invalid)
            low=Decimal("155.00"),   # Low > High (invalid)
            close=Decimal("145.00"),
            volume=Decimal("-1000")   # Negative volume (invalid)
        ),
        metadata={"asset_class": AssetClass.EQUITY.value}
    )
    
    # Validate invalid point
    result = await validator.validate_data_point(invalid_point, [])
    
    print(f"\n✓ Invalid data point - Is Valid: {result.is_valid}")
    print(f"  Overall Score: {result.quality_score.overall_score:.2f}")
    print(f"  Validation Errors: {len(result.validation_errors)}")
    for error in result.validation_errors[:3]:
        print(f"    - {error}")


async def test_anomaly_detection():
    """Test anomaly detection."""
    print("\n\nTesting anomaly detection...")
    
    # Create detector
    detector = AnomalyDetector()
    
    # Create historical context
    historical_data = []
    for i in range(20):
        historical_data.append(
            MarketDataPoint(
                symbol="AAPL",
                timestamp=datetime.utcnow() - timedelta(minutes=i+1),
                ohlcv=OHLCV(
                    open=Decimal("150.00"),
                    high=Decimal("151.00"),
                    low=Decimal("149.00"),
                    close=Decimal("150.50"),
                    volume=Decimal("1000000")
                )
            )
        )
    
    # Create anomalous data point (50% price jump)
    anomalous_point = MarketDataPoint(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        ohlcv=OHLCV(
            open=Decimal("225.00"),  # 50% jump
            high=Decimal("230.00"),
            low=Decimal("220.00"),
            close=Decimal("225.00"),
            volume=Decimal("50000000")  # 50x volume
        )
    )
    
    # Detect anomalies
    results = await detector.detect_anomalies(
        anomalous_point,
        historical_data,
        methods=[AnomalyMethod.Z_SCORE, AnomalyMethod.IQR]
    )
    
    print(f"✓ Detected {len(results)} anomaly detection results")
    for result in results:
        print(f"  Method: {result.method.value}")
        print(f"    Is Anomaly: {result.is_anomaly}")
        print(f"    Severity: {result.severity.value}")
        print(f"    Score: {result.anomaly_score:.2f}")
        print(f"    Affected Fields: {', '.join(result.affected_fields)}")


async def test_quality_pipeline():
    """Test quality pipeline orchestration."""
    print("\n\nTesting data quality pipeline...")
    
    # Create pipeline config
    config = PipelineConfig(
        mode=PipelineMode.BATCH,
        enable_validation=True,
        enable_anomaly_detection=True,
        enable_metrics_collection=True
    )
    
    # Create pipeline
    pipeline = DataQualityPipeline(config)
    
    # Create test data
    test_data = []
    now = datetime.utcnow()
    
    # Normal data
    for i in range(5):
        test_data.append(
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
        )
    
    # Process data
    print(f"✓ Processing {len(test_data)} data points through pipeline")
    results = []
    for i, data_point in enumerate(test_data):
        result = await pipeline.process_data_point(data_point)
        results.append(result)
        print(f"  Point {i+1}: Status={result.status.value}, "
              f"Valid={result.validation_result.is_valid if result.validation_result else 'N/A'}, "
              f"Processing Time={result.processing_time_ms:.1f}ms")
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    print(f"\n✓ Pipeline Status:")
    print(f"  Total Processed: {status['processing_stats']['total_processed']}")
    print(f"  Validation Count: {status['processing_stats']['validation_count']}")
    print(f"  Anomaly Detection Count: {status['processing_stats']['anomaly_detection_count']}")
    print(f"  Average Processing Time: {status['processing_stats']['avg_processing_time_ms']:.1f}ms")


async def main():
    """Run all tests."""
    print("=== Data Quality Pipeline Test ===\n")
    
    try:
        await test_basic_validation()
        await test_anomaly_detection()
        await test_quality_pipeline()
        
        print("\n\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())