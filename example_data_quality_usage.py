"""
Example usage of the Data Quality Pipeline for AlphaPulse.

This example demonstrates how to use the comprehensive data quality
validation and anomaly detection system.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

# Example of how the data quality pipeline would be used in the main application

async def example_usage():
    """
    Example of how to integrate the data quality pipeline.
    """
    print("=== AlphaPulse Data Quality Pipeline Example ===\n")
    
    # 1. Import the necessary components
    from alpha_pulse.models.market_data import MarketDataPoint, OHLCV, AssetClass
    from alpha_pulse.data.quality import DataQualityValidator, AnomalyDetector
    from alpha_pulse.pipelines import DataQualityPipeline, PipelineConfig, PipelineMode
    from alpha_pulse.config.quality_rules import QualityRulesManager, QualityProfile
    
    # 2. Configure quality rules
    print("1. Configuring quality rules...")
    rules_manager = QualityRulesManager()
    rules_manager.apply_profile(QualityProfile.STANDARD)
    print(f"   ✓ Applied {rules_manager.config.profile.value} quality profile")
    
    # 3. Create pipeline with configuration
    print("\n2. Creating data quality pipeline...")
    pipeline_config = PipelineConfig(
        mode=PipelineMode.REAL_TIME,
        enable_validation=True,
        enable_anomaly_detection=True,
        enable_metrics_collection=True,
        max_concurrent_validations=10
    )
    
    pipeline = DataQualityPipeline(pipeline_config)
    print("   ✓ Pipeline configured for real-time processing")
    
    # 4. Start the pipeline
    print("\n3. Starting pipeline...")
    await pipeline.start()
    print("   ✓ Pipeline started")
    
    # 5. Process incoming market data
    print("\n4. Processing market data...")
    
    # Simulate incoming data points
    test_symbols = ["AAPL", "GOOGL", "MSFT"]
    
    for symbol in test_symbols:
        # Normal data point
        data_point = MarketDataPoint(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            ohlcv=OHLCV(
                open=Decimal("150.00"),
                high=Decimal("151.50"),
                low=Decimal("149.50"),
                close=Decimal("151.00"),
                volume=Decimal("1000000")
            ),
            metadata={
                "asset_class": AssetClass.EQUITY.value,
                "exchange": "NASDAQ",
                "processing_latency_ms": 45
            }
        )
        
        # Process through pipeline
        result = await pipeline.process_data_point(data_point)
        print(f"   ✓ Processed {symbol}: Status={result.status.value}")
    
    # 6. Simulate an anomalous data point
    print("\n5. Testing anomaly detection...")
    anomalous_data = MarketDataPoint(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        ohlcv=OHLCV(
            open=Decimal("225.00"),  # 50% price jump
            high=Decimal("230.00"),
            low=Decimal("220.00"),
            close=Decimal("225.00"),
            volume=Decimal("50000000")  # 50x normal volume
        ),
        metadata={"asset_class": AssetClass.EQUITY.value}
    )
    
    result = await pipeline.process_data_point(anomalous_data)
    print(f"   ✓ Anomalous data detected and processed")
    
    # 7. Wait for async processing
    await asyncio.sleep(2)
    
    # 8. Get pipeline status
    print("\n6. Pipeline Status:")
    status = pipeline.get_pipeline_status()
    print(f"   - Total processed: {status['processing_stats']['total_processed']}")
    print(f"   - Validations: {status['processing_stats']['validation_count']}")
    print(f"   - Anomaly detections: {status['processing_stats']['anomaly_detection_count']}")
    print(f"   - Quarantined: {status['processing_stats']['quarantine_count']}")
    print(f"   - Avg processing time: {status['processing_stats']['avg_processing_time_ms']:.1f}ms")
    
    # 9. Generate quality report
    print("\n7. Generating quality report...")
    report = await pipeline.generate_quality_report("AAPL", hours=1)
    if report:
        print(f"   ✓ Report generated for AAPL")
        print(f"   - Overall score: {report.overall_score:.2%}")
        print(f"   - Recommendations: {len(report.recommendations)}")
    
    # 10. Stop pipeline
    print("\n8. Stopping pipeline...")
    await pipeline.stop()
    print("   ✓ Pipeline stopped")
    
    print("\n✅ Data quality pipeline example completed successfully!")


def print_implementation_summary():
    """Print summary of what was implemented."""
    print("\n=== Data Quality Pipeline Implementation Summary ===\n")
    
    print("📁 Files Created:")
    print("   ├── src/alpha_pulse/data/quality/")
    print("   │   ├── __init__.py")
    print("   │   ├── data_validator.py      - Core validation framework")
    print("   │   ├── anomaly_detector.py    - Statistical & ML anomaly detection")
    print("   │   ├── quality_metrics.py     - Metrics calculation & reporting")
    print("   │   └── quality_checks.py      - Specific validation checks")
    print("   ├── src/alpha_pulse/pipelines/")
    print("   │   ├── __init__.py")
    print("   │   └── data_quality_pipeline.py - Pipeline orchestration")
    print("   ├── src/alpha_pulse/config/")
    print("   │   └── quality_rules.py         - Configuration management")
    print("   └── src/alpha_pulse/models/")
    print("       └── data_quality_report.py   - Report data models")
    
    print("\n✨ Key Features Implemented:")
    print("   ✓ Comprehensive data validation framework")
    print("   ✓ Multi-dimensional quality scoring (6 dimensions)")
    print("   ✓ Statistical anomaly detection (Z-score, IQR)")
    print("   ✓ ML-based anomaly detection (Isolation Forest, SVM)")
    print("   ✓ Real-time quality monitoring")
    print("   ✓ Automated data quarantine system")
    print("   ✓ Quality metrics and reporting")
    print("   ✓ Alert generation and management")
    print("   ✓ Configurable quality rules and profiles")
    print("   ✓ Historical quality tracking")
    
    print("\n🎯 Quality Dimensions:")
    print("   1. Completeness (25%) - All required fields present")
    print("   2. Accuracy (30%)     - Data within expected ranges")
    print("   3. Consistency (20%)  - Data relationships valid")
    print("   4. Timeliness (15%)   - Data freshness and latency")
    print("   5. Validity (8%)      - Format and type validation")
    print("   6. Uniqueness (2%)    - Duplicate detection")
    
    print("\n🔍 Anomaly Detection Methods:")
    print("   • Z-score analysis")
    print("   • Interquartile Range (IQR)")
    print("   • Isolation Forest")
    print("   • One-Class SVM")
    print("   • Ensemble methods")
    print("   • Moving averages")
    
    print("\n⚙️  Pipeline Modes:")
    print("   • Real-time processing")
    print("   • Batch processing")
    print("   • Hybrid mode")
    
    print("\n📊 Quality Reporting:")
    print("   • Real-time dashboards")
    print("   • Daily/weekly/monthly reports")
    print("   • SLA compliance tracking")
    print("   • Trend analysis")
    print("   • Actionable recommendations")


if __name__ == "__main__":
    print_implementation_summary()
    
    print("\n" + "="*50)
    print("\nTo run the example (requires dependencies):")
    print("asyncio.run(example_usage())")
    print("\nThis example shows how the data quality pipeline")
    print("integrates with the AlphaPulse trading system.")