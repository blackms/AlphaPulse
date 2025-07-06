"""
Data Lake Exploration Demo.

This script demonstrates the complete data exploration interface capabilities:
1. Dataset discovery and browsing
2. Interactive SQL querying
3. Data profiling and statistics
4. Schema exploration and metadata analysis
5. Data quality assessment
6. Performance analytics and optimization
"""
import asyncio
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json
import time

# Import data lake components
from alpha_pulse.data_lake.manager import DataLakeManager
from alpha_pulse.data_lake.catalog import DataCatalog
from alpha_pulse.data_lake.query import DataLakeQueryEngine
from alpha_pulse.data_lake.profiler import DataProfiler
from alpha_pulse.data_lake.models import Dataset, DatasetType, Layer


async def demo_data_lake_exploration():
    """Main demonstration function."""
    logger.info("🚀 Starting Data Lake Exploration Demo")
    
    # Initialize components
    try:
        manager = DataLakeManager()
        catalog = DataCatalog()
        query_engine = DataLakeQueryEngine()
        profiler = DataProfiler()
        
        logger.info("✅ Data lake components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize data lake components: {e}")
        logger.info("📝 This demo requires the full data lake infrastructure")
        return
    
    # Demo sections
    await demo_dataset_discovery(catalog)
    await demo_schema_exploration(catalog)
    await demo_interactive_querying(query_engine)
    await demo_data_profiling(profiler, catalog)
    await demo_quality_assessment(catalog)
    await demo_performance_analytics(manager)
    await demo_data_lineage(catalog)
    
    logger.info("🎉 Data Lake Exploration Demo completed successfully!")


async def demo_dataset_discovery(catalog: DataCatalog):
    """Demonstrate dataset discovery and search capabilities."""
    logger.info("\n📊 === Dataset Discovery Demo ===")
    
    try:
        # Search all datasets
        logger.info("🔍 Searching all datasets...")
        all_datasets = catalog.search_datasets(limit=10)
        
        if all_datasets:
            logger.info(f"Found {len(all_datasets)} datasets:")
            
            for dataset in all_datasets:
                logger.info(f"  📈 {dataset.name} ({dataset.layer})")
                logger.info(f"     Type: {dataset.dataset_type}")
                logger.info(f"     Owner: {dataset.owner}")
                logger.info(f"     Tags: {', '.join(dataset.tags)}")
                
                # Get statistics
                stats = catalog.get_dataset_statistics(dataset.id)
                if stats:
                    size_mb = stats.get('size_bytes', 0) / (1024 * 1024)
                    records = stats.get('record_count', 0)
                    quality = stats.get('quality_score', 0)
                    logger.info(f"     Size: {size_mb:.1f} MB, Records: {records:,}, Quality: {quality:.1%}")
                
                logger.info("")
        else:
            logger.warning("No datasets found in catalog")
        
        # Search by layer
        logger.info("🥈 Searching Silver layer datasets...")
        silver_datasets = catalog.search_datasets(layer="silver", limit=5)
        
        if silver_datasets:
            logger.info(f"Found {len(silver_datasets)} Silver layer datasets:")
            for dataset in silver_datasets:
                logger.info(f"  🥈 {dataset.name} - {dataset.description}")
        
        # Search by type
        logger.info("📊 Searching processed datasets...")
        processed_datasets = catalog.search_datasets(dataset_type="PROCESSED", limit=5)
        
        if processed_datasets:
            logger.info(f"Found {len(processed_datasets)} processed datasets:")
            for dataset in processed_datasets:
                logger.info(f"  ⚙️ {dataset.name}")
        
        # Search by tag
        logger.info("🏷️ Searching datasets with 'market_data' tag...")
        market_datasets = catalog.search_datasets(tags=["market_data"], limit=5)
        
        if market_datasets:
            logger.info(f"Found {len(market_datasets)} market data datasets:")
            for dataset in market_datasets:
                logger.info(f"  📈 {dataset.name}")
        
    except Exception as e:
        logger.error(f"Dataset discovery failed: {e}")


async def demo_schema_exploration(catalog: DataCatalog):
    """Demonstrate schema exploration capabilities."""
    logger.info("\n🔬 === Schema Exploration Demo ===")
    
    try:
        # Get a dataset for schema exploration
        datasets = catalog.search_datasets(limit=1)
        
        if not datasets:
            logger.warning("No datasets available for schema exploration")
            return
        
        dataset = datasets[0]
        logger.info(f"🔍 Exploring schema for dataset: {dataset.name}")
        
        # Display basic schema info
        schema = dataset.schema
        if schema and 'columns' in schema:
            columns = schema['columns']
            logger.info(f"📋 Schema contains {len(columns)} columns:")
            
            for col_name, col_info in columns.items():
                col_type = col_info.get('type', 'Unknown')
                nullable = col_info.get('nullable', True)
                description = col_info.get('description', '')
                
                nullable_text = "nullable" if nullable else "not null"
                logger.info(f"  📊 {col_name} ({col_type}, {nullable_text})")
                
                if description:
                    logger.info(f"      Description: {description}")
        
        # Display partition information
        if dataset.partition_keys:
            logger.info(f"🗂️ Partitioned by: {', '.join(dataset.partition_keys)}")
        else:
            logger.info("🗂️ No partitioning")
        
        # Display primary keys
        if schema and 'primary_keys' in schema:
            primary_keys = schema['primary_keys']
            if primary_keys:
                logger.info(f"🔑 Primary keys: {', '.join(primary_keys)}")
        
        # Display foreign keys
        if schema and 'foreign_keys' in schema:
            foreign_keys = schema['foreign_keys']
            if foreign_keys:
                logger.info("🔗 Foreign key relationships:")
                for fk in foreign_keys:
                    logger.info(f"  {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
        
        # Display indexes
        if schema and 'indexes' in schema:
            indexes = schema['indexes']
            if indexes:
                logger.info("📇 Indexes:")
                for idx in indexes:
                    logger.info(f"  {idx['name']}: {', '.join(idx['columns'])}")
        
    except Exception as e:
        logger.error(f"Schema exploration failed: {e}")


async def demo_interactive_querying(query_engine: DataLakeQueryEngine):
    """Demonstrate interactive SQL querying capabilities."""
    logger.info("\n💻 === Interactive Querying Demo ===")
    
    # Sample queries to demonstrate
    sample_queries = [
        {
            "name": "Dataset Discovery",
            "sql": "SHOW TABLES",
            "description": "List all available tables/datasets"
        },
        {
            "name": "Schema Information",
            "sql": "DESCRIBE market_data_ohlcv",
            "description": "Show schema for OHLCV data"
        },
        {
            "name": "Sample Data",
            "sql": "SELECT * FROM market_data_ohlcv LIMIT 10",
            "description": "Get sample records from market data"
        },
        {
            "name": "Aggregation Query",
            "sql": """
                SELECT 
                    symbol,
                    COUNT(*) as record_count,
                    AVG(close) as avg_close,
                    MIN(date_time) as earliest_date,
                    MAX(date_time) as latest_date
                FROM market_data_ohlcv 
                WHERE symbol IN ('AAPL', 'GOOGL', 'MSFT')
                GROUP BY symbol
                ORDER BY avg_close DESC
            """,
            "description": "Aggregate statistics by symbol"
        },
        {
            "name": "Time Series Analysis",
            "sql": """
                SELECT 
                    DATE(date_time) as trade_date,
                    symbol,
                    close,
                    LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date_time) as prev_close,
                    (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date_time)) / 
                    LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date_time) * 100 as daily_return
                FROM market_data_ohlcv 
                WHERE symbol = 'AAPL' 
                    AND date_time >= '2024-01-01'
                ORDER BY date_time DESC
                LIMIT 20
            """,
            "description": "Calculate daily returns for AAPL"
        }
    ]
    
    for query_info in sample_queries:
        logger.info(f"\n🔍 Running query: {query_info['name']}")
        logger.info(f"📝 Description: {query_info['description']}")
        logger.info(f"💻 SQL: {query_info['sql'].strip()}")
        
        try:
            start_time = time.time()
            
            # Execute query
            result_df = query_engine.execute_query(
                sql=query_info['sql'],
                limit=20,
                timeout_seconds=30
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            if not result_df.empty:
                logger.info(f"✅ Query completed in {execution_time:.0f}ms")
                logger.info(f"📊 Result: {len(result_df)} rows x {len(result_df.columns)} columns")
                
                # Show sample results
                if len(result_df) > 0:
                    logger.info("📋 Sample results:")
                    # Display first few rows
                    sample_rows = min(3, len(result_df))
                    for i in range(sample_rows):
                        row_data = result_df.iloc[i].to_dict()
                        formatted_row = {k: (f"{v:.4f}" if isinstance(v, float) else str(v)) 
                                       for k, v in row_data.items()}
                        logger.info(f"  Row {i+1}: {formatted_row}")
                
                # Show query plan if available
                try:
                    plan = query_engine.get_query_plan(query_info['sql'])
                    if plan:
                        logger.info(f"🗂️ Query plan: {plan}")
                except:
                    pass
                
            else:
                logger.info(f"✅ Query completed in {execution_time:.0f}ms (no results)")
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
        
        # Small delay between queries
        await asyncio.sleep(1)


async def demo_data_profiling(profiler: DataProfiler, catalog: DataCatalog):
    """Demonstrate data profiling capabilities."""
    logger.info("\n📊 === Data Profiling Demo ===")
    
    try:
        # Get a dataset to profile
        datasets = catalog.search_datasets(limit=1)
        
        if not datasets:
            logger.warning("No datasets available for profiling")
            return
        
        dataset = datasets[0]
        logger.info(f"🔍 Profiling dataset: {dataset.name}")
        
        # Generate profile
        profile = profiler.profile_dataset(
            dataset_id=dataset.id,
            include_histogram=True,
            include_correlations=True,
            max_categorical_values=20
        )
        
        if profile:
            logger.info("📈 Profile Results:")
            logger.info(f"  📊 Total Records: {profile.get('record_count', 0):,}")
            logger.info(f"  📋 Total Columns: {profile.get('column_count', 0)}")
            
            # Quality metrics
            quality_metrics = profile.get('quality_metrics', {})
            if quality_metrics:
                logger.info("🎯 Quality Metrics:")
                overall_score = quality_metrics.get('overall_score', 0)
                logger.info(f"  📊 Overall Score: {overall_score:.1%}")
                
                completeness = quality_metrics.get('completeness', 0)
                validity = quality_metrics.get('validity', 0)
                consistency = quality_metrics.get('consistency', 0)
                
                logger.info(f"  ✅ Completeness: {completeness:.1%}")
                logger.info(f"  ✅ Validity: {validity:.1%}")
                logger.info(f"  ✅ Consistency: {consistency:.1%}")
            
            # Column profiles
            column_profiles = profile.get('column_profiles', {})
            if column_profiles:
                logger.info("📋 Column Analysis:")
                
                for col_name, col_profile in list(column_profiles.items())[:5]:  # Show first 5 columns
                    logger.info(f"  📊 {col_name}:")
                    
                    data_type = col_profile.get('data_type', 'Unknown')
                    null_count = col_profile.get('null_count', 0)
                    unique_count = col_profile.get('unique_count', 0)
                    
                    logger.info(f"    Type: {data_type}")
                    logger.info(f"    Nulls: {null_count:,}")
                    logger.info(f"    Unique: {unique_count:,}")
                    
                    # Numeric statistics
                    if 'min_value' in col_profile:
                        min_val = col_profile['min_value']
                        max_val = col_profile['max_value']
                        mean_val = col_profile.get('mean_value', 0)
                        std_val = col_profile.get('std_deviation', 0)
                        
                        logger.info(f"    Range: {min_val} to {max_val}")
                        logger.info(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                    
                    # Categorical statistics
                    if 'top_values' in col_profile:
                        top_values = col_profile['top_values']
                        logger.info(f"    Top values: {list(top_values.keys())[:3]}")
            
            # Correlations
            correlations = profile.get('correlations')
            if correlations:
                logger.info("🔗 Column Correlations:")
                
                # Show strongest correlations
                strong_correlations = []
                for col1, corr_data in correlations.items():
                    for col2, corr_value in corr_data.items():
                        if col1 != col2 and abs(corr_value) > 0.7:
                            strong_correlations.append((col1, col2, corr_value))
                
                if strong_correlations:
                    for col1, col2, corr in strong_correlations[:5]:
                        logger.info(f"  {col1} ↔ {col2}: {corr:.3f}")
                else:
                    logger.info("  No strong correlations found")
            
            # Recommendations
            recommendations = profile.get('recommendations', [])
            if recommendations:
                logger.info("💡 Recommendations:")
                for rec in recommendations[:3]:  # Show first 3 recommendations
                    logger.info(f"  💡 {rec}")
        
    except Exception as e:
        logger.error(f"Data profiling failed: {e}")


async def demo_quality_assessment(catalog: DataCatalog):
    """Demonstrate data quality assessment capabilities."""
    logger.info("\n🎯 === Data Quality Assessment Demo ===")
    
    try:
        # Get datasets and their quality scores
        datasets = catalog.search_datasets(limit=10)
        
        if not datasets:
            logger.warning("No datasets available for quality assessment")
            return
        
        logger.info("📊 Dataset Quality Scores:")
        
        quality_data = []
        for dataset in datasets:
            stats = catalog.get_dataset_statistics(dataset.id)
            if stats:
                quality_score = stats.get('quality_score', 0)
                quality_data.append({
                    'name': dataset.name,
                    'layer': dataset.layer,
                    'quality': quality_score
                })
        
        # Sort by quality score
        quality_data.sort(key=lambda x: x['quality'], reverse=True)
        
        for item in quality_data:
            quality_icon = "🟢" if item['quality'] >= 0.8 else "🟡" if item['quality'] >= 0.6 else "🔴"
            logger.info(f"  {quality_icon} {item['name']} ({item['layer']}): {item['quality']:.1%}")
        
        # Quality distribution by layer
        logger.info("\n📈 Quality Distribution by Layer:")
        layer_quality = {}
        
        for item in quality_data:
            layer = item['layer']
            if layer not in layer_quality:
                layer_quality[layer] = []
            layer_quality[layer].append(item['quality'])
        
        for layer, scores in layer_quality.items():
            avg_quality = sum(scores) / len(scores)
            min_quality = min(scores)
            max_quality = max(scores)
            
            logger.info(f"  🏷️ {layer.title()}: avg={avg_quality:.1%}, range={min_quality:.1%}-{max_quality:.1%}")
        
        # Quality improvement suggestions
        logger.info("\n💡 Quality Improvement Suggestions:")
        
        low_quality = [item for item in quality_data if item['quality'] < 0.7]
        if low_quality:
            logger.info(f"  🔍 {len(low_quality)} datasets need attention:")
            for item in low_quality[:3]:
                logger.info(f"    - {item['name']}: Consider data validation and cleanup")
        else:
            logger.info("  ✅ All datasets meet quality thresholds")
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")


async def demo_performance_analytics(manager: DataLakeManager):
    """Demonstrate performance analytics capabilities."""
    logger.info("\n⚡ === Performance Analytics Demo ===")
    
    try:
        # Get overall statistics
        stats = manager.get_data_lake_statistics()
        
        if stats:
            logger.info("📊 Data Lake Statistics:")
            
            total_datasets = stats.get('total_datasets', 0)
            total_size = stats.get('total_size_bytes', 0)
            total_records = stats.get('total_records', 0)
            
            logger.info(f"  📈 Total Datasets: {total_datasets:,}")
            logger.info(f"  💾 Total Size: {total_size / (1024**3):.2f} GB")
            logger.info(f"  📊 Total Records: {total_records:,}")
            
            # Layer breakdown
            layer_breakdown = stats.get('layer_breakdown', {})
            if layer_breakdown:
                logger.info("\n🏷️ Layer Breakdown:")
                for layer, layer_stats in layer_breakdown.items():
                    dataset_count = layer_stats.get('dataset_count', 0)
                    layer_size = layer_stats.get('total_size', 0)
                    layer_records = layer_stats.get('total_records', 0)
                    
                    logger.info(f"  {layer.title()}:")
                    logger.info(f"    Datasets: {dataset_count:,}")
                    logger.info(f"    Size: {layer_size / (1024**2):.1f} MB")
                    logger.info(f"    Records: {layer_records:,}")
            
            # Storage costs
            storage_costs = stats.get('storage_costs', {})
            if storage_costs:
                logger.info("\n💰 Storage Cost Analysis:")
                total_monthly_cost = 0
                
                for storage_type, cost_info in storage_costs.items():
                    monthly_cost = cost_info.get('monthly_cost', 0)
                    efficiency = cost_info.get('efficiency', 'Unknown')
                    
                    logger.info(f"  {storage_type}: ${monthly_cost:.2f}/month ({efficiency} efficiency)")
                    total_monthly_cost += monthly_cost
                
                logger.info(f"  Total: ${total_monthly_cost:.2f}/month")
            
            # Performance metrics
            performance_metrics = stats.get('performance_metrics', {})
            if performance_metrics:
                logger.info("\n⚡ Performance Metrics:")
                
                avg_query_time = performance_metrics.get('avg_query_time_ms', 0)
                cache_hit_rate = performance_metrics.get('cache_hit_rate', 0)
                compression_ratio = performance_metrics.get('compression_ratio', 0)
                
                logger.info(f"  ⏱️ Avg Query Time: {avg_query_time:.0f}ms")
                logger.info(f"  🎯 Cache Hit Rate: {cache_hit_rate:.1%}")
                logger.info(f"  🗜️ Compression Ratio: {compression_ratio:.1f}x")
        
        # Health check
        logger.info("\n🏥 Health Check:")
        health = manager.check_health()
        
        if health:
            overall_healthy = health.get('overall_healthy', False)
            status_icon = "✅" if overall_healthy else "❌"
            logger.info(f"  {status_icon} Overall Status: {'Healthy' if overall_healthy else 'Unhealthy'}")
            
            checks = health.get('checks', {})
            for check_name, check_result in checks.items():
                check_status = check_result.get('status', 'unknown')
                check_icon = "✅" if check_status == 'pass' else "⚠️" if check_status == 'warn' else "❌"
                logger.info(f"  {check_icon} {check_name.replace('_', ' ').title()}: {check_status}")
            
            recommendations = health.get('recommendations', [])
            if recommendations:
                logger.info("  💡 Recommendations:")
                for rec in recommendations[:3]:
                    logger.info(f"    - {rec}")
        
    except Exception as e:
        logger.error(f"Performance analytics failed: {e}")


async def demo_data_lineage(catalog: DataCatalog):
    """Demonstrate data lineage tracking capabilities."""
    logger.info("\n🌳 === Data Lineage Demo ===")
    
    try:
        # Get datasets to explore lineage
        datasets = catalog.search_datasets(limit=5)
        
        if not datasets:
            logger.warning("No datasets available for lineage exploration")
            return
        
        for dataset in datasets[:3]:  # Show lineage for first 3 datasets
            logger.info(f"\n🔍 Lineage for: {dataset.name}")
            
            # Get lineage information
            lineage = catalog.get_dataset_lineage(dataset.id, include_graph=True)
            
            if lineage:
                # Upstream dependencies
                upstream = lineage.get('upstream', [])
                if upstream:
                    logger.info("  ⬆️ Upstream dependencies:")
                    for dep in upstream:
                        dep_name = dep.get('name', 'Unknown')
                        dep_type = dep.get('relationship_type', 'derived_from')
                        logger.info(f"    📊 {dep_name} ({dep_type})")
                else:
                    logger.info("  ⬆️ No upstream dependencies (source dataset)")
                
                # Downstream consumers
                downstream = lineage.get('downstream', [])
                if downstream:
                    logger.info("  ⬇️ Downstream consumers:")
                    for consumer in downstream:
                        consumer_name = consumer.get('name', 'Unknown')
                        consumer_type = consumer.get('relationship_type', 'feeds_into')
                        logger.info(f"    📈 {consumer_name} ({consumer_type})")
                else:
                    logger.info("  ⬇️ No downstream consumers")
                
                # Lineage depth
                lineage_depth = lineage.get('depth', 0)
                logger.info(f"  📏 Lineage depth: {lineage_depth} levels")
                
                # Data flow path
                flow_path = lineage.get('flow_path', [])
                if flow_path and len(flow_path) > 1:
                    logger.info("  🌊 Data flow path:")
                    path_str = " → ".join(flow_path)
                    logger.info(f"    {path_str}")
            else:
                logger.info("  ❌ No lineage information available")
    
    except Exception as e:
        logger.error(f"Data lineage exploration failed: {e}")


if __name__ == "__main__":
    # Run the comprehensive demo
    logger.info("🎯 Starting comprehensive Data Lake Exploration Demo")
    logger.info("This demo showcases all data exploration interface capabilities")
    
    asyncio.run(demo_data_lake_exploration())