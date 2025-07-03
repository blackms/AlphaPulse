"""
Demo script for data lake functionality.

Demonstrates:
- Setting up multi-layer data lake architecture
- Ingesting raw data into Bronze layer
- Processing data to Silver layer
- Creating business datasets in Gold layer
- Data catalog and lineage tracking
- Query optimization and cost analysis
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

from alpha_pulse.data_lake import (
    DataLakeManager,
    DataCatalog,
    DatasetType,
    DatasetSchema,
    BatchIngestionPipeline,
    IngestionConfig,
    IngestionMode,
    PartitionConfig,
    PartitionScheme,
    CompressionHandler,
    DataLakeUtils,
    get_data_lake_config,
    DataLakeEnvironment
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLakeDemo:
    """Demonstrates data lake functionality."""
    
    def __init__(self):
        """Initialize demo."""
        # Get configuration
        self.config = get_data_lake_config(DataLakeEnvironment.DEVELOPMENT)
        
        # Initialize components
        self.lake_manager = DataLakeManager(self.config)
        self.catalog = DataCatalog()
        self.compression_handler = CompressionHandler()
        self.utils = DataLakeUtils()
        
        logger.info("Data Lake Demo initialized")
    
    def generate_sample_market_data(self, num_days: int = 30) -> pd.DataFrame:
        """Generate sample market data for demo."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        exchanges = ['NASDAQ', 'NYSE']
        
        data = []
        base_date = datetime.utcnow() - timedelta(days=num_days)
        
        for day in range(num_days):
            current_date = base_date + timedelta(days=day)
            
            for symbol in symbols:
                for hour in range(9, 17):  # Market hours
                    base_price = np.random.uniform(100, 500)
                    
                    # Generate OHLCV data
                    record = {
                        'timestamp': current_date.replace(hour=hour),
                        'symbol': symbol,
                        'exchange': np.random.choice(exchanges),
                        'open': base_price + np.random.uniform(-2, 2),
                        'high': base_price + np.random.uniform(0, 5),
                        'low': base_price - np.random.uniform(0, 5),
                        'close': base_price + np.random.uniform(-3, 3),
                        'volume': np.random.randint(1000000, 10000000),
                        'trades': np.random.randint(1000, 50000),
                        'vwap': base_price + np.random.uniform(-1, 1)
                    }
                    data.append(record)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} market data records")
        return df
    
    async def demonstrate_bronze_layer(self):
        """Demonstrate Bronze layer ingestion."""
        logger.info("\n=== Bronze Layer Demo ===")
        
        # Generate sample data
        market_data = self.generate_sample_market_data(7)  # 1 week
        
        # Analyze compression options
        logger.info("Analyzing compression options...")
        compression_analysis = self.compression_handler.analyze_compression_ratio(
            market_data,
            test_all=False
        )
        
        for profile, stats in compression_analysis.items():
            if isinstance(stats, dict) and 'compression_ratio' in stats:
                logger.info(
                    f"{profile}: ratio={stats['compression_ratio']:.2f}, "
                    f"savings={stats['space_savings_pct']:.1f}%"
                )
        
        # Register dataset in catalog
        schema = DatasetSchema(
            columns={col: str(market_data[col].dtype) for col in market_data.columns},
            primary_keys=['timestamp', 'symbol', 'exchange'],
            partition_keys=['timestamp'],
            description="Raw market data OHLCV"
        )
        
        dataset_id = self.catalog.register_dataset(
            name="market_data_ohlcv",
            type=DatasetType.RAW,
            layer="bronze",
            path="bronze/market_data/ohlcv",
            schema=schema,
            owner="demo_user",
            tags=["market_data", "ohlcv", "raw"]
        )
        
        logger.info(f"Registered dataset in catalog: {dataset_id}")
        
        # Create ingestion pipeline
        ingestion_config = IngestionConfig(
            source_name="demo_market_feed",
            dataset_name="market_data_ohlcv",
            mode=IngestionMode.BATCH,
            partition_config=PartitionConfig(
                scheme=PartitionScheme.TIME_BASED,
                columns=['timestamp'],
                time_granularity='day'
            ),
            validation_enabled=True,
            quality_threshold=0.8
        )
        
        # Save data to temporary file for batch ingestion
        temp_file = Path("/tmp/market_data_demo.csv")
        market_data.to_csv(temp_file, index=False)
        
        pipeline = BatchIngestionPipeline(
            lake_manager=self.lake_manager,
            config=ingestion_config,
            source_path=str(temp_file)
        )
        
        # Execute ingestion
        job = await pipeline.ingest()
        
        logger.info(f"Ingestion job completed: {job.status.value}")
        logger.info(f"Records processed: {job.records_processed}")
        logger.info(f"Duration: {job.metrics.get('duration_seconds', 0):.2f}s")
        
        # Update catalog with statistics
        self.catalog.update_dataset(
            dataset_id=dataset_id,
            row_count=job.records_processed,
            size_bytes=int(market_data.memory_usage(deep=True).sum())
        )
        
        # Clean up
        temp_file.unlink()
    
    async def demonstrate_silver_layer(self):
        """Demonstrate Silver layer processing."""
        logger.info("\n=== Silver Layer Demo ===")
        
        # Define processing function
        def process_market_data(df: pd.DataFrame) -> pd.DataFrame:
            """Process raw market data for Silver layer."""
            # Add calculated fields
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['price_change'] / df['open']) * 100
            df['high_low_spread'] = df['high'] - df['low']
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Add moving averages
            df = df.sort_values(['symbol', 'timestamp'])
            for window in [5, 10, 20]:
                df[f'ma_{window}'] = df.groupby('symbol')['close'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            
            # Add volatility
            df['volatility'] = df.groupby('symbol')['price_change_pct'].transform(
                lambda x: x.rolling(window=20, min_periods=1).std()
            )
            
            # Quality score
            df['_quality_score'] = 1.0  # Simplified
            
            return df
        
        # Process from Bronze to Silver
        path = self.lake_manager.process_to_silver(
            dataset_name="market_data_ohlcv",
            processing_func=process_market_data
        )
        
        if path:
            logger.info(f"Processed data to Silver layer: {path}")
            
            # Register Silver dataset
            silver_dataset_id = self.catalog.register_dataset(
                name="market_data_processed",
                type=DatasetType.PROCESSED,
                layer="silver",
                path=path,
                schema=DatasetSchema(
                    columns={
                        'timestamp': 'timestamp',
                        'symbol': 'string',
                        'exchange': 'string',
                        'open': 'double',
                        'high': 'double',
                        'low': 'double',
                        'close': 'double',
                        'volume': 'bigint',
                        'price_change': 'double',
                        'price_change_pct': 'double',
                        'volatility': 'double'
                    },
                    primary_keys=['timestamp', 'symbol'],
                    description="Processed market data with technical indicators"
                ),
                owner="demo_user",
                tags=["market_data", "processed", "technical_indicators"]
            )
            
            logger.info(f"Registered Silver dataset: {silver_dataset_id}")
    
    async def demonstrate_gold_layer(self):
        """Demonstrate Gold layer business datasets."""
        logger.info("\n=== Gold Layer Demo ===")
        
        # Define aggregation for business dataset
        def create_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
            """Create daily summary for business users."""
            # Daily aggregation
            daily = df.groupby(['symbol', pd.Grouper(key='timestamp', freq='D')]).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trades': 'sum',
                'volatility': 'mean',
                'price_change_pct': 'mean'
            }).reset_index()
            
            # Add business metrics
            daily['daily_range'] = daily['high'] - daily['low']
            daily['daily_return'] = (daily['close'] - daily['open']) / daily['open'] * 100
            
            # Rank by performance
            daily['performance_rank'] = daily.groupby('timestamp')['daily_return'].rank(
                ascending=False,
                method='dense'
            )
            
            return daily
        
        # Create Gold dataset
        path = self.lake_manager.create_gold_dataset(
            dataset_name="daily_market_summary",
            aggregation_func=create_daily_summary,
            source_datasets=["market_data_processed"],
            business_date=datetime.utcnow().date()
        )
        
        if path:
            logger.info(f"Created Gold dataset: {path}")
            
            # Register Gold dataset
            gold_dataset_id = self.catalog.register_dataset(
                name="daily_market_summary",
                type=DatasetType.BUSINESS,
                layer="gold",
                path=path,
                schema=DatasetSchema(
                    columns={
                        'symbol': 'string',
                        'timestamp': 'date',
                        'open': 'double',
                        'high': 'double',
                        'low': 'double',
                        'close': 'double',
                        'volume': 'bigint',
                        'daily_return': 'double',
                        'performance_rank': 'int'
                    },
                    primary_keys=['symbol', 'timestamp'],
                    description="Daily market summary for business intelligence"
                ),
                owner="demo_user",
                tags=["business", "daily_summary", "market_performance"]
            )
            
            logger.info(f"Registered Gold dataset: {gold_dataset_id}")
    
    async def demonstrate_querying(self):
        """Demonstrate data querying capabilities."""
        logger.info("\n=== Query Demo ===")
        
        try:
            # Query Silver layer data
            query = """
            SELECT 
                symbol,
                DATE(timestamp) as trade_date,
                AVG(close) as avg_close,
                MAX(high) as max_high,
                MIN(low) as min_low,
                SUM(volume) as total_volume,
                AVG(volatility) as avg_volatility
            FROM silver_view
            WHERE symbol IN ('AAPL', 'GOOGL')
            GROUP BY symbol, DATE(timestamp)
            ORDER BY symbol, trade_date
            """
            
            result = self.lake_manager.query_data(
                query=query,
                layer="silver",
                format="pandas"
            )
            
            if not result.empty:
                logger.info(f"Query returned {len(result)} rows")
                logger.info(f"Sample results:\n{result.head()}")
            
        except Exception as e:
            logger.warning(f"Query demo skipped (Spark not available): {e}")
    
    async def demonstrate_optimization(self):
        """Demonstrate storage optimization."""
        logger.info("\n=== Storage Optimization Demo ===")
        
        # Get lake statistics before optimization
        stats_before = self.lake_manager.get_lake_statistics()
        logger.info(f"Before optimization: {stats_before['total_files']} files, "
                   f"{stats_before['total_size'] / (1024**2):.2f} MB")
        
        # Run optimization
        self.lake_manager.optimize_storage(layer="all")
        
        # Get statistics after optimization
        stats_after = self.lake_manager.get_lake_statistics()
        logger.info(f"After optimization: {stats_after['total_files']} files, "
                   f"{stats_after['total_size'] / (1024**2):.2f} MB")
        
        # Show catalog statistics
        catalog_stats = self.catalog.get_catalog_statistics()
        logger.info(f"\nCatalog Statistics:")
        logger.info(f"Total datasets: {catalog_stats['total_datasets']}")
        logger.info(f"Active datasets: {catalog_stats['active_datasets']}")
        logger.info(f"Datasets by layer: {catalog_stats['datasets_by_layer']}")
    
    async def demonstrate_cost_analysis(self):
        """Demonstrate cost analysis."""
        logger.info("\n=== Cost Analysis Demo ===")
        
        # Analyze storage costs
        stats = self.lake_manager.get_lake_statistics()
        total_size_gb = stats['total_size'] / (1024**3)
        
        # Estimate costs for different storage classes
        storage_classes = ["standard", "infrequent", "archive"]
        
        for storage_class in storage_classes:
            cost_estimate = self.compression_handler.estimate_storage_cost(
                uncompressed_size_gb=total_size_gb * 3,  # Assume 3x compression
                compression_profile=self.compression_handler.PROFILES["warm_data"],
                storage_class=storage_class
            )
            
            logger.info(f"\n{storage_class.upper()} storage:")
            logger.info(f"  Compressed size: {cost_estimate['compressed_size_gb']:.2f} GB")
            logger.info(f"  Monthly cost: ${cost_estimate['monthly_cost_usd']:.2f}")
            logger.info(f"  Annual cost: ${cost_estimate['annual_cost_usd']:.2f}")
    
    async def run(self):
        """Run all demonstrations."""
        try:
            # Bronze layer
            await self.demonstrate_bronze_layer()
            
            # Silver layer
            await self.demonstrate_silver_layer()
            
            # Gold layer
            await self.demonstrate_gold_layer()
            
            # Querying
            await self.demonstrate_querying()
            
            # Optimization
            await self.demonstrate_optimization()
            
            # Cost analysis
            await self.demonstrate_cost_analysis()
            
            # Export catalog
            logger.info("\n=== Exporting Catalog ===")
            self.catalog.export_catalog("/tmp/data_lake_catalog.json", format="json")
            logger.info("Catalog exported to /tmp/data_lake_catalog.json")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise
        finally:
            # Cleanup
            self.lake_manager.shutdown()
            self.catalog.close()


async def main():
    """Main entry point."""
    logger.info("🚀 Starting AlphaPulse Data Lake Demo")
    logger.info("=" * 50)
    
    demo = DataLakeDemo()
    await demo.run()
    
    logger.info("\n✅ Data Lake Demo completed!")
    logger.info("\nKey Features Demonstrated:")
    logger.info("- Bronze/Silver/Gold layer architecture")
    logger.info("- Intelligent partitioning and compression")
    logger.info("- Data catalog with lineage tracking")
    logger.info("- Query optimization")
    logger.info("- Storage lifecycle management")
    logger.info("- Cost analysis and optimization")


if __name__ == "__main__":
    asyncio.run(main())