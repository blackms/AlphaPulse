"""
Data Lake Exploration API endpoints.

Provides comprehensive REST API access to the data lake infrastructure,
including dataset browsing, querying, profiling, and visualization.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, status, Query, Depends
from pydantic import BaseModel, validator
from loguru import logger
import pandas as pd
import json

from alpha_pulse.data_lake.manager import DataLakeManager
from alpha_pulse.data_lake.catalog import DataCatalog
from alpha_pulse.data_lake.query import DataLakeQueryEngine
from alpha_pulse.data_lake.profiler import DataProfiler


router = APIRouter(prefix="/datalake", tags=["data-lake"])


class DatasetSearchRequest(BaseModel):
    """Request model for dataset search."""
    query: Optional[str] = None
    layer: Optional[str] = None  # bronze, silver, gold
    dataset_type: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 50
    offset: int = 0


class QueryRequest(BaseModel):
    """Request model for data lake queries."""
    sql: str
    limit: int = 1000
    timeout_seconds: int = 30
    cache_enabled: bool = True
    
    @validator('sql')
    def validate_sql(cls, v):
        # Basic SQL validation
        v = v.strip()
        if not v:
            raise ValueError("SQL query cannot be empty")
        
        # Prevent destructive operations
        destructive_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        sql_upper = v.upper()
        for keyword in destructive_keywords:
            if keyword in sql_upper:
                raise ValueError(f"Destructive operation '{keyword}' not allowed")
        
        return v


class DatasetSampleRequest(BaseModel):
    """Request model for dataset sampling."""
    limit: int = 100
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None


class DatasetProfileRequest(BaseModel):
    """Request model for dataset profiling."""
    columns: Optional[List[str]] = None
    include_histogram: bool = True
    include_correlations: bool = True
    max_categorical_values: int = 20


class DatasetResponse(BaseModel):
    """Response model for dataset metadata."""
    id: str
    name: str
    description: str
    layer: str
    dataset_type: str
    owner: str
    created_at: datetime
    updated_at: datetime
    schema: Dict[str, Any]
    size_bytes: int
    record_count: int
    partition_keys: List[str]
    quality_score: float
    tags: List[str]
    lineage: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for query results."""
    query_id: str
    sql: str
    status: str
    execution_time_ms: float
    record_count: int
    columns: List[str]
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DataProfileResponse(BaseModel):
    """Response model for data profiling."""
    dataset_id: str
    profile_timestamp: datetime
    record_count: int
    column_count: int
    column_profiles: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    correlations: Optional[Dict[str, Any]] = None
    recommendations: List[str]


# Dependency to get data lake manager
def get_data_lake_manager() -> DataLakeManager:
    """Get data lake manager instance."""
    return DataLakeManager()


# Dependency to get data catalog
def get_data_catalog() -> DataCatalog:
    """Get data catalog instance."""
    return DataCatalog()


# Dependency to get query engine
def get_query_engine() -> DataLakeQueryEngine:
    """Get query engine instance."""
    return DataLakeQueryEngine()


# Dependency to get data profiler
def get_data_profiler() -> DataProfiler:
    """Get data profiler instance."""
    return DataProfiler()


@router.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets(
    search: DatasetSearchRequest = Depends(),
    catalog: DataCatalog = Depends(get_data_catalog)
) -> List[DatasetResponse]:
    """
    List and search datasets in the data lake.
    
    Supports filtering by layer, type, owner, tags, and full-text search.
    """
    logger.info(f"Searching datasets with query: {search.query}")
    
    try:
        # Search datasets using catalog
        datasets = catalog.search_datasets(
            query=search.query,
            layer=search.layer,
            dataset_type=search.dataset_type,
            owner=search.owner,
            tags=search.tags,
            limit=search.limit,
            offset=search.offset
        )
        
        # Convert to response format
        results = []
        for dataset in datasets:
            # Get additional metadata
            stats = catalog.get_dataset_statistics(dataset.id)
            lineage = catalog.get_dataset_lineage(dataset.id)
            
            response = DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                layer=dataset.layer,
                dataset_type=dataset.dataset_type,
                owner=dataset.owner,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
                schema=dataset.schema,
                size_bytes=stats.get('size_bytes', 0),
                record_count=stats.get('record_count', 0),
                partition_keys=dataset.partition_keys,
                quality_score=stats.get('quality_score', 0.0),
                tags=dataset.tags,
                lineage=lineage
            )
            results.append(response)
        
        logger.info(f"Found {len(results)} datasets")
        return results
        
    except Exception as e:
        logger.error(f"Failed to search datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset search failed: {str(e)}"
        )


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    catalog: DataCatalog = Depends(get_data_catalog)
) -> DatasetResponse:
    """Get detailed information about a specific dataset."""
    logger.info(f"Getting dataset details: {dataset_id}")
    
    try:
        # Get dataset metadata
        dataset = catalog.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        # Get statistics and lineage
        stats = catalog.get_dataset_statistics(dataset_id)
        lineage = catalog.get_dataset_lineage(dataset_id)
        
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            layer=dataset.layer,
            dataset_type=dataset.dataset_type,
            owner=dataset.owner,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            schema=dataset.schema,
            size_bytes=stats.get('size_bytes', 0),
            record_count=stats.get('record_count', 0),
            partition_keys=dataset.partition_keys,
            quality_score=stats.get('quality_score', 0.0),
            tags=dataset.tags,
            lineage=lineage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset: {str(e)}"
        )


@router.post("/datasets/{dataset_id}/sample")
async def sample_dataset(
    dataset_id: str,
    request: DatasetSampleRequest,
    manager: DataLakeManager = Depends(get_data_lake_manager)
) -> Dict[str, Any]:
    """Get a sample of data from a dataset."""
    logger.info(f"Sampling dataset {dataset_id} with limit {request.limit}")
    
    try:
        # Get dataset metadata
        catalog = get_data_catalog()
        dataset = catalog.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        # Sample data
        sample_df = manager.sample_dataset(
            dataset_id=dataset_id,
            limit=request.limit,
            columns=request.columns,
            filters=request.filters
        )
        
        if sample_df.empty:
            return {
                "dataset_id": dataset_id,
                "columns": [],
                "data": [],
                "sample_size": 0,
                "total_columns": 0
            }
        
        # Convert to JSON-serializable format
        columns = sample_df.columns.tolist()
        data = sample_df.to_dict('records')
        
        # Handle datetime and other non-serializable types
        for record in data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                    record[key] = value.item() if pd.notna(value) else None
        
        return {
            "dataset_id": dataset_id,
            "columns": columns,
            "data": data,
            "sample_size": len(data),
            "total_columns": len(columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sample dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sample dataset: {str(e)}"
        )


@router.post("/datasets/{dataset_id}/profile", response_model=DataProfileResponse)
async def profile_dataset(
    dataset_id: str,
    request: DatasetProfileRequest,
    profiler: DataProfiler = Depends(get_data_profiler)
) -> DataProfileResponse:
    """Generate a comprehensive profile of a dataset."""
    logger.info(f"Profiling dataset {dataset_id}")
    
    try:
        # Get dataset metadata
        catalog = get_data_catalog()
        dataset = catalog.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        # Generate profile
        profile = profiler.profile_dataset(
            dataset_id=dataset_id,
            columns=request.columns,
            include_histogram=request.include_histogram,
            include_correlations=request.include_correlations,
            max_categorical_values=request.max_categorical_values
        )
        
        return DataProfileResponse(
            dataset_id=dataset_id,
            profile_timestamp=datetime.now(),
            record_count=profile.get('record_count', 0),
            column_count=profile.get('column_count', 0),
            column_profiles=profile.get('column_profiles', {}),
            quality_metrics=profile.get('quality_metrics', {}),
            correlations=profile.get('correlations'),
            recommendations=profile.get('recommendations', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to profile dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to profile dataset: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    query_engine: DataLakeQueryEngine = Depends(get_query_engine)
) -> QueryResponse:
    """Execute a SQL query against the data lake."""
    import time
    import uuid
    
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Executing query {query_id}: {request.sql[:100]}...")
    
    try:
        # Execute query
        result_df = query_engine.execute_query(
            sql=request.sql,
            limit=request.limit,
            timeout_seconds=request.timeout_seconds,
            cache_enabled=request.cache_enabled
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        if result_df.empty:
            return QueryResponse(
                query_id=query_id,
                sql=request.sql,
                status="completed",
                execution_time_ms=execution_time,
                record_count=0,
                columns=[],
                data=[],
                metadata={}
            )
        
        # Convert to JSON format
        columns = result_df.columns.tolist()
        data = result_df.to_dict('records')
        
        # Handle non-serializable types
        for record in data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif hasattr(value, 'item'):
                    record[key] = value.item()
        
        return QueryResponse(
            query_id=query_id,
            sql=request.sql,
            status="completed",
            execution_time_ms=execution_time,
            record_count=len(data),
            columns=columns,
            data=data,
            metadata={
                "query_plan": query_engine.get_query_plan(request.sql),
                "cache_hit": query_engine.was_cache_hit()
            }
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"Query {query_id} failed: {e}")
        
        return QueryResponse(
            query_id=query_id,
            sql=request.sql,
            status="failed",
            execution_time_ms=execution_time,
            record_count=0,
            columns=[],
            data=[],
            metadata={"error": str(e)}
        )


@router.get("/datasets/{dataset_id}/lineage")
async def get_dataset_lineage(
    dataset_id: str,
    catalog: DataCatalog = Depends(get_data_catalog)
) -> Dict[str, Any]:
    """Get the lineage graph for a dataset."""
    logger.info(f"Getting lineage for dataset {dataset_id}")
    
    try:
        # Get full lineage graph
        lineage = catalog.get_dataset_lineage(dataset_id, include_graph=True)
        
        if not lineage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        return lineage
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get lineage for {dataset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get lineage: {str(e)}"
        )


@router.get("/statistics")
async def get_data_lake_statistics(
    manager: DataLakeManager = Depends(get_data_lake_manager)
) -> Dict[str, Any]:
    """Get overall data lake statistics and health metrics."""
    logger.info("Getting data lake statistics")
    
    try:
        stats = manager.get_data_lake_statistics()
        
        return {
            "total_datasets": stats.get('total_datasets', 0),
            "total_size_bytes": stats.get('total_size_bytes', 0),
            "total_records": stats.get('total_records', 0),
            "layer_breakdown": stats.get('layer_breakdown', {}),
            "storage_costs": stats.get('storage_costs', {}),
            "quality_metrics": stats.get('quality_metrics', {}),
            "performance_metrics": stats.get('performance_metrics', {}),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get data lake statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/health")
async def check_data_lake_health(
    manager: DataLakeManager = Depends(get_data_lake_manager)
) -> Dict[str, Any]:
    """Check the health status of the data lake."""
    logger.info("Checking data lake health")
    
    try:
        health = manager.check_health()
        
        return {
            "status": "healthy" if health.get('overall_healthy', False) else "unhealthy",
            "checks": health.get('checks', {}),
            "recommendations": health.get('recommendations', []),
            "last_check": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }


@router.delete("/cache")
async def clear_query_cache(
    query_engine: DataLakeQueryEngine = Depends(get_query_engine)
) -> Dict[str, str]:
    """Clear the query result cache."""
    logger.info("Clearing query cache")
    
    try:
        query_engine.clear_cache()
        return {"status": "success", "message": "Query cache cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {"status": "error", "message": str(e)}