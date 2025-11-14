"""
Example: Using WorkerTenantContext with Celery Tasks

Story 1.4: Create WorkerTenantContext mixin for background jobs
EPIC-001: Database Infrastructure

This example demonstrates how to use the WorkerTenantContext mixin
with Celery tasks to ensure proper tenant isolation using PostgreSQL RLS.
"""

from uuid import UUID
from celery import Task
from alpha_pulse.celery_app import celery_app
from alpha_pulse.mixins.worker_tenant_context import (
    WorkerTenantContext,
    require_tenant_context,
)
from alpha_pulse.config.database import get_db_connection


# Example 1: Custom Celery Task Base Class with WorkerTenantContext
class TenantAwareTask(Task, WorkerTenantContext):
    """
    Custom Celery task base class with tenant context management.

    Usage:
        @celery_app.task(base=TenantAwareTask)
        @require_tenant_context
        async def my_task(tenant_id: UUID, data: dict):
            # Task automatically has tenant context methods available
            pass
    """

    pass


# Example 2: Simple Tenant-Scoped Background Job
@celery_app.task(base=TenantAwareTask)
@require_tenant_context
async def process_tenant_trades(tenant_id: UUID, symbol: str):
    """
    Process trades for a specific tenant.

    The @require_tenant_context decorator ensures:
    1. tenant_id parameter is present
    2. tenant_id is a valid UUID
    3. Function is async

    Args:
        tenant_id: Tenant UUID
        symbol: Trading symbol to process

    Returns:
        Dict with processing results
    """
    async with get_db_connection() as conn:
        # Set tenant context for RLS
        task = TenantAwareTask()
        await task.set_tenant_context(conn, tenant_id)

        try:
            # Query will automatically be filtered by tenant_id via RLS
            trades = await conn.fetch(
                """
                SELECT * FROM trades
                WHERE symbol = $1
                ORDER BY created_at DESC
                LIMIT 100
                """,
                symbol,
            )

            # Verify tenant isolation (optional, for debugging)
            is_isolated = await task.verify_tenant_isolation(conn, tenant_id, "trades")
            assert is_isolated, "Tenant isolation violated!"

            # Process trades...
            processed_count = len(trades)

            return {
                "tenant_id": str(tenant_id),
                "symbol": symbol,
                "processed": processed_count,
            }

        finally:
            # Clear tenant context
            await task.clear_tenant_context(conn)


# Example 3: Batch Processing with Tenant Context
@celery_app.task(base=TenantAwareTask)
@require_tenant_context
async def aggregate_tenant_portfolio(tenant_id: UUID, date: str):
    """
    Aggregate portfolio metrics for a tenant.

    Args:
        tenant_id: Tenant UUID
        date: Date to aggregate (YYYY-MM-DD)

    Returns:
        Dict with aggregated metrics
    """
    async with get_db_connection() as conn:
        task = TenantAwareTask()
        await task.set_tenant_context(conn, tenant_id)

        try:
            # All queries automatically filtered by tenant_id
            portfolio_value = await conn.fetchrow(
                """
                SELECT SUM(quantity * current_price) as total_value
                FROM positions
                WHERE DATE(created_at) = $1
                """,
                date,
            )

            trade_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM trades
                WHERE DATE(created_at) = $1
                """,
                date,
            )

            return {
                "tenant_id": str(tenant_id),
                "date": date,
                "portfolio_value": float(portfolio_value["total_value"] or 0),
                "trade_count": trade_count,
            }

        finally:
            await task.clear_tenant_context(conn)


# Example 4: Multi-Tenant Job (Scans All Tenants)
@celery_app.task(base=TenantAwareTask)
async def scan_all_tenant_portfolios():
    """
    Scan all tenants and process their portfolios.

    This job iterates over tenants and processes each one with proper context.

    Returns:
        Dict with processing summary
    """
    async with get_db_connection() as conn:
        # Get all tenant IDs (this query doesn't need RLS)
        tenants = await conn.fetch("SELECT DISTINCT tenant_id FROM tenants")

        results = []
        task = TenantAwareTask()

        for tenant_row in tenants:
            tenant_id = tenant_row["tenant_id"]

            # Set context for this tenant
            await task.set_tenant_context(conn, tenant_id)

            try:
                # Process tenant data (RLS automatically filters)
                trade_count = await conn.fetchval("SELECT COUNT(*) FROM trades")

                results.append(
                    {"tenant_id": str(tenant_id), "trade_count": trade_count}
                )

            finally:
                # Clear context before moving to next tenant
                await task.clear_tenant_context(conn)

        return {"tenants_processed": len(results), "results": results}


# Example 5: Manual Context Management (Advanced)
from alpha_pulse.mixins.worker_tenant_context import with_tenant_context


async def advanced_processing(tenant_id: UUID):
    """
    Example of manual tenant context management using context manager.

    Use this pattern when you need fine-grained control over context lifecycle.
    """
    async with get_db_connection() as conn:
        # with_tenant_context automatically sets and clears context
        result = await with_tenant_context(
            conn,
            tenant_id,
            lambda: conn.fetch("SELECT * FROM trades LIMIT 10"),
        )

        return result


# Example 6: Error Handling with Tenant Context
@celery_app.task(base=TenantAwareTask, autoretry_for=(Exception,), max_retries=3)
@require_tenant_context
async def resilient_tenant_job(tenant_id: UUID):
    """
    Background job with automatic retries and proper error handling.

    Celery will automatically retry on exceptions, and WorkerTenantContext
    ensures tenant context is properly cleaned up even on failure.
    """
    async with get_db_connection() as conn:
        task = TenantAwareTask()
        await task.set_tenant_context(conn, tenant_id)

        try:
            # Risky operation that might fail
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM high_volume_table WHERE tenant_id = $1", tenant_id
            )

            return {"tenant_id": str(tenant_id), "count": result}

        except Exception as e:
            # Tenant context will be cleared in finally block
            raise

        finally:
            await task.clear_tenant_context(conn)


# Example 7: Scheduled Periodic Task with Tenant Context
from celery.schedules import crontab


@celery_app.task(base=TenantAwareTask)
@require_tenant_context
async def daily_tenant_report(tenant_id: UUID):
    """
    Generate daily report for a tenant.

    Schedule in celery beat config:
    ```python
    celery_app.conf.beat_schedule = {
        'daily-reports-tenant-1': {
            'task': 'tasks.daily_tenant_report',
            'schedule': crontab(hour=0, minute=0),
            'args': ['00000000-0000-0000-0000-000000000001'],
        },
    }
    ```
    """
    async with get_db_connection() as conn:
        task = TenantAwareTask()
        await task.set_tenant_context(conn, tenant_id)

        try:
            # Generate report with RLS-filtered data
            summary = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as trade_count,
                    SUM(quantity * price) as total_volume
                FROM trades
                WHERE created_at >= NOW() - INTERVAL '1 day'
                """
            )

            return {
                "tenant_id": str(tenant_id),
                "trade_count": summary["trade_count"],
                "total_volume": float(summary["total_volume"] or 0),
            }

        finally:
            await task.clear_tenant_context(conn)


# Example 8: Chain Multiple Tenant-Aware Tasks
from celery import chain


def process_tenant_pipeline(tenant_id: UUID):
    """
    Chain multiple tenant-aware tasks together.

    Each task in the chain will receive tenant_id and maintain proper context.
    """
    workflow = chain(
        process_tenant_trades.s(tenant_id, "BTC_USDT"),
        aggregate_tenant_portfolio.s(tenant_id, "2025-11-14"),
        daily_tenant_report.s(tenant_id),
    )

    return workflow.apply_async()
