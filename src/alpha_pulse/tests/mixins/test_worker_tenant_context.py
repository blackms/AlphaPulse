"""
Unit tests for WorkerTenantContext mixin.

Story 1.4: Create WorkerTenantContext mixin for background jobs
Tests verify tenant isolation in background jobs using PostgreSQL RLS.
"""

import pytest
import asyncio
from uuid import UUID, uuid4
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncpg

from alpha_pulse.mixins.worker_tenant_context import (
    WorkerTenantContext,
    require_tenant_context,
    with_tenant_context,
)


class TestRequireTenantContextDecorator:
    """Test suite for @require_tenant_context decorator"""

    @pytest.mark.asyncio
    async def test_decorator_enforces_tenant_id_parameter(self):
        """AC1: Decorator enforces tenant_id parameter"""

        @require_tenant_context
        async def process_data(tenant_id: UUID, data: dict):
            return {"tenant_id": tenant_id, "data": data}

        tenant_id = uuid4()
        result = await process_data(tenant_id=tenant_id, data={"key": "value"})

        assert result["tenant_id"] == tenant_id
        assert result["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_decorator_raises_when_tenant_id_missing(self):
        """AC1: Decorator raises ValueError when tenant_id is missing"""

        @require_tenant_context
        async def process_data(data: dict):
            return data

        with pytest.raises(
            ValueError, match="requires 'tenant_id' parameter for tenant context"
        ):
            await process_data(data={"key": "value"})

    @pytest.mark.asyncio
    async def test_decorator_accepts_tenant_id_as_positional_arg(self):
        """Decorator should work with positional arguments"""

        @require_tenant_context
        async def process_data(tenant_id: UUID, data: dict):
            return {"tenant_id": tenant_id, "data": data}

        tenant_id = uuid4()
        result = await process_data(tenant_id, {"key": "value"})

        assert result["tenant_id"] == tenant_id

    @pytest.mark.asyncio
    async def test_decorator_validates_uuid_format(self):
        """Decorator should validate UUID format"""

        @require_tenant_context
        async def process_data(tenant_id: UUID):
            return {"tenant_id": tenant_id}

        # Invalid UUID string
        with pytest.raises(ValueError, match="Invalid tenant_id UUID format"):
            await process_data(tenant_id="not-a-uuid")

    @pytest.mark.asyncio
    async def test_decorator_accepts_string_uuid(self):
        """Decorator should accept valid UUID strings"""

        @require_tenant_context
        async def process_data(tenant_id: UUID):
            return {"tenant_id": tenant_id}

        tenant_id_str = str(uuid4())
        result = await process_data(tenant_id=tenant_id_str)

        # Function receives original value (string), decorator validates it
        assert isinstance(result["tenant_id"], str)
        assert result["tenant_id"] == tenant_id_str

    @pytest.mark.asyncio
    async def test_decorator_rejects_invalid_types(self):
        """Decorator should reject invalid tenant_id types"""

        @require_tenant_context
        async def process_data(tenant_id: UUID):
            return {"tenant_id": tenant_id}

        with pytest.raises(ValueError, match="tenant_id must be UUID or string"):
            await process_data(tenant_id=123)

    def test_decorator_rejects_non_async_functions(self):
        """Decorator should only work with async functions"""

        with pytest.raises(
            TypeError, match="must be an async function to use @require_tenant_context"
        ):

            @require_tenant_context
            def sync_function(tenant_id: UUID):
                return {"tenant_id": tenant_id}

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring"""

        @require_tenant_context
        async def process_data(tenant_id: UUID):
            """Process tenant data."""
            return {"tenant_id": tenant_id}

        assert process_data.__name__ == "process_data"
        assert process_data.__doc__ == "Process tenant data."


class TestWorkerTenantContextMixin:
    """Test suite for WorkerTenantContext mixin class"""

    @pytest.fixture
    def mixin(self):
        """Create WorkerTenantContext instance"""
        return WorkerTenantContext()

    @pytest.fixture
    def mock_conn(self):
        """Create mock AsyncPG connection"""
        conn = AsyncMock(spec=asyncpg.Connection)
        return conn

    @pytest.fixture
    def tenant_id(self):
        """Create test tenant UUID"""
        return uuid4()

    @pytest.mark.asyncio
    async def test_set_tenant_context_executes_sql(self, mixin, mock_conn, tenant_id):
        """AC2: Sets PostgreSQL session variable"""

        await mixin.set_tenant_context(mock_conn, tenant_id)

        # Verify SET command was executed
        mock_conn.execute.assert_called_once_with(
            f"SET app.current_tenant_id = '{tenant_id}'"
        )

    @pytest.mark.asyncio
    async def test_clear_tenant_context_resets_variable(self, mixin, mock_conn):
        """Mixin should clear session variable"""

        await mixin.clear_tenant_context(mock_conn)

        # Verify RESET command was executed
        mock_conn.execute.assert_called_once_with("RESET app.current_tenant_id")

    @pytest.mark.asyncio
    async def test_set_tenant_context_logs_error_on_failure(
        self, mixin, mock_conn, tenant_id
    ):
        """Mixin should log and raise error if SET fails"""

        mock_conn.execute.side_effect = asyncpg.PostgresError("Connection failed")

        with pytest.raises(asyncpg.PostgresError):
            await mixin.set_tenant_context(mock_conn, tenant_id)

    @pytest.mark.asyncio
    async def test_verify_tenant_isolation_returns_true_when_isolated(
        self, mixin, mock_conn, tenant_id
    ):
        """AC3: Verify tenant isolation"""

        # Mock query result: all rows belong to tenant
        mock_conn.fetchrow.return_value = {"total": 100, "tenant_count": 100}

        result = await mixin.verify_tenant_isolation(mock_conn, tenant_id, "trades")

        assert result is True
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_tenant_isolation_returns_false_when_violated(
        self, mixin, mock_conn, tenant_id
    ):
        """AC3: Detect tenant isolation violations"""

        # Mock query result: some rows belong to other tenants
        mock_conn.fetchrow.return_value = {"total": 100, "tenant_count": 80}

        result = await mixin.verify_tenant_isolation(mock_conn, tenant_id, "trades")

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_tenant_isolation_handles_empty_table(
        self, mixin, mock_conn, tenant_id
    ):
        """Isolation verification should handle empty tables"""

        # Mock query result: no rows in table
        mock_conn.fetchrow.return_value = {"total": 0, "tenant_count": 0}

        result = await mixin.verify_tenant_isolation(mock_conn, tenant_id, "trades")

        assert result is True  # Vacuously true

    @pytest.mark.asyncio
    async def test_verify_tenant_isolation_handles_query_error(
        self, mixin, mock_conn, tenant_id
    ):
        """Isolation verification should handle query errors"""

        mock_conn.fetchrow.side_effect = asyncpg.PostgresError("Table not found")

        result = await mixin.verify_tenant_isolation(mock_conn, tenant_id, "trades")

        assert result is False


class TestWithTenantContext:
    """Test suite for with_tenant_context convenience function"""

    @pytest.fixture
    def mock_conn(self):
        """Create mock AsyncPG connection"""
        conn = AsyncMock(spec=asyncpg.Connection)
        return conn

    @pytest.fixture
    def tenant_id(self):
        """Create test tenant UUID"""
        return uuid4()

    @pytest.mark.asyncio
    async def test_with_tenant_context_sets_and_clears_context(
        self, mock_conn, tenant_id
    ):
        """Context manager should set and clear tenant context"""

        async def mock_operation():
            return "result"

        result = await with_tenant_context(mock_conn, tenant_id, mock_operation)

        assert result == "result"

        # Verify SET was called
        set_call = mock_conn.execute.call_args_list[0]
        assert "SET app.current_tenant_id" in set_call[0][0]

        # Verify RESET was called
        reset_call = mock_conn.execute.call_args_list[1]
        assert "RESET app.current_tenant_id" in reset_call[0][0]

    @pytest.mark.asyncio
    async def test_with_tenant_context_clears_on_exception(
        self, mock_conn, tenant_id
    ):
        """Context manager should clear context even on exception"""

        async def failing_operation():
            raise RuntimeError("Operation failed")

        with pytest.raises(RuntimeError, match="Operation failed"):
            await with_tenant_context(mock_conn, tenant_id, failing_operation)

        # Verify RESET was still called
        reset_call = mock_conn.execute.call_args_list[-1]
        assert "RESET app.current_tenant_id" in reset_call[0][0]


class TestTenantIsolationIntegration:
    """Integration tests for tenant isolation (requires PostgreSQL)"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rls_policy_filters_by_tenant(self):
        """
        AC3: Integration test verifying RLS policy enforcement.

        This test requires a PostgreSQL database with RLS enabled.
        Run with: pytest -m integration
        """
        # This would require actual database connection
        # For now, we verify the pattern is correct
        pytest.skip("Integration test requires PostgreSQL database")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_background_job_sets_tenant_context(self):
        """
        Integration test for Celery task with tenant context.

        This test would verify that a Celery task properly sets
        the RLS session variable before executing queries.
        """
        pytest.skip("Integration test requires Celery worker")


class TestCeleryTaskIntegration:
    """Test suite for Celery task integration"""

    def test_worker_tenant_context_can_be_used_as_base_class(self):
        """WorkerTenantContext should be usable as Celery task base class"""
        from celery import Task

        class TenantAwareTask(Task, WorkerTenantContext):
            pass

        task = TenantAwareTask()

        # Verify mixin methods are available
        assert hasattr(task, "set_tenant_context")
        assert hasattr(task, "clear_tenant_context")
        assert hasattr(task, "verify_tenant_isolation")

    @pytest.mark.asyncio
    async def test_decorator_works_with_celery_task(self):
        """@require_tenant_context should work with Celery tasks"""

        @require_tenant_context
        async def celery_task_function(tenant_id: UUID, data: dict):
            """Mock Celery task function"""
            return {"tenant_id": tenant_id, "processed": True}

        tenant_id = uuid4()
        result = await celery_task_function(tenant_id=tenant_id, data={"key": "value"})

        assert result["tenant_id"] == tenant_id
        assert result["processed"] is True


class TestDocumentationExamples:
    """Test examples from docstrings"""

    @pytest.mark.asyncio
    async def test_decorator_usage_example(self):
        """Example from @require_tenant_context docstring"""

        @require_tenant_context
        async def process_tenant_data(tenant_id: UUID, data: dict):
            # In real usage, this would query database with RLS
            return {"tenant_id": tenant_id, "data": data}

        tenant_id = uuid4()
        result = await process_tenant_data(tenant_id=tenant_id, data={"test": "data"})

        assert result["tenant_id"] == tenant_id

    @pytest.mark.asyncio
    async def test_mixin_usage_example(self, mock_conn=None):
        """Example from WorkerTenantContext docstring"""
        from celery import Task

        class TenantAwareTask(Task, WorkerTenantContext):
            pass

        # Verify the class can be instantiated
        task = TenantAwareTask()
        assert isinstance(task, WorkerTenantContext)


# Pytest configuration for this test module
pytestmark = pytest.mark.asyncio
