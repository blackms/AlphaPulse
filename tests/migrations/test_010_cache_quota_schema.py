"""
Tests for cache quota schema migration - RED Phase (Story 4.2).

This test suite validates:
- tenant_cache_quotas table creation
- tenant_cache_metrics table creation
- Indexes for performance
- RLS policies for tenant isolation
- Computed columns (hit_rate)
- Constraints (unique tenant_id, positive quotas)
"""
import pytest
from datetime import date, datetime
from uuid import UUID, uuid4
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def test_tenant_id():
    """Test tenant UUID fixture."""
    return UUID('00000000-0000-0000-0000-000000000001')


@pytest.fixture
def test_tenant_id_2():
    """Second test tenant UUID fixture."""
    return UUID('00000000-0000-0000-0000-000000000002')


class TestTenantCacheQuotasTable:
    """Test tenant_cache_quotas table structure and constraints."""

    @pytest.mark.asyncio
    async def test_table_exists(self, db_session: AsyncSession):
        """Test tenant_cache_quotas table exists."""
        result = await db_session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'tenant_cache_quotas'
            )
        """))
        exists = result.scalar()
        assert exists, "tenant_cache_quotas table should exist"

    @pytest.mark.asyncio
    async def test_table_columns(self, db_session: AsyncSession):
        """Test tenant_cache_quotas has all required columns."""
        result = await db_session.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'tenant_cache_quotas'
            ORDER BY ordinal_position
        """))
        columns = {row[0]: {'type': row[1], 'nullable': row[2]}
                   for row in result.fetchall()}

        # Required columns
        assert 'id' in columns
        assert 'tenant_id' in columns
        assert 'quota_mb' in columns
        assert 'current_usage_mb' in columns
        assert 'quota_reset_at' in columns
        assert 'overage_allowed' in columns
        assert 'overage_limit_mb' in columns
        assert 'created_at' in columns
        assert 'updated_at' in columns

        # Check types
        assert columns['tenant_id']['type'] == 'uuid'
        assert columns['quota_mb']['type'] == 'integer'
        assert columns['current_usage_mb']['type'] == 'numeric'
        assert columns['overage_allowed']['type'] == 'boolean'

        # Check nullable constraints
        assert columns['tenant_id']['nullable'] == 'NO'
        assert columns['quota_mb']['nullable'] == 'NO'

    @pytest.mark.asyncio
    async def test_unique_constraint(self, db_session: AsyncSession, test_tenant_id):
        """Test unique constraint on tenant_id."""
        # Insert first quota
        await db_session.execute(text("""
            INSERT INTO tenant_cache_quotas (tenant_id, quota_mb)
            VALUES (:tenant_id, 100)
        """), {'tenant_id': test_tenant_id})
        await db_session.commit()

        # Try to insert duplicate (should fail)
        with pytest.raises(Exception) as exc_info:
            await db_session.execute(text("""
                INSERT INTO tenant_cache_quotas (tenant_id, quota_mb)
                VALUES (:tenant_id, 200)
            """), {'tenant_id': test_tenant_id})
            await db_session.commit()

        assert 'unique' in str(exc_info.value).lower() or 'duplicate' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_default_values(self, db_session: AsyncSession, test_tenant_id):
        """Test default values for columns."""
        await db_session.execute(text("""
            INSERT INTO tenant_cache_quotas (tenant_id, quota_mb)
            VALUES (:tenant_id, 100)
        """), {'tenant_id': test_tenant_id})
        await db_session.commit()

        result = await db_session.execute(text("""
            SELECT quota_mb, current_usage_mb, overage_allowed, overage_limit_mb
            FROM tenant_cache_quotas
            WHERE tenant_id = :tenant_id
        """), {'tenant_id': test_tenant_id})
        row = result.fetchone()

        assert row[0] == 100  # quota_mb (explicit)
        assert row[1] == 0  # current_usage_mb (default 0)
        assert row[2] is False  # overage_allowed (default false)
        assert row[3] == 10  # overage_limit_mb (default 10)

    @pytest.mark.asyncio
    async def test_tenant_id_index(self, db_session: AsyncSession):
        """Test index exists on tenant_id."""
        result = await db_session.execute(text("""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'tenant_cache_quotas'
            AND indexname LIKE '%tenant_id%'
        """))
        indexes = [row[0] for row in result.fetchall()]

        assert len(indexes) > 0, "Should have index on tenant_id"

    @pytest.mark.asyncio
    async def test_overage_index(self, db_session: AsyncSession):
        """Test partial index for overage detection."""
        result = await db_session.execute(text("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'tenant_cache_quotas'
            AND indexname LIKE '%overage%'
        """))
        indexes = [(row[0], row[1]) for row in result.fetchall()]

        assert len(indexes) > 0, "Should have overage index"
        # Check it's a partial index (WHERE clause)
        assert any('WHERE' in indexdef.upper() for _, indexdef in indexes)


class TestTenantCacheMetricsTable:
    """Test tenant_cache_metrics table structure and constraints."""

    @pytest.mark.asyncio
    async def test_table_exists(self, db_session: AsyncSession):
        """Test tenant_cache_metrics table exists."""
        result = await db_session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'tenant_cache_metrics'
            )
        """))
        exists = result.scalar()
        assert exists, "tenant_cache_metrics table should exist"

    @pytest.mark.asyncio
    async def test_table_columns(self, db_session: AsyncSession):
        """Test tenant_cache_metrics has all required columns."""
        result = await db_session.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'tenant_cache_metrics'
            ORDER BY ordinal_position
        """))
        columns = {row[0]: {'type': row[1], 'nullable': row[2]}
                   for row in result.fetchall()}

        # Required columns
        assert 'id' in columns
        assert 'tenant_id' in columns
        assert 'metric_date' in columns
        assert 'total_requests' in columns
        assert 'cache_hits' in columns
        assert 'cache_misses' in columns
        assert 'hit_rate' in columns
        assert 'avg_response_time_ms' in columns
        assert 'total_bytes_served' in columns
        assert 'created_at' in columns

        # Check types
        assert columns['tenant_id']['type'] == 'uuid'
        assert columns['metric_date']['type'] == 'date'
        assert columns['total_requests']['type'] == 'bigint'
        assert columns['cache_hits']['type'] == 'bigint'
        assert columns['cache_misses']['type'] == 'bigint'
        assert columns['hit_rate']['type'] == 'numeric'

    @pytest.mark.asyncio
    async def test_computed_hit_rate(self, db_session: AsyncSession, test_tenant_id):
        """Test hit_rate computed column calculation."""
        today = date.today()

        # Insert metrics with 80 hits, 20 misses
        await db_session.execute(text("""
            INSERT INTO tenant_cache_metrics
            (tenant_id, metric_date, total_requests, cache_hits, cache_misses)
            VALUES (:tenant_id, :metric_date, 100, 80, 20)
        """), {'tenant_id': test_tenant_id, 'metric_date': today})
        await db_session.commit()

        # Check computed hit_rate
        result = await db_session.execute(text("""
            SELECT hit_rate
            FROM tenant_cache_metrics
            WHERE tenant_id = :tenant_id AND metric_date = :metric_date
        """), {'tenant_id': test_tenant_id, 'metric_date': today})
        hit_rate = result.scalar()

        assert hit_rate is not None
        assert abs(hit_rate - 80.0) < 0.01, "Hit rate should be 80% (80/100 * 100)"

    @pytest.mark.asyncio
    async def test_unique_tenant_date_constraint(
        self,
        db_session: AsyncSession,
        test_tenant_id
    ):
        """Test unique constraint on (tenant_id, metric_date)."""
        today = date.today()

        # Insert first metric
        await db_session.execute(text("""
            INSERT INTO tenant_cache_metrics
            (tenant_id, metric_date, total_requests, cache_hits, cache_misses)
            VALUES (:tenant_id, :metric_date, 100, 80, 20)
        """), {'tenant_id': test_tenant_id, 'metric_date': today})
        await db_session.commit()

        # Try to insert duplicate (should fail)
        with pytest.raises(Exception) as exc_info:
            await db_session.execute(text("""
                INSERT INTO tenant_cache_metrics
                (tenant_id, metric_date, total_requests, cache_hits, cache_misses)
                VALUES (:tenant_id, :metric_date, 50, 40, 10)
            """), {'tenant_id': test_tenant_id, 'metric_date': today})
            await db_session.commit()

        assert 'unique' in str(exc_info.value).lower() or 'duplicate' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_tenant_date_index(self, db_session: AsyncSession):
        """Test composite index on (tenant_id, metric_date DESC)."""
        result = await db_session.execute(text("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'tenant_cache_metrics'
            AND indexname LIKE '%tenant%date%'
        """))
        indexes = [(row[0], row[1]) for row in result.fetchall()]

        assert len(indexes) > 0, "Should have index on tenant_id + metric_date"
        # Check index includes both columns
        assert any('tenant_id' in indexdef and 'metric_date' in indexdef
                   for _, indexdef in indexes)


class TestRLSPolicies:
    """Test Row-Level Security policies on cache tables."""

    @pytest.mark.asyncio
    async def test_rls_enabled_on_quotas(self, db_session: AsyncSession):
        """Test RLS is enabled on tenant_cache_quotas."""
        result = await db_session.execute(text("""
            SELECT relrowsecurity
            FROM pg_class
            WHERE relname = 'tenant_cache_quotas'
        """))
        rls_enabled = result.scalar()
        assert rls_enabled, "RLS should be enabled on tenant_cache_quotas"

    @pytest.mark.asyncio
    async def test_rls_enabled_on_metrics(self, db_session: AsyncSession):
        """Test RLS is enabled on tenant_cache_metrics."""
        result = await db_session.execute(text("""
            SELECT relrowsecurity
            FROM pg_class
            WHERE relname = 'tenant_cache_metrics'
        """))
        rls_enabled = result.scalar()
        assert rls_enabled, "RLS should be enabled on tenant_cache_metrics"

    @pytest.mark.asyncio
    async def test_rls_policy_exists_quotas(self, db_session: AsyncSession):
        """Test RLS policy exists for tenant_cache_quotas."""
        result = await db_session.execute(text("""
            SELECT policyname
            FROM pg_policies
            WHERE tablename = 'tenant_cache_quotas'
        """))
        policies = [row[0] for row in result.fetchall()]

        assert len(policies) > 0, "Should have at least one RLS policy"
        assert any('isolation' in policy.lower() for policy in policies)

    @pytest.mark.asyncio
    async def test_rls_policy_exists_metrics(self, db_session: AsyncSession):
        """Test RLS policy exists for tenant_cache_metrics."""
        result = await db_session.execute(text("""
            SELECT policyname
            FROM pg_policies
            WHERE tablename = 'tenant_cache_metrics'
        """))
        policies = [row[0] for row in result.fetchall()]

        assert len(policies) > 0, "Should have at least one RLS policy"
        assert any('isolation' in policy.lower() for policy in policies)

    @pytest.mark.asyncio
    async def test_rls_isolation_quotas(
        self,
        db_session: AsyncSession,
        test_tenant_id,
        test_tenant_id_2
    ):
        """Test RLS prevents cross-tenant access for quotas."""
        # Insert quota for tenant 1
        await db_session.execute(text("""
            INSERT INTO tenant_cache_quotas (tenant_id, quota_mb)
            VALUES (:tenant_id, 100)
        """), {'tenant_id': test_tenant_id})
        await db_session.commit()

        # Set tenant context to tenant 1
        await db_session.execute(text(
            "SET LOCAL app.current_tenant_id = :tenant_id"
        ), {'tenant_id': str(test_tenant_id)})

        # Should see tenant 1's quota
        result = await db_session.execute(text("""
            SELECT COUNT(*) FROM tenant_cache_quotas
        """))
        count = result.scalar()
        assert count == 1

        # Set tenant context to tenant 2
        await db_session.execute(text(
            "SET LOCAL app.current_tenant_id = :tenant_id"
        ), {'tenant_id': str(test_tenant_id_2)})

        # Should NOT see tenant 1's quota
        result = await db_session.execute(text("""
            SELECT COUNT(*) FROM tenant_cache_quotas
        """))
        count = result.scalar()
        assert count == 0, "Tenant 2 should not see tenant 1's quota"


class TestDataIntegrity:
    """Test data integrity constraints and defaults."""

    @pytest.mark.asyncio
    async def test_default_quota_values(self, db_session: AsyncSession, test_tenant_id):
        """Test inserting with default values."""
        await db_session.execute(text("""
            INSERT INTO tenant_cache_quotas (tenant_id, quota_mb)
            VALUES (:tenant_id, 100)
        """), {'tenant_id': test_tenant_id})
        await db_session.commit()

        result = await db_session.execute(text("""
            SELECT quota_mb, current_usage_mb, overage_allowed
            FROM tenant_cache_quotas
            WHERE tenant_id = :tenant_id
        """), {'tenant_id': test_tenant_id})
        row = result.fetchone()

        assert row[0] == 100  # quota_mb
        assert row[1] == 0  # current_usage_mb default
        assert row[2] is False  # overage_allowed default

    @pytest.mark.asyncio
    async def test_metrics_default_values(self, db_session: AsyncSession, test_tenant_id):
        """Test inserting metrics with defaults."""
        today = date.today()

        await db_session.execute(text("""
            INSERT INTO tenant_cache_metrics
            (tenant_id, metric_date, total_requests, cache_hits, cache_misses)
            VALUES (:tenant_id, :metric_date, 0, 0, 0)
        """), {'tenant_id': test_tenant_id, 'metric_date': today})
        await db_session.commit()

        result = await db_session.execute(text("""
            SELECT total_requests, cache_hits, cache_misses, total_bytes_served
            FROM tenant_cache_metrics
            WHERE tenant_id = :tenant_id
        """), {'tenant_id': test_tenant_id})
        row = result.fetchone()

        assert row[0] == 0  # total_requests
        assert row[1] == 0  # cache_hits
        assert row[2] == 0  # cache_misses
        assert row[3] == 0  # total_bytes_served default

    @pytest.mark.asyncio
    async def test_timestamps_auto_populated(self, db_session: AsyncSession, test_tenant_id):
        """Test created_at and updated_at timestamps."""
        before = datetime.utcnow()

        await db_session.execute(text("""
            INSERT INTO tenant_cache_quotas (tenant_id, quota_mb)
            VALUES (:tenant_id, 100)
        """), {'tenant_id': test_tenant_id})
        await db_session.commit()

        after = datetime.utcnow()

        result = await db_session.execute(text("""
            SELECT created_at, updated_at
            FROM tenant_cache_quotas
            WHERE tenant_id = :tenant_id
        """), {'tenant_id': test_tenant_id})
        row = result.fetchone()

        created_at = row[0]
        updated_at = row[1]

        assert before <= created_at <= after
        assert before <= updated_at <= after
        assert created_at == updated_at  # Should be same on insert


# Mark all tests as requiring asyncio
pytest_plugins = ('pytest_asyncio',)
