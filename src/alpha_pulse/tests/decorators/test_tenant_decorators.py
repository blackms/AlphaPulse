"""
Tests for multi-tenant decorators.

Following TDD approach: RED -> GREEN -> REFACTOR
"""
import pytest
from alpha_pulse.decorators.tenant_decorators import require_tenant_id


class TestRequireTenantIdDecorator:
    """Test suite for @require_tenant_id decorator."""

    @pytest.mark.asyncio
    async def test_async_function_requires_tenant_id(self, default_tenant_id):
        """Test that decorator raises ValueError when tenant_id is missing from async function."""

        @require_tenant_id
        async def sample_async_function(market_data, tenant_id: str = None):
            return f"Processing for tenant {tenant_id}"

        # Act & Assert - calling without tenant_id should raise
        with pytest.raises(ValueError, match="requires 'tenant_id' parameter"):
            await sample_async_function("mock_data")

    @pytest.mark.asyncio
    async def test_async_function_with_tenant_id(self, default_tenant_id):
        """Test that decorator allows async function to proceed when tenant_id is provided."""

        @require_tenant_id
        async def sample_async_function(market_data, tenant_id: str = None):
            return f"Processing for tenant {tenant_id}"

        # Act
        result = await sample_async_function("mock_data", tenant_id=default_tenant_id)

        # Assert
        assert result == f"Processing for tenant {default_tenant_id}"

    def test_sync_function_requires_tenant_id(self, default_tenant_id):
        """Test that decorator raises ValueError when tenant_id is missing from sync function."""

        @require_tenant_id
        def sample_sync_function(market_data, tenant_id: str = None):
            return f"Processing for tenant {tenant_id}"

        # Act & Assert
        with pytest.raises(ValueError, match="requires 'tenant_id' parameter"):
            sample_sync_function("mock_data")

    def test_sync_function_with_tenant_id(self, default_tenant_id):
        """Test that decorator allows sync function to proceed when tenant_id is provided."""

        @require_tenant_id
        def sample_sync_function(market_data, tenant_id: str = None):
            return f"Processing for tenant {tenant_id}"

        # Act
        result = sample_sync_function("mock_data", tenant_id=default_tenant_id)

        # Assert
        assert result == f"Processing for tenant {default_tenant_id}"

    @pytest.mark.asyncio
    async def test_empty_string_tenant_id_raises_error(self):
        """Test that empty string tenant_id is rejected."""

        @require_tenant_id
        async def sample_function(tenant_id: str = None):
            return "ok"

        # Act & Assert
        with pytest.raises(ValueError, match="requires 'tenant_id' parameter"):
            await sample_function(tenant_id="")

    @pytest.mark.asyncio
    async def test_none_tenant_id_raises_error(self):
        """Test that None tenant_id is rejected."""

        @require_tenant_id
        async def sample_function(tenant_id: str = None):
            return "ok"

        # Act & Assert
        with pytest.raises(ValueError, match="requires 'tenant_id' parameter"):
            await sample_function(tenant_id=None)
