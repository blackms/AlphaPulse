"""
Quota checker service for enforcement logic.

Implements quota check algorithm with atomic usage tracking.
"""

import logging
from typing import Optional
from uuid import UUID

from alpha_pulse.models.quota import (
    QuotaConfig,
    QuotaDecision,
    QuotaCheckResult
)
from alpha_pulse.services.quota_cache_service import QuotaCacheService
from alpha_pulse.services.usage_tracker import UsageTracker

logger = logging.getLogger(__name__)


class QuotaChecker:
    """
    Quota enforcement checker with atomic usage tracking.

    Implements three-level decision logic:
    - ALLOW: Within quota
    - WARN: Over quota but within overage limit
    - REJECT: Exceeds hard limit
    """

    def __init__(
        self,
        cache_service: QuotaCacheService,
        usage_tracker: UsageTracker
    ):
        """
        Initialize quota checker.

        Args:
            cache_service: Quota cache service
            usage_tracker: Usage tracking service
        """
        self.cache_service = cache_service
        self.usage_tracker = usage_tracker

    async def check_quota(
        self,
        tenant_id: UUID,
        write_size_mb: float
    ) -> QuotaCheckResult:
        """
        Check if write is allowed under quota.

        Algorithm:
        1. Load quota config (with caching)
        2. Atomically increment usage
        3. Check against limits
        4. Rollback if rejected

        Args:
            tenant_id: Tenant identifier
            write_size_mb: Size of write in megabytes

        Returns:
            QuotaCheckResult with decision and metadata

        Raises:
            Exception: If quota cannot be determined
        """
        # Step 1: Get quota configuration
        quota_config = await self.cache_service.get_quota_config(tenant_id)
        if quota_config is None:
            logger.error(
                f"Quota config not found: tenant_id={tenant_id}"
            )
            raise ValueError(f"No quota configuration for tenant {tenant_id}")

        # Step 2: Atomically increment usage (optimistic)
        try:
            new_usage_mb = await self.usage_tracker.increment_usage(
                tenant_id,
                write_size_mb
            )
        except Exception as e:
            logger.error(
                f"Usage increment failed: tenant_id={tenant_id}, error={e}"
            )
            raise

        # Step 3: Check against limits
        hard_limit_mb = quota_config.hard_limit_mb

        # REJECT: Exceeds hard limit
        if new_usage_mb > hard_limit_mb:
            # Rollback usage increment
            try:
                await self.usage_tracker.decrement_usage(
                    tenant_id,
                    write_size_mb
                )
            except Exception as e:
                logger.error(
                    f"Usage rollback failed: tenant_id={tenant_id}, error={e}"
                )

            logger.warning(
                f"Quota rejected: tenant_id={tenant_id}, write_size_mb={write_size_mb}, "
                f"new_usage_mb={new_usage_mb}, hard_limit_mb={hard_limit_mb}"
            )

            return QuotaCheckResult(
                decision=QuotaDecision.REJECT,
                quota_config=QuotaConfig(
                    tenant_id=tenant_id,
                    quota_mb=quota_config.quota_mb,
                    current_usage_mb=new_usage_mb - write_size_mb,  # Before increment
                    overage_allowed=quota_config.overage_allowed,
                    overage_limit_mb=quota_config.overage_limit_mb
                ),
                requested_mb=write_size_mb,
                new_usage_mb=None,  # Not allocated
                message=f"Write would exceed hard limit ({hard_limit_mb:.1f} MB)"
            )

        # WARN: Over quota but within overage limit
        elif new_usage_mb > quota_config.quota_mb:
            logger.info(
                f"Quota warning: tenant_id={tenant_id}, write_size_mb={write_size_mb}, "
                f"new_usage_mb={new_usage_mb}, quota_mb={quota_config.quota_mb}, hard_limit_mb={hard_limit_mb}"
            )

            return QuotaCheckResult(
                decision=QuotaDecision.WARN,
                quota_config=QuotaConfig(
                    tenant_id=tenant_id,
                    quota_mb=quota_config.quota_mb,
                    current_usage_mb=new_usage_mb,
                    overage_allowed=quota_config.overage_allowed,
                    overage_limit_mb=quota_config.overage_limit_mb
                ),
                requested_mb=write_size_mb,
                new_usage_mb=new_usage_mb,
                message=f"Usage over quota but within overage limit"
            )

        # ALLOW: Within quota
        else:
            logger.debug(
                f"Quota allowed: tenant_id={tenant_id}, write_size_mb={write_size_mb}, "
                f"new_usage_mb={new_usage_mb}, quota_mb={quota_config.quota_mb}"
            )

            return QuotaCheckResult(
                decision=QuotaDecision.ALLOW,
                quota_config=QuotaConfig(
                    tenant_id=tenant_id,
                    quota_mb=quota_config.quota_mb,
                    current_usage_mb=new_usage_mb,
                    overage_allowed=quota_config.overage_allowed,
                    overage_limit_mb=quota_config.overage_limit_mb
                ),
                requested_mb=write_size_mb,
                new_usage_mb=new_usage_mb,
                message="Write allowed"
            )

    async def get_quota_status(self, tenant_id: UUID) -> Optional[QuotaConfig]:
        """
        Get current quota status without modifying usage.

        Args:
            tenant_id: Tenant identifier

        Returns:
            QuotaConfig with current usage, or None if no quota
        """
        return await self.cache_service.get_quota_config(tenant_id)

    async def release_quota(
        self,
        tenant_id: UUID,
        size_mb: float
    ) -> None:
        """
        Release allocated quota (decrement usage).

        Args:
            tenant_id: Tenant identifier
            size_mb: Size to release in megabytes
        """
        try:
            await self.usage_tracker.decrement_usage(tenant_id, size_mb)

            logger.debug(
                f"Quota released: tenant_id={tenant_id}, size_mb={size_mb}"
            )

        except Exception as e:
            logger.error(
                f"Quota release failed: tenant_id={tenant_id}, size_mb={size_mb}, error={e}"
            )
            raise
