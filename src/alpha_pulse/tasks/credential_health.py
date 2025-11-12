"""
Credential health check task.

Periodically validates all stored tenant credentials and sends webhook notifications on failure.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any
from uuid import UUID

from celery import shared_task
from loguru import logger
from prometheus_client import Counter, Histogram

from alpha_pulse.services.credentials import TenantCredentialService
from alpha_pulse.services.credentials.validator import CredentialValidator
from alpha_pulse.services.webhook_notifier import WebhookNotifier
from alpha_pulse.utils.secrets_manager import HashiCorpVaultProvider
from alpha_pulse.config.secure_settings import get_settings
from alpha_pulse.config.database import get_db_session
from alpha_pulse.models.credential_health import (
    CredentialHealthCheck,
    get_tenant_webhook,
    get_latest_health_check,
)

# Prometheus metrics
credential_health_check_total = Counter(
    "credential_health_check_total",
    "Total credential health checks performed",
    ["tenant_id", "exchange", "result"],
)

credential_health_check_duration_seconds = Histogram(
    "credential_health_check_duration_seconds",
    "Time taken to check credential health",
    ["exchange"],
)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
    soft_time_limit=240,
    time_limit=300,
)
def check_all_credentials_health(self):
    """
    Check health of all stored credentials.

    For each tenant:
    1. Retrieve all credentials from Vault
    2. Validate each credential via CCXT
    3. Track consecutive failures
    4. Send webhook on Nth consecutive failure
    5. Log results and emit metrics

    Runs as a Celery periodic task every N hours (configured in beat_schedule).
    """
    logger.info("Starting credential health check task")

    try:
        # Run async validation in sync context
        result = asyncio.run(_check_credentials_async())
        logger.info(
            f"Credential health check completed: "
            f"{result['checked']} checked, "
            f"{result['failed']} failed, "
            f"{result['webhooks_sent']} webhooks sent"
        )
        return result
    except Exception as e:
        logger.error(f"Credential health check task failed: {e}", exc_info=True)
        raise


async def _list_all_credentials(
    vault_provider: HashiCorpVaultProvider,
) -> List[Dict[str, Any]]:
    """
    Scan Vault for all stored credentials.

    Recursively scans the Vault path structure:
    tenants/{tenant_id}/exchanges/{exchange}/{credential_type}

    Args:
        vault_provider: HashiCorpVaultProvider instance

    Returns:
        List of dicts with {tenant_id, exchange, credential_type}
    """
    credentials = []

    try:
        # List all tenants
        tenant_paths = vault_provider.list_secrets("tenants")

        if not tenant_paths:
            logger.warning("No tenants found in Vault at 'tenants/' path")
            return credentials

        for tenant_path in tenant_paths:
            # tenant_path is like "00000000-0000-0000-0000-000000000001/"
            tenant_id = tenant_path.rstrip("/")

            # List exchanges for this tenant
            try:
                exchange_paths = vault_provider.list_secrets(
                    f"tenants/{tenant_id}/exchanges"
                )

                if not exchange_paths:
                    logger.debug(f"No exchanges found for tenant {tenant_id}")
                    continue

                for exchange_path in exchange_paths:
                    # exchange_path is like "binance/"
                    exchange = exchange_path.rstrip("/")

                    # List credential types for this exchange
                    try:
                        cred_type_paths = vault_provider.list_secrets(
                            f"tenants/{tenant_id}/exchanges/{exchange}"
                        )

                        if not cred_type_paths:
                            logger.debug(
                                f"No credentials found for {tenant_id}/{exchange}"
                            )
                            continue

                        for cred_type_path in cred_type_paths:
                            # cred_type_path is like "trading" (actual secret key, not a folder)
                            credential_type = cred_type_path.rstrip("/")

                            credentials.append(
                                {
                                    "tenant_id": tenant_id,
                                    "exchange": exchange,
                                    "credential_type": credential_type,
                                }
                            )
                            logger.debug(
                                f"Found credential: tenant={tenant_id}, "
                                f"exchange={exchange}, type={credential_type}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error listing credential types for {tenant_id}/{exchange}: {e}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Error listing exchanges for tenant {tenant_id}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error listing credentials from Vault: {e}", exc_info=True)

    logger.info(f"Discovered {len(credentials)} credentials in Vault")
    return credentials


async def _check_credentials_async() -> Dict[str, int]:
    """
    Async implementation of credential health checks.

    Returns:
        Dict with counts: checked, failed, webhooks_sent
    """
    settings = get_settings()

    # Initialize services
    vault_provider = HashiCorpVaultProvider(
        vault_addr=settings.vault_addr,
        vault_token=settings.vault_token,
    )
    validator = CredentialValidator(timeout=10)
    credential_service = TenantCredentialService(
        vault_provider=vault_provider,
        validator=validator,
    )

    # Get all tenant credentials from Vault
    credentials_to_check = await _list_all_credentials(vault_provider)
    logger.info(f"Found {len(credentials_to_check)} credentials to check")

    # Get database session
    session = get_db_session()

    checked_count = 0
    failed_count = 0
    webhooks_sent = 0

    try:
        for cred_info in credentials_to_check:
            tenant_id = UUID(cred_info["tenant_id"])
            exchange = cred_info["exchange"]
            credential_type = cred_info.get("credential_type", "trading")

            # Retrieve and validate credential
            try:
                with credential_health_check_duration_seconds.labels(
                    exchange=exchange
                ).time():
                    credentials = await credential_service.get_credentials(
                        tenant_id, exchange, credential_type
                    )

                    if not credentials:
                        logger.warning(
                            f"No credentials found for tenant={tenant_id}, exchange={exchange}"
                        )
                        continue

                    # Validate via CCXT
                    validation_result = await validator.validate(
                        exchange=exchange,
                        api_key=credentials.api_key,
                        secret=credentials.api_secret,
                        passphrase=credentials.passphrase,
                        testnet=credentials.testnet,
                    )

                    checked_count += 1

                    if validation_result.valid:
                        # Success - reset failure counter
                        logger.debug(
                            f"Credential valid: tenant={tenant_id}, exchange={exchange}"
                        )
                        credential_health_check_total.labels(
                            tenant_id=str(tenant_id),
                            exchange=exchange,
                            result="success",
                        ).inc()

                        # Record success in database
                        health_check = CredentialHealthCheck(
                            tenant_id=tenant_id,
                            exchange=exchange,
                            credential_type=credential_type,
                            check_timestamp=datetime.utcnow(),
                            is_valid=True,
                            validation_error=None,
                            consecutive_failures=0,
                            last_success_at=datetime.utcnow(),
                        )
                        session.add(health_check)
                        session.commit()

                    else:
                        # Failure - increment counter
                        failed_count += 1
                        logger.warning(
                            f"Credential invalid: tenant={tenant_id}, exchange={exchange}, "
                            f"error={validation_result.error}"
                        )
                        credential_health_check_total.labels(
                            tenant_id=str(tenant_id),
                            exchange=exchange,
                            result="failure",
                        ).inc()

                        # Get previous check to track consecutive failures
                        latest_check = await asyncio.to_thread(
                            get_latest_health_check,
                            session,
                            tenant_id,
                            exchange,
                            credential_type,
                        )

                        consecutive_failures = (
                            latest_check.consecutive_failures + 1
                            if latest_check and not latest_check.is_valid
                            else 1
                        )

                        # Record failure in database
                        health_check = CredentialHealthCheck(
                            tenant_id=tenant_id,
                            exchange=exchange,
                            credential_type=credential_type,
                            check_timestamp=datetime.utcnow(),
                            is_valid=False,
                            validation_error=validation_result.error,
                            consecutive_failures=consecutive_failures,
                            first_failure_at=(
                                latest_check.first_failure_at
                                if latest_check and not latest_check.is_valid
                                else datetime.utcnow()
                            ),
                        )
                        session.add(health_check)
                        session.commit()

                        # Send webhook if threshold reached
                        if (
                            consecutive_failures
                            >= settings.credential_consecutive_failures_before_alert
                        ):
                            # Send webhook notification
                            webhook_sent = await _send_credential_failure_webhook(
                                session=session,
                                tenant_id=tenant_id,
                                exchange=exchange,
                                credential_type=credential_type,
                                error=validation_result.error,
                                consecutive_failures=consecutive_failures,
                            )
                            if webhook_sent:
                                webhooks_sent += 1

            except Exception as e:
                logger.error(
                    f"Error checking credential for tenant={tenant_id}, exchange={exchange}: {e}",
                    exc_info=True,
                )
                credential_health_check_total.labels(
                    tenant_id=str(tenant_id),
                    exchange=exchange,
                    result="error",
                ).inc()

    finally:
        session.close()

    return {
        "checked": checked_count,
        "failed": failed_count,
        "webhooks_sent": webhooks_sent,
    }


async def _send_credential_failure_webhook(
    session,
    tenant_id: UUID,
    exchange: str,
    credential_type: str,
    error: str,
    consecutive_failures: int,
) -> bool:
    """
    Send webhook notification for credential failure.

    Args:
        session: Database session
        tenant_id: Tenant UUID
        exchange: Exchange name
        credential_type: Credential type
        error: Validation error message
        consecutive_failures: Number of consecutive failures

    Returns:
        True if webhook sent successfully, False otherwise
    """
    logger.info(
        f"Sending credential failure webhook: tenant={tenant_id}, "
        f"exchange={exchange}, failures={consecutive_failures}"
    )

    try:
        # Get webhook configuration from database
        webhook_config = await asyncio.to_thread(
            get_tenant_webhook, session, tenant_id, "credential.failed"
        )

        if not webhook_config or not webhook_config.is_active:
            logger.debug(
                f"No active webhook configured for tenant={tenant_id}, event=credential.failed"
            )
            return False
        # Initialize webhook notifier
        notifier = WebhookNotifier()

        # Build webhook payload
        payload = {
            "exchange": exchange,
            "credential_type": credential_type,
            "error": error,
            "consecutive_failures": consecutive_failures,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Send webhook
        success = await notifier.send_credential_failure_webhook(
            tenant_id=tenant_id,
            webhook_url=webhook_config.webhook_url,
            payload=payload,
            webhook_secret=webhook_config.webhook_secret,
        )

        # Update webhook delivery stats
        if success:
            webhook_config.last_success_at = datetime.utcnow()
            webhook_config.success_count = (webhook_config.success_count or 0) + 1
            logger.info(f"Webhook delivered successfully for tenant={tenant_id}")
        else:
            webhook_config.last_failure_at = datetime.utcnow()
            webhook_config.failure_count = (webhook_config.failure_count or 0) + 1
            logger.warning(f"Webhook delivery failed for tenant={tenant_id}")

        webhook_config.last_triggered_at = datetime.utcnow()
        session.commit()

        return success

    except Exception as e:
        logger.error(
            f"Failed to send webhook for tenant={tenant_id}: {e}", exc_info=True
        )
        return False


@shared_task(bind=True)
def check_credential_health_manual(
    self, tenant_id: str, exchange: str, credential_type: str = "trading"
):
    """
    Manual on-demand credential health check.

    Can be triggered via API endpoint for immediate feedback.

    Args:
        tenant_id: Tenant UUID (string)
        exchange: Exchange name
        credential_type: Credential type (default: "trading")

    Returns:
        Dict with validation result
    """
    logger.info(
        f"Manual health check requested: tenant={tenant_id}, exchange={exchange}"
    )

    try:
        result = asyncio.run(
            _check_single_credential(UUID(tenant_id), exchange, credential_type)
        )
        return result
    except Exception as e:
        logger.error(
            f"Manual health check failed for tenant={tenant_id}, exchange={exchange}: {e}",
            exc_info=True,
        )
        raise


async def _check_single_credential(
    tenant_id: UUID, exchange: str, credential_type: str
) -> Dict[str, Any]:
    """
    Check health of a single credential.

    Args:
        tenant_id: Tenant UUID
        exchange: Exchange name
        credential_type: Credential type

    Returns:
        Dict with validation result and details
    """
    settings = get_settings()

    # Initialize services
    vault_provider = HashiCorpVaultProvider(
        vault_addr=settings.vault_addr,
        vault_token=settings.vault_token,
    )
    validator = CredentialValidator(timeout=10)
    credential_service = TenantCredentialService(
        vault_provider=vault_provider,
        validator=validator,
    )

    # Get credential
    credentials = await credential_service.get_credentials(
        tenant_id, exchange, credential_type
    )

    if not credentials:
        return {
            "valid": False,
            "error": "Credentials not found",
            "tenant_id": str(tenant_id),
            "exchange": exchange,
        }

    # Validate
    validation_result = await validator.validate(
        exchange=exchange,
        api_key=credentials.api_key,
        secret=credentials.api_secret,
        passphrase=credentials.passphrase,
        testnet=credentials.testnet,
    )

    return {
        "valid": validation_result.valid,
        "error": validation_result.error,
        "credential_type": validation_result.credential_type,
        "exchange_account_id": validation_result.exchange_account_id,
        "tenant_id": str(tenant_id),
        "exchange": exchange,
        "checked_at": datetime.utcnow().isoformat(),
    }
