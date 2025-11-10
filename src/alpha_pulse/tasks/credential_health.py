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

from alpha_pulse.services.credentials import TenantCredentialService, TenantCredentials
from alpha_pulse.services.credentials.validator import CredentialValidator
from alpha_pulse.utils.secrets_manager import HashiCorpVaultProvider
from alpha_pulse.config.secure_settings import get_settings

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
    settings = get_settings()

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


async def _check_credentials_async() -> Dict[str, int]:
    """
    Async implementation of credential health checks.

    Returns:
        Dict with counts: checked, failed, webhooks_sent
    """
    settings = get_settings()

    # Initialize services
    vault_provider = HashiCorpVaultProvider(
        vault_addr="http://localhost:8200",  # TODO: Get from settings
        vault_token=None,  # Will use VAULT_TOKEN env var
    )
    validator = CredentialValidator(timeout=10)
    credential_service = TenantCredentialService(
        vault_provider=vault_provider,
        validator=validator,
    )

    # Get all tenant credentials from Vault
    # TODO: Implement list_all_credentials() method or query from database
    # For now, this is a placeholder that would need integration with tenant management
    credentials_to_check: List[Dict[str, Any]] = []

    checked_count = 0
    failed_count = 0
    webhooks_sent = 0

    for cred_info in credentials_to_check:
        tenant_id = UUID(cred_info["tenant_id"])
        exchange = cred_info["exchange"]
        credential_type = cred_info.get("credential_type", "trading")

        # Retrieve and validate credential
        start_time = datetime.utcnow()
        try:
            with credential_health_check_duration_seconds.labels(exchange=exchange).time():
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

                    # TODO: Update database - reset consecutive_failures to 0
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

                    # TODO: Increment consecutive_failures in database
                    # TODO: If consecutive_failures >= threshold, send webhook
                    consecutive_failures = 1  # Placeholder

                    if (
                        consecutive_failures
                        >= settings.credential_consecutive_failures_before_alert
                    ):
                        # Send webhook notification
                        webhook_sent = await _send_credential_failure_webhook(
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

    return {
        "checked": checked_count,
        "failed": failed_count,
        "webhooks_sent": webhooks_sent,
    }


async def _send_credential_failure_webhook(
    tenant_id: UUID,
    exchange: str,
    credential_type: str,
    error: str,
    consecutive_failures: int,
) -> bool:
    """
    Send webhook notification for credential failure.

    Args:
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

    # TODO: Implement webhook delivery
    # 1. Get webhook URL and secret from database (tenant_webhooks table)
    # 2. Build webhook payload (JSON)
    # 3. Generate HMAC-SHA256 signature
    # 4. Send HTTP POST with retry logic
    # 5. Log delivery status

    # Placeholder
    webhook_url = None  # Query from database
    if not webhook_url:
        logger.debug(f"No webhook configured for tenant={tenant_id}")
        return False

    try:
        # TODO: Use WebhookNotifier service
        # notifier = WebhookNotifier()
        # success = await notifier.send_credential_failure_webhook(
        #     tenant_id=tenant_id,
        #     webhook_url=webhook_url,
        #     payload={...},
        #     webhook_secret=webhook_secret,
        # )
        # return success
        return False  # Placeholder
    except Exception as e:
        logger.error(f"Failed to send webhook for tenant={tenant_id}: {e}")
        return False


@shared_task(bind=True)
def check_credential_health_manual(self, tenant_id: str, exchange: str, credential_type: str = "trading"):
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
        vault_addr="http://localhost:8200",
        vault_token=None,
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
