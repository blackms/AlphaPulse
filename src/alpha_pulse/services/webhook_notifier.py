"""
Webhook notification service with retry logic and HMAC signatures.

Sends webhook notifications to tenants for credential failures and other events.
"""
import hmac
import hashlib
import time
from typing import Dict, Any, Optional
from uuid import UUID

import httpx
from loguru import logger
from prometheus_client import Counter

from alpha_pulse.config.secure_settings import get_settings

# Prometheus metrics
webhook_delivery_total = Counter(
    "webhook_delivery_total",
    "Total webhook deliveries attempted",
    ["event_type", "status"],
)


class WebhookNotifier:
    """
    Sends webhook notifications to tenants with retry logic and HMAC security.
    """

    def __init__(self):
        """Initialize webhook notifier with settings."""
        self.settings = get_settings()
        self.timeout = self.settings.webhook_timeout_seconds
        self.max_retries = self.settings.webhook_retry_attempts
        self.backoff_base = self.settings.webhook_retry_backoff_base

    async def send_credential_failure_webhook(
        self,
        tenant_id: UUID,
        webhook_url: str,
        payload: Dict[str, Any],
        webhook_secret: str,
    ) -> bool:
        """
        Send webhook notification for credential failure.

        Args:
            tenant_id: Tenant UUID
            webhook_url: Tenant's webhook URL
            payload: Webhook payload (JSON)
            webhook_secret: Secret for HMAC signature

        Returns:
            True if delivered successfully, False otherwise
        """
        event_type = "credential.failed"

        # Add metadata to payload
        full_payload = {
            **payload,
            "event": event_type,
            "tenant_id": str(tenant_id),
            "timestamp": int(time.time()),
        }

        # Generate HMAC-SHA256 signature
        signature = self._generate_signature(full_payload, webhook_secret)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-AlphaPulse-Signature": f"sha256={signature}",
            "X-AlphaPulse-Timestamp": str(full_payload["timestamp"]),
            "X-AlphaPulse-Event": event_type,
            "User-Agent": "AlphaPulse-Webhook/1.0",
        }

        # Retry with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        webhook_url,
                        json=full_payload,
                        headers=headers,
                    )

                    if response.status_code in (200, 201, 202, 204):
                        logger.info(
                            f"Webhook delivered successfully: tenant={tenant_id}, "
                            f"event={event_type}, attempt={attempt}"
                        )
                        webhook_delivery_total.labels(
                            event_type=event_type, status="success"
                        ).inc()
                        return True
                    else:
                        logger.warning(
                            f"Webhook delivery failed: tenant={tenant_id}, "
                            f"status={response.status_code}, attempt={attempt}"
                        )

            except httpx.TimeoutException:
                logger.warning(
                    f"Webhook delivery timeout: tenant={tenant_id}, "
                    f"attempt={attempt}/{self.max_retries}"
                )
                webhook_delivery_total.labels(
                    event_type=event_type, status="timeout"
                ).inc()

            except Exception as e:
                logger.error(
                    f"Webhook delivery error: tenant={tenant_id}, "
                    f"attempt={attempt}, error={e}"
                )
                webhook_delivery_total.labels(
                    event_type=event_type, status="error"
                ).inc()

            # Exponential backoff before retry (except on last attempt)
            if attempt < self.max_retries:
                backoff_delay = self.backoff_base * (2 ** (attempt - 1))
                logger.debug(f"Retrying webhook in {backoff_delay}s...")
                await self._sleep(backoff_delay)

        # All retries exhausted
        logger.error(
            f"Webhook delivery failed after {self.max_retries} attempts: "
            f"tenant={tenant_id}, event={event_type}"
        )
        webhook_delivery_total.labels(
            event_type=event_type, status="failed"
        ).inc()
        return False

    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload.

        Args:
            payload: Webhook payload (will be JSON-serialized)
            secret: Webhook secret key

        Returns:
            Hex-encoded HMAC signature
        """
        import json

        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(
            secret.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    async def _sleep(seconds: float):
        """Async sleep helper."""
        import asyncio

        await asyncio.sleep(seconds)

    async def send_test_webhook(
        self,
        webhook_url: str,
        webhook_secret: str,
        tenant_id: Optional[UUID] = None,
    ) -> bool:
        """
        Send test webhook to verify configuration.

        Args:
            webhook_url: Webhook URL to test
            webhook_secret: Secret for signature
            tenant_id: Optional tenant ID for testing

        Returns:
            True if test successful, False otherwise
        """
        test_payload = {
            "event": "webhook.test",
            "message": "This is a test webhook from AlphaPulse",
            "tenant_id": str(tenant_id) if tenant_id else "test",
            "timestamp": int(time.time()),
        }

        signature = self._generate_signature(test_payload, webhook_secret)

        headers = {
            "Content-Type": "application/json",
            "X-AlphaPulse-Signature": f"sha256={signature}",
            "X-AlphaPulse-Timestamp": str(test_payload["timestamp"]),
            "X-AlphaPulse-Event": "webhook.test",
            "User-Agent": "AlphaPulse-Webhook/1.0",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    webhook_url,
                    json=test_payload,
                    headers=headers,
                )

                if response.status_code in (200, 201, 202, 204):
                    logger.info(f"Test webhook delivered successfully to {webhook_url}")
                    return True
                else:
                    logger.warning(
                        f"Test webhook failed: status={response.status_code}, url={webhook_url}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Test webhook error: {e}, url={webhook_url}")
            return False


def verify_webhook_signature(
    payload: str, signature: str, secret: str
) -> bool:
    """
    Verify HMAC-SHA256 signature for incoming webhook.

    This helper function can be used by the receiving endpoint to verify
    that webhooks are authentic.

    Args:
        payload: Raw JSON payload string
        signature: Signature from X-AlphaPulse-Signature header (without "sha256=" prefix)
        secret: Webhook secret

    Returns:
        True if signature is valid, False otherwise

    Example:
        ```python
        # In FastAPI endpoint
        @app.post("/webhooks/alphapulse")
        async def receive_webhook(
            request: Request,
            signature: str = Header(None, alias="X-AlphaPulse-Signature")
        ):
            body = await request.body()
            payload_str = body.decode("utf-8")

            # Extract signature (remove "sha256=" prefix)
            sig = signature.replace("sha256=", "")

            if not verify_webhook_signature(payload_str, sig, WEBHOOK_SECRET):
                raise HTTPException(status_code=401, detail="Invalid signature")

            # Process webhook...
        ```
    """
    expected_signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected_signature)
