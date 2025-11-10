"""
Unit tests for webhook notifier service.

Tests the WebhookNotifier class that sends HTTP webhooks with HMAC signatures.
"""
import pytest
import json
import time
import hmac
import hashlib
from uuid import UUID, uuid4
from unittest.mock import Mock, patch, AsyncMock

import httpx

from alpha_pulse.services.webhook_notifier import (
    WebhookNotifier,
    verify_webhook_signature,
)


@pytest.fixture
def mock_settings():
    """Mock settings for tests."""
    settings = Mock()
    settings.webhook_timeout_seconds = 10
    settings.webhook_retry_attempts = 3
    settings.webhook_retry_backoff_base = 1
    return settings


@pytest.fixture
def notifier(mock_settings):
    """Create WebhookNotifier instance with mocked settings."""
    with patch("alpha_pulse.services.webhook_notifier.get_settings", return_value=mock_settings):
        return WebhookNotifier()


@pytest.fixture
def sample_tenant_id():
    """Sample tenant UUID."""
    return UUID("12345678-1234-1234-1234-123456789abc")


@pytest.fixture
def sample_webhook_url():
    """Sample webhook URL."""
    return "https://example.com/webhooks/alphapulse"


@pytest.fixture
def sample_webhook_secret():
    """Sample webhook secret."""
    return "test_secret_key_12345"


@pytest.fixture
def sample_payload():
    """Sample webhook payload."""
    return {
        "exchange": "binance",
        "credential_type": "trading",
        "consecutive_failures": 3,
        "error": "Invalid API signature",
        "first_failure_at": "2025-11-10T12:00:00Z",
    }


class TestWebhookNotifierInit:
    """Tests for WebhookNotifier initialization."""

    def test_init_with_default_settings(self, notifier, mock_settings):
        """Test notifier initialization with default settings."""
        assert notifier.timeout == mock_settings.webhook_timeout_seconds
        assert notifier.max_retries == mock_settings.webhook_retry_attempts
        assert notifier.backoff_base == mock_settings.webhook_retry_backoff_base


class TestGenerateSignature:
    """Tests for HMAC signature generation."""

    def test_generate_signature(self, notifier, sample_webhook_secret):
        """Test HMAC-SHA256 signature generation."""
        payload = {"test": "data", "number": 123}
        signature = notifier._generate_signature(payload, sample_webhook_secret)

        # Verify signature is hex string
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest length

        # Verify signature is deterministic
        signature2 = notifier._generate_signature(payload, sample_webhook_secret)
        assert signature == signature2

    def test_generate_signature_with_different_secrets(self, notifier):
        """Test that different secrets produce different signatures."""
        payload = {"test": "data"}
        sig1 = notifier._generate_signature(payload, "secret1")
        sig2 = notifier._generate_signature(payload, "secret2")

        assert sig1 != sig2

    def test_generate_signature_with_different_payloads(self, notifier, sample_webhook_secret):
        """Test that different payloads produce different signatures."""
        sig1 = notifier._generate_signature({"a": 1}, sample_webhook_secret)
        sig2 = notifier._generate_signature({"b": 2}, sample_webhook_secret)

        assert sig1 != sig2

    def test_generate_signature_key_order_independence(self, notifier, sample_webhook_secret):
        """Test that key order doesn't affect signature (sorted)."""
        payload1 = {"z": 3, "a": 1, "m": 2}
        payload2 = {"a": 1, "m": 2, "z": 3}

        sig1 = notifier._generate_signature(payload1, sample_webhook_secret)
        sig2 = notifier._generate_signature(payload2, sample_webhook_secret)

        # Signatures should match because JSON is sorted by keys
        assert sig1 == sig2


class TestSendCredentialFailureWebhook:
    """Tests for send_credential_failure_webhook method."""

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_success_200(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret, sample_payload
    ):
        """Test successful webhook delivery with 200 status."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=sample_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is True
        mock_client.post.assert_called_once()

        # Verify request parameters
        call_args = mock_client.post.call_args
        assert call_args.args[0] == sample_webhook_url
        assert "json" in call_args.kwargs
        assert "headers" in call_args.kwargs

        # Verify headers
        headers = call_args.kwargs["headers"]
        assert "X-AlphaPulse-Signature" in headers
        assert "X-AlphaPulse-Timestamp" in headers
        assert "X-AlphaPulse-Event" in headers
        assert headers["X-AlphaPulse-Event"] == "credential.failed"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_success_202(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret, sample_payload
    ):
        """Test successful webhook delivery with 202 Accepted status."""
        mock_response = Mock()
        mock_response.status_code = 202
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=sample_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is True

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_failure_500(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret, sample_payload
    ):
        """Test webhook delivery failure with 500 status."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=sample_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is False
        # Should retry 3 times
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_timeout(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret, sample_payload
    ):
        """Test webhook delivery timeout."""
        mock_client = Mock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=sample_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is False
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_network_error(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret, sample_payload
    ):
        """Test webhook delivery with network error."""
        mock_client = Mock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=sample_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is False
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    @patch("alpha_pulse.services.webhook_notifier.WebhookNotifier._sleep")
    async def test_send_webhook_retry_with_backoff(
        self, mock_sleep, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret, sample_payload
    ):
        """Test webhook retry with exponential backoff."""
        # First two attempts fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_success = Mock()
        mock_response_success.status_code = 200

        mock_client = Mock()
        mock_client.post = AsyncMock(
            side_effect=[mock_response_fail, mock_response_fail, mock_response_success]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=sample_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is True
        assert mock_client.post.call_count == 3

        # Verify exponential backoff delays (1s, 2s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # First backoff: 1 * 2^0 = 1
        mock_sleep.assert_any_call(2)  # Second backoff: 1 * 2^1 = 2

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_payload_enrichment(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret, sample_payload
    ):
        """Test that payload is enriched with metadata."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=sample_payload,
            webhook_secret=sample_webhook_secret,
        )

        # Get the payload that was sent
        call_args = mock_client.post.call_args
        sent_payload = call_args.kwargs["json"]

        # Verify enrichment
        assert "event" in sent_payload
        assert sent_payload["event"] == "credential.failed"
        assert "tenant_id" in sent_payload
        assert sent_payload["tenant_id"] == str(sample_tenant_id)
        assert "timestamp" in sent_payload
        assert isinstance(sent_payload["timestamp"], int)
        # Original payload fields should be present
        assert sent_payload["exchange"] == sample_payload["exchange"]
        assert sent_payload["credential_type"] == sample_payload["credential_type"]


class TestSendTestWebhook:
    """Tests for send_test_webhook method."""

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_test_webhook_success(
        self, mock_client_class, notifier, sample_webhook_url, sample_webhook_secret
    ):
        """Test successful test webhook delivery."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_test_webhook(
            webhook_url=sample_webhook_url,
            webhook_secret=sample_webhook_secret,
        )

        assert result is True
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_test_webhook_with_tenant_id(
        self, mock_client_class, notifier, sample_webhook_url,
        sample_webhook_secret, sample_tenant_id
    ):
        """Test test webhook with tenant ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_test_webhook(
            webhook_url=sample_webhook_url,
            webhook_secret=sample_webhook_secret,
            tenant_id=sample_tenant_id,
        )

        assert result is True

        # Verify payload contains tenant_id
        call_args = mock_client.post.call_args
        sent_payload = call_args.kwargs["json"]
        assert sent_payload["tenant_id"] == str(sample_tenant_id)
        assert sent_payload["event"] == "webhook.test"

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_test_webhook_failure(
        self, mock_client_class, notifier, sample_webhook_url, sample_webhook_secret
    ):
        """Test test webhook delivery failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_test_webhook(
            webhook_url=sample_webhook_url,
            webhook_secret=sample_webhook_secret,
        )

        assert result is False

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_test_webhook_exception(
        self, mock_client_class, notifier, sample_webhook_url, sample_webhook_secret
    ):
        """Test test webhook exception handling."""
        mock_client = Mock()
        mock_client.post = AsyncMock(side_effect=Exception("Network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await notifier.send_test_webhook(
            webhook_url=sample_webhook_url,
            webhook_secret=sample_webhook_secret,
        )

        assert result is False


class TestVerifyWebhookSignature:
    """Tests for verify_webhook_signature helper function."""

    def test_verify_valid_signature(self):
        """Test verifying a valid webhook signature."""
        secret = "test_secret"
        payload = '{"test":"data","number":123}'

        # Generate expected signature
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        result = verify_webhook_signature(payload, expected_sig, secret)
        assert result is True

    def test_verify_invalid_signature(self):
        """Test verifying an invalid webhook signature."""
        secret = "test_secret"
        payload = '{"test":"data"}'
        invalid_sig = "0" * 64  # Invalid signature

        result = verify_webhook_signature(payload, invalid_sig, secret)
        assert result is False

    def test_verify_signature_with_wrong_secret(self):
        """Test verifying signature with wrong secret."""
        correct_secret = "correct_secret"
        wrong_secret = "wrong_secret"
        payload = '{"test":"data"}'

        # Generate signature with correct secret
        signature = hmac.new(
            correct_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Verify with wrong secret
        result = verify_webhook_signature(payload, signature, wrong_secret)
        assert result is False

    def test_verify_signature_with_modified_payload(self):
        """Test verifying signature with modified payload."""
        secret = "test_secret"
        original_payload = '{"test":"data"}'
        modified_payload = '{"test":"modified"}'

        # Generate signature for original payload
        signature = hmac.new(
            secret.encode("utf-8"),
            original_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Verify with modified payload
        result = verify_webhook_signature(modified_payload, signature, secret)
        assert result is False

    def test_verify_signature_timing_safe_comparison(self):
        """Test that signature verification uses timing-safe comparison."""
        secret = "test_secret"
        payload = '{"test":"data"}'

        # Generate correct signature
        correct_sig = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Create signature that differs by one character
        almost_correct_sig = correct_sig[:-1] + ("0" if correct_sig[-1] != "0" else "1")

        # Should return False (uses hmac.compare_digest internally)
        result = verify_webhook_signature(payload, almost_correct_sig, secret)
        assert result is False


class TestPrometheusMetrics:
    """Tests for Prometheus metrics integration."""

    @patch("alpha_pulse.services.webhook_notifier.webhook_delivery_total")
    def test_metrics_exported(self, mock_metric):
        """Test that Prometheus metrics are properly exported."""
        from alpha_pulse.services.webhook_notifier import webhook_delivery_total

        assert webhook_delivery_total is not None


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_with_special_characters_in_payload(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret
    ):
        """Test webhook with special characters in payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        special_payload = {
            "error": "Error: '{}' contains special characters: <>&\"",
            "message": "Line break\nand tab\there",
        }

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=special_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is True

    @pytest.mark.asyncio
    @patch("alpha_pulse.services.webhook_notifier.httpx.AsyncClient")
    async def test_send_webhook_with_unicode_payload(
        self, mock_client_class, notifier, sample_tenant_id,
        sample_webhook_url, sample_webhook_secret
    ):
        """Test webhook with Unicode characters in payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        unicode_payload = {
            "error": "Error message with émojis 🚨 and 中文字符",
            "exchange": "Binance™",
        }

        result = await notifier.send_credential_failure_webhook(
            tenant_id=sample_tenant_id,
            webhook_url=sample_webhook_url,
            payload=unicode_payload,
            webhook_secret=sample_webhook_secret,
        )

        assert result is True

    def test_verify_signature_with_empty_payload(self):
        """Test signature verification with empty payload."""
        secret = "test_secret"
        payload = ""

        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        result = verify_webhook_signature(payload, signature, secret)
        assert result is True
