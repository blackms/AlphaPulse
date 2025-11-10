"""
Unit tests for CredentialValidator.

Tests credential validation logic using mocked CCXT exchange clients.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import UUID

import ccxt

from alpha_pulse.services.credentials.validator import (
    CredentialValidator,
    ValidationResult,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult(
            valid=True,
            credential_type="trading",
            exchange_account_id="test-account-123",
        )
        assert result.valid is True
        assert result.credential_type == "trading"
        assert result.exchange_account_id == "test-account-123"
        assert result.error is None

    def test_validation_result_failure(self):
        """Test failed validation result."""
        result = ValidationResult(valid=False, error="Invalid API key")
        assert result.valid is False
        assert result.credential_type is None
        assert result.exchange_account_id is None
        assert result.error == "Invalid API key"


class TestCredentialValidator:
    """Tests for CredentialValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a CredentialValidator instance."""
        return CredentialValidator(timeout=5)

    @pytest.fixture
    def mock_exchange_class(self):
        """Create a mock CCXT exchange class."""
        mock_class = Mock()
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        return mock_class, mock_instance

    @pytest.mark.asyncio
    async def test_validate_unsupported_exchange(self, validator):
        """Test validation with unsupported exchange."""
        result = await validator.validate(
            exchange="unsupported_exchange",
            api_key="test_key",
            secret="test_secret",
        )

        assert result.valid is False
        assert "Unsupported exchange" in result.error
        assert result.credential_type is None

    @pytest.mark.asyncio
    async def test_validate_success_trading_permissions(
        self, validator, mock_exchange_class
    ):
        """Test successful validation with trading permissions."""
        mock_class, mock_instance = mock_exchange_class

        # Mock successful balance fetch
        mock_balance = {
            "total": {"BTC": 1.0, "USDT": 10000.0},
            "info": {"accountId": "test-account-123"},
        }
        mock_instance.fetch_balance = Mock(return_value=mock_balance)

        # Mock successful test order (indicates trading permissions)
        mock_instance.create_test_order = Mock(return_value={"id": "test-order"})

        # Mock uid attribute
        mock_instance.uid = None

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="test_key",
                secret="test_secret",
            )

        assert result.valid is True
        assert result.credential_type == "trading"
        assert result.exchange_account_id == "test-account-123"
        assert result.error is None

        # Verify API calls
        mock_instance.fetch_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_success_readonly_permissions(
        self, validator, mock_exchange_class
    ):
        """Test successful validation with readonly permissions."""
        mock_class, mock_instance = mock_exchange_class

        # Mock successful balance fetch
        mock_balance = {
            "total": {"BTC": 1.0, "USDT": 10000.0},
            "info": {"uid": "test-uid-456"},
        }
        mock_instance.fetch_balance = Mock(return_value=mock_balance)

        # Mock test order failure (no trading permissions)
        mock_instance.create_test_order = Mock(
            side_effect=ccxt.PermissionDenied("No trading permissions")
        )

        # Mock uid attribute
        mock_instance.uid = None

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="test_key",
                secret="test_secret",
            )

        assert result.valid is True
        assert result.credential_type == "readonly"
        assert result.exchange_account_id == "test-uid-456"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_validate_authentication_error(self, validator, mock_exchange_class):
        """Test validation with authentication error."""
        mock_class, mock_instance = mock_exchange_class

        # Mock authentication failure
        mock_instance.fetch_balance = Mock(
            side_effect=ccxt.AuthenticationError("Invalid API key")
        )

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="invalid_key",
                secret="invalid_secret",
            )

        assert result.valid is False
        assert result.error == "Invalid API key or secret"
        assert result.credential_type is None

    @pytest.mark.asyncio
    async def test_validate_permission_denied(self, validator, mock_exchange_class):
        """Test validation with permission denied error."""
        mock_class, mock_instance = mock_exchange_class

        # Mock permission denied
        mock_instance.fetch_balance = Mock(
            side_effect=ccxt.PermissionDenied("Insufficient permissions")
        )

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="test_key",
                secret="test_secret",
            )

        assert result.valid is False
        assert "Insufficient API permissions" in result.error
        assert result.credential_type is None

    @pytest.mark.asyncio
    async def test_validate_network_error(self, validator, mock_exchange_class):
        """Test validation with network error."""
        mock_class, mock_instance = mock_exchange_class

        # Mock network error
        mock_instance.fetch_balance = Mock(
            side_effect=ccxt.NetworkError("Connection timeout")
        )

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="test_key",
                secret="test_secret",
            )

        assert result.valid is False
        assert "Network error" in result.error
        assert "binance" in result.error
        assert result.credential_type is None

    @pytest.mark.asyncio
    async def test_validate_timeout(self, validator, mock_exchange_class):
        """Test validation with timeout."""
        mock_class, mock_instance = mock_exchange_class

        # Mock slow fetch_balance that times out
        async def slow_fetch():
            await asyncio.sleep(10)  # Longer than validator timeout (5s)
            return {}

        mock_instance.fetch_balance = Mock(side_effect=slow_fetch)

        with patch.object(ccxt, "binance", mock_class):
            # Use asyncio.wait_for to simulate timeout
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await validator.validate(
                    exchange="binance",
                    api_key="test_key",
                    secret="test_secret",
                )

        assert result.valid is False
        assert "timeout" in result.error.lower()
        assert result.credential_type is None

    @pytest.mark.asyncio
    async def test_validate_with_passphrase(self, validator, mock_exchange_class):
        """Test validation with passphrase (e.g., Coinbase, KuCoin)."""
        mock_class, mock_instance = mock_exchange_class

        # Mock successful balance fetch
        mock_balance = {"total": {"BTC": 1.0}, "info": {}}
        mock_instance.fetch_balance = Mock(return_value=mock_balance)
        mock_instance.create_test_order = Mock(
            side_effect=ccxt.PermissionDenied("No trading")
        )
        mock_instance.uid = None

        with patch.object(ccxt, "coinbase", mock_class):
            result = await validator.validate(
                exchange="coinbase",
                api_key="test_key",
                secret="test_secret",
                passphrase="test_passphrase",
            )

        assert result.valid is True
        # Verify passphrase was passed to exchange config
        call_config = mock_class.call_args[0][0]
        assert call_config["password"] == "test_passphrase"

    @pytest.mark.asyncio
    async def test_validate_testnet_mode(self, validator, mock_exchange_class):
        """Test validation with testnet mode enabled."""
        mock_class, mock_instance = mock_exchange_class

        # Mock set_sandbox_mode method
        mock_instance.set_sandbox_mode = Mock()

        # Mock successful balance fetch
        mock_balance = {"total": {"BTC": 1.0}, "info": {}}
        mock_instance.fetch_balance = Mock(return_value=mock_balance)
        mock_instance.create_test_order = Mock(
            side_effect=ccxt.PermissionDenied("No trading")
        )
        mock_instance.uid = None

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="test_key",
                secret="test_secret",
                testnet=True,
            )

        assert result.valid is True
        # Verify sandbox mode was enabled
        mock_instance.set_sandbox_mode.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_validate_unexpected_error(self, validator, mock_exchange_class):
        """Test validation with unexpected error."""
        mock_class, mock_instance = mock_exchange_class

        # Mock unexpected exception
        mock_instance.fetch_balance = Mock(
            side_effect=ValueError("Unexpected error")
        )

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="test_key",
                secret="test_secret",
            )

        assert result.valid is False
        assert "Validation error" in result.error
        assert "Unexpected error" in result.error
        assert result.credential_type is None

    @pytest.mark.asyncio
    async def test_test_trading_permission_no_create_order_method(
        self, validator, mock_exchange_class
    ):
        """Test _test_trading_permission when exchange doesn't support create_order."""
        mock_class, mock_instance = mock_exchange_class

        # Remove create_order method
        del mock_instance.create_order

        has_trading = await validator._test_trading_permission(
            mock_instance, "test_exchange"
        )

        assert has_trading is False

    @pytest.mark.asyncio
    async def test_test_trading_permission_with_account_info_can_trade(
        self, validator, mock_exchange_class
    ):
        """Test _test_trading_permission using account info with canTrade field."""
        mock_class, mock_instance = mock_exchange_class

        # Mock privateGetAccount with canTrade field
        mock_instance.privateGetAccount = Mock(
            return_value={"canTrade": True, "accountType": "SPOT"}
        )

        # No create_test_order method
        if hasattr(mock_instance, "create_test_order"):
            del mock_instance.create_test_order

        has_trading = await validator._test_trading_permission(
            mock_instance, "binance"
        )

        # Should detect trading permissions from account info
        assert has_trading is True

    @pytest.mark.asyncio
    async def test_test_trading_permission_with_permissions_array(
        self, validator, mock_exchange_class
    ):
        """Test _test_trading_permission using permissions array (Coinbase style)."""
        mock_class, mock_instance = mock_exchange_class

        # Mock privateGetAccount with permissions array
        mock_instance.privateGetAccount = Mock(
            return_value={"permissions": ["read", "trade", "withdraw"]}
        )

        # No create_test_order method
        if hasattr(mock_instance, "create_test_order"):
            del mock_instance.create_test_order

        has_trading = await validator._test_trading_permission(
            mock_instance, "coinbase"
        )

        # Should detect trading permissions from permissions array
        assert has_trading is True

    @pytest.mark.asyncio
    async def test_test_trading_permission_error_handling(
        self, validator, mock_exchange_class
    ):
        """Test _test_trading_permission with error during test."""
        mock_class, mock_instance = mock_exchange_class

        # Mock create_test_order with generic error
        mock_instance.create_test_order = Mock(
            side_effect=ValueError("Unexpected error")
        )

        has_trading = await validator._test_trading_permission(
            mock_instance, "binance"
        )

        # Should default to False on error
        assert has_trading is False

    @pytest.mark.asyncio
    async def test_validate_exchange_account_id_extraction_uid(
        self, validator, mock_exchange_class
    ):
        """Test account ID extraction from client.uid attribute."""
        mock_class, mock_instance = mock_exchange_class

        # Set uid attribute on client
        mock_instance.uid = "client-uid-789"

        # Mock successful balance fetch
        mock_balance = {"total": {"BTC": 1.0}, "info": {}}
        mock_instance.fetch_balance = Mock(return_value=mock_balance)
        mock_instance.create_test_order = Mock(
            side_effect=ccxt.PermissionDenied("No trading")
        )

        with patch.object(ccxt, "binance", mock_class):
            result = await validator.validate(
                exchange="binance",
                api_key="test_key",
                secret="test_secret",
            )

        assert result.valid is True
        assert result.exchange_account_id == "client-uid-789"

    @pytest.mark.asyncio
    async def test_validate_config_parameters(self, validator, mock_exchange_class):
        """Test that exchange client is configured with correct parameters."""
        mock_class, mock_instance = mock_exchange_class

        # Mock successful balance fetch
        mock_balance = {"total": {}, "info": {}}
        mock_instance.fetch_balance = Mock(return_value=mock_balance)
        mock_instance.create_test_order = Mock(
            side_effect=ccxt.PermissionDenied("No trading")
        )
        mock_instance.uid = None

        with patch.object(ccxt, "binance", mock_class):
            await validator.validate(
                exchange="binance",
                api_key="test_key_123",
                secret="test_secret_456",
            )

        # Verify exchange was initialized with correct config
        call_config = mock_class.call_args[0][0]
        assert call_config["apiKey"] == "test_key_123"
        assert call_config["secret"] == "test_secret_456"
        assert call_config["enableRateLimit"] is True
        assert call_config["timeout"] == validator.timeout * 1000  # milliseconds
