"""
Credential validator for exchange API keys using CCXT.

Validates credentials by making test API calls to exchanges before storing in Vault.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional
import ccxt
from loguru import logger


@dataclass
class ValidationResult:
    """Result of credential validation."""

    valid: bool
    credential_type: Optional[str] = None  # 'trading' or 'readonly'
    exchange_account_id: Optional[str] = None
    error: Optional[str] = None


class CredentialValidator:
    """Validates exchange API credentials using CCXT test calls."""

    def __init__(self, timeout: int = 10):
        """
        Initialize credential validator.

        Args:
            timeout: Timeout in seconds for validation API calls
        """
        self.timeout = timeout

    async def validate(
        self,
        exchange: str,
        api_key: str,
        secret: str,
        passphrase: Optional[str] = None,
        testnet: bool = False,
    ) -> ValidationResult:
        """
        Validate exchange credentials by making test API calls.

        Args:
            exchange: Exchange name (e.g., 'binance', 'coinbase')
            api_key: API key
            secret: API secret
            passphrase: Optional passphrase (required for some exchanges)
            testnet: Whether to use testnet

        Returns:
            ValidationResult with validation status and detected permissions
        """
        try:
            # Get CCXT exchange class
            if not hasattr(ccxt, exchange.lower()):
                return ValidationResult(
                    valid=False, error=f"Unsupported exchange: {exchange}"
                )

            exchange_class = getattr(ccxt, exchange.lower())

            # Initialize exchange client
            client_config = {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "timeout": self.timeout * 1000,  # CCXT uses milliseconds
            }

            # Add passphrase if provided (required for Coinbase, KuCoin, etc.)
            if passphrase:
                client_config["password"] = passphrase

            client = exchange_class(client_config)

            # Enable sandbox/testnet mode if requested
            if testnet and hasattr(client, "set_sandbox_mode"):
                client.set_sandbox_mode(True)

            logger.info(
                f"Validating credentials for {exchange} (testnet={testnet})..."
            )

            # Test 1: Read permission (fetch_balance)
            # This is the minimal permission required
            try:
                balance = await asyncio.wait_for(
                    asyncio.to_thread(client.fetch_balance), timeout=self.timeout
                )
                logger.debug(f"Successfully fetched balance from {exchange}")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout fetching balance from {exchange} after {self.timeout}s"
                )
                return ValidationResult(
                    valid=False,
                    error=f"Validation timeout ({self.timeout}s). Exchange may be slow or unreachable.",
                )
            except ccxt.AuthenticationError as e:
                logger.warning(f"Authentication failed for {exchange}: {e}")
                return ValidationResult(valid=False, error="Invalid API key or secret")
            except ccxt.PermissionDenied as e:
                logger.warning(f"Permission denied for {exchange}: {e}")
                return ValidationResult(
                    valid=False, error="Insufficient API permissions (cannot read balance)"
                )

            # Test 2: Trading permission (optional, best-effort)
            # Try to determine if credentials have trading permissions
            has_trading = await self._test_trading_permission(client, exchange)

            # Get account ID if available
            account_id = None
            if hasattr(client, "uid") and client.uid:
                account_id = client.uid
            elif hasattr(balance, "info") and isinstance(balance.get("info"), dict):
                # Try to extract account ID from balance response
                info = balance.get("info", {})
                account_id = info.get("accountId") or info.get("uid") or info.get("id")

            credential_type = "trading" if has_trading else "readonly"

            logger.info(
                f"Credentials validated for {exchange}: {credential_type} permissions"
            )

            return ValidationResult(
                valid=True,
                credential_type=credential_type,
                exchange_account_id=str(account_id) if account_id else None,
            )

        except ccxt.NetworkError as e:
            logger.error(f"Network error validating {exchange}: {e}")
            return ValidationResult(
                valid=False, error=f"Network error: Unable to connect to {exchange}"
            )
        except Exception as e:
            logger.error(f"Unexpected error validating {exchange}: {e}")
            return ValidationResult(valid=False, error=f"Validation error: {str(e)}")

    async def _test_trading_permission(self, client, exchange: str) -> bool:
        """
        Test if credentials have trading permissions (best-effort).

        Args:
            client: CCXT exchange client
            exchange: Exchange name

        Returns:
            True if trading permissions detected, False otherwise
        """
        try:
            # Strategy 1: Check if create_order method exists and is callable
            if not hasattr(client, "create_order"):
                logger.debug(f"{exchange} does not support create_order")
                return False

            # Strategy 2: Use create_test_order if available (Binance, etc.)
            if hasattr(client, "create_test_order"):
                try:
                    # Try to create a test order that won't execute
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            client.create_test_order,
                            symbol="BTC/USDT",
                            type="limit",
                            side="buy",
                            amount=0.001,
                            price=1000,  # Way below market to avoid execution
                        ),
                        timeout=self.timeout,
                    )
                    logger.debug(f"{exchange} test order succeeded - trading permissions detected")
                    return True
                except ccxt.InsufficientPermissions:
                    logger.debug(f"{exchange} test order failed - readonly permissions")
                    return False
                except Exception as e:
                    # Test order might fail for other reasons (invalid symbol, etc.)
                    # Don't treat as conclusive evidence of no trading permissions
                    logger.debug(f"{exchange} test order error (inconclusive): {e}")

            # Strategy 3: Check account permissions via API (exchange-specific)
            # This varies by exchange, so we're conservative here
            if hasattr(client, "privateGetAccount"):
                try:
                    account_info = await asyncio.wait_for(
                        asyncio.to_thread(client.privateGetAccount), timeout=self.timeout
                    )
                    # Check for trading-related fields in account info
                    if isinstance(account_info, dict):
                        # Binance: check canTrade field
                        if "canTrade" in account_info:
                            return bool(account_info.get("canTrade"))
                        # Coinbase: check permissions array
                        if "permissions" in account_info:
                            perms = account_info.get("permissions", [])
                            return "trade" in perms or "trading" in perms
                except Exception as e:
                    logger.debug(f"Could not fetch account permissions for {exchange}: {e}")

            # Default: Assume readonly if we can't confirm trading permissions
            logger.debug(f"{exchange} trading permissions unclear - defaulting to readonly")
            return False

        except Exception as e:
            logger.warning(f"Error testing trading permission for {exchange}: {e}")
            return False
