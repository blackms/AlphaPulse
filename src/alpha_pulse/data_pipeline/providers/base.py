"""
Base provider implementation with common functionality.
"""
from typing import Dict, List, Optional, Any
import aiohttp
from loguru import logger
import asyncio
from datetime import datetime, timedelta

from ..interfaces import DataFetchError


def retry_on_error(retries: int = 3, delay: float = 1.0):
    """
    Retry decorator for API requests.

    Args:
        retries: Maximum number of retries
        delay: Delay between retries in seconds
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Attempt {attempt + 1} failed, "
                            f"retrying in {wait_time}s: {str(e)}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {retries} attempts failed: {str(e)}"
                        )
                        raise last_error
                        
        return wrapper
    return decorator


class CacheMixin:
    """Mixin providing caching functionality."""
    
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize cache.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self._cache = {}
        self._cache_ttl = cache_ttl
        self._cache_timestamps = {}

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp = self._cache_timestamps.get(key)
            if timestamp:
                age = (datetime.now() - timestamp).total_seconds()
                if age < self._cache_ttl:
                    logger.debug(f"Cache hit for {key}")
                    return self._cache[key]
                else:
                    logger.debug(f"Cache expired for {key}")
                    del self._cache[key]
                    del self._cache_timestamps[key]
        return None

    def _set_in_cache(self, key: str, value: Any):
        """Set value in cache with current timestamp."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()
        logger.debug(f"Cached value for {key}")


class BaseDataProvider:
    """Base class for all data providers."""

    def __init__(
        self,
        provider_name: str,
        provider_type: str,
        base_url: Optional[str],
        default_headers: Optional[Dict[str, str]] = None,
        request_timeout: int = 30,
        tcp_connector_limit: int = 100
    ):
        """
        Initialize base provider.

        Args:
            provider_name: Name of the provider
            provider_type: Type of data provided
            base_url: Base URL for API requests
            default_headers: Default headers to include in requests
            request_timeout: Request timeout in seconds
            tcp_connector_limit: Maximum number of concurrent connections
        """
        self.provider_name = provider_name
        self.provider_type = provider_type
        self.base_url = base_url
        self._default_headers = default_headers or {}
        self._request_timeout = request_timeout
        self._tcp_connector_limit = tcp_connector_limit
        self._session = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    logger.debug(f"Creating new session for {self.provider_name}")
                    connector = aiohttp.TCPConnector(
                        limit=self._tcp_connector_limit,
                        enable_cleanup_closed=True
                    )
                    timeout = aiohttp.ClientTimeout(total=self._request_timeout)
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers=self._default_headers
                    )
        return self._session

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return endpoint

    async def _execute_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> aiohttp.ClientResponse:
        """
        Execute HTTP request.

        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            headers: Request headers
            data: Request body
            timeout: Request timeout in seconds (overrides default)

        Returns:
            aiohttp.ClientResponse
        """
        url = self._build_url(endpoint)
        request_headers = {**self._default_headers, **(headers or {})}
        
        try:
            session = await self._get_session()
            logger.debug(f"Executing {method} request to {url}")
            
            timeout_obj = aiohttp.ClientTimeout(
                total=timeout or self._request_timeout
            )
            
            async with session.request(
                method=method,
                url=url,
                params=params,
                headers=request_headers,
                json=data,
                timeout=timeout_obj,
                raise_for_status=True
            ) as response:
                # Log API errors
                if response.status >= 400:
                    response_text = await response.text()
                    logger.error(
                        f"API Error: Status {response.status} "
                        f"URL: {url} "
                        f"Headers: {request_headers} "
                        f"Params: {params}\n"
                        f"Response: {response_text}"
                    )
                    response.raise_for_status()
                
                # Ensure response is fully read
                await response.read()
                return response
                
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {str(e)}")
            raise DataFetchError(f"Failed to fetch data: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout or self._request_timeout}s: {url}")
            raise DataFetchError(f"Request timed out: {url}")
        except Exception as e:
            logger.exception(f"Unexpected error during request: {str(e)}")
            raise

    async def _process_response(self, response: aiohttp.ClientResponse) -> Any:
        """
        Process API response.

        Args:
            response: aiohttp response object

        Returns:
            Processed response data
        """
        try:
            # Response should already be read in _execute_request
            return await response.json(content_type=None)
        except Exception as e:
            logger.exception(f"Error processing response: {str(e)}")
            raise DataFetchError(f"Failed to process response: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            try:
                if not self._session.closed:
                    logger.debug(f"Closing session for {self.provider_name}")
                    await self._session.close()
                    
                # Wait for all connections to close
                if hasattr(self._session, '_connector'):
                    await self._session._connector.close()
                    
            except Exception as e:
                logger.exception(f"Error closing session: {str(e)}")
            finally:
                self._session = None