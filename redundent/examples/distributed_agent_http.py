"""
Comprehensive HTTPX patterns for distributed agent communication.

This module provides robust HTTP client architecture patterns for agent-to-agent
communication with strict timeout requirements (2-3 seconds) and paramount reliability.

Key Features:
- AsyncClient lifecycle management with proper resource cleanup
- Granular timeout configuration (connect, read, write, pool)
- Comprehensive error handling with retry strategies
- Connection pooling and performance optimization
- Integration with FastAPI background tasks

Source: Based on HTTPX documentation from async_support_httpx.md, timeouts_httpx.md,
clients_httpx.md, exceptions_httpx.md, http_2_support_httpx.md, and transports_httpx.md
"""

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Callable, AsyncGenerator
import time
import httpx
from httpx import AsyncClient, Timeout, Limits, Response, RequestError, TimeoutException
from httpx import ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout, NetworkError


# Configure logging for monitoring
logger = logging.getLogger(__name__)


class AgentHTTPClientConfig:
    """Configuration for agent HTTP communication with strict timeout requirements."""

    def __init__(
        self,
        connect_timeout: float = 2.0,
        read_timeout: float = 2.0,
        write_timeout: float = 2.0,
        pool_timeout: float = 1.0,
        total_timeout: float = 3.0,
        max_connections: int = 20,
        max_keepalive_connections: int = 10,
        keepalive_expiry: float = 5.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        retry_jitter: bool = True,
        enable_http2: bool = True
    ):
        """
        Initialize agent HTTP client configuration.

        Args:
            connect_timeout: Maximum time to establish connection (seconds)
            read_timeout: Maximum time to read response data (seconds)
            write_timeout: Maximum time to send request data (seconds)
            pool_timeout: Maximum time to acquire connection from pool (seconds)
            total_timeout: Overall request timeout (seconds)
            max_connections: Maximum total connections in pool
            max_keepalive_connections: Maximum persistent connections
            keepalive_expiry: Connection keep-alive timeout (seconds)
            max_retries: Maximum retry attempts for failed requests
            retry_backoff_factor: Base factor for exponential backoff
            retry_jitter: Whether to add jitter to retry delays
            enable_http2: Enable HTTP/2 support for multiplexing
        """
        # Timeout configuration based on timeouts_httpx.md
        self.timeout = Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout
        )
        self.total_timeout = total_timeout

        # Connection limits for performance based on clients_httpx.md
        self.limits = Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry
        )

        # Retry configuration for resilience
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_jitter = retry_jitter

        # Performance optimization
        self.enable_http2 = enable_http2


class CircuitBreaker:
    """Circuit breaker pattern for agent communication resilience."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = RequestError
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __enter__(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            self.on_success()
        elif issubclass(exc_type, self.expected_exception):
            # Expected failure
            self.on_failure()

        return False

    def on_success(self):
        """Handle successful request."""
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class AgentHTTPClient:
    """
    High-performance HTTP client for distributed agent communication.

    Features:
    - Proper AsyncClient lifecycle management
    - Granular timeout configuration
    - Automatic retry with exponential backoff and jitter
    - Circuit breaker pattern for fault tolerance
    - Connection pooling optimization
    - HTTP/2 support for multiplexing

    Based on HTTPX documentation patterns from async_support_httpx.md and clients_httpx.md.
    """

    def __init__(self, config: AgentHTTPClientConfig, base_url: Optional[str] = None):
        """
        Initialize the agent HTTP client.

        Args:
            config: Client configuration
            base_url: Base URL for agent endpoints
        """
        self.config = config
        self.base_url = base_url
        self._client: Optional[AsyncClient] = None
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._is_closed = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.close()

    async def start(self):
        """
        Initialize the HTTP client with optimized configuration.

        Based on async_support_httpx.md patterns for efficient resource usage.
        """
        if self._client is not None:
            return

        # Create transport with retry support from transports_httpx.md
        transport = httpx.AsyncHTTPTransport(
            retries=1,  # Transport-level retries for connection errors
            limits=self.config.limits
        )

        # Initialize AsyncClient with comprehensive configuration
        self._client = AsyncClient(
            timeout=self.config.timeout,
            limits=self.config.limits,
            transport=transport,
            base_url=self.base_url,
            http2=self.config.enable_http2,
            # Trust environment variables for proxy configuration
            trust_env=True
        )

        logger.info(f"Agent HTTP client started with base_url={self.base_url}")

    async def close(self):
        """
        Properly close the HTTP client and clean up resources.

        Critical for preventing resource leaks as noted in async_support_httpx.md.
        """
        if self._client is not None and not self._is_closed:
            await self._client.aclose()
            self._client = None
            self._is_closed = True
            logger.info("Agent HTTP client closed")

    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """Get or create circuit breaker for URL domain."""
        try:
            domain = httpx.URL(url).host
        except Exception:
            domain = "default"

        if domain not in self._circuit_breakers:
            self._circuit_breakers[domain] = CircuitBreaker()

        return self._circuit_breakers[domain]

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff with optional jitter.

        Args:
            attempt: Current retry attempt (0-based)

        Returns:
            Delay in seconds
        """
        base_delay = self.config.retry_backoff_factor * (2 ** attempt)

        if self.config.retry_jitter:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, 0.1 * base_delay)
            return base_delay + jitter

        return base_delay

    async def request_with_resilience(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Response:
        """
        Make HTTP request with comprehensive error handling and resilience patterns.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response

        Raises:
            RequestError: After all retries exhausted
            TimeoutException: For timeout scenarios

        Based on exceptions_httpx.md error hierarchy and resilience patterns.
        """
        if self._client is None:
            raise RuntimeError("Client not started. Use async context manager or call start()")

        circuit_breaker = self._get_circuit_breaker(url)
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                with circuit_breaker:
                    # Apply total timeout at request level
                    request_timeout = kwargs.get('timeout', self.config.total_timeout)
                    kwargs['timeout'] = request_timeout

                    response = await self._client.request(method, url, **kwargs)

                    # Check HTTP status and raise for error codes if needed
                    response.raise_for_status()

                    logger.debug(
                        f"Agent request succeeded: {method} {url} "
                        f"-> {response.status_code} (attempt {attempt + 1})"
                    )

                    return response

            except (ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout) as e:
                # Timeout exceptions from exceptions_httpx.md hierarchy
                last_exception = e
                logger.warning(
                    f"Agent request timeout: {method} {url} "
                    f"-> {type(e).__name__} (attempt {attempt + 1})"
                )

            except (NetworkError, RequestError) as e:
                # Network and request errors from exceptions_httpx.md
                last_exception = e
                logger.warning(
                    f"Agent request failed: {method} {url} "
                    f"-> {type(e).__name__}: {e} (attempt {attempt + 1})"
                )

            except Exception as e:
                # Circuit breaker open or other unexpected errors
                last_exception = e
                logger.error(
                    f"Agent request error: {method} {url} "
                    f"-> {type(e).__name__}: {e} (attempt {attempt + 1})"
                )
                # Don't retry on unexpected errors
                break

            # Calculate backoff delay for retry
            if attempt < self.config.max_retries:
                delay = self._calculate_backoff(attempt)
                logger.debug(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"Agent request failed after {self.config.max_retries + 1} attempts")
        if last_exception:
            raise last_exception
        raise RequestError("Request failed after all retries")

    async def get(self, url: str, **kwargs) -> Response:
        """GET request with resilience."""
        return await self.request_with_resilience("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Response:
        """POST request with resilience."""
        return await self.request_with_resilience("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> Response:
        """PUT request with resilience."""
        return await self.request_with_resilience("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> Response:
        """DELETE request with resilience."""
        return await self.request_with_resilience("DELETE", url, **kwargs)


# Global client instance for FastAPI integration
_global_client: Optional[AgentHTTPClient] = None


@asynccontextmanager
async def get_agent_http_client(
    config: Optional[AgentHTTPClientConfig] = None,
    base_url: Optional[str] = None
) -> AsyncGenerator[AgentHTTPClient, None]:
    """
    Context manager for agent HTTP client with proper lifecycle management.

    Args:
        config: Client configuration (uses default if None)
        base_url: Base URL for requests

    Yields:
        Configured AgentHTTPClient instance

    Example usage based on async_support_httpx.md patterns:
        async with get_agent_http_client() as client:
            response = await client.get("/health")
    """
    if config is None:
        config = AgentHTTPClientConfig()

    client = AgentHTTPClient(config, base_url)
    try:
        await client.start()
        yield client
    finally:
        await client.close()


async def initialize_global_client(
    config: Optional[AgentHTTPClientConfig] = None,
    base_url: Optional[str] = None
):
    """
    Initialize global HTTP client for FastAPI background tasks.

    This should be called during FastAPI startup to ensure proper connection pooling
    as recommended in clients_httpx.md to avoid instantiating multiple clients.
    """
    global _global_client

    if _global_client is not None:
        await _global_client.close()

    if config is None:
        config = AgentHTTPClientConfig()

    _global_client = AgentHTTPClient(config, base_url)
    await _global_client.start()


async def close_global_client():
    """Close global HTTP client during FastAPI shutdown."""
    global _global_client

    if _global_client is not None:
        await _global_client.close()
        _global_client = None


def get_global_client() -> AgentHTTPClient:
    """
    Get the global HTTP client instance.

    Raises:
        RuntimeError: If global client is not initialized
    """
    if _global_client is None:
        raise RuntimeError(
            "Global HTTP client not initialized. "
            "Call initialize_global_client() during FastAPI startup."
        )
    return _global_client


# Example FastAPI integration
async def agent_communication_task(target_url: str, data: Dict[str, Any]):
    """
    Example FastAPI background task using global HTTP client.

    This pattern ensures connection reuse and proper resource management
    as recommended in clients_httpx.md and async_support_httpx.md.
    """
    try:
        client = get_global_client()
        response = await client.post(target_url, json=data)

        logger.info(
            f"Agent communication successful: {target_url} "
            f"-> {response.status_code} ({response.http_version})"
        )

        return response.json()

    except Exception as e:
        logger.error(f"Agent communication failed: {target_url} -> {e}")
        raise


if __name__ == "__main__":
    # Example usage demonstrating the patterns
    async def example_usage():
        """Demonstrate agent HTTP communication patterns."""

        # Configuration for strict 2-3 second requirements
        config = AgentHTTPClientConfig(
            connect_timeout=1.0,
            read_timeout=2.0,
            write_timeout=1.0,
            pool_timeout=0.5,
            total_timeout=3.0,
            max_retries=2,
            enable_http2=True
        )

        # Context manager usage (recommended)
        async with get_agent_http_client(config) as client:
            try:
                response = await client.get("https://httpbin.org/delay/1")
                print(f"Success: {response.status_code} ({response.http_version})")

            except TimeoutException as e:
                print(f"Timeout: {e}")

            except RequestError as e:
                print(f"Request failed: {e}")

    # Run example
    asyncio.run(example_usage())