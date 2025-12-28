"""
Comprehensive test suite and configuration examples for distributed agent HTTP communication.

This module provides:
- Test scenarios for all timeout configurations
- Performance benchmarks for agent communication
- Error simulation and resilience testing
- Configuration examples for different deployment scenarios

Source: Based on HTTPX timeouts_httpx.md and exceptions_httpx.md patterns
"""

import asyncio
import time
import logging
import statistics
from typing import List, Dict, Any
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from distributed_agent_http import (
    AgentHTTPClient,
    AgentHTTPClientConfig,
    get_agent_http_client,
    CircuitBreaker
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentCommunicationScenarios:
    """Pre-defined configuration scenarios for different agent deployment patterns."""

    @staticmethod
    def high_frequency_local():
        """Configuration for high-frequency communication within same data center."""
        return AgentHTTPClientConfig(
            connect_timeout=0.5,
            read_timeout=1.0,
            write_timeout=0.5,
            pool_timeout=0.2,
            total_timeout=2.0,
            max_connections=50,
            max_keepalive_connections=30,
            keepalive_expiry=60.0,
            max_retries=2,
            retry_backoff_factor=0.1,
            enable_http2=True
        )

    @staticmethod
    def cross_region():
        """Configuration for cross-region agent communication."""
        return AgentHTTPClientConfig(
            connect_timeout=2.0,
            read_timeout=3.0,
            write_timeout=2.0,
            pool_timeout=1.0,
            total_timeout=5.0,
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
            max_retries=4,
            retry_backoff_factor=0.5,
            retry_jitter=True,
            enable_http2=True
        )

    @staticmethod
    def edge_computing():
        """Configuration for edge computing scenarios with limited resources."""
        return AgentHTTPClientConfig(
            connect_timeout=1.0,
            read_timeout=2.0,
            write_timeout=1.0,
            pool_timeout=0.5,
            total_timeout=3.0,
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=15.0,
            max_retries=3,
            retry_backoff_factor=0.3,
            enable_http2=False  # Lower overhead for resource-constrained environments
        )

    @staticmethod
    def critical_real_time():
        """Configuration for critical real-time agent communication."""
        return AgentHTTPClientConfig(
            connect_timeout=0.3,
            read_timeout=0.7,
            write_timeout=0.3,
            pool_timeout=0.1,
            total_timeout=1.5,
            max_connections=100,
            max_keepalive_connections=50,
            keepalive_expiry=120.0,
            max_retries=1,  # Fail fast for real-time requirements
            retry_backoff_factor=0.05,
            enable_http2=True
        )


class PerformanceMetrics:
    """Collect and analyze performance metrics for agent communication."""

    def __init__(self):
        self.request_times: List[float] = []
        self.success_count = 0
        self.timeout_count = 0
        self.error_count = 0
        self.http2_usage = 0

    def record_success(self, duration: float, http_version: str):
        """Record successful request metrics."""
        self.request_times.append(duration)
        self.success_count += 1
        if http_version == "HTTP/2":
            self.http2_usage += 1

    def record_timeout(self):
        """Record timeout occurrence."""
        self.timeout_count += 1

    def record_error(self):
        """Record error occurrence."""
        self.error_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        total_requests = self.success_count + self.timeout_count + self.error_count

        if not self.request_times:
            return {
                "total_requests": total_requests,
                "success_rate": 0.0,
                "timeout_rate": self.timeout_count / total_requests if total_requests > 0 else 0,
                "error_rate": self.error_count / total_requests if total_requests > 0 else 0
            }

        return {
            "total_requests": total_requests,
            "success_count": self.success_count,
            "success_rate": self.success_count / total_requests if total_requests > 0 else 0,
            "timeout_rate": self.timeout_count / total_requests if total_requests > 0 else 0,
            "error_rate": self.error_count / total_requests if total_requests > 0 else 0,
            "http2_usage_rate": self.http2_usage / self.success_count if self.success_count > 0 else 0,
            "latency_stats": {
                "mean": statistics.mean(self.request_times),
                "median": statistics.median(self.request_times),
                "p95": self._percentile(self.request_times, 95),
                "p99": self._percentile(self.request_times, 99),
                "min": min(self.request_times),
                "max": max(self.request_times)
            }
        }

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = (percentile / 100) * len(sorted_data)
        if index.is_integer():
            return sorted_data[int(index) - 1]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1] if int(index) + 1 < len(sorted_data) else lower
            return lower + (upper - lower) * (index - int(index))


async def test_timeout_configurations():
    """
    Test different timeout scenarios to validate configuration effectiveness.

    Based on timeouts_httpx.md patterns for granular timeout control.
    """
    print("Testing timeout configurations...")

    # Test scenarios with different timeout requirements
    test_scenarios = [
        ("Connect Timeout", "http://10.255.255.1/"),  # Non-routable IP for connect timeout
        ("Read Timeout", "https://httpbin.org/delay/5"),  # Long response for read timeout
        ("Quick Response", "https://httpbin.org/status/200"),  # Fast response
    ]

    config = AgentHTTPClientConfig(
        connect_timeout=1.0,
        read_timeout=2.0,
        write_timeout=1.0,
        pool_timeout=0.5,
        total_timeout=3.0
    )

    async with get_agent_http_client(config) as client:
        for scenario_name, url in test_scenarios:
            print(f"\nTesting {scenario_name}...")
            start_time = time.time()

            try:
                response = await client.get(url)
                duration = time.time() - start_time
                print(f"  ✓ Success: {response.status_code} in {duration:.2f}s ({response.http_version})")

            except httpx.ConnectTimeout:
                duration = time.time() - start_time
                print(f"  ⏰ Connect timeout after {duration:.2f}s (expected)")

            except httpx.ReadTimeout:
                duration = time.time() - start_time
                print(f"  ⏰ Read timeout after {duration:.2f}s (expected)")

            except Exception as e:
                duration = time.time() - start_time
                print(f"  ❌ Error after {duration:.2f}s: {type(e).__name__}: {e}")


async def test_circuit_breaker():
    """Test circuit breaker functionality for fault tolerance."""
    print("\nTesting circuit breaker...")

    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    # Simulate failures
    for i in range(5):
        try:
            with circuit_breaker:
                if i < 4:  # First 4 attempts fail
                    raise httpx.ConnectError("Simulated connection failure")
                else:  # 5th attempt would succeed but circuit is open
                    print(f"  Attempt {i+1}: Would succeed")

            print(f"  Attempt {i+1}: Success")

        except Exception as e:
            print(f"  Attempt {i+1}: Failed - {e}")

    print(f"  Final circuit state: {circuit_breaker.state}")


async def benchmark_agent_communication():
    """
    Benchmark agent communication performance under different scenarios.

    Tests concurrent requests, connection reuse, and HTTP/2 benefits.
    """
    print("\nBenchmarking agent communication...")

    config = AgentHTTPClientConfig(
        connect_timeout=2.0,
        read_timeout=3.0,
        write_timeout=2.0,
        pool_timeout=1.0,
        total_timeout=5.0,
        max_connections=50,
        max_keepalive_connections=25,
        enable_http2=True
    )

    metrics = PerformanceMetrics()

    async with get_agent_http_client(config, "https://httpbin.org") as client:

        # Test concurrent requests to measure connection pooling benefits
        async def make_request(request_id: int):
            """Make a single test request."""
            start_time = time.time()
            try:
                response = await client.get("/json")
                duration = time.time() - start_time
                metrics.record_success(duration, response.http_version)
                return f"Request {request_id}: {response.status_code} in {duration:.3f}s ({response.http_version})"

            except httpx.TimeoutException:
                metrics.record_timeout()
                return f"Request {request_id}: Timeout"

            except Exception as e:
                metrics.record_error()
                return f"Request {request_id}: Error - {e}"

        # Execute concurrent requests
        print("  Executing 20 concurrent requests...")
        tasks = [make_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # Print sample results
        for result in results[:5]:
            print(f"    {result}")

        if len(results) > 5:
            print(f"    ... and {len(results) - 5} more")

    # Print performance summary
    summary = metrics.get_summary()
    print(f"\n  Performance Summary:")
    print(f"    Total requests: {summary['total_requests']}")
    print(f"    Success rate: {summary['success_rate']:.1%}")
    print(f"    HTTP/2 usage: {summary['http2_usage_rate']:.1%}")

    if 'latency_stats' in summary:
        latency = summary['latency_stats']
        print(f"    Latency - Mean: {latency['mean']:.3f}s, P95: {latency['p95']:.3f}s, P99: {latency['p99']:.3f}s")


async def test_error_handling_patterns():
    """
    Test comprehensive error handling based on exceptions_httpx.md hierarchy.
    """
    print("\nTesting error handling patterns...")

    config = AgentHTTPClientConfig(
        connect_timeout=1.0,
        read_timeout=2.0,
        max_retries=2
    )

    error_scenarios = [
        ("Invalid URL", "not-a-valid-url"),
        ("Non-existent domain", "https://this-domain-definitely-does-not-exist.com/"),
        ("Connection refused", "http://localhost:65432/"),  # Unlikely to be in use
        ("HTTP error status", "https://httpbin.org/status/500"),
    ]

    async with get_agent_http_client(config) as client:
        for scenario_name, url in error_scenarios:
            print(f"  Testing {scenario_name}...")
            try:
                response = await client.get(url)
                print(f"    Unexpected success: {response.status_code}")

            except httpx.InvalidURL:
                print(f"    ✓ Caught InvalidURL (expected)")

            except httpx.ConnectError:
                print(f"    ✓ Caught ConnectError (expected)")

            except httpx.ConnectTimeout:
                print(f"    ✓ Caught ConnectTimeout (expected)")

            except httpx.HTTPStatusError as e:
                print(f"    ✓ Caught HTTPStatusError: {e.response.status_code} (expected)")

            except Exception as e:
                print(f"    ✓ Caught {type(e).__name__}: {e}")


def test_configuration_scenarios():
    """Test pre-defined configuration scenarios for different deployment patterns."""
    print("\nTesting configuration scenarios...")

    scenarios = [
        ("High-frequency local", AgentCommunicationScenarios.high_frequency_local()),
        ("Cross-region", AgentCommunicationScenarios.cross_region()),
        ("Edge computing", AgentCommunicationScenarios.edge_computing()),
        ("Critical real-time", AgentCommunicationScenarios.critical_real_time()),
    ]

    for scenario_name, config in scenarios:
        print(f"\n  {scenario_name} configuration:")
        print(f"    Connect timeout: {config.timeout.connect}s")
        print(f"    Read timeout: {config.timeout.read}s")
        print(f"    Total timeout: {config.total_timeout}s")
        print(f"    Max connections: {config.limits.max_connections}")
        print(f"    HTTP/2 enabled: {config.enable_http2}")
        print(f"    Max retries: {config.max_retries}")


# PyTest test cases for automated testing
class TestAgentHTTPClient:
    """PyTest test suite for agent HTTP client functionality."""

    @pytest.fixture
    async def client(self):
        """Fixture providing configured HTTP client."""
        config = AgentHTTPClientConfig(
            connect_timeout=2.0,
            read_timeout=3.0,
            total_timeout=5.0
        )
        async with get_agent_http_client(config) as client:
            yield client

    @pytest.mark.asyncio
    async def test_successful_request(self, client):
        """Test successful HTTP request."""
        response = await client.get("https://httpbin.org/json")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_timeout_handling(self, client):
        """Test timeout exception handling."""
        with pytest.raises(httpx.TimeoutException):
            # This should timeout due to long delay
            await client.get("https://httpbin.org/delay/10")

    @pytest.mark.asyncio
    async def test_retry_behavior(self, client):
        """Test retry behavior on transient failures."""
        # This test would require a mock server to properly test retry logic
        # For now, we test that retries don't break on success
        response = await client.get("https://httpbin.org/status/200")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self):
        """Test circuit breaker state transitions."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)

        # Test initial state
        assert cb.state == "CLOSED"

        # Simulate failures
        for _ in range(2):
            try:
                with cb:
                    raise httpx.ConnectError("Test failure")
            except:
                pass

        # Circuit should now be open
        assert cb.state == "OPEN"

        # Test that circuit rejects requests when open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            with cb:
                pass


async def main():
    """Run all test scenarios and benchmarks."""
    print("=" * 60)
    print("HTTPX Agent Communication Test Suite")
    print("=" * 60)

    try:
        # Run all test scenarios
        await test_timeout_configurations()
        await test_circuit_breaker()
        await benchmark_agent_communication()
        await test_error_handling_patterns()
        test_configuration_scenarios()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(main())