"""
Tests for configuration constants in consts/__init__.py.
"""

import pytest

from consts import (
    # Network Configuration
    LEAGUE_MANAGER_PORT, REFEREE_PORT, PLAYER_PORTS,
    LEAGUE_MANAGER_URL, REFEREE_URL,

    # External Services
    OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL,

    # Timing Configuration
    DEFAULT_TIMEOUT, MOVE_TIMEOUT, MATCH_TIMEOUT, REGISTRATION_TIMEOUT,

    # Retry & Circuit Breaker
    MAX_RETRIES, CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_RESET_TIME,
    EXPONENTIAL_BACKOFF_BASE, JITTER_RANGE,

    # HTTP Client Configuration
    HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT, HTTP_TOTAL_TIMEOUT,
    MAX_CONNECTIONS, MAX_KEEPALIVE,

    # Protocol Configuration
    PROTOCOL_VERSION, USER_AGENT_PREFIX,

    # Game Configuration
    GAME_TYPE, MAX_ROUNDS_PER_MATCH, RANDOM_NUMBER_MIN, RANDOM_NUMBER_MAX,

    # Logging Configuration
    LOG_LEVEL, JSON_LOG_FORMAT, LOG_REQUEST_IDS,

    # Agent Registration
    REGISTRATION_ENDPOINT, HEALTH_ENDPOINT, MCP_ENDPOINT,

    # Strategy Configuration
    AVAILABLE_STRATEGIES, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,

    # Match Orchestration
    MAX_CONCURRENT_MATCHES, ROUND_ROBIN_SCHEDULING, MATCH_CLEANUP_DELAY
)


class TestNetworkConfiguration:
    """Tests for network configuration constants."""

    def test_league_manager_port(self):
        """Test League Manager port is configured."""
        assert LEAGUE_MANAGER_PORT == 8000
        assert isinstance(LEAGUE_MANAGER_PORT, int)

    def test_referee_port(self):
        """Test Referee port is configured."""
        assert REFEREE_PORT == 8001
        assert isinstance(REFEREE_PORT, int)

    def test_player_ports(self):
        """Test Player ports are configured."""
        assert len(PLAYER_PORTS) == 4
        assert PLAYER_PORTS == [8101, 8102, 8103, 8104]

    def test_player_ports_are_sequential(self):
        """Test player ports are sequential."""
        for i in range(len(PLAYER_PORTS) - 1):
            assert PLAYER_PORTS[i + 1] == PLAYER_PORTS[i] + 1

    def test_no_port_conflicts(self):
        """Test no port conflicts between agents."""
        all_ports = [LEAGUE_MANAGER_PORT, REFEREE_PORT] + PLAYER_PORTS
        assert len(all_ports) == len(set(all_ports))

    def test_league_manager_url_format(self):
        """Test League Manager URL is properly formatted."""
        assert LEAGUE_MANAGER_URL == f"http://localhost:{LEAGUE_MANAGER_PORT}"
        assert LEAGUE_MANAGER_URL.startswith("http://")

    def test_referee_url_format(self):
        """Test Referee URL is properly formatted."""
        assert REFEREE_URL == f"http://localhost:{REFEREE_PORT}"


class TestExternalServices:
    """Tests for external service configuration."""

    def test_ollama_base_url(self):
        """Test Ollama base URL is configured."""
        assert OLLAMA_BASE_URL == "http://localhost:11434"

    def test_default_ollama_model(self):
        """Test default Ollama model is set."""
        assert DEFAULT_OLLAMA_MODEL == "llama3"
        assert isinstance(DEFAULT_OLLAMA_MODEL, str)


class TestTimingConfiguration:
    """Tests for timing configuration constants."""

    def test_default_timeout(self):
        """Test default timeout is reasonable."""
        assert DEFAULT_TIMEOUT == 2.0
        assert DEFAULT_TIMEOUT > 0

    def test_move_timeout(self):
        """Test move timeout is in milliseconds."""
        assert MOVE_TIMEOUT == 2000
        assert MOVE_TIMEOUT >= 500  # Reasonable minimum

    def test_match_timeout(self):
        """Test match timeout is reasonable."""
        assert MATCH_TIMEOUT == 30.0
        assert MATCH_TIMEOUT > MOVE_TIMEOUT / 1000  # Match should be longer than single move

    def test_registration_timeout(self):
        """Test registration timeout is reasonable."""
        assert REGISTRATION_TIMEOUT == 5.0
        assert REGISTRATION_TIMEOUT > 0


class TestRetryConfiguration:
    """Tests for retry and circuit breaker configuration."""

    def test_max_retries(self):
        """Test max retries is reasonable."""
        assert MAX_RETRIES == 3
        assert MAX_RETRIES >= 1

    def test_circuit_breaker_threshold(self):
        """Test circuit breaker threshold."""
        assert CIRCUIT_BREAKER_THRESHOLD == 5
        assert CIRCUIT_BREAKER_THRESHOLD > MAX_RETRIES

    def test_circuit_breaker_reset_time(self):
        """Test circuit breaker reset time."""
        assert CIRCUIT_BREAKER_RESET_TIME == 30.0
        assert CIRCUIT_BREAKER_RESET_TIME > 0

    def test_exponential_backoff_base(self):
        """Test exponential backoff base."""
        assert EXPONENTIAL_BACKOFF_BASE == 2.0
        assert EXPONENTIAL_BACKOFF_BASE > 1.0  # Must be > 1 for exponential growth

    def test_jitter_range(self):
        """Test jitter range for backoff."""
        assert JITTER_RANGE == 0.5
        assert 0 <= JITTER_RANGE <= 1.0


class TestHTTPClientConfiguration:
    """Tests for HTTP client configuration."""

    def test_connect_timeout(self):
        """Test HTTP connect timeout."""
        assert HTTP_CONNECT_TIMEOUT == 1.5
        assert HTTP_CONNECT_TIMEOUT > 0

    def test_read_timeout(self):
        """Test HTTP read timeout."""
        assert HTTP_READ_TIMEOUT == 2.0
        assert HTTP_READ_TIMEOUT > 0

    def test_total_timeout(self):
        """Test HTTP total timeout."""
        assert HTTP_TOTAL_TIMEOUT == 3.0
        assert HTTP_TOTAL_TIMEOUT >= HTTP_CONNECT_TIMEOUT + HTTP_READ_TIMEOUT

    def test_max_connections(self):
        """Test max connections."""
        assert MAX_CONNECTIONS == 20
        assert MAX_CONNECTIONS > 0

    def test_max_keepalive(self):
        """Test max keepalive connections."""
        assert MAX_KEEPALIVE == 10
        assert MAX_KEEPALIVE <= MAX_CONNECTIONS


class TestProtocolConfiguration:
    """Tests for protocol configuration."""

    def test_protocol_version(self):
        """Test protocol version format."""
        assert PROTOCOL_VERSION == "league.v2"
        assert "league" in PROTOCOL_VERSION

    def test_user_agent_prefix(self):
        """Test user agent prefix."""
        assert USER_AGENT_PREFIX == "MCP-Agent-Client"


class TestGameConfiguration:
    """Tests for game configuration constants."""

    def test_game_type(self):
        """Test game type is configured."""
        assert GAME_TYPE == "even_odd"

    def test_max_rounds_per_match(self):
        """Test max rounds per match."""
        assert MAX_ROUNDS_PER_MATCH == 10
        assert MAX_ROUNDS_PER_MATCH > 0

    def test_random_number_range(self):
        """Test random number range is valid."""
        assert RANDOM_NUMBER_MIN == 1
        assert RANDOM_NUMBER_MAX == 10
        assert RANDOM_NUMBER_MIN < RANDOM_NUMBER_MAX
        assert RANDOM_NUMBER_MIN > 0

    def test_random_range_allows_both_parities(self):
        """Test random range includes both even and odd numbers."""
        numbers = list(range(RANDOM_NUMBER_MIN, RANDOM_NUMBER_MAX + 1))
        evens = [n for n in numbers if n % 2 == 0]
        odds = [n for n in numbers if n % 2 == 1]

        assert len(evens) > 0
        assert len(odds) > 0


class TestLoggingConfiguration:
    """Tests for logging configuration."""

    def test_log_level(self):
        """Test log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert LOG_LEVEL in valid_levels

    def test_json_log_format(self):
        """Test JSON log format setting."""
        assert isinstance(JSON_LOG_FORMAT, bool)

    def test_log_request_ids(self):
        """Test request ID logging setting."""
        assert isinstance(LOG_REQUEST_IDS, bool)


class TestEndpointConfiguration:
    """Tests for API endpoint configuration."""

    def test_registration_endpoint(self):
        """Test registration endpoint."""
        assert REGISTRATION_ENDPOINT == "/register"
        assert REGISTRATION_ENDPOINT.startswith("/")

    def test_health_endpoint(self):
        """Test health endpoint."""
        assert HEALTH_ENDPOINT == "/health"
        assert HEALTH_ENDPOINT.startswith("/")

    def test_mcp_endpoint(self):
        """Test MCP endpoint."""
        assert MCP_ENDPOINT == "/mcp"
        assert MCP_ENDPOINT.startswith("/")


class TestStrategyConfiguration:
    """Tests for strategy configuration."""

    def test_available_strategies(self):
        """Test available strategies list."""
        assert "random" in AVAILABLE_STRATEGIES
        assert "history" in AVAILABLE_STRATEGIES
        assert "llm" in AVAILABLE_STRATEGIES
        assert len(AVAILABLE_STRATEGIES) == 3

    def test_llm_temperature(self):
        """Test LLM temperature setting."""
        assert LLM_TEMPERATURE == 0.7
        assert 0.0 <= LLM_TEMPERATURE <= 2.0

    def test_llm_max_tokens(self):
        """Test LLM max tokens setting."""
        assert LLM_MAX_TOKENS == 100
        assert LLM_MAX_TOKENS > 0

    def test_llm_timeout(self):
        """Test LLM timeout setting."""
        assert LLM_TIMEOUT == 3.0
        assert LLM_TIMEOUT > 0


class TestMatchOrchestration:
    """Tests for match orchestration configuration."""

    def test_max_concurrent_matches(self):
        """Test max concurrent matches."""
        assert MAX_CONCURRENT_MATCHES == 4
        assert MAX_CONCURRENT_MATCHES > 0

    def test_round_robin_scheduling(self):
        """Test round robin scheduling setting."""
        assert ROUND_ROBIN_SCHEDULING is True
        assert isinstance(ROUND_ROBIN_SCHEDULING, bool)

    def test_match_cleanup_delay(self):
        """Test match cleanup delay."""
        assert MATCH_CLEANUP_DELAY == 1.0
        assert MATCH_CLEANUP_DELAY >= 0


class TestConfigurationConsistency:
    """Tests for configuration consistency across values."""

    def test_timeout_hierarchy(self):
        """Test timeout values form sensible hierarchy."""
        # Move timeout should be less than match timeout
        assert MOVE_TIMEOUT / 1000 < MATCH_TIMEOUT

        # LLM timeout should be less than or equal to move timeout
        assert LLM_TIMEOUT <= MOVE_TIMEOUT / 1000

    def test_player_ports_match_count(self):
        """Test player ports match expected player count."""
        # With round robin, need at least 2 players
        assert len(PLAYER_PORTS) >= 2

        # Max concurrent matches shouldn't exceed possible matches
        max_possible_matches = len(PLAYER_PORTS) // 2
        assert MAX_CONCURRENT_MATCHES <= len(PLAYER_PORTS)

    def test_circuit_breaker_allows_retries(self):
        """Test circuit breaker threshold allows for retries."""
        assert CIRCUIT_BREAKER_THRESHOLD > MAX_RETRIES
