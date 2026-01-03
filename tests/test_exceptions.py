"""
Tests for custom exception hierarchy in shared/exceptions.py.
"""

import pytest

from shared.exceptions import (
    LeagueError,
    RegistrationError,
    MatchError,
    StrategyError,
    NetworkError,
    OperationTimeoutError,
    LLMError,
    ProtocolError,
    SchedulingError,
    CircuitBreakerError
)


class TestLeagueErrorBase:
    """Tests for the base LeagueError class."""

    def test_league_error_creation(self):
        """Test basic LeagueError creation."""
        error = LeagueError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_league_error_is_exception(self):
        """Test LeagueError inherits from Exception."""
        error = LeagueError("Test")
        assert isinstance(error, Exception)

    def test_league_error_can_be_raised(self):
        """Test LeagueError can be raised and caught."""
        with pytest.raises(LeagueError) as exc_info:
            raise LeagueError("Test error")
        assert "Test error" in str(exc_info.value)


class TestRegistrationError:
    """Tests for RegistrationError."""

    def test_registration_error_creation(self):
        """Test RegistrationError creation."""
        error = RegistrationError("Registration failed", agent_id="player:P01")
        assert "Registration failed" in str(error)

    def test_registration_error_with_agent_id(self):
        """Test RegistrationError stores agent_id."""
        error = RegistrationError("Failed", agent_id="player:P01")
        assert error.agent_id == "player:P01"

    def test_registration_error_without_agent_id(self):
        """Test RegistrationError works without agent_id."""
        error = RegistrationError("Failed")
        assert error.agent_id is None

    def test_registration_error_inherits_league_error(self):
        """Test RegistrationError inherits from LeagueError."""
        error = RegistrationError("Test")
        assert isinstance(error, LeagueError)


class TestMatchError:
    """Tests for MatchError."""

    def test_match_error_creation(self):
        """Test MatchError creation."""
        error = MatchError("Match failed", match_id="M-001")
        assert "Match failed" in str(error)

    def test_match_error_with_match_id(self):
        """Test MatchError stores match_id."""
        error = MatchError("Failed", match_id="M-001")
        assert error.match_id == "M-001"

    def test_match_error_without_match_id(self):
        """Test MatchError works without match_id."""
        error = MatchError("Failed")
        assert error.match_id is None

    def test_match_error_inherits_league_error(self):
        """Test MatchError inherits from LeagueError."""
        error = MatchError("Test")
        assert isinstance(error, LeagueError)


class TestStrategyError:
    """Tests for StrategyError."""

    def test_strategy_error_creation(self):
        """Test StrategyError creation."""
        error = StrategyError("Strategy failed", strategy_name="llm")
        assert "Strategy failed" in str(error)

    def test_strategy_error_with_strategy_name(self):
        """Test StrategyError stores strategy_name."""
        error = StrategyError("Failed", strategy_name="llm")
        assert error.strategy_name == "llm"

    def test_strategy_error_inherits_league_error(self):
        """Test StrategyError inherits from LeagueError."""
        error = StrategyError("Test")
        assert isinstance(error, LeagueError)


class TestNetworkError:
    """Tests for NetworkError."""

    def test_network_error_creation(self):
        """Test NetworkError creation."""
        error = NetworkError("Connection refused", endpoint="http://localhost:8000")
        assert "Connection refused" in str(error)

    def test_network_error_with_endpoint(self):
        """Test NetworkError stores endpoint."""
        error = NetworkError("Failed", endpoint="http://localhost:8000")
        assert error.endpoint == "http://localhost:8000"

    def test_network_error_with_status_code(self):
        """Test NetworkError stores status_code."""
        error = NetworkError("Failed", status_code=500)
        assert error.status_code == 500

    def test_network_error_inherits_league_error(self):
        """Test NetworkError inherits from LeagueError."""
        error = NetworkError("Test")
        assert isinstance(error, LeagueError)


class TestOperationTimeoutError:
    """Tests for OperationTimeoutError."""

    def test_timeout_error_creation(self):
        """Test OperationTimeoutError creation."""
        error = OperationTimeoutError("Timed out", timeout_ms=2000)
        assert "Timed out" in str(error)

    def test_timeout_error_with_timeout_ms(self):
        """Test OperationTimeoutError stores timeout_ms."""
        error = OperationTimeoutError("Failed", timeout_ms=2000)
        assert error.timeout_ms == 2000

    def test_timeout_error_with_operation(self):
        """Test OperationTimeoutError stores operation."""
        error = OperationTimeoutError("Failed", operation="move_request")
        assert error.operation == "move_request"

    def test_timeout_error_inherits_league_error(self):
        """Test OperationTimeoutError inherits from LeagueError."""
        error = OperationTimeoutError("Test")
        assert isinstance(error, LeagueError)


class TestLLMError:
    """Tests for LLMError."""

    def test_llm_error_creation(self):
        """Test LLMError creation."""
        error = LLMError("LLM failed", model_name="llama3")
        assert "LLM failed" in str(error)

    def test_llm_error_with_model_name(self):
        """Test LLMError stores model_name."""
        error = LLMError("Failed", model_name="llama3")
        assert error.model_name == "llama3"

    def test_llm_error_with_response_data(self):
        """Test LLMError stores response_data."""
        error = LLMError("Failed", response_data="invalid json")
        assert error.response_data == "invalid json"

    def test_llm_error_inherits_league_error(self):
        """Test LLMError inherits from LeagueError."""
        error = LLMError("Test")
        assert isinstance(error, LeagueError)


class TestProtocolError:
    """Tests for ProtocolError."""

    def test_protocol_error_creation(self):
        """Test ProtocolError creation."""
        error = ProtocolError("Invalid message", message_type="UNKNOWN")
        assert "Invalid message" in str(error)

    def test_protocol_error_with_message_type(self):
        """Test ProtocolError stores message_type."""
        error = ProtocolError("Failed", message_type="UNKNOWN")
        assert error.message_type == "UNKNOWN"

    def test_protocol_error_inherits_league_error(self):
        """Test ProtocolError inherits from LeagueError."""
        error = ProtocolError("Test")
        assert isinstance(error, LeagueError)


class TestSchedulingError:
    """Tests for SchedulingError."""

    def test_scheduling_error_creation(self):
        """Test SchedulingError creation."""
        error = SchedulingError("Scheduling failed")
        assert "Scheduling failed" in str(error)

    def test_scheduling_error_inherits_league_error(self):
        """Test SchedulingError inherits from LeagueError."""
        error = SchedulingError("Test")
        assert isinstance(error, LeagueError)


class TestCircuitBreakerError:
    """Tests for CircuitBreakerError."""

    def test_circuit_breaker_error_creation(self):
        """Test CircuitBreakerError creation."""
        error = CircuitBreakerError("Circuit open", service_name="league-manager")
        assert "Circuit open" in str(error)

    def test_circuit_breaker_error_with_service_name(self):
        """Test CircuitBreakerError stores service_name."""
        error = CircuitBreakerError("Failed", service_name="league-manager")
        assert error.service_name == "league-manager"

    def test_circuit_breaker_error_inherits_network_error(self):
        """Test CircuitBreakerError inherits from NetworkError."""
        error = CircuitBreakerError("Test")
        assert isinstance(error, NetworkError)
        assert isinstance(error, LeagueError)


class TestExceptionHierarchy:
    """Tests for the exception hierarchy structure."""

    def test_all_errors_are_league_errors(self):
        """Test all custom exceptions inherit from LeagueError."""
        errors = [
            RegistrationError("test"),
            MatchError("test"),
            StrategyError("test"),
            NetworkError("test"),
            OperationTimeoutError("test"),
            LLMError("test"),
            ProtocolError("test"),
            SchedulingError("test"),
            CircuitBreakerError("test")
        ]

        for error in errors:
            assert isinstance(error, LeagueError), f"{type(error).__name__} is not a LeagueError"

    def test_catch_all_with_league_error(self):
        """Test catching all errors with LeagueError."""
        errors_caught = 0

        for ErrorClass in [RegistrationError, MatchError, StrategyError, NetworkError]:
            try:
                raise ErrorClass("test")
            except LeagueError:
                errors_caught += 1

        assert errors_caught == 4

    def test_specific_catch_before_general(self):
        """Test specific exception types can be caught before general."""
        try:
            raise RegistrationError("test", agent_id="player:P01")
        except RegistrationError as e:
            assert e.agent_id == "player:P01"
        except LeagueError:
            pytest.fail("Should have caught RegistrationError specifically")
