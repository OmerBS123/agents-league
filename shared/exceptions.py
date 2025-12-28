"""
Custom exception hierarchy for the Distributed AI Agent League System.
Provides specific error types for different failure modes.
"""

from typing import Optional, Dict, Any


class LeagueError(Exception):
    """Base exception for all league operations"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "LEAGUE_ERROR"
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Code: {self.error_code}, Details: {self.details})"
        return f"{self.message} (Code: {self.error_code})"


class RegistrationError(LeagueError):
    """Registration failures with the League Manager"""
    def __init__(self, message: str, agent_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "REGISTRATION_ERROR", details)
        self.agent_id = agent_id


class MatchError(LeagueError):
    """Match execution and orchestration failures"""
    def __init__(self, message: str, match_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MATCH_ERROR", details)
        self.match_id = match_id


class StrategyError(LeagueError):
    """Player strategy failures and move generation errors"""
    def __init__(self, message: str, strategy_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "STRATEGY_ERROR", details)
        self.strategy_type = strategy_type


class NetworkError(LeagueError):
    """Network communication failures between agents"""
    def __init__(self, message: str, endpoint: Optional[str] = None, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "NETWORK_ERROR", details)
        self.endpoint = endpoint
        self.status_code = status_code


class CircuitBreakerError(NetworkError):
    """Circuit breaker is open, requests are being blocked"""
    def __init__(self, message: str, service_name: Optional[str] = None, reset_time: Optional[float] = None):
        details = {
            "service_name": service_name,
            "reset_time": reset_time
        }
        super().__init__(message, None, None, details)
        self.service_name = service_name
        self.reset_time = reset_time


class OperationTimeoutError(LeagueError):
    """Operation timeout errors"""
    def __init__(self, message: str, timeout_duration: Optional[float] = None, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.timeout_duration = timeout_duration
        self.operation = operation


class PayloadValidationError(LeagueError):
    """Protocol and data validation errors"""
    def __init__(self, message: str, field_name: Optional[str] = None, invalid_value: Optional[Any] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field_name = field_name
        self.invalid_value = invalid_value


class ProtocolError(LeagueError):
    """Message protocol and communication format errors"""
    def __init__(self, message: str, message_type: Optional[str] = None, protocol_version: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PROTOCOL_ERROR", details)
        self.message_type = message_type
        self.protocol_version = protocol_version


class LLMError(StrategyError):
    """Large Language Model integration failures"""
    def __init__(self, message: str, model_name: Optional[str] = None, response_data: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "llm", details)
        self.model_name = model_name
        self.response_data = response_data


class SchedulingError(LeagueError):
    """Tournament scheduling and match organization errors"""
    def __init__(self, message: str, tournament_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SCHEDULING_ERROR", details)
        self.tournament_id = tournament_id


class AgentStateError(LeagueError):
    """Invalid agent state transitions and operations"""
    def __init__(self, message: str, agent_id: Optional[str] = None, current_state: Optional[str] = None, requested_state: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AGENT_STATE_ERROR", details)
        self.agent_id = agent_id
        self.current_state = current_state
        self.requested_state = requested_state