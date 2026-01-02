"""
Pytest configuration and shared fixtures for the Distributed AI Agent League tests.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timezone

from shared.schemas import (
    MCPEnvelope, MessageType, ParityChoice, AgentStatus, MatchStatus,
    RegistrationData, RegistrationResponse, InvitationData, InvitationResponse,
    MoveRequestData, MoveResponseData, GameResultData, MatchResultData,
    ScheduleMatchData, StandingsData, ErrorData, HeartbeatData
)


# --- Async Event Loop Configuration ---

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# --- Sample Data Fixtures ---

@pytest.fixture
def sample_player_id() -> str:
    """Sample player ID in correct format."""
    return "player:P01"


@pytest.fixture
def sample_referee_id() -> str:
    """Sample referee ID in correct format."""
    return "referee:REF-MAIN"


@pytest.fixture
def sample_manager_id() -> str:
    """Sample manager ID in correct format."""
    return "manager:league"


@pytest.fixture
def sample_match_id() -> str:
    """Sample match ID in correct format."""
    return "M-001"


# --- Registration Fixtures ---

@pytest.fixture
def valid_registration_data() -> Dict[str, Any]:
    """Valid registration data payload."""
    return {
        "display_name": "Test Player",
        "contact_endpoint": "http://localhost:8101",
        "strategies": ["random"],
        "capabilities": {},
        "agent_version": "1.0.0"
    }


@pytest.fixture
def registration_envelope(sample_player_id, sample_manager_id, valid_registration_data) -> MCPEnvelope:
    """Complete registration envelope."""
    return MCPEnvelope(
        message_type=MessageType.REGISTER,
        sender=sample_player_id,
        recipient=sample_manager_id,
        data=valid_registration_data
    )


# --- Invitation Fixtures ---

@pytest.fixture
def valid_invitation_data(sample_match_id) -> Dict[str, Any]:
    """Valid invitation data payload."""
    return {
        "match_id": sample_match_id,
        "round_id": 1,
        "opponent_id": "player:P02",
        "role_in_match": "A",
        "timeout_ms": 2000
    }


@pytest.fixture
def invitation_envelope(sample_referee_id, sample_player_id, valid_invitation_data) -> MCPEnvelope:
    """Complete invitation envelope."""
    return MCPEnvelope(
        message_type=MessageType.INVITATION,
        sender=sample_referee_id,
        recipient=sample_player_id,
        data=valid_invitation_data
    )


# --- Move Request/Response Fixtures ---

@pytest.fixture
def valid_move_request_data(sample_match_id) -> Dict[str, Any]:
    """Valid move request data payload."""
    return {
        "match_id": sample_match_id,
        "round_id": 1,
        "opponent_history": [],
        "timeout_ms": 2000
    }


@pytest.fixture
def valid_move_response_data(sample_match_id) -> Dict[str, Any]:
    """Valid move response data payload."""
    return {
        "match_id": sample_match_id,
        "round_id": 1,
        "parity_choice": "even",
        "confidence": 0.75,
        "reasoning": "Random selection",
        "strategy_used": "random"
    }


@pytest.fixture
def move_request_envelope(sample_referee_id, sample_player_id, valid_move_request_data) -> MCPEnvelope:
    """Complete move request envelope."""
    return MCPEnvelope(
        message_type=MessageType.MOVE_CALL,
        sender=sample_referee_id,
        recipient=sample_player_id,
        data=valid_move_request_data
    )


# --- Game Result Fixtures ---

@pytest.fixture
def sample_game_result(sample_match_id) -> GameResultData:
    """Sample game result for a single round."""
    return GameResultData(
        match_id=sample_match_id,
        round_id=1,
        player_a_choice=ParityChoice.EVEN,
        player_b_choice=ParityChoice.ODD,
        random_number=5,
        winner_id="player:P01",
        calculation="(0+1+5) = 6 (even)",
        round_duration_ms=150.5
    )


@pytest.fixture
def sample_match_result(sample_match_id, sample_game_result) -> MatchResultData:
    """Sample complete match result."""
    return MatchResultData(
        match_id=sample_match_id,
        winner_id="player:P01",
        loser_id="player:P02",
        final_score={"player:P01": 6, "player:P02": 4},
        total_rounds=10,
        match_duration_ms=5000.0,
        game_results=[sample_game_result],
        match_stats={"avg_round_duration": 150.5}
    )


# --- Opponent History Fixtures ---

@pytest.fixture
def empty_opponent_history() -> List[ParityChoice]:
    """Empty opponent history for first round."""
    return []


@pytest.fixture
def short_opponent_history() -> List[ParityChoice]:
    """Short opponent history (2 moves)."""
    return [ParityChoice.EVEN, ParityChoice.ODD]


@pytest.fixture
def long_opponent_history() -> List[ParityChoice]:
    """Long opponent history with pattern (10 moves)."""
    return [
        ParityChoice.EVEN, ParityChoice.ODD,
        ParityChoice.EVEN, ParityChoice.ODD,
        ParityChoice.EVEN, ParityChoice.ODD,
        ParityChoice.EVEN, ParityChoice.ODD,
        ParityChoice.EVEN, ParityChoice.ODD
    ]


@pytest.fixture
def biased_opponent_history() -> List[ParityChoice]:
    """Opponent history with strong EVEN bias."""
    return [
        ParityChoice.EVEN, ParityChoice.EVEN,
        ParityChoice.EVEN, ParityChoice.ODD,
        ParityChoice.EVEN, ParityChoice.EVEN
    ]


# --- Schedule Fixtures ---

@pytest.fixture
def valid_schedule_data(sample_match_id) -> Dict[str, Any]:
    """Valid schedule match data."""
    return {
        "match_id": sample_match_id,
        "player_a_id": "player:P01",
        "player_b_id": "player:P02",
        "priority": 1
    }


# --- Standings Fixtures ---

@pytest.fixture
def sample_standings_data() -> Dict[str, Any]:
    """Sample standings data."""
    return {
        "tournament_id": "TOURNAMENT-1234567890",
        "standings": [
            {
                "agent_id": "player:P01",
                "display_name": "Player 1",
                "matches_played": 3,
                "matches_won": 2,
                "matches_lost": 1,
                "points": 7,
                "win_rate": 0.667
            },
            {
                "agent_id": "player:P02",
                "display_name": "Player 2",
                "matches_played": 3,
                "matches_won": 1,
                "matches_lost": 2,
                "points": 5,
                "win_rate": 0.333
            }
        ],
        "total_matches": 6,
        "completed_matches": 3
    }


# --- Error Fixtures ---

@pytest.fixture
def sample_error_data() -> Dict[str, Any]:
    """Sample error data payload."""
    return {
        "error_code": "TIMEOUT",
        "error_message": "Operation timed out after 2000ms",
        "error_context": {"match_id": "M-001", "round_id": 5},
        "recoverable": True,
        "suggested_action": "Retry the operation"
    }


# --- Heartbeat Fixtures ---

@pytest.fixture
def sample_heartbeat_data() -> Dict[str, Any]:
    """Sample heartbeat data payload."""
    return {
        "agent_status": "idle",
        "uptime_seconds": 3600.5,
        "current_matches": [],
        "performance_stats": {"avg_response_time_ms": 45.2}
    }
