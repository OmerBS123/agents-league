"""
Tests for Pydantic schemas and validation in shared/schemas.py.
"""

import pytest
import json
from datetime import datetime, timezone
from pydantic import ValidationError

from shared.schemas import (
    MCPEnvelope, MessageType, ParityChoice, AgentStatus, MatchStatus,
    AgentRole, MatchRole, ProtocolVersion,
    RegistrationData, RegistrationResponse, InvitationData, InvitationResponse,
    MoveRequestData, MoveResponseData, GameResultData, MatchResultData,
    ScheduleMatchData, StandingsData, ErrorData, HeartbeatData, BaseResponse,
    create_registration_message, create_move_response
)


class TestEnums:
    """Tests for enum definitions."""

    def test_parity_choice_values(self):
        """Test ParityChoice enum has correct values."""
        assert ParityChoice.EVEN.value == "even"
        assert ParityChoice.ODD.value == "odd"

    def test_agent_status_values(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.OFFLINE.value == "offline"
        assert AgentStatus.ERROR.value == "error"

    def test_match_status_values(self):
        """Test MatchStatus enum values."""
        assert MatchStatus.SCHEDULED.value == "scheduled"
        assert MatchStatus.IN_PROGRESS.value == "in_progress"
        assert MatchStatus.COMPLETED.value == "completed"
        assert MatchStatus.CANCELLED.value == "cancelled"

    def test_message_type_values(self):
        """Test key MessageType enum values."""
        assert MessageType.REGISTER.value == "LEAGUE_REGISTER_REQUEST"
        assert MessageType.MOVE_CALL.value == "CHOOSE_PARITY"
        assert MessageType.GAME_OVER.value == "GAME_OVER"

    def test_agent_role_values(self):
        """Test AgentRole enum values."""
        assert AgentRole.PLAYER.value == "player"
        assert AgentRole.REFEREE.value == "referee"
        assert AgentRole.MANAGER.value == "manager"


class TestMCPEnvelope:
    """Tests for the main MCPEnvelope schema."""

    def test_valid_envelope_creation(self, sample_player_id, sample_manager_id):
        """Test creating a valid envelope with minimal data."""
        envelope = MCPEnvelope(
            message_type=MessageType.HEARTBEAT,
            sender=sample_player_id,
            data={}
        )
        assert envelope.sender == sample_player_id
        assert envelope.message_type == MessageType.HEARTBEAT
        assert envelope.protocol == ProtocolVersion.V2

    def test_envelope_with_recipient(self, sample_player_id, sample_manager_id):
        """Test envelope with explicit recipient."""
        envelope = MCPEnvelope(
            message_type=MessageType.REGISTER,
            sender=sample_player_id,
            recipient=sample_manager_id,
            data={
                "display_name": "Test",
                "contact_endpoint": "http://localhost:8101",
                "strategies": ["random"]
            }
        )
        assert envelope.recipient == sample_manager_id

    def test_invalid_sender_format_no_colon(self):
        """Test that sender without colon is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MCPEnvelope(
                message_type=MessageType.HEARTBEAT,
                sender="invalidformat",
                data={}
            )
        assert "must be in format" in str(exc_info.value)

    def test_invalid_sender_format_wrong_role(self):
        """Test that sender with invalid role is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MCPEnvelope(
                message_type=MessageType.HEARTBEAT,
                sender="invalid:P01",
                data={}
            )
        assert "Invalid role" in str(exc_info.value)

    def test_invalid_sender_empty_id(self):
        """Test that sender with empty ID is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MCPEnvelope(
                message_type=MessageType.HEARTBEAT,
                sender="player:",
                data={}
            )
        assert "cannot be empty" in str(exc_info.value)

    def test_broadcast_recipient_allowed(self, sample_player_id):
        """Test that 'broadcast' is valid recipient."""
        envelope = MCPEnvelope(
            message_type=MessageType.HEARTBEAT,
            sender=sample_player_id,
            recipient="broadcast",
            data={}
        )
        assert envelope.recipient == "broadcast"

    def test_envelope_json_serialization(self, registration_envelope):
        """Test envelope serializes to valid JSON."""
        json_str = registration_envelope.to_json()
        data = json.loads(json_str)

        assert data["message_type"] == MessageType.REGISTER.value
        assert "timestamp" in data
        assert "conversation_id" in data

    def test_envelope_json_deserialization(self, registration_envelope):
        """Test envelope deserializes from JSON correctly."""
        json_str = registration_envelope.to_json()
        parsed = MCPEnvelope.from_json(json_str)

        assert parsed.message_type == registration_envelope.message_type
        assert parsed.sender == registration_envelope.sender

    def test_envelope_create_reply(self, registration_envelope, sample_manager_id):
        """Test creating a reply envelope."""
        reply = registration_envelope.create_reply(
            response_type=MessageType.REGISTER_ACK,
            data={"accepted": True, "agent_id": "player:P01", "league_status": "active"},
            sender_id=sample_manager_id
        )

        assert reply.message_type == MessageType.REGISTER_ACK
        assert reply.sender == sample_manager_id
        assert reply.recipient == registration_envelope.sender
        assert reply.conversation_id == registration_envelope.conversation_id

    def test_envelope_create_error_reply(self, registration_envelope, sample_manager_id):
        """Test creating an error reply envelope."""
        error_reply = registration_envelope.create_error_reply(
            error_code="VALIDATION_ERROR",
            error_message="Invalid data",
            sender_id=sample_manager_id
        )

        assert error_reply.message_type == MessageType.ERROR
        assert error_reply.data["error_code"] == "VALIDATION_ERROR"

    def test_timestamp_parsing_iso_format(self, sample_player_id):
        """Test timestamp parsing from ISO string."""
        envelope = MCPEnvelope(
            message_type=MessageType.HEARTBEAT,
            sender=sample_player_id,
            timestamp="2026-01-02T12:00:00+00:00",
            data={}
        )
        assert isinstance(envelope.timestamp, datetime)

    def test_timestamp_parsing_with_z(self, sample_player_id):
        """Test timestamp parsing with Z suffix."""
        envelope = MCPEnvelope(
            message_type=MessageType.HEARTBEAT,
            sender=sample_player_id,
            timestamp="2026-01-02T12:00:00Z",
            data={}
        )
        assert isinstance(envelope.timestamp, datetime)


class TestRegistrationData:
    """Tests for RegistrationData payload."""

    def test_valid_registration(self):
        """Test valid registration data."""
        data = RegistrationData(
            display_name="Test Player",
            contact_endpoint="http://localhost:8101",
            strategies=["random"]
        )
        assert data.display_name == "Test Player"

    def test_registration_display_name_too_short(self):
        """Test display name minimum length."""
        with pytest.raises(ValidationError):
            RegistrationData(
                display_name="AB",  # Too short (min 3)
                contact_endpoint="http://localhost:8101",
                strategies=["random"]
            )

    def test_registration_display_name_too_long(self):
        """Test display name maximum length."""
        with pytest.raises(ValidationError):
            RegistrationData(
                display_name="A" * 51,  # Too long (max 50)
                contact_endpoint="http://localhost:8101",
                strategies=["random"]
            )

    def test_registration_invalid_endpoint_format(self):
        """Test contact endpoint must match pattern."""
        with pytest.raises(ValidationError):
            RegistrationData(
                display_name="Test Player",
                contact_endpoint="not-a-valid-url",
                strategies=["random"]
            )

    def test_registration_invalid_strategy(self):
        """Test unsupported strategy is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RegistrationData(
                display_name="Test Player",
                contact_endpoint="http://localhost:8101",
                strategies=["invalid_strategy"]
            )
        assert "Unsupported strategy" in str(exc_info.value)

    def test_registration_multiple_strategies(self):
        """Test multiple valid strategies."""
        data = RegistrationData(
            display_name="Test Player",
            contact_endpoint="http://localhost:8101",
            strategies=["random", "history"]
        )
        assert len(data.strategies) == 2


class TestInvitationData:
    """Tests for InvitationData payload."""

    def test_valid_invitation(self, sample_match_id):
        """Test valid invitation data."""
        data = InvitationData(
            match_id=sample_match_id,
            round_id=1,
            opponent_id="player:P02",
            role_in_match="A",
            timeout_ms=2000
        )
        assert data.match_id == sample_match_id

    def test_invitation_invalid_match_id(self):
        """Test match ID must match pattern."""
        with pytest.raises(ValidationError):
            InvitationData(
                match_id="invalid",  # Must be M-XXX
                round_id=1,
                opponent_id="player:P02",
                role_in_match="A"
            )

    def test_invitation_invalid_round_id(self, sample_match_id):
        """Test round ID must be positive."""
        with pytest.raises(ValidationError):
            InvitationData(
                match_id=sample_match_id,
                round_id=0,  # Must be >= 1
                opponent_id="player:P02",
                role_in_match="A"
            )

    def test_invitation_timeout_bounds(self, sample_match_id):
        """Test timeout within allowed range."""
        with pytest.raises(ValidationError):
            InvitationData(
                match_id=sample_match_id,
                round_id=1,
                opponent_id="player:P02",
                role_in_match="A",
                timeout_ms=100  # Too low (min 500)
            )


class TestMoveData:
    """Tests for MoveRequestData and MoveResponseData."""

    def test_valid_move_request(self, sample_match_id):
        """Test valid move request data."""
        data = MoveRequestData(
            match_id=sample_match_id,
            round_id=5,
            opponent_history=[ParityChoice.EVEN, ParityChoice.ODD],
            timeout_ms=2000
        )
        assert len(data.opponent_history) == 2

    def test_valid_move_response(self, sample_match_id):
        """Test valid move response data."""
        data = MoveResponseData(
            match_id=sample_match_id,
            round_id=5,
            parity_choice=ParityChoice.EVEN,
            confidence=0.85,
            reasoning="Pattern detected",
            strategy_used="history"
        )
        assert data.parity_choice == ParityChoice.EVEN
        assert data.confidence == 0.85

    def test_move_response_confidence_bounds(self, sample_match_id):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            MoveResponseData(
                match_id=sample_match_id,
                round_id=5,
                parity_choice=ParityChoice.EVEN,
                confidence=1.5,  # Too high
                strategy_used="random"
            )


class TestGameResultData:
    """Tests for GameResultData payload."""

    def test_valid_game_result(self, sample_match_id):
        """Test valid game result."""
        result = GameResultData(
            match_id=sample_match_id,
            round_id=1,
            player_a_choice=ParityChoice.EVEN,
            player_b_choice=ParityChoice.ODD,
            random_number=7,
            winner_id="player:P02",
            calculation="(0+1+7) = 8 (even)",
            round_duration_ms=145.3
        )
        assert result.random_number == 7

    def test_game_result_random_number_bounds(self, sample_match_id):
        """Test random number must be 1-10."""
        with pytest.raises(ValidationError):
            GameResultData(
                match_id=sample_match_id,
                round_id=1,
                player_a_choice=ParityChoice.EVEN,
                player_b_choice=ParityChoice.ODD,
                random_number=15,  # Too high
                calculation="test",
                round_duration_ms=100.0
            )


class TestFactoryFunctions:
    """Tests for schema factory functions."""

    def test_create_registration_message(self):
        """Test registration message factory."""
        msg = create_registration_message(
            sender_id="player:P01",
            display_name="Test Player",
            contact_endpoint="http://localhost:8101",
            strategies=["random", "history"]
        )

        assert msg.message_type == MessageType.REGISTER
        assert msg.sender == "player:P01"
        assert msg.recipient == "manager:league"
        assert msg.data["display_name"] == "Test Player"

    def test_create_move_response(self):
        """Test move response factory."""
        msg = create_move_response(
            sender_id="player:P01",
            match_id="M-001",
            round_id=3,
            parity_choice=ParityChoice.ODD,
            strategy_used="llm",
            conversation_id="test-conv-id",
            confidence=0.9,
            reasoning="LLM analysis"
        )

        assert msg.message_type == MessageType.MOVE_RESPONSE
        assert msg.data["parity_choice"] == ParityChoice.ODD.value
        assert msg.conversation_id == "test-conv-id"


class TestCrossFieldValidation:
    """Tests for envelope cross-field validation."""

    def test_register_message_validates_data(self, sample_player_id, sample_manager_id):
        """Test REGISTER message validates RegistrationData."""
        # Valid data should pass
        envelope = MCPEnvelope(
            message_type=MessageType.REGISTER,
            sender=sample_player_id,
            recipient=sample_manager_id,
            data={
                "display_name": "Test Player",
                "contact_endpoint": "http://localhost:8101",
                "strategies": ["random"]
            }
        )
        assert envelope is not None

    def test_register_message_rejects_invalid_data(self, sample_player_id, sample_manager_id):
        """Test REGISTER message rejects invalid data."""
        with pytest.raises(ValidationError) as exc_info:
            MCPEnvelope(
                message_type=MessageType.REGISTER,
                sender=sample_player_id,
                recipient=sample_manager_id,
                data={
                    "display_name": "AB",  # Too short
                    "contact_endpoint": "http://localhost:8101",
                    "strategies": ["random"]
                }
            )
        assert "Invalid payload" in str(exc_info.value)

    def test_move_call_validates_data(self, sample_referee_id, sample_player_id):
        """Test MOVE_CALL message validates MoveRequestData."""
        envelope = MCPEnvelope(
            message_type=MessageType.MOVE_CALL,
            sender=sample_referee_id,
            recipient=sample_player_id,
            data={
                "match_id": "M-001",
                "round_id": 1,
                "opponent_history": [],
                "timeout_ms": 2000
            }
        )
        assert envelope is not None


class TestBaseResponse:
    """Tests for BaseResponse API wrapper."""

    def test_success_response(self):
        """Test successful response creation."""
        response = BaseResponse(
            status="success",
            request_id="test-123",
            data={"result": "ok"}
        )
        assert response.status == "success"
        assert response.error is None

    def test_error_response(self):
        """Test error response creation."""
        response = BaseResponse(
            status="error",
            request_id="test-123",
            error="Something went wrong"
        )
        assert response.status == "error"
        assert response.error == "Something went wrong"

    def test_response_has_timestamp(self):
        """Test response includes timestamp."""
        response = BaseResponse(
            status="success",
            request_id="test-123"
        )
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)
