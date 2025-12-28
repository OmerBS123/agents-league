"""
Strict Pydantic models for the Distributed AI Agent League System.
Implements the MCPEnvelope pattern with cross-field validation.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, Union, Literal, List
import uuid
import json

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    field_serializer,
    model_validator,
    ConfigDict,
    ValidationError
)

from consts import PROTOCOL_VERSION, AVAILABLE_STRATEGIES, GAME_TYPE


# --- Enums ---

class ProtocolVersion(str, Enum):
    """Protocol version for message compatibility"""
    V1 = "league.v1"
    V2 = "league.v2"


class AgentRole(str, Enum):
    """Agent types in the league system"""
    PLAYER = "player"
    REFEREE = "referee"
    MANAGER = "manager"


class MessageType(str, Enum):
    """Message types for agent communication"""
    # Lifecycle Management
    REGISTER = "LEAGUE_REGISTER_REQUEST"
    REGISTER_ACK = "LEAGUE_REGISTER_RESPONSE"
    HEARTBEAT = "HEARTBEAT"
    HEALTH_CHECK = "HEALTH_CHECK"

    # Match Flow
    INVITATION = "GAME_INVITATION"
    INVITATION_ACK = "GAME_JOIN_ACK"
    MOVE_CALL = "CHOOSE_PARITY"
    MOVE_RESPONSE = "CHOOSE_PARITY_RESPONSE"
    GAME_OVER = "GAME_OVER"
    MATCH_REPORT = "MATCH_RESULT_REPORT"

    # Tournament Management
    SCHEDULE_MATCH = "SCHEDULE_MATCH"
    SCHEDULE_ACK = "SCHEDULE_ACK"
    TOURNAMENT_START = "TOURNAMENT_START"
    TOURNAMENT_END = "TOURNAMENT_END"
    STANDINGS_UPDATE = "STANDINGS_UPDATE"

    # Errors and Status
    ERROR = "ERROR"
    STATUS_UPDATE = "STATUS_UPDATE"


class MatchRole(str, Enum):
    """Player roles in a specific match"""
    PLAYER_A = "A"
    PLAYER_B = "B"


class ParityChoice(str, Enum):
    """Game move choices"""
    EVEN = "even"
    ODD = "odd"


class AgentStatus(str, Enum):
    """Agent operational status"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class MatchStatus(str, Enum):
    """Match execution status"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


# --- Base Models ---

class BasePayload(BaseModel):
    """Base class for all data payloads with strict configuration"""
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )


class BaseResponse(BaseModel):
    """Standard API response wrapper"""
    status: str = "success"
    request_id: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        """Serialize datetime to ISO format string"""
        return dt.isoformat()

    model_config = ConfigDict(
        populate_by_name=True
    )


# --- Specific Data Payloads ---

class RegistrationData(BasePayload):
    """Player registration with League Manager"""
    display_name: str = Field(..., min_length=3, max_length=50)
    contact_endpoint: str = Field(..., pattern=r"^http://.*:\d+$")
    game_types: List[str] = Field(default=[GAME_TYPE])
    strategies: List[str] = Field(default_factory=list)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    agent_version: str = Field(default="1.0.0")

    @field_validator("strategies")
    @classmethod
    def validate_strategies(cls, v: List[str]) -> List[str]:
        """Ensure all strategies are supported"""
        for strategy in v:
            if strategy not in AVAILABLE_STRATEGIES:
                raise ValueError(f"Unsupported strategy '{strategy}'. Must be one of {AVAILABLE_STRATEGIES}")
        return v


class RegistrationResponse(BasePayload):
    """League Manager response to registration"""
    accepted: bool
    agent_id: str
    league_status: str
    next_match_eta: Optional[int] = None  # seconds
    tournament_info: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


class InvitationData(BasePayload):
    """Match invitation from Referee to Players"""
    match_id: str = Field(..., pattern=r"^M-\d+$")
    round_id: int = Field(..., ge=1)
    opponent_id: str = Field(..., pattern=r"^player:.+$")
    role_in_match: MatchRole
    timeout_ms: int = Field(default=2000, ge=500, le=10000)
    match_context: Dict[str, Any] = Field(default_factory=dict)


class InvitationResponse(BasePayload):
    """Player response to match invitation"""
    match_id: str
    accepted: bool
    reason: Optional[str] = None
    estimated_ready_time: Optional[int] = None  # milliseconds


class MoveRequestData(BasePayload):
    """Move request from Referee to Player"""
    match_id: str = Field(..., pattern=r"^M-\d+$")
    round_id: int = Field(..., ge=1)
    opponent_history: List[ParityChoice] = Field(default_factory=list)
    opponent_history_hash: Optional[str] = None
    timeout_ms: int = Field(default=2000)


class MoveResponseData(BasePayload):
    """Player move response"""
    match_id: str
    round_id: int
    parity_choice: ParityChoice
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, max_length=500)
    strategy_used: str
    decision_time_ms: Optional[float] = None


class GameResultData(BasePayload):
    """Single game round result"""
    match_id: str
    round_id: int
    player_a_choice: ParityChoice
    player_b_choice: ParityChoice
    random_number: int = Field(..., ge=1, le=10)
    winner_id: Optional[str] = None
    calculation: str  # Show how winner was determined
    round_duration_ms: float


class MatchResultData(BasePayload):
    """Complete match results"""
    match_id: str
    winner_id: Optional[str] = None
    loser_id: Optional[str] = None
    final_score: Dict[str, int]  # {"player:P1": 3, "player:P2": 1}
    total_rounds: int
    match_duration_ms: float
    game_results: List[GameResultData]
    match_stats: Dict[str, Any] = Field(default_factory=dict)


class ScheduleMatchData(BasePayload):
    """Match scheduling request from Manager to Referee"""
    match_id: str
    player_a_id: str = Field(..., pattern=r"^player:.+$")
    player_b_id: str = Field(..., pattern=r"^player:.+$")
    scheduled_time: Optional[datetime] = None
    priority: int = Field(default=1, ge=1, le=5)
    match_config: Dict[str, Any] = Field(default_factory=dict)


class StandingsData(BasePayload):
    """Tournament standings"""
    tournament_id: str
    standings: List[Dict[str, Any]]  # [{"agent_id": "player:P1", "wins": 3, "losses": 1, "points": 6}]
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_matches: int
    completed_matches: int

    @field_serializer('last_updated')
    def serialize_last_updated(self, dt: datetime) -> str:
        """Serialize datetime to ISO format string"""
        return dt.isoformat()


class ErrorData(BasePayload):
    """Error information payload"""
    error_code: str
    error_message: str
    error_context: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = True
    suggested_action: Optional[str] = None


class HeartbeatData(BasePayload):
    """Agent heartbeat data"""
    agent_status: AgentStatus
    uptime_seconds: float
    current_matches: List[str] = Field(default_factory=list)
    performance_stats: Dict[str, Any] = Field(default_factory=dict)


# --- Main Envelope ---

class MCPEnvelope(BaseModel):
    """
    Universal message wrapper for all agent communication.
    Enforces protocol structure and cross-field validation.
    """
    protocol: ProtocolVersion = Field(default=ProtocolVersion.V2)
    message_type: MessageType
    sender: str
    recipient: Optional[str] = None
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    auth_token: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)

    # Polymorphic data field
    data: Dict[str, Any] = Field(default_factory=dict)

    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime) -> str:
        """Serialize datetime to ISO format string"""
        return dt.isoformat()

    model_config = ConfigDict(
        populate_by_name=True
    )

    # --- Validators ---

    @field_validator("sender")
    @classmethod
    def validate_sender_format(cls, v: str) -> str:
        """Ensures sender follows 'role:id' format"""
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError("Sender must be in format 'role:unique_id'")

        role, agent_id = parts
        try:
            AgentRole(role)
        except ValueError:
            raise ValueError(f"Invalid role '{role}'. Must be one of {[r.value for r in AgentRole]}")

        if len(agent_id) < 1:
            raise ValueError("Agent ID cannot be empty")

        return v

    @field_validator("recipient")
    @classmethod
    def validate_recipient_format(cls, v: Optional[str]) -> Optional[str]:
        """Validates recipient format if provided"""
        if v is None:
            return v

        if v == "broadcast":
            return v

        # Same format as sender
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError("Recipient must be in format 'role:unique_id' or 'broadcast'")

        role, agent_id = parts
        try:
            AgentRole(role)
        except ValueError:
            raise ValueError(f"Invalid recipient role '{role}'. Must be one of {[r.value for r in AgentRole]}")

        return v

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        """Robust timestamp parsing"""
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("Invalid ISO 8601 timestamp format")
        raise ValueError("Timestamp must be string or datetime object")

    @model_validator(mode="after")
    def validate_data_content(self) -> 'MCPEnvelope':
        """
        Cross-field validation: Ensures 'data' matches 'message_type'.
        Validates payload structure based on message type.
        """
        try:
            if self.message_type == MessageType.REGISTER:
                RegistrationData(**self.data)
            elif self.message_type == MessageType.REGISTER_ACK:
                RegistrationResponse(**self.data)
            elif self.message_type == MessageType.INVITATION:
                InvitationData(**self.data)
            elif self.message_type == MessageType.INVITATION_ACK:
                InvitationResponse(**self.data)
            elif self.message_type == MessageType.MOVE_CALL:
                MoveRequestData(**self.data)
            elif self.message_type == MessageType.MOVE_RESPONSE:
                MoveResponseData(**self.data)
            elif self.message_type == MessageType.GAME_OVER:
                GameResultData(**self.data)
            elif self.message_type == MessageType.MATCH_REPORT:
                MatchResultData(**self.data)
            elif self.message_type == MessageType.SCHEDULE_MATCH:
                ScheduleMatchData(**self.data)
            elif self.message_type == MessageType.STANDINGS_UPDATE:
                StandingsData(**self.data)
            elif self.message_type == MessageType.ERROR:
                ErrorData(**self.data)
            elif self.message_type == MessageType.HEARTBEAT:
                HeartbeatData(**self.data)
            # Other message types may have flexible data
        except ValidationError as e:
            raise ValueError(f"Invalid payload for message type {self.message_type}: {e}")

        return self

    # --- Helper Methods ---

    def to_json(self) -> str:
        """Safe JSON serialization"""
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPEnvelope':
        """Safe JSON deserialization factory"""
        try:
            data = json.loads(json_str)
            return cls(**data)
        except json.JSONDecodeError:
            raise ValueError("Malformed JSON string")
        except ValidationError as e:
            raise ValueError(f"Schema validation error: {e}")

    def create_reply(
        self,
        response_type: MessageType,
        data: Dict[str, Any],
        sender_id: str,
        recipient_id: Optional[str] = None
    ) -> 'MCPEnvelope':
        """
        Factory method to create a response maintaining conversation thread.

        Args:
            response_type: Type of response message
            data: Response payload
            sender_id: ID of responding agent
            recipient_id: Specific recipient (defaults to original sender)

        Returns:
            New MCPEnvelope with linked conversation
        """
        return MCPEnvelope(
            message_type=response_type,
            sender=sender_id,
            recipient=recipient_id or self.sender,
            conversation_id=self.conversation_id,  # Maintain thread
            data=data
        )

    def create_error_reply(
        self,
        error_code: str,
        error_message: str,
        sender_id: str,
        recoverable: bool = True,
        error_context: Optional[Dict[str, Any]] = None
    ) -> 'MCPEnvelope':
        """
        Factory method to create error responses.

        Args:
            error_code: Error classification code
            error_message: Human-readable error description
            sender_id: ID of agent reporting error
            recoverable: Whether the error can be recovered from
            error_context: Additional error context

        Returns:
            Error MCPEnvelope
        """
        error_data = ErrorData(
            error_code=error_code,
            error_message=error_message,
            error_context=error_context or {},
            recoverable=recoverable
        )

        return self.create_reply(
            response_type=MessageType.ERROR,
            data=error_data.model_dump(),
            sender_id=sender_id
        )


# --- Factory Functions ---

def create_registration_message(
    sender_id: str,
    display_name: str,
    contact_endpoint: str,
    strategies: List[str],
    recipient_id: str = "manager:league"
) -> MCPEnvelope:
    """Create a registration message for a player"""
    registration_data = RegistrationData(
        display_name=display_name,
        contact_endpoint=contact_endpoint,
        strategies=strategies
    )

    return MCPEnvelope(
        message_type=MessageType.REGISTER,
        sender=sender_id,
        recipient=recipient_id,
        data=registration_data.model_dump()
    )


def create_move_response(
    sender_id: str,
    match_id: str,
    round_id: int,
    parity_choice: ParityChoice,
    strategy_used: str,
    conversation_id: str,
    confidence: float = 0.5,
    reasoning: Optional[str] = None
) -> MCPEnvelope:
    """Create a move response message"""
    move_data = MoveResponseData(
        match_id=match_id,
        round_id=round_id,
        parity_choice=parity_choice,
        confidence=confidence,
        reasoning=reasoning,
        strategy_used=strategy_used
    )

    return MCPEnvelope(
        message_type=MessageType.MOVE_RESPONSE,
        sender=sender_id,
        conversation_id=conversation_id,
        data=move_data.model_dump()
    )