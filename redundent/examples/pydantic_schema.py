from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, Union, Literal
import uuid
import json

from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict, 
    ValidationError
)

# --- Enums ---

class ProtocolVersion(str, Enum):
    V1 = "league.v1"
    V2 = "league.v2"

class AgentRole(str, Enum):
    PLAYER = "player"
    REFEREE = "referee"
    MANAGER = "manager"

class MessageType(str, Enum):
    # Lifecycle
    REGISTER = "LEAGUE_REGISTER_REQUEST"
    REGISTER_ACK = "LEAGUE_REGISTER_RESPONSE"
    HEARTBEAT = "HEARTBEAT"
    
    # Match Flow
    INVITATION = "GAME_INVITATION"
    INVITATION_ACK = "GAME_JOIN_ACK"
    MOVE_CALL = "CHOOSE_PARITY"
    MOVE_RESPONSE = "CHOOSE_PARITY_RESPONSE"
    GAME_OVER = "GAME_OVER"
    MATCH_REPORT = "MATCH_RESULT_REPORT"
    
    # Errors
    ERROR = "ERROR"

# --- Shared Fields & Mixins ---

class BasePayload(BaseModel):
    """Base class for all data payloads to ensure strict config"""
    model_config = ConfigDict(extra="forbid", frozen=True)

# --- Specific Data Payloads ---

class RegistrationData(BasePayload):
    display_name: str = Field(..., min_length=3, max_length=50)
    contact_endpoint: str = Field(..., pattern=r"^http://.*")
    game_types: list[str] = Field(default=["even_odd"])
    capabilities: Dict[str, Any] = Field(default_factory=dict)

class InvitationData(BasePayload):
    match_id: str
    round_id: int
    opponent_id: str
    role_in_match: Literal["A", "B"]
    timeout_ms: int = 2000

class MoveRequestData(BasePayload):
    match_id: str
    round_id: int
    opponent_history_hash: Optional[str] = None # For cache invalidation

class MoveResponseData(BasePayload):
    match_id: str
    parity_choice: Literal["even", "odd"]
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

class MatchResultData(BasePayload):
    match_id: str
    winner_id: Optional[str]
    score_a: int
    score_b: int
    details: Dict[str, Any]

# --- The Main Envelope ---

class MCPEnvelope(BaseModel):
    """
    The Universal Message Wrapper.
    Enforces the protocol structure for all agents.
    """
    protocol: ProtocolVersion = Field(default=ProtocolVersion.V2)
    message_type: MessageType
    sender: str
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    auth_token: Optional[str] = None
    
    # The data field is polymorphic - it can be any dict, 
    # but we validate it manually or via Union if we want strictness.
    data: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    # --- Validators ---

    @field_validator("sender")
    @classmethod
    def validate_sender_format(cls, v: str) -> str:
        """Ensures sender follows 'role:id' format"""
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError("Sender must be in format 'role:unique_id'")
        
        role, _id = parts
        try:
            AgentRole(role) # Validate role is known
        except ValueError:
            raise ValueError(f"Invalid role '{role}'. Must be one of {[r.value for r in AgentRole]}")
            
        if len(_id) < 2:
            raise ValueError("Sender ID is too short")
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
        This acts as a router validation check.
        """
        try:
            if self.message_type == MessageType.REGISTER:
                RegistrationData(**self.data)
            elif self.message_type == MessageType.INVITATION:
                InvitationData(**self.data)
            elif self.message_type == MessageType.MOVE_RESPONSE:
                MoveResponseData(**self.data)
            elif self.message_type == MessageType.GAME_OVER:
                MatchResultData(**self.data)
            # Add other mappings as needed
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
            raise ValueError(f"Schema Validation Error: {e}")

    def create_reply(self, response_type: MessageType, data: Dict[str, Any], sender_id: str) -> 'MCPEnvelope':
        """Factory method to create a response to this specific message"""
        return MCPEnvelope(
            message_type=response_type,
            sender=sender_id,
            conversation_id=self.conversation_id, # Maintain thread
            data=data
        )

# --- Usage Example (Test Block) ---

if __name__ == "__main__":
    try:
        # 1. Create a valid payload
        invitation_payload = {
            "match_id": "M-101",
            "round_id": 1,
            "opponent_id": "player:P02",
            "role_in_match": "A",
            "timeout_ms": 5000
        }

        # 2. Wrap in Envelope
        envelope = MCPEnvelope(
            message_type=MessageType.INVITATION,
            sender="referee:REF-MAIN",
            data=invitation_payload
        )
        
        print("✅ Serialization Success:")
        print(envelope.to_json())

        # 3. Test Invalid Logic (Cross-field validation)
        print("\n🧪 Testing Validation Error:")
        bad_envelope = MCPEnvelope(
            message_type=MessageType.INVITATION, # Expects InvitationData
            sender="referee:REF-MAIN",
            data={"foo": "bar"} # Invalid data
        )
        
    except ValueError as e:
        print(f"❌ Caught Expected Validation Error: {e}")