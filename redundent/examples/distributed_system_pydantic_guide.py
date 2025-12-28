"""
Comprehensive Pydantic V2 Best Practices for Distributed Systems
================================================================

This implementation demonstrates advanced Pydantic patterns for:
1. Model Design: Strict typing, frozen models, extra field handling, inheritance
2. Validation: Field validators, model validators, cross-field validation
3. Serialization: JSON encoding/decoding, datetime handling, custom serializers
4. Error Handling: Validation errors, custom messages, error propagation
5. Performance: Optimal configurations for high-throughput scenarios
6. Protocol Design: Message envelopes and polymorphic data for agent communication

Based on official Pydantic documentation:
- Models: docs.pydantic.dev/latest/concepts/models
- Configuration: docs.pydantic.dev/latest/concepts/config
- Validators: docs.pydantic.dev/latest/concepts/validators
- Serialization: docs.pydantic.dev/latest/concepts/serialization
- Error Handling: docs.pydantic.dev/latest/errors/errors
- Performance: docs.pydantic.dev/latest/concepts/performance
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Annotated, Literal, Optional, Union
from uuid import uuid4, UUID
import json
from decimal import Decimal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    field_serializer,
    model_serializer,
    ValidationError,
    AfterValidator,
    BeforeValidator,
    WrapValidator,
    PlainSerializer,
    WrapSerializer,
    SerializeAsAny,
    StrictInt,
    StrictStr,
    StrictFloat,
    StrictBool
)
from pydantic_core import PydanticCustomError
from typing_extensions import Self

# ============================================================================
# 1. MODEL DESIGN PATTERNS
# ============================================================================

class AgentRole(str, Enum):
    """Strictly typed enumeration for agent roles."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    OBSERVER = "observer"
    GATEWAY = "gateway"

class BaseProtocolMessage(BaseModel):
    """
    Base protocol message with strict validation and frozen configuration.

    Key patterns demonstrated:
    - Frozen models for immutability
    - Strict typing with custom types
    - Extra field handling
    - Field constraints and validation
    """
    model_config = ConfigDict(
        # Performance optimizations
        frozen=True,           # Immutable for thread safety
        extra="forbid",        # Strict protocol compliance
        str_strip_whitespace=True,  # Clean inputs
        validate_assignment=True,   # Re-validate on assignment
        # JSON optimization
        cache_strings=True,    # Performance boost for repeated strings
        # Validation behavior
        strict=False,          # Allow some coercion for JSON compatibility
        use_enum_values=True,  # Serialize enums as values
    )

    # Required protocol fields with strict validation
    message_id: Annotated[
        UUID,
        Field(description="Unique message identifier", default_factory=uuid4)
    ]
    timestamp: Annotated[
        datetime,
        Field(description="Message creation timestamp", default_factory=lambda: datetime.now(timezone.utc))
    ]
    version: Annotated[
        StrictStr,
        Field(description="Protocol version", pattern=r"^\d+\.\d+\.\d+$", default="1.0.0")
    ]
    sender_id: Annotated[
        StrictStr,
        Field(description="Sending agent identifier", min_length=1, max_length=128)
    ]

    @field_validator('timestamp', mode='after')
    @classmethod
    def ensure_utc_timezone(cls, v: datetime) -> datetime:
        """Ensure all timestamps are UTC for distributed consistency."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

class AgentMetadata(BaseModel):
    """
    Agent metadata with inheritance and validation patterns.

    Demonstrates:
    - Nested model design
    - Cross-field validation
    - Custom validators for business logic
    """
    model_config = ConfigDict(frozen=True, extra="allow")

    agent_id: StrictStr
    role: AgentRole
    capabilities: list[StrictStr] = Field(default_factory=list)
    load_factor: Annotated[
        StrictFloat,
        Field(ge=0.0, le=1.0, description="Current agent load (0.0 to 1.0)")
    ]
    max_concurrent_tasks: StrictInt = Field(ge=1, le=1000, default=10)

    @model_validator(mode='after')
    def validate_role_capabilities(self) -> Self:
        """Cross-field validation ensuring role-capability consistency."""
        role_requirements = {
            AgentRole.COORDINATOR: ["coordination", "task_distribution"],
            AgentRole.WORKER: ["task_execution"],
            AgentRole.OBSERVER: ["monitoring", "logging"],
            AgentRole.GATEWAY: ["request_routing", "load_balancing"]
        }

        required_caps = role_requirements.get(self.role, [])
        missing_caps = set(required_caps) - set(self.capabilities)

        if missing_caps:
            raise ValueError(f"Role {self.role.value} requires capabilities: {missing_caps}")

        return self

# ============================================================================
# 2. VALIDATION PATTERNS
# ============================================================================

def validate_currency_amount(value: Any) -> Decimal:
    """
    Custom validator for financial amounts with precision handling.

    Demonstrates before validation with type coercion.
    """
    if isinstance(value, (int, float)):
        value = str(value)
    if isinstance(value, str):
        try:
            decimal_value = Decimal(value)
            if decimal_value < 0:
                raise ValueError("Amount cannot be negative")
            # Ensure proper precision for financial calculations
            return decimal_value.quantize(Decimal('0.01'))
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid currency amount: {e}")
    raise ValueError(f"Cannot convert {type(value)} to currency amount")

def validate_agent_endpoint(value: Any, handler) -> str:
    """
    Wrap validator for endpoint URL validation.

    Demonstrates wrap validation with fallback logic.
    """
    if isinstance(value, str) and value.startswith("agent://"):
        # Custom protocol handling
        return value

    # Delegate to standard validation for http/https
    result = handler(value)

    # Post-process: ensure no trailing slash
    return result.rstrip('/')

class TaskRequest(BaseModel):
    """
    Task request with comprehensive field validation patterns.

    Demonstrates:
    - Multiple validator types
    - Custom error messages
    - Context-aware validation
    """
    model_config = ConfigDict(
        str_max_length=10000,  # Global string limit
        validate_default=True,  # Validate default values
    )

    task_id: UUID = Field(default_factory=uuid4)
    task_type: StrictStr = Field(min_length=1, max_length=100)
    priority: Annotated[
        StrictInt,
        Field(ge=1, le=10, description="Task priority (1=lowest, 10=highest)")
    ]

    # Financial amount with custom validation
    budget: Annotated[
        Decimal,
        BeforeValidator(validate_currency_amount),
        Field(description="Task budget in USD")
    ]

    # Endpoint with wrap validation
    callback_url: Annotated[
        str,
        WrapValidator(validate_agent_endpoint),
        Field(description="Callback endpoint for task completion")
    ]

    # Validated parameters
    parameters: dict[str, Any] = Field(default_factory=dict)
    deadline: Optional[datetime] = None

    @field_validator('task_type', mode='after')
    @classmethod
    def validate_task_type_format(cls, v: str) -> str:
        """Validate task type follows naming convention."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise PydanticCustomError(
                'invalid_task_type',
                'Task type must be alphanumeric with underscores/hyphens: {input}',
                {'input': v}
            )
        return v.lower()

    @field_validator('deadline', mode='after')
    @classmethod
    def validate_deadline_future(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure deadline is in the future."""
        if v is not None and v <= datetime.now(timezone.utc):
            raise ValueError("Deadline must be in the future")
        return v

    @model_validator(mode='after')
    def validate_budget_priority_relationship(self) -> Self:
        """Business rule: high priority tasks require sufficient budget."""
        if self.priority >= 8 and self.budget < Decimal('1000.00'):
            raise ValueError("High priority tasks (8+) require budget >= $1000")
        return self

# ============================================================================
# 3. SERIALIZATION PATTERNS
# ============================================================================

class DecimalSerializer:
    """Custom serializer for Decimal values to ensure JSON compatibility."""

    @staticmethod
    def serialize_decimal(value: Decimal, info=None) -> str:
        """Serialize Decimal as string to preserve precision."""
        return str(value)

class TimestampSerializer:
    """Custom timestamp serialization for consistent format across agents."""

    @staticmethod
    def serialize_timestamp(value: datetime, info=None) -> str:
        """Serialize datetime in ISO format with UTC timezone."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

class TaskResult(BaseModel):
    """
    Task result with custom serialization patterns.

    Demonstrates:
    - Custom field serializers
    - Model-level serialization
    - Performance-optimized serialization
    """
    model_config = ConfigDict(
        # Serialization optimization
        ser_json_bytes=False,  # Return str not bytes
        ser_json_timedelta='float',  # Timedelta as seconds
    )

    task_id: UUID
    status: Literal["completed", "failed", "timeout"]

    # Custom serialized fields
    completion_time: Annotated[
        datetime,
        PlainSerializer(TimestampSerializer.serialize_timestamp)
    ]

    processing_cost: Annotated[
        Decimal,
        PlainSerializer(DecimalSerializer.serialize_decimal)
    ]

    result_data: dict[str, Any] = Field(default_factory=dict)
    error_details: Optional[str] = None

    @field_serializer('result_data', mode='wrap')
    def serialize_result_data(self, value: dict, handler, info) -> dict:
        """Custom serialization for result data with filtering."""
        # Filter sensitive data in serialization
        filtered_data = {
            k: v for k, v in value.items()
            if not k.startswith('_private')
        }

        # Apply default serialization to filtered data
        return handler(filtered_data)

    @model_serializer(mode='wrap')
    def serialize_model(self, handler, info) -> dict:
        """Model-level serialization with metadata injection."""
        data = handler(self)

        # Add serialization metadata
        data['_meta'] = {
            'serialized_at': datetime.now(timezone.utc).isoformat(),
            'serializer_version': '1.0'
        }

        return data

# ============================================================================
# 4. ERROR HANDLING PATTERNS
# ============================================================================

class DistributedSystemError(Exception):
    """Base exception for distributed system errors."""
    pass

class ValidationErrorHandler:
    """
    Centralized error handling for validation errors with custom messages.

    Based on: docs.pydantic.dev/latest/errors/errors
    """

    CUSTOM_ERROR_MESSAGES = {
        'string_too_long': 'Text is too long (max {max_length} characters)',
        'missing': 'This field is required for protocol compliance',
        'value_error': 'Invalid value: {error}',
        'invalid_task_type': 'Task type format is invalid: {input}',
        'greater_than_equal': 'Value must be at least {ge}',
        'less_than_equal': 'Value must be at most {le}',
    }

    @classmethod
    def format_validation_error(cls, exc: ValidationError) -> dict[str, Any]:
        """
        Format validation errors for distributed system logging.

        Returns structured error information for agent communication.
        """
        formatted_errors = []

        for error in exc.errors():
            error_type = error['type']
            custom_message = cls.CUSTOM_ERROR_MESSAGES.get(error_type)

            formatted_error = {
                'field_path': '.'.join(str(loc) for loc in error['loc']),
                'error_type': error_type,
                'input_value': error['input'],
                'message': custom_message.format(**error.get('ctx', {})) if custom_message else error['msg'],
            }
            formatted_errors.append(formatted_error)

        return {
            'error_count': exc.error_count(),
            'errors': formatted_errors,
            'raw_errors': exc.errors()
        }

    @classmethod
    def create_error_response(cls, exc: ValidationError, request_id: UUID) -> dict:
        """Create standardized error response for agent communication."""
        error_data = cls.format_validation_error(exc)

        return {
            'success': False,
            'request_id': str(request_id),
            'error_type': 'validation_error',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **error_data
        }

# ============================================================================
# 5. PERFORMANCE OPTIMIZATION PATTERNS
# ============================================================================

class HighThroughputConfig(ConfigDict):
    """
    Optimized configuration for high-throughput distributed systems.

    Based on: docs.pydantic.dev/latest/concepts/performance
    """
    # Validation optimizations
    validate_default: bool = False  # Skip default validation for speed
    use_enum_values: bool = True   # Faster enum serialization

    # JSON optimizations
    cache_strings: bool = True     # Cache repeated strings
    ser_json_bytes: bool = False   # Return strings not bytes

    # Memory optimizations
    extra: str = "ignore"          # Don't store extra fields
    frozen: bool = True            # Immutable for memory efficiency

class OptimizedMessage(BaseModel):
    """
    Message model optimized for high-throughput scenarios.

    Demonstrates performance best practices:
    - Concrete types over abstract (list vs Sequence)
    - Minimal validation overhead
    - Efficient serialization
    """
    model_config = HighThroughputConfig()

    # Use concrete types for better performance
    message_type: Literal["request", "response", "notification"]
    payload: dict[str, Any]  # dict instead of Mapping
    headers: list[tuple[str, str]]  # list instead of Sequence

    # Minimal validation for high throughput
    content_length: int = Field(ge=0, default=0)

    @field_validator('content_length', mode='before')
    @classmethod
    def calculate_content_length(cls, v, info):
        """Calculate content length if not provided."""
        if v == 0 and 'payload' in info.data:
            return len(json.dumps(info.data['payload'], default=str))
        return v

# ============================================================================
# 6. PROTOCOL DESIGN PATTERNS
# ============================================================================

class MessageEnvelope(BaseModel):
    """
    Message envelope for distributed agent communication.

    Demonstrates:
    - Polymorphic message design
    - Protocol versioning
    - Error propagation patterns
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        discriminator='message_type'  # For union discrimination
    )

    envelope_id: UUID = Field(default_factory=uuid4)
    message_type: str
    correlation_id: Optional[UUID] = None
    reply_to: Optional[str] = None
    ttl: int = Field(ge=1, le=300, default=30)  # Time to live in seconds

    # Routing information
    source_agent: AgentMetadata
    target_agent: Optional[str] = None  # None for broadcast

    @model_validator(mode='after')
    def validate_reply_semantics(self) -> Self:
        """Validate reply-to semantics for request-response patterns."""
        if self.message_type in ["request", "query"] and not self.reply_to:
            raise ValueError("Request messages must specify reply_to address")
        return self

# Polymorphic message payloads using discriminated unions
class TaskRequestMessage(BaseModel):
    """Task request message payload."""
    message_type: Literal["task_request"] = "task_request"
    request: TaskRequest
    execution_context: dict[str, Any] = Field(default_factory=dict)

class TaskResponseMessage(BaseModel):
    """Task response message payload."""
    message_type: Literal["task_response"] = "task_response"
    response: TaskResult
    execution_metadata: dict[str, Any] = Field(default_factory=dict)

class HeartbeatMessage(BaseModel):
    """Agent heartbeat message payload."""
    message_type: Literal["heartbeat"] = "heartbeat"
    agent_status: AgentMetadata
    metrics: dict[str, float] = Field(default_factory=dict)

class ErrorMessage(BaseModel):
    """Error message payload for distributed error propagation."""
    message_type: Literal["error"] = "error"
    error_code: str
    error_message: str
    error_details: dict[str, Any] = Field(default_factory=dict)
    original_message_id: Optional[UUID] = None

# Union type for polymorphic message handling
MessagePayload = Union[
    TaskRequestMessage,
    TaskResponseMessage,
    HeartbeatMessage,
    ErrorMessage
]

class DistributedMessage(BaseModel):
    """
    Complete distributed message with envelope and polymorphic payload.

    This is the top-level protocol message for agent communication.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    envelope: MessageEnvelope
    payload: SerializeAsAny[MessagePayload]  # Enable duck typing for subclasses

    def to_json_rpc(self) -> dict[str, Any]:
        """Convert to JSON-RPC compatible format."""
        return {
            "jsonrpc": "2.0",
            "id": str(self.envelope.envelope_id),
            "method": self.payload.message_type,
            "params": self.payload.model_dump(exclude={'message_type'})
        }

    @classmethod
    def from_json_rpc(cls, data: dict[str, Any], source_agent: AgentMetadata) -> Self:
        """Create from JSON-RPC format."""
        method = data.get("method")
        params = data.get("params", {})

        # Create envelope
        envelope = MessageEnvelope(
            envelope_id=UUID(data["id"]),
            message_type=method,
            source_agent=source_agent
        )

        # Create appropriate payload based on method
        payload_data = {"message_type": method, **params}

        if method == "task_request":
            payload = TaskRequestMessage(**payload_data)
        elif method == "task_response":
            payload = TaskResponseMessage(**payload_data)
        elif method == "heartbeat":
            payload = HeartbeatMessage(**payload_data)
        elif method == "error":
            payload = ErrorMessage(**payload_data)
        else:
            raise ValueError(f"Unknown message type: {method}")

        return cls(envelope=envelope, payload=payload)

# ============================================================================
# 7. USAGE EXAMPLES AND TESTING
# ============================================================================

def demonstrate_validation_patterns():
    """Demonstrate validation error handling patterns."""
    print("=== Validation Patterns Demo ===")

    try:
        # This will trigger multiple validation errors
        task = TaskRequest(
            task_type="invalid type!",  # Invalid format
            priority=15,  # Out of range
            budget="-100.50",  # Negative amount
            callback_url="invalid-url",  # Invalid URL
            deadline=datetime(2020, 1, 1)  # Past date
        )
    except ValidationError as e:
        error_data = ValidationErrorHandler.format_validation_error(e)
        print(f"Validation errors: {error_data['error_count']}")
        for error in error_data['errors']:
            print(f"  - {error['field_path']}: {error['message']}")

def demonstrate_serialization_patterns():
    """Demonstrate custom serialization patterns."""
    print("\n=== Serialization Patterns Demo ===")

    # Create task result with custom serialization
    result = TaskResult(
        task_id=uuid4(),
        status="completed",
        completion_time=datetime.now(timezone.utc),
        processing_cost=Decimal("123.45"),
        result_data={
            "output": "Task completed successfully",
            "_private_key": "secret123",  # Will be filtered out
            "metrics": {"duration": 5.2}
        }
    )

    # Serialize to JSON
    json_data = result.model_dump_json(indent=2)
    print("Serialized result:")
    print(json_data)

def demonstrate_message_protocol():
    """Demonstrate the distributed message protocol."""
    print("\n=== Message Protocol Demo ===")

    # Create agent metadata
    coordinator = AgentMetadata(
        agent_id="coordinator-1",
        role=AgentRole.COORDINATOR,
        capabilities=["coordination", "task_distribution"],
        load_factor=0.3
    )

    worker = AgentMetadata(
        agent_id="worker-1",
        role=AgentRole.WORKER,
        capabilities=["task_execution"],
        load_factor=0.7
    )

    # Create task request
    task_request = TaskRequest(
        task_type="data_processing",
        priority=5,
        budget=Decimal("500.00"),
        callback_url="agent://coordinator-1/callbacks",
        parameters={"dataset": "user_data.csv", "algorithm": "kmeans"}
    )

    # Create message payload
    request_payload = TaskRequestMessage(request=task_request)

    # Create envelope
    envelope = MessageEnvelope(
        message_type="task_request",
        source_agent=coordinator,
        target_agent="worker-1",
        reply_to="agent://coordinator-1/responses"
    )

    # Create complete message
    message = DistributedMessage(envelope=envelope, payload=request_payload)

    # Convert to JSON-RPC format
    json_rpc = message.to_json_rpc()
    print("JSON-RPC format:")
    print(json.dumps(json_rpc, indent=2, default=str))

    # Demonstrate round-trip
    reconstructed = DistributedMessage.from_json_rpc(json_rpc, coordinator)
    print(f"\nRound-trip successful: {reconstructed.envelope.envelope_id == message.envelope.envelope_id}")

def demonstrate_performance_patterns():
    """Demonstrate performance optimization patterns."""
    print("\n=== Performance Patterns Demo ===")

    # High-throughput message
    optimized_msg = OptimizedMessage(
        message_type="notification",
        payload={"event": "agent_status_update", "data": {"load": 0.8}},
        headers=[("content-type", "application/json"), ("priority", "low")]
    )

    print(f"Optimized message: {optimized_msg.model_dump_json()}")
    print(f"Content length calculated: {optimized_msg.content_length}")

if __name__ == "__main__":
    """
    Run demonstrations of all patterns.

    This showcases comprehensive Pydantic V2 usage for distributed systems
    including strict validation, error handling, serialization, and protocol design.
    """
    demonstrate_validation_patterns()
    demonstrate_serialization_patterns()
    demonstrate_message_protocol()
    demonstrate_performance_patterns()

    print("\n=== Summary ===")
    print("Demonstrated patterns:")
    print("✓ Strict typing and validation")
    print("✓ Custom field and model validators")
    print("✓ Advanced serialization with custom serializers")
    print("✓ Comprehensive error handling")
    print("✓ Performance-optimized configurations")
    print("✓ JSON-RPC style message protocol")
    print("✓ Polymorphic payloads with discriminated unions")
    print("✓ Agent communication patterns")