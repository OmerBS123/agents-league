"""
Advanced Pydantic V2 Validation Patterns for Distributed Systems
================================================================

This module demonstrates sophisticated validation patterns including:
- Context-aware validation
- Conditional validation
- Cross-model validation
- Async validation patterns
- Custom validation errors
- Validation with external dependencies

Based on: docs.pydantic.dev/latest/concepts/validators/
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4
import asyncio
from contextlib import asynccontextmanager

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
    ValidationInfo,
    AfterValidator,
    BeforeValidator,
    WrapValidator
)
from pydantic_core import PydanticCustomError
from typing_extensions import Self


# ============================================================================
# CONTEXT-AWARE VALIDATION
# ============================================================================

class SecurityLevel(str, Enum):
    """Security levels for validation context."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

class ValidationContext(BaseModel):
    """Context information for validation decisions."""
    security_level: SecurityLevel
    requesting_agent_id: str
    operation_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

def validate_with_security_context(value: str, info: ValidationInfo) -> str:
    """Validate field based on security context."""
    if not info.context:
        raise ValueError("Security context required")

    context = ValidationContext(**info.context)

    # Apply different validation rules based on security level
    if context.security_level == SecurityLevel.SECRET:
        if len(value) < 32:
            raise ValueError("Secret level fields require at least 32 characters")
        if not any(c.isupper() for c in value):
            raise ValueError("Secret level fields must contain uppercase characters")

    elif context.security_level == SecurityLevel.CONFIDENTIAL:
        if len(value) < 16:
            raise ValueError("Confidential level fields require at least 16 characters")

    return value

class SecureDocument(BaseModel):
    """Document with context-aware field validation."""

    document_id: UUID = Field(default_factory=uuid4)
    title: str

    # Context-aware validation
    content: Annotated[
        str,
        AfterValidator(validate_with_security_context)
    ]

    classification: SecurityLevel

    @model_validator(mode='after')
    def validate_classification_consistency(self, info: ValidationInfo) -> Self:
        """Ensure document classification matches validation context."""
        if info.context:
            context = ValidationContext(**info.context)
            if context.security_level != self.classification:
                raise ValueError(
                    f"Document classification {self.classification} "
                    f"doesn't match context level {context.security_level}"
                )
        return self


# ============================================================================
# CONDITIONAL VALIDATION
# ============================================================================

def validate_conditional_field(value: Any, handler, info: ValidationInfo) -> Any:
    """
    Conditional validation based on other field values.

    This wrap validator demonstrates conditional validation logic.
    """
    # First, apply standard validation
    validated_value = handler(value)

    # Then apply conditional logic
    if 'mode' in info.data:
        mode = info.data['mode']

        if mode == 'strict' and isinstance(validated_value, str):
            if not validated_value.isalnum():
                raise ValueError("Strict mode requires alphanumeric values only")

        elif mode == 'relaxed':
            # In relaxed mode, convert to string if possible
            if not isinstance(validated_value, str):
                validated_value = str(validated_value)

    return validated_value

class ConfigurableModel(BaseModel):
    """Model with conditional validation based on mode setting."""

    mode: Literal["strict", "relaxed", "normal"] = "normal"

    # This field's validation depends on the mode
    data_field: Annotated[
        Any,
        WrapValidator(validate_conditional_field)
    ]

    optional_strict_field: Optional[str] = None

    @field_validator('optional_strict_field', mode='after')
    @classmethod
    def validate_optional_when_strict(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Make optional field required in strict mode."""
        mode = info.data.get('mode', 'normal')

        if mode == 'strict' and v is None:
            raise ValueError("This field is required in strict mode")

        return v


# ============================================================================
# CROSS-MODEL VALIDATION
# ============================================================================

class ModelRegistry:
    """Registry for cross-model validation."""

    def __init__(self):
        self._models: Dict[str, BaseModel] = {}

    def register(self, key: str, model: BaseModel):
        """Register a model for cross-validation."""
        self._models[key] = model

    def get(self, key: str) -> Optional[BaseModel]:
        """Get a registered model."""
        return self._models.get(key)

    def validate_reference(self, ref_id: str, model_type: str) -> bool:
        """Validate that a reference exists and is of correct type."""
        model = self._models.get(ref_id)
        return model is not None and model.__class__.__name__ == model_type

# Global registry for cross-model validation
model_registry = ModelRegistry()

def validate_model_reference(value: str, info: ValidationInfo) -> str:
    """Validate reference to another model."""
    if info.context and 'registry' in info.context:
        registry: ModelRegistry = info.context['registry']
        expected_type = info.context.get('expected_type')

        if not registry.validate_reference(value, expected_type):
            raise ValueError(f"Invalid reference to {expected_type}: {value}")

    return value

class User(BaseModel):
    """User model for cross-reference validation."""
    user_id: str
    username: str
    email: str

    def __post_init__(self):
        """Register this model after creation."""
        model_registry.register(self.user_id, self)

class Project(BaseModel):
    """Project model with user reference validation."""
    project_id: str
    name: str

    # Cross-model reference validation
    owner_id: Annotated[
        str,
        AfterValidator(validate_model_reference)
    ]

    @model_validator(mode='before')
    @classmethod
    def setup_validation_context(cls, data: Any, info: ValidationInfo) -> Any:
        """Setup context for cross-model validation."""
        if info.context is None:
            info.context = {}

        info.context.update({
            'registry': model_registry,
            'expected_type': 'User'
        })

        return data


# ============================================================================
# ASYNC VALIDATION PATTERNS
# ============================================================================

class AsyncValidationCache:
    """Cache for async validation results."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def get_or_validate(self, key: str, validator_func, *args, **kwargs):
        """Get cached result or perform async validation."""
        if key in self._cache:
            return self._cache[key]

        # Ensure we don't validate the same key concurrently
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            if key in self._cache:  # Double-check after acquiring lock
                return self._cache[key]

            result = await validator_func(*args, **kwargs)
            self._cache[key] = result
            return result

# Global async validation cache
async_cache = AsyncValidationCache()

async def validate_external_reference(reference_id: str) -> bool:
    """Simulate async validation against external service."""
    # Simulate network delay
    await asyncio.sleep(0.1)

    # Simulate validation logic (e.g., checking external API)
    return reference_id.startswith("valid_") and len(reference_id) > 10

def async_reference_validator(value: str) -> str:
    """
    Synchronous wrapper for async validation.

    Note: This pattern should be used carefully as it blocks the event loop.
    In a real distributed system, consider using background validation.
    """
    async def _validate():
        return await async_cache.get_or_validate(
            f"ref_{value}",
            validate_external_reference,
            value
        )

    # Create event loop if none exists
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, we can't use run_until_complete
            # This would typically be handled differently in real applications
            raise RuntimeError("Cannot perform async validation in running event loop")
        else:
            is_valid = loop.run_until_complete(_validate())
    except RuntimeError:
        # Create new event loop for validation
        is_valid = asyncio.run(_validate())

    if not is_valid:
        raise ValueError(f"External reference validation failed: {value}")

    return value

class ExternalReferencedModel(BaseModel):
    """Model with async validation for external references."""

    model_config = ConfigDict(
        # Disable validation of defaults for performance
        validate_default=False,
        # Use assignment validation for async fields
        validate_assignment=True
    )

    name: str
    external_ref: Annotated[
        str,
        AfterValidator(async_reference_validator)
    ]


# ============================================================================
# CUSTOM VALIDATION ERRORS
# ============================================================================

class BusinessRuleError(Exception):
    """Custom business rule validation error."""

    def __init__(self, rule_name: str, message: str, context: Dict[str, Any] = None):
        self.rule_name = rule_name
        self.message = message
        self.context = context or {}
        super().__init__(message)

def validate_business_rules(value: Any, info: ValidationInfo) -> Any:
    """Validate against business rules with custom errors."""

    # Example business rule: certain operations are restricted during maintenance
    if info.context and info.context.get('maintenance_mode'):
        if info.context.get('operation') in ['create', 'update', 'delete']:
            raise PydanticCustomError(
                'maintenance_mode_restriction',
                'Operation {operation} is not allowed during maintenance mode',
                {'operation': info.context.get('operation')}
            )

    # Example business rule: validate against regulatory compliance
    if isinstance(value, str) and 'financial' in info.field_name.lower():
        if any(keyword in value.lower() for keyword in ['insider', 'manipulation']):
            raise PydanticCustomError(
                'compliance_violation',
                'Content violates financial compliance rules: {content}',
                {'content': value[:50] + '...' if len(value) > 50 else value}
            )

    return value

class ComplianceModel(BaseModel):
    """Model with business rule validation."""

    transaction_id: str
    financial_description: Annotated[
        str,
        AfterValidator(validate_business_rules)
    ]
    amount: float

    @model_validator(mode='after')
    def validate_compliance_rules(self, info: ValidationInfo) -> Self:
        """Model-level compliance validation."""

        # Large transaction rule
        if self.amount > 10000:
            if 'high_value_approved' not in (info.context or {}):
                raise PydanticCustomError(
                    'large_transaction_approval',
                    'Transactions over $10,000 require pre-approval. Amount: ${amount}',
                    {'amount': self.amount}
                )

        return self


# ============================================================================
# VALIDATION WITH EXTERNAL DEPENDENCIES
# ============================================================================

class ValidationDependencies:
    """Container for external validation dependencies."""

    def __init__(self):
        self.user_service = None
        self.config_service = None
        self.audit_service = None

    def set_user_service(self, service):
        self.user_service = service

    def set_config_service(self, service):
        self.config_service = service

    def set_audit_service(self, service):
        self.audit_service = service

# Global dependencies
validation_deps = ValidationDependencies()

def validate_with_user_service(value: str, info: ValidationInfo) -> str:
    """Validate using external user service."""
    if not validation_deps.user_service:
        raise ValueError("User service not available for validation")

    # Simulate user service call
    if not validation_deps.user_service.user_exists(value):
        raise ValueError(f"User not found: {value}")

    if not validation_deps.user_service.user_active(value):
        raise ValueError(f"User is not active: {value}")

    return value

class UserService:
    """Mock user service for validation."""

    def __init__(self):
        self.users = {
            "user1": {"active": True},
            "user2": {"active": False},
            "admin": {"active": True}
        }

    def user_exists(self, user_id: str) -> bool:
        return user_id in self.users

    def user_active(self, user_id: str) -> bool:
        return self.users.get(user_id, {}).get("active", False)

class DependentValidationModel(BaseModel):
    """Model that depends on external services for validation."""

    task_id: str
    assigned_user: Annotated[
        str,
        AfterValidator(validate_with_user_service)
    ]
    priority: int = Field(ge=1, le=5)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_context_aware_validation():
    """Demonstrate context-aware validation."""
    print("=== Context-Aware Validation ===")

    # Setup validation context
    secret_context = {
        "security_level": SecurityLevel.SECRET,
        "requesting_agent_id": "agent-123",
        "operation_type": "create_document"
    }

    try:
        # This should fail - content too short for SECRET level
        doc = SecureDocument.model_validate(
            {
                "title": "Secret Document",
                "content": "short",
                "classification": SecurityLevel.SECRET
            },
            context=secret_context
        )
    except ValidationError as e:
        print(f"Expected validation error: {e.errors()[0]['msg']}")

    # This should succeed
    doc = SecureDocument.model_validate(
        {
            "title": "Secret Document",
            "content": "THIS_IS_A_VERY_LONG_SECRET_CONTENT_WITH_UPPERCASE",
            "classification": SecurityLevel.SECRET
        },
        context=secret_context
    )
    print(f"Successfully created document: {doc.title}")

def demonstrate_conditional_validation():
    """Demonstrate conditional validation patterns."""
    print("\n=== Conditional Validation ===")

    # Normal mode - relaxed validation
    model1 = ConfigurableModel(
        mode="relaxed",
        data_field=123,  # Will be converted to string
        optional_strict_field=None  # Optional in relaxed mode
    )
    print(f"Relaxed mode result: {model1.data_field}")

    # Strict mode validation
    try:
        model2 = ConfigurableModel(
            mode="strict",
            data_field="invalid-chars!",  # Should fail in strict mode
        )
    except ValidationError as e:
        print(f"Strict mode validation error: {e.errors()[0]['msg']}")

def demonstrate_cross_model_validation():
    """Demonstrate cross-model validation."""
    print("\n=== Cross-Model Validation ===")

    # Create a user first
    user = User(user_id="user123", username="john_doe", email="john@example.com")

    # Create project referencing the user
    project = Project(
        project_id="proj1",
        name="Test Project",
        owner_id="user123"
    )
    print(f"Successfully created project with owner: {project.owner_id}")

    # Try to create project with invalid user reference
    try:
        invalid_project = Project(
            project_id="proj2",
            name="Invalid Project",
            owner_id="nonexistent_user"
        )
    except ValidationError as e:
        print(f"Cross-model validation error: {e.errors()[0]['msg']}")

def demonstrate_external_dependencies():
    """Demonstrate validation with external dependencies."""
    print("\n=== External Dependencies Validation ===")

    # Setup mock services
    validation_deps.set_user_service(UserService())

    # Valid user
    model1 = DependentValidationModel(
        task_id="task1",
        assigned_user="admin",
        priority=3
    )
    print(f"Task assigned to active user: {model1.assigned_user}")

    # Invalid user (inactive)
    try:
        model2 = DependentValidationModel(
            task_id="task2",
            assigned_user="user2",  # This user is inactive
            priority=2
        )
    except ValidationError as e:
        print(f"External service validation error: {e.errors()[0]['msg']}")

def demonstrate_business_rules():
    """Demonstrate business rule validation."""
    print("\n=== Business Rules Validation ===")

    # Normal transaction
    model1 = ComplianceModel(
        transaction_id="txn1",
        financial_description="Regular payment for services",
        amount=500.0
    )
    print(f"Normal transaction approved: {model1.transaction_id}")

    # Large transaction without approval
    try:
        model2 = ComplianceModel(
            transaction_id="txn2",
            financial_description="Large investment transaction",
            amount=50000.0
        )
    except ValidationError as e:
        print(f"Business rule violation: {e.errors()[0]['msg']}")

    # Large transaction with approval
    model3 = ComplianceModel.model_validate(
        {
            "transaction_id": "txn3",
            "financial_description": "Approved large investment",
            "amount": 50000.0
        },
        context={"high_value_approved": True}
    )
    print(f"Large transaction with approval: {model3.transaction_id}")

if __name__ == "__main__":
    """Run demonstrations of advanced validation patterns."""

    demonstrate_context_aware_validation()
    demonstrate_conditional_validation()
    demonstrate_cross_model_validation()
    demonstrate_external_dependencies()
    demonstrate_business_rules()

    print("\n=== Advanced Validation Summary ===")
    print("✓ Context-aware validation with security levels")
    print("✓ Conditional validation based on model state")
    print("✓ Cross-model reference validation")
    print("✓ External service dependency validation")
    print("✓ Business rule compliance validation")
    print("✓ Custom error messages and types")