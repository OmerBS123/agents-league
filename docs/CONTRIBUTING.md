# Contributing Guide

## Distributed AI Agent League System

Thank you for your interest in contributing to the Distributed AI Agent League System! This document provides guidelines and best practices for contributing to the project.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Standards](#code-standards)
4. [Testing Guidelines](#testing-guidelines)
5. [Commit Guidelines](#commit-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Adding New Features](#adding-new-features)

---

## 1. Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (>= 0.4.0)
- Git
- (Optional) Ollama for LLM strategy development

### Quick Setup

```bash
# Clone the repository
git clone <repo-url>
cd agents-league

# Install all dependencies including dev tools
uv sync --all-groups

# Verify installation
uv run python -c "import fastapi, pydantic, httpx; print('OK')"

# Run tests
uv run pytest test_integration.py -v
```

---

## 2. Development Setup

### Project Structure

```
agents-league/
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── league_manager.py   # Central coordinator
│   ├── player.py           # Player with strategies
│   └── referee.py          # Match orchestrator
├── shared/                 # Shared utilities
│   ├── __init__.py
│   ├── schemas.py          # Pydantic models
│   ├── exceptions.py       # Custom exceptions
│   ├── logger.py           # Logging utilities
│   └── ollama_strategy.py  # LLM integration
├── consts/                 # Configuration
│   └── __init__.py
├── docs/                   # Documentation
├── tests/                  # Test files
├── pyproject.toml          # Project config
└── README.md
```

### IDE Configuration

**VS Code (Recommended)**

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    }
}
```

---

## 3. Code Standards

### 3.1 Style Guide

We follow PEP 8 with these specifics:

| Rule | Standard |
|------|----------|
| Line length | 100 characters max |
| Indentation | 4 spaces (no tabs) |
| Quotes | Double quotes for strings |
| Naming | snake_case for functions/variables |
| Naming | PascalCase for classes |
| Naming | UPPER_CASE for constants |

### 3.2 Type Hints

All functions must have type hints:

```python
# Good
def calculate_score(player_id: str, rounds: List[int]) -> float:
    """Calculate player score from round results."""
    return sum(rounds) / len(rounds) if rounds else 0.0

# Bad - missing type hints
def calculate_score(player_id, rounds):
    return sum(rounds) / len(rounds) if rounds else 0.0
```

### 3.3 Docstrings

Use NumPy-style docstrings:

```python
def process_move(
    match_id: str,
    player_id: str,
    choice: ParityChoice
) -> MoveResult:
    """
    Process a player's move in an active match.

    Parameters
    ----------
    match_id : str
        Unique identifier for the match.
    player_id : str
        Identifier of the player making the move.
    choice : ParityChoice
        The player's parity choice (EVEN or ODD).

    Returns
    -------
    MoveResult
        Result of processing the move.

    Raises
    ------
    MatchError
        If match_id is not found or match is not active.
    ValidationError
        If choice is invalid.

    Examples
    --------
    >>> result = process_move("M-001", "player:P01", ParityChoice.EVEN)
    >>> result.success
    True
    """
```

### 3.4 Code Quality Tools

Run these before committing:

```bash
# Format code
uv run ruff format agents/ shared/

# Lint code
uv run ruff check agents/ shared/ --fix

# Type checking
uv run mypy agents/ shared/

# Security scan
uv run bandit -r agents/ shared/

# Check for unused code
uv run vulture agents/ shared/
```

---

## 4. Testing Guidelines

### 4.1 Test Structure

```python
# tests/test_example.py

import pytest
from shared.schemas import MCPEnvelope, MessageType


class TestMCPEnvelope:
    """Tests for MCPEnvelope schema."""

    def test_valid_envelope_creation(self):
        """Test creating a valid envelope."""
        envelope = MCPEnvelope(
            message_type=MessageType.REGISTER,
            sender="player:P01",
            data={"display_name": "Test", ...}
        )
        assert envelope.sender == "player:P01"

    def test_invalid_sender_format(self):
        """Test that invalid sender format raises error."""
        with pytest.raises(ValueError, match="must be in format"):
            MCPEnvelope(
                message_type=MessageType.REGISTER,
                sender="invalid",
                data={}
            )

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async functionality."""
        result = await some_async_function()
        assert result is not None
```

### 4.2 Test Categories

| Category | Location | Command |
|----------|----------|---------|
| Unit tests | `tests/test_*.py` | `uv run pytest tests/` |
| Integration | `test_integration.py` | `uv run pytest test_integration.py` |
| All tests | - | `uv run pytest` |

### 4.3 Coverage Requirements

- Minimum coverage: 70%
- New code should have >80% coverage

```bash
# Run with coverage
uv run pytest --cov=agents --cov=shared --cov-report=html

# View report
open htmlcov/index.html
```

---

## 5. Commit Guidelines

### 5.1 Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code restructuring |
| `test` | Adding tests |
| `chore` | Build, CI, deps |

**Examples:**

```bash
# Feature
git commit -m "feat(player): add Monte Carlo strategy"

# Bug fix
git commit -m "fix(referee): handle timeout in move collection"

# Documentation
git commit -m "docs(readme): add troubleshooting section"

# Refactor
git commit -m "refactor(schemas): split into multiple modules"
```

### 5.2 Branch Naming

```
<type>/<short-description>

Examples:
feature/add-websocket-support
fix/timeout-handling
docs/update-architecture
refactor/split-schemas
```

---

## 6. Pull Request Process

### 6.1 Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] No linting errors
- [ ] Type hints added
- [ ] Commit messages follow convention

### 6.2 PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed.

## Checklist
- [ ] Tests pass
- [ ] Docs updated
- [ ] No linting errors
```

### 6.3 Review Process

1. Create PR against `main` branch
2. Automated checks run (if configured)
3. Request review from maintainers
4. Address feedback
5. Squash and merge when approved

---

## 7. Adding New Features

### 7.1 Adding a New Strategy

1. Create strategy class in `agents/player.py`:

```python
class NewStrategy(Strategy):
    """Description of strategy."""

    @property
    def strategy_name(self) -> str:
        return "new_strategy"

    async def choose_move(
        self,
        match_id: str,
        opponent_id: str,
        round_id: int,
        opponent_history: List[ParityChoice]
    ) -> tuple[ParityChoice, float, Optional[str]]:
        # Implementation
        choice = ...
        confidence = 0.7
        reasoning = "Explanation"
        return choice, confidence, reasoning
```

2. Register in factory function:

```python
def create_strategy(strategy_name: str, agent_id: str) -> Strategy:
    if strategy_name == "new_strategy":
        return NewStrategy(agent_id)
    # ... existing strategies
```

3. Update `AVAILABLE_STRATEGIES` in `consts/__init__.py`

4. Add tests in `tests/test_strategies.py`

5. Update documentation

### 7.2 Adding a New Message Type

1. Add to `MessageType` enum in `shared/schemas.py`:

```python
class MessageType(str, Enum):
    # ... existing types
    NEW_TYPE = "NEW_MESSAGE_TYPE"
```

2. Create payload model:

```python
class NewTypeData(BasePayload):
    """Payload for NEW_TYPE messages."""
    field1: str
    field2: int = Field(..., ge=0)
```

3. Add validation in `MCPEnvelope.validate_data_content()`:

```python
elif self.message_type == MessageType.NEW_TYPE:
    NewTypeData(**self.data)
```

4. Implement handler in relevant agent

5. Add tests

### 7.3 Adding a New Agent

1. Create new module in `agents/`:

```python
# agents/new_agent.py
"""
New Agent - Description.
"""

from fastapi import FastAPI
# ... implementation
```

2. Add configuration in `consts/__init__.py`

3. Update `start_league.sh` to include new agent

4. Add to documentation

---

## 8. Getting Help

- **Issues:** Open a GitHub issue for bugs or features
- **Discussions:** Use GitHub Discussions for questions
- **Documentation:** Check `docs/` directory

---

## 9. Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project guidelines

---

Thank you for contributing!
