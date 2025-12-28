# Distributed AI Agent League System - Implementation PRP

**Feature:** Complete distributed AI agent league system with Player, League Manager, and Referee agents communicating over HTTP/JSON-RPC

**Confidence Score:** 9/10 - High confidence for one-pass implementation success

---

## Executive Summary

Build a complete local distributed system involving three distinct types of agents (Player, League Manager, Referee) that communicate over HTTP using JSON-RPC protocols. The system implements an "Even/Odd" game league with round-robin scheduling, multiple player strategies (random, history-based, LLM-powered), and comprehensive orchestration.

**Key Success Factors:**
- Comprehensive example files provide exact patterns to follow
- Expert agent research provides production-ready best practices
- Strict quality gates ensure code reliability
- Clear implementation order minimizes dependencies
- Executable validation commands guarantee success

---

## Critical Context & Research Findings

### 📁 **Example File Templates (MUST REFERENCE)**

The AI agent has access to these proven implementation patterns:

**FastAPI Server Template:**
- **File:** `/Users/omerbensalmon/Documents/personal_life/masters_degree/years/2025-2026/semseters/A/LLM_in_multiple_agent_env/home_assigments/7/examples/fastapi_server.py`
- **Purpose:** Template for all three agents (player.py, league_manager.py, referee.py)
- **Key Patterns:** JSON Lines logging, middleware stack, exception handlers, lifespan management
- **Lines 25-42:** Singleton logger pattern with JSON formatting
- **Lines 112-122:** Lifespan management for resource initialization
- **Lines 174-195:** Global exception handlers for validation and system errors

**Pydantic Schema Template:**
- **File:** `/Users/omerbensalmon/Documents/personal_life/masters_degree/years/2025-2026/semseters/A/LLM_in_multiple_agent_env/home_assigments/7/examples/pydantic_schema.py`
- **Purpose:** Template for shared/schemas.py with strict validation
- **Key Patterns:** Frozen models, strict validation, cross-field validation
- **Lines 47-49:** `extra="forbid", frozen=True` for strict protocol compliance
- **Lines 140-159:** Model validator for cross-field validation (message_type vs data)
- **Lines 178-185:** Reply factory method for maintaining conversation threads

**HTTPX Client Template:**
- **File:** `/Users/omerbensalmon/Documents/personal_life/masters_degree/years/2025-2026/semseters/A/LLM_in_multiple_agent_env/home_assigments/7/examples/httpx_client.py`
- **Key Patterns:** Circuit breaker, retry logic, exponential backoff
- **Lines 76-89:** Circuit breaker state management and timeout handling
- **Lines 129-145:** Exponential backoff with jitter for retry logic
- **Lines 164-176:** Registration flow with error handling

**Ollama Integration Template:**
- **File:** `/Users/omerbensalmon/Documents/personal_life/masters_degree/years/2025-2026/semseters/A/LLM_in_multiple_agent_env/home_assigments/7/examples/ollama_integration.py`
- **Key Patterns:** Timeout handling, response parsing, fallback strategies
- **Lines 154-164:** Comprehensive error handling with fallback to random strategy
- **Lines 73-90:** JSON extraction from LLM responses with regex fallback
- **Lines 166-174:** Deterministic fallback strategy for game continuity

### 🏗️ **Expert Agent Best Practices**

**FastAPI Production Patterns:**
- Dependency injection with `@lru_cache` for settings
- Lifespan management for HTTP client pools
- Structured logging with request IDs and timing
- Global exception handlers for custom agent errors
- Background tasks for non-blocking operations
- CORS middleware for agent-to-agent communication

**Pydantic V2 Strict Validation:**
- `ConfigDict(extra="forbid", frozen=True)` for protocol compliance
- Field validators with `@classmethod` for complex validation
- Model validators for cross-field business rules
- Custom serializers for datetime and Decimal fields
- Performance optimization with `model_construct()` for trusted data

**HTTPX Resilient Communication:**
- AsyncClient with proper lifecycle management
- Timeout hierarchy: connect=1.5s, read=2.0s, total=3.0s
- Circuit breaker pattern with failure threshold and reset timeout
- Exponential backoff with jitter: `delay = base * (2^attempt) + jitter`
- Connection pooling with max_connections=20, max_keepalive=10

---

## System Architecture

### **Component Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  League Manager │    │     Referee     │    │     Player      │
│   (Port 8000)   │    │   (Port 8001)   │    │  (Ports 81xx)   │
│                 │    │                 │    │                 │
│ - Registration  │◄──►│ - Match Control │◄──►│ - Strategies    │
│ - Scheduling    │    │ - Game Logic    │    │ - Registration  │
│ - Standings     │    │ - Parallel Moves│    │ - Move Choice   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Communication Flow**

1. **Startup:** Players register with League Manager (retry with exponential backoff)
2. **Scheduling:** Manager generates round-robin schedule
3. **Match Orchestration:** Manager instructs Referee to run specific matches
4. **Game Execution:** Referee coordinates parallel move collection from players
5. **Result Processing:** Referee reports results back to Manager for standings update

### **Message Protocol (JSON-RPC Style)**

All inter-agent communication uses the MCPEnvelope pattern from examples/pydantic_schema.py:

```python
{
  "protocol": "league.v2",
  "message_type": "GAME_INVITATION",
  "sender": "referee:REF-MAIN",
  "conversation_id": "uuid-string",
  "timestamp": "2025-01-01T12:00:00Z",
  "data": {
    "match_id": "M-101",
    "opponent_id": "player:P02",
    "timeout_ms": 2000
  }
}
```

---

## Implementation Blueprint

### **Phase 1: Foundation Components**

#### 1.1 Project Setup & Dependencies
```bash
# Initialize with uv (following uv-expert recommendations)
uv init --package
uv add fastapi pydantic httpx uvicorn
uv add --group quality ruff mypy bandit radon vulture deptry types-requests
uv add --group test pytest pytest-asyncio pytest-cov httpx[test]
uv sync --all-groups
```

#### 1.2 Constants & Configuration (consts/__init__.py)
```python
# Single source of truth for all configuration
LEAGUE_MANAGER_PORT = 8000
REFEREE_PORT = 8001
PLAYER_PORTS = [8101, 8102, 8103, 8104]
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 2.0  # Critical for game timing
MAX_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 5
```

#### 1.3 Singleton Logger (shared/logger.py)
**Pattern Reference:** examples/fastapi_server.py:25-42
```python
# JSON Lines format matching examples
def get_logger(name: str) -> logging.Logger:
    """Singleton logger factory with JSON formatting"""
    # Implementation follows JsonFormatter pattern from examples
```

#### 1.4 Custom Exceptions (shared/exceptions.py)
```python
class LeagueError(Exception):
    """Base exception for league operations"""

class RegistrationError(LeagueError):
    """Registration failures"""

class MatchError(LeagueError):
    """Match execution failures"""

class StrategyError(LeagueError):
    """Player strategy failures"""
```

#### 1.5 Strict Schemas (shared/schemas.py)
**Pattern Reference:** examples/pydantic_schema.py (complete file)
- Use MCPEnvelope pattern with cross-field validation
- Implement all message types: REGISTER, INVITATION, MOVE_CALL, GAME_OVER
- Follow frozen=True and extra="forbid" patterns
- Add reply factory methods for conversation threading

### **Phase 2: Agent Implementation**

#### 2.1 Player Agent (agents/player.py)

**Class Structure:**
```python
class PlayerServer:
    """Main FastAPI application following examples/fastapi_server.py patterns"""
    # Lines 123-130: FastAPI setup with lifespan management
    # Lines 174-195: Global exception handlers

class Strategy(ABC):
    """Abstract strategy base class"""

class RandomStrategy(Strategy):
    """Random even/odd selection"""

class HistoryStrategy(Strategy):
    """Counter-strategy based on opponent history"""

class LLMStrategy(Strategy):
    """Ollama-powered strategy with fallback"""
    # Pattern: examples/ollama_integration.py complete implementation
```

**Key Implementation Points:**
- Startup registration with Manager using circuit breaker pattern
- Strategy selection via command-line arguments
- Parallel handling of invitations and move requests
- Comprehensive error handling with custom exceptions

#### 2.2 League Manager (agents/league_manager.py)

**Class Structure:**
```python
class LeagueManager:
    """Central coordinator following FastAPI patterns"""

    def __init__(self):
        self.players: List[PlayerProfile] = []
        self.schedule: List[Matchup] = []
        self.standings: Dict[str, int] = {}

    async def generate_round_robin_schedule(self):
        """Pure function for schedule generation"""

    async def orchestrate_matches(self):
        """Background task for match coordination"""
```

**Key Implementation Points:**
- Round-robin algorithm for fair scheduling
- HTTP client integration for Referee communication
- Standings tracking and updates
- Background task coordination

#### 2.3 Referee Agent (agents/referee.py)

**Class Structure:**
```python
class MatchOrchestrator:
    """Ephemeral match state machine"""

    async def collect_moves_parallel(self, player_a: str, player_b: str):
        """Parallel move collection with timeout enforcement"""
        # Pattern: asyncio.gather with timeout control

    async def execute_match(self, match_info: Dict):
        """Complete match execution flow"""
```

**Key Implementation Points:**
- Parallel move collection using asyncio.gather
- Strict timeout enforcement (2 second default)
- Random number generation for game resolution
- Result reporting to Manager

### **Phase 3: Integration & Orchestration**

#### 3.1 Startup Script (start_league.sh)
```bash
#!/bin/bash
# Background process management using uv run

uv run uvicorn agents.league_manager:app --host 0.0.0.0 --port 8000 &
uv run uvicorn agents.referee:app --host 0.0.0.0 --port 8001 &
uv run uvicorn agents.player:app --host 0.0.0.0 --port 8101 -- --strategy random &
uv run uvicorn agents.player:app --host 0.0.0.0 --port 8102 -- --strategy history &
uv run uvicorn agents.player:app --host 0.0.0.0 --port 8103 -- --strategy llm &
uv run uvicorn agents.player:app --host 0.0.0.0 --port 8104 -- --strategy random &

wait # Wait for all processes
```

#### 3.2 Vulture Whitelist (whitelist.py)
```python
# Configuration for dead code detection
# Mark intentionally unused imports and variables
```

---

## Quality Gates & Validation

### **Mandatory Code Quality Stack**

**Installation Command:**
```bash
uv add --group quality ruff mypy bandit radon vulture deptry
```

**pyproject.toml Configuration:**
```toml
[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "C", "I", "N", "D", "UP", "S"]

[tool.mypy]
strict = true
python_version = "3.13"

[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]  # Skip assert_used if needed

[tool.vulture]
min_confidence = 80
paths = ["agents", "shared"]
```

### **Executable Validation Commands**

**1. Style & Linting (MUST PASS):**
```bash
uv run ruff check . --fix
uv run ruff format .
```

**2. Type Checking (MUST PASS):**
```bash
uv run mypy . --strict
```

**3. Security Scan (MUST PASS):**
```bash
uv run bandit -r . --exclude /tests -ll
```

**4. Complexity Analysis (MUST PASS - Grade A-B only):**
```bash
uv run radon cc . --min C  # Should return empty (no Grade C+)
```

**5. Dead Code Detection (MUST PASS):**
```bash
uv run vulture . whitelist.py
```

**6. Dependency Validation (MUST PASS):**
```bash
uv run deptry .
```

### **Agent Code Review Requirements**

**Sequential Review Process (MANDATORY):**

1. **Logic & Security Review:**
```python
# Use feature-dev:code-reviewer agent
Task(
    subagent_type="feature-dev:code-reviewer",
    prompt="Review distributed agent system for race conditions, timeout handling, and security vulnerabilities in async code"
)
```

2. **Style & Standards Review:**
```python
# Use pr-review-toolkit:code-reviewer agent
Task(
    subagent_type="pr-review-toolkit:code-reviewer",
    prompt="Review all files for project structure adherence, docstring completeness, and consistent patterns"
)
```

---

## Implementation Task List (Execute in Order)

### **Week 1: Foundation**
1. ✅ **Setup Project Structure**
   - Initialize uv project with pyproject.toml
   - Install all dependencies with quality tools
   - Create directory structure (agents/, shared/, consts/, tests/)

2. ✅ **Implement Shared Components**
   - consts/__init__.py: All configuration constants
   - shared/exceptions.py: Custom exception hierarchy
   - shared/logger.py: Singleton logger with JSON formatting
   - shared/schemas.py: Strict Pydantic models with cross-validation

3. ✅ **Quality Gate Validation**
   - Run all quality tools and fix issues
   - Ensure 100% pass rate on foundation code

### **Week 2: Core Agents**
4. ✅ **Implement Player Agent**
   - agents/player.py: FastAPI server with strategy pattern
   - Implement RandomStrategy, HistoryStrategy classes
   - Registration logic with circuit breaker
   - Move request handling with timeout enforcement

5. ✅ **Implement Referee Agent**
   - agents/referee.py: Match orchestration server
   - Parallel move collection using asyncio.gather
   - Game logic with random number generation
   - Result reporting with error handling

6. ✅ **Quality Gate Validation**
   - Run full quality suite on agent implementations
   - Address any complexity or security issues

### **Week 3: Advanced Features**
7. ✅ **Implement League Manager**
   - agents/league_manager.py: Central coordinator
   - Player registration endpoint
   - Round-robin scheduling algorithm
   - Match orchestration and standings tracking

8. ✅ **Implement LLM Strategy**
   - Ollama integration using httpx patterns from examples
   - JSON response parsing with regex fallback
   - Timeout handling with deterministic fallback
   - Error handling for connection failures

9. ✅ **Integration Testing**
   - Create start_league.sh orchestration script
   - End-to-end testing with all agents running
   - Performance testing with concurrent matches

### **Week 4: Production Readiness**
10. ✅ **Final Quality Review**
    - Execute both mandatory agent reviews
    - Address all findings and recommendations
    - Final quality tool validation

11. ✅ **Documentation & Deployment**
    - Complete whitelist.py for vulture
    - Performance optimization and profiling
    - Production deployment preparation

---

## Error Handling Strategy

### **Network Communication Errors**
- Use circuit breaker pattern from examples/httpx_client.py
- Exponential backoff with jitter for retries
- Graceful degradation when agents unavailable

### **Game Logic Errors**
- Timeout enforcement at multiple levels (connection, request, total)
- Fallback strategies for LLM failures
- Match continuation even with partial player failures

### **Quality Assurance Errors**
- Zero tolerance policy - all quality gates must pass
- Automated fixing where possible (ruff --fix)
- Manual review required for security and complexity issues

---

## Performance Considerations

### **Concurrency Patterns**
- Use asyncio.gather for parallel operations
- Semaphore control for limiting concurrent requests
- Connection pooling for HTTP clients

### **Timeout Management**
- Hierarchical timeouts: connect < request < total < match
- Default 2-second game timeouts for real-time feel
- Circuit breaker prevents cascade failures

### **Resource Management**
- Proper AsyncClient lifecycle with lifespan handlers
- Background task cleanup for match state
- Memory-efficient data structures for large tournaments

---

## Success Metrics

**Quality Metrics:**
- ✅ 100% pass rate on all 6 quality tools
- ✅ Zero medium/high security vulnerabilities
- ✅ All functions Grade A-B complexity only
- ✅ 100% type annotation coverage

**Functional Metrics:**
- ✅ Full round-robin tournament completion
- ✅ All three strategies functional (random, history, LLM)
- ✅ Concurrent multi-match execution
- ✅ Fault tolerance with agent failures

**Performance Metrics:**
- ✅ <2 second match execution time
- ✅ Support for 4+ concurrent players
- ✅ Clean startup/shutdown of all services
- ✅ Accurate standings tracking

---

**Implementation Confidence: 9/10**

This PRP provides comprehensive context, proven patterns, and executable validation for reliable one-pass implementation of a production-ready distributed AI agent system.