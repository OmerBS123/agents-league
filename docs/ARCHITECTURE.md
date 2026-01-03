# Architecture Documentation

## Distributed AI Agent League System

**Version:** 1.0.0
**Last Updated:** January 2026

---

## 1. System Overview

The Distributed AI Agent League is a microservices-based multi-agent system implementing automated tournament orchestration for the Even/Odd parity game.

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Distributed AI Agent League                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │  League Manager  │◄───────►│     Referee      │              │
│  │    (Port 8000)   │         │   (Port 8001)    │              │
│  └────────┬─────────┘         └────────┬─────────┘              │
│           │                            │                         │
│           │ Registration               │ Match Control           │
│           │                            │                         │
│           ▼                            ▼                         │
│  ┌────────────────────────────────────────────────────┐         │
│  │                   Player Agents                     │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │         │
│  │  │ P01     │ │ P02     │ │ P03     │ │ P04     │  │         │
│  │  │ random  │ │ history │ │ llm     │ │ random  │  │         │
│  │  │ :8101   │ │ :8102   │ │ :8103   │ │ :8104   │  │         │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Communication

```
                    HTTP/JSON-RPC
    ┌─────────────────────────────────────────┐
    │                                         │
    ▼                                         │
┌────────┐  REGISTER   ┌────────┐  SCHEDULE  ┌────────┐
│ Player │────────────►│ League │───────────►│Referee │
│        │◄────────────│Manager │◄───────────│        │
└────────┘  REG_ACK    └────────┘  MATCH_RPT └────────┘
    ▲                                         │
    │           INVITATION / MOVES            │
    └─────────────────────────────────────────┘
```

---

## 2. Component Details

### 2.1 League Manager

**Purpose:** Central coordinator for tournament lifecycle management

**Architecture Pattern:** Service + FastAPI Controller

```
┌─────────────────────────────────────────────┐
│              League Manager                  │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐   │
│  │         TournamentManager           │   │
│  │  - players: Dict[str, PlayerProfile]│   │
│  │  - schedule: List[MatchSchedule]    │   │
│  │  - completed_matches: Dict          │   │
│  └─────────────────────────────────────┘   │
│                    │                        │
│  ┌─────────────────────────────────────┐   │
│  │       LeagueManagerService          │   │
│  │  - handle_registration()            │   │
│  │  - handle_match_report()            │   │
│  │  - _match_orchestration_loop()      │   │
│  └─────────────────────────────────────┘   │
│                    │                        │
│  ┌─────────────────────────────────────┐   │
│  │         FastAPI Routes              │   │
│  │  - /health                          │   │
│  │  - /register                        │   │
│  │  - /mcp                             │   │
│  │  - /tournament/status               │   │
│  │  - /tournament/standings            │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Key Responsibilities:**
1. Accept and validate player registrations
2. Generate round-robin tournament schedule
3. Dispatch match assignments to Referee
4. Aggregate match results and update standings
5. Broadcast tournament completion

### 2.2 Referee

**Purpose:** Match orchestrator enforcing game rules and timing

**Architecture Pattern:** State Machine + Service

```
┌─────────────────────────────────────────────┐
│                  Referee                     │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐   │
│  │         MatchOrchestrator           │   │
│  │  - status: MatchStatus              │   │
│  │  - scores: Dict[str, int]           │   │
│  │  - game_history: List[GameResult]   │   │
│  │                                      │   │
│  │  States: SCHEDULED -> IN_PROGRESS   │   │
│  │          -> COMPLETED/ERROR         │   │
│  └─────────────────────────────────────┘   │
│                    │                        │
│  ┌─────────────────────────────────────┐   │
│  │          RefereeService             │   │
│  │  - schedule_match()                 │   │
│  │  - _execute_match_workflow()        │   │
│  │  - _report_match_results()          │   │
│  └─────────────────────────────────────┘   │
│                    │                        │
│  ┌─────────────────────────────────────┐   │
│  │         FastAPI Routes              │   │
│  │  - /health                          │   │
│  │  - /mcp                             │   │
│  │  - /matches/{id}                    │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Match State Machine:**

```
    ┌──────────┐
    │SCHEDULED │
    └────┬─────┘
         │ invite_players()
         ▼
    ┌──────────┐
    │IN_PROGRESS│◄────┐
    └────┬─────┘      │
         │            │ next round
         ▼            │
    ┌──────────┐      │
    │ execute  │──────┘
    │  round   │
    └────┬─────┘
         │ all rounds complete
         ▼
    ┌──────────┐
    │COMPLETED │
    └──────────┘
```

### 2.3 Player Agent

**Purpose:** Game participant with pluggable strategy

**Architecture Pattern:** Strategy Pattern + Service

```
┌─────────────────────────────────────────────┐
│               Player Agent                   │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐   │
│  │         Strategy (Abstract)         │   │
│  │  + choose_move() -> ParityChoice    │   │
│  └──────────────┬──────────────────────┘   │
│                 │                           │
│     ┌───────────┼───────────┐              │
│     ▼           ▼           ▼              │
│  ┌──────┐  ┌────────┐  ┌────────┐         │
│  │Random│  │History │  │  LLM   │         │
│  └──────┘  └────────┘  └────────┘         │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │          PlayerService              │   │
│  │  - handle_invitation()              │   │
│  │  - handle_move_request()            │   │
│  │  - handle_game_over()               │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Strategy Implementations:**

| Strategy | Algorithm | Complexity | Use Case |
|----------|-----------|------------|----------|
| Random | Uniform random | O(1) | Baseline |
| History | Pattern analysis | O(n) | Counter-play |
| LLM | Ollama inference | O(1)* | Intelligent |

*LLM has constant algorithmic complexity but variable latency

---

## 3. Communication Protocol

### 3.1 MCP Envelope Structure

All inter-agent communication uses the MCP (Message Communication Protocol) envelope:

```python
MCPEnvelope {
    protocol: "league.v2"           # Protocol version
    message_type: MessageType       # Enum of message types
    sender: "role:id"               # e.g., "player:P01"
    recipient: "role:id" | None     # Target agent
    conversation_id: UUID           # Thread correlation
    timestamp: ISO-8601             # Message timestamp
    data: Dict[str, Any]            # Type-specific payload
}
```

### 3.2 Message Flow: Registration

```
Player                  League Manager
   │                         │
   │  REGISTER               │
   │  {display_name,         │
   │   contact_endpoint,     │
   │   strategies}           │
   │────────────────────────►│
   │                         │
   │  REGISTER_ACK           │
   │  {accepted: true,       │
   │   agent_id,             │
   │   tournament_info}      │
   │◄────────────────────────│
   │                         │
```

### 3.3 Message Flow: Match Execution

```
League Manager          Referee              Player A    Player B
      │                    │                    │            │
      │  SCHEDULE_MATCH    │                    │            │
      │───────────────────►│                    │            │
      │                    │                    │            │
      │                    │  INVITATION        │            │
      │                    │───────────────────►│            │
      │                    │                    │            │
      │                    │  INVITATION        │            │
      │                    │───────────────────────────────►│
      │                    │                    │            │
      │                    │  INVITATION_ACK    │            │
      │                    │◄───────────────────│            │
      │                    │                    │            │
      │                    │  INVITATION_ACK    │            │
      │                    │◄───────────────────────────────│
      │                    │                    │            │
      │                    │  ══════════════════════════════│
      │                    │  │ For each round:             │
      │                    │  │                             │
      │                    │  MOVE_CALL (parallel)          │
      │                    │─────────────────►│◄────────────│
      │                    │                  │             │
      │                    │  MOVE_RESPONSE   │             │
      │                    │◄─────────────────│             │
      │                    │◄───────────────────────────────│
      │                    │  │                             │
      │                    │  ══════════════════════════════│
      │                    │                    │            │
      │                    │  GAME_OVER        │            │
      │                    │───────────────────►│            │
      │                    │───────────────────────────────►│
      │                    │                    │            │
      │  MATCH_REPORT      │                    │            │
      │◄───────────────────│                    │            │
      │                    │                    │            │
```

---

## 4. Data Models

### 4.1 Core Enums

```python
class MessageType(Enum):
    REGISTER = "LEAGUE_REGISTER_REQUEST"
    REGISTER_ACK = "LEAGUE_REGISTER_RESPONSE"
    INVITATION = "GAME_INVITATION"
    INVITATION_ACK = "GAME_JOIN_ACK"
    MOVE_CALL = "CHOOSE_PARITY"
    MOVE_RESPONSE = "CHOOSE_PARITY_RESPONSE"
    GAME_OVER = "GAME_OVER"
    MATCH_REPORT = "MATCH_RESULT_REPORT"
    SCHEDULE_MATCH = "SCHEDULE_MATCH"

class ParityChoice(Enum):
    EVEN = "even"
    ODD = "odd"

class MatchStatus(Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
```

### 4.2 Key Data Classes

```python
@dataclass
class PlayerProfile:
    agent_id: str
    display_name: str
    contact_endpoint: str
    strategies: List[str]
    matches_played: int = 0
    matches_won: int = 0

@dataclass
class MatchSchedule:
    match_id: str
    player_a_id: str
    player_b_id: str
    status: str = "scheduled"
    winner_id: Optional[str] = None

class GameResultData(BaseModel):
    match_id: str
    round_id: int
    player_a_choice: ParityChoice
    player_b_choice: ParityChoice
    random_number: int
    winner_id: Optional[str]
```

---

## 5. Error Handling

### 5.1 Exception Hierarchy

```
LeagueError (Base)
├── RegistrationError
│   └── Used when player registration fails
├── MatchError
│   └── Used for match execution failures
├── StrategyError
│   └── Used when strategy selection fails
├── NetworkError
│   └── Used for HTTP communication failures
├── OperationTimeoutError
│   └── Used when operations exceed timeout
├── LLMError
│   └── Used for Ollama integration failures
└── ProtocolError
    └── Used for MCP validation failures
```

### 5.2 Retry Strategy

```
┌─────────────────────────────────────────┐
│           Exponential Backoff            │
├─────────────────────────────────────────┤
│  Attempt 1: Immediate                    │
│  Attempt 2: 2s + jitter (0-0.5s)        │
│  Attempt 3: 4s + jitter (0-0.5s)        │
│  Max Retries: 3                          │
│  Circuit Breaker: 5 failures → open     │
│  Reset Time: 30 seconds                  │
└─────────────────────────────────────────┘
```

---

## 6. Deployment Architecture

### 6.1 Local Development

```
┌─────────────────────────────────────────┐
│            localhost                     │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Manager │  │ Referee │  │ Players │ │
│  │  :8000  │  │  :8001  │  │:8101-04 │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│                    │                    │
│              ┌─────┴─────┐              │
│              │  Ollama   │              │
│              │  :11434   │              │
│              └───────────┘              │
└─────────────────────────────────────────┘
```

### 6.2 Startup Sequence

```
1. start_league.sh
   │
   ├─► Install dependencies (uv sync)
   │
   ├─► Start League Manager (port 8000)
   │   └─► Wait for health check
   │
   ├─► Start Referee (port 8001)
   │   └─► Wait for health check
   │
   ├─► Start Players (ports 8101-8104)
   │   ├─► Each player registers with League Manager
   │   └─► Wait for health check
   │
   └─► Tournament begins automatically
```

---

## 7. Security Considerations

### 7.1 Current Security Model

| Aspect | Implementation | Status |
|--------|----------------|--------|
| Authentication | None (localhost only) | Acceptable for dev |
| Input Validation | Pydantic strict mode | Implemented |
| CORS | Allow all origins | Configurable |
| Secrets | No hardcoded values | Verified |

### 7.2 Future Security Enhancements

1. JWT-based agent authentication
2. Rate limiting per agent
3. TLS for inter-agent communication
4. Audit logging for all actions

---

## 8. Monitoring and Observability

### 8.1 Logging Strategy

```python
# Structured JSON logging with context
{
    "timestamp": "2026-01-02T12:00:00Z",
    "level": "INFO",
    "service": "league-manager",
    "request_id": "uuid",
    "agent_id": "player:P01",
    "match_id": "M-001",
    "message": "Registration accepted",
    "duration_ms": 42.5
}
```

### 8.2 Health Checks

All agents expose `/health` endpoint returning:

```json
{
    "status": "healthy",
    "service": "league-manager",
    "agent_id": "manager:league",
    "active_matches": 2,
    "tournament_active": true
}
```

---

## 9. Architectural Decision Records (ADRs)

### ADR-001: FastAPI over Flask

**Context:** Need async-capable web framework
**Decision:** Use FastAPI for native async support
**Consequences:** Better performance, automatic OpenAPI docs

### ADR-002: Pydantic v2 for Validation

**Context:** Need strict message validation
**Decision:** Use Pydantic v2 with strict mode
**Consequences:** Type safety, cross-field validation, performance

### ADR-003: HTTP over WebSocket

**Context:** Choose communication protocol
**Decision:** Use HTTP/JSON-RPC for simplicity
**Consequences:** Easier debugging, stateless agents, slight latency

### ADR-004: Strategy Pattern for Players

**Context:** Need pluggable decision algorithms
**Decision:** Implement Strategy pattern
**Consequences:** Easy to add strategies, clean separation

### ADR-005: Local LLM with Ollama

**Context:** Want LLM strategy without API costs
**Decision:** Use Ollama for local inference
**Consequences:** Free operation, requires local setup

---

## 10. Performance Characteristics

| Metric | Typical Value | Worst Case |
|--------|---------------|------------|
| Registration latency | 10-50ms | 500ms |
| Move decision (random) | 1-5ms | 10ms |
| Move decision (history) | 5-20ms | 50ms |
| Move decision (LLM) | 500-1500ms | 3000ms |
| Match completion | 5-15s | 30s |
| Tournament (4 players) | 30-60s | 120s |

---

## 11. Future Architecture

### 11.1 Planned Enhancements

```
┌─────────────────────────────────────────────────────┐
│              Future Architecture                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │   Web   │  │   API   │  │ Message │             │
│  │Dashboard│◄►│ Gateway │◄►│  Queue  │             │
│  └─────────┘  └─────────┘  └────┬────┘             │
│                                  │                   │
│       ┌──────────────────────────┼──────────┐       │
│       ▼                          ▼          ▼       │
│  ┌─────────┐            ┌─────────┐  ┌─────────┐   │
│  │ League  │            │ Referee │  │ Referee │   │
│  │ Manager │            │   (1)   │  │   (2)   │   │
│  └─────────┘            └─────────┘  └─────────┘   │
│       │                                             │
│       ▼                                             │
│  ┌─────────┐                                        │
│  │PostgreSQL│  ← Persistent storage                 │
│  └─────────┘                                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```
