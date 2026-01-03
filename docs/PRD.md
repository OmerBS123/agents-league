# Product Requirements Document (PRD)

## Distributed AI Agent League System

**Version:** 1.0.0
**Date:** January 2026
**Author:** Development Team
**Status:** Implementation Complete

---

## 1. Executive Summary

The Distributed AI Agent League System is a multi-agent platform that orchestrates automated tournaments for the Even/Odd parity game. The system demonstrates distributed systems concepts including service orchestration, message protocols, circuit breakers, and resilient inter-agent communication.

### 1.1 Problem Statement

Traditional game simulations and AI agent competitions often suffer from:
- Tight coupling between game logic and agent implementations
- Lack of fault tolerance in multi-agent communication
- Difficulty in adding new strategies or game types
- No standardized protocol for agent communication
- Limited observability into agent decision-making

### 1.2 Solution Overview

A microservices-based architecture where autonomous agents communicate via HTTP/JSON-RPC, enabling:
- Decoupled agent implementations with pluggable strategies
- Resilient communication with circuit breakers and retries
- Centralized tournament coordination with decentralized execution
- Type-safe message passing with Pydantic validation
- Optional LLM integration for intelligent decision-making

---

## 2. Goals and Objectives

### 2.1 Primary Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| **Distributed Architecture** | Build a functioning multi-agent system | 3+ agents communicating successfully |
| **Fault Tolerance** | Handle network failures gracefully | 95%+ match completion rate |
| **Extensibility** | Easy addition of new strategies | New strategy in <50 lines of code |
| **Observability** | Comprehensive logging and monitoring | Request tracing across all agents |

### 2.2 Secondary Goals

- Demonstrate modern Python async patterns
- Showcase Pydantic v2 validation capabilities
- Provide educational example of distributed systems
- Enable LLM-powered game strategies

### 2.3 Non-Goals

- Production-scale deployment (this is an educational project)
- Persistent storage across restarts
- User authentication and authorization
- Multi-game support (future enhancement)

---

## 3. Functional Requirements

### 3.1 Agent Registration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-001 | Players must register with League Manager before participating | Must Have |
| FR-002 | Registration must include contact endpoint and supported strategies | Must Have |
| FR-003 | League Manager must validate registration data | Must Have |
| FR-004 | Duplicate registrations must be rejected | Should Have |

### 3.2 Tournament Management

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-010 | League Manager must generate round-robin schedule | Must Have |
| FR-011 | Schedule must ensure each player faces every other player once | Must Have |
| FR-012 | Tournament must start automatically when 2+ players registered | Should Have |
| FR-013 | Standings must update in real-time after each match | Must Have |

### 3.3 Match Execution

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-020 | Referee must send invitations to both players before match | Must Have |
| FR-021 | Players must respond to invitations within 5 seconds | Must Have |
| FR-022 | Moves must be collected in parallel from both players | Must Have |
| FR-023 | Move timeout must be enforced (2 seconds default) | Must Have |
| FR-024 | Match consists of 10 rounds; majority wins | Must Have |
| FR-025 | Match results must be reported to League Manager | Must Have |

### 3.4 Game Rules (Even/Odd Parity)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-030 | Random number generated between 1-10 each round | Must Have |
| FR-031 | Players choose "even" or "odd" simultaneously | Must Have |
| FR-032 | Winner determined by sum parity matching choice | Must Have |
| FR-033 | Ties possible when both players choose same and win/lose | Must Have |

### 3.5 Strategy Support

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-040 | Random strategy: uniform random selection | Must Have |
| FR-041 | History strategy: pattern-based counter-play | Must Have |
| FR-042 | LLM strategy: Ollama-powered decision making | Should Have |
| FR-043 | Strategy must be selectable at player startup | Must Have |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Move decision latency | < 2000ms |
| NFR-002 | Match completion time | < 30 seconds |
| NFR-003 | Agent startup time | < 5 seconds |
| NFR-004 | Concurrent matches supported | 4 simultaneous |

### 4.2 Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-010 | Match completion rate | > 95% |
| NFR-011 | Agent recovery from network errors | Automatic with exponential backoff |
| NFR-012 | Graceful shutdown on SIGINT | All agents cleanup properly |

### 4.3 Maintainability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-020 | Code documentation | 100% public API documented |
| NFR-021 | Type hints coverage | 100% of function signatures |
| NFR-022 | Test coverage | > 70% |

### 4.4 Security

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-030 | No secrets in source code | 0 hardcoded credentials |
| NFR-031 | Input validation | All external input validated |
| NFR-032 | CORS configuration | Configurable origins |

---

## 5. System Components

### 5.1 League Manager (Port 8000)

**Purpose:** Central coordinator for tournament management

**Responsibilities:**
- Player registration and validation
- Round-robin schedule generation
- Match assignment to Referee
- Standings calculation and broadcasting
- Tournament lifecycle management

**Endpoints:**
- `GET /health` - Health check
- `POST /register` - Player registration
- `POST /mcp` - MCP protocol messages
- `GET /tournament/status` - Tournament state
- `GET /tournament/standings` - Player rankings

### 5.2 Referee (Port 8001)

**Purpose:** Match orchestrator enforcing game rules

**Responsibilities:**
- Player invitation and acceptance
- Parallel move collection
- Random number generation
- Winner determination per round
- Match result reporting

**Endpoints:**
- `GET /health` - Health check
- `POST /mcp` - MCP protocol messages
- `GET /matches/{id}` - Match status

### 5.3 Player Agents (Ports 8101-8104)

**Purpose:** Game participants with configurable strategies

**Responsibilities:**
- Registration with League Manager
- Responding to match invitations
- Move selection using configured strategy
- History tracking for pattern analysis

**Endpoints:**
- `GET /health` - Health check
- `POST /mcp` - MCP protocol messages

---

## 6. Data Models

### 6.1 MCP Envelope

```python
{
    "protocol": "league.v2",
    "message_type": "CHOOSE_PARITY",
    "sender": "referee:REF-MAIN",
    "recipient": "player:P01",
    "conversation_id": "uuid",
    "timestamp": "ISO-8601",
    "data": { ... }
}
```

### 6.2 Key Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| LEAGUE_REGISTER_REQUEST | Player -> Manager | Registration |
| GAME_INVITATION | Referee -> Player | Match invite |
| CHOOSE_PARITY | Referee -> Player | Move request |
| MATCH_RESULT_REPORT | Referee -> Manager | Results |

---

## 7. Dependencies

### 7.1 Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | >= 0.100.0 | Web framework |
| Pydantic | >= 2.0.0 | Data validation |
| HTTPX | >= 0.25.0 | Async HTTP client |
| Uvicorn | >= 0.20.0 | ASGI server |

### 7.2 Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Ollama | Local install | LLM strategy |

---

## 8. Constraints and Assumptions

### 8.1 Constraints

- Python 3.13+ required
- All agents run on localhost
- No persistent storage between runs
- Single tournament at a time

### 8.2 Assumptions

- Network connectivity between agents is reliable
- Ports 8000-8001 and 8101-8104 are available
- Users have basic command-line familiarity
- Ollama installed separately if using LLM strategy

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| System starts successfully | 100% | All agents healthy |
| Matches complete without errors | > 95% | Match completion rate |
| Move latency within timeout | 100% | Performance logs |
| All tests pass | 100% | pytest results |

---

## 10. Timeline and Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1: Core Infrastructure | Schemas, logging, exceptions | Complete |
| M2: League Manager | Registration, scheduling | Complete |
| M3: Referee | Match orchestration | Complete |
| M4: Player Agent | Strategies, gameplay | Complete |
| M5: Integration | End-to-end testing | Complete |
| M6: Documentation | PRD, Architecture, README | Complete |

---

## 11. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Network timeouts | Match failures | Medium | Circuit breakers, retries |
| LLM unavailable | Strategy fallback | Low | Automatic random fallback |
| Port conflicts | Startup failure | Low | Clear error messages |

---

## 12. Future Enhancements

1. **WebSocket Support** - Real-time bidirectional communication
2. **Web Dashboard** - React-based tournament visualization
3. **Persistent Storage** - PostgreSQL for history
4. **More Games** - Rock-paper-scissors, tic-tac-toe
5. **Docker Compose** - Containerized deployment
6. **Metrics & Observability** - Prometheus/Grafana integration
