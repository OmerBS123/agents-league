# Distributed AI Agent League System (Python/FastAPI)

We need to build a complete local distributed system involving three distinct types of agents communicating over HTTP/JSON-RPC.

## 1. The Player Agent (agents/player.py)

**Role:** The participant in the game.

**Server:** Implements a FastAPI server to receive invitations and game calls.

**Registration:** Must automatically register itself with the League Manager on startup (with retries).

**Strategies:** The player must support 3 selectable strategies via command-line arguments:

- `random`: Randomly chooses "even" or "odd".
- `history`: Keeps a local record of opponent moves and chooses the counter to their most frequent choice.
- `llm`: Connects to a local Ollama instance (URL: http://localhost:11434) to decide the move using a prompt.

**Tools:** Must implement `handle_game_invitation`, `choose_parity`, and `notify_match_result`.

## 2. The League Manager (agents/league_manager.py)

**Role:** The administrator of the league.

**Server:** A FastAPI server (Port 8000) that accepts `LEAGUE_REGISTER_REQUEST`.

**Logic:**
- Stores the list of registered players in memory.
- Generates a Round-Robin schedule.
- Broadcasts round announcements.
- Instructs the Referee to play specific matches.
- Maintains and updates the league standings based on match reports.

## 3. The Referee (agents/referee.py)

**Role:** The neutral orchestrator of individual matches.

**Server:** A FastAPI server (Port 8001).

**Logic (Game Loop):**
- Receives match instructions from Manager.
- Sends `GAME_INVITATION` to two players.
- Requests `CHOOSE_PARITY` from both players simultaneously.
- Draws a random number (1-10) to determine the result.
- Sends `GAME_OVER` to players.
- Sends `MATCH_RESULT_REPORT` to the Manager.

## 4. Orchestration

A shell script (`start_league.sh`) to launch the Manager, Referee, and 4 Players (Ports 8101-8104) in parallel using `uv run`.

## 🛡️ MANDATORY CODE QUALITY REQUIREMENTS

**ALL Python code in this project MUST pass the complete quality audit before deployment. This is non-negotiable.**

### Quality Tools Stack (Required)
Install the complete quality stack:
```bash
uv add --dev ruff mypy bandit radon vulture deptry
```

### Mandatory Quality Checks

#### Ruff (Style & Linting) - MUST PASS
- No style violations (spacing, quotes, imports).
- All unused imports removed.
- Consistent code formatting.
- **Command:** `uv run ruff check . --fix && uv run ruff format .`

#### MyPy (Strict Type Checking) - MUST PASS
- All functions must have type annotations.
- No type mismatches allowed.
- Strict mode required: `--strict`.
- **Command:** `uv run mypy . --strict`

#### Bandit (Security Scan) - MUST PASS
- Zero medium/high security vulnerabilities.
- No hardcoded secrets.
- No shell injection risks.
- **Command:** `uv run bandit -r . --exclude /tests -ll`

#### Radon (Complexity Analysis) - MUST PASS
- Cyclomatic complexity ≤ 10 (Grade A-B only).
- No functions with Grade C, D, or F complexity.
- **Command:** `uv run radon cc . --min C`

#### Vulture (Dead Code Detection) - MUST PASS
- No unused functions, classes, or variables.
- No unreachable code.
- **Command:** `uv run vulture . whitelist.py`

#### Deptry (Dependency Validation) - MUST PASS
- All imported packages declared in pyproject.toml.
- No unused dependencies.
- **Command:** `uv run deptry .`

## 🔍 MANDATORY CODE REVIEW AGENTS

After passing all quality tools, code MUST be reviewed by specialized agents before deployment. This provides an additional layer of logic, security, and standards validation.

### Required Agent Reviews (Sequential Order)

#### 1. Logic & Security Review: `feature-dev:code-reviewer` (REQUIRED FIRST)
- **Target:** `agents/`, `shared/`
- **Focus:** Logic errors, edge cases, race conditions in async code, and security vulnerabilities (injection, data leaks).
- **Usage:**
```python
Task(subagent_type="feature-dev:code-reviewer", prompt="Review referee logic for race conditions and timeout handling")
```

#### 2. Style & Standards Review: `pr-review-toolkit:code-reviewer` (REQUIRED SECOND)
- **Target:** All files.
- **Focus:** Project structure adherence, docstring completeness, variable naming, and consistent pattern usage.
- **Usage:**
```python
Task(subagent_type="pr-review-toolkit:code-reviewer", prompt="Review code for style violations and project standards compliance")
```

### Complete Development Workflow (MANDATORY SEQUENCE)
1. **Quality Tools Audit:** Run the full uv stack (Ruff, MyPy, etc.).
2. **Logic & Security Review:** Pass `feature-dev:code-reviewer`.
3. **Style & Standards Review:** Pass `pr-review-toolkit:code-reviewer`.
4. **Commit/Deploy:** Only allowed after steps 1-3 pass.

## EXAMPLES

In the `examples/` folder, there are referenced implementations. Read these files to understand the coding standards for this project.

- `examples/fastapi_server.py` - Template for `agents/player.py`, `agents/league_manager.py`, and `agents/referee.py`. Demonstrates FastAPI app structure, strict Pydantic models, and exception handling integration.

- `examples/pydantic_schema.py` - Template for `shared/schemas.py`. Shows strict TypedDict or Pydantic V2 usage with field validators.

- `examples/httpx_client.py` - Template for inter-agent communication. Demonstrates AsyncClient usage, timeout settings, and exponential backoff.

- `examples/ollama_integration.py` - Template for the Player's llm strategy, including the fallback mechanism.

## DOCUMENTATION & REFERENCES

### Context Window Configuration
Include the following Expert Agents in your context window to ensure implementation follows best practices:

- **agent: claude-code-guide** - why: Use Claude Code guide agent - has all Claude Code and Agent SDK documentation.
- **agent: uv-expert** - why: Use uv agent-expert - has documentation for modern Python project management, dependency resolution, and uv run commands.
- **agent: fastapi-expert** - why: Use FastAPI agent-expert - has documentation for building high-performance APIs, dependency injection, and Pydantic integration.
- **agent: pydantic-expert** - why: Use Pydantic agent-expert - has documentation for V2 strict data validation, serialization, and field validators.
- **agent: httpx-expert** - why: Use HTTPX agent-expert - has documentation for async HTTP clients, timeout configurations, and connection pooling.

## DETAILED DESIGN SPECIFICATION

### 1. File Responsibilities & Class Design

#### shared/logger.py (Singleton Pattern)
- **Responsibility:** Centralized logging configuration.
- **Requirement:** Must expose a `get_logger(name: str)` function.
- **Implementation:** Ensures only one logger configuration exists for the application lifecycle. Formats logs as JSON Lines (`{"timestamp": "...", "level": "INFO", "agent_id": "...", "message": "..."}`).

#### shared/exceptions.py
- **Responsibility:** centralized custom exception classes.
- **Classes:** `LeagueError` (base), `RegistrationError`, `MatchError`, `StrategyError`.
- **Usage:** All agents must raise these specific exceptions rather than generic `Exception`.

#### agents/player.py
- **Class `PlayerServer`:**
  - **Responsibility:** Main entry point. Initializes FastAPI and Strategy.
  - **Method `startup()`:** Async task for registering with Manager (retries required).
  - **Method `dispatch()`:** Routes JSON-RPC envelopes to tools.

- **Class `Strategy` (Abstract):**
  - **Subclasses:** `RandomStrategy`, `HistoryStrategy`, `LLMStrategy`.
  - **Requirement:** `LLMStrategy` must handle Ollama timeouts gracefully.

#### agents/league_manager.py
- **Class `LeagueManager`:**
  - **State:** `players: List[PlayerProfile]`, `schedule: List[Matchup]`, `standings: Dict`.
  - **Logic:** Round-Robin scheduling and match orchestration loop.

#### agents/referee.py
- **Class `MatchOrchestrator`:**
  - **Responsibility:** Ephemeral state machine for a single match.
  - **Logic:** Parallel `collect_moves()` calls. Enforces strict timeouts (default 2s) on players.

#### shared/schemas.py
- **Responsibility:** Data contracts.
- **Requirement:** Use strict Pydantic models (or TypedDict where appropriate) for `MCPEnvelope`, `GameInvitation`, `MatchResult`.

#### consts/
- **Purpose:** Single Source of Truth.
- **Content:** `PORTS` (8000/8001/81xx), `URLS` (Ollama), `TIMEOUTS`.

## OTHER CONSIDERATIONS

- **Dependency Management:** Strictly use `uv` and `pyproject.toml`. Do not use `requirements.txt` or `pip` directly.

- **Code Quality Compliance:** All code must pass the complete 3-step validation: (1) Quality tools audit (Ruff/MyPy/Bandit/etc.), (2) `feature-dev:code-reviewer` for logic/security, (3) `pr-review-toolkit:code-reviewer` for style/standards.

- **Singleton Logger:** The `logger.py` must expose a `get_logger()` function that ensures only one logger instance exists for the application lifecycle.

- **Concurrency:** Use `async/await` for all agent interactions, SDK calls, and external requests. Never use blocking I/O in the async context.

- **Error Handling:** Implement global exception handlers in FastAPI to catch custom errors from `shared/exceptions.py` and return clean JSON error responses (HTTP 4xx/5xx).

## Project Structure

```
my_mcp_league/
├── pyproject.toml         # uv managed dependencies
├── start_league.sh        # Startup script
├── whitelist.py           # Configuration for Vulture (dead code)
├── consts/
│   └── __init__.py
├── agents/
│   ├── __init__.py
│   ├── player.py
│   ├── league_manager.py
│   └── referee.py
├── shared/
│   ├── __init__.py
│   ├── schemas.py
│   ├── logger.py          # Singleton Logger
│   └── exceptions.py      # Custom Error Handling
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```