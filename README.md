# Distributed AI Agent League System

A multi-agent system for playing an even/odd parity game with automated tournament management and multiple player strategies.

**Architecture:** Distributed microservices with FastAPI agents communicating via HTTP/JSON-RPC.
- **League Manager** - Central coordinator managing agent registration and tournament scheduling
- **Referee** - Match orchestrator enforcing game rules and timing constraints
- **Player Agents** - Game participants with pluggable strategies (random, history-based, LLM)
- **Round-Robin Tournament** - Automated scheduling with real-time standings tracking

---

## Table of Contents

- [🚀 Quick Start](#-quick-start)
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Development](#development)
- [Documentation](#documentation)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install uv package manager (if not already installed)
brew install uv

# Clone and install dependencies
git clone <repo-url>
cd distributed-ai-agent-league
uv sync --all-groups
```

### Minimal Example

**Start the entire league system:**
```bash
./start_league.sh
```

**What this does:**
1. ✅ **Installs** all dependencies using uv
2. ✅ **Starts** League Manager (port 8000) - central coordinator
3. ✅ **Starts** Referee (port 8001) - match orchestrator
4. ✅ **Starts** 4 Player agents (ports 8101-8104) with different strategies
5. ✅ **Registers** all players with the League Manager
6. ✅ **Begins** automated round-robin tournament
7. ✅ **Monitors** all processes and handles graceful shutdown on Ctrl+C

### Common Usage Patterns

**Monitor tournament status:**
```bash
# Get current tournament status
curl http://localhost:8000/tournament/status

# View live standings
curl http://localhost:8000/tournament/standings

# Check agent health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8101/health
```

**Start individual agents manually:**
```bash
# Start League Manager only
uv run python -m agents.league_manager

# Start a single player with specific strategy
uv run python -m agents.player --port 8101 --strategy random --agent-id player:P01

# Start Referee
uv run python -m agents.referee
```

**Run integration tests:**
```bash
uv run pytest test_integration.py -v
```

### Output

**When running start_league.sh, you'll see:**
```
🎮 Distributed AI Agent League System
=====================================
📦 Installing dependencies...
🏗️  Starting distributed agent system...
🚀 Starting League Manager on port 8000...
   Started League Manager (PID: 12345)
✅ League Manager is ready
🚀 Starting Referee on port 8001...
   Started Referee (PID: 12346)
✅ Referee is ready
🚀 Starting Player P01 (random) on port 8101...
   Started Player P01 (random) (PID: 12347)
✅ Player P01 (random) is ready
[... more player startups ...]
🎉 All agents started successfully!
```

### Accessing Your System

Once running, the system exposes several endpoints:

**League Manager (localhost:8000):**
- `/health` - Health check endpoint
- `/register` - Agent registration
- `/mcp` - MCP protocol endpoint
- `/tournament/status` - Current tournament state
- `/tournament/standings` - Player rankings and statistics

**Referee (localhost:8001):**
- `/health` - Health check
- `/mcp` - MCP protocol endpoint for match orchestration

**Player Agents (localhost:8101-8104):**
- `/health` - Health check
- `/mcp` - MCP protocol endpoint for game moves

---

## Overview

A distributed multi-agent system implementing an automated tournament for the even/odd parity game. The system demonstrates key distributed systems concepts including service orchestration, message protocols, circuit breakers, and resilient inter-agent communication.

**Key capabilities:**
- **Distributed Architecture** - Microservices communicating via HTTP with JSON-RPC message envelopes
- **Multiple Agent Strategies** - Random, history-based, and LLM-powered decision making
- **Automated Tournament Management** - Round-robin scheduling with concurrent match execution
- **Resilient Communication** - Circuit breakers, exponential backoff, and timeout handling
- **Real-time Monitoring** - Health checks, heartbeats, and tournament standings tracking

**Technology stack:**
- **FastAPI** - High-performance async web framework for agent APIs
- **Pydantic** - Strict schema validation with cross-field validation rules
- **HTTPX** - Async HTTP client with connection pooling
- **Uvicorn** - ASGI server with graceful shutdown support
- **Ollama** (optional) - Local LLM integration for AI-powered strategies

**What problem it solves:**

This system demonstrates how to build distributed multi-agent systems with:
- Reliable message passing between autonomous agents
- Centralized coordination with decentralized execution
- Fault tolerance through circuit breakers and retry mechanisms
- Type-safe communication protocols using Pydantic schemas
- Automated tournament scheduling and result tracking

## Features

- **Strategy Pattern Implementation** - Pluggable player strategies allowing easy addition of new decision-making algorithms (random, history-based, LLM)
- **MCP Envelope Protocol** - Type-safe message passing with strict validation, versioning, and cross-field consistency checks
- **Circuit Breaker Pattern** - Automatic failure detection and recovery to prevent cascade failures between agents
- **Graceful Orchestration** - Startup script manages agent lifecycle with health checks, dependency ordering, and clean shutdown
- **Round-Robin Scheduling** - Automatic tournament generation ensuring every player faces every other player exactly once
- **Real-time Standings** - Live tracking of wins, losses, and rankings with concurrent match execution
- **Resilient HTTP Communication** - Exponential backoff, jitter, connection pooling, and timeout handling for reliable inter-agent messaging
- **Comprehensive Logging** - Structured JSON logging with request IDs, performance metrics, and contextual information
- **Optional LLM Integration** - Ollama-powered strategy using local language models for intelligent game decisions

## Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (>=0.4.0)
- (Optional) [Ollama](https://ollama.ai/) for LLM strategy

### Development Mode

Clone the repository and install all dependencies:

```bash
# Clone the repository
git clone <repo-url>
cd distributed-ai-agent-league

# Install dependencies (includes dev and test groups)
uv sync --all-groups
```

This installs the project in development mode where code changes take effect immediately without reinstallation.

### Optional: Ollama Setup

If you want to use the LLM strategy, install Ollama and download a model:

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama3

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

## Configuration

### Startup Script Options

```bash
./start_league.sh [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message with detailed information |
| (no options) | Start all agents with default configuration |

### Environment Configuration

The system uses configuration constants defined in `consts/__init__.py`:

| Configuration | Value | Description |
|---------------|-------|-------------|
| **Network Ports** | | |
| `LEAGUE_MANAGER_PORT` | `8000` | League Manager API port |
| `REFEREE_PORT` | `8001` | Referee API port |
| `PLAYER_PORTS` | `[8101-8104]` | Player agent ports |
| **Timing** | | |
| `DEFAULT_TIMEOUT` | `2.0` seconds | Default HTTP operation timeout |
| `MOVE_TIMEOUT` | `2000` milliseconds | Player move decision timeout |
| `MATCH_TIMEOUT` | `30.0` seconds | Maximum match duration |
| **Game Rules** | | |
| `GAME_TYPE` | `even_odd` | Parity game type |
| `MAX_ROUNDS_PER_MATCH` | `10` | Rounds per match |
| `RANDOM_NUMBER_MIN` | `1` | Minimum random number |
| `RANDOM_NUMBER_MAX` | `10` | Maximum random number |
| **Strategies** | | |
| `AVAILABLE_STRATEGIES` | `["random", "history", "llm"]` | Supported player strategies |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `DEFAULT_OLLAMA_MODEL` | `llama3` | Default LLM model |

### Agent Command-Line Arguments

**Player Agent:**
```bash
uv run python -m agents.player [OPTIONS]
```

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `--port` | `int` | HTTP server port | Yes |
| `--strategy` | `str` | Strategy name (random, history, llm) | Yes |
| `--agent-id` | `str` | Unique agent identifier (format: player:ID) | Yes |

**Example:**
```bash
uv run python -m agents.player --port 8101 --strategy random --agent-id player:P01
```

**League Manager and Referee:**

No command-line arguments required. Configuration is loaded from `consts/__init__.py`.

## Development

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (>=0.4.0)
- All ports 8000-8001 and 8101-8104 available

### Project Structure

```
distributed-ai-agent-league/
├── agents/
│   ├── __init__.py
│   ├── league_manager.py   # Central coordinator
│   ├── referee.py           # Match orchestrator
│   └── player.py            # Player agent with strategies
├── shared/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic models and MCP protocol
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── logger.py            # Structured logging utilities
│   └── ollama_strategy.py   # LLM strategy implementation
├── consts/
│   └── __init__.py          # Configuration constants
├── tests/
│   └── test_*.py            # Test files
├── test_integration.py      # Integration tests
├── start_league.sh          # Orchestration script
├── whitelist.py             # License validation
├── pyproject.toml           # Project configuration
└── README.md
```

### Installation for Development

Follow the [Development Mode](#development-mode) installation instructions above.

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run integration tests
uv run pytest test_integration.py -v

# Run specific test file
uv run pytest tests/test_schemas.py -v

# Run with coverage
uv run pytest tests/ --cov=agents --cov=shared --cov-report=html
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --all-groups
```

### Code Quality

```bash
# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=agents --cov=shared --cov-report=html

# Format code with ruff
uv run ruff format agents/ shared/

# Lint code
uv run ruff check agents/ shared/

# Type checking with mypy
uv run mypy agents/ shared/

# Security scanning with bandit
uv run bandit -r agents/ shared/

# Check for unused code with vulture
uv run vulture agents/ shared/

# Check dependency issues
uv run deptry .
```

## Documentation

### How It Works

The system implements a distributed tournament using a three-tier architecture:

1. **Agent Registration** - Players register with League Manager, providing their contact endpoint and supported strategies
2. **Tournament Initialization** - League Manager generates round-robin schedule and sends match assignments to Referee
3. **Match Execution** - Referee orchestrates matches by sending game invitations and move requests to players
4. **Move Decision** - Players use their strategy (random, history-based, or LLM) to choose even/odd within 2-second timeout
5. **Result Aggregation** - Referee reports match results to League Manager, which updates standings in real-time

**Game Rules (Even/Odd Parity):**
1. Referee generates a random number between 1-10
2. Both players independently choose "even" or "odd" within 2 seconds
3. Players whose choice matches the number's parity win the round
4. Match consists of 10 rounds; player with most wins takes the match
5. Timeouts or errors count as round losses

**Communication Protocol:**

All inter-agent messages use the MCP (Message Communication Protocol) envelope format:

```python
{
  "protocol_version": "league.v2",
  "message_type": "CHOOSE_PARITY",
  "message_id": "uuid-here",
  "timestamp": "2024-12-24T12:00:00Z",
  "sender": "referee:main",
  "data": {
    "match_id": "uuid-here",
    "round_number": 1,
    "timeout_ms": 2000
  }
}
```

### Usage Examples

**Example 1: Start full system and monitor**
```bash
# Start all agents
./start_league.sh

# In another terminal, monitor tournament
curl http://localhost:8000/tournament/standings | jq '.'

# Example output:
# {
#   "standings": [
#     {
#       "agent_id": "player:P01",
#       "wins": 5,
#       "losses": 1,
#       "rank": 1
#     },
#     ...
#   ],
#   "matches_completed": 6,
#   "matches_total": 12
# }
```

**Example 2: Start agents individually**
```bash
# Terminal 1: Start League Manager
uv run python -m agents.league_manager

# Terminal 2: Start Referee
uv run python -m agents.referee

# Terminal 3-6: Start Players with different strategies
uv run python -m agents.player --port 8101 --strategy random --agent-id player:P01
uv run python -m agents.player --port 8102 --strategy history --agent-id player:P02
uv run python -m agents.player --port 8103 --strategy llm --agent-id player:P03
uv run python -m agents.player --port 8104 --strategy random --agent-id player:P04
```

**Example 3: Test individual components**
```bash
# Test schema validation
uv run python test_integration.py

# Test League Manager health
curl http://localhost:8000/health

# Send manual registration (for testing)
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_version": "league.v2",
    "message_type": "LEAGUE_REGISTER_REQUEST",
    "message_id": "'$(uuidgen)'",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
    "sender": "player:TEST",
    "data": {
      "display_name": "Test Player",
      "contact_endpoint": "http://localhost:9999",
      "strategies": ["random"]
    }
  }'
```

### Player Strategies

**Random Strategy:**
- Randomly selects "even" or "odd" each round
- No memory of previous rounds
- Fast decision time (~1ms)
- Baseline strategy for comparison

**History Strategy:**
- Analyzes opponent's previous choices
- Predicts opponent's next move based on patterns
- Adapts to opponent behavior over multiple rounds
- More effective against predictable opponents

**LLM Strategy:**
- Uses Ollama-hosted language model (e.g., llama3)
- Receives game context and opponent history as prompt
- Generates strategic decision with reasoning
- Requires Ollama service running locally
- Decision time ~500ms-1500ms (within 2s timeout)

### Message Types

The system uses several message types for agent communication:

| Message Type | Direction | Purpose |
|--------------|-----------|---------|
| `LEAGUE_REGISTER_REQUEST` | Player → Manager | Register agent with league |
| `LEAGUE_REGISTER_RESPONSE` | Manager → Player | Acknowledge registration |
| `GAME_INVITATION` | Referee → Player | Invite player to match |
| `GAME_JOIN_ACK` | Player → Referee | Accept invitation |
| `CHOOSE_PARITY` | Referee → Player | Request move (even/odd) |
| `CHOOSE_PARITY_RESPONSE` | Player → Referee | Submit move choice |
| `GAME_OVER` | Referee → Player | Notify match completion |
| `MATCH_RESULT_REPORT` | Referee → Manager | Report match outcome |
| `STANDINGS_UPDATE` | Manager → All | Broadcast updated standings |

### Requirements

- **Python 3.13 or higher** - Required for modern async features and type hints
- **Network connectivity** - Agents communicate via HTTP on localhost
- **Available ports** - System requires ports 8000-8001 and 8101-8104 to be free
- **Memory** - Minimum 512MB RAM, recommended 1GB+ for LLM strategy
- **CPU** - Any modern CPU sufficient; LLM strategy benefits from faster processors
- **Ollama** (optional) - Required only for LLM strategy; must be running locally

### Validation Commands

Test your installation with these commands:

```bash
# Verify Python version
python --version  # Should be 3.13+

# Verify uv installation
uv --version

# Verify dependencies installed correctly
uv run python -c "import fastapi, pydantic, httpx; print('✅ All core imports OK')"

# Run integration tests
uv run pytest test_integration.py -v

# Start system and check health
./start_league.sh

# In another terminal:
curl http://localhost:8000/health  # Should return {"status": "healthy"}
curl http://localhost:8001/health  # Should return {"status": "healthy"}
curl http://localhost:8101/health  # Should return {"status": "healthy"}

# Check tournament status
curl http://localhost:8000/tournament/status

# View standings
curl http://localhost:8000/tournament/standings
```

### Troubleshooting

**Problem: "Port already in use" error**
```bash
# Find and kill processes using required ports
lsof -i :8000
lsof -i :8001
lsof -i :8101-8104

# Kill specific process
kill -9 <PID>

# Or use the cleanup in start_league.sh (Ctrl+C gracefully stops all agents)
```

**Problem: "Connection refused" between agents**
```bash
# Verify all agents are running
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8101/health

# Check startup logs for errors
./start_league.sh  # Look for error messages in red

# Verify no firewall blocking localhost communication
```

**Problem: LLM strategy timing out**
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Check model is downloaded
ollama list

# Pull model if missing
ollama pull llama3

# Test Ollama directly
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Hello",
  "stream": false
}'
```

**Problem: "Module not found" errors**
```bash
# Reinstall all dependencies
uv sync --all-groups

# Verify virtual environment is activated
uv run python -c "import sys; print(sys.prefix)"

# Check if packages are installed
uv pip list
```

**Problem: Tests failing**
```bash
# Run with verbose output to see details
uv run pytest test_integration.py -v -s

# Run specific test function
uv run pytest test_integration.py::test_imports -v

# Clear pytest cache
rm -rf .pytest_cache __pycache__
uv run pytest test_integration.py -v
```

### Future Enhancements

- **WebSocket Support** - Real-time bidirectional communication for faster match execution and live tournament updates
- **Web Dashboard** - React-based UI for visualizing tournament progress, agent status, and match history
- **Persistent Storage** - PostgreSQL/SQLite integration for tournament history and player statistics across sessions
- **More Strategies** - Minimax, Monte Carlo Tree Search, reinforcement learning agents
- **Authentication** - JWT-based authentication for secure agent registration and communication
- **Docker Compose** - Containerized deployment with orchestration for easy multi-host scaling
- **Metrics & Observability** - Prometheus metrics and Grafana dashboards for system monitoring
- **Tournament Formats** - Swiss system, double elimination, group stages with playoffs
- **Multi-Game Support** - Extensible framework for rock-paper-scissors, tic-tac-toe, and other games

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Review the integration tests in `test_integration.py` for usage examples
- Check agent logs for detailed error messages and context
