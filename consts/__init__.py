"""
Configuration constants for the Distributed AI Agent League System.
Single source of truth for all configuration values.
"""

# Network Configuration
LEAGUE_MANAGER_PORT = 8000
REFEREE_PORT = 8001
PLAYER_PORTS = [8101, 8102, 8103, 8104]

# Service Endpoints
LEAGUE_MANAGER_URL = f"http://localhost:{LEAGUE_MANAGER_PORT}"
REFEREE_URL = f"http://localhost:{REFEREE_PORT}"

# External Services
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

# Timing Configuration (Critical for game timing)
DEFAULT_TIMEOUT = 2.0  # seconds
MOVE_TIMEOUT = 2000    # milliseconds
MATCH_TIMEOUT = 30.0   # seconds
REGISTRATION_TIMEOUT = 5.0  # seconds

# Retry & Circuit Breaker Configuration
MAX_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_TIME = 30.0  # seconds
EXPONENTIAL_BACKOFF_BASE = 2.0
JITTER_RANGE = 0.5

# HTTP Client Configuration
HTTP_CONNECT_TIMEOUT = 1.5  # seconds
HTTP_READ_TIMEOUT = 2.0     # seconds
HTTP_TOTAL_TIMEOUT = 3.0    # seconds
MAX_CONNECTIONS = 20
MAX_KEEPALIVE = 10

# Protocol Configuration
PROTOCOL_VERSION = "league.v2"
USER_AGENT_PREFIX = "MCP-Agent-Client"

# Game Configuration
GAME_TYPE = "even_odd"
MAX_ROUNDS_PER_MATCH = 10
RANDOM_NUMBER_MIN = 1
RANDOM_NUMBER_MAX = 10

# Logging Configuration
LOG_LEVEL = "INFO"
JSON_LOG_FORMAT = True
LOG_REQUEST_IDS = True

# Agent Registration
REGISTRATION_ENDPOINT = "/register"
HEALTH_ENDPOINT = "/health"
MCP_ENDPOINT = "/mcp"

# Strategy Configuration
AVAILABLE_STRATEGIES = ["random", "history", "llm"]
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 100
LLM_TIMEOUT = 3.0  # seconds

# Match Orchestration
MAX_CONCURRENT_MATCHES = 4
ROUND_ROBIN_SCHEDULING = True
MATCH_CLEANUP_DELAY = 1.0  # seconds