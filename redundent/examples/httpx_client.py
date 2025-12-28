import httpx
import asyncio
import logging
import random
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

# --- Logging Setup ---
logger = logging.getLogger("ResilientClient")
logging.basicConfig(level=logging.INFO)

# --- Configuration Constants ---
DEFAULT_TIMEOUT = 5.0
MAX_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_TIME = 30.0 # seconds

class CircuitState(Enum):
    CLOSED = "CLOSED" # Normal operation
    OPEN = "OPEN"     # Failing, reject requests
    HALF_OPEN = "HALF_OPEN" # Testing recovery

@dataclass
class RequestContext:
    url: str
    method: str
    attempt: int
    duration: float
    status_code: Optional[int]

class NetworkError(Exception):
    """Base exception for networking issues"""
    pass

class CircuitBreakerOpen(NetworkError):
    """Raised when the circuit is open"""
    pass

# --- The Client Class ---

class AsyncResilientClient:
    """
    A wrapper around httpx.AsyncClient that provides:
    1. Automatic Retries with Exponential Backoff & Jitter
    2. Circuit Breaker Pattern
    3. JSON Parsing & Error Handling
    4. Detailed Instrumentation
    """

    def __init__(self, base_url: str = "", agent_id: str = "unknown"):
        self.base_url = base_url
        self.agent_id = agent_id
        self.headers = {
            "User-Agent": f"MCP-Agent-Client/{agent_id}",
            "Content-Type": "application/json"
        }
        
        # Circuit Breaker State
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    def _update_circuit_success(self):
        """Reset circuit on success"""
        if self._circuit_state != CircuitState.CLOSED:
            logger.info("Circuit Breaker recovering -> CLOSED")
            self._circuit_state = CircuitState.CLOSED
            self._failure_count = 0

    def _update_circuit_failure(self):
        """Track failures and trip circuit if needed"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._circuit_state == CircuitState.CLOSED and self._failure_count >= CIRCUIT_BREAKER_THRESHOLD:
            logger.error(f"Circuit Breaker TRIPPED -> OPEN (Failures: {self._failure_count})")
            self._circuit_state = CircuitState.OPEN

    def _check_circuit(self):
        """Raise error if circuit is open, or check for reset timeout"""
        if self._circuit_state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed > CIRCUIT_BREAKER_RESET_TIME:
                logger.warning("Circuit Breaker probe -> HALF_OPEN")
                self._circuit_state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpen(f"Circuit is OPEN. Retry in {CIRCUIT_BREAKER_RESET_TIME - elapsed:.1f}s")

    async def _execute_request(
        self, 
        method: str, 
        endpoint: str, 
        payload: Optional[Dict] = None,
        retries: int = MAX_RETRIES
    ) -> Dict[str, Any]:
        """
        Internal execution logic with retry loop.
        """
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        
        timeout_config = httpx.Timeout(DEFAULT_TIMEOUT, connect=3.0)
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            for attempt in range(1, retries + 1):
                start_time = time.time()
                try:
                    # 1. Check Circuit
                    self._check_circuit()
                    
                    # 2. Make Request
                    logger.debug(f"[{method}] {url} (Attempt {attempt})")
                    if method == "POST":
                        response = await client.post(url, json=payload, headers=self.headers)
                    else:
                        response = await client.get(url, headers=self.headers)

                    # 3. Validate Response
                    response.raise_for_status()
                    
                    # 4. Success Logic
                    duration = (time.time() - start_time) * 1000
                    self._update_circuit_success()
                    logger.info(f"✅ Success {method} {url} in {duration:.2f}ms")
                    
                    return response.json() if response.content else {}

                except (httpx.HTTPError, httpx.TimeoutException) as e:
                    # 5. Failure Logic
                    duration = (time.time() - start_time) * 1000
                    logger.warning(f"⚠️ Failed {method} {url}: {str(e)} ({duration:.2f}ms)")
                    
                    if attempt == retries:
                        self._update_circuit_failure()
                        logger.error(f"❌ Max retries reached for {url}")
                        raise NetworkError(f"Request failed after {retries} attempts: {e}")
                    
                    # 6. Backoff Strategy (Exponential + Jitter)
                    base_delay = 2.0 ** attempt
                    jitter = random.uniform(0, 0.5)
                    sleep_time = base_delay + jitter
                    
                    logger.info(f"Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)

                except CircuitBreakerOpen as e:
                    # Propagate immediately, don't retry locally if circuit is blown
                    logger.error(f"⛔ Request blocked by Circuit Breaker: {e}")
                    raise

        return {}

    # --- Public API ---

    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request"""
        return await self._execute_request("POST", endpoint, payload=data)

    async def get(self, endpoint: str) -> Dict[str, Any]:
        """Send a GET request"""
        return await self._execute_request("GET", endpoint)

    async def register_agent(self, manager_url: str, registration_payload: Dict[str, Any]) -> bool:
        """
        Specific helper for the complex registration flow.
        """
        try:
            logger.info("Attempting to register with League Manager...")
            resp = await self.post(manager_url, registration_payload)
            logger.info(f"Registration Response: {resp}")
            return True
        except NetworkError:
            logger.critical("Failed to register. Agent cannot start.")
            return False

# --- Usage Example ---

async def main():
    # 1. Initialize Client
    client = AsyncResilientClient(base_url="http://localhost:8000", agent_id="P01")
    
    # 2. Define a payload
    payload = {
        "protocol": "league.v2",
        "message_type": "HEARTBEAT",
        "sender": "player:P01",
        "timestamp": "2025-01-01T12:00:00Z",
        "conversation_id": "test-1",
        "data": {}
    }

    # 3. Run Request
    try:
        data = await client.post("/mcp", payload)
        print(f"Result: {data}")
    except NetworkError as e:
        print(f"Operation failed: {e}")
    except CircuitBreakerOpen as e:
        print(f"System overloaded: {e}")

if __name__ == "__main__":
    asyncio.run(main())