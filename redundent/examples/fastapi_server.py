import uvicorn
import logging
import uuid
import time
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# --- Configuration & Constants ---
# In a real app, these would come from consts/ or env vars
SERVICE_NAME = "example-agent-server"
API_VERSION = "v1"
HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = logging.INFO

# --- Logging Setup (JSON Lines Pattern) ---
# This mimics the singleton logger requirement
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        return str(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(LOG_LEVEL)
logger.addHandler(handler)

# --- Data Models (Request/Response) ---

class MessageType(str, Enum):
    HEARTBEAT = "HEARTBEAT"
    GAME_INVITATION = "GAME_INVITATION"
    CHOOSE_PARITY = "CHOOSE_PARITY"
    GAME_OVER = "GAME_OVER"
    ERROR = "ERROR"

class BaseResponse(BaseModel):
    """Standard API Response Wrapper"""
    status: str = "success"
    request_id: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    model_config = ConfigDict(populate_by_name=True)

class MCPPayload(BaseModel):
    """Incoming JSON-RPC style payload"""
    protocol: str = Field(..., pattern=r"^league\.v\d+$")
    message_type: MessageType
    sender: str
    conversation_id: str
    timestamp: str
    data: Dict[str, Any] = Field(default_factory=dict)

# --- Service Layer (Business Logic) ---

class AgentService:
    """
    Encapsulates the core business logic, decoupled from HTTP details.
    This makes unit testing easier (you can test AgentService without FastAPI).
    """
    def __init__(self):
        self._state = "IDLE"
        self._active_matches: List[str] = []

    async def handle_heartbeat(self, sender: str, timestamp: str) -> Dict[str, Any]:
        logger.info(f"Processing heartbeat from {sender} at {timestamp}")
        # Simulate some async work
        await asyncio.sleep(0.01)
        return {"status": "acknowledged", "agent_state": self._state}

    async def handle_invitation(self, match_id: str, opponent: str) -> Dict[str, Any]:
        logger.info(f"Evaluating invitation for match {match_id} vs {opponent}")
        if self._state == "BUSY":
            logger.warning(f"Rejecting match {match_id}: Agent is busy")
            return {"accept": False, "reason": "busy"}
        
        self._state = "BUSY"
        self._active_matches.append(match_id)
        return {"accept": True}

    async def cleanup_match(self, match_id: str):
        """Called via background task to cleanup state after game over"""
        logger.info(f"Cleaning up match resources for {match_id}...")
        await asyncio.sleep(1.0) # Simulate teardown
        if match_id in self._active_matches:
            self._active_matches.remove(match_id)
        if not self._active_matches:
            self._state = "IDLE"
        logger.info(f"Match {match_id} cleanup complete.")

agent_service = AgentService()

# --- Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Server starting up...")
    logger.info("Initializing Agent Service connections...")
    # e.g., await agent_service.connect_db()
    yield
    # Shutdown logic
    logger.info("Server shutting down...")
    # e.g., await agent_service.disconnect_db()

# --- FastAPI App Setup ---

app = FastAPI(
    title="Standard Agent Server",
    description="Reference implementation for League Agents",
    version=API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Middleware ---

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Adds Request ID and Request Timing to every request.
    Handles global exception safety.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Inject request_id into logger context (using a context var in real production)
    # For simplicity, we just attach it to the request state
    request.state.request_id = request_id
    
    logger.info(f"Started request {request.method} {request.url.path} ID={request_id}")

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        logger.info(f"Completed request ID={request_id} in {process_time:.2f}ms Status={response.status_code}")
        return response
        
    except Exception as exc:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"Request ID={request_id} failed: {str(exc)}", exc_info=True)
        raise exc # Re-raise to let the Exception Handler catch it

# --- Exception Handlers ---

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=BaseResponse(
            status="error",
            request_id=getattr(request.state, "request_id", "unknown"),
            error="Validation Failed",
            data={"details": exc.errors()}
        ).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=BaseResponse(
            status="error",
            request_id=getattr(request.state, "request_id", "unknown"),
            error="Internal Server Error"
        ).model_dump()
    )

# --- Routes ---

@app.get("/health")
async def health_check(request: Request):
    """Standard health probe for orchestration tools"""
    return BaseResponse(
        status="ok",
        request_id=request.state.request_id,
        data={"uptime": "ok", "service": SERVICE_NAME}
    )

@app.post("/mcp", response_model=BaseResponse)
async def handle_mcp_message(
    payload: MCPPayload, 
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Main entry point for the Agent Protocol.
    Dispatches to the AgentService based on message_type.
    """
    req_id = request.state.request_id
    logger.info(f"Received MCP Message: {payload.message_type} from {payload.sender}")

    response_data = {}

    if payload.message_type == MessageType.HEARTBEAT:
        response_data = await agent_service.handle_heartbeat(
            payload.sender, payload.timestamp
        )

    elif payload.message_type == MessageType.GAME_INVITATION:
        match_id = payload.data.get("match_id")
        opponent = payload.data.get("opponent_id")
        
        if not match_id or not opponent:
            raise HTTPException(status_code=400, detail="Missing match_id or opponent_id")
            
        response_data = await agent_service.handle_invitation(match_id, opponent)

    elif payload.message_type == MessageType.GAME_OVER:
        # Example of using BackgroundTasks for non-blocking cleanup
        match_id = payload.data.get("match_id")
        if match_id:
            background_tasks.add_task(agent_service.cleanup_match, match_id)
        response_data = {"status": "acknowledged"}

    else:
        logger.warning(f"Unhandled message type: {payload.message_type}")
        response_data = {"status": "ignored", "reason": "unsupported_type"}

    return BaseResponse(
        status="success",
        request_id=req_id,
        data=response_data
    )

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host=HOST, port=PORT, reload=True)