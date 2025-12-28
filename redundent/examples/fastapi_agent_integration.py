"""
FastAPI integration example for distributed agent HTTP communication.

This module demonstrates how to integrate the AgentHTTPClient with FastAPI
for reliable agent-to-agent communication with proper lifecycle management.

Key patterns:
- FastAPI lifespan management for HTTP client
- Background task integration
- Middleware for request monitoring
- Health check endpoints with circuit breaker status

Source: Based on HTTPX async_support_httpx.md and clients_httpx.md patterns
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time

from distributed_agent_http import (
    AgentHTTPClientConfig,
    initialize_global_client,
    close_global_client,
    get_global_client,
    agent_communication_task
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class AgentMessage(BaseModel):
    target_agent_url: str
    message_type: str
    payload: Dict[str, Any]
    priority: str = "normal"


class AgentResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class HealthStatus(BaseModel):
    status: str
    timestamp: float
    http_client: Dict[str, Any]
    circuit_breakers: Dict[str, Dict[str, Any]]


# FastAPI lifespan management based on async_support_httpx.md patterns
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager for HTTP client initialization and cleanup.

    This ensures proper connection pooling and resource management
    as recommended in clients_httpx.md.
    """
    # Startup: Initialize HTTP client with agent-specific configuration
    config = AgentHTTPClientConfig(
        connect_timeout=1.5,
        read_timeout=2.0,
        write_timeout=1.5,
        pool_timeout=0.5,
        total_timeout=3.0,
        max_connections=50,  # Higher for agent communication
        max_keepalive_connections=20,
        keepalive_expiry=30.0,  # Longer for agent persistence
        max_retries=3,
        retry_backoff_factor=0.3,
        retry_jitter=True,
        enable_http2=True
    )

    logger.info("Starting agent HTTP client...")
    await initialize_global_client(config)
    logger.info("Agent HTTP client initialized successfully")

    yield  # Application runs here

    # Shutdown: Clean up HTTP client
    logger.info("Shutting down agent HTTP client...")
    await close_global_client()
    logger.info("Agent HTTP client shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Distributed Agent Communication Service",
    description="FastAPI service with optimized HTTPX client for agent-to-agent communication",
    version="1.0.0",
    lifespan=lifespan
)


# Middleware for request monitoring
@app.middleware("http")
async def monitor_requests(request, call_next):
    """Monitor HTTP requests for performance and debugging."""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"-> {response.status_code} ({process_time:.3f}s)"
    )

    return response


# Dependency for getting HTTP client
async def get_http_client():
    """Dependency to get the global HTTP client instance."""
    return get_global_client()


# Health check endpoint with circuit breaker status
@app.get("/health", response_model=HealthStatus)
async def health_check(client=Depends(get_http_client)):
    """
    Health check endpoint that includes HTTP client status.

    Returns detailed information about the client state and circuit breakers.
    """
    # Get circuit breaker status
    circuit_breaker_status = {}
    for domain, cb in client._circuit_breakers.items():
        circuit_breaker_status[domain] = {
            "state": cb.state,
            "failure_count": cb.failure_count,
            "last_failure_time": cb.last_failure_time
        }

    return HealthStatus(
        status="healthy",
        timestamp=time.time(),
        http_client={
            "is_closed": client._is_closed,
            "base_url": str(client.base_url) if client.base_url else None,
            "config": {
                "connect_timeout": client.config.timeout.connect,
                "read_timeout": client.config.timeout.read,
                "write_timeout": client.config.timeout.write,
                "pool_timeout": client.config.timeout.pool,
                "total_timeout": client.config.total_timeout,
                "http2_enabled": client.config.enable_http2,
                "max_connections": client.config.limits.max_connections,
                "max_keepalive": client.config.limits.max_keepalive_connections
            }
        },
        circuit_breakers=circuit_breaker_status
    )


# Agent communication endpoint
@app.post("/communicate", response_model=AgentResponse)
async def communicate_with_agent(
    message: AgentMessage,
    background_tasks: BackgroundTasks,
    client=Depends(get_http_client)
):
    """
    Send message to another agent with background task processing.

    This demonstrates the recommended pattern from async_support_httpx.md
    for FastAPI integration with background tasks.
    """
    try:
        # Validate target URL
        if not message.target_agent_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail="Invalid target agent URL"
            )

        # For high-priority messages, send immediately
        if message.priority == "high":
            try:
                response = await client.post(
                    f"{message.target_agent_url}/receive",
                    json={
                        "type": message.message_type,
                        "payload": message.payload,
                        "timestamp": time.time()
                    },
                    timeout=2.0  # Stricter timeout for high priority
                )

                return AgentResponse(
                    status="success",
                    message="High-priority message sent successfully",
                    data={"response": response.json(), "http_version": response.http_version}
                )

            except Exception as e:
                logger.error(f"High-priority message failed: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to communicate with agent: {str(e)}"
                )

        # For normal priority, use background task
        else:
            background_tasks.add_task(
                agent_communication_task,
                f"{message.target_agent_url}/receive",
                {
                    "type": message.message_type,
                    "payload": message.payload,
                    "timestamp": time.time()
                }
            )

            return AgentResponse(
                status="accepted",
                message="Message queued for delivery"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Communication request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during agent communication"
        )


# Endpoint to receive messages from other agents
@app.post("/receive", response_model=AgentResponse)
async def receive_agent_message(message: Dict[str, Any]):
    """
    Receive message from another agent.

    This endpoint would typically process the incoming message
    and potentially trigger further agent communications.
    """
    logger.info(f"Received agent message: {message.get('type', 'unknown')}")

    # Process the message (implement your logic here)
    await process_incoming_message(message)

    return AgentResponse(
        status="success",
        message="Message received and processed"
    )


# Batch communication endpoint for multiple agents
@app.post("/broadcast", response_model=AgentResponse)
async def broadcast_to_agents(
    targets: list[str],
    message_type: str,
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    client=Depends(get_http_client)
):
    """
    Broadcast message to multiple agents efficiently.

    Uses concurrent requests to minimize total latency while respecting
    individual agent timeout requirements.
    """
    try:
        # Validate all target URLs
        for target in targets:
            if not target.startswith(('http://', 'https://')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid target URL: {target}"
                )

        # Create message payload
        broadcast_message = {
            "type": message_type,
            "payload": payload,
            "timestamp": time.time(),
            "broadcast": True
        }

        # Use background task for batch processing
        background_tasks.add_task(
            broadcast_task,
            targets,
            broadcast_message
        )

        return AgentResponse(
            status="accepted",
            message=f"Broadcast queued for {len(targets)} agents"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Broadcast request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during broadcast"
        )


# Circuit breaker status endpoint
@app.get("/circuit-breakers")
async def get_circuit_breaker_status(client=Depends(get_http_client)):
    """Get current circuit breaker status for monitoring."""
    status = {}
    for domain, cb in client._circuit_breakers.items():
        status[domain] = {
            "state": cb.state,
            "failure_count": cb.failure_count,
            "failure_threshold": cb.failure_threshold,
            "last_failure_time": cb.last_failure_time,
            "recovery_timeout": cb.recovery_timeout
        }

    return {"circuit_breakers": status}


# Reset circuit breaker endpoint (for admin/debugging)
@app.post("/circuit-breakers/{domain}/reset")
async def reset_circuit_breaker(domain: str, client=Depends(get_http_client)):
    """Reset circuit breaker for a specific domain."""
    if domain in client._circuit_breakers:
        cb = client._circuit_breakers[domain]
        cb.failure_count = 0
        cb.state = "CLOSED"
        cb.last_failure_time = None

        return {"message": f"Circuit breaker for {domain} reset"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit breaker for domain {domain} not found"
        )


# Helper functions
async def process_incoming_message(message: Dict[str, Any]):
    """
    Process incoming agent message.

    Implement your specific message processing logic here.
    """
    message_type = message.get("type", "unknown")
    payload = message.get("payload", {})

    logger.info(f"Processing message type: {message_type}")

    # Add your message processing logic here
    # This could trigger additional agent communications
    # using the global HTTP client

    await asyncio.sleep(0.1)  # Simulate processing time


async def broadcast_task(targets: list[str], message: Dict[str, Any]):
    """
    Background task for broadcasting to multiple agents concurrently.

    This uses asyncio.gather for concurrent requests while maintaining
    individual timeout controls.
    """
    client = get_global_client()

    async def send_to_agent(target: str) -> Dict[str, Any]:
        """Send message to individual agent."""
        try:
            response = await client.post(
                f"{target}/receive",
                json=message,
                timeout=3.0
            )
            return {
                "target": target,
                "status": "success",
                "status_code": response.status_code,
                "http_version": response.http_version
            }
        except Exception as e:
            logger.error(f"Broadcast to {target} failed: {e}")
            return {
                "target": target,
                "status": "failed",
                "error": str(e)
            }

    # Execute all requests concurrently
    try:
        results = await asyncio.gather(
            *[send_to_agent(target) for target in targets],
            return_exceptions=True
        )

        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")

        logger.info(f"Broadcast completed: {success_count}/{len(targets)} successful")

    except Exception as e:
        logger.error(f"Broadcast task failed: {e}")


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run(
        "fastapi_agent_integration:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )