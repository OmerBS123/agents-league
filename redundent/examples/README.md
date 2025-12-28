# HTTPX Best Practices for Distributed Agent Communication

This repository provides comprehensive patterns and implementations for reliable HTTP communication between distributed agents using HTTPX, with strict timeout requirements (2-3 seconds) and paramount reliability.

## 🚀 Key Features

- **AsyncClient Lifecycle Management**: Proper resource management with connection pooling
- **Granular Timeout Configuration**: Fine-tuned connect, read, write, and pool timeouts
- **Comprehensive Error Handling**: Complete exception hierarchy handling with retry strategies
- **Resilience Patterns**: Circuit breaker, exponential backoff with jitter
- **Performance Optimization**: HTTP/2 support, connection reuse, concurrent requests
- **FastAPI Integration**: Background tasks and lifespan management patterns

## 📁 Project Structure

```
examples/
├── distributed_agent_http.py      # Core HTTP client implementation
├── fastapi_agent_integration.py   # FastAPI service integration
├── test_agent_communication.py    # Test suite and benchmarks
└── README.md                      # This file
```

## 🏗️ Architecture Overview

### 1. Client Architecture

Based on **HTTPX async_support_httpx.md** and **clients_httpx.md** patterns:

```python
# Proper AsyncClient lifecycle with context management
async with get_agent_http_client(config) as client:
    response = await client.get("/agent-endpoint")
```

**Key Benefits:**
- Connection pooling and reuse (reduces latency, CPU usage, network congestion)
- Automatic resource cleanup preventing connection leaks
- HTTP/2 multiplexing for concurrent requests over single TCP connection

### 2. Timeout Configuration

Based on **HTTPX timeouts_httpx.md** hierarchy:

```python
# Granular timeout control for agent communication
config = AgentHTTPClientConfig(
    connect_timeout=1.5,    # Time to establish connection
    read_timeout=2.0,       # Time to read response data
    write_timeout=1.5,      # Time to send request data
    pool_timeout=0.5,       # Time to acquire connection from pool
    total_timeout=3.0       # Overall request timeout
)
```

**Timeout Types:**
- **Connect Timeout**: Prevents hanging on unreachable agents
- **Read Timeout**: Handles slow-responding agents
- **Write Timeout**: Manages large request payloads
- **Pool Timeout**: Controls connection pool contention

### 3. Error Handling

Based on **HTTPX exceptions_httpx.md** hierarchy:

```python
try:
    response = await client.request_with_resilience("POST", url, json=data)
except ConnectTimeout:
    # Handle connection timeout
except ReadTimeout:
    # Handle response timeout
except NetworkError:
    # Handle network issues
except RequestError:
    # Handle other request failures
```

**Exception Hierarchy:**
```
HTTPError
├── RequestError
│   ├── TransportError
│   │   ├── TimeoutException (ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout)
│   │   ├── NetworkError (ConnectError, ReadError, WriteError)
│   │   └── ProtocolError
│   └── DecodingError
└── HTTPStatusError
```

### 4. Resilience Patterns

**Circuit Breaker Pattern:**
- Prevents cascading failures in agent networks
- Automatic recovery with configurable thresholds
- Per-domain isolation for multi-agent communication

**Exponential Backoff with Jitter:**
```python
# Prevents thundering herd in agent swarms
delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
```

## 🎯 Configuration Scenarios

### High-Frequency Local Communication
```python
config = AgentHTTPClientConfig(
    connect_timeout=0.5,
    read_timeout=1.0,
    total_timeout=2.0,
    max_connections=50,
    max_keepalive_connections=30,
    enable_http2=True
)
```

### Cross-Region Agent Communication
```python
config = AgentHTTPClientConfig(
    connect_timeout=2.0,
    read_timeout=3.0,
    total_timeout=5.0,
    max_retries=4,
    retry_backoff_factor=0.5,
    retry_jitter=True
)
```

### Critical Real-Time Systems
```python
config = AgentHTTPClientConfig(
    connect_timeout=0.3,
    read_timeout=0.7,
    total_timeout=1.5,
    max_retries=1,  # Fail fast
    max_connections=100
)
```

## 🔧 FastAPI Integration

### Lifespan Management

Based on **HTTPX async_support_httpx.md** recommendations:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize global client for connection reuse
    await initialize_global_client(config)
    yield
    # Shutdown: Clean up resources
    await close_global_client()

app = FastAPI(lifespan=lifespan)
```

### Background Tasks

```python
@app.post("/communicate")
async def communicate_with_agent(message: AgentMessage, background_tasks: BackgroundTasks):
    # High-priority: immediate response
    if message.priority == "high":
        response = await get_global_client().post(message.target_url, json=message.payload)
        return {"status": "success", "response": response.json()}

    # Normal priority: background processing
    background_tasks.add_task(agent_communication_task, message.target_url, message.payload)
    return {"status": "accepted"}
```

## 📊 Performance Optimization

### HTTP/2 Benefits

Based on **HTTPX http_2_support_httpx.md**:

- **Multiplexing**: Multiple concurrent requests over single TCP connection
- **Binary Protocol**: More efficient than HTTP/1.1 text format
- **Header Compression**: Reduces bandwidth usage
- **Stream Prioritization**: Better resource allocation

```python
# Enable HTTP/2 for agent communication
client = AsyncClient(http2=True)
```

### Connection Pooling

```python
limits = Limits(
    max_connections=50,           # Total connection pool size
    max_keepalive_connections=25, # Persistent connections
    keepalive_expiry=30.0        # Connection lifetime
)
```

### Memory Management

- Streaming responses for large payloads
- Automatic connection cleanup
- Resource monitoring and alerting

## 🧪 Testing and Validation

### Performance Benchmarks

```bash
python test_agent_communication.py
```

**Metrics Collected:**
- Request latency (mean, P95, P99)
- Success/timeout/error rates
- HTTP/2 usage statistics
- Connection pool efficiency

### Error Scenarios

- Connect timeout testing (non-routable IPs)
- Read timeout testing (slow endpoints)
- Circuit breaker validation
- Retry behavior verification

## 🚀 Quick Start

### 1. Basic Agent Client

```python
from distributed_agent_http import AgentHTTPClientConfig, get_agent_http_client

# Configure for your requirements
config = AgentHTTPClientConfig(
    connect_timeout=1.0,
    read_timeout=2.0,
    total_timeout=3.0
)

# Use with context manager
async with get_agent_http_client(config) as client:
    response = await client.post("/agent/endpoint", json={"message": "hello"})
    print(f"Agent responded: {response.status_code} ({response.http_version})")
```

### 2. FastAPI Service

```python
from fastapi import FastAPI
from fastapi_agent_integration import app

# Run the service
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Run Tests

```bash
# Install dependencies
pip install httpx fastapi uvicorn pytest pytest-asyncio

# Run comprehensive tests
python test_agent_communication.py

# Run pytest suite
pytest test_agent_communication.py -v
```

## 📝 Best Practices Summary

### ✅ Do's

1. **Use AsyncClient with context managers** for proper resource management
2. **Configure granular timeouts** based on your SLA requirements
3. **Enable HTTP/2** for better performance in concurrent scenarios
4. **Implement circuit breakers** for fault tolerance
5. **Use global client instances** in FastAPI to maximize connection reuse
6. **Add jitter to retries** to prevent thundering herd
7. **Monitor performance metrics** (latency, success rate, HTTP/2 usage)

### ❌ Don'ts

1. **Don't instantiate multiple clients** in hot loops (breaks connection pooling)
2. **Don't use infinite timeouts** in agent communication
3. **Don't ignore timeout hierarchy** - configure all timeout types
4. **Don't retry indefinitely** - implement circuit breakers
5. **Don't forget to close clients** - use context managers or explicit cleanup

## 📚 Sources and References

This implementation is based on official HTTPX documentation:

- **async_support_httpx.md**: AsyncClient patterns and lifecycle management
- **timeouts_httpx.md**: Granular timeout configuration and hierarchy
- **clients_httpx.md**: Connection pooling and performance optimization
- **exceptions_httpx.md**: Comprehensive error handling patterns
- **http_2_support_httpx.md**: HTTP/2 multiplexing and performance benefits
- **transports_httpx.md**: Custom transport configuration and retry handling

## 🔍 Monitoring and Observability

### Health Check Endpoint

```
GET /health
```

Returns client status, connection pool metrics, and circuit breaker states.

### Circuit Breaker Status

```
GET /circuit-breakers
```

Monitor circuit breaker states across different agent domains.

### Performance Metrics

Track key metrics for agent communication:
- Request latency percentiles
- Success/failure rates
- HTTP/2 adoption
- Connection pool utilization
- Circuit breaker trips

## 🤝 Contributing

This implementation follows HTTPX best practices and is designed for production distributed agent systems. Contributions welcome for additional patterns and optimizations.