#!/bin/bash

# Distributed AI Agent League System - Startup Script
# Orchestrates all agents in the correct order with proper cleanup

set -e  # Exit on any error

# Configuration
LEAGUE_MANAGER_PORT=8000
REFEREE_PORT=8001
PLAYER_PORTS=(8101 8102 8103 8104)
PLAYER_STRATEGIES=("random" "history" "llm" "random")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID tracking
declare -a PIDS=()
CLEANUP_DONE=false

# Cleanup function
cleanup() {
    if [[ "$CLEANUP_DONE" == "true" ]]; then
        return
    fi

    echo -e "\n${YELLOW}🧹 Shutting down all agents...${NC}"
    CLEANUP_DONE=true

    # Kill all background processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}   Stopping process $pid${NC}"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Wait a moment for graceful shutdown
    sleep 2

    # Force kill if needed
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}   Force stopping process $pid${NC}"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}✅ All agents stopped${NC}"
}

# Set up signal handlers
trap cleanup EXIT
trap cleanup INT
trap cleanup TERM

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -i :$port >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    echo -e "${BLUE}🔄 Waiting for $name to be ready...${NC}"

    while [[ $attempt -le $max_attempts ]]; do
        if curl -s "$url/health" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $name is ready${NC}"
            return 0
        fi

        if [[ $((attempt % 5)) -eq 0 ]]; then
            echo -e "${YELLOW}   Still waiting for $name (attempt $attempt/$max_attempts)${NC}"
        fi

        sleep 1
        ((attempt++))
    done

    echo -e "${RED}❌ $name failed to start within $max_attempts seconds${NC}"
    return 1
}

# Function to start a service
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local health_url=$4

    echo -e "${BLUE}🚀 Starting $name on port $port...${NC}"

    # Check if port is available
    if ! check_port $port; then
        return 1
    fi

    # Start the service in background
    eval "$command" &
    local pid=$!
    PIDS+=($pid)

    echo -e "${GREEN}   Started $name (PID: $pid)${NC}"

    # Wait for service to be ready
    if [[ -n "$health_url" ]]; then
        if ! wait_for_service "$name" "$health_url"; then
            return 1
        fi
    fi

    return 0
}

# Main execution
main() {
    echo -e "${BLUE}🎮 Distributed AI Agent League System${NC}"
    echo -e "${BLUE}=====================================${NC}"

    # Check uv availability
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv package manager not found. Please install uv first.${NC}"
        exit 1
    fi

    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "agents" ]]; then
        echo -e "${RED}❌ Please run this script from the project root directory${NC}"
        exit 1
    fi

    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    uv sync --all-groups || {
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    }

    echo -e "\n${BLUE}🏗️  Starting distributed agent system...${NC}"

    # 1. Start League Manager (central coordinator)
    if ! start_service \
        "League Manager" \
        "uv run python -m agents.league_manager" \
        $LEAGUE_MANAGER_PORT \
        "http://localhost:$LEAGUE_MANAGER_PORT"; then
        exit 1
    fi

    echo ""

    # 2. Start Referee (match orchestrator)
    if ! start_service \
        "Referee" \
        "uv run python -m agents.referee" \
        $REFEREE_PORT \
        "http://localhost:$REFEREE_PORT"; then
        exit 1
    fi

    echo ""

    # 3. Start Player agents with different strategies
    for i in "${!PLAYER_PORTS[@]}"; do
        local port=${PLAYER_PORTS[$i]}
        local strategy=${PLAYER_STRATEGIES[$i]}
        local agent_id="P$(printf "%02d" $((i+1)))"

        if ! start_service \
            "Player $agent_id ($strategy)" \
            "uv run python -m agents.player --port $port --strategy $strategy --agent-id player:$agent_id" \
            $port \
            "http://localhost:$port"; then
            exit 1
        fi

        # Small delay between player startups to avoid registration conflicts
        sleep 2
    done

    echo -e "\n${GREEN}🎉 All agents started successfully!${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${YELLOW}System Status:${NC}"
    echo -e "  • League Manager: http://localhost:$LEAGUE_MANAGER_PORT"
    echo -e "  • Referee:        http://localhost:$REFEREE_PORT"

    for i in "${!PLAYER_PORTS[@]}"; do
        local port=${PLAYER_PORTS[$i]}
        local strategy=${PLAYER_STRATEGIES[$i]}
        local agent_id="P$(printf "%02d" $((i+1)))"
        echo -e "  • Player $agent_id ($strategy): http://localhost:$port"
    done

    echo -e "\n${YELLOW}📊 Monitor the tournament:${NC}"
    echo -e "  • Tournament Status: curl http://localhost:$LEAGUE_MANAGER_PORT/tournament/status"
    echo -e "  • Live Standings:    curl http://localhost:$LEAGUE_MANAGER_PORT/tournament/standings"
    echo -e "  • Health Checks:     curl http://localhost:{port}/health"

    echo -e "\n${YELLOW}📱 Press Ctrl+C to stop all agents${NC}"

    # Wait for user interruption or process completion
    echo -e "\n${BLUE}🏃 Tournament is now running...${NC}"

    # Monitor processes and wait
    while true; do
        # Check if all processes are still running
        local running_count=0
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running_count++))
            fi
        done

        if [[ $running_count -eq 0 ]]; then
            echo -e "\n${YELLOW}🏁 All agents have stopped${NC}"
            break
        fi

        sleep 5
    done
}

# Show usage if help requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Distributed AI Agent League System Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script starts all components of the distributed agent system:"
    echo "  - League Manager (port 8000) - Central coordinator"
    echo "  - Referee (port 8001) - Match orchestrator"
    echo "  - 4 Player agents (ports 8101-8104) - Game participants"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Requirements:"
    echo "  - uv package manager installed"
    echo "  - Run from project root directory"
    echo "  - Ports 8000-8004 and 8101-8104 must be available"
    echo ""
    echo "The system will automatically:"
    echo "  1. Install all dependencies"
    echo "  2. Start services in correct order"
    echo "  3. Wait for each service to be ready"
    echo "  4. Register players with the League Manager"
    echo "  5. Begin round-robin tournament"
    echo ""
    echo "Press Ctrl+C to stop all agents cleanly."
    exit 0
fi

# Run main function
main "$@"