"""
Player Agent - Distributed AI Agent League System.
Implements the strategy pattern for different game-playing approaches.
"""

import asyncio
import argparse
import sys
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import random

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

# Internal imports
from shared.logger import get_logger, log_with_context, log_performance
from shared.exceptions import (
    LeagueError, RegistrationError, MatchError, StrategyError,
    NetworkError, CircuitBreakerError, OperationTimeoutError
)
from shared.schemas import (
    MCPEnvelope, MessageType, BaseResponse, ParityChoice, AgentStatus,
    InvitationData, InvitationResponse, MoveRequestData, MoveResponseData,
    create_registration_message, create_move_response
)
from consts import (
    LEAGUE_MANAGER_URL, REGISTRATION_ENDPOINT, MCP_ENDPOINT, HEALTH_ENDPOINT,
    DEFAULT_TIMEOUT, MAX_RETRIES, CIRCUIT_BREAKER_THRESHOLD,
    CIRCUIT_BREAKER_RESET_TIME, AVAILABLE_STRATEGIES
)

# --- Configuration ---
SERVICE_NAME = "player-agent"
logger = get_logger(__name__, SERVICE_NAME)


# --- Strategy Pattern Implementation ---

class Strategy(ABC):
    """Abstract base class for player strategies"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.match_history: Dict[str, List[ParityChoice]] = {}
        self.opponent_histories: Dict[str, List[ParityChoice]] = {}

    @abstractmethod
    async def choose_move(
        self,
        match_id: str,
        opponent_id: str,
        round_id: int,
        opponent_history: List[ParityChoice]
    ) -> tuple[ParityChoice, float, Optional[str]]:
        """
        Choose a move for the current round.

        Returns:
            Tuple of (choice, confidence, reasoning)
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this strategy"""
        pass

    def update_history(self, match_id: str, my_choice: ParityChoice, opponent_choice: Optional[ParityChoice] = None):
        """Update match history after a round"""
        if match_id not in self.match_history:
            self.match_history[match_id] = []
        self.match_history[match_id].append(my_choice)


class RandomStrategy(Strategy):
    """Completely random strategy"""

    @property
    def strategy_name(self) -> str:
        return "random"

    async def choose_move(
        self,
        match_id: str,
        opponent_id: str,
        round_id: int,
        opponent_history: List[ParityChoice]
    ) -> tuple[ParityChoice, float, Optional[str]]:
        """Choose randomly between even and odd"""
        choice = random.choice([ParityChoice.EVEN, ParityChoice.ODD])
        confidence = 0.5  # Completely random, no confidence
        reasoning = "Random selection - no pattern analysis"

        log_with_context(
            logger,
            logger.level,
            f"Random strategy chose {choice.value}",
            agent_id=self.agent_id,
            match_id=match_id
        )

        return choice, confidence, reasoning


class HistoryStrategy(Strategy):
    """Counter-strategy based on opponent history"""

    @property
    def strategy_name(self) -> str:
        return "history"

    async def choose_move(
        self,
        match_id: str,
        opponent_id: str,
        round_id: int,
        opponent_history: List[ParityChoice]
    ) -> tuple[ParityChoice, float, Optional[str]]:
        """Analyze opponent patterns and counter them"""

        if not opponent_history or len(opponent_history) < 2:
            # Not enough history, use random
            choice = random.choice([ParityChoice.EVEN, ParityChoice.ODD])
            return choice, 0.3, "Insufficient history for pattern analysis"

        # Analyze recent patterns
        recent_moves = opponent_history[-5:]  # Last 5 moves
        even_count = sum(1 for move in recent_moves if move == ParityChoice.EVEN)
        odd_count = len(recent_moves) - even_count

        # Check for alternating pattern
        is_alternating = True
        if len(recent_moves) >= 3:
            for i in range(1, len(recent_moves)):
                if recent_moves[i] == recent_moves[i-1]:
                    is_alternating = False
                    break

        confidence = 0.6
        reasoning = ""

        if is_alternating and len(recent_moves) >= 3:
            # Predict next move in alternating pattern
            last_move = recent_moves[-1]
            predicted_next = ParityChoice.ODD if last_move == ParityChoice.EVEN else ParityChoice.EVEN
            # Counter the predicted move
            choice = ParityChoice.EVEN if predicted_next == ParityChoice.ODD else ParityChoice.ODD
            confidence = 0.8
            reasoning = f"Detected alternating pattern, countering predicted {predicted_next.value}"

        elif even_count > odd_count * 1.5:
            # Opponent favors EVEN
            choice = ParityChoice.ODD
            confidence = 0.7
            reasoning = f"Opponent tends toward EVEN ({even_count}/{len(recent_moves)}), countering with ODD"

        elif odd_count > even_count * 1.5:
            # Opponent favors ODD
            choice = ParityChoice.EVEN
            confidence = 0.7
            reasoning = f"Opponent tends toward ODD ({odd_count}/{len(recent_moves)}), countering with EVEN"

        else:
            # No clear pattern, slight bias toward less common recent choice
            if even_count < odd_count:
                choice = ParityChoice.EVEN
                reasoning = "No strong pattern, slight bias toward EVEN"
            else:
                choice = ParityChoice.ODD
                reasoning = "No strong pattern, slight bias toward ODD"
            confidence = 0.4

        log_with_context(
            logger,
            logger.level,
            f"History strategy chose {choice.value} (confidence: {confidence:.2f})",
            agent_id=self.agent_id,
            match_id=match_id
        )

        return choice, confidence, reasoning


class LLMStrategy(Strategy):
    """LLM-powered strategy using Ollama integration"""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        # Import here to avoid circular imports
        from shared.ollama_strategy import OllamaClient, GameContext

        self.ollama_client = OllamaClient()
        self.GameContext = GameContext

    @property
    def strategy_name(self) -> str:
        return "llm"

    async def choose_move(
        self,
        match_id: str,
        opponent_id: str,
        round_id: int,
        opponent_history: List[ParityChoice]
    ) -> tuple[ParityChoice, float, Optional[str]]:
        """Use Ollama LLM for strategic decision making"""
        try:
            # Build current game context
            my_history = self.match_history.get(match_id, [])

            # Estimate current score based on history (simplified)
            current_score = {self.agent_id: 0, opponent_id: 0}
            if len(my_history) > 0 and len(opponent_history) > 0:
                # This is a simplified score calculation - in real game it would come from match state
                my_wins = len(my_history) // 2  # Rough estimate
                opp_wins = len(opponent_history) // 2
                current_score = {self.agent_id: my_wins, opponent_id: opp_wins}

            context = self.GameContext(
                agent_id=self.agent_id,
                opponent_id=opponent_id,
                match_id=match_id,
                round_number=round_id,
                opponent_history=opponent_history,
                my_history=my_history,
                current_score=current_score
            )

            # Get LLM decision
            choice, confidence, reasoning = await self.ollama_client.get_strategic_move(context)

            log_with_context(
                logger,
                logger.level,
                f"LLM strategy chose {choice.value} (confidence: {confidence:.2f})",
                agent_id=self.agent_id,
                match_id=match_id
            )

            return choice, confidence, reasoning

        except Exception as e:
            logger.error(f"LLM strategy failed: {str(e)}, falling back to random")

            # Fallback to random choice
            choice = random.choice([ParityChoice.EVEN, ParityChoice.ODD])
            confidence = 0.3
            reasoning = f"LLM failed ({str(e)[:50]}), random fallback"

            return choice, confidence, reasoning


# --- Agent Service Logic ---

class PlayerService:
    """
    Core business logic for the Player Agent.
    Handles registration, match management, and strategy coordination.
    """

    def __init__(self, agent_id: str, strategy: Strategy, contact_port: int):
        self.agent_id = agent_id
        self.strategy = strategy
        self.contact_port = contact_port
        self.status = AgentStatus.OFFLINE
        self.active_matches: Dict[str, Dict[str, Any]] = {}
        self.registration_complete = False

        # HTTP client for League Manager communication
        self.http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """Initialize service connections"""
        timeout_config = httpx.Timeout(DEFAULT_TIMEOUT, connect=3.0)
        self.http_client = httpx.AsyncClient(timeout=timeout_config)
        self.status = AgentStatus.IDLE

        # Attempt registration with League Manager
        await self._register_with_manager()

    async def shutdown(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
        self.status = AgentStatus.OFFLINE

    async def _register_with_manager(self):
        """Register this player with the League Manager"""
        try:
            logger.info(f"Attempting registration with League Manager")

            contact_endpoint = f"http://localhost:{self.contact_port}"
            registration_msg = create_registration_message(
                sender_id=self.agent_id,
                display_name=f"Player-{self.agent_id.split(':')[-1]}",
                contact_endpoint=contact_endpoint,
                strategies=[self.strategy.strategy_name]
            )

            if not self.http_client:
                raise RegistrationError("HTTP client not initialized")

            response = await self.http_client.post(
                f"{LEAGUE_MANAGER_URL}{REGISTRATION_ENDPOINT}",
                content=registration_msg.model_dump_json(),
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()

            self.registration_complete = True
            logger.info(f"Successfully registered with League Manager")

        except httpx.HTTPError as e:
            error_msg = f"Registration failed: {str(e)}"
            logger.error(error_msg)
            raise RegistrationError(error_msg, self.agent_id)
        except Exception as e:
            error_msg = f"Unexpected registration error: {str(e)}"
            logger.error(error_msg)
            raise RegistrationError(error_msg, self.agent_id)

    async def handle_invitation(self, envelope: MCPEnvelope) -> MCPEnvelope:
        """Handle match invitation from Referee"""
        try:
            invitation_data = InvitationData(**envelope.data)
            match_id = invitation_data.match_id

            logger.info(f"Received match invitation for {match_id}")

            # Check if we're available
            if self.status != AgentStatus.IDLE:
                response_data = InvitationResponse(
                    match_id=match_id,
                    accepted=False,
                    reason=f"Agent status is {self.status.value}"
                )
            else:
                # Accept the invitation
                self.status = AgentStatus.BUSY
                self.active_matches[match_id] = {
                    "opponent_id": invitation_data.opponent_id,
                    "role": invitation_data.role_in_match,
                    "start_time": time.time()
                }

                response_data = InvitationResponse(
                    match_id=match_id,
                    accepted=True,
                    estimated_ready_time=100  # 100ms to be ready
                )

                logger.info(f"Accepted match invitation for {match_id}")

            return envelope.create_reply(
                response_type=MessageType.INVITATION_ACK,
                data=response_data.model_dump(),
                sender_id=self.agent_id
            )

        except Exception as e:
            logger.error(f"Error handling invitation: {str(e)}")
            return envelope.create_error_reply(
                error_code="INVITATION_ERROR",
                error_message=str(e),
                sender_id=self.agent_id
            )

    async def handle_move_request(self, envelope: MCPEnvelope) -> MCPEnvelope:
        """Handle move request from Referee"""
        start_time = time.time()

        try:
            move_request = MoveRequestData(**envelope.data)
            match_id = move_request.match_id

            if match_id not in self.active_matches:
                raise MatchError(f"Unknown match: {match_id}", match_id)

            match_info = self.active_matches[match_id]
            opponent_id = match_info["opponent_id"]

            # Use strategy to make move
            choice, confidence, reasoning = await self.strategy.choose_move(
                match_id=match_id,
                opponent_id=opponent_id,
                round_id=move_request.round_id,
                opponent_history=move_request.opponent_history
            )

            decision_time_ms = (time.time() - start_time) * 1000

            # Create response
            response_msg = create_move_response(
                sender_id=self.agent_id,
                match_id=match_id,
                round_id=move_request.round_id,
                parity_choice=choice,
                strategy_used=self.strategy.strategy_name,
                conversation_id=envelope.conversation_id,
                confidence=confidence,
                reasoning=reasoning
            )

            # Update strategy history
            self.strategy.update_history(match_id, choice)

            log_performance(
                logger=logger,
                operation="move_decision",
                duration_ms=decision_time_ms,
                success=True,
                match_id=match_id,
                choice=choice.value,
                confidence=confidence
            )

            return response_msg

        except Exception as e:
            decision_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error handling move request: {str(e)}")

            log_performance(
                logger=logger,
                operation="move_decision",
                duration_ms=decision_time_ms,
                success=False,
                error=str(e)
            )

            return envelope.create_error_reply(
                error_code="MOVE_ERROR",
                error_message=str(e),
                sender_id=self.agent_id
            )

    async def handle_game_over(self, envelope: MCPEnvelope):
        """Handle game over notification"""
        try:
            match_id = envelope.data.get("match_id")
            if match_id and match_id in self.active_matches:
                del self.active_matches[match_id]

                # Reset status if no active matches
                if not self.active_matches:
                    self.status = AgentStatus.IDLE

                logger.info(f"Match {match_id} completed, status: {self.status.value}")

        except Exception as e:
            logger.error(f"Error handling game over: {str(e)}")


# --- FastAPI Application ---

# Global service instance
player_service: Optional[PlayerService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global player_service

    logger.info("Player Agent starting up...")

    try:
        if player_service:
            await player_service.initialize()
            logger.info("Player Agent initialization complete")

        yield

    finally:
        logger.info("Player Agent shutting down...")
        if player_service:
            await player_service.shutdown()


app = FastAPI(
    title="Player Agent",
    description="Distributed AI Agent League - Player Component",
    version="1.0.0",
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
    """Add request context and timing"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    request.state.request_id = request_id

    log_with_context(
        logger,
        logger.level,
        f"Started {request.method} {request.url.path}",
        request_id=request_id
    )

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

        log_with_context(
            logger,
            logger.level,
            f"Completed request in {process_time:.2f}ms",
            request_id=request_id,
            duration_ms=process_time
        )

        return response

    except Exception as exc:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"Request {request_id} failed: {str(exc)}", exc_info=True)
        raise


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


@app.exception_handler(LeagueError)
async def league_exception_handler(request: Request, exc: LeagueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=BaseResponse(
            status="error",
            request_id=getattr(request.state, "request_id", "unknown"),
            error=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=BaseResponse(
            status="error",
            request_id=getattr(request.state, "request_id", "unknown"),
            error="Internal Server Error"
        ).model_dump()
    )


# --- Routes ---

@app.get(HEALTH_ENDPOINT)
async def health_check(request: Request):
    """Health check endpoint"""
    global player_service

    health_data = {
        "status": "healthy",
        "service": SERVICE_NAME,
        "agent_id": player_service.agent_id if player_service else "unknown",
        "agent_status": player_service.status.value if player_service else "unknown",
        "active_matches": len(player_service.active_matches) if player_service else 0,
        "registration_complete": player_service.registration_complete if player_service else False
    }

    return BaseResponse(
        status="success",
        request_id=request.state.request_id,
        data=health_data
    )


@app.post(MCP_ENDPOINT, response_model=BaseResponse)
async def handle_mcp_message(
    envelope: MCPEnvelope,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Main MCP message handler"""
    global player_service

    if not player_service:
        raise HTTPException(status_code=503, detail="Player service not initialized")

    req_id = request.state.request_id
    logger.info(f"Received {envelope.message_type.value} from {envelope.sender}")

    try:
        if envelope.message_type == MessageType.INVITATION:
            response_envelope = await player_service.handle_invitation(envelope)
            return BaseResponse(
                status="success",
                request_id=req_id,
                data=response_envelope.model_dump()
            )

        elif envelope.message_type == MessageType.MOVE_CALL:
            response_envelope = await player_service.handle_move_request(envelope)
            return BaseResponse(
                status="success",
                request_id=req_id,
                data=response_envelope.model_dump()
            )

        elif envelope.message_type == MessageType.GAME_OVER:
            background_tasks.add_task(player_service.handle_game_over, envelope)
            return BaseResponse(
                status="success",
                request_id=req_id,
                data={"acknowledged": True}
            )

        else:
            logger.warning(f"Unhandled message type: {envelope.message_type}")
            return BaseResponse(
                status="success",
                request_id=req_id,
                data={"status": "ignored", "reason": "unsupported_message_type"}
            )

    except Exception as e:
        logger.error(f"Error handling MCP message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Main Entry Point ---

def create_strategy(strategy_name: str, agent_id: str) -> Strategy:
    """Factory function for creating strategies"""
    if strategy_name == "random":
        return RandomStrategy(agent_id)
    elif strategy_name == "history":
        return HistoryStrategy(agent_id)
    elif strategy_name == "llm":
        return LLMStrategy(agent_id)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {AVAILABLE_STRATEGIES}")


def main():
    """Main entry point with command line argument parsing"""
    global player_service

    parser = argparse.ArgumentParser(description="Player Agent for Distributed AI League")
    parser.add_argument("--strategy", choices=AVAILABLE_STRATEGIES, default="random",
                      help="Strategy to use for gameplay")
    parser.add_argument("--port", type=int, default=8101,
                      help="Port to run the player server on")
    parser.add_argument("--agent-id", type=str,
                      help="Agent ID (auto-generated if not provided)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind to")

    args = parser.parse_args()

    # Generate agent ID if not provided
    if not args.agent_id:
        args.agent_id = f"player:P{args.port - 8100:02d}"

    try:
        # Create strategy
        strategy = create_strategy(args.strategy, args.agent_id)

        # Initialize player service
        player_service = PlayerService(
            agent_id=args.agent_id,
            strategy=strategy,
            contact_port=args.port
        )

        logger.info(f"Starting Player Agent {args.agent_id} with {args.strategy} strategy on port {args.port}")

        # Run FastAPI server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("Player Agent stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start Player Agent: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()