"""
Referee Agent - Distributed AI Agent League System.
Orchestrates matches between players with parallel move collection.
"""

import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Tuple
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
    LeagueError, MatchError, NetworkError, OperationTimeoutError, ProtocolError
)
from shared.schemas import (
    MCPEnvelope, MessageType, BaseResponse, ParityChoice, MatchStatus,
    ScheduleMatchData, InvitationData, InvitationResponse,
    MoveRequestData, MoveResponseData, GameResultData, MatchResultData
)
from consts import (
    REFEREE_PORT, LEAGUE_MANAGER_URL, MCP_ENDPOINT, HEALTH_ENDPOINT,
    DEFAULT_TIMEOUT, MOVE_TIMEOUT, MATCH_TIMEOUT, MAX_RETRIES,
    RANDOM_NUMBER_MIN, RANDOM_NUMBER_MAX, MATCH_CLEANUP_DELAY
)

# --- Configuration ---
SERVICE_NAME = "referee-agent"
AGENT_ID = "referee:REF-MAIN"
logger = get_logger(__name__, SERVICE_NAME)


# --- Match State Management ---

class MatchOrchestrator:
    """
    Ephemeral match state machine for orchestrating games between players.
    Each match runs independently with its own state and lifecycle.
    """

    def __init__(self, match_id: str, player_a_id: str, player_b_id: str):
        self.match_id = match_id
        self.player_a_id = player_a_id
        self.player_b_id = player_b_id
        self.status = MatchStatus.SCHEDULED
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Match state
        self.current_round = 0
        self.max_rounds = 10  # Best of 10 rounds
        self.scores = {player_a_id: 0, player_b_id: 0}
        self.game_history: List[GameResultData] = []

        # Player endpoints (will be set when match starts)
        self.player_endpoints: Dict[str, str] = {}

        logger.info(f"Created match orchestrator for {match_id}: {player_a_id} vs {player_b_id}")

    async def invite_players(self, http_client: httpx.AsyncClient) -> bool:
        """
        Send invitations to both players and wait for acceptance.

        Returns:
            True if both players accepted, False otherwise
        """
        try:
            self.status = MatchStatus.IN_PROGRESS
            self.start_time = time.time()

            logger.info(f"Sending invitations for match {self.match_id}")

            # Create invitation messages
            invitation_a = MCPEnvelope(
                message_type=MessageType.INVITATION,
                sender=AGENT_ID,
                recipient=self.player_a_id,
                data=InvitationData(
                    match_id=self.match_id,
                    round_id=1,
                    opponent_id=self.player_b_id,
                    role_in_match="A",
                    timeout_ms=MOVE_TIMEOUT
                ).model_dump()
            )

            invitation_b = MCPEnvelope(
                message_type=MessageType.INVITATION,
                sender=AGENT_ID,
                recipient=self.player_b_id,
                data=InvitationData(
                    match_id=self.match_id,
                    round_id=1,
                    opponent_id=self.player_a_id,
                    role_in_match="B",
                    timeout_ms=MOVE_TIMEOUT
                ).model_dump()
            )

            # Send invitations in parallel
            tasks = [
                self._send_invitation(http_client, self.player_a_id, invitation_a),
                self._send_invitation(http_client, self.player_b_id, invitation_b)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check results
            for i, result in enumerate(results):
                player_id = [self.player_a_id, self.player_b_id][i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to invite {player_id}: {str(result)}")
                    return False
                elif not result:
                    logger.warning(f"Player {player_id} declined invitation")
                    return False

            logger.info(f"Both players accepted invitations for match {self.match_id}")
            return True

        except Exception as e:
            logger.error(f"Error during player invitations: {str(e)}")
            self.status = MatchStatus.ERROR
            return False

    async def _send_invitation(
        self,
        http_client: httpx.AsyncClient,
        player_id: str,
        invitation: MCPEnvelope
    ) -> bool:
        """Send invitation to a specific player"""
        try:
            # Extract port from player ID (e.g., "player:P01" -> port 8101)
            player_num = int(player_id.split("P")[-1])
            player_port = 8100 + player_num
            player_url = f"http://localhost:{player_port}{MCP_ENDPOINT}"

            response = await http_client.post(
                player_url,
                json=invitation.model_dump(),
                timeout=5.0
            )

            response.raise_for_status()
            response_data = response.json()

            # Parse response to check if invitation was accepted
            if "data" in response_data and isinstance(response_data["data"], dict):
                envelope_data = response_data["data"]
                if envelope_data.get("message_type") == MessageType.INVITATION_ACK.value:
                    ack_data = InvitationResponse(**envelope_data.get("data", {}))
                    if ack_data.accepted:
                        # Store player endpoint for future communication
                        self.player_endpoints[player_id] = player_url
                        return True

            return False

        except Exception as e:
            logger.error(f"Failed to send invitation to {player_id}: {str(e)}")
            return False

    async def execute_match(self, http_client: httpx.AsyncClient) -> MatchResultData:
        """
        Execute the complete match with multiple rounds.

        Returns:
            Complete match results
        """
        try:
            logger.info(f"Starting match execution: {self.match_id}")

            # Execute rounds until we have a winner or max rounds reached
            while (self.current_round < self.max_rounds and
                   max(self.scores.values()) <= self.max_rounds // 2):

                self.current_round += 1

                game_result = await self._execute_round(http_client, self.current_round)
                if game_result:
                    self.game_history.append(game_result)

                    # Update scores
                    if game_result.winner_id:
                        self.scores[game_result.winner_id] += 1

                # Small delay between rounds
                await asyncio.sleep(0.1)

            # Determine final winner
            winner_id = None
            if self.scores[self.player_a_id] > self.scores[self.player_b_id]:
                winner_id = self.player_a_id
            elif self.scores[self.player_b_id] > self.scores[self.player_a_id]:
                winner_id = self.player_b_id

            self.end_time = time.time()
            self.status = MatchStatus.COMPLETED

            # Create match result
            match_duration_ms = (self.end_time - self.start_time) * 1000 if self.start_time else 0

            match_result = MatchResultData(
                match_id=self.match_id,
                winner_id=winner_id,
                loser_id=self.player_b_id if winner_id == self.player_a_id else self.player_a_id,
                final_score=self.scores.copy(),
                total_rounds=self.current_round,
                match_duration_ms=match_duration_ms,
                game_results=self.game_history,
                match_stats={
                    "avg_round_duration": sum(gr.round_duration_ms for gr in self.game_history) / len(self.game_history) if self.game_history else 0,
                    "timeouts": 0,  # TODO: Track timeouts
                    "strategy_breakdown": {}  # TODO: Collect strategy stats
                }
            )

            logger.info(f"Match {self.match_id} completed: {winner_id} wins {self.scores}")
            return match_result

        except Exception as e:
            logger.error(f"Error executing match {self.match_id}: {str(e)}")
            self.status = MatchStatus.ERROR
            raise MatchError(f"Match execution failed: {str(e)}", self.match_id)

    async def _execute_round(self, http_client: httpx.AsyncClient, round_id: int) -> Optional[GameResultData]:
        """Execute a single round of the Even/Odd game"""
        round_start = time.time()

        try:
            logger.info(f"Executing round {round_id} of match {self.match_id}")

            # Collect moves from both players in parallel
            moves = await self._collect_moves_parallel(http_client, round_id)

            if not moves or len(moves) != 2:
                logger.error(f"Failed to collect moves for round {round_id}")
                return None

            player_a_choice = moves.get(self.player_a_id)
            player_b_choice = moves.get(self.player_b_id)

            if not player_a_choice or not player_b_choice:
                logger.error(f"Missing moves for round {round_id}")
                return None

            # Generate random number for game resolution
            random_number = random.randint(RANDOM_NUMBER_MIN, RANDOM_NUMBER_MAX)

            # Calculate game result using Even/Odd rules
            # Sum of choices + random number, check if even or odd
            choice_sum = (1 if player_a_choice == ParityChoice.ODD else 0) + \
                        (1 if player_b_choice == ParityChoice.ODD else 0)

            total = choice_sum + random_number
            is_even = (total % 2) == 0

            # Determine winner based on game rules
            # If total is even and player chose EVEN, they get a point
            # If total is odd and player chose ODD, they get a point
            winner_id = None
            if is_even:
                if player_a_choice == ParityChoice.EVEN:
                    winner_id = self.player_a_id
                elif player_b_choice == ParityChoice.EVEN:
                    winner_id = self.player_b_id
            else:  # is_odd
                if player_a_choice == ParityChoice.ODD:
                    winner_id = self.player_a_id
                elif player_b_choice == ParityChoice.ODD:
                    winner_id = self.player_b_id

            round_duration_ms = (time.time() - round_start) * 1000

            # Create game result
            game_result = GameResultData(
                match_id=self.match_id,
                round_id=round_id,
                player_a_choice=player_a_choice,
                player_b_choice=player_b_choice,
                random_number=random_number,
                winner_id=winner_id,
                calculation=f"({player_a_choice.value[0]}+{player_b_choice.value[0]}+{random_number}) = {total} ({'even' if is_even else 'odd'})",
                round_duration_ms=round_duration_ms
            )

            logger.info(f"Round {round_id} result: {game_result.calculation} -> {winner_id or 'tie'}")
            return game_result

        except Exception as e:
            logger.error(f"Error executing round {round_id}: {str(e)}")
            return None

    async def _collect_moves_parallel(
        self,
        http_client: httpx.AsyncClient,
        round_id: int
    ) -> Dict[str, ParityChoice]:
        """
        Collect moves from both players in parallel with timeout enforcement.

        Returns:
            Dictionary mapping player_id to their move choice
        """
        try:
            # Prepare opponent histories for context
            player_a_history = []
            player_b_history = []

            for game in self.game_history:
                player_a_history.append(game.player_a_choice)
                player_b_history.append(game.player_b_choice)

            # Create move request tasks
            tasks = [
                self._request_move(
                    http_client,
                    self.player_a_id,
                    round_id,
                    player_b_history  # A sees B's history
                ),
                self._request_move(
                    http_client,
                    self.player_b_id,
                    round_id,
                    player_a_history  # B sees A's history
                )
            ]

            # Execute in parallel with timeout
            timeout_seconds = MOVE_TIMEOUT / 1000.0
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_seconds
            )

            # Process results
            moves = {}
            for i, result in enumerate(results):
                player_id = [self.player_a_id, self.player_b_id][i]

                if isinstance(result, Exception):
                    logger.warning(f"Error getting move from {player_id}: {str(result)}")
                elif result:
                    moves[player_id] = result
                    logger.debug(f"{player_id} chose {result.value}")

            return moves

        except asyncio.TimeoutError:
            logger.error(f"Timeout collecting moves for round {round_id}")
            return {}
        except Exception as e:
            logger.error(f"Error collecting moves: {str(e)}")
            return {}

    async def _request_move(
        self,
        http_client: httpx.AsyncClient,
        player_id: str,
        round_id: int,
        opponent_history: List[ParityChoice]
    ) -> Optional[ParityChoice]:
        """Request a move from a specific player"""
        try:
            if player_id not in self.player_endpoints:
                logger.error(f"No endpoint for player {player_id}")
                return None

            # Create move request
            move_request = MCPEnvelope(
                message_type=MessageType.MOVE_CALL,
                sender=AGENT_ID,
                recipient=player_id,
                data=MoveRequestData(
                    match_id=self.match_id,
                    round_id=round_id,
                    opponent_history=opponent_history,
                    timeout_ms=MOVE_TIMEOUT
                ).model_dump()
            )

            # Send request
            response = await http_client.post(
                self.player_endpoints[player_id],
                json=move_request.model_dump(),
                timeout=MOVE_TIMEOUT / 1000.0
            )

            response.raise_for_status()
            response_data = response.json()

            # Parse move response
            if "data" in response_data and isinstance(response_data["data"], dict):
                envelope_data = response_data["data"]
                if envelope_data.get("message_type") == MessageType.MOVE_RESPONSE.value:
                    move_data = MoveResponseData(**envelope_data.get("data", {}))
                    return move_data.parity_choice

            return None

        except Exception as e:
            logger.error(f"Error requesting move from {player_id}: {str(e)}")
            return None


# --- Referee Service Logic ---

class RefereeService:
    """
    Core business logic for the Referee Agent.
    Manages match lifecycle and communicates with League Manager.
    """

    def __init__(self):
        self.active_matches: Dict[str, MatchOrchestrator] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.match_counter = 0

    async def initialize(self):
        """Initialize service connections"""
        timeout_config = httpx.Timeout(DEFAULT_TIMEOUT, connect=3.0)
        self.http_client = httpx.AsyncClient(timeout=timeout_config)
        logger.info("Referee service initialized")

    async def shutdown(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()

    async def schedule_match(self, schedule_data: ScheduleMatchData) -> str:
        """
        Schedule a new match between two players.

        Returns:
            Match ID of the scheduled match
        """
        try:
            match_id = schedule_data.match_id
            if not match_id:
                self.match_counter += 1
                match_id = f"M-{self.match_counter:03d}"

            # Create match orchestrator
            orchestrator = MatchOrchestrator(
                match_id=match_id,
                player_a_id=schedule_data.player_a_id,
                player_b_id=schedule_data.player_b_id
            )

            self.active_matches[match_id] = orchestrator

            logger.info(f"Scheduled match {match_id}: {schedule_data.player_a_id} vs {schedule_data.player_b_id}")

            # Start match execution in background
            asyncio.create_task(self._execute_match_workflow(match_id))

            return match_id

        except Exception as e:
            logger.error(f"Error scheduling match: {str(e)}")
            raise MatchError(f"Failed to schedule match: {str(e)}")

    async def _execute_match_workflow(self, match_id: str):
        """Execute the complete match workflow in background"""
        try:
            orchestrator = self.active_matches.get(match_id)
            if not orchestrator or not self.http_client:
                return

            # 1. Invite players
            if not await orchestrator.invite_players(self.http_client):
                logger.error(f"Failed to get player acceptance for match {match_id}")
                orchestrator.status = MatchStatus.CANCELLED
                return

            # 2. Execute match
            match_result = await orchestrator.execute_match(self.http_client)

            # 3. Notify players of match completion
            await self._notify_match_completion(orchestrator, match_result)

            # 4. Report results to League Manager
            await self._report_match_results(match_result)

            # 5. Cleanup
            await asyncio.sleep(MATCH_CLEANUP_DELAY)
            if match_id in self.active_matches:
                del self.active_matches[match_id]

        except Exception as e:
            logger.error(f"Error in match workflow for {match_id}: {str(e)}")
            if match_id in self.active_matches:
                self.active_matches[match_id].status = MatchStatus.ERROR

    async def _notify_match_completion(self, orchestrator: MatchOrchestrator, match_result: MatchResultData):
        """Notify both players that the match is complete"""
        try:
            if not self.http_client:
                return

            # Create game over notification
            game_over_msg = MCPEnvelope(
                message_type=MessageType.GAME_OVER,
                sender=AGENT_ID,
                data=match_result.model_dump()
            )

            # Send to both players
            notifications = []
            for player_id in [orchestrator.player_a_id, orchestrator.player_b_id]:
                if player_id in orchestrator.player_endpoints:
                    notifications.append(
                        self.http_client.post(
                            orchestrator.player_endpoints[player_id],
                            json=game_over_msg.model_dump()
                        )
                    )

            if notifications:
                await asyncio.gather(*notifications, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error notifying match completion: {str(e)}")

    async def _report_match_results(self, match_result: MatchResultData):
        """Report match results back to League Manager"""
        try:
            if not self.http_client:
                return

            # Create match report
            report_msg = MCPEnvelope(
                message_type=MessageType.MATCH_REPORT,
                sender=AGENT_ID,
                recipient="manager:league",
                data=match_result.model_dump()
            )

            # Send to League Manager
            await self.http_client.post(
                f"{LEAGUE_MANAGER_URL}{MCP_ENDPOINT}",
                json=report_msg.model_dump()
            )

            logger.info(f"Reported match results for {match_result.match_id}")

        except Exception as e:
            logger.error(f"Error reporting match results: {str(e)}")

    def get_match_status(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a match"""
        orchestrator = self.active_matches.get(match_id)
        if not orchestrator:
            return None

        return {
            "match_id": match_id,
            "status": orchestrator.status.value,
            "current_round": orchestrator.current_round,
            "scores": orchestrator.scores,
            "players": [orchestrator.player_a_id, orchestrator.player_b_id]
        }


# --- FastAPI Application ---

# Global service instance
referee_service: Optional[RefereeService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global referee_service

    logger.info("Referee Agent starting up...")

    try:
        referee_service = RefereeService()
        await referee_service.initialize()
        logger.info("Referee Agent initialization complete")

        yield

    finally:
        logger.info("Referee Agent shutting down...")
        if referee_service:
            await referee_service.shutdown()


app = FastAPI(
    title="Referee Agent",
    description="Distributed AI Agent League - Referee Component",
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
    global referee_service

    health_data = {
        "status": "healthy",
        "service": SERVICE_NAME,
        "agent_id": AGENT_ID,
        "active_matches": len(referee_service.active_matches) if referee_service else 0,
        "match_statuses": {
            mid: orch.status.value
            for mid, orch in (referee_service.active_matches.items() if referee_service else {})
        }
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
    global referee_service

    if not referee_service:
        raise HTTPException(status_code=503, detail="Referee service not initialized")

    req_id = request.state.request_id
    logger.info(f"Received {envelope.message_type.value} from {envelope.sender}")

    try:
        if envelope.message_type == MessageType.SCHEDULE_MATCH:
            schedule_data = ScheduleMatchData(**envelope.data)
            match_id = await referee_service.schedule_match(schedule_data)

            return BaseResponse(
                status="success",
                request_id=req_id,
                data={"match_id": match_id, "status": "scheduled"}
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


@app.get("/matches/{match_id}")
async def get_match_status(match_id: str, request: Request):
    """Get status of a specific match"""
    global referee_service

    if not referee_service:
        raise HTTPException(status_code=503, detail="Referee service not initialized")

    match_status = referee_service.get_match_status(match_id)
    if not match_status:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    return BaseResponse(
        status="success",
        request_id=request.state.request_id,
        data=match_status
    )


# --- Main Entry Point ---

def main():
    """Main entry point"""
    try:
        logger.info(f"Starting Referee Agent on port {REFEREE_PORT}")

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=REFEREE_PORT,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("Referee Agent stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start Referee Agent: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()