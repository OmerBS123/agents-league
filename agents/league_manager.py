"""
League Manager Agent - Distributed AI Agent League System.
Central coordinator for player registration, match scheduling, and tournament management.
"""

import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from itertools import combinations
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
    LeagueError, RegistrationError, SchedulingError, NetworkError
)
from shared.schemas import (
    MCPEnvelope, MessageType, BaseResponse, AgentStatus,
    RegistrationData, RegistrationResponse, ScheduleMatchData,
    MatchResultData, StandingsData, create_registration_message
)
from consts import (
    LEAGUE_MANAGER_PORT, REFEREE_URL, MCP_ENDPOINT, HEALTH_ENDPOINT,
    REGISTRATION_ENDPOINT, DEFAULT_TIMEOUT, MAX_CONCURRENT_MATCHES,
    ROUND_ROBIN_SCHEDULING
)

# --- Configuration ---
SERVICE_NAME = "league-manager"
AGENT_ID = "manager:league"
logger = get_logger(__name__, SERVICE_NAME)


# --- Data Models ---

@dataclass
class PlayerProfile:
    """Registered player information and statistics"""
    agent_id: str
    display_name: str
    contact_endpoint: str
    strategies: List[str]
    capabilities: Dict[str, Any]

    # Registration metadata
    registration_time: float = field(default_factory=time.time)
    agent_version: str = "1.0.0"

    # Tournament statistics
    matches_played: int = 0
    matches_won: int = 0
    matches_lost: int = 0
    total_rounds_won: int = 0
    total_rounds_played: int = 0

    # Status tracking
    status: AgentStatus = AgentStatus.IDLE
    last_seen: float = field(default_factory=time.time)
    current_match_id: Optional[str] = None

    @property
    def win_rate(self) -> float:
        """Calculate win percentage"""
        return (self.matches_won / self.matches_played) if self.matches_played > 0 else 0.0

    @property
    def round_win_rate(self) -> float:
        """Calculate round win percentage"""
        return (self.total_rounds_won / self.total_rounds_played) if self.total_rounds_played > 0 else 0.0

    @property
    def points(self) -> int:
        """Calculate tournament points (3 for win, 1 for loss)"""
        return (self.matches_won * 3) + (self.matches_lost * 1)


@dataclass
class MatchSchedule:
    """Scheduled match information"""
    match_id: str
    player_a_id: str
    player_b_id: str
    scheduled_time: float
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled
    priority: int = 1

    # Results (populated when match completes)
    winner_id: Optional[str] = None
    final_score: Optional[Dict[str, int]] = None
    match_duration_ms: Optional[float] = None
    completed_time: Optional[float] = None


# --- Tournament Management ---

class TournamentManager:
    """
    Manages tournament logic including round-robin scheduling,
    standings calculation, and match coordination.
    """

    def __init__(self):
        self.tournament_id = f"TOURNAMENT-{int(time.time())}"
        self.players: Dict[str, PlayerProfile] = {}
        self.schedule: List[MatchSchedule] = []
        self.completed_matches: Dict[str, MatchResultData] = {}
        self.tournament_start_time: Optional[float] = None
        self.tournament_active = False

    def register_player(self, registration_data: RegistrationData, agent_id: str) -> bool:
        """
        Register a new player in the tournament.

        Args:
            registration_data: Player registration information
            agent_id: Unique agent identifier

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Check if player already registered
            if agent_id in self.players:
                logger.warning(f"Player {agent_id} already registered")
                return False

            # Create player profile
            player = PlayerProfile(
                agent_id=agent_id,
                display_name=registration_data.display_name,
                contact_endpoint=registration_data.contact_endpoint,
                strategies=registration_data.strategies,
                capabilities=registration_data.capabilities,
                agent_version=getattr(registration_data, 'agent_version', '1.0.0')
            )

            self.players[agent_id] = player

            logger.info(f"Registered player {agent_id}: {registration_data.display_name}")
            logger.info(f"Total registered players: {len(self.players)}")

            # If we have enough players and tournament not started, generate schedule
            if len(self.players) >= 2 and not self.tournament_active:
                self._generate_round_robin_schedule()

            return True

        except Exception as e:
            logger.error(f"Error registering player {agent_id}: {str(e)}")
            return False

    def _generate_round_robin_schedule(self):
        """Generate round-robin tournament schedule"""
        try:
            if not ROUND_ROBIN_SCHEDULING or len(self.players) < 2:
                return

            logger.info("Generating round-robin schedule")

            # Clear existing schedule
            self.schedule.clear()

            # Get all player IDs
            player_ids = list(self.players.keys())

            # Generate all possible pairings
            match_counter = 1
            current_time = time.time()

            for player_a, player_b in combinations(player_ids, 2):
                match_id = f"M-{match_counter:03d}"

                # Schedule match with some time spacing
                scheduled_time = current_time + (match_counter * 10)  # 10 second intervals

                match = MatchSchedule(
                    match_id=match_id,
                    player_a_id=player_a,
                    player_b_id=player_b,
                    scheduled_time=scheduled_time,
                    priority=1
                )

                self.schedule.append(match)
                match_counter += 1

            logger.info(f"Generated {len(self.schedule)} matches for {len(self.players)} players")

            # Start tournament if we have a complete schedule
            if len(self.schedule) > 0:
                self.tournament_active = True
                self.tournament_start_time = time.time()

        except Exception as e:
            logger.error(f"Error generating schedule: {str(e)}")

    def get_next_matches(self, limit: int = MAX_CONCURRENT_MATCHES) -> List[MatchSchedule]:
        """
        Get the next matches ready to be executed.

        Args:
            limit: Maximum number of matches to return

        Returns:
            List of matches ready for execution
        """
        current_time = time.time()

        # Find scheduled matches that are ready to run
        ready_matches = [
            match for match in self.schedule
            if match.status == "scheduled" and match.scheduled_time <= current_time
        ]

        # Sort by priority and scheduled time
        ready_matches.sort(key=lambda m: (m.priority, m.scheduled_time))

        return ready_matches[:limit]

    def update_match_status(self, match_id: str, status: str, result: Optional[MatchResultData] = None):
        """Update match status and results"""
        try:
            # Find and update schedule entry
            for match in self.schedule:
                if match.match_id == match_id:
                    match.status = status

                    if result and status == "completed":
                        match.winner_id = result.winner_id
                        match.final_score = result.final_score
                        match.match_duration_ms = result.match_duration_ms
                        match.completed_time = time.time()

                        # Store complete results
                        self.completed_matches[match_id] = result

                        # Update player statistics
                        self._update_player_stats(result)

                        logger.info(f"Match {match_id} completed: {result.winner_id} wins")

                    break

        except Exception as e:
            logger.error(f"Error updating match status: {str(e)}")

    def _update_player_stats(self, result: MatchResultData):
        """Update player statistics based on match result"""
        try:
            player_a_id = None
            player_b_id = None

            # Extract player IDs from final score
            if result.final_score and len(result.final_score) == 2:
                player_ids = list(result.final_score.keys())
                player_a_id, player_b_id = player_ids[0], player_ids[1]

            if not player_a_id or not player_b_id:
                logger.error(f"Could not extract player IDs from match result {result.match_id}")
                return

            # Update match counts
            for player_id in [player_a_id, player_b_id]:
                if player_id in self.players:
                    self.players[player_id].matches_played += 1
                    self.players[player_id].current_match_id = None
                    self.players[player_id].status = AgentStatus.IDLE

                    # Update round statistics
                    if player_id in result.final_score:
                        rounds_won = result.final_score[player_id]
                        self.players[player_id].total_rounds_won += rounds_won
                        self.players[player_id].total_rounds_played += result.total_rounds

            # Update win/loss records
            if result.winner_id and result.winner_id in self.players:
                self.players[result.winner_id].matches_won += 1

                # Determine loser
                loser_id = player_b_id if result.winner_id == player_a_id else player_a_id
                if loser_id in self.players:
                    self.players[loser_id].matches_lost += 1

        except Exception as e:
            logger.error(f"Error updating player stats: {str(e)}")

    def get_standings(self) -> StandingsData:
        """Generate current tournament standings"""
        try:
            standings_list = []

            for player in self.players.values():
                standings_entry = {
                    "agent_id": player.agent_id,
                    "display_name": player.display_name,
                    "matches_played": player.matches_played,
                    "matches_won": player.matches_won,
                    "matches_lost": player.matches_lost,
                    "points": player.points,
                    "win_rate": round(player.win_rate, 3),
                    "rounds_won": player.total_rounds_won,
                    "rounds_played": player.total_rounds_played,
                    "round_win_rate": round(player.round_win_rate, 3)
                }
                standings_list.append(standings_entry)

            # Sort by points (descending), then by win rate, then by rounds won
            standings_list.sort(key=lambda x: (x["points"], x["win_rate"], x["rounds_won"]), reverse=True)

            completed_matches = len([m for m in self.schedule if m.status == "completed"])

            return StandingsData(
                tournament_id=self.tournament_id,
                standings=standings_list,
                total_matches=len(self.schedule),
                completed_matches=completed_matches
            )

        except Exception as e:
            logger.error(f"Error generating standings: {str(e)}")
            return StandingsData(
                tournament_id=self.tournament_id,
                standings=[],
                total_matches=0,
                completed_matches=0
            )

    def is_tournament_complete(self) -> bool:
        """Check if all matches have been completed"""
        if not self.schedule:
            return False
        return all(match.status == "completed" for match in self.schedule)


# --- League Manager Service ---

class LeagueManagerService:
    """
    Core business logic for the League Manager Agent.
    Handles registration, scheduling, and tournament coordination.
    """

    def __init__(self):
        self.tournament = TournamentManager()
        self.http_client: Optional[httpx.AsyncClient] = None
        self.background_tasks_running = False

    async def initialize(self):
        """Initialize service connections"""
        timeout_config = httpx.Timeout(DEFAULT_TIMEOUT, connect=3.0)
        self.http_client = httpx.AsyncClient(timeout=timeout_config)

        # Start background match orchestration
        self.background_tasks_running = True
        asyncio.create_task(self._match_orchestration_loop())

        logger.info("League Manager service initialized")

    async def shutdown(self):
        """Cleanup resources"""
        self.background_tasks_running = False

        if self.http_client:
            await self.http_client.aclose()

    async def handle_registration(self, envelope: MCPEnvelope) -> MCPEnvelope:
        """Handle player registration request"""
        try:
            registration_data = RegistrationData(**envelope.data)
            agent_id = envelope.sender

            success = self.tournament.register_player(registration_data, agent_id)

            if success:
                response_data = RegistrationResponse(
                    accepted=True,
                    agent_id=agent_id,
                    league_status="active",
                    tournament_info={
                        "tournament_id": self.tournament.tournament_id,
                        "total_players": len(self.tournament.players),
                        "matches_scheduled": len(self.tournament.schedule)
                    }
                )

                logger.info(f"Registration accepted for {agent_id}")
            else:
                response_data = RegistrationResponse(
                    accepted=False,
                    agent_id=agent_id,
                    league_status="error",
                    reason="Registration failed"
                )

            return envelope.create_reply(
                response_type=MessageType.REGISTER_ACK,
                data=response_data.model_dump(),
                sender_id=AGENT_ID
            )

        except Exception as e:
            logger.error(f"Error handling registration: {str(e)}")
            return envelope.create_error_reply(
                error_code="REGISTRATION_ERROR",
                error_message=str(e),
                sender_id=AGENT_ID
            )

    async def handle_match_report(self, envelope: MCPEnvelope):
        """Handle match result report from Referee"""
        try:
            match_result = MatchResultData(**envelope.data)
            match_id = match_result.match_id

            logger.info(f"Received match report for {match_id}")

            # Update tournament with match results
            self.tournament.update_match_status(match_id, "completed", match_result)

            # Check if tournament is complete
            if self.tournament.is_tournament_complete():
                await self._handle_tournament_completion()

        except Exception as e:
            logger.error(f"Error handling match report: {str(e)}")

    async def _match_orchestration_loop(self):
        """Background task for match orchestration"""
        logger.info("Starting match orchestration loop")

        while self.background_tasks_running:
            try:
                # Check for matches ready to execute
                ready_matches = self.tournament.get_next_matches()

                if ready_matches:
                    logger.info(f"Found {len(ready_matches)} matches ready for execution")

                    # Schedule matches with Referee
                    for match in ready_matches:
                        await self._schedule_match_with_referee(match)

                # Sleep before next check
                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in match orchestration loop: {str(e)}")
                await asyncio.sleep(10.0)  # Longer delay on error

    async def _schedule_match_with_referee(self, match: MatchSchedule):
        """Schedule a specific match with the Referee"""
        try:
            if not self.http_client:
                return

            logger.info(f"Scheduling match {match.match_id} with Referee")

            # Update match status
            match.status = "in_progress"

            # Update player status
            for player_id in [match.player_a_id, match.player_b_id]:
                if player_id in self.tournament.players:
                    self.tournament.players[player_id].status = AgentStatus.BUSY
                    self.tournament.players[player_id].current_match_id = match.match_id

            # Create schedule request
            schedule_request = MCPEnvelope(
                message_type=MessageType.SCHEDULE_MATCH,
                sender=AGENT_ID,
                recipient="referee:REF-MAIN",
                data=ScheduleMatchData(
                    match_id=match.match_id,
                    player_a_id=match.player_a_id,
                    player_b_id=match.player_b_id,
                    priority=match.priority
                ).model_dump()
            )

            # Send to Referee
            response = await self.http_client.post(
                f"{REFEREE_URL}{MCP_ENDPOINT}",
                json=schedule_request.model_dump()
            )

            response.raise_for_status()
            logger.info(f"Successfully scheduled match {match.match_id}")

        except Exception as e:
            logger.error(f"Error scheduling match {match.match_id}: {str(e)}")
            # Reset match status on error
            match.status = "scheduled"

    async def _handle_tournament_completion(self):
        """Handle tournament completion"""
        try:
            logger.info("Tournament completed! Generating final standings...")

            standings = self.tournament.get_standings()

            # Log final results
            logger.info("=== TOURNAMENT RESULTS ===")
            for i, entry in enumerate(standings.standings, 1):
                logger.info(
                    f"{i}. {entry['display_name']} - "
                    f"Points: {entry['points']}, "
                    f"W-L: {entry['matches_won']}-{entry['matches_lost']}, "
                    f"Win Rate: {entry['win_rate']:.1%}"
                )

            # TODO: Could send standings update to all players here

        except Exception as e:
            logger.error(f"Error handling tournament completion: {str(e)}")

    def get_tournament_status(self) -> Dict[str, Any]:
        """Get current tournament status"""
        standings = self.tournament.get_standings()

        return {
            "tournament_id": self.tournament.tournament_id,
            "active": self.tournament.tournament_active,
            "start_time": self.tournament.tournament_start_time,
            "total_players": len(self.tournament.players),
            "matches_scheduled": len(self.tournament.schedule),
            "matches_completed": standings.completed_matches,
            "is_complete": self.tournament.is_tournament_complete(),
            "standings": standings.standings[:5]  # Top 5 for status
        }


# --- FastAPI Application ---

# Global service instance
league_service: Optional[LeagueManagerService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global league_service

    logger.info("League Manager starting up...")

    try:
        league_service = LeagueManagerService()
        await league_service.initialize()
        logger.info("League Manager initialization complete")

        yield

    finally:
        logger.info("League Manager shutting down...")
        if league_service:
            await league_service.shutdown()


app = FastAPI(
    title="League Manager",
    description="Distributed AI Agent League - Central Coordinator",
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
    global league_service

    health_data = {
        "status": "healthy",
        "service": SERVICE_NAME,
        "agent_id": AGENT_ID
    }

    if league_service:
        health_data.update(league_service.get_tournament_status())

    return BaseResponse(
        status="success",
        request_id=request.state.request_id,
        data=health_data
    )


@app.post(REGISTRATION_ENDPOINT, response_model=BaseResponse)
async def handle_registration(
    envelope: MCPEnvelope,
    request: Request
):
    """Player registration endpoint"""
    global league_service

    if not league_service:
        raise HTTPException(status_code=503, detail="League Manager service not initialized")

    req_id = request.state.request_id
    logger.info(f"Registration request from {envelope.sender}")

    try:
        response_envelope = await league_service.handle_registration(envelope)

        return BaseResponse(
            status="success",
            request_id=req_id,
            data=response_envelope.model_dump()
        )

    except Exception as e:
        logger.error(f"Error handling registration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(MCP_ENDPOINT, response_model=BaseResponse)
async def handle_mcp_message(
    envelope: MCPEnvelope,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Main MCP message handler"""
    global league_service

    if not league_service:
        raise HTTPException(status_code=503, detail="League Manager service not initialized")

    req_id = request.state.request_id
    logger.info(f"Received {envelope.message_type.value} from {envelope.sender}")

    try:
        if envelope.message_type == MessageType.MATCH_REPORT:
            background_tasks.add_task(league_service.handle_match_report, envelope)

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


@app.get("/tournament/status")
async def get_tournament_status(request: Request):
    """Get current tournament status"""
    global league_service

    if not league_service:
        raise HTTPException(status_code=503, detail="League Manager service not initialized")

    status_data = league_service.get_tournament_status()

    return BaseResponse(
        status="success",
        request_id=request.state.request_id,
        data=status_data
    )


@app.get("/tournament/standings")
async def get_standings(request: Request):
    """Get current tournament standings"""
    global league_service

    if not league_service:
        raise HTTPException(status_code=503, detail="League Manager service not initialized")

    standings = league_service.tournament.get_standings()

    return BaseResponse(
        status="success",
        request_id=request.state.request_id,
        data=standings.model_dump()
    )


# --- Main Entry Point ---

def main():
    """Main entry point"""
    try:
        logger.info(f"Starting League Manager on port {LEAGUE_MANAGER_PORT}")

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=LEAGUE_MANAGER_PORT,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("League Manager stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start League Manager: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()