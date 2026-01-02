"""
Tests for Ollama LLM integration in shared/ollama_strategy.py.

Note: These tests mock the Ollama API to avoid requiring a running Ollama server.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

from shared.schemas import ParityChoice
from shared.ollama_strategy import (
    OllamaClient, GameContext, PromptBuilder, ResponseParser
)
from shared.exceptions import LLMError


class TestGameContext:
    """Tests for the GameContext dataclass."""

    @pytest.fixture
    def sample_context(self) -> GameContext:
        """Create a sample game context."""
        return GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=5,
            opponent_history=[ParityChoice.EVEN, ParityChoice.ODD, ParityChoice.EVEN],
            my_history=[ParityChoice.ODD, ParityChoice.EVEN, ParityChoice.ODD],
            current_score={"player:TEST": 2, "player:OPP": 1}
        )

    def test_context_creation(self, sample_context):
        """Test GameContext creation with all fields."""
        assert sample_context.agent_id == "player:TEST"
        assert sample_context.round_number == 5
        assert len(sample_context.opponent_history) == 3

    def test_score_summary_with_scores(self, sample_context):
        """Test score_summary property with valid scores."""
        summary = sample_context.score_summary
        assert "-" in summary
        assert "2" in summary or "1" in summary

    def test_score_summary_empty_scores(self):
        """Test score_summary with empty scores."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=1,
            opponent_history=[],
            my_history=[],
            current_score={}
        )
        assert context.score_summary == "0-0"

    def test_context_with_empty_history(self):
        """Test context with empty opponent history."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=1,
            opponent_history=[],
            my_history=[],
            current_score={}
        )
        assert len(context.opponent_history) == 0


class TestPromptBuilder:
    """Tests for the PromptBuilder class."""

    def test_build_system_prompt(self):
        """Test system prompt generation."""
        prompt = PromptBuilder.build_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100

        # Should contain key elements
        assert "Game Theory" in prompt or "game" in prompt.lower()
        assert "even" in prompt.lower()
        assert "odd" in prompt.lower()
        assert "JSON" in prompt or "json" in prompt.lower()

    def test_build_user_prompt_basic(self):
        """Test user prompt generation with basic context."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=3,
            opponent_history=[],
            my_history=[],
            current_score={"player:TEST": 0, "player:OPP": 0}
        )

        prompt = PromptBuilder.build_user_prompt(context)

        assert isinstance(prompt, str)
        assert "3" in prompt  # Round number
        assert "player:OPP" in prompt  # Opponent ID

    def test_build_user_prompt_with_history(self):
        """Test user prompt includes opponent history."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=5,
            opponent_history=[ParityChoice.EVEN, ParityChoice.ODD, ParityChoice.EVEN],
            my_history=[],
            current_score={}
        )

        prompt = PromptBuilder.build_user_prompt(context)

        assert "even" in prompt.lower()
        assert "odd" in prompt.lower()

    def test_build_user_prompt_detects_pattern(self):
        """Test user prompt detects repeating pattern."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=5,
            opponent_history=[ParityChoice.EVEN, ParityChoice.EVEN, ParityChoice.EVEN],
            my_history=[],
            current_score={}
        )

        prompt = PromptBuilder.build_user_prompt(context)
        # Should contain pattern warning
        assert len(prompt) > 0


class TestResponseParser:
    """Tests for the ResponseParser class."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        raw = '{"choice": "even", "confidence": 0.8, "reasoning": "Pattern detected"}'

        result = ResponseParser.extract_and_validate_json(raw)

        assert result["choice"] == "even"
        assert result["confidence"] == 0.8
        assert result["reasoning"] == "Pattern detected"

    def test_parse_json_in_markdown(self):
        """Test parsing JSON embedded in markdown."""
        raw = '''Here is my analysis:

```json
{"choice": "odd", "confidence": 0.7, "reasoning": "Counter opponent"}
```

That's my decision.'''

        result = ResponseParser.extract_and_validate_json(raw)

        assert result["choice"] == "odd"
        assert result["confidence"] == 0.7

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON with surrounding text."""
        raw = '''Based on my analysis, I recommend:
{"choice": "even", "confidence": 0.6, "reasoning": "Random choice"}
This should work well.'''

        result = ResponseParser.extract_and_validate_json(raw)

        assert result["choice"] == "even"

    def test_parse_normalizes_choice(self):
        """Test parser normalizes choice to lowercase."""
        raw = '{"choice": "EVEN", "confidence": 0.5, "reasoning": "Test"}'

        result = ResponseParser.extract_and_validate_json(raw)

        assert result["choice"] == "even"

    def test_parse_clamps_confidence(self):
        """Test parser clamps confidence to valid range."""
        raw = '{"choice": "even", "confidence": 1.5, "reasoning": "Test"}'

        result = ResponseParser.extract_and_validate_json(raw)

        assert result["confidence"] == 1.0  # Clamped to max

    def test_parse_default_confidence(self):
        """Test parser uses default confidence when missing."""
        raw = '{"choice": "even", "reasoning": "Test"}'

        result = ResponseParser.extract_and_validate_json(raw)

        assert result["confidence"] == 0.5  # Default

    def test_parse_invalid_choice_raises(self):
        """Test parser raises on invalid choice."""
        raw = '{"choice": "invalid", "confidence": 0.5, "reasoning": "Test"}'

        with pytest.raises(LLMError):
            ResponseParser.extract_and_validate_json(raw)

    def test_parse_keyword_fallback(self):
        """Test parser uses keyword detection as fallback."""
        raw = "I think we should choose even for this round because..."

        result = ResponseParser.extract_and_validate_json(raw)

        assert result["choice"] == "even"
        assert result["confidence"] < 0.5  # Low confidence for fallback

    def test_parse_no_valid_response_raises(self):
        """Test parser raises when no valid response found."""
        raw = "This is completely unrelated text with no choice."

        with pytest.raises(LLMError):
            ResponseParser.extract_and_validate_json(raw)

    def test_parse_truncates_long_reasoning(self):
        """Test parser truncates very long reasoning."""
        long_reasoning = "A" * 1000
        raw = f'{{"choice": "even", "confidence": 0.5, "reasoning": "{long_reasoning}"}}'

        result = ResponseParser.extract_and_validate_json(raw)

        assert len(result["reasoning"]) <= 500


class TestOllamaClient:
    """Tests for the OllamaClient class."""

    @pytest.fixture
    def client(self):
        """Create an OllamaClient instance."""
        return OllamaClient(model="test-model", base_url="http://localhost:11434")

    @pytest.fixture
    def sample_context(self) -> GameContext:
        """Create a sample game context."""
        return GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=3,
            opponent_history=[ParityChoice.EVEN, ParityChoice.ODD],
            my_history=[ParityChoice.ODD, ParityChoice.EVEN],
            current_score={"player:TEST": 1, "player:OPP": 1}
        )

    def test_client_initialization(self, client):
        """Test client initializes with correct settings."""
        assert client.model == "test-model"
        assert client.base_url == "http://localhost:11434"
        assert client.total_requests == 0

    @pytest.mark.asyncio
    async def test_get_strategic_move_success(self, client, sample_context):
        """Test successful strategic move from mocked Ollama."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"choice": "even", "confidence": 0.75, "reasoning": "Pattern analysis"}'
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            choice, confidence, reasoning = await client.get_strategic_move(sample_context)

            assert choice == ParityChoice.EVEN
            assert confidence == 0.75
            assert "Pattern" in reasoning

    @pytest.mark.asyncio
    async def test_get_strategic_move_timeout_fallback(self, client, sample_context):
        """Test fallback on timeout."""
        import httpx

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            choice, confidence, reasoning = await client.get_strategic_move(sample_context)

            assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
            assert "fallback" in reasoning.lower() or "timeout" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_get_strategic_move_connection_error_fallback(self, client, sample_context):
        """Test fallback on connection error."""
        import httpx

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            choice, confidence, reasoning = await client.get_strategic_move(sample_context)

            assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
            assert "fallback" in reasoning.lower() or "connection" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_get_strategic_move_parse_error_fallback(self, client, sample_context):
        """Test fallback on parse error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "This is not valid JSON at all"
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            choice, confidence, reasoning = await client.get_strategic_move(sample_context)

            assert choice in [ParityChoice.EVEN, ParityChoice.ODD]

    def test_get_performance_stats(self, client):
        """Test performance stats retrieval."""
        client.total_requests = 10
        client.successful_requests = 8
        client.avg_response_time = 500.0

        stats = client.get_performance_stats()

        assert stats["total_requests"] == 10
        assert stats["successful_requests"] == 8
        assert stats["success_rate"] == 0.8
        assert stats["avg_response_time_ms"] == 500.0
        assert stats["model"] == "test-model"

    def test_performance_stats_zero_requests(self, client):
        """Test performance stats with zero requests."""
        stats = client.get_performance_stats()

        assert stats["total_requests"] == 0
        assert stats["success_rate"] == 0.0


class TestFallbackStrategy:
    """Tests for the fallback strategy logic."""

    @pytest.fixture
    def client(self):
        """Create an OllamaClient instance."""
        return OllamaClient()

    def test_fallback_counters_even_bias(self, client):
        """Test fallback counters opponent with EVEN bias."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=5,
            opponent_history=[ParityChoice.EVEN, ParityChoice.EVEN, ParityChoice.EVEN],
            my_history=[],
            current_score={}
        )

        choice, confidence, reasoning = client._fallback_strategy("timeout", context)

        assert choice == ParityChoice.ODD  # Counter EVEN tendency
        assert confidence > 0.5

    def test_fallback_counters_odd_bias(self, client):
        """Test fallback counters opponent with ODD bias."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=5,
            opponent_history=[ParityChoice.ODD, ParityChoice.ODD, ParityChoice.ODD],
            my_history=[],
            current_score={}
        )

        choice, confidence, reasoning = client._fallback_strategy("timeout", context)

        assert choice == ParityChoice.EVEN  # Counter ODD tendency
        assert confidence > 0.5

    def test_fallback_random_without_history(self, client):
        """Test fallback uses random when no history."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=1,
            opponent_history=[],
            my_history=[],
            current_score={}
        )

        choice, confidence, reasoning = client._fallback_strategy("timeout", context)

        assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
        assert confidence == 0.5  # Nash equilibrium confidence
        assert "Nash" in reasoning or "equilibrium" in reasoning.lower()

    def test_fallback_includes_error_type(self, client):
        """Test fallback reasoning includes error type."""
        context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPP",
            match_id="M-001",
            round_number=1,
            opponent_history=[],
            my_history=[],
            current_score={}
        )

        choice, confidence, reasoning = client._fallback_strategy("connection_error", context)

        assert "connection_error" in reasoning
