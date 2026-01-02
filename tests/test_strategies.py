"""
Tests for player strategy implementations.

Note: These tests focus on strategy logic without requiring the full agent framework.
"""

import pytest
import asyncio
from typing import List
from unittest.mock import AsyncMock, patch, MagicMock

from shared.schemas import ParityChoice


class TestRandomStrategy:
    """Tests for the RandomStrategy implementation."""

    @pytest.fixture
    def random_strategy(self):
        """Create a RandomStrategy instance for testing."""
        # Import here to avoid import issues if agents not fully loadable
        from agents.player import RandomStrategy
        return RandomStrategy(agent_id="player:TEST")

    def test_strategy_name(self, random_strategy):
        """Test strategy reports correct name."""
        assert random_strategy.strategy_name == "random"

    @pytest.mark.asyncio
    async def test_choose_move_returns_valid_choice(self, random_strategy):
        """Test choose_move returns valid ParityChoice."""
        choice, confidence, reasoning = await random_strategy.choose_move(
            match_id="M-001",
            opponent_id="player:P02",
            round_id=1,
            opponent_history=[]
        )

        assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reasoning, str)

    @pytest.mark.asyncio
    async def test_random_strategy_confidence_is_low(self, random_strategy):
        """Test random strategy reports low confidence (0.5)."""
        choice, confidence, reasoning = await random_strategy.choose_move(
            match_id="M-001",
            opponent_id="player:P02",
            round_id=1,
            opponent_history=[]
        )

        assert confidence == 0.5  # Random should be 50% confident

    @pytest.mark.asyncio
    async def test_random_strategy_ignores_history(self, random_strategy, long_opponent_history):
        """Test random strategy doesn't change behavior based on history."""
        # Run multiple times and verify it still works with history
        for _ in range(5):
            choice, confidence, reasoning = await random_strategy.choose_move(
                match_id="M-001",
                opponent_id="player:P02",
                round_id=5,
                opponent_history=long_opponent_history
            )
            assert choice in [ParityChoice.EVEN, ParityChoice.ODD]


class TestHistoryStrategy:
    """Tests for the HistoryStrategy implementation."""

    @pytest.fixture
    def history_strategy(self):
        """Create a HistoryStrategy instance for testing."""
        from agents.player import HistoryStrategy
        return HistoryStrategy(agent_id="player:TEST")

    def test_strategy_name(self, history_strategy):
        """Test strategy reports correct name."""
        assert history_strategy.strategy_name == "history"

    @pytest.mark.asyncio
    async def test_choose_move_no_history(self, history_strategy):
        """Test behavior with no opponent history."""
        choice, confidence, reasoning = await history_strategy.choose_move(
            match_id="M-001",
            opponent_id="player:P02",
            round_id=1,
            opponent_history=[]
        )

        assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
        assert confidence < 0.5  # Low confidence without history

    @pytest.mark.asyncio
    async def test_choose_move_short_history(self, history_strategy, short_opponent_history):
        """Test behavior with short history (< 3 moves)."""
        choice, confidence, reasoning = await history_strategy.choose_move(
            match_id="M-001",
            opponent_id="player:P02",
            round_id=3,
            opponent_history=short_opponent_history
        )

        assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
        # Should still work but with lower confidence

    @pytest.mark.asyncio
    async def test_counters_even_bias(self, history_strategy, biased_opponent_history):
        """Test strategy counters opponent with EVEN bias."""
        choice, confidence, reasoning = await history_strategy.choose_move(
            match_id="M-001",
            opponent_id="player:P02",
            round_id=7,
            opponent_history=biased_opponent_history
        )

        # Should detect EVEN bias and counter with ODD
        assert choice == ParityChoice.ODD
        assert confidence > 0.5
        assert "EVEN" in reasoning or "even" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_detects_alternating_pattern(self, history_strategy, long_opponent_history):
        """Test strategy detects alternating pattern."""
        choice, confidence, reasoning = await history_strategy.choose_move(
            match_id="M-001",
            opponent_id="player:P02",
            round_id=11,
            opponent_history=long_opponent_history
        )

        # Should detect alternating and have high confidence
        assert confidence >= 0.7
        assert "alternating" in reasoning.lower() or "pattern" in reasoning.lower()


class TestStrategyUpdateHistory:
    """Tests for strategy history tracking."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy for testing history updates."""
        from agents.player import RandomStrategy
        return RandomStrategy(agent_id="player:TEST")

    def test_update_history_new_match(self, strategy):
        """Test updating history for new match."""
        strategy.update_history("M-001", ParityChoice.EVEN)

        assert "M-001" in strategy.match_history
        assert strategy.match_history["M-001"] == [ParityChoice.EVEN]

    def test_update_history_existing_match(self, strategy):
        """Test updating history for existing match."""
        strategy.update_history("M-001", ParityChoice.EVEN)
        strategy.update_history("M-001", ParityChoice.ODD)

        assert len(strategy.match_history["M-001"]) == 2
        assert strategy.match_history["M-001"] == [ParityChoice.EVEN, ParityChoice.ODD]

    def test_update_history_multiple_matches(self, strategy):
        """Test tracking history across multiple matches."""
        strategy.update_history("M-001", ParityChoice.EVEN)
        strategy.update_history("M-002", ParityChoice.ODD)

        assert len(strategy.match_history) == 2
        assert "M-001" in strategy.match_history
        assert "M-002" in strategy.match_history


class TestStrategyFactory:
    """Tests for strategy factory function."""

    def test_create_random_strategy(self):
        """Test factory creates random strategy."""
        from agents.player import create_strategy
        strategy = create_strategy("random", "player:TEST")
        assert strategy.strategy_name == "random"

    def test_create_history_strategy(self):
        """Test factory creates history strategy."""
        from agents.player import create_strategy
        strategy = create_strategy("history", "player:TEST")
        assert strategy.strategy_name == "history"

    def test_create_llm_strategy(self):
        """Test factory creates LLM strategy."""
        from agents.player import create_strategy
        strategy = create_strategy("llm", "player:TEST")
        assert strategy.strategy_name == "llm"

    def test_invalid_strategy_raises_error(self):
        """Test factory raises error for unknown strategy."""
        from agents.player import create_strategy
        with pytest.raises(ValueError) as exc_info:
            create_strategy("invalid_strategy", "player:TEST")
        assert "Unknown strategy" in str(exc_info.value)


class TestLLMStrategyFallback:
    """Tests for LLM strategy fallback behavior."""

    @pytest.fixture
    def llm_strategy(self):
        """Create an LLMStrategy instance for testing."""
        from agents.player import LLMStrategy
        return LLMStrategy(agent_id="player:TEST")

    def test_strategy_name(self, llm_strategy):
        """Test strategy reports correct name."""
        assert llm_strategy.strategy_name == "llm"

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self, llm_strategy):
        """Test LLM strategy falls back on error."""
        # Mock the ollama client to raise an exception
        with patch.object(llm_strategy.ollama_client, 'get_strategic_move',
                         side_effect=Exception("Connection refused")):
            choice, confidence, reasoning = await llm_strategy.choose_move(
                match_id="M-001",
                opponent_id="player:P02",
                round_id=1,
                opponent_history=[]
            )

            # Should still return valid choice (fallback)
            assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
            assert "fallback" in reasoning.lower() or "failed" in reasoning.lower()


class TestStrategyPerformance:
    """Performance-related tests for strategies."""

    @pytest.mark.asyncio
    async def test_random_strategy_is_fast(self):
        """Test random strategy completes quickly."""
        from agents.player import RandomStrategy
        import time

        strategy = RandomStrategy(agent_id="player:TEST")

        start = time.time()
        for _ in range(100):
            await strategy.choose_move(
                match_id="M-001",
                opponent_id="player:P02",
                round_id=1,
                opponent_history=[]
            )
        elapsed = time.time() - start

        # 100 calls should complete in under 1 second
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_history_strategy_handles_long_history(self):
        """Test history strategy handles long opponent history."""
        from agents.player import HistoryStrategy
        import time

        strategy = HistoryStrategy(agent_id="player:TEST")

        # Create very long history
        long_history = [ParityChoice.EVEN, ParityChoice.ODD] * 500  # 1000 moves

        start = time.time()
        choice, confidence, reasoning = await strategy.choose_move(
            match_id="M-001",
            opponent_id="player:P02",
            round_id=1001,
            opponent_history=long_history
        )
        elapsed = time.time() - start

        # Should complete in under 100ms even with long history
        assert elapsed < 0.1
        assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
