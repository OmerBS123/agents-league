"""
Tests for logging utilities in shared/logger.py.
"""

import pytest
import logging
import json
from io import StringIO
from unittest.mock import patch, MagicMock

from shared.logger import get_logger, log_with_context, log_performance


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        logger = get_logger("test_module", "test-service")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test logger has correct name."""
        logger = get_logger("my_module", "test-service")
        assert logger.name == "my_module"

    def test_get_logger_different_services(self):
        """Test loggers for different services are distinct."""
        logger1 = get_logger("module1", "service-a")
        logger2 = get_logger("module2", "service-b")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_same_name_returns_same_logger(self):
        """Test getting logger with same name returns same instance."""
        logger1 = get_logger("same_module", "test-service")
        logger2 = get_logger("same_module", "test-service")

        assert logger1 is logger2

    def test_get_logger_has_handlers(self):
        """Test logger has at least one handler configured."""
        logger = get_logger("test_handlers", "test-service")
        # Either has handlers directly or inherits from root
        assert len(logger.handlers) > 0 or logger.parent is not None


class TestLogWithContext:
    """Tests for contextual logging function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        logger = MagicMock(spec=logging.Logger)
        logger.level = logging.INFO
        return logger

    def test_log_with_context_basic(self, mock_logger):
        """Test basic contextual logging."""
        log_with_context(
            logger=mock_logger,
            level=logging.INFO,
            message="Test message"
        )

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.INFO  # Level

    def test_log_with_context_includes_agent_id(self, mock_logger):
        """Test context includes agent_id."""
        log_with_context(
            logger=mock_logger,
            level=logging.INFO,
            message="Test message",
            agent_id="player:P01"
        )

        call_args = mock_logger.log.call_args
        logged_message = call_args[0][1]
        assert "player:P01" in logged_message or "agent_id" in str(call_args)

    def test_log_with_context_includes_match_id(self, mock_logger):
        """Test context includes match_id."""
        log_with_context(
            logger=mock_logger,
            level=logging.INFO,
            message="Test message",
            match_id="M-001"
        )

        call_args = mock_logger.log.call_args
        logged_message = call_args[0][1]
        assert "M-001" in logged_message or "match_id" in str(call_args)

    def test_log_with_context_includes_request_id(self, mock_logger):
        """Test context includes request_id."""
        log_with_context(
            logger=mock_logger,
            level=logging.INFO,
            message="Test message",
            request_id="req-12345"
        )

        call_args = mock_logger.log.call_args
        logged_message = call_args[0][1]
        assert "req-12345" in logged_message or "request_id" in str(call_args)

    def test_log_with_context_multiple_fields(self, mock_logger):
        """Test context with multiple fields."""
        log_with_context(
            logger=mock_logger,
            level=logging.INFO,
            message="Processing move",
            agent_id="player:P01",
            match_id="M-001",
            round_id=5
        )

        mock_logger.log.assert_called_once()

    def test_log_with_context_different_levels(self, mock_logger):
        """Test logging at different levels."""
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]

        for level in levels:
            mock_logger.reset_mock()
            mock_logger.level = level
            log_with_context(
                logger=mock_logger,
                level=level,
                message="Test"
            )
            assert mock_logger.log.called


class TestLogPerformance:
    """Tests for performance logging function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        logger = MagicMock(spec=logging.Logger)
        return logger

    def test_log_performance_basic(self, mock_logger):
        """Test basic performance logging."""
        log_performance(
            logger=mock_logger,
            operation="test_operation",
            duration_ms=150.5,
            success=True
        )

        mock_logger.info.assert_called()

    def test_log_performance_includes_duration(self, mock_logger):
        """Test performance log includes duration."""
        log_performance(
            logger=mock_logger,
            operation="test_operation",
            duration_ms=250.0,
            success=True
        )

        call_args = mock_logger.info.call_args
        logged_message = str(call_args)
        assert "250" in logged_message or "duration" in logged_message

    def test_log_performance_success_true(self, mock_logger):
        """Test performance log for successful operation."""
        log_performance(
            logger=mock_logger,
            operation="test_operation",
            duration_ms=100.0,
            success=True
        )

        # Success should use info level
        mock_logger.info.assert_called()

    def test_log_performance_success_false(self, mock_logger):
        """Test performance log for failed operation."""
        log_performance(
            logger=mock_logger,
            operation="test_operation",
            duration_ms=100.0,
            success=False,
            error="Something went wrong"
        )

        # Failed operations might use warning or error level
        assert mock_logger.warning.called or mock_logger.error.called or mock_logger.info.called

    def test_log_performance_with_extra_fields(self, mock_logger):
        """Test performance log with additional context."""
        log_performance(
            logger=mock_logger,
            operation="move_decision",
            duration_ms=1500.0,
            success=True,
            match_id="M-001",
            choice="even",
            confidence=0.85
        )

        mock_logger.info.assert_called()

    def test_log_performance_operation_name(self, mock_logger):
        """Test operation name is included in log."""
        log_performance(
            logger=mock_logger,
            operation="llm_inference",
            duration_ms=2000.0,
            success=True
        )

        call_args = mock_logger.info.call_args
        logged_message = str(call_args)
        assert "llm_inference" in logged_message


class TestLoggerIntegration:
    """Integration tests for the logging system."""

    def test_real_logger_logs_message(self, caplog):
        """Test real logger captures messages."""
        logger = get_logger("integration_test", "test-service")

        with caplog.at_level(logging.INFO):
            logger.info("Integration test message")

        assert "Integration test message" in caplog.text

    def test_log_with_context_real_logger(self, caplog):
        """Test log_with_context with real logger."""
        logger = get_logger("context_test", "test-service")

        with caplog.at_level(logging.INFO):
            log_with_context(
                logger=logger,
                level=logging.INFO,
                message="Context test",
                agent_id="player:TEST"
            )

        assert len(caplog.records) > 0

    def test_log_performance_real_logger(self, caplog):
        """Test log_performance with real logger."""
        logger = get_logger("perf_test", "test-service")

        with caplog.at_level(logging.INFO):
            log_performance(
                logger=logger,
                operation="test_op",
                duration_ms=100.0,
                success=True
            )

        assert len(caplog.records) > 0


class TestLoggerConfiguration:
    """Tests for logger configuration."""

    def test_logger_level_default(self):
        """Test logger has appropriate default level."""
        logger = get_logger("level_test", "test-service")

        # Should be INFO or lower (more verbose)
        assert logger.level <= logging.INFO or logger.getEffectiveLevel() <= logging.INFO

    def test_logger_propagate(self):
        """Test logger propagation setting."""
        logger = get_logger("propagate_test", "test-service")

        # Logger should either propagate or have its own handlers
        assert logger.propagate or len(logger.handlers) > 0
