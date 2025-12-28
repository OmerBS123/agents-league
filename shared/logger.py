"""
Singleton logger factory with JSON Lines formatting.
Provides structured logging for all agents in the distributed system.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from functools import lru_cache

from consts import LOG_LEVEL, JSON_LOG_FORMAT, LOG_REQUEST_IDS


class JsonFormatter(logging.Formatter):
    """
    JSON Lines formatter for structured logging.
    Each log entry is a single JSON object on one line.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string"""
        # Base log object structure
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "service": getattr(record, "service_name", "unknown"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add request ID if available (from middleware)
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        # Add agent ID if available
        if hasattr(record, "agent_id"):
            log_obj["agent_id"] = record.agent_id

        # Add match ID if available (for match-specific operations)
        if hasattr(record, "match_id"):
            log_obj["match_id"] = record.match_id

        # Add conversation ID if available (for message threading)
        if hasattr(record, "conversation_id"):
            log_obj["conversation_id"] = record.conversation_id

        # Add performance metrics if available
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }

        # Add extra fields if present
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj, default=str, separators=(',', ':'))


class ContextFilter(logging.Filter):
    """
    Filter that adds contextual information to log records.
    """

    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        """Add service name and other context to log record"""
        record.service_name = self.service_name
        return True


@lru_cache(maxsize=None)
def get_logger(name: str, service_name: str = "league-agent") -> logging.Logger:
    """
    Singleton logger factory with JSON formatting.

    Args:
        name: Logger name (typically __name__ or module name)
        service_name: Service identifier for structured logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Set log level from configuration
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Configure formatter based on configuration
        if JSON_LOG_FORMAT:
            formatter = JsonFormatter()
        else:
            # Standard text format as fallback
            formatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] %(service_name)s.%(module)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Add context filter
        context_filter = ContextFilter(service_name)
        logger.addFilter(context_filter)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    request_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    match_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log message with contextual information.

    Args:
        logger: Logger instance
        level: Log level (logging.INFO, etc.)
        message: Log message
        request_id: HTTP request ID
        agent_id: Agent identifier
        match_id: Match identifier
        conversation_id: Message conversation ID
        duration_ms: Operation duration in milliseconds
        extra_data: Additional fields to include
    """
    # Create a log record with extra context
    extra = {}

    if request_id:
        extra['request_id'] = request_id
    if agent_id:
        extra['agent_id'] = agent_id
    if match_id:
        extra['match_id'] = match_id
    if conversation_id:
        extra['conversation_id'] = conversation_id
    if duration_ms is not None:
        extra['duration_ms'] = duration_ms
    if extra_data:
        extra['extra_data'] = extra_data

    logger.log(level, message, extra=extra)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    success: bool = True,
    **context
) -> None:
    """
    Log performance metrics for operations.

    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        success: Whether operation succeeded
        **context: Additional context fields
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"{operation} completed in {duration_ms:.2f}ms - {status}"

    extra_data = {
        "operation": operation,
        "success": success,
        **context
    }

    level = logging.INFO if success else logging.WARNING
    log_with_context(
        logger=logger,
        level=level,
        message=message,
        duration_ms=duration_ms,
        extra_data=extra_data
    )