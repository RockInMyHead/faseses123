"""
Centralized logging configuration
"""
import sys
import logging
from typing import Optional
from pathlib import Path

import structlog

from .config import settings


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False
) -> None:
    """
    Configure structured logging for the application

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
        json_format: Whether to use JSON format for logs
    """

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *(log_file and [logging.FileHandler(log_file)] or [])
        ]
    )

    # Configure structlog
    if json_format:
        # JSON format for production
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Human-readable format for development
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.BoundLoggerWithContext:
    """
    Get a structured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


# Initialize default logging
setup_logging(
    level="DEBUG" if settings.debug else "INFO",
    json_format=not settings.debug
)
