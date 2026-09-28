"""Structured JSON logging configuration for Wikipedia MCP Agent.

Formats all application logs as structured JSON records (JSON Lines - JSONL)
and writes them to a rotating log file, while maintaining readable console output.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union


class JSONFormatter(logging.Formatter):
    """Custom logging formatter outputting single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializes LogRecord into a structured JSON string."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func_name": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_name": record.threadName,
        }

        # Include exception stack trace if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include custom extra metadata if supplied
        extra_fields = {}
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message",
        }
        for key, val in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                try:
                    json.dumps(val)  # Test serialization
                    extra_fields[key] = val
                except (TypeError, OverflowError):
                    extra_fields[key] = str(val)

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, ensure_ascii=False)


def setup_json_logging(
    log_file_path: Optional[Union[str, Path]] = None,
    level: Union[int, str] = logging.INFO,
    console_output: bool = True,
    console_stream: Any = sys.stderr,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,
) -> logging.Logger:
    """Configures root logging with both a JSON rotating file handler and a console handler.

    Args:
        log_file_path: Target path for structured JSON logs (defaults to logs/app.log.jsonl).
        level: Logging level (e.g. logging.INFO or "INFO").
        console_output: Whether to also write formatted logs to the console stream.
        console_stream: Stream for console output (default sys.stderr).
        max_bytes: Max size before rotating log file.
        backup_count: Number of rotated backup files to retain.

    Returns:
        Root logger instance.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Resolve default log file path
    if log_file_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        log_file_path = base_dir / "logs" / "app.log.jsonl"
    else:
        log_file_path = Path(log_file_path)

    # Ensure log directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Structured JSON File Handler
    # Avoid duplicate file handlers if setup is called multiple times
    has_json_handler = any(
        isinstance(h, RotatingFileHandler) and getattr(h, "_is_json_handler", False)
        for h in root_logger.handlers
    )

    if not has_json_handler:
        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        file_handler._is_json_handler = True  # Tag to prevent duplicates
        root_logger.addHandler(file_handler)

    # 2. Console Stream Handler (Human-readable)
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root_logger.handlers
    )

    if console_output and not has_console_handler:
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    return root_logger
