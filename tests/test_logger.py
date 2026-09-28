"""Unit tests for structured JSON logging."""

import json
import logging
from pathlib import Path
import pytest

from app.logger import JSONFormatter, setup_json_logging


def test_json_formatter_valid_json():
    """Verify that JSONFormatter emits valid parseable JSON with all required keys."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/path/to/test.py",
        lineno=42,
        msg="Test message with param: %s",
        args=("val1",),
        exc_info=None,
    )
    formatted = formatter.format(record)

    # Must be valid JSON
    data = json.loads(formatted)
    assert data["level"] == "INFO"
    assert data["logger"] == "test.logger"
    assert data["message"] == "Test message with param: val1"
    assert data["line"] == 42
    assert "timestamp" in data
    assert "process_id" in data
    assert "thread_name" in data


def test_json_formatter_with_exception():
    """Verify that exceptions are formatted and included in the JSON record."""
    formatter = JSONFormatter()
    try:
        raise ValueError("Simulated error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test.error",
        level=logging.ERROR,
        pathname="/path/to/test.py",
        lineno=99,
        msg="An error occurred",
        args=(),
        exc_info=exc_info,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["level"] == "ERROR"
    assert "exception" in data
    assert "ValueError: Simulated error" in data["exception"]


def test_setup_json_logging_writes_to_file(tmp_path: Path):
    """Verify that setup_json_logging creates log file and writes valid JSON lines."""
    log_file = tmp_path / "test_app.log.jsonl"
    logger = setup_json_logging(log_file_path=log_file, level="DEBUG", console_output=False)

    test_logger = logging.getLogger("app.test_component")
    test_logger.info("Structured JSON log test event")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip()
    assert len(content) > 0

    # Each line should be valid JSON
    lines = content.splitlines()
    last_line = lines[-1]
    parsed = json.loads(last_line)
    assert parsed["message"] == "Structured JSON log test event"
    assert parsed["logger"] == "app.test_component"
