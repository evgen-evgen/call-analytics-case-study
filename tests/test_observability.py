import json
import logging

import pytest

from app.observability import (
    bind_request_id,
    configure_logging,
    logger,
    operation,
    reset_request_id,
)
from app.llm.client import LLMRequestError
from app.openwebui.formatter import OpenWebUIResponseFormatter


class EventHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(json.loads(record.getMessage()))


@pytest.fixture
def captured_events():
    handler = EventHandler()
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    try:
        yield handler.events
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)


def test_operation_events_are_correlated_and_do_not_leak_error_content(
    captured_events,
) -> None:
    token = bind_request_id("request-123")
    try:
        with operation("audio.normalize"):
            pass

        with pytest.raises(RuntimeError):
            with operation("llm.generate"):
                raise RuntimeError("secret prompt and provider response")
    finally:
        reset_request_id(token)

    assert [event["event"] for event in captured_events] == [
        "operation.started",
        "operation.completed",
        "operation.started",
        "operation.failed",
    ]
    assert all(
        event["request_id"] == "request-123"
        for event in captured_events
    )
    completed, failed = captured_events[1], captured_events[3]
    assert completed["operation"] == "audio.normalize"
    assert failed["operation"] == "llm.generate"
    assert "duration_ms" in completed
    assert "duration_ms" in failed
    serialized = json.dumps(captured_events)
    assert "secret prompt" not in serialized
    assert "provider response" not in serialized


def test_logging_levels_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RUNTIME_LOG_LEVEL", "ERROR")

    configure_logging()

    assert logger.level == logging.DEBUG
    assert logging.getLogger().level == logging.ERROR


def test_service_name_is_written_from_configuration(
    monkeypatch,
    captured_events,
) -> None:
    monkeypatch.setenv("SERVICE_NAME", "mtbank-analysis-api")
    configure_logging()

    with operation("api.test"):
        pass

    assert {event["service"] for event in captured_events} == {
        "mtbank-analysis-api"
    }


def test_llm_provider_response_is_not_written_to_logs(
    captured_events,
) -> None:
    OpenWebUIResponseFormatter.error(
        LLMRequestError("provider response with secret transcript")
    )

    serialized = json.dumps(captured_events)
    assert "provider response" not in serialized
    assert "secret transcript" not in serialized
