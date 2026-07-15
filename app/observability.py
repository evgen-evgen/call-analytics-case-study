import json
import asyncio
import logging
import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Any, Iterator

from app.config import AppSettings, RuntimeSettings


_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

logger = logging.getLogger("mtbank.pipeline")
_service_name = "mtbank-pipelines"


def configure_logging(settings: RuntimeSettings | None = None) -> None:
    global _service_name
    settings = settings or AppSettings.from_env().runtime
    _service_name = settings.service_name
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    runtime_level_name = settings.runtime_log_level.upper()
    runtime_level = getattr(
        logging,
        runtime_level_name,
        logging.WARNING,
    )

    # The base Open WebUI Pipelines runtime logs the complete value returned
    # by pipe() through the root logger. Suppress that potentially sensitive
    # payload while keeping our dedicated structured logger independent.
    logging.getLogger().setLevel(runtime_level)

    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def bind_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "service": _service_name,
        "request_id": _request_id.get(),
        **fields,
    }

    if exc_info:
        # Keep call sites and line numbers without appending exception text.
        # Provider exceptions may embed generated content or response bodies.
        payload["stack_trace"] = [
            {
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in traceback.extract_tb(sys.exc_info()[2])
        ]

    logger.log(
        level,
        json.dumps(payload, ensure_ascii=False, default=str),
    )


@contextmanager
def operation(
    name: str,
    **fields: Any,
) -> Iterator[None]:
    started_at = perf_counter()
    log_event("operation.started", operation=name, **fields)

    try:
        yield
    except asyncio.CancelledError:
        log_event(
            "operation.cancelled",
            operation=name,
            duration_ms=round(
                (perf_counter() - started_at) * 1000,
                2,
            ),
            **fields,
        )
        raise
    except Exception as exc:
        log_event(
            "operation.failed",
            level=logging.ERROR,
            exc_info=True,
            operation=name,
            duration_ms=round(
                (perf_counter() - started_at) * 1000,
                2,
            ),
            error_type=type(exc).__name__,
            error_message=f"{type(exc).__name__} during {name}",
            **fields,
        )
        raise

    log_event(
        "operation.completed",
        operation=name,
        duration_ms=round(
            (perf_counter() - started_at) * 1000,
            2,
        ),
        **fields,
    )
