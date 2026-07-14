import json
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Any, Iterator


_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

logger = logging.getLogger("mtbank.pipeline")


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    runtime_level_name = os.getenv(
        "RUNTIME_LOG_LEVEL",
        "WARNING",
    ).upper()
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
        "service": "mtbank-pipelines",
        "request_id": _request_id.get(),
        **fields,
    }

    if exc_info:
        payload["stack_trace"] = traceback.format_exc()

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
            error_message=str(exc),
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
