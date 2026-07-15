import asyncio
import queue
import threading
from collections.abc import Coroutine, Iterator
from typing import Any, Callable


class SyncStreamBridge:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def run(self, coroutine: Coroutine[Any, Any, str]) -> str:
        return asyncio.run_coroutine_threadsafe(
            coroutine,
            self.event_loop,
        ).result()

    def stream(
        self,
        operation: Callable[
            [Callable[[str], None], threading.Event],
            Coroutine[Any, Any, str],
        ],
    ) -> Iterator[str]:
        progress_queue: queue.Queue[str] = queue.Queue()
        cancel_event = threading.Event()
        future = asyncio.run_coroutine_threadsafe(
            operation(progress_queue.put, cancel_event),
            self.event_loop,
        )

        try:
            while not future.done() or not progress_queue.empty():
                try:
                    yield progress_queue.get(timeout=0.25)
                except queue.Empty:
                    yield ""
            yield future.result()
        finally:
            cancel_event.set()
            if not future.done():
                future.cancel()
