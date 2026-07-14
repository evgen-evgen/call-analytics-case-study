import asyncio
import threading
from types import MethodType

from pipelines.pipeline import Pipeline


def test_pipeline_requires_async_startup() -> None:
    pipeline = Pipeline()

    result = pipeline.pipe(
        user_message="",
        model_id="test",
        messages=[],
        body={},
    )

    assert result == "Pipeline не был корректно инициализирован."


def test_pipeline_streams_progress_before_final_result() -> None:
    pipeline = Pipeline()
    pipeline.downloader = object()
    pipeline.transcriber = object()
    pipeline.diarizer = object()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    pipeline._event_loop = loop

    async def fake_pipe_async(
        self,
        user_message,
        model_id,
        messages,
        body,
        progress=None,
    ):
        progress("Файл получен.\n\n")
        await asyncio.sleep(0.01)
        progress("Анализ готов.\n\n")
        return "FINAL"

    pipeline._pipe_async = MethodType(
        fake_pipe_async,
        pipeline,
    )

    try:
        chunks = list(
            pipeline.pipe(
                user_message="<file />",
                model_id="test",
                messages=[],
                body={"stream": True},
            )
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()

    assert chunks == [
        "Файл получен.\n\n",
        "Анализ готов.\n\n",
        "FINAL",
    ]
