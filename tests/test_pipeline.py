import asyncio
import threading
from types import MethodType

from pipelines.pipeline import Pipeline
from app.schemas import AudioSource


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
        cancel_event=None,
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


def test_detects_openwebui_internal_task() -> None:
    assert Pipeline._internal_task(
        {
            "stream": False,
            "metadata": {"task": "query_generation"},
        }
    ) == "query_generation"
    assert Pipeline._internal_task(
        {"stream": True, "metadata": {}}
    ) is None


def test_non_stream_audio_request_is_not_processed() -> None:
    pipeline = Pipeline()
    pipeline.downloader = object()
    pipeline.transcriber = object()
    pipeline.diarizer = object()

    result = pipeline.pipe(
        user_message=(
            '<file type="file" url="file-id" '
            'content_type="audio/mpeg" name="call.mp3" />'
        ),
        model_id="pipeline",
        messages=[],
        body={"stream": False},
    )

    assert "streaming-запросе" in result


def test_audio_jobs_are_limited_and_queued() -> None:
    async def scenario() -> None:
        pipeline = Pipeline()
        pipeline._audio_semaphore = asyncio.Semaphore(1)

        first_started = asyncio.Event()
        release_first = asyncio.Event()
        active_jobs = 0
        max_active_jobs = 0

        async def fake_process_audio(
            self,
            source,
            *,
            progress=None,
            cancel_event=None,
        ):
            nonlocal active_jobs, max_active_jobs
            active_jobs += 1
            max_active_jobs = max(max_active_jobs, active_jobs)
            try:
                if source.file_id == "first":
                    first_started.set()
                    await release_first.wait()
                return source.file_id
            finally:
                active_jobs -= 1

        pipeline._process_audio = MethodType(
            fake_process_audio,
            pipeline,
        )

        first = asyncio.create_task(
            pipeline._process_audio_with_limit(
                AudioSource(
                    file_id="first",
                    filename="first.wav",
                    content_type="audio/wav",
                )
            )
        )
        await first_started.wait()

        queued_progress: list[str] = []
        second = asyncio.create_task(
            pipeline._process_audio_with_limit(
                AudioSource(
                    file_id="second",
                    filename="second.wav",
                    content_type="audio/wav",
                ),
                progress=queued_progress.append,
            )
        )
        await asyncio.sleep(0)

        assert not second.done()
        assert queued_progress == [
            pipeline.presenter.format_progress("audio_queued")
        ]

        release_first.set()
        assert await asyncio.gather(first, second) == [
            "first",
            "second",
        ]
        assert max_active_jobs == 1

    asyncio.run(scenario())
