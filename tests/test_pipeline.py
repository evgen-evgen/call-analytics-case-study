import asyncio
import json
from types import MethodType, SimpleNamespace

from app.config import AppSettings
from app.schemas import AnalyzeResponse
from app.openwebui import (
    OpenWebUIRequestHandler,
    OpenWebUIRequestParser,
    OpenWebUIResponseFormatter,
)
from pipelines.pipeline import Pipeline, PipelineRuntime


class ImmediateBridge:
    def run(self, coroutine):
        return asyncio.run(coroutine)

    def stream(self, operation):
        chunks: list[str] = []
        result = asyncio.run(operation(chunks.append, None))
        return iter([*chunks, result])


class FakeHandler:
    async def handle(self, request, *, progress=None, cancel_event=None):
        if progress is not None:
            progress("Файл получен.\n\n")
            progress("Анализ готов.\n\n")
        return "FINAL"


def test_valves_use_shared_settings_without_duplicating_defaults(
    monkeypatch,
) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("MAX_CONCURRENT_AUDIO_JOBS", "3")

    valves = Pipeline.Valves.from_settings(AppSettings.from_env())

    assert valves.WHISPER_MODEL == "small"
    assert valves.MAX_CONCURRENT_AUDIO_JOBS == 3
    assert valves.WHISPER_DEVICE == "cpu"


def test_pipeline_requires_async_startup() -> None:
    pipeline = Pipeline()

    result = pipeline.pipe(
        user_message="",
        model_id="test",
        messages=[],
        body={},
    )

    assert result == "Pipeline не был корректно инициализирован."


def test_pipeline_only_routes_streaming_request() -> None:
    pipeline = Pipeline()
    pipeline.runtime = PipelineRuntime(
        llm_client=SimpleNamespace(),
        analysis_service=SimpleNamespace(),
        handler=FakeHandler(),
        bridge=ImmediateBridge(),
    )

    chunks = list(
        pipeline.pipe(
            user_message="",
            model_id="test",
            messages=[],
            body={"stream": True},
        )
    )

    assert chunks == [
        "Файл получен.\n\n",
        "Анализ готов.\n\n",
        "FINAL",
    ]


def test_openwebui_analysis_uses_same_json_contract_as_api() -> None:
    response = AnalyzeResponse.model_validate(
        {
            "transcript": [
                {
                    "speaker": "Оператор",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Добрый день.",
                }
            ],
            "classification": {"topic": "карты", "priority": "low"},
            "quality_score": {
                "total": 100,
                "checklist": {
                    "greeting": True,
                    "need_detection": True,
                    "solution_provided": True,
                    "farewell": True,
                },
            },
            "compliance": {"passed": True, "issues": []},
            "summary": "Звонок обработан.",
            "action_items": [],
        }
    )
    formatter = OpenWebUIResponseFormatter()
    formatter.response_mapper = SimpleNamespace(map=lambda _: response)

    rendered = formatter.analysis(None)

    assert json.loads(rendered) == response.model_dump(mode="json")
    assert formatter.latest_analysis(
        [{"role": "assistant", "content": rendered}]
    ) == response.model_dump_json(indent=2)


def test_request_parser_prioritizes_internal_task() -> None:
    parser = OpenWebUIRequestParser()
    request = parser.parse(
        user_message=(
            '<file type="file" url="file-id" '
            'content_type="audio/mpeg" name="call.mp3" />'
        ),
        model_id="pipeline",
        messages=[],
        body={
            "stream": True,
            "metadata": {"task": "query_generation"},
        },
    )

    assert request.internal_task == "query_generation"
    assert request.audio_source is None


def make_handler(max_jobs: int = 1) -> OpenWebUIRequestHandler:
    return OpenWebUIRequestHandler(
        downloader=SimpleNamespace(),
        analysis_service=SimpleNamespace(),
        chat_agent=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        formatter=OpenWebUIResponseFormatter(),
        max_concurrent_audio_jobs=max_jobs,
    )


def test_handler_skips_non_stream_audio_request() -> None:
    request = OpenWebUIRequestParser().parse(
        user_message=(
            '<file type="file" url="file-id" '
            'content_type="audio/mpeg" name="call.mp3" />'
        ),
        model_id="pipeline",
        messages=[],
        body={"stream": False},
    )

    result = asyncio.run(make_handler().handle(request))

    assert "streaming-запросе" in result


def test_handler_limits_audio_jobs_and_reports_queue() -> None:
    async def scenario() -> None:
        handler = make_handler(max_jobs=1)
        parser = OpenWebUIRequestParser()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        active_jobs = 0
        max_active_jobs = 0

        async def fake_analyze(self, request, progress, cancel_event):
            nonlocal active_jobs, max_active_jobs
            active_jobs += 1
            max_active_jobs = max(max_active_jobs, active_jobs)
            try:
                if request.audio_source.file_id == "first":
                    first_started.set()
                    await release_first.wait()
                return request.audio_source.file_id
            finally:
                active_jobs -= 1

        handler._analyze_audio = MethodType(fake_analyze, handler)

        def request(file_id: str):
            return parser.parse(
                user_message=(
                    f'<file type="file" url="{file_id}" '
                    f'content_type="audio/wav" name="{file_id}.wav" />'
                ),
                model_id="pipeline",
                messages=[],
                body={"stream": True},
            )

        first = asyncio.create_task(handler.handle(request("first")))
        await first_started.wait()
        queued_progress: list[str] = []
        second = asyncio.create_task(
            handler.handle(request("second"), progress=queued_progress.append)
        )
        await asyncio.sleep(0)

        assert not second.done()
        assert queued_progress == [handler.formatter.progress("audio_queued")]
        release_first.set()
        assert await asyncio.gather(first, second) == ["first", "second"]
        assert max_active_jobs == 1

    asyncio.run(scenario())
