import asyncio

from openwebui_functions.mtbank_progress import Pipe


def test_replacing_progress_closes_previous_status() -> None:
    async def scenario() -> list[dict]:
        events: list[dict] = []

        async def emit(event: dict) -> None:
            events.append(event)

        active = await Pipe._replace_status(emit, None, "Первый")
        active = await Pipe._replace_status(emit, active, "Второй")
        await Pipe._emit_status(emit, active, done=True)
        return events

    events = asyncio.run(scenario())

    assert [event["data"] for event in events] == [
        {
            "action": "mtbank_call_analysis",
            "description": "Первый",
            "done": False,
        },
        {
            "action": "mtbank_call_analysis",
            "description": "Первый",
            "done": True,
        },
        {
            "action": "mtbank_call_analysis",
            "description": "Второй",
            "done": False,
        },
        {
            "action": "mtbank_call_analysis",
            "description": "Второй",
            "done": True,
        },
    ]


def test_stream_error_has_user_visible_fallback() -> None:
    assert Pipe._error_message() == (
        "## Ошибка модели\n\n"
        "Не удалось завершить анализ. Попробуйте повторить запрос."
    )
