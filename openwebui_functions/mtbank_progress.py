"""
title: MTBank Call Analytics
author: MTBank
version: 0.1.0
description: Proxies MTBank ASR and renders pipeline progress as one status.
"""

import json
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx


class Pipe:
    PROGRESS_MESSAGES = {
        "📎 Файл получен. Начинаю обработку…": "Файл получен. Начинаю обработку…",
        "⏳ Другой аудиофайл уже обрабатывается. Ваш запрос поставлен в очередь…": (
            "Запрос поставлен в очередь…"
        ),
        "🎧 Аудио подготовлено. Выполняю транскрибацию…": (
            "Выполняю транскрибацию…"
        ),
        "📝 Транскрибация готова. Разделяю спикеров…": (
            "Разделяю спикеров…"
        ),
        "👥 Спикеры определены. Готовлю финальный транскрипт…": (
            "Готовлю финальный транскрипт…"
        ),
        "🧠 Запускаю аналитических агентов…": (
            "Анализирую звонок…"
        ),
        "✅ Анализ готов.": "Анализ готов.",
    }

    def __init__(self) -> None:
        self.name = "MTBank Call Analytics"
        self.base_url = os.getenv(
            "OPENAI_API_BASE_URL",
            "http://pipelines:9099/v1",
        ).rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Callable | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = dict(body)
        payload["model"] = "mtbank-asr"
        payload["stream"] = True

        await self._emit_status(
            __event_emitter__,
            "Подготавливаю запрос…",
            done=False,
        )

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        raw_data = line[5:].strip()
                        if not raw_data or raw_data == "[DONE]":
                            continue

                        data = json.loads(raw_data)
                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        content = choices[0].get("delta", {}).get("content")
                        if not content:
                            continue

                        progress = self.PROGRESS_MESSAGES.get(content.strip())
                        if progress is not None:
                            await self._emit_status(
                                __event_emitter__,
                                progress,
                                done=False,
                            )
                            continue

                        yield content
        finally:
            await self._emit_status(
                __event_emitter__,
                "Обработка завершена.",
                done=True,
            )

    @staticmethod
    async def _emit_status(
        emitter: Callable | None,
        description: str,
        *,
        done: bool,
    ) -> None:
        if emitter is None:
            return
        await emitter(
            {
                "type": "status",
                "data": {
                    "action": "mtbank_call_analysis",
                    "description": description,
                    "done": done,
                },
            }
        )
