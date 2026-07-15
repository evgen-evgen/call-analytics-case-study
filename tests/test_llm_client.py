import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import BadRequestError, RateLimitError

from app.llm.client import LLMClient
from app.schemas import ClassificationResult


def test_rate_limit_uses_retry_after_before_retrying() -> None:
    async def scenario() -> None:
        client = LLMClient(
            base_url="http://llm.test/v1",
            api_key="test-key",
            model="test-model",
        )
        request = httpx.Request(
            "POST",
            "http://llm.test/v1/chat/completions",
        )
        error = RateLimitError(
            "Please try again in 5.64s.",
            response=httpx.Response(
                429,
                request=request,
                headers={"retry-after": "5.64"},
            ),
            body={"error": {"code": "rate_limit_exceeded"}},
        )
        response = SimpleNamespace(choices=[], usage=None)
        create = AsyncMock(side_effect=[error, response])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        with (
            patch(
                "app.llm.client.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
            patch("app.llm.client.random.uniform", return_value=0.0),
        ):
            result = await client._create_completion(model="test-model")

        assert result is response
        assert create.await_count == 2
        sleep.assert_awaited_once_with(5.64)

    asyncio.run(scenario())


def test_completion_has_wall_clock_timeout() -> None:
    async def scenario() -> None:
        client = LLMClient(
            base_url="http://llm.test/v1",
            api_key="test-key",
            model="test-model",
            timeout_seconds=0.01,
        )

        async def never_returns(**kwargs):
            await asyncio.sleep(60)

        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=never_returns)
            )
        )

        with pytest.raises(TimeoutError):
            await client._create_completion(model="test-model")

    asyncio.run(scenario())


def test_structured_generation_retries_with_json_object() -> None:
    async def scenario() -> None:
        client = LLMClient(
            base_url="http://llm.test/v1",
            api_key="test-key",
            model="test-model",
        )
        request = httpx.Request(
            "POST",
            "http://llm.test/v1/chat/completions",
        )
        error = BadRequestError(
            "Failed to validate JSON",
            response=httpx.Response(400, request=request),
            body={
                "error": {
                    "code": "json_validate_failed",
                    "failed_generation": "",
                }
            },
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"topic":"карты","priority":"low",'
                            '"reasoning":"Тест."}'
                        )
                    )
                )
            ],
            usage=None,
        )
        create = AsyncMock(side_effect=[error, response])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        result = await client.generate_structured(
            system_prompt="Проверь требования.",
            user_prompt="Транскрипт",
            response_model=ClassificationResult,
        )

        assert result == ClassificationResult(
            topic="карты",
            priority="low",
            reasoning="Тест.",
        )
        assert create.await_count == 2
        assert (
            create.await_args_list[0].kwargs["response_format"]["type"]
            == "json_schema"
        )
        assert create.await_args_list[1].kwargs["response_format"] == {
            "type": "json_object"
        }

    asyncio.run(scenario())


def test_structured_generation_falls_back_to_plain_json() -> None:
    async def scenario() -> None:
        client = LLMClient(
            base_url="http://llm.test/v1",
            api_key="test-key",
            model="test-model",
        )
        request = httpx.Request(
            "POST",
            "http://llm.test/v1/chat/completions",
        )

        def validation_error() -> BadRequestError:
            return BadRequestError(
                "Failed to validate JSON",
                response=httpx.Response(400, request=request),
                body={"error": {"code": "json_validate_failed"}},
            )

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "```json\n"
                                '{"topic":"карты","priority":"low",'
                                '"reasoning":"Тест."}'
                            "\n```"
                        )
                    )
                )
            ],
            usage=None,
        )
        create = AsyncMock(
            side_effect=[
                validation_error(),
                validation_error(),
                response,
            ]
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        result = await client.generate_structured(
            system_prompt="Проверь требования.",
            user_prompt="Транскрипт",
            response_model=ClassificationResult,
        )

        assert result == ClassificationResult(
            topic="карты",
            priority="low",
            reasoning="Тест.",
        )
        assert create.await_count == 3
        assert "response_format" not in create.await_args_list[2].kwargs

    asyncio.run(scenario())


def test_structured_generation_retries_truncated_json_once() -> None:
    async def scenario() -> None:
        client = LLMClient(
            base_url="http://llm.test/v1",
            api_key="test-key",
            model="test-model",
        )
        truncated = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content='{"topic":"карты"'),
                )
            ],
            usage=None,
        )
        valid = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"topic":"карты","priority":"low",'
                            '"reasoning":"Тест."}'
                        )
                    ),
                )
            ],
            usage=None,
        )
        create = AsyncMock(side_effect=[truncated, valid])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        result = await client.generate_structured(
            system_prompt="Проверь требования.",
            user_prompt="Транскрипт",
            response_model=ClassificationResult,
        )

        assert result.reasoning == "Тест."
        assert create.await_count == 2
        assert create.await_args_list[1].kwargs["response_format"] == {
            "type": "json_object"
        }

    asyncio.run(scenario())
