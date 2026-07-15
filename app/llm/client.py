import asyncio
import json
import random
import re
from typing import TypeVar

from openai import AsyncOpenAI, BadRequestError, RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import AppSettings, LLMSettings
from app.observability import log_event, operation


ResultT = TypeVar(
    "ResultT",
    bound=BaseModel,
)


def build_strict_json_schema(
    response_model: type[BaseModel],
) -> dict:
    schema = response_model.model_json_schema()
    _make_schema_strict(schema)
    return schema


def _make_schema_strict(node: object) -> None:
    if isinstance(node, list):
        for item in node:
            _make_schema_strict(item)
        return

    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["additionalProperties"] = False
        node["required"] = list(properties)

    for value in node.values():
        _make_schema_strict(value)


class LLMClientError(RuntimeError):
    pass


class LLMConfigurationError(LLMClientError):
    pass


class LLMRequestError(LLMClientError):
    pass


class LLMResponseValidationError(LLMClientError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        settings = settings or AppSettings.from_env().llm
        self.model = model or settings.model
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.timeout_seconds
        )

        self.client = AsyncOpenAI(
            base_url=(
                base_url or settings.base_url
            ),
            api_key=(
                api_key or settings.api_key
            ),
            timeout=effective_timeout,
            max_retries=0,
        )
        self._request_semaphore = asyncio.Semaphore(
            settings.max_concurrent_requests
        )
        self._rate_limit_max_retries = int(
            settings.rate_limit_max_retries
        )
        self._rate_limit_max_delay = settings.rate_limit_max_delay_seconds
        self._max_completion_tokens = settings.max_completion_tokens
        self._request_timeout_seconds = effective_timeout

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResultT],
        temperature: float = 0.0,
        response_format_mode: str = "json_schema",
    ) -> ResultT:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "Верни только один валидный JSON-объект без "
                    "Markdown и пояснений. JSON должен соответствовать "
                    "этой схеме:\n"
                    f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False)}"
                ),
            },
            messages[1],
        ]

        try:
            if response_format_mode == "json_object":
                with operation(
                    "llm.generate",
                    model=self.model,
                    response_schema=response_model.__name__,
                    response_format="json_object",
                ):
                    response = await self._create_completion(
                        model=self.model,
                        temperature=temperature,
                        max_completion_tokens=self._max_completion_tokens,
                        messages=fallback_messages,
                        response_format={"type": "json_object"},
                    )
            elif response_format_mode == "json_schema":
                try:
                    with operation(
                        "llm.generate",
                        model=self.model,
                        response_schema=response_model.__name__,
                        response_format="json_schema",
                    ):
                        response = await self._create_completion(
                            model=self.model,
                            temperature=temperature,
                            max_completion_tokens=self._max_completion_tokens,
                            messages=messages,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": response_model.__name__,
                                    "strict": True,
                                    "schema": build_strict_json_schema(
                                        response_model
                                    ),
                                },
                            },
                        )
                except BadRequestError as exc:
                    if not self._is_json_validation_failure(exc):
                        raise

                    log_event(
                        "llm.structured_retry",
                        model=self.model,
                        response_schema=response_model.__name__,
                        reason="provider_json_validation_failed",
                        fallback_format="json_object",
                    )
                    try:
                        with operation(
                            "llm.generate",
                            model=self.model,
                            response_schema=response_model.__name__,
                            response_format="json_object",
                            retry=True,
                        ):
                            response = await self._create_completion(
                                model=self.model,
                                temperature=temperature,
                                max_completion_tokens=self._max_completion_tokens,
                                messages=fallback_messages,
                                response_format={"type": "json_object"},
                            )
                    except BadRequestError as fallback_exc:
                        if not self._is_json_validation_failure(fallback_exc):
                            raise

                        log_event(
                            "llm.structured_retry",
                            model=self.model,
                            response_schema=response_model.__name__,
                            reason="provider_json_mode_validation_failed",
                            fallback_format="plain_json",
                        )
                        with operation(
                            "llm.generate",
                            model=self.model,
                            response_schema=response_model.__name__,
                            response_format="plain_json",
                            retry=True,
                        ):
                            response = await self._create_completion(
                                model=self.model,
                                temperature=temperature,
                                max_completion_tokens=self._max_completion_tokens,
                                messages=fallback_messages,
                            )
            else:
                raise ValueError(
                    f"Unsupported response format mode: {response_format_mode}"
                )

            self._log_usage(response)

            if not response.choices:
                raise LLMRequestError(
                    "LLM returned no choices."
                )

            content = response.choices[0].message.content

            if not content:
                raise LLMRequestError(
                    "LLM returned an empty response."
                )

            try:
                return self._validate_response(content, response_model)
            except json.JSONDecodeError as exc:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                log_event(
                    "llm.response_validation_retry",
                    model=self.model,
                    response_schema=response_model.__name__,
                    reason="invalid_json",
                    finish_reason=finish_reason,
                )
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n\n"
                            "Предыдущий ответ оказался оборванным. Повтори "
                            "ответ кратко и верни ровно один полностью "
                            "завершённый JSON-объект без Markdown. Не повторяй "
                            "одно и то же объяснение. JSON должен соответствовать "
                            "этой схеме:\n"
                            f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False)}"
                        ),
                    },
                    messages[1],
                ]
                with operation(
                    "llm.generate",
                    model=self.model,
                    response_schema=response_model.__name__,
                    response_format="json_object",
                    retry=True,
                ):
                    retry_response = await self._create_completion(
                        model=self.model,
                        temperature=temperature,
                        max_completion_tokens=self._max_completion_tokens,
                        messages=retry_messages,
                        response_format={"type": "json_object"},
                    )
                self._log_usage(retry_response)
                if not retry_response.choices:
                    raise LLMRequestError("LLM returned no choices on retry.")
                retry_content = retry_response.choices[0].message.content
                if not retry_content:
                    raise LLMRequestError("LLM returned an empty response on retry.")
                try:
                    return self._validate_response(retry_content, response_model)
                except (ValidationError, json.JSONDecodeError) as retry_exc:
                    raise LLMResponseValidationError(
                        "LLM response does not match "
                        f"{response_model.__name__} after retry: {retry_exc}"
                    ) from retry_exc
            except ValidationError as exc:
                raise LLMResponseValidationError(
                    "LLM response does not match "
                    f"{response_model.__name__}: {exc}"
                ) from exc

        except LLMClientError:
            raise

        except Exception as exc:
            raise LLMRequestError(
                f"Structured LLM request failed: {exc}"
            ) from exc

    @staticmethod
    def _is_json_validation_failure(exc: BadRequestError) -> bool:
        body = exc.body
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return error.get("code") == "json_validate_failed"

        return "json_validate_failed" in str(exc)

    @staticmethod
    def _unwrap_json_content(content: str) -> str:
        value = content.strip()
        if value.startswith("```") and value.endswith("```"):
            first_newline = value.find("\n")
            if first_newline != -1:
                value = value[first_newline + 1 : -3].strip()
        return value

    @classmethod
    def _validate_response(
        cls,
        content: str,
        response_model: type[ResultT],
    ) -> ResultT:
        parsed = json.loads(cls._unwrap_json_content(content))
        return response_model.model_validate(parsed)

    async def _create_completion(self, **kwargs):
        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                async with self._request_semaphore:
                    async with asyncio.timeout(self._request_timeout_seconds):
                        return await self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if attempt >= self._rate_limit_max_retries:
                    raise

                retry_after = self._retry_after_seconds(exc)
                delay = min(
                    retry_after if retry_after is not None else 2 ** attempt,
                    self._rate_limit_max_delay,
                )
                delay += random.uniform(0.0, min(1.0, delay * 0.1))
                log_event(
                    "llm.rate_limit_retry",
                    model=self.model,
                    attempt=attempt + 1,
                    max_retries=self._rate_limit_max_retries,
                    delay_seconds=round(delay, 2),
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Unreachable rate-limit retry state.")

    @staticmethod
    def _retry_after_seconds(exc: RateLimitError) -> float | None:
        response = getattr(exc, "response", None)
        if response is not None:
            value = response.headers.get("retry-after")
            if value is not None:
                try:
                    return max(0.0, float(value))
                except ValueError:
                    pass

        match = re.search(
            r"try again in\s+([0-9]+(?:\.[0-9]+)?)s",
            str(exc),
            flags=re.IGNORECASE,
        )
        return float(match.group(1)) if match else None

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        try:
            with operation(
                "llm.generate_text",
                model=self.model,
            ):
                response = await self._create_completion(
                    model=self.model,
                    temperature=temperature,
                    max_completion_tokens=self._max_completion_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                )
        except Exception as exc:
            raise LLMRequestError(
                f"LLM request failed: {exc}"
            ) from exc

        self._log_usage(response)

        if not response.choices:
            raise LLMRequestError(
                "LLM returned no choices."
            )

        content = response.choices[0].message.content

        if not content:
            raise LLMRequestError(
                "LLM returned an empty response."
            )

        return content.strip()

    def _log_usage(self, response: object) -> None:
        choices = getattr(response, "choices", None) or []
        finish_reason = (
            getattr(choices[0], "finish_reason", None) if choices else None
        )
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        log_event(
            "llm.usage",
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            finish_reason=finish_reason,
        )
