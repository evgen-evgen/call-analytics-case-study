import json
import os
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

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
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = (
            model
            or os.getenv("LLM_MODEL", "analysis-model")
        )

        self.client = AsyncOpenAI(
            base_url=(
                base_url
                or os.getenv(
                    "LLM_BASE_URL",
                    "http://litellm:4000/v1",
                )
            ),
            api_key=(
                api_key
                or os.getenv(
                    "LLM_API_KEY",
                    "sk-local-litellm",
                )
            ),
            timeout=timeout_seconds,
            max_retries=1 
        )

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResultT],
        temperature: float = 0.0,
    ) -> ResultT:
        try:
            with operation(
                "llm.generate",
                model=self.model,
                response_schema=response_model.__name__,
            ):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
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

            if response.usage is not None:
                log_event(
                    "llm.usage",
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

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
                return response_model.model_validate_json(
                    content
                )
            except (ValidationError, json.JSONDecodeError) as exc:
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
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
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
