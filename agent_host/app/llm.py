"""The language model behind a tiny interface, so the graph never imports an SDK.

The graph only needs "send a system prompt and a user message, get text back". The planner
asks for JSON and we parse it ourselves, which keeps this identical for every provider and
makes the graph easy to test with a fake model.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from app.config import Settings, get_settings

logger = logging.getLogger("agent.llm")


class LLMError(Exception):
    """A model call failed in a way that is safe to show to the user."""


@runtime_checkable
class ChatModel(Protocol):
    last_stop_reason: str  # "end_turn" when the model finished normally

    async def complete(self, system: str, user: str, max_tokens: int = 1500) -> str: ...


class AnthropicChat:
    def __init__(self, model: str, api_key: str = "", timeout: float = 90.0) -> None:
        import anthropic  # imported here so Bedrock-only images do not need it

        self._anthropic = anthropic
        self._model = model
        self.last_stop_reason = "end_turn"
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or None,
            timeout=timeout,
            max_retries=2,  # None key -> env var
        )

    async def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        try:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except self._anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from exc
        except self._anthropic.APITimeoutError as exc:
            raise LLMError(f"Anthropic did not answer in time ({type(exc).__name__}).") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached. Wait a minute and retry.") from exc
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error ({type(exc).__name__}).") from exc
        self.last_stop_reason = str(getattr(msg, "stop_reason", "") or "unknown")
        logger.info(
            "model stop_reason=%s output_tokens=%s",
            self.last_stop_reason,
            getattr(getattr(msg, "usage", None), "output_tokens", "?"),
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class BedrockChat:
    def __init__(self, model_id: str, region: str) -> None:
        import boto3

        self._model_id = model_id
        self.last_stop_reason = "end_turn"
        self._client = boto3.client("bedrock-runtime", region_name=region)

    async def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        def call() -> str:
            resp = self._client.converse(
                modelId=self._model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": max_tokens},
            )
            self.last_stop_reason = str(resp.get("stopReason", "unknown")).lower()
            parts = resp["output"]["message"]["content"]
            return "".join(p.get("text", "") for p in parts)

        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            raise LLMError(
                f"Bedrock call failed ({type(exc).__name__}). Check AWS_PROFILE, region, "
                "and that model access is enabled for BEDROCK_MODEL_ID."
            ) from exc


def get_chat_model(settings: Settings | None = None) -> ChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicChat(
            settings.llm_model, settings.anthropic_api_key, settings.llm_timeout_seconds
        )
    return BedrockChat(settings.bedrock_model_id, settings.aws_region)
