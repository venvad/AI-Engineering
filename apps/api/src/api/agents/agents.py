import logging
from collections.abc import Callable
from functools import cache
from typing import cast

from anthropic import Anthropic, omit
from anthropic.types import MessageParam
from google import genai
from google.genai import types
from langsmith.wrappers import wrap_anthropic, wrap_gemini, wrap_openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from api.agents.constants import (
    DEFAULT_REASONING_EFFORT,
    LLM_TIMEOUT_SECONDS,
    MAX_TOKENS,
)
from api.api.models import ChatMessage
from api.core.config import config
from api.core.types import Provider, Role

logger = logging.getLogger(__name__)


# wrapped so every LLM call (incl. RAG generation) shows up in LangSmith with token usage
@cache
def _openai() -> OpenAI:
    return wrap_openai(
        OpenAI(api_key=config.OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
    )


@cache
def _anthropic() -> Anthropic:
    return wrap_anthropic(
        Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
    )


@cache
def _gemini() -> genai.Client:
    return wrap_gemini(
        genai.Client(
            api_key=config.GOOGLE_API_KEY,
            http_options=types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000),
        )
    )


def _extract_system_prompt(
    messages: list[ChatMessage],
) -> tuple[str, list[ChatMessage]]:
    system = "\n".join(m.content for m in messages if m.role == Role.SYSTEM)
    return system, [m for m in messages if m.role != Role.SYSTEM]


def _run_openai(model: str, messages: list[ChatMessage], max_tokens: int) -> str:
    resp = _openai().chat.completions.create(
        model=model,
        messages=cast(
            list[ChatCompletionMessageParam], [m.model_dump() for m in messages]
        ),
        max_completion_tokens=max_tokens,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    return (resp.choices[0].message.content or "") if resp.choices else ""


def _run_anthropic(model: str, messages: list[ChatMessage], max_tokens: int) -> str:
    system, chat = _extract_system_prompt(messages)
    resp = _anthropic().messages.create(
        model=model,
        system=system or omit,
        messages=cast(list[MessageParam], [m.model_dump() for m in chat]),
        max_tokens=max_tokens,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _run_gemini(model: str, messages: list[ChatMessage], max_tokens: int) -> str:
    system, chat = _extract_system_prompt(messages)
    contents = [
        types.Content(
            role="model" if m.role == Role.ASSISTANT else "user",
            parts=[types.Part(text=m.content)],
        )
        for m in chat
    ]
    resp = _gemini().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            # thinking tokens count toward max_output_tokens
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            ),
        ),
    )
    return resp.text or ""


_RUNNERS: dict[Provider, Callable[[str, list[ChatMessage], int], str]] = {
    Provider.OPENAI: _run_openai,
    Provider.ANTHROPIC: _run_anthropic,
    Provider.GOOGLE: _run_gemini,
}


def run_llm(
    provider: Provider,
    model_name: str,
    messages: list[ChatMessage],
    max_tokens: int = MAX_TOKENS,
) -> str:
    logger.info(
        "Calling %s/%s (messages=%d, max_tokens=%d)",
        provider,
        model_name,
        len(messages),
        max_tokens,
    )
    result = _RUNNERS[provider](model_name, messages, max_tokens)
    if not result.strip():
        raise RuntimeError(f"Empty response (max_tokens={max_tokens} may be too low)")
    return result
