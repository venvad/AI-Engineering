import logging
from collections.abc import Callable
from functools import cache
from typing import Literal, cast

from anthropic import Anthropic, omit
from anthropic.types import MessageParam
from fastapi import FastAPI, HTTPException, Request
from google import genai
from google.genai import types
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from api.core.config import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Provider = Literal["OpenAI", "Anthropic", "Google"]


@cache
def _openai() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


@cache
def _anthropic() -> Anthropic:
    return Anthropic(api_key=config.ANTHROPIC_API_KEY)


@cache
def _gemini() -> genai.Client:
    return genai.Client(api_key=config.GOOGLE_API_KEY)


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def _run_openai(model: str, messages: list[dict], max_tokens: int) -> str:
    resp = _openai().chat.completions.create(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        max_completion_tokens=max_tokens,
        reasoning_effort="low",
    )
    return resp.choices[0].message.content or ""


def _run_anthropic(model: str, messages: list[dict], max_tokens: int) -> str:
    system, chat = _split_system(messages)
    resp = _anthropic().messages.create(
        model=model,
        system=system or omit,
        messages=cast(list[MessageParam], chat),
        max_tokens=max_tokens,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _run_gemini(model: str, messages: list[dict], max_tokens: int) -> str:
    system, chat = _split_system(messages)
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
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


_RUNNERS: dict[Provider, Callable[[str, list[dict], int], str]] = {
    "OpenAI": _run_openai,
    "Anthropic": _run_anthropic,
    "Google": _run_gemini,
}


def run_llm(
    provider: Provider, model_name: str, messages: list[dict], max_tokens: int = 2000
) -> str:
    try:
        runner = _RUNNERS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider: {provider!r}") from None
    logger.info(
        "Calling %s model %s (messages=%d, max_tokens=%d)",
        provider,
        model_name,
        len(messages),
        max_tokens,
    )
    result = runner(model_name, messages, max_tokens)
    if not result.strip():
        # reasoning models can spend the whole token limit thinking
        raise ValueError(f"Empty response (max_tokens={max_tokens} may be too low)")
    return result


class ChatRequest(BaseModel):
    provider: Provider
    models_name: str
    messages: list[dict]


class ChatResponse(BaseModel):
    message: str


app = FastAPI()


@app.post("/chat")
def chat(request: Request, payload: ChatRequest) -> ChatResponse:

    try:
        result = run_llm(payload.provider, payload.models_name, payload.messages)
    except Exception as e:
        logger.exception(
            "LLM call failed: %s %s", payload.provider, payload.models_name
        )
        raise HTTPException(
            status_code=502, detail=f"{payload.provider} request failed: {e}"
        ) from e

    return ChatResponse(message=result)
