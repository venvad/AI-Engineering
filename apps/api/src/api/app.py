from fastapi import FastAPI, Request
from pydantic import BaseModel

from google import genai
from groq import Groq
from openai import OpenAI
from api.core.config import config

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_llm(provider: str, model_name:str, messages: list[dict], max_tokens: int = 500) -> str:
    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal"
        ).choices[0].message.content
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens
        ).choices[0].message.content
    else:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        return client.models.generate_content(
            model=model_name,
            contents=[message["content"] for message in messages]
        ).text

class ChatRequest(BaseModel):
    provider: str
    models_name: str
    messages: list[dict]

class ChatResponse(BaseModel):
    message: str


app = FastAPI()

@app.post("/chat")
def chat(
    request: Request,
    payload: ChatRequest
) -> ChatResponse:

    result = run_llm(payload.provider, payload.models_name, payload.messages)

    return ChatResponse(message=result)