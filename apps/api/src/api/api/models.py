from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from api.agents.constants import DEFAULT_MODEL_NAME, DEFAULT_PROVIDER
from api.core.types import Provider, Role


class ChatMessage(BaseModel):
    role: Role
    content: str


class HistoryMessage(ChatMessage):
    role: Literal[Role.USER, Role.ASSISTANT]


class RAGRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    provider: Provider = DEFAULT_PROVIDER
    model_name: str = DEFAULT_MODEL_NAME
    history: list[HistoryMessage] = []


class RAGResponse(BaseModel):
    answer: str


class RAGResult(BaseModel):
    answer: str
    query: str
    retrieved_context: list[str]
    retrieved_context_ids: list[str]


class RetrievedItem(BaseModel):
    parent_asin: str
    description: str
    similarity_score: float
    average_rating: float | None = None
