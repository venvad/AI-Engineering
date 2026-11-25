from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.agents.constants import DEFAULT_MODEL_NAME, DEFAULT_PROVIDER
from api.core.types import Provider, Role


class ChatMessage(BaseModel):
    role: Role
    content: str


class RAGRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    provider: Provider = DEFAULT_PROVIDER
    model_name: str = DEFAULT_MODEL_NAME
    history: list[ChatMessage] = []

    @field_validator("history")
    @classmethod
    def no_system_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if any(m.role == Role.SYSTEM for m in v):
            raise ValueError("history cannot contain system messages")
        return v


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
