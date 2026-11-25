from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from api.agents.constants import DEFAULT_MODEL_NAME, DEFAULT_PROVIDER
from api.core.types import Provider, Role


class ChatMessage(BaseModel):
    role: Role
    content: str


class HistoryMessage(ChatMessage):
    role: Literal[Role.USER, Role.ASSISTANT]


class RAGRequest(BaseModel):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ] = Field(..., description="Query to be answered")
    provider: Provider = Field(default=DEFAULT_PROVIDER, description="LLM provider")
    model_name: str = Field(default=DEFAULT_MODEL_NAME, description="Model name")
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="Previous conversation messages (most recent last)",
    )


class RAGResult(BaseModel):
    answer: str = Field(..., description="Answer to the query")
    query: str = Field(..., description="Query to be answered")
    retrieved_context: list[str] = Field(
        ..., description="Preprocessed descriptions of the retrieved products"
    )
    retrieved_context_ids: list[str] = Field(
        ..., description="Parent ASINs of the retrieved products"
    )


class RAGResponse(BaseModel):
    answer: str = Field(..., description="Answer to the query")


class RetrievedItem(BaseModel):
    parent_asin: str = Field(..., description="Parent ASIN of the product")
    description: str = Field(..., description="Description of the product")
    similarity_score: float = Field(..., description="Similarity score of the product")
    average_rating: float | None = Field(
        ..., description="Average rating of the product"
    )
