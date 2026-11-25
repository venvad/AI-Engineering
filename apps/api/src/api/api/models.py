from pydantic import BaseModel, Field

from api.core.types import Provider, Role


class ChatMessage(BaseModel):
    role: Role
    content: str


class RAGRequest(BaseModel):
    query: str = Field(..., description="Query to be answered")
    provider: Provider = Field(default=Provider.OPENAI, description="LLM provider")
    model_name: str = Field(default="gpt-6-luna", description="Model name")
    history: list[ChatMessage] = Field(
        default_factory=list, description="Previous conversation turns"
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


class ChatRequest(BaseModel):
    provider: Provider
    models_name: str
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    message: str
