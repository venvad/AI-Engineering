import logging

from fastapi import APIRouter, HTTPException, Request

from api.agents.retrieval_generation import rag_pipeline
from api.api.models import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    try:
        result = rag_pipeline(
            payload.query,
            provider=payload.provider,
            model_name=payload.model_name,
            history=payload.history,
        )
        return RAGResponse(answer=result.answer)
    except Exception as e:
        logger.exception("Request failed: %s %s", request.method, request.url.path)
        raise HTTPException(status_code=500, detail=f"Request failed: {e}") from e


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
