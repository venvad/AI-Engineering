import asyncio

from langsmith import Client
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from api.agents.retrieval_generation import rag_pipeline

EVAL_JUDGE_MODEL = "gpt-6-luna"
EVAL_JUDGE_REASONING_EFFORT = "low"
EVAL_EMBEDDING_MODEL = "text-embedding-3-small"

ls_client = Client()
async_openai_client = AsyncOpenAI()
ragas_llm = llm_factory(
    EVAL_JUDGE_MODEL,
    client=async_openai_client,
    max_tokens=4096,
    reasoning_effort=EVAL_JUDGE_REASONING_EFFORT,
)
ragas_embeddings = OpenAIEmbeddings(
    client=async_openai_client, model=EVAL_EMBEDDING_MODEL
)


def context_precision_id_based(run: Run, example: Example | None) -> EvaluationResult:
    assert (
        run.outputs is not None and example is not None and example.outputs is not None
    )
    retrieved = set(map(str, run.outputs["retrieved_context_ids"]))
    reference = set(map(str, example.outputs["reference_context_ids"]))

    if not retrieved:
        return EvaluationResult(
            key="ragas_context_precision_id_based",
            score=None,
            comment="No retrieved context IDs",
        )
    return EvaluationResult(
        key="ragas_context_precision_id_based",
        score=len(retrieved & reference) / len(retrieved),
    )


def context_recall_id_based(run: Run, example: Example | None) -> EvaluationResult:
    assert (
        run.outputs is not None and example is not None and example.outputs is not None
    )
    retrieved = set(map(str, run.outputs["retrieved_context_ids"]))
    reference = set(map(str, example.outputs["reference_context_ids"]))

    if not reference:
        return EvaluationResult(
            key="ragas_context_recall_id_based",
            score=None,
            comment="No reference context IDs",
        )
    return EvaluationResult(
        key="ragas_context_recall_id_based",
        score=len(retrieved & reference) / len(reference),
    )


async def ragas_faithfulness(run: Run, example: Example | None) -> EvaluationResult:
    assert run.outputs is not None
    scorer = Faithfulness(llm=ragas_llm)
    result = await scorer.ascore(
        user_input=run.outputs["query"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    return EvaluationResult(
        key="ragas_faithfulness",
        score=result.value,
    )


async def ragas_relevancy(run: Run, example: Example | None) -> EvaluationResult:
    assert run.outputs is not None
    scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    result = await scorer.ascore(
        user_input=run.outputs["query"],
        response=run.outputs["answer"],
    )
    return EvaluationResult(
        key="ragas_relevancy",
        score=result.value,
    )


async def target(inputs: dict) -> dict:
    result = await asyncio.to_thread(rag_pipeline, inputs["query"])
    return result.model_dump()


async def main() -> None:
    try:
        await ls_client.aevaluate(
            target,
            data="rag-evaluation-dataset",
            evaluators=[
                context_precision_id_based,
                context_recall_id_based,
                ragas_faithfulness,
                ragas_relevancy,
            ],
            experiment_prefix="retriever",
            max_concurrency=5,
        )
    finally:
        await async_openai_client.close()


if __name__ == "__main__":
    asyncio.run(main())
