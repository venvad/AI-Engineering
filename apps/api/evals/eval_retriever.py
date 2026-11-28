import asyncio

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    IDBasedContextPrecision,
    IDBasedContextRecall,
    ResponseRelevancy,
)

from api.agents.retrieval_generation import rag_pipeline

EVAL_JUDGE_MODEL = "gpt-6-sol"
EVAL_EMBEDDING_MODEL = "text-embedding-3-small"

ls_client = Client()
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model=EVAL_JUDGE_MODEL), bypass_temperature=True
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model=EVAL_EMBEDDING_MODEL)
)


async def ragas_context_precision_id_based(
    run: Run, example: Example | None
) -> EvaluationResult:
    assert (
        run.outputs is not None and example is not None and example.outputs is not None
    )
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"],
    )
    scorer = IDBasedContextPrecision()
    return EvaluationResult(
        key="ragas_context_precision_id_based",
        score=await scorer.single_turn_ascore(sample),
    )


async def ragas_context_recall_id_based(
    run: Run, example: Example | None
) -> EvaluationResult:
    assert (
        run.outputs is not None and example is not None and example.outputs is not None
    )
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"],
    )
    scorer = IDBasedContextRecall()
    return EvaluationResult(
        key="ragas_context_recall_id_based",
        score=await scorer.single_turn_ascore(sample),
    )


async def ragas_faithfulness(run: Run, example: Example | None) -> EvaluationResult:
    assert run.outputs is not None
    sample = SingleTurnSample(
        user_input=run.outputs["query"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    scorer = Faithfulness(llm=ragas_llm)
    return EvaluationResult(
        key="ragas_faithfulness",
        score=await scorer.single_turn_ascore(sample),
    )


async def ragas_relevancy(run: Run, example: Example | None) -> EvaluationResult:
    assert run.outputs is not None
    sample = SingleTurnSample(
        user_input=run.outputs["query"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return EvaluationResult(
        key="ragas_relevancy",
        score=await scorer.single_turn_ascore(sample),
    )


async def target(inputs: dict) -> dict:
    result = await asyncio.to_thread(rag_pipeline, inputs["query"])
    return result.model_dump()


async def main() -> None:
    await ls_client.aevaluate(
        target,
        data="rag-evaluation-dataset",
        evaluators=[
            ragas_context_precision_id_based,
            ragas_context_recall_id_based,
            ragas_faithfulness,
            ragas_relevancy,
        ],
        experiment_prefix="retriever",
    )


if __name__ == "__main__":
    asyncio.run(main())
