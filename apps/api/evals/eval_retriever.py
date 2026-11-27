import asyncio

from api.agents.retrieval_generation import rag_pipeline
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

ls_client = Client()
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-6-sol"), bypass_temperature=True)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)


async def ragas_context_precision_id_based(
    run: Run, example: Example | None
) -> EvaluationResult:
    assert (
        run.outputs is not None
        and example is not None
        and example.outputs is not None
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
        run.outputs is not None
        and example is not None
        and example.outputs is not None
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
    return await asyncio.to_thread(lambda: rag_pipeline(inputs["query"]).model_dump())


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
