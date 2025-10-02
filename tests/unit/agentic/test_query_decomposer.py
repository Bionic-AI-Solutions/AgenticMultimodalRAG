import pytest
from app.agentic.query_decomposer import QueryDecomposer
from app.agentic.models import DecompositionPlan


def test_decompose_text_query():
    decomposer = QueryDecomposer()
    plan = decomposer.decompose(query="What is the summary?", app_id="app1", user_id="user1", modality="text", context={})
    assert isinstance(plan, DecompositionPlan)
    assert len(plan.plan) == 2
    assert plan.plan[0].type == "vector_search"
    assert plan.plan[0].modality == "text"
    assert plan.plan[1].type == "graph_query"
    assert plan.plan[1].dependencies == [1]
    assert plan.traceability is True


def test_decompose_audio_query():
    decomposer = QueryDecomposer()
    plan = decomposer.decompose(
        query="Summarize the audio.", app_id="app1", user_id="user1", modality="audio", context={"file": "audio.mp3"}
    )
    assert isinstance(plan, DecompositionPlan)
    assert len(plan.plan) == 3
    assert plan.plan[0].type == "audio_transcription"
    assert plan.plan[1].type == "vector_search"
    assert plan.plan[1].dependencies == [1]
    assert plan.plan[2].type == "graph_query"
    assert plan.plan[2].dependencies == [2]
    assert plan.traceability is True


def test_decompose_llm_mock():
    decomposer = QueryDecomposer(llm_backend="mock", use_llm=True)
    plan = decomposer.decompose(
        query="Summarize the main findings from the attached PDF and find related images in the knowledge base.",
        app_id="myapp",
        user_id="user1",
        modality="multimodal",
        context={},
    )
    assert isinstance(plan, DecompositionPlan)
    assert len(plan.plan) == 2
    assert plan.plan[0].type == "vector_search"
    assert plan.plan[1].type == "graph_query"
    assert plan.traceability is True
