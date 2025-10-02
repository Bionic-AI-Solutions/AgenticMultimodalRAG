import pytest
from app.agentic.response_synthesizer import ResponseSynthesizer, ResponseSynthesisRequest


class MockLLMClient:
    def generate(self, prompt):
        return f"LLM: {prompt[:40]}..."


def sample_trace():
    return [
        {"step_id": 1, "type": "vector_search", "result": {"results": [{"doc_id": "a", "score": 0.9}]}},
        {"step_id": 2, "type": "filter", "result": {"results": [{"doc_id": "a", "score": 0.9}], "filter_method": "min_score"}},
        {"step_id": 3, "type": "llm_call", "result": {"llm_call": True, "synthesized": "summary"}},
    ]


def test_synthesize_answer_template():
    synthesizer = ResponseSynthesizer()
    req = ResponseSynthesisRequest(plan=None, execution_trace=sample_trace(), app_id="app1", user_id="user1")
    result = synthesizer.synthesize_answer(req)
    assert "Based on the results" in result.answer
    assert "Step 1" in result.explanation
    assert result.supporting_evidence is not None
    assert result.trace == sample_trace()


def test_synthesize_answer_llm():
    synthesizer = ResponseSynthesizer(llm_client=MockLLMClient())
    req = ResponseSynthesisRequest(plan=None, execution_trace=sample_trace(), app_id="app1", user_id="user1")
    result = synthesizer.synthesize_answer(req)
    assert result.answer.startswith("LLM:")
    assert "Step 1" in result.explanation
    assert result.supporting_evidence is not None


def test_generate_explanation():
    synthesizer = ResponseSynthesizer()
    trace = [
        {"step_id": 1, "type": "vector_search", "result": {"results": [1, 2, 3]}},
        {"step_id": 2, "type": "filter", "result": {"results": [1, 2]}},
        {"step_id": 3, "type": "rerank", "result": {"results": [2, 1]}},
    ]
    explanation = synthesizer.generate_explanation(trace)
    assert "Step 1" in explanation and "Step 2" in explanation and "Step 3" in explanation


def test_generate_explanation_skipped():
    synthesizer = ResponseSynthesizer()
    trace = [
        {"step_id": 1, "type": "vector_search", "result": {"results": [1, 2, 3]}},
        {"step_id": 2, "type": "filter", "skipped": True, "condition": "step_1.result == []"},
    ]
    explanation = synthesizer.generate_explanation(trace)
    assert "skipped" in explanation
    assert "condition" in explanation


def test_synthesize_answer_no_final_result():
    synthesizer = ResponseSynthesizer()
    req = ResponseSynthesisRequest(
        plan=None, execution_trace=[{"step_id": 1, "type": "vector_search", "result": None}], app_id="app1", user_id="user1"
    )
    result = synthesizer.synthesize_answer(req)
    assert "No answer could be synthesized" in result.answer
    assert result.supporting_evidence is None
