import pytest
from unittest.mock import patch, MagicMock
from app.agentic.agent_executor import AgentExecutor
from app.agentic.models import DecompositionPlan, DecompositionStep

@pytest.fixture
def simple_plan():
    return DecompositionPlan(plan=[
        DecompositionStep(step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}),
        DecompositionStep(step_id=2, type="graph_query", modality="text", parameters={"related_to": "foo"}, dependencies=[1], trace={}),
        DecompositionStep(step_id=3, type="audio_transcription", modality="audio", parameters={"file": "audio.mp3"}, dependencies=[], trace={}),
        DecompositionStep(step_id=4, type="tool_call", modality="text", parameters={"tool": "search"}, dependencies=[], trace={}),
    ], traceability=True)

def test_agent_executor_simple_plan(simple_plan):
    executor = AgentExecutor(base_url="http://mockserver")
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"mock": "result"})
        result = executor.execute_plan(simple_plan, app_id="app1", user_id="user1")
    assert "final_result" in result
    assert "trace" in result
    assert len(result["trace"]) == 4
    assert result["trace"][0]["type"] == "vector_search"
    assert result["trace"][1]["type"] == "graph_query"
    assert result["trace"][2]["type"] == "audio_transcription"
    assert result["trace"][3]["type"] == "tool_call"
    # Check that mocked API calls were made for vector_search and graph_query
    assert mock_post.call_count == 2
    # Check that audio_transcription and tool_call are simulated
    assert result["trace"][2]["result"]["transcription"] == "[transcribed text]"
    assert result["trace"][3]["result"]["tool_result"] == "[tool output]" 