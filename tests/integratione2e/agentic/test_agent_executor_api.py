import pytest
from app.agentic.agent_executor import AgentExecutor
from app.agentic.models import DecompositionPlan, DecompositionStep
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.integration
def test_agent_executor_multistep_live():
    # This test assumes the FastAPI app is running at localhost:8000
    plan = DecompositionPlan(plan=[
        DecompositionStep(step_id=1, type="audio_transcription", modality="audio", parameters={"file": "audio.mp3"}, dependencies=[], trace={}),
        DecompositionStep(step_id=2, type="vector_search", modality="text", parameters={"query": "transcription from step 1"}, dependencies=[1], trace={}),
        DecompositionStep(step_id=3, type="graph_query", modality="text", parameters={"related_to": "topics from step 2"}, dependencies=[2], trace={}),
        DecompositionStep(step_id=4, type="tool_call", modality="text", parameters={"tool": "search"}, dependencies=[], trace={}),
    ], traceability=True)
    executor = None  # Not used in this test
    result = client.post("/agent/execute", json={**plan.dict(), "app_id": "app1", "user_id": "user1"})
    assert result.status_code == 200
    data = result.json()
    assert "final_result" in data
    assert "trace" in data
    assert len(data["trace"]) == 4
    assert data["trace"][0]["type"] == "audio_transcription"
    assert data["trace"][1]["type"] == "vector_search"
    assert data["trace"][2]["type"] == "graph_query"
    assert data["trace"][3]["type"] == "tool_call"
    # Check that audio_transcription and tool_call are simulated
    assert data["trace"][0]["result"]["transcription"] == "[transcribed text]"
    assert data["trace"][3]["result"]["tool_result"] == "[tool output]"
    # Check that vector_search and graph_query return dicts (from API)
    assert isinstance(data["trace"][1]["result"], dict)
    assert isinstance(data["trace"][2]["result"], dict) 