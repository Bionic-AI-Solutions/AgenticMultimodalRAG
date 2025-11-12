import pytest
from app.agentic.agent_executor import AgentExecutor
from app.agentic.models import DecompositionPlan, DecompositionStep
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.mark.integration
def test_agent_executor_multistep_live():
    # This test assumes the FastAPI app is running at localhost:8000
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="audio_transcription",
                modality="audio",
                parameters={"file": "audio.mp3"},
                dependencies=[],
                trace={},
            ),
            DecompositionStep(
                step_id=2,
                type="vector_search",
                modality="text",
                parameters={"query": "transcription from step 1"},
                dependencies=[1],
                trace={},
            ),
            DecompositionStep(
                step_id=3,
                type="graph_query",
                modality="text",
                parameters={"related_to": "topics from step 2"},
                dependencies=[2],
                trace={},
            ),
            DecompositionStep(
                step_id=4, type="tool_call", modality="text", parameters={"tool": "search"}, dependencies=[], trace={}
            ),
        ],
        traceability=True,
    )
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


@pytest.mark.integration
def test_mcp_tools_list_endpoint():
    """Test the /agent/tools/list endpoint"""
    response = client.get("/agent/tools/list")
    # If fastapi-mcp is not installed, this might fail gracefully
    if response.status_code == 500:
        # Check if it's because MCP is not available
        error_data = response.json()
        if "mcp" in error_data.get("message", "").lower() or "not installed" in error_data.get("message", "").lower():
            pytest.skip("FastAPI MCP not installed or not available")
        else:
            pytest.fail(f"Unexpected error: {error_data}")
    
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert "count" in data
    assert isinstance(data["tools"], list)
    # If tools are available, verify structure
    if len(data["tools"]) > 0:
        tool = data["tools"][0]
        assert "name" in tool or "tool" in tool  # Different MCP formats


@pytest.mark.integration
def test_fastapi_mcp_tool_call_in_plan():
    """Test using FastAPI MCP tools in an agentic plan"""
    # First check if MCP is available
    tools_response = client.get("/agent/tools/list")
    if tools_response.status_code != 200:
        pytest.skip("FastAPI MCP not available")
    
    # Use a simple tool like health_check if available
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="tool_call",
                modality="text",
                parameters={
                    "tool": "fastapi_mcp",
                    "tool_name": "health_check",
                    "arguments": {}
                },
                dependencies=[],
                trace={},
            )
        ],
        traceability=True,
    )
    result = client.post("/agent/execute", json={**plan.dict(), "app_id": "app1", "user_id": "user1"})
    assert result.status_code == 200
    data = result.json()
    assert "final_result" in data
    assert "trace" in data
    # The result might be an error if the tool doesn't exist, but the execution should complete
    assert len(data["trace"]) == 1
    assert data["trace"][0]["type"] == "tool_call"


@pytest.mark.integration
def test_fastapi_mcp_tool_call_with_vector_search():
    """Test using FastAPI MCP tool call for vector search"""
    # First check if MCP is available
    tools_response = client.get("/agent/tools/list")
    if tools_response.status_code != 200:
        pytest.skip("FastAPI MCP not available")
    
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="tool_call",
                modality="text",
                parameters={
                    "tool": "fastapi_mcp",
                    "tool_name": "query_vector",
                    "arguments": {
                        "query": "test query",
                        "app_id": "testapp",
                        "user_id": "testuser",
                        "top_k": 5
                    }
                },
                dependencies=[],
                trace={},
            )
        ],
        traceability=True,
    )
    result = client.post("/agent/execute", json={**plan.dict(), "app_id": "testapp", "user_id": "testuser"})
    assert result.status_code == 200
    data = result.json()
    assert "final_result" in data
    assert "trace" in data
    # Even if the query returns no results, the tool call should complete
    assert len(data["trace"]) == 1
