import pytest
from unittest.mock import patch, MagicMock
from app.agentic.agent_executor import AgentExecutor
from app.agentic.models import DecompositionPlan, DecompositionStep


@pytest.fixture
def simple_plan():
    return DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2, type="graph_query", modality="text", parameters={"related_to": "foo"}, dependencies=[1], trace={}
            ),
            DecompositionStep(
                step_id=3,
                type="audio_transcription",
                modality="audio",
                parameters={"file": "audio.mp3"},
                dependencies=[],
                trace={},
            ),
            DecompositionStep(
                step_id=4, type="tool_call", modality="text", parameters={"tool": "search"}, dependencies=[], trace={}
            ),
        ],
        traceability=True,
    )


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


# --- MCP tool_call test ---
def test_agent_executor_mcp_tool_call():
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="tool_call",
                modality="text",
                parameters={
                    "tool": "mcp",
                    "endpoint": "https://mcp.example.com/api/tool",
                    "payload": {"query": "What is the weather in Paris?"},
                    "headers": {"Authorization": "Bearer testtoken"},
                },
                dependencies=[],
                trace={},
            )
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"answer": "It is sunny in Paris."}, status_code=200, raise_for_status=lambda: None
        )
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert "final_result" in result
    assert result["final_result"] == {"answer": "It is sunny in Paris."}
    assert result["trace"][0]["type"] == "tool_call"
    assert result["trace"][0]["result"] == {"answer": "It is sunny in Paris."}
    # Test error handling
    with patch("app.agentic.agent_executor.requests.post", side_effect=Exception("MCP error")):
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert "error" in result["final_result"]
    assert "MCP tool_call failed" in result["final_result"]["error"]


# --- Conditional step test ---
def test_agent_executor_conditional_step():
    # Step 2 only runs if step_1.result["run"] == True
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="tool_call", modality="text", parameters={"tool": "search"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2,
                type="tool_call",
                modality="text",
                parameters={"tool": "search"},
                dependencies=[1],
                trace={},
                condition="step_1.result['run'] == True",
            ),
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    # Case 1: step_1 returns {"run": True} -> step_2 runs
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"run": True}, status_code=200, raise_for_status=lambda: None)
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert len(result["trace"]) == 2
    assert result["trace"][1]["type"] == "tool_call"
    # Case 2: step_1 returns {"run": False} -> step_2 skipped
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"run": False}, status_code=200, raise_for_status=lambda: None)
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert len(result["trace"]) == 2
    assert result["trace"][1]["skipped"] is True
    assert result["trace"][1]["condition"] == "step_1.result['run'] == True"


def test_agent_executor_rerank_step():
    # Plan: vector_search -> rerank
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2, type="rerank", modality="text", parameters={"input_step": 1}, dependencies=[1], trace={}
            ),
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    # Mock vector_search to return results
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {
                "results": [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.8}, {"doc_id": "c", "score": 0.7}]
            },
            status_code=200,
            raise_for_status=lambda: None,
        )
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert len(result["trace"]) == 2
    rerank_result = result["trace"][1]["result"]
    assert rerank_result["rerank_method"].startswith("reverse_order")
    assert [r["doc_id"] for r in rerank_result["results"]] == ["c", "b", "a"]
    # Test error: missing input step
    plan2 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="rerank", modality="text", parameters={"input_step": 99}, dependencies=[], trace={}
            ),
        ],
        traceability=True,
    )
    result2 = executor.execute_plan(plan2, app_id="app1", user_id="user1")
    assert "error" in result2["final_result"]
    # Test error: no results to rerank
    plan3 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2, type="rerank", modality="text", parameters={"input_step": 1}, dependencies=[1], trace={}
            ),
        ],
        traceability=True,
    )
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"noresults": []}, status_code=200, raise_for_status=lambda: None)
        result3 = executor.execute_plan(plan3, app_id="app1", user_id="user1")
    assert "error" in result3["trace"][1]["result"]


def test_agent_executor_filter_step():
    # Plan: vector_search -> filter
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2,
                type="filter",
                modality="text",
                parameters={"input_step": 1, "min_score": 0.8, "metadata": {"label": "important"}},
                dependencies=[1],
                trace={},
            ),
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    # Mock vector_search to return results
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {
                "results": [
                    {"doc_id": "a", "score": 0.85, "content": "...", "metadata": {"label": "important"}},
                    {"doc_id": "b", "score": 0.7, "content": "...", "metadata": {"label": "important"}},
                    {"doc_id": "c", "score": 0.9, "content": "...", "metadata": {"label": "other"}},
                ]
            },
            status_code=200,
            raise_for_status=lambda: None,
        )
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert len(result["trace"]) == 2
    filter_result = result["trace"][1]["result"]
    # Only doc_id 'a' should pass both min_score and metadata filter
    assert filter_result["results"] == [{"doc_id": "a", "score": 0.85, "content": "...", "metadata": {"label": "important"}}]
    assert filter_result["filter_method"].startswith("min_score")

    # Test error: missing input step
    plan2 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="filter",
                modality="text",
                parameters={"input_step": 99, "min_score": 0.8},
                dependencies=[],
                trace={},
            ),
        ],
        traceability=True,
    )
    result2 = executor.execute_plan(plan2, app_id="app1", user_id="user1")
    assert "error" in result2["final_result"]
    # Test error: no results to filter
    plan3 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2,
                type="filter",
                modality="text",
                parameters={"input_step": 1, "min_score": 0.8},
                dependencies=[1],
                trace={},
            ),
        ],
        traceability=True,
    )
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"noresults": []}, status_code=200, raise_for_status=lambda: None)
        result3 = executor.execute_plan(plan3, app_id="app1", user_id="user1")
    assert "error" in result3["trace"][1]["result"]


def test_agent_executor_aggregate_step():
    # Plan: vector_search, graph_query -> aggregate (union)
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2, type="graph_query", modality="text", parameters={"related_to": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=3,
                type="aggregate",
                modality="text",
                parameters={"input_steps": [1, 2], "method": "union"},
                dependencies=[1, 2],
                trace={},
            ),
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    # Mock both search steps to return results
    with patch("app.agentic.agent_executor.requests.post") as mock_post:

        def side_effect(*args, **kwargs):
            if "/query/vector" in args[0]:
                return MagicMock(
                    json=lambda: {"results": [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.8}]},
                    status_code=200,
                    raise_for_status=lambda: None,
                )
            else:
                return MagicMock(
                    json=lambda: {"results": [{"doc_id": "b", "score": 0.85}, {"doc_id": "c", "score": 0.7}]},
                    status_code=200,
                    raise_for_status=lambda: None,
                )

        mock_post.side_effect = side_effect
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    agg_result = result["trace"][2]["result"]
    # Union: a, b, c (unique by doc_id)
    assert {r["doc_id"] for r in agg_result["results"]} == {"a", "b", "c"}
    assert agg_result["aggregate_method"] == "union"

    # Intersection: only b
    plan2 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2, type="graph_query", modality="text", parameters={"related_to": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=3,
                type="aggregate",
                modality="text",
                parameters={"input_steps": [1, 2], "method": "intersection"},
                dependencies=[1, 2],
                trace={},
            ),
        ],
        traceability=True,
    )
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.side_effect = side_effect
        result2 = executor.execute_plan(plan2, app_id="app1", user_id="user1")
    agg_result2 = result2["trace"][2]["result"]
    # Intersection: only b
    assert {r["doc_id"] for r in agg_result2["results"]} == {"b"}
    assert agg_result2["aggregate_method"] == "intersection"

    # Error: missing input steps
    plan3 = DecompositionPlan(
        plan=[
            DecompositionStep(step_id=1, type="aggregate", modality="text", parameters={}, dependencies=[], trace={}),
        ],
        traceability=True,
    )
    result3 = executor.execute_plan(plan3, app_id="app1", user_id="user1")
    assert "error" in result3["final_result"]
    # Error: unknown method
    plan4 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2,
                type="aggregate",
                modality="text",
                parameters={"input_steps": [1], "method": "bogus"},
                dependencies=[1],
                trace={},
            ),
        ],
        traceability=True,
    )
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"results": [{"doc_id": "a"}]}, status_code=200, raise_for_status=lambda: None
        )
        result4 = executor.execute_plan(plan4, app_id="app1", user_id="user1")
    assert "error" in result4["trace"][1]["result"]
    # Error: no results to aggregate
    plan5 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2, type="aggregate", modality="text", parameters={"input_steps": [1]}, dependencies=[1], trace={}
            ),
        ],
        traceability=True,
    )
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"noresults": []}, status_code=200, raise_for_status=lambda: None)
        result5 = executor.execute_plan(plan5, app_id="app1", user_id="user1")
    assert "error" in result5["trace"][1]["result"]


def test_agent_executor_multihop_step():
    # Plan: graph_query -> multi-hop
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="graph_query", modality="text", parameters={"related_to": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2,
                type="multi-hop",
                modality="text",
                parameters={"input_step": 1, "hops": 3},
                dependencies=[1],
                trace={},
            ),
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    # Mock graph_query and multi-hop graph_query
    with patch("app.agentic.agent_executor.requests.post") as mock_post:

        def side_effect(*args, **kwargs):
            if "/query/graph" in args[0] and "depth" not in kwargs.get("json", {}).get("graph_expansion", {}):
                return MagicMock(
                    json=lambda: {"results": [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.8}]},
                    status_code=200,
                    raise_for_status=lambda: None,
                )
            else:
                return MagicMock(
                    json=lambda: {"results": [{"doc_id": "c", "score": 0.7}, {"doc_id": "d", "score": 0.6}]},
                    status_code=200,
                    raise_for_status=lambda: None,
                )

        mock_post.side_effect = side_effect
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    multihop_result = result["trace"][1]["result"]
    assert multihop_result["multi_hop"] == 3
    assert {r["doc_id"] for r in multihop_result["results"]} == {"c", "d"}

    # Error: missing input step
    plan2 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="multi-hop",
                modality="text",
                parameters={"input_step": 99, "hops": 2},
                dependencies=[],
                trace={},
            ),
        ],
        traceability=True,
    )
    result2 = executor.execute_plan(plan2, app_id="app1", user_id="user1")
    assert "error" in result2["final_result"]
    # Error: no doc_id to multi-hop from
    plan3 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="graph_query", modality="text", parameters={"related_to": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2,
                type="multi-hop",
                modality="text",
                parameters={"input_step": 1, "hops": 2},
                dependencies=[1],
                trace={},
            ),
        ],
        traceability=True,
    )
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(json=lambda: {"results": []}, status_code=200, raise_for_status=lambda: None)
        result3 = executor.execute_plan(plan3, app_id="app1", user_id="user1")
    assert "error" in result3["trace"][1]["result"]


def test_agent_executor_llm_call_step():
    # Plan: vector_search -> llm_call
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1, type="vector_search", modality="text", parameters={"query": "foo"}, dependencies=[], trace={}
            ),
            DecompositionStep(
                step_id=2,
                type="llm_call",
                modality="text",
                parameters={"input_step": 1, "prompt": "Summarize the results."},
                dependencies=[1],
                trace={},
            ),
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    # Mock vector_search to return results
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"results": [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.8}]},
            status_code=200,
            raise_for_status=lambda: None,
        )
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    llm_result = result["trace"][1]["result"]
    assert llm_result["llm_call"] is True
    assert llm_result["synthesized"].startswith("[LLM SYNTHESIS] Summarize the results.")

    # Error: missing input step
    plan2 = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="llm_call",
                modality="text",
                parameters={"input_step": 99, "prompt": "Summarize the results."},
                dependencies=[],
                trace={},
            ),
        ],
        traceability=True,
    )
    result2 = executor.execute_plan(plan2, app_id="app1", user_id="user1")
    assert "error" in result2["final_result"]


# --- FastAPI MCP tool_call tests ---
def test_agent_executor_fastapi_mcp_tool_call():
    """Test FastAPI MCP tool call functionality"""
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
                        "user_id": "testuser"
                    }
                },
                dependencies=[],
                trace={},
            )
        ],
        traceability=True,
    )
    executor = AgentExecutor(base_url="http://mockserver")
    with patch("app.agentic.agent_executor.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"results": [{"doc_id": "test", "score": 0.9, "content": "test", "metadata": {}}]},
            status_code=200,
            raise_for_status=lambda: None
        )
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert "final_result" in result
    assert "trace" in result
    assert result["trace"][0]["type"] == "tool_call"
    # Verify the MCP endpoint was called correctly
    assert mock_post.called
    call_args = mock_post.call_args
    assert "/mcp/tools/call" in call_args[0][0] or "/tools/call" in call_args[0][0]
    assert call_args[1]["json"]["name"] == "query_vector"
    assert call_args[1]["json"]["arguments"]["query"] == "test query"


def test_agent_executor_fastapi_mcp_tool_call_missing_tool_name():
    """Test FastAPI MCP tool call with missing tool_name"""
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="tool_call",
                modality="text",
                parameters={
                    "tool": "fastapi_mcp",
                    "arguments": {"query": "test"}
                },
                dependencies=[],
                trace={},
            )
        ],
        traceability=True,
    )
    executor = AgentExecutor()
    result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert "error" in result["final_result"]
    assert "tool_name required" in result["final_result"]["error"]


def test_agent_executor_fastapi_mcp_tool_call_error():
    """Test FastAPI MCP tool call error handling"""
    plan = DecompositionPlan(
        plan=[
            DecompositionStep(
                step_id=1,
                type="tool_call",
                modality="text",
                parameters={
                    "tool": "fastapi_mcp",
                    "tool_name": "nonexistent_tool",
                    "arguments": {}
                },
                dependencies=[],
                trace={},
            )
        ],
        traceability=True,
    )
    executor = AgentExecutor(base_url="http://mockserver")
    with patch("app.agentic.agent_executor.requests.post", side_effect=Exception("MCP error")):
        result = executor.execute_plan(plan, app_id="app1", user_id="user1")
    assert "error" in result["final_result"]
    assert "FastAPI MCP tool_call failed" in result["final_result"]["error"]


def test_agent_executor_list_mcp_tools():
    """Test listing MCP tools"""
    executor = AgentExecutor(base_url="http://mockserver")
    with patch("app.agentic.agent_executor.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            json=lambda: {"tools": [{"name": "query_vector", "description": "Query vector search"}]},
            status_code=200,
            raise_for_status=lambda: None
        )
        tools = executor.list_mcp_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "query_vector"
    assert tools[0]["description"] == "Query vector search"


def test_agent_executor_list_mcp_tools_list_format():
    """Test listing MCP tools with list format response"""
    executor = AgentExecutor(base_url="http://mockserver")
    with patch("app.agentic.agent_executor.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            json=lambda: [{"name": "query_vector"}, {"name": "ingest_document"}],
            status_code=200,
            raise_for_status=lambda: None
        )
        tools = executor.list_mcp_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "query_vector"
    assert tools[1]["name"] == "ingest_document"


def test_agent_executor_list_mcp_tools_error():
    """Test listing MCP tools error handling"""
    executor = AgentExecutor(base_url="http://mockserver")
    with patch("app.agentic.agent_executor.requests.get", side_effect=Exception("Connection error")):
        tools = executor.list_mcp_tools()
    assert tools == []
