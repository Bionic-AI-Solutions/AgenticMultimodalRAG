import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agentic.query_decomposer import QueryDecomposer

client = TestClient(app)


def test_decompose_api_text():
    resp = client.post(
        "/agent/query/decompose",
        json={"query": "What is the summary?", "app_id": "app1", "user_id": "user1", "modality": "text", "context": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "plan" in data
    assert data["traceability"] is True
    assert len(data["plan"]) == 2
    assert data["plan"][0]["type"] == "vector_search"
    assert data["plan"][0]["modality"] == "text"
    assert data["plan"][1]["type"] == "graph_query"
    assert data["plan"][1]["dependencies"] == [1]


def test_decompose_api_audio():
    resp = client.post(
        "/agent/query/decompose",
        json={
            "query": "Summarize the audio.",
            "app_id": "app1",
            "user_id": "user1",
            "modality": "audio",
            "context": {"file": "audio.mp3"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "plan" in data
    assert data["traceability"] is True
    assert len(data["plan"]) == 3
    assert data["plan"][0]["type"] == "audio_transcription"
    assert data["plan"][1]["type"] == "vector_search"
    assert data["plan"][1]["dependencies"] == [1]
    assert data["plan"][2]["type"] == "graph_query"
    assert data["plan"][2]["dependencies"] == [2]


def test_decompose_api_llm_mock(monkeypatch):
    # Patch the QueryDecomposer in the API to use llm_backend='mock', use_llm=True
    from app.api import agentic

    agentic.decomposer = QueryDecomposer(llm_backend="mock", use_llm=True)
    resp = client.post(
        "/agent/query/decompose",
        json={
            "query": "Summarize the main findings from the attached PDF and find related images in the knowledge base.",
            "app_id": "myapp",
            "user_id": "user1",
            "modality": "multimodal",
            "context": {},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "plan" in data
    assert data["traceability"] is True
    assert len(data["plan"]) == 2
    assert data["plan"][0]["type"] == "vector_search"
    assert data["plan"][1]["type"] == "graph_query"


def test_decompose_api_llm_ollama(monkeypatch):
    from app.api import agentic

    agentic.decomposer = QueryDecomposer(llm_backend="local", use_llm=True)
    resp = client.post(
        "/agent/query/decompose",
        json={
            "query": "Summarize the main findings from the attached PDF and find related images in the knowledge base.",
            "app_id": "myapp",
            "user_id": "user1",
            "modality": "multimodal",
            "context": {},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "plan" in data
    assert data["traceability"] is True
    assert isinstance(data["plan"], list)
    assert len(data["plan"]) >= 1
