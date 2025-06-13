import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_decompose_api_text():
    resp = client.post("/agent/query/decompose", json={
        "query": "What is the summary?",
        "app_id": "app1",
        "user_id": "user1",
        "modality": "text",
        "context": {}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "plan" in data
    assert data["traceability"] is True
    assert len(data["plan"]) == 1
    assert data["plan"][0]["type"] == "vector_search"
    assert data["plan"][0]["modality"] == "text"

def test_decompose_api_audio():
    resp = client.post("/agent/query/decompose", json={
        "query": "Summarize the audio.",
        "app_id": "app1",
        "user_id": "user1",
        "modality": "audio",
        "context": {"file": "audio.mp3"}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "plan" in data
    assert data["traceability"] is True
    assert len(data["plan"]) == 2
    assert data["plan"][0]["type"] == "audio_transcription"
    assert data["plan"][1]["type"] == "vector_search"
    assert data["plan"][1]["dependencies"] == [1] 