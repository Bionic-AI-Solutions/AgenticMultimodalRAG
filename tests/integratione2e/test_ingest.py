import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLES = [
    ('sample.txt', 'text/plain'),
    ('sample.pdf', 'application/pdf'),
    ('sample.jpg', 'image/jpeg'),
    ('sample.mp3', 'audio/mpeg'),
    ('sample.csv', 'text/csv'),
    ('sample.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
    ('sample.mp4', 'video/mp4'),
]

@pytest.mark.parametrize('filename,expected_mime', SAMPLES)
def test_ingest_sample_file(filename, expected_mime):
    path = os.path.join('samples', filename)
    if not os.path.exists(path):
        pytest.skip(f"Sample file {filename} not found")
    with open(path, 'rb') as f:
        response = client.post(
            '/docs/ingest',
            files={'file': (filename, f, expected_mime)},
            data={'app_id': 'testapp', 'user_id': 'testuser'}
        )
    assert response.status_code in (200, 422, 415, 413)  # Acceptable: success or validation error
    data = response.json()
    assert 'status' in data

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert 'services' in data

@pytest.mark.integration
def test_query_vector_integration():
    # This test assumes Milvus is running and at least one document is ingested for app_id=user_id='test'
    req = {
        "query": "sample",
        "app_id": "test",
        "user_id": "test",
        "top_k": 3,
        "filters": {"doc_type": "txt"}
    }
    resp = client.post("/query/vector", json=req)
    assert resp.status_code == 200
    data = resp.json()
    # Should not error, may return 0 or more results
    assert "results" in data
    # Optionally check structure of results
    for r in data["results"]:
        assert "doc_id" in r
        assert "score" in r
        assert "content" in r
        assert "metadata" in r 