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

@pytest.mark.parametrize('filename,expected_mime', [
    ('sample.jpg', 'image/jpeg'),
    ('sample.mp3', 'audio/mpeg'),
    ('sample.pdf', 'application/pdf'),
    ('sample.mp4', 'video/mp4'),
])
def test_query_vector_multimodal(filename, expected_mime):
    path = os.path.join('samples', filename)
    if not os.path.exists(path):
        pytest.skip(f"Sample file {filename} not found")
    with open(path, 'rb') as f:
        response = client.post(
            '/query/vector',
            files={'file': (filename, f, expected_mime)},
            data={'app_id': 'testapp', 'user_id': 'testuser'}
        )
    assert response.status_code == 200
    data = response.json()
    assert 'results' in data
    # For video, results may be empty or placeholder
    if filename == 'sample.mp4':
        assert isinstance(data['results'], list)

@pytest.mark.integration
def test_query_graph_text_context():
    req = {
        "query": "sample",
        "app_id": "test",
        "user_id": "test",
        "graph_expansion": {"depth": 1, "type": "context"}
    }
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    for r in data["results"]:
        assert "graph_context" in r
        assert "nodes" in r["graph_context"]
        assert "edges" in r["graph_context"]

@pytest.mark.integration
def test_query_graph_image_semantic():
    path = os.path.join('samples', 'sample.jpg')
    if not os.path.exists(path):
        pytest.skip("Sample image not found")
    with open(path, 'rb') as f:
        resp = client.post(
            '/query/graph',
            files={'file': ('sample.jpg', f, 'image/jpeg')},
            data={'app_id': 'test', 'user_id': 'test', 'graph_expansion': '{"depth": 2, "type": "semantic"}'}
        )
    assert resp.status_code == 200
    data = resp.json()
    for r in data["results"]:
        assert "graph_context" in r
        assert "nodes" in r["graph_context"]
        assert "edges" in r["graph_context"]

@pytest.mark.integration
def test_query_graph_weighted_expansion_explicit():
    req = {
        "query": "sample",
        "app_id": "test",
        "user_id": "test",
        "graph_expansion": {
            "depth": 1,
            "type": "context",
            "weights": {"context_of": 2.0, "about_topic": 0.0, "temporal_neighbor": 1.0}
        }
    }
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "explain" in data
    explain = data["explain"]
    # Only edge types with weight > 0 should be used
    used = explain["used_edge_types"]
    assert "context_of" in used and used["context_of"] == 2.0
    assert "about_topic" in used and used["about_topic"] == 0.0
    assert "temporal_neighbor" in used and used["temporal_neighbor"] == 1.0
    # Rerank info present
    assert "rerank" in explain
    # Edges in results should only be of allowed types
    for r in data["results"]:
        for e in r["graph_context"]["edges"]:
            assert e["type"] in [k for k, v in used.items() if v > 0]

@pytest.mark.integration
def test_query_graph_weighted_expansion_config():
    # This test assumes config/edge_graph.yaml has weights for test app or default
    req = {
        "query": "sample",
        "app_id": "test",
        "user_id": "test",
        "graph_expansion": {"depth": 1, "type": "context"}
    }
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "explain" in data
    explain = data["explain"]
    used = explain["used_edge_types"]
    # Should match config (at least keys/values)
    assert isinstance(used, dict)
    assert all(isinstance(v, float) for v in used.values())
    # Rerank info present
    assert "rerank" in explain
    # Edges in results should only be of allowed types
    for r in data["results"]:
        for e in r["graph_context"]["edges"]:
            assert e["type"] in [k for k, v in used.items() if v > 0]

@pytest.mark.integration
def test_ingest_sample2_pdf():
    path = os.path.join('samples', 'sample2.pdf')
    if not os.path.exists(path):
        pytest.skip("sample2.pdf not found")
    with open(path, 'rb') as f:
        response = client.post(
            '/docs/ingest',
            files={'file': ('sample2.pdf', f, 'application/pdf')},
            data={'app_id': 'testapp', 'user_id': 'testuser'}
        )
    assert response.status_code == 200
    data = response.json()
    assert data.get('status') == 'embedded'
    assert 'doc_id' in data
    assert 'embedding complete' in data.get('message', '').lower() 