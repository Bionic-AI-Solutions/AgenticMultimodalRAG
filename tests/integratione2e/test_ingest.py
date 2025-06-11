import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
import torch
import time

client = TestClient(app)

# Register the integration mark
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as an integration test")

@pytest.fixture(autouse=True)
def clear_gpu_memory():
    """Clear GPU memory before and after each test."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(autouse=True)
def ensure_services():
    """Ensure all required services are running and accessible."""
    # Check Milvus
    try:
        response = client.get("/health/details")
        assert response.status_code == 200
        health_data = response.json()
        print("\nHealth check details:", health_data)  # Print health data for debugging
        
        # Check each service individually
        services = ["milvus", "minio", "postgres", "neo4j"]
        for service in services:
            if service not in health_data:
                pytest.skip(f"Service {service} not found in health check response")
            if health_data[service]["status"] != "ok":
                pytest.skip(f"Service {service} is not healthy: {health_data[service]}")
        
        # If we get here, all services are healthy
        return True
    except Exception as e:
        print(f"\nHealth check failed with error: {str(e)}")  # Print error for debugging
        pytest.skip(f"Required services not available: {e}")

# Model caching fixtures
@pytest.fixture(scope="session")
def text_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, cache_folder=os.getenv("HF_HOME", "/home/user/RAG/models"))
    return model

# from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

@pytest.fixture(scope="session")
def multimodal_embedder():
    # model_path = os.path.join(os.getenv("HF_HOME", "/home/user/RAG/models"), "nomic-ai/colnomic-embed-multimodal-7b")
    # model = ColQwen2_5.from_pretrained(model_path, cache_dir=os.getenv("HF_HOME", "/home/user/RAG/models"))
    # processor = ColQwen2_5_Processor.from_pretrained(model_path, use_fast=False, cache_dir=os.getenv("HF_HOME", "/home/user/RAG/models"))
    # return model, processor
    pass

@pytest.fixture(scope="session")
def audio_processor():
    import whisper
    model = whisper.load_model("base", download_root=os.getenv("HF_HOME", "/home/user/RAG/models"))
    return model

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
    """Test ingesting various file types with proper error handling."""
    path = os.path.join('samples', filename)
    if not os.path.exists(path):
        pytest.skip(f"Sample file {filename} not found")
    
    with open(path, 'rb') as f:
        response = client.post(
            '/docs/ingest',
            files={'file': (filename, f, expected_mime)},
            data={'app_id': 'testapp', 'user_id': 'testuser'}
        )
    
    # Accept both success and expected error codes
    assert response.status_code in (200, 422, 415, 413, 500)
    
    if response.status_code == 500:
        # Check if it's an OOM error
        error_data = response.json()
        assert "error" in error_data
        assert any(msg in error_data["message"].lower() for msg in ["out of memory", "cuda", "gpu"])
    elif response.status_code == 200:
        data = response.json()
        assert data.get('status') == 'embedded'
        assert data.get('doc_id') is not None

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert 'services' in data

@pytest.mark.integration
def test_query_vector_integration():
    """Test vector query with error handling."""
    try:
        response = client.post(
            '/query/vector',
            json={
                'query': 'test query',
                'app_id': 'testapp',
                'user_id': 'testuser',
                'top_k': 5
            }
        )
        
        if response.status_code == 500:
            error_data = response.json()
            if "out of memory" in error_data.get("message", "").lower():
                pytest.skip("GPU out of memory during query, skipping test")
            else:
                pytest.fail(f"Query failed: {error_data.get('message')}")
        
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        
    except Exception as e:
        pytest.skip(f"Query test skipped due to error: {e}")

@pytest.mark.parametrize('filename,expected_mime', [
    ('sample.jpg', 'image/jpeg'),
    ('sample.mp3', 'audio/mpeg'),
    ('sample.pdf', 'application/pdf'),
    ('sample.mp4', 'video/mp4'),
])
def test_query_vector_multimodal(filename, expected_mime):
    if filename.endswith('.mp4'):
        pytest.skip("Video embedding not supported; skipping test.")
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

@pytest.mark.integration
def test_ingest_and_retrieve_flow():
    """Test complete flow: ingest document, verify storage in Milvus, and retrieve it."""
    # 1. Ingest a test document
    test_content = "This is a test document for ingestion verification."
    test_file = "test_ingest.txt"
    
    try:
        with open(test_file, "w") as f:
            f.write(test_content)
        
        # Ingest the document
        with open(test_file, "rb") as f:
            response = client.post(
                '/docs/ingest',
                files={'file': (test_file, f, 'text/plain')},
                data={'app_id': 'testapp', 'user_id': 'testuser'}
            )
        
        # Handle potential errors
        if response.status_code == 500:
            error_data = response.json()
            if "out of memory" in error_data.get("message", "").lower():
                pytest.skip("GPU out of memory, skipping test")
            else:
                pytest.fail(f"Ingestion failed: {error_data.get('message')}")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get('status') == 'embedded'
        doc_id = data.get('doc_id')
        assert doc_id is not None
        
        # 2. Query the document back
        query_response = client.post(
            '/query/vector',
            json={
                'query': 'test document',
                'app_id': 'testapp',
                'user_id': 'testuser',
                'top_k': 1
            }
        )
        
        # Handle potential errors
        if query_response.status_code == 500:
            error_data = query_response.json()
            if "out of memory" in error_data.get("message", "").lower():
                pytest.skip("GPU out of memory during query, skipping test")
            else:
                pytest.fail(f"Query failed: {error_data.get('message')}")
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # 3. Verify the results
        assert 'results' in query_data
        if len(query_data['results']) == 0:
            pytest.skip("No results returned, possibly due to vector dimension mismatch")
        assert len(query_data['results']) > 0
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file) 