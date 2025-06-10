import pytest
from app.main import detect_file_type, extract_text, extract_pdf, extract_image, extract_audio, extract_content_by_type, chunk_text_recursive
from unittest.mock import patch, MagicMock
from app.main import app, VectorQueryRequest
from fastapi.testclient import TestClient

class TestExtraction:
    def test_detect_file_type_text(self):
        content = b'Hello world!'
        assert detect_file_type('test.txt', content).startswith('text/')

    def test_extract_text_utf8(self):
        content = 'Hello world!'.encode('utf-8')
        assert extract_text(content) == 'Hello world!'

    def test_extract_text_latin1(self):
        content = 'Olá mundo!'.encode('latin1')
        assert 'Olá' in extract_text(content)

    def test_extract_pdf_stub(self):
        content = b'%PDF-1.4...'
        assert 'PDF extraction' in extract_pdf(content)

    def test_extract_image_stub(self):
        content = b'\x89PNG...'
        assert 'OCR' in extract_image(content)

    def test_extract_audio_stub(self):
        content = b'ID3...'
        assert 'ASR' in extract_audio(content)

    def test_extract_content_by_type_text(self):
        content = b'Hello world!'
        assert extract_content_by_type('text/plain', content) == 'Hello world!'

    def test_chunk_text_recursive(self):
        text = ' '.join(['word']*1000)
        chunks = chunk_text_recursive(text, chunk_size=100, overlap=20)
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
        assert len(chunks) > 1 

client = TestClient(app)

@patch("app.main.jina_embedder")
@patch("pymilvus.Collection")
def test_query_vector_basic(mock_collection, mock_embedder):
    mock_embedder.encode.return_value = [[0.1]*768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "doc1", "content": "foo", "metadata": {"created_at": "2024-06-10"}}.get(k, d)
    mock_hit.score = 0.99
    mock_collection.return_value.search.return_value = [[mock_hit]]
    req = {
        "query": "test",
        "app_id": "app1",
        "user_id": "user1",
        "top_k": 1,
        "filters": {"doc_type": "pdf", "created_after": "2024-06-01"}
    }
    resp = client.post("/query/vector", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["doc_id"] == "doc1"
    assert data["results"][0]["metadata"]["created_at"] == "2024-06-10"

@patch("app.main.jina_embedder")
@patch("pymilvus.Collection")
def test_query_vector_error_handling(mock_collection, mock_embedder):
    mock_embedder.encode.side_effect = Exception("fail")
    req = {"query": "fail", "app_id": "a", "user_id": "u"}
    resp = client.post("/query/vector", json=req)
    assert resp.status_code == 200
    assert resp.json()["results"] == [] 

@patch("app.main.jina_embedder")
@patch("pymilvus.Collection")
def test_query_vector_image(mock_collection, mock_embedder):
    mock_embedder.encode.return_value = [[0.1]*768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docimg", "content": "img", "metadata": {}}.get(k, d)
    mock_hit.score = 0.88
    mock_collection.return_value.search.return_value = [[mock_hit]]
    with open("samples/sample.jpg", "rb") as f:
        resp = client.post("/query/vector", files={"file": ("sample.jpg", f, "image/jpeg")}, data={"app_id": "app1", "user_id": "user1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data

@patch("app.main.embed_audio_whisper")
@patch("pymilvus.Collection")
def test_query_vector_audio(mock_collection, mock_embed_audio):
    mock_embed_audio.return_value = [[0.2]*768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docaud", "content": "aud", "metadata": {}}.get(k, d)
    mock_hit.score = 0.77
    mock_collection.return_value.search.return_value = [[mock_hit]]
    with open("samples/sample.mp3", "rb") as f:
        resp = client.post("/query/vector", files={"file": ("sample.mp3", f, "audio/mpeg")}, data={"app_id": "app1", "user_id": "user1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data

@patch("app.main.embed_pdf_nomic")
@patch("pymilvus.Collection")
def test_query_vector_pdf(mock_collection, mock_embed_pdf):
    mock_embed_pdf.return_value = [[0.3]*768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docpdf", "content": "pdf", "metadata": {}}.get(k, d)
    mock_hit.score = 0.66
    mock_collection.return_value.search.return_value = [[mock_hit]]
    with open("samples/sample.pdf", "rb") as f:
        resp = client.post("/query/vector", files={"file": ("sample.pdf", f, "application/pdf")}, data={"app_id": "app1", "user_id": "user1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data

@patch("pymilvus.Collection")
def test_query_vector_video_placeholder(mock_collection):
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docvid", "content": "vid", "metadata": {}}.get(k, d)
    mock_hit.score = 0.55
    mock_collection.return_value.search.return_value = [[mock_hit]]
    with open("samples/sample.mp4", "rb") as f:
        resp = client.post("/query/vector", files={"file": ("sample.mp4", f, "video/mp4")}, data={"app_id": "app1", "user_id": "user1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data

@patch("app.main.jina_embedder")
@patch("pymilvus.Collection")
def test_query_vector_legacy_json(mock_collection, mock_embedder):
    mock_embedder.encode.return_value = [[0.1]*768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "doc1", "content": "foo", "metadata": {"created_at": "2024-06-10"}}.get(k, d)
    mock_hit.score = 0.99
    mock_collection.return_value.search.return_value = [[mock_hit]]
    req = {
        "query": "test",
        "app_id": "app1",
        "user_id": "user1",
        "top_k": 1,
        "filters": {"doc_type": "pdf", "created_after": "2024-06-01"}
    }
    resp = client.post("/query/vector", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["doc_id"] == "doc1"
    assert data["results"][0]["metadata"]["created_at"] == "2024-06-10" 

@patch("app.main.GraphDatabase")
@patch("pymilvus.Collection")
def test_query_graph_context_expansion(mock_collection, mock_neo4j):
    # Mock Milvus search result
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "doc123", "content": "chunk", "metadata": {}}.get(k, d)
    mock_hit.score = 0.99
    mock_collection.return_value.search.return_value = [[mock_hit]]
    # Mock Neo4j session and result
    mock_session = MagicMock()
    mock_neo4j.driver.return_value.session.return_value.__enter__.return_value = mock_session
    mock_node = {"doc_id": "doc123", "label": "Result Chunk", "type": "result"}
    mock_rel = MagicMock()
    mock_rel.start_node = {"doc_id": "doc123"}
    mock_rel.end_node = {"doc_id": "doc456"}
    mock_rel.type = "context"
    mock_session.run.return_value = [{"nodes": [mock_node], "relationships": [mock_rel]}]
    req = {
        "query": "test",
        "app_id": "app1",
        "user_id": "user1",
        "graph_expansion": {"depth": 1, "type": "context"}
    }
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["results"][0]["graph_context"]["nodes"][0]["id"] == "doc123"
    assert data["results"][0]["graph_context"]["edges"][0]["source"] == "doc123"

@patch("app.main.GraphDatabase")
@patch("pymilvus.Collection")
def test_query_graph_semantic_expansion(mock_collection, mock_neo4j):
    # Similar to above, but with type="semantic"
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "doc789", "content": "sem", "metadata": {}}.get(k, d)
    mock_hit.score = 0.88
    mock_collection.return_value.search.return_value = [[mock_hit]]
    mock_session = MagicMock()
    mock_neo4j.driver.return_value.session.return_value.__enter__.return_value = mock_session
    mock_node = {"doc_id": "doc789", "label": "Semantic Chunk", "type": "semantic"}
    mock_rel = MagicMock()
    mock_rel.start_node = {"doc_id": "doc789"}
    mock_rel.end_node = {"doc_id": "doc999"}
    mock_rel.type = "semantic"
    mock_session.run.return_value = [{"nodes": [mock_node], "relationships": [mock_rel]}]
    req = {
        "query": "semantics",
        "app_id": "app2",
        "user_id": "user2",
        "graph_expansion": {"depth": 2, "type": "semantic"}
    }
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["graph_context"]["nodes"][0]["type"] == "semantic"
    assert data["results"][0]["graph_context"]["edges"][0]["type"] == "semantic"

@patch("app.main.GraphDatabase")
@patch("pymilvus.Collection")
def test_query_graph_neo4j_error(mock_collection, mock_neo4j):
    # Simulate Neo4j error
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docerr", "content": "err", "metadata": {}}.get(k, d)
    mock_hit.score = 0.77
    mock_collection.return_value.search.return_value = [[mock_hit]]
    mock_neo4j.driver.return_value.session.side_effect = Exception("Neo4j down")
    req = {
        "query": "fail",
        "app_id": "app3",
        "user_id": "user3",
        "graph_expansion": {"depth": 1, "type": "context"}
    }
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["graph_context"]["nodes"][0]["id"] == "docerr"
    assert data["results"][0]["graph_context"]["edges"] == [] 