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