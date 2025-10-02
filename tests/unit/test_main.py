import pytest
from app.main import (
    detect_file_type,
    extract_text,
    extract_pdf,
    extract_image,
    extract_audio,
    extract_content_by_type,
    chunk_text_recursive,
)
from unittest.mock import patch, MagicMock
from app.main import app, VectorQueryRequest
from fastapi.testclient import TestClient


class TestExtraction:
    def test_detect_file_type_text(self):
        content = b"Hello world!"
        assert detect_file_type("test.txt", content).startswith("text/")

    def test_extract_text_utf8(self):
        content = "Hello world!".encode("utf-8")
        assert extract_text(content) == "Hello world!"

    def test_extract_text_latin1(self):
        content = "Olá mundo!".encode("latin1")
        assert "Olá" in extract_text(content)

    def test_extract_pdf_stub(self):
        content = b"%PDF-1.4..."
        assert "PDF extraction" in extract_pdf(content)

    def test_extract_image_stub(self):
        content = b"\x89PNG..."
        assert "OCR" in extract_image(content)

    def test_extract_audio_stub(self):
        content = b"ID3..."
        assert "ASR" in extract_audio(content)

    def test_extract_content_by_type_text(self):
        content = b"Hello world!"
        assert extract_content_by_type("text/plain", content) == "Hello world!"

    def test_chunk_text_recursive(self):
        text = " ".join(["word"] * 1000)
        chunks = chunk_text_recursive(text, chunk_size=100, overlap=20)
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
        assert len(chunks) > 1


client = TestClient(app)

# Universal embedding/model patch for all relevant tests
universal_patches = [
    patch("app.main.jina_embedder"),
    patch("app.main.embed_image_nomic", return_value=[[0.1] * 768]),
    patch("app.main.embed_pdf_nomic", return_value=[[0.3] * 768]),
    patch("app.main.embed_audio_whisper", return_value=[[0.2] * 768]),
    patch("app.main.connections.connect"),  # Mock Milvus connection
    patch("app.main.connections.has_connection", return_value=True),  # Mock connection check
    patch("app.main.utility.list_collections", return_value=[]),  # Mock collection listing
    patch("os.path.exists", return_value=True),  # Mock file system checks
]


def apply_universal_patches(test_func):
    for p in reversed(universal_patches):
        test_func = p(test_func)
    return test_func


@apply_universal_patches
@patch("app.main.Collection")
def test_query_vector_basic(mock_collection, mock_os_path_exists, mock_list_collections, mock_has_connection, mock_connect, mock_embed_audio, mock_embed_pdf, mock_embed_image, mock_embedder):
    mock_embedder.encode.return_value = [[0.1] * 768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {
        "doc_id": "doc1",
        "content": "foo",
        "metadata": {"created_at": "2024-06-10"},
    }.get(k, d)
    mock_hit.score = 0.99

    def search_side_effect(*args, **kwargs):
        return [[mock_hit]]

    mock_collection.return_value.search.side_effect = search_side_effect
    req = {
        "query": "test",
        "app_id": "app1",
        "user_id": "user1",
        "top_k": 1,
        "filters": {"doc_type": "pdf", "created_after": "2024-06-01"},
    }
    resp = client.post("/query/vector", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["doc_id"] == "doc1"
    assert data["results"][0]["metadata"]["created_at"] == "2024-06-10"


@apply_universal_patches
@patch("app.main.Collection")
def test_query_vector_error_handling(mock_collection, mock_os_path_exists, mock_list_collections, mock_has_connection, mock_connect, mock_embed_audio, mock_embed_pdf, mock_embed_image, mock_embedder):
    mock_embedder.encode.side_effect = Exception("fail")
    req = {"query": "fail", "app_id": "a", "user_id": "u"}
    resp = client.post("/query/vector", json=req)
    assert resp.status_code == 422
    assert resp.json()["status"] == "error"


@patch("app.main.embed_image_nomic", return_value=[[0.1] * 768])
@patch("app.main.jina_embedder")
@patch("app.main.Collection")
@patch("os.path.exists", return_value=True)
@patch("app.main.utility.list_collections", return_value=[])
@patch("app.main.connections.has_connection", return_value=True)
@patch("app.main.connections.connect")
def test_query_vector_image(mock_connect, mock_has_connection, mock_list_collections, mock_os_path_exists, mock_collection, mock_embedder, mock_embed_image):
    mock_embedder.encode.return_value = [[0.1] * 768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docimg", "content": "img", "metadata": {}}.get(k, d)
    mock_hit.score = 0.88

    def search_side_effect(*args, **kwargs):
        return [[mock_hit]]

    mock_collection.return_value.search.side_effect = search_side_effect
    with open("samples/sample.jpg", "rb") as f:
        resp = client.post(
            "/query/vector", files={"file": ("sample.jpg", f, "image/jpeg")}, data={"app_id": "app1", "user_id": "user1"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


@patch("app.main.embed_audio_whisper", return_value=[[0.2] * 768])
@patch("app.main.Collection")
@patch("os.path.exists", return_value=True)
@patch("app.main.utility.list_collections", return_value=[])
@patch("app.main.connections.has_connection", return_value=True)
@patch("app.main.connections.connect")
def test_query_vector_audio(mock_connect, mock_has_connection, mock_list_collections, mock_os_path_exists, mock_collection, mock_embed_audio):
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docaud", "content": "aud", "metadata": {}}.get(k, d)
    mock_hit.score = 0.77

    def search_side_effect(*args, **kwargs):
        return [[mock_hit]]

    mock_collection.return_value.search.side_effect = search_side_effect
    with open("samples/sample.mp3", "rb") as f:
        resp = client.post(
            "/query/vector", files={"file": ("sample.mp3", f, "audio/mpeg")}, data={"app_id": "app1", "user_id": "user1"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


@patch("app.main.embed_pdf_nomic", return_value=[[0.3] * 768])
@patch("app.main.Collection")
@patch("app.main.get_text_embedder")
@patch("os.path.exists", return_value=True)
@patch("app.main.utility.list_collections", return_value=[])
@patch("app.main.connections.has_connection", return_value=True)
@patch("app.main.connections.connect")
def test_query_vector_pdf(mock_connect, mock_has_connection, mock_list_collections, mock_os_path_exists, mock_get_text_embedder, mock_collection, mock_embed_pdf):
    # Mock the Jina embedder
    mock_jina = MagicMock()
    mock_jina.encode.return_value = [[0.1] * 768]
    mock_get_text_embedder.return_value = mock_jina
    
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docpdf", "content": "pdf", "metadata": {}}.get(k, d)
    mock_hit.score = 0.66

    def search_side_effect(*args, **kwargs):
        return [[mock_hit]]

    mock_collection.return_value.search.side_effect = search_side_effect
    with open("samples/sample.pdf", "rb") as f:
        resp = client.post(
            "/query/vector", files={"file": ("sample.pdf", f, "application/pdf")}, data={"app_id": "app1", "user_id": "user1"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


@patch("app.main.embed_image_nomic", return_value=[[0.1] * 768])
@patch("app.main.embed_audio_whisper", return_value=[[0.2] * 768])
@patch("app.main.embed_pdf_nomic", return_value=[[0.3] * 768])
@patch("app.main.jina_embedder")
@patch("app.main.Collection")
def test_query_vector_video_placeholder(mock_collection, mock_embedder, mock_embed_pdf, mock_embed_audio, mock_embed_image):
    mock_embedder.encode.return_value = [[0.1] * 768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docvid", "content": "vid", "metadata": {}}.get(k, d)
    mock_hit.score = 0.55

    def search_side_effect(*args, **kwargs):
        return [[mock_hit]]

    mock_collection.return_value.search.side_effect = search_side_effect
    with open("samples/sample.mp4", "rb") as f:
        resp = client.post(
            "/query/vector", files={"file": ("sample.mp4", f, "video/mp4")}, data={"app_id": "app1", "user_id": "user1"}
        )
    assert resp.status_code == 415
    data = resp.json()
    assert data["status"] == "error"


@apply_universal_patches
@patch("app.main.Collection")
def test_query_vector_legacy_json(mock_collection, mock_os_path_exists, mock_list_collections, mock_has_connection, mock_connect, mock_embed_audio, mock_embed_pdf, mock_embed_image, mock_embedder):
    mock_embedder.encode.return_value = [[0.1] * 768]
    mock_hit = MagicMock()
    mock_hit.entity.get.side_effect = lambda k, d=None: {
        "doc_id": "doc1",
        "content": "foo",
        "metadata": {"created_at": "2024-06-10"},
    }.get(k, d)
    mock_hit.score = 0.99

    def search_side_effect(*args, **kwargs):
        return [[mock_hit]]

    mock_collection.return_value.search.side_effect = search_side_effect
    req = {
        "query": "test",
        "app_id": "app1",
        "user_id": "user1",
        "top_k": 1,
        "filters": {"doc_type": "pdf", "created_after": "2024-06-01"},
    }
    resp = client.post("/query/vector", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["doc_id"] == "doc1"
    assert data["results"][0]["metadata"]["created_at"] == "2024-06-10"


class MockRel:
    def __init__(self, start_doc_id, end_doc_id, etype, weight=1.0, expanded_by=None, config_source=None, label=None):
        self.type = etype
        self.start_node = {"doc_id": start_doc_id}
        self.end_node = {"doc_id": end_doc_id}
        self._data = {"weight": weight, "expanded_by": expanded_by or etype, "config_source": config_source or "app"}
        if label:
            self._data["label"] = label

    def get(self, k, d=None):
        return self._data.get(k, d)


@pytest.mark.skip(reason="Known mock/patch issue with /query/graph endpoint. See tracker.")
@apply_universal_patches
@patch("app.main.GraphDatabase")
@patch("app.main.Collection")
def test_query_graph_context_expansion(
    mock_collection, mock_neo4j, mock_embed_audio, mock_embed_pdf, mock_embed_image, mock_embedder
):
    with patch("app.main.edge_graph_config_loader.get_app_edge_weights", return_value={"context": 1.0}):
        mock_embedder.encode.return_value = [[0.1] * 768]
        # Mock Milvus search result
        mock_hit = MagicMock()
        mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "doc123", "content": "chunk", "metadata": {}}.get(k, d)
        mock_hit.score = 0.99

        def search_side_effect(*args, **kwargs):
            return [[mock_hit]]

        mock_collection.return_value.search.side_effect = search_side_effect
        # Mock Neo4j session and result
        mock_session = MagicMock()
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_node = {
            "doc_id": "doc123",
            "label": "Result Chunk",
            "type": "result",
            "expanded_by": "context",
            "config_source": "app",
        }
        mock_rel = MockRel("doc123", "doc456", "context", weight=1.0)
        mock_record = {"nodes": [mock_node], "relationships": [mock_rel]}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        req = {"query": "test", "app_id": "app1", "user_id": "user1", "graph_expansion": {"depth": 1, "type": "context"}}
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert data["results"][0]["graph_context"]["nodes"][0]["id"] == "doc123"
        assert data["results"][0]["graph_context"]["edges"][0]["source"] == "doc123"


@pytest.mark.skip(reason="Known mock/patch issue with /query/graph endpoint. See tracker.")
@apply_universal_patches
@patch("app.main.GraphDatabase")
@patch("app.main.Collection")
def test_query_graph_semantic_expansion(
    mock_collection, mock_neo4j, mock_embed_audio, mock_embed_pdf, mock_embed_image, mock_embedder
):
    with patch("app.main.edge_graph_config_loader.get_app_edge_weights", return_value={"semantic": 1.0, "semantic_of": 1.0}):
        mock_embedder.encode.return_value = [[0.1] * 768]
        # Similar to above, but with type="semantic"
        mock_hit = MagicMock()
        mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "doc789", "content": "sem", "metadata": {}}.get(k, d)
        mock_hit.score = 0.88

        def search_side_effect(*args, **kwargs):
            return [[mock_hit]]

        mock_collection.return_value.search.side_effect = search_side_effect
        mock_session = MagicMock()
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_node = {
            "doc_id": "doc789",
            "label": "Semantic Chunk",
            "type": "semantic",
            "expanded_by": "semantic",
            "config_source": "app",
        }
        mock_rel = MockRel("doc789", "doc999", "semantic", weight=1.0)
        mock_record = {"nodes": [mock_node], "relationships": [mock_rel]}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        req = {"query": "semantics", "app_id": "app2", "user_id": "user2", "graph_expansion": {"depth": 2, "type": "semantic"}}
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["graph_context"]["nodes"][0]["type"] in ("semantic", "semantic_of")
        assert data["results"][0]["graph_context"]["edges"][0]["type"] in ("semantic", "semantic_of")


@pytest.mark.skip(reason="Known mock/patch issue with /query/graph endpoint. See tracker.")
@apply_universal_patches
@patch("app.main.GraphDatabase")
@patch("app.main.Collection")
def test_query_graph_neo4j_error(
    mock_collection, mock_neo4j, mock_embed_audio, mock_embed_pdf, mock_embed_image, mock_embedder
):
    with patch("app.main.edge_graph_config_loader.get_app_edge_weights", return_value={"context": 1.0}):
        mock_embedder.encode.return_value = [[0.1] * 768]
        # Simulate Neo4j error
        mock_hit = MagicMock()
        mock_hit.entity.get.side_effect = lambda k, d=None: {"doc_id": "docerr", "content": "err", "metadata": {}}.get(k, d)
        mock_hit.score = 0.77

        def search_side_effect(*args, **kwargs):
            return [[mock_hit]]

        mock_collection.return_value.search.side_effect = search_side_effect
        mock_neo4j.driver.return_value.session.side_effect = Exception("Neo4j down")
        # Even if Neo4j is down, the mocked Milvus still returns a result
        mock_node = {
            "doc_id": "docerr",
            "label": "Error Chunk",
            "type": "error",
            "expanded_by": "context",
            "config_source": "app",
        }
        mock_rel = MockRel("docerr", "doc456", "context", weight=1.0)
        mock_record = {"nodes": [mock_node], "relationships": [mock_rel]}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session = MagicMock()
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        req = {"query": "fail", "app_id": "app3", "user_id": "user3", "graph_expansion": {"depth": 1, "type": "context"}}
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["graph_context"]["nodes"][0]["id"] == "docerr"
        # Edges may be empty if Neo4j is down, but nodes should be present


@pytest.mark.skip(reason="Known mock/patch issue with /query/graph endpoint. See tracker.")
@apply_universal_patches
@patch("app.main.GraphDatabase")
@patch("app.main.Collection")
def test_query_graph_filtering_and_traceability(
    mock_collection, mock_neo4j, mock_embed_audio, mock_embed_pdf, mock_embed_image, mock_embedder
):
    with patch("app.main.edge_graph_config_loader.get_app_edge_weights", return_value={"context_of": 1.0, "context": 1.0}):
        mock_embedder.encode.return_value = [[0.1] * 768]
        # Mock Milvus search result
        mock_hit = MagicMock()
        mock_hit.entity.get.side_effect = lambda k, d=None: {
            "doc_id": "doc123",
            "content": "chunk",
            "metadata": {"label": "important"},
        }.get(k, d)
        mock_hit.score = 0.99

        def search_side_effect(*args, **kwargs):
            return [[mock_hit]]

        mock_collection.return_value.search.side_effect = search_side_effect
        # Mock Neo4j session and result
        mock_session = MagicMock()
        mock_neo4j.driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_node = {
            "doc_id": "doc123",
            "label": "Result Chunk",
            "type": "result",
            "expanded_by": "context_of",
            "config_source": "app",
        }
        mock_rel = MockRel("doc123", "doc456", "context_of", weight=0.7, label="important")
        mock_record = {"nodes": [mock_node], "relationships": [mock_rel]}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        # Test filtering by edge type
        req = {
            "query": "test",
            "app_id": "app1",
            "user_id": "user1",
            "filters": {"edge_types": ["context_of"]},
            "graph_expansion": {"depth": 1, "type": "context_of"},
        }
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        edge = data["results"][0]["graph_context"]["edges"][0]
        assert edge["type"] in ("context", "context_of")
        # Test filtering by min_weight
        req["filters"] = {"min_weight": 0.5}
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        edge = data["results"][0]["graph_context"]["edges"][0]
        assert edge["weight"] >= 0.5
        # Test filtering by metadata
        req["filters"] = {"metadata": {"label": "important"}}
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        edge = data["results"][0]["graph_context"]["edges"][0]
        assert edge["type"] == "context_of"
        # Check traceability fields
        for node in data["results"][0]["graph_context"]["nodes"]:
            assert "expanded_by" in node
            assert "config_source" in node
        for edge in data["results"][0]["graph_context"]["edges"]:
            assert "expanded_by" in edge
            assert "config_source" in edge
            assert "weight" in edge
            assert "type" in edge
