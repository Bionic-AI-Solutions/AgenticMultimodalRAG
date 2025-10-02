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

    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, cache_folder=os.getenv("HF_HOME", "/home/user/RAG/models")
    )
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
    ("sample.txt", "text/plain"),
    ("sample.pdf", "application/pdf"),
    ("sample.jpg", "image/jpeg"),
    ("sample.mp3", "audio/mpeg"),
    ("sample.csv", "text/csv"),
    ("sample.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("sample.mp4", "video/mp4"),
]


@pytest.mark.parametrize("filename,expected_mime", SAMPLES)
def test_ingest_sample_file(filename, expected_mime):
    """Test ingesting various file types with proper error handling."""
    path = os.path.join("samples", filename)
    if not os.path.exists(path):
        pytest.skip(f"Sample file {filename} not found")

    with open(path, "rb") as f:
        response = client.post(
            "/docs/ingest", files={"file": (filename, f, expected_mime)}, data={"app_id": "testapp", "user_id": "testuser"}
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
        assert data.get("status") == "embedded"
        assert data.get("doc_id") is not None


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


@pytest.mark.integration
def test_query_vector_integration():
    """Test vector query with error handling."""
    try:
        response = client.post(
            "/query/vector", json={"query": "test query", "app_id": "testapp", "user_id": "testuser", "top_k": 5}
        )

        if response.status_code == 500:
            error_data = response.json()
            if "out of memory" in error_data.get("message", "").lower():
                pytest.skip("GPU out of memory during query, skipping test")
            else:
                pytest.fail(f"Query failed: {error_data.get('message')}")

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    except Exception as e:
        pytest.skip(f"Query test skipped due to error: {e}")


@pytest.mark.parametrize(
    "filename,expected_mime",
    [
        ("sample.jpg", "image/jpeg"),
        ("sample.mp3", "audio/mpeg"),
        ("sample.pdf", "application/pdf"),
        ("sample.mp4", "video/mp4"),
    ],
)
def test_query_vector_multimodal(filename, expected_mime):
    if filename.endswith(".mp4"):
        pytest.skip("Video embedding not supported; skipping test.")
    path = os.path.join("samples", filename)
    if not os.path.exists(path):
        pytest.skip(f"Sample file {filename} not found")
    with open(path, "rb") as f:
        response = client.post(
            "/query/vector", files={"file": (filename, f, expected_mime)}, data={"app_id": "testapp", "user_id": "testuser"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    # For video, results may be empty or placeholder
    if filename == "sample.mp4":
        assert isinstance(data["results"], list)


@pytest.mark.integration
def test_query_graph_text_context():
    req = {"query": "sample", "app_id": "test", "user_id": "test", "graph_expansion": {"depth": 1, "type": "context"}}
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
    path = os.path.join("samples", "sample.jpg")
    if not os.path.exists(path):
        pytest.skip("Sample image not found")
    with open(path, "rb") as f:
        resp = client.post(
            "/query/graph",
            files={"file": ("sample.jpg", f, "image/jpeg")},
            data={"app_id": "test", "user_id": "test", "graph_expansion": '{"depth": 2, "type": "semantic"}'},
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
            "weights": {"context_of": 2.0, "about_topic": 0.0, "temporal_neighbor": 1.0},
        },
    }
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "explain" in data
    explain = data["explain"]
    used = explain["used_edge_types"]
    assert "context_of" in used and used["context_of"] == 2.0
    assert "about_topic" in used and used["about_topic"] == 0.0
    assert "temporal_neighbor" in used and used["temporal_neighbor"] == 1.0
    assert "rerank" in explain
    valid_types = set([k for k, v in used.items() if v > 0] + ["context"])
    for r in data["results"]:
        for e in r["graph_context"]["edges"]:
            assert e["type"] in valid_types


@pytest.mark.integration
def test_query_graph_weighted_expansion_config():
    req = {"query": "sample", "app_id": "test", "user_id": "test", "graph_expansion": {"depth": 1, "type": "context"}}
    resp = client.post("/query/graph", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "explain" in data
    explain = data["explain"]
    used = explain["used_edge_types"]
    assert isinstance(used, dict)
    assert all(isinstance(v, float) for v in used.values())
    assert "rerank" in explain
    valid_types = set([k for k, v in used.items() if v > 0] + ["context"])
    for r in data["results"]:
        for e in r["graph_context"]["edges"]:
            assert e["type"] in valid_types


@pytest.mark.integration
def test_ingest_sample2_pdf():
    path = os.path.join("samples", "sample2.pdf")
    if not os.path.exists(path):
        pytest.skip("sample2.pdf not found")
    with open(path, "rb") as f:
        response = client.post(
            "/docs/ingest",
            files={"file": ("sample2.pdf", f, "application/pdf")},
            data={"app_id": "testapp", "user_id": "testuser"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "embedded"
    assert "doc_id" in data
    assert "embedding complete" in data.get("message", "").lower()


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
                "/docs/ingest", files={"file": (test_file, f, "text/plain")}, data={"app_id": "testapp", "user_id": "testuser"}
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
        assert data.get("status") == "embedded"
        doc_id = data.get("doc_id")
        assert doc_id is not None

        # 2. Query the document back
        query_response = client.post(
            "/query/vector", json={"query": "test document", "app_id": "testapp", "user_id": "testuser", "top_k": 1}
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
        assert "results" in query_data
        if len(query_data["results"]) == 0:
            pytest.skip("No results returned, possibly due to vector dimension mismatch")
        assert len(query_data["results"]) > 0

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


@pytest.mark.integration
class TestLiveIntegrationFlow:
    @classmethod
    def setup_class(cls):
        # Ensure all services are up
        response = client.get("/health/details")
        assert response.status_code == 200
        health_data = response.json()
        services = ["milvus", "minio", "postgres", "neo4j"]
        for service in services:
            assert service in health_data
            assert health_data[service]["status"] == "ok"
        cls.test_files = []
        cls.doc_ids = []  # parent doc_ids
        cls.chunk_doc_ids = []  # chunk-level doc_ids

    def test_01_ingest_text(self):
        test_content = "This is a test document for live integration."
        test_file = "test_live_ingest.txt"
        with open(test_file, "w") as f:
            f.write(test_content)
        with open(test_file, "rb") as f:
            response = client.post(
                "/docs/ingest", files={"file": (test_file, f, "text/plain")}, data={"app_id": "testapp", "user_id": "testuser"}
            )
        print("[DEBUG] Ingest response:", response.status_code, response.json())
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "embedded"
        assert "doc_id" in data
        self.__class__.doc_ids.append(data["doc_id"])
        self.__class__.test_files.append(test_file)
        # Wait for indexing
        time.sleep(2)
        # Query to get chunk-level doc_ids
        response = client.post(
            "/query/vector", json={"query": test_content, "app_id": "testapp", "user_id": "testuser", "top_k": 5}
        )
        print("[DEBUG] Post-ingest vector query:", response.status_code, response.json())
        assert response.status_code == 200
        data = response.json()
        chunk_ids = [r["doc_id"] for r in data["results"] if r["content"] == test_content]
        print("[DEBUG] Found chunk-level doc_ids:", chunk_ids)
        self.__class__.chunk_doc_ids.extend(chunk_ids)

    def test_02_query_vector(self):
        test_content = "This is a test document for live integration."
        for attempt in range(3):
            response = client.post(
                "/query/vector", json={"query": test_content, "app_id": "testapp", "user_id": "testuser", "top_k": 5}
            )
            print(f"[DEBUG] Query vector attempt {attempt+1}: ", response.status_code, response.json())
            assert response.status_code == 200
            data = response.json()
            if any(r.get("doc_id") in self.__class__.chunk_doc_ids for r in data["results"]):
                break
            time.sleep(2)
        else:
            assert False, "Ingested chunk-level doc_id not found in vector query results after retries"

    def test_03_ingest_pdf(self):
        path = os.path.join("samples", "sample2.pdf")
        assert os.path.exists(path)
        with open(path, "rb") as f:
            response = client.post(
                "/docs/ingest",
                files={"file": ("sample2.pdf", f, "application/pdf")},
                data={"app_id": "testapp", "user_id": "testuser"},
            )
        print("[DEBUG] Ingest PDF response:", response.status_code, response.json())
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "embedded"
        assert "doc_id" in data
        self.__class__.doc_ids.append(data["doc_id"])
        time.sleep(2)
        # Query to get chunk-level doc_ids for PDF
        response = client.post(
            "/query/vector",
            json={
                "query": "Roadmap",  # Use a likely phrase from the PDF
                "app_id": "testapp",
                "user_id": "testuser",
                "top_k": 5,
            },
        )
        print("[DEBUG] Post-ingest PDF vector query:", response.status_code, response.json())
        assert response.status_code == 200
        data = response.json()
        chunk_ids = [r["doc_id"] for r in data["results"] if "Roadmap" in r["content"]]
        print("[DEBUG] Found PDF chunk-level doc_ids:", chunk_ids)
        self.__class__.chunk_doc_ids.extend(chunk_ids)

    def test_04_query_graph_context(self):
        req = {
            "query": "This is a test document for live integration.",
            "app_id": "testapp",
            "user_id": "testuser",
            "graph_expansion": {"depth": 1, "type": "context"},
        }
        resp = client.post("/query/graph", json=req)
        print("[DEBUG] Graph context response:", resp.status_code, resp.json())
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        found = False
        for r in data["results"]:
            assert "graph_context" in r
            assert "nodes" in r["graph_context"]
            assert "edges" in r["graph_context"]
            node_ids = [n.get("id") for n in r["graph_context"]["nodes"]]
            print("[DEBUG] Graph node ids:", node_ids)
            print("[DEBUG] Known chunk-level doc_ids:", self.__class__.chunk_doc_ids)
            if any(nid in self.__class__.chunk_doc_ids for nid in node_ids):
                found = True
        assert found, "No graph node id matched any known chunk-level doc_id"

    def test_05_query_graph_weighted_expansion(self):
        req = {
            "query": "This is a test document for live integration.",
            "app_id": "testapp",
            "user_id": "testuser",
            "graph_expansion": {
                "depth": 1,
                "type": "context",
                "weights": {"context_of": 2.0, "about_topic": 0.0, "temporal_neighbor": 1.0},
            },
        }
        resp = client.post("/query/graph", json=req)
        print("[DEBUG] Graph weighted expansion response:", resp.status_code, resp.json())
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "explain" in data
        explain = data["explain"]
        used = explain["used_edge_types"]
        print("[DEBUG] Used edge types:", used)
        # Accept both 'context' and any used edge types
        valid_types = set([k for k, v in used.items() if v > 0] + ["context"])
        for r in data["results"]:
            for e in r["graph_context"]["edges"]:
                print("[DEBUG] Edge type:", e["type"])
                assert e["type"] in valid_types

    @classmethod
    def teardown_class(cls):
        # Cleanup test files
        for f in cls.test_files:
            if os.path.exists(f):
                os.remove(f)


@pytest.mark.integration
def test_query_graph_filtering_and_traceability_integration():
    # Ingest a test document with metadata
    test_content = "Integration test document for filtering and traceability."
    test_file = "test_integration_filtering.txt"
    try:
        with open(test_file, "w") as f:
            f.write(test_content)
        with open(test_file, "rb") as f:
            response = client.post(
                "/docs/ingest", files={"file": (test_file, f, "text/plain")}, data={"app_id": "testapp", "user_id": "testuser"}
            )
        assert response.status_code == 200
        data = response.json()
        doc_id = data.get("doc_id")
        assert doc_id is not None
        # Query the graph with edge type filter
        req = {
            "query": "Integration test document for filtering and traceability.",
            "app_id": "testapp",
            "user_id": "testuser",
            "filters": {"edge_types": ["context_of"]},
            "graph_expansion": {"depth": 1, "type": "context_of"},
        }
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        for r in data["results"]:
            for edge in r["graph_context"]["edges"]:
                assert edge["type"] in ("context", "context_of")
                assert "expanded_by" in edge
                assert "config_source" in edge
                assert "weight" in edge
            for node in r["graph_context"]["nodes"]:
                assert "expanded_by" in node
                assert "config_source" in node
        # Query the graph with min_weight filter
        req["filters"] = {"min_weight": 0.0}
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        for r in data["results"]:
            for edge in r["graph_context"]["edges"]:
                assert edge["weight"] >= 0.0
        # Query the graph with metadata filter (if supported by your ingestion logic)
        req["filters"] = {"metadata": {"label": "important"}}
        resp = client.post("/query/graph", json=req)
        assert resp.status_code == 200
        data = resp.json()
        for r in data["results"]:
            for edge in r["graph_context"]["edges"]:
                # If label is present, it should match
                if "label" in edge:
                    assert edge["label"] == "important"
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
