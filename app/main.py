from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Body, Request, Depends
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Any, List, Union
from pydantic import BaseModel
import logging
import os
from pymilvus import connections, exceptions as milvus_exceptions, Collection, CollectionSchema, FieldSchema, DataType, list_collections
import asyncio
from minio import Minio
import asyncpg
from neo4j import GraphDatabase
import traceback
from fastapi.responses import JSONResponse
import mimetypes
try:
    import magic  # python-magic for magic byte detection
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
from sentence_transformers import SentenceTransformer
# Nomic and Whisper imports (stubs for now)
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import whisper
from PIL import Image
import fitz  # PyMuPDF
import io
import torch
import huggingface_hub
from app.edge_graph_config import EdgeGraphConfigLoader
import numpy as np

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Set HuggingFace cache at the very top
print(f"[RAG Startup] HF_HOME={os.getenv('HF_HOME')}")
print(f"[RAG Startup] TRANSFORMERS_CACHE={os.getenv('TRANSFORMERS_CACHE')}")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/Volumes/ssd/models")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("HF_HOME", "/Volumes/ssd/models")

# Move the lifespan definition here (before app = FastAPI(...))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG System...")
    # ... (existing lifespan code) ...
    # Set HuggingFace cache to /Volumes/ssd/models
    os.environ["HF_HOME"] = "/Volumes/ssd/models"
    os.environ["TRANSFORMERS_CACHE"] = "/Volumes/ssd/models"
    # Global edge-graph config loader (Phase 1)
    edge_graph_config_loader = EdgeGraphConfigLoader()
    yield
    # Shutdown
    logger.info("Shutting down RAG System...")

# FastAPI app
app = FastAPI(
    title="Agentic Multimodal RAG System",
    description="Advanced RAG system with Milvus integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    timestamp: str

class IngestResponse(BaseModel):
    doc_id: str
    status: str
    message: Optional[str] = None

class VectorQueryRequest(BaseModel):
    query: str
    app_id: str
    user_id: str
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None  # Flexible key-value filters

class VectorQueryResult(BaseModel):
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]

class VectorQueryResponse(BaseModel):
    results: List[VectorQueryResult]

class GraphContextNode(BaseModel):
    id: str
    label: str
    type: str

class GraphContextEdge(BaseModel):
    source: str
    target: str
    type: str

class GraphQueryResult(BaseModel):
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    graph_context: Dict[str, Any]  # nodes/edges

class GraphQueryResponse(BaseModel):
    results: List[GraphQueryResult]
    explain: Optional[Dict[str, Any]] = None

# Milvus connection utility
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Utility to ensure Milvus connection
def ensure_milvus_connection():
    try:
        if not connections.has_connection("default"):
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise

# Utility to ensure collection exists (create if missing)
def ensure_collection(collection_name: str):
    ensure_milvus_connection()
    try:
        if collection_name not in list_collections():
            # Define schema (should match ingestion logic)
            fields = [
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, description="RAG chunks")
            collection = Collection(collection_name, schema)
            logger.info(f"Created Milvus collection: {collection_name}")
            # Create index on embedding
            index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Created index on 'embedding' for collection: {collection_name}")
        else:
            collection = Collection(collection_name)
            # Check if index exists, create if missing
            if not collection.has_index():
                index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
                collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Created index on 'embedding' for collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to ensure Milvus collection '{collection_name}': {e}")
        raise

async def check_milvus():
    try:
        # pymilvus is not async, but this is a quick check
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        # Try a simple operation
        _ = connections.get_connection_addr("default")
        return "ok"
    except milvus_exceptions.MilvusException as e:
        return f"unreachable: {str(e)}"
    except Exception as e:
        return f"unreachable: {str(e)}"

# Minio health check
async def check_minio():
    try:
        minio_host = os.getenv("MINIO_HOST", "localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        client = Minio(minio_host, access_key=minio_access_key, secret_key=minio_secret_key, secure=minio_secure)
        # List buckets as a health check
        client.list_buckets()
        return "ok"
    except Exception as e:
        return f"unreachable: {str(e)}"

# Postgres health check
async def check_postgres():
    try:
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        pg_db = os.getenv("POSTGRES_DB", "postgres")
        conn = await asyncpg.connect(
            user=pg_user,
            password=pg_password,
            database=pg_db,
            host=pg_host,
            port=pg_port,
            timeout=2
        )
        await conn.close()
        return "ok"
    except Exception as e:
        return f"unreachable: {str(e)}"

# Neo4j health check
async def check_neo4j():
    try:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "test")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return "ok"
    except Exception as e:
        return f"unreachable: {str(e)}"

# Enhanced health checks with detailed error info
async def check_minio_detailed():
    try:
        minio_host = os.getenv("MINIO_HOST", "localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        client = Minio(minio_host, access_key=minio_access_key, secret_key=minio_secret_key, secure=minio_secure)
        client.list_buckets()
        return {"status": "ok"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Minio health check failed: {e}\n{tb}")
        return {"status": "unreachable", "error": str(e), "trace": tb}

async def check_postgres_detailed():
    try:
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        pg_db = os.getenv("POSTGRES_DB", "postgres")
        conn = await asyncpg.connect(
            user=pg_user,
            password=pg_password,
            database=pg_db,
            host=pg_host,
            port=pg_port,
            timeout=2
        )
        await conn.close()
        return {"status": "ok"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Postgres health check failed: {e}\n{tb}")
        return {"status": "unreachable", "error": str(e), "trace": tb}

async def check_neo4j_detailed():
    try:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "test")
        with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
            with driver.session() as session:
                session.run("RETURN 1")
        return {"status": "ok"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Neo4j health check failed: {e}\n{tb}")
        return {"status": "unreachable", "error": str(e), "trace": tb}

async def check_milvus_detailed():
    try:
        result = await check_milvus()
        if result == "ok":
            return {"status": "ok"}
        else:
            return {"status": "unreachable", "error": result}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Milvus health check failed: {e}\n{tb}")
        return {"status": "unreachable", "error": str(e), "trace": tb}

def detect_file_type(filename: str, content: bytes) -> str:
    # Try magic bytes first if available
    if HAS_MAGIC:
        try:
            mime = magic.from_buffer(content, mime=True)
            if mime:
                return mime
        except Exception as e:
            logger.warning(f"magic type detection failed: {e}")
    # Fallback to mimetypes
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    return "application/octet-stream"

# Extraction helpers (stubs for now)
def extract_text(content: bytes) -> str:
    # TODO: Use chardet or similar for encoding detection
    try:
        return content.decode('utf-8')
    except Exception:
        return content.decode('latin1', errors='replace')

def extract_pdf(content: bytes) -> str:
    # TODO: Use PyPDF2 or pdfplumber for real extraction
    logger.info("Would use PyPDF2/pdfplumber for PDF extraction here.")
    return "[PDF extraction not yet implemented]"

def extract_image(content: bytes) -> str:
    # TODO: Use pytesseract for OCR
    logger.info("Would use pytesseract for OCR here.")
    return "[Image OCR not yet implemented]"

def extract_audio(content: bytes) -> str:
    # TODO: Use openai-whisper or SpeechRecognition for ASR
    logger.info("Would use Whisper/SpeechRecognition for ASR here.")
    return "[Audio ASR not yet implemented]"

def extract_content_by_type(detected_type: str, content: bytes) -> str:
    if detected_type.startswith('text/'):
        return extract_text(content)
    elif detected_type == 'application/pdf':
        return extract_pdf(content)
    elif detected_type.startswith('image/'):
        return extract_image(content)
    elif detected_type.startswith('audio/'):
        return extract_audio(content)
    else:
        logger.info(f"No extractor for type: {detected_type}")
        return "[Unsupported file type for extraction]"

def chunk_text_recursive(text: str, chunk_size: int = 512, overlap: int = 102, separator: str = " ") -> list:
    """
    LangChain-inspired recursive/fixed-size chunking with overlap.
    - chunk_size: number of words per chunk (default 512)
    - overlap: number of words to overlap (default 20% of chunk_size)
    - separator: word separator (default: space)
    """
    words = text.split(separator)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(separator.join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

# Text embedding with Jina Embeddings v2 via SentenceTransformers
jina_embedder = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, cache_folder=os.getenv("HF_HOME", "/Volumes/ssd/models"))

# Fallback: Sentence Transformers (BGE-M3, etc.)
sbert_embedder = SentenceTransformer("BAAI/bge-m3", cache_folder=os.getenv("HF_HOME", "/Volumes/ssd/models"))

def embed_text_jina(chunks: list[str]) -> list:
    try:
        return jina_embedder.encode(chunks, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"Jina embedding failed: {e}, falling back to SBERT.")
        return sbert_embedder.encode(chunks, show_progress_bar=False)

# Nomic imports
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Nomic model/processor (load once)
nomic_model = None
nomic_processor = None

def get_nomic_model():
    global nomic_model, nomic_processor
    model_path = os.path.join(os.getenv("HF_HOME", "/Volumes/ssd/models"), "nomic-ai/colnomic-embed-multimodal-7b")
    processor_path = model_path
    if nomic_model is None or nomic_processor is None:
        logger.info(f"Checking for Nomic model at {model_path}")
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            logger.info("Nomic model not found locally. Downloading from HuggingFace...")
            huggingface_hub.snapshot_download(repo_id="nomic-ai/colnomic-embed-multimodal-7b", local_dir=model_path, local_dir_use_symlinks=False)
        logger.info("Loading Nomic model...")
        nomic_model = ColQwen2_5.from_pretrained(model_path, cache_dir=os.getenv("HF_HOME", "/Volumes/ssd/models"))
        logger.info("Loading Nomic processor with use_fast=False...")
        nomic_processor = ColQwen2_5_Processor.from_pretrained(processor_path, use_fast=False, cache_dir=os.getenv("HF_HOME", "/Volumes/ssd/models"))
        logger.info("Loaded slow processor.")
    return nomic_model, nomic_processor

# Image/PDF embedding with Nomic
def embed_image_nomic(image_bytes: bytes) -> list:
    try:
        model, processor = get_nomic_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch_images = processor.process_images([image]).to(model.device)
        with torch.no_grad():
            image_embeddings = model(**batch_images)
        arr = image_embeddings.cpu().tolist()
        # Pool patch embeddings to a single vector (mean across patches)
        if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
            pooled = np.mean(np.array(arr[0]), axis=0).tolist()
            return pooled
        return arr
    except Exception as e:
        logger.error(f"Nomic image embedding failed: {e}")
        return [0.0]*1024

# PDF embedding: extract first page as image, then embed
def embed_pdf_nomic(pdf_bytes: bytes) -> list:
    try:
        logger.info("Getting Nomic model and processor...")
        model, processor = get_nomic_model()
        logger.info("Opening PDF with fitz...")
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        if pdf.page_count == 0:
            logger.error("PDF has no pages")
            raise ValueError("PDF has no pages")
        logger.info("Loading first page of PDF...")
        page = pdf.load_page(0)
        logger.info("Rendering page to pixmap...")
        pix = page.get_pixmap()
        logger.info("Converting pixmap to image...")
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        logger.info("Processing image with processor...")
        batch_images = processor.process_images([img]).to(model.device)
        logger.info("Running model to get embeddings...")
        with torch.no_grad():
            image_embeddings = model(**batch_images)
        logger.info("Embedding complete. Returning result.")
        arr = image_embeddings.cpu().tolist()
        # Pool patch embeddings to a single vector (mean across patches)
        if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
            pooled = np.mean(np.array(arr[0]), axis=0).tolist()
            return pooled
        return arr
    except Exception as e:
        logger.error(f"Nomic PDF embedding failed: {e}")
        return [0.0]*1024

# Audio embedding with Whisper + text embedding
whisper_model = whisper.load_model("base", download_root=os.getenv("HF_HOME", "/Volumes/ssd/models"))
def embed_audio_whisper(audio_bytes: bytes) -> list:
    try:
        # Save to temp file for whisper
        with open("/tmp/audio.wav", "wb") as f:
            f.write(audio_bytes)
        result = whisper_model.transcribe("/tmp/audio.wav")
        transcript = result.get("text", "")
        logger.info(f"Whisper transcript: {transcript[:100]}")
        if transcript:
            return embed_text_jina([transcript])
        else:
            return [[0.0]*1024]
    except Exception as e:
        logger.error(f"Whisper audio embedding failed: {e}")
        return [[0.0]*1024]

# Global edge-graph config loader (Phase 1)
edge_graph_config_loader = EdgeGraphConfigLoader()

def get_edge_graph_config():
    return edge_graph_config_loader.get_config()

@app.get("/edge-graph/config")
def get_edge_graph_config_endpoint(config: dict = Depends(get_edge_graph_config)):
    """Return the current edge-graph config (for debugging/validation)."""
    return config

@app.get("/health", response_model=HealthResponse)
async def health_check():
    services = {}
    # Milvus health check
    services["milvus"] = await check_milvus()
    # Minio health check
    services["minio"] = await check_minio()
    # Postgres health check
    services["postgres"] = await check_postgres()
    # Neo4j health check
    services["neo4j"] = await check_neo4j()
    return HealthResponse(
        status="ok" if all(v == "ok" for v in services.values()) else "degraded",
        services=services,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/health/details")
async def health_details():
    milvus = await check_milvus_detailed()
    minio = await check_minio_detailed()
    postgres = await check_postgres_detailed()
    neo4j = await check_neo4j_detailed()
    return {
        "milvus": milvus,
        "minio": minio,
        "postgres": postgres,
        "neo4j": neo4j,
        "timestamp": datetime.utcnow().isoformat()
    }

# Utility to normalize embeddings for Milvus (always returns list of flat float lists)
def normalize_embeddings(embeddings, num_chunks):
    # Recursively flatten to a list of floats
    def flatten_recursive(e):
        if isinstance(e, (list, tuple)):
            result = []
            for item in e:
                result.extend(flatten_recursive(item))
            return result
        elif isinstance(e, float):
            return [e]
        elif isinstance(e, int):
            return [float(e)]
        else:
            return []
    # If embeddings is a list, flatten each
    if isinstance(embeddings, list) and len(embeddings) > 0:
        return [flatten_recursive(e) for e in embeddings]
    # If embeddings is a single embedding, repeat for each chunk
    if isinstance(embeddings, (list, tuple)) and all(isinstance(x, (float, int)) for x in embeddings):
        return [list(map(float, embeddings))] * num_chunks
    # Fallback: return zero vectors
    return [[0.0]*1024 for _ in range(num_chunks)]

@app.post("/docs/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    app_id: str = Form(...),
    user_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    # Step 1: File Upload & Validation
    try:
        # Validate file size (example: max 100MB)
        MAX_SIZE = 100 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_SIZE:
            return JSONResponse(status_code=413, content={"status": "error", "message": "File too large"})
        if not file.filename:
            return JSONResponse(status_code=422, content={"status": "error", "message": "Filename required"})
        # Generate a unique doc_id (could use UUID, here use timestamp+filename)
        import uuid
        doc_id = f"{app_id}_{user_id or 'anon'}_{uuid.uuid4().hex}"
        # Store file in Minio
        minio_host = os.getenv("MINIO_HOST", "localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        bucket_name = os.getenv("MINIO_BUCKET", "rag-docs")
        client = Minio(minio_host, access_key=minio_access_key, secret_key=minio_secret_key, secure=minio_secure)
        # Ensure bucket exists
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
        # Store file
        minio_path = f"{app_id}/{user_id or 'anon'}/{doc_id}/{file.filename}"
        import io
        client.put_object(
            bucket_name,
            minio_path,
            io.BytesIO(contents),
            length=len(contents),
            content_type=file.content_type or "application/octet-stream"
        )
        logger.info(f"File uploaded to Minio: {bucket_name}/{minio_path}")
        # Step 2: Type Detection
        detected_type = detect_file_type(file.filename, contents)
        logger.info(f"Detected file type: {detected_type}")
        # Step 3: Extraction
        extracted_content = extract_content_by_type(detected_type, contents)
        logger.info(f"Extracted content (truncated): {extracted_content[:200]}")
        # Step 4: Chunking
        if isinstance(extracted_content, str):
            chunks = chunk_text_recursive(extracted_content)
            logger.info(f"Chunked into {len(chunks)} chunks. First chunk size: {len(chunks[0].split()) if chunks else 0} words.")
        else:
            chunks = []
            logger.warning("Extracted content is not a string; skipping chunking.")
        # Step 5: Embedding (route by MIME)
        if detected_type.startswith("text"):
            logger.info("Routing to Jina Embeddings v3 for text embedding.")
            embeddings = embed_text_jina(chunks)
        elif detected_type.startswith("image"):
            logger.info("Routing to Nomic Embed Multimodal 7B for image embedding.")
            embedding = embed_image_nomic(contents)
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                embedding = embedding[0]
            embeddings = [embedding] * len(chunks)
        elif "pdf" in detected_type:
            logger.info("Routing to Nomic Embed Multimodal 7B for PDF embedding.")
            embedding = embed_pdf_nomic(contents)
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                embedding = embedding[0]
            embeddings = [embedding] * len(chunks)
        elif detected_type.startswith("audio"):
            logger.info("Routing to Whisper for audio embedding.")
            embeddings = [embed_audio_whisper(contents)] * len(chunks)
        elif detected_type in ("text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
            logger.info("Detected CSV/Excel. Extracting text and embedding as text.")
            # For CSV/Excel, treat as text for embedding
            embeddings = embed_text_jina(chunks)
        elif detected_type in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            logger.info("Detected Word doc. Extracting text and embedding as text.")
            embeddings = embed_text_jina(chunks)
        else:
            logger.warning(f"Unsupported MIME type for embedding: {detected_type}")
            embeddings = []
        # Normalize embeddings for Milvus
        embeddings = normalize_embeddings(embeddings, len(chunks))
        # Pad all embeddings to 1024-dim
        for i in range(len(embeddings)):
            if len(embeddings[i]) < 1024:
                embeddings[i] = embeddings[i] + [0.0] * (1024 - len(embeddings[i]))
            elif len(embeddings[i]) > 1024:
                embeddings[i] = embeddings[i][:1024]
        if embeddings:
            logger.debug(f"First embedding: {embeddings[0][:10]} (type: {type(embeddings[0])}, length: {len(embeddings[0]) if isinstance(embeddings[0], (list, tuple)) else 'N/A'})")
        logger.info(f"Generated {len(embeddings)} embeddings.")
        # Store embeddings in Milvus
        if len(embeddings) > 0 and len(chunks) == len(embeddings):
            from pymilvus import Collection
            collection_name = f"{app_id}_{user_id or 'anon'}"
            ensure_collection(collection_name)
            collection = Collection(collection_name)
            collection.load()
            # Prepare data for Milvus insert (row-oriented, list of dicts)
            insert_data = [
                {
                    "doc_id": str(f"{doc_id}_chunk{i}"),
                    "embedding": list(map(float, embeddings[i])),
                    "content": str(chunks[i]),
                    "metadata": {"source_doc_id": doc_id, "chunk_index": i, "minio_path": minio_path}
                }
                for i in range(len(chunks))
            ]
            # Insert into Milvus
            mr = collection.insert(insert_data)
            logger.info(f"Inserted {len(chunks)} docs into Milvus collection {collection_name}. Insert result: {mr}")
        else:
            logger.warning("No embeddings or chunk/embedding count mismatch; skipping Milvus insert.")
        return IngestResponse(doc_id=doc_id, status="embedded", message=f"File uploaded. Type: {detected_type}. Embedding complete. {len(embeddings)} chunks.")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Ingestion failed: {e}\n{tb}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/query/vector", response_model=VectorQueryResponse)
async def query_vector(
    request: Request,
    file: UploadFile = File(None),
    query: str = Form(None),
    app_id: str = Form(None),
    user_id: str = Form(None),
    filters: Optional[str] = Form(None),
    legacy_body: Optional[VectorQueryRequest] = Body(None)
):
    """
    Vector search endpoint for multimodal queries (text, image, audio, PDF, video) with flexible metadata/temporal filtering.
    Accepts either JSON (text, legacy VectorQueryRequest) or multipart/form-data (file).
    """
    # Legacy JSON body (VectorQueryRequest) for backwards compatibility
    if legacy_body is not None:
        try:
            query_embedding = jina_embedder.encode([legacy_body.query])[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return VectorQueryResponse(results=[])
        app_id = legacy_body.app_id
        user_id = legacy_body.user_id
        filters = legacy_body.filters
        top_k = legacy_body.top_k
    # If JSON, parse as before
    elif request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        query = body.get("query")
        app_id = body.get("app_id")
        user_id = body.get("user_id")
        filters = body.get("filters")
        top_k = body.get("top_k", 10)
        if not query or not app_id or not user_id:
            return VectorQueryResponse(results=[])
        try:
            query_embedding = jina_embedder.encode([query])[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return VectorQueryResponse(results=[])
    # If multipart/form-data, handle file
    elif file is not None:
        if not app_id or not user_id:
            return VectorQueryResponse(results=[])
        contents = await file.read()
        detected_type = detect_file_type(file.filename, contents)
        logger.info(f"Detected file type: {detected_type}")
        # Route to embedding/model pipeline
        if detected_type.startswith("text"):
            text = extract_text(contents)
            query_embedding = jina_embedder.encode([text])[0]
        elif detected_type.startswith("image"):
            query_embedding = embed_image_nomic(contents)[0]
        elif "pdf" in detected_type:
            query_embedding = embed_pdf_nomic(contents)[0]
        elif detected_type.startswith("audio"):
            query_embedding = embed_audio_whisper(contents)[0]
        elif detected_type.startswith("video"):
            # TODO: Implement video embedding (extract key frames, use image embedding)
            logger.warning("Video embedding not implemented; returning placeholder embedding.")
            query_embedding = [0.0]*1024
        else:
            logger.warning(f"Unsupported MIME type for embedding: {detected_type}")
            return VectorQueryResponse(results=[])
        # Parse filters if present
        if filters:
            import json
            try:
                filters = json.loads(filters)
            except Exception:
                filters = None
        top_k = 10
    else:
        logger.warning("No valid input provided to /query/vector.")
        return VectorQueryResponse(results=[])

    # 2. Milvus search (scoped to app_id/user_id)
    collection_name = f"{app_id}_{user_id}"  # Example naming
    try:
        ensure_collection(collection_name)
        collection = Collection(collection_name)
        collection.load()
        # Build filter expression (Milvus supports limited filtering)
        exprs = []
        if filters:
            for k, v in filters.items():
                if k in ("created_after", "created_before"):
                    # Assume metadata.created_at is ISO string
                    if k == "created_after":
                        exprs.append(f"metadata[\"created_at\"] >= '{v}'")
                    else:
                        exprs.append(f"metadata[\"created_at\"] <= '{v}'")
                elif isinstance(v, list):
                    exprs.append(f"metadata[\"{k}\"] in {v}")
                else:
                    exprs.append(f"metadata[\"{k}\"] == '{v}'")
        expr = " and ".join(exprs) if exprs else None
        logger.info(f"Milvus search expr: {expr}")
        # Search in Milvus
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr
        )
        # Format results
        formatted = []
        for hit in results[0]:
            entity = hit.entity
            formatted.append(VectorQueryResult(
                doc_id=entity.get("doc_id", ""),
                score=hit.score,
                content=entity.get("content", ""),
                metadata=entity.get("metadata", {})
            ))
        return VectorQueryResponse(results=formatted)
    except Exception as e:
        logger.error(f"Milvus search failed: {e}")
        return VectorQueryResponse(results=[])

@app.post("/query/graph", response_model=GraphQueryResponse)
async def query_graph(
    request: Request,
    file: UploadFile = File(None),
    query: str = Form(None),
    app_id: str = Form(None),
    user_id: str = Form(None),
    filters: Optional[str] = Form(None),
    graph_expansion: Optional[str] = Form(None),
):
    explain = {}
    # Parse input (JSON or form)
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        query = body.get("query")
        app_id = body.get("app_id")
        user_id = body.get("user_id")
        filters = body.get("filters")
        graph_expansion = body.get("graph_expansion")
        if not query or not app_id or not user_id:
            return {"results": [], "explain": explain}
        try:
            query_embedding = jina_embedder.encode([query])[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {"results": [], "explain": explain}
    elif file is not None:
        if not app_id or not user_id:
            return {"results": [], "explain": explain}
        contents = await file.read()
        detected_type = detect_file_type(file.filename, contents)
        logger.info(f"Detected file type: {detected_type}")
        if detected_type.startswith("text"):
            text = extract_text(contents)
            query_embedding = jina_embedder.encode([text])[0]
        elif detected_type.startswith("image"):
            query_embedding = embed_image_nomic(contents)[0]
        elif "pdf" in detected_type:
            query_embedding = embed_pdf_nomic(contents)[0]
        elif detected_type.startswith("audio"):
            query_embedding = embed_audio_whisper(contents)[0]
        elif detected_type.startswith("video"):
            logger.warning("Video embedding not implemented; returning placeholder embedding.")
            query_embedding = [0.0]*1024
        else:
            logger.warning(f"Unsupported MIME type for embedding: {detected_type}")
            return {"results": [], "explain": explain}
        if filters:
            import json
            try:
                filters = json.loads(filters)
            except Exception:
                filters = None
        if graph_expansion:
            import json
            try:
                graph_expansion = json.loads(graph_expansion)
            except Exception:
                graph_expansion = None
    else:
        logger.warning("No valid input provided to /query/graph.")
        return {"results": [], "explain": explain}

    # --- Phase 2: Weighted edge expansion ---
    # 1. Get edge weights (from config, or from graph_expansion if provided)
    edge_weights = None
    explain = {}
    all_edge_types = None
    if graph_expansion and isinstance(graph_expansion, dict) and "weights" in graph_expansion:
        # Use weights from request if provided
        edge_weights = {k: float(v) for k, v in graph_expansion["weights"].items()}
        all_edge_types = list(graph_expansion["weights"].keys())
    if not edge_weights:
        # Fallback to config
        edge_weights = edge_graph_config_loader.get_app_edge_weights(app_id)
        all_edge_types = list(edge_weights.keys())
    # Only use edge types with weight > 0 for expansion
    edge_types = [k for k, v in edge_weights.items() if v > 0]
    # Always include all edge types in explain, even if weight is zero
    explain["used_edge_types"] = {k: edge_weights.get(k, 0.0) for k in all_edge_types}
    explain["rerank"] = "Nodes/edges prioritized by edge weights"

    # Vector search in Milvus
    collection_name = f"{app_id}_{user_id}"
    try:
        ensure_collection(collection_name)
        collection = Collection(collection_name)
        collection.load()
        exprs = []
        if filters:
            for k, v in filters.items():
                if k in ("created_after", "created_before"):
                    if k == "created_after":
                        exprs.append(f"metadata[\"created_at\"] >= '{v}'")
                    else:
                        exprs.append(f"metadata[\"created_at\"] <= '{v}'")
                elif isinstance(v, list):
                    exprs.append(f"metadata[\"{k}\"] in {v}")
                else:
                    exprs.append(f"metadata[\"{k}\"] == '{v}'")
        expr = " and ".join(exprs) if exprs else None
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=10,
            expr=expr
        )
        formatted = []
        for hit in results[0]:
            entity = hit.entity
            doc_id = entity.get("doc_id", "")
            # --- Neo4j graph expansion ---
            nodes = []
            edges = []
            driver = None
            try:
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                neo4j_user = os.getenv("NEO4J_USER", "neo4j")
                neo4j_password = os.getenv("NEO4J_PASSWORD", "test")
                driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                with driver.session() as session:
                    depth = 1
                    exp_type = "context"
                    time_window = None
                    if graph_expansion:
                        depth = int(graph_expansion.get("depth", 1))
                        exp_type = graph_expansion.get("type", "context")
                        time_window = graph_expansion.get("time_window")
                    # Build Cypher to only expand selected edge types, order by weight
                    edge_types_cypher = ", ".join(f"'{et}'" for et in edge_types)
                    cypher = f"""
                    MATCH (n:Chunk {{doc_id: $doc_id, app_id: $app_id, user_id: $user_id}})
                    CALL apoc.path.subgraphAll(n, {{maxLevel: $depth, relationshipFilter: {edge_types_cypher}}})
                    YIELD nodes, relationships
                    RETURN nodes, relationships
                    """
                    params = {"doc_id": doc_id, "app_id": app_id, "user_id": user_id, "depth": depth}
                    result = session.run(cypher, **params)
                    for record in result:
                        for node in record["nodes"]:
                            nodes.append({
                                "id": node["doc_id"],
                                "label": node.get("label", node["doc_id"]),
                                "type": node.get("type", "chunk")
                            })
                        for rel in record["relationships"]:
                            edges.append({
                                "source": rel.start_node["doc_id"],
                                "target": rel.end_node["doc_id"],
                                "type": rel.type,
                                "weight": edge_weights.get(rel.type, 0)
                            })
            except Exception as e:
                logger.error(f"Neo4j expansion failed for doc_id {doc_id}: {e}")
                nodes = [
                    {"id": doc_id, "label": "Result Chunk", "type": "result"}
                ]
                edges = []
            finally:
                if driver is not None:
                    driver.close()
            # Rerank nodes/edges by cumulative edge weights (simple sum for now)
            node_weights = {}
            for edge in edges:
                node_weights[edge["source"]] = node_weights.get(edge["source"], 0) + edge["weight"]
                node_weights[edge["target"]] = node_weights.get(edge["target"], 0) + edge["weight"]
            # Sort nodes by weight (desc), fallback to original order
            nodes = sorted(nodes, key=lambda n: node_weights.get(n["id"], 0), reverse=True)
            explain["rerank"] = "Nodes/edges prioritized by edge weights"
            graph_context = {"nodes": nodes, "edges": edges}
            formatted.append(GraphQueryResult(
                doc_id=doc_id,
                score=hit.score,
                content=entity.get("content", ""),
                metadata=entity.get("metadata", {}),
                graph_context=graph_context
            ))
        return {"results": formatted, "explain": explain}
    except Exception as e:
        logger.error(f"Milvus/Graph search failed: {e}")
        return {"results": [], "explain": explain}

# ---
# Required Models for Application (for download.py)
#
# Text Embedding:
#   - jinaai/jina-embeddings-v2-base-en (SentenceTransformers)
#   - BAAI/bge-m3 (SentenceTransformers fallback)
# Image/PDF Embedding:
#   - nomic-ai/colnomic-embed-multimodal-7b (ColQwen2_5, ColQwen2_5_Processor)
# Audio Embedding:
#   - openai/whisper-base (Whisper)
#
# All models should be downloaded to the directory specified by the environment variable (e.g., HF_HOME or TRANSFORMERS_CACHE)
# --- 