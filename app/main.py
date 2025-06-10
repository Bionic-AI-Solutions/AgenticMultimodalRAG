from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from pydantic import BaseModel
import logging
import os
from pymilvus import connections, exceptions as milvus_exceptions
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

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Move the lifespan definition here (before app = FastAPI(...))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG System...")
    # ... (existing lifespan code) ...
    # Set HuggingFace cache to /Volumes/ssd/models
    os.environ["HF_HOME"] = "/Volumes/ssd/models"
    os.environ["TRANSFORMERS_CACHE"] = "/Volumes/ssd/models"
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

# Milvus connection utility
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

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
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
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
jina_embedder = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)

# Fallback: Sentence Transformers (BGE-M3, etc.)
sbert_embedder = SentenceTransformer("BAAI/bge-m3")

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
    model_path = "/Volumes/ssd/models/nomic-ai/colnomic-embed-multimodal-7b"
    processor_path = "/Volumes/ssd/models/nomic-ai/colnomic-embed-multimodal-7b"
    if nomic_model is None or nomic_processor is None:
        logger.info(f"Checking for Nomic model at {model_path}")
        if not os.path.exists(model_path):
            logger.info("Nomic model not found locally. Downloading from HuggingFace...")
            huggingface_hub.snapshot_download(repo_id="nomic-ai/colnomic-embed-multimodal-7b", local_dir=model_path, local_dir_use_symlinks=False)
        logger.info("Loading Nomic model...")
        nomic_model = ColQwen2_5.from_pretrained(model_path)
        logger.info("Loading Nomic processor with use_fast=False...")
        nomic_processor = ColQwen2_5_Processor.from_pretrained(processor_path, use_fast=False)
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
        return image_embeddings.cpu().tolist()
    except Exception as e:
        logger.error(f"Nomic image embedding failed: {e}")
        return [[0.0]*1024]

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
        return image_embeddings.cpu().tolist()
    except Exception as e:
        logger.error(f"Nomic PDF embedding failed: {e}")
        return [[0.0]*1024]

# Audio embedding with Whisper + text embedding
whisper_model = whisper.load_model("base")
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
            embeddings = [embed_image_nomic(contents)] * len(chunks)
        elif "pdf" in detected_type:
            logger.info("Routing to Nomic Embed Multimodal 7B for PDF embedding.")
            embeddings = [embed_pdf_nomic(contents)] * len(chunks)
        elif detected_type.startswith("audio"):
            logger.info("Routing to Whisper for audio embedding.")
            embeddings = [embed_audio_whisper(contents)] * len(chunks)
        else:
            logger.warning(f"Unsupported MIME type for embedding: {detected_type}")
            embeddings = []
        logger.info(f"Generated {len(embeddings)} embeddings.")
        # TODO: Store embeddings in Milvus, metadata in Postgres
        return IngestResponse(doc_id=doc_id, status="embedded", message=f"File uploaded. Type: {detected_type}. Embedding complete. {len(embeddings)} chunks.")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Ingestion failed: {e}\n{tb}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

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