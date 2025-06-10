# Technical Design Document: Agentic Multimodal GraphRAG

## 1. Data Model

- **User**: id, email, hashed_pw, roles
- **Application**: id, owner_id, config
- **Document**: id, app_id, user_id, type, metadata, storage_uri
- **Chunk/Node**: id, doc_id, content, embedding, graph_edges, timestamps
- **Graph**: Edges for context, time, semantic links

## 2. Vector Storage Strategy

- Each (app, user) pair gets a logical collection/partition in Milvus
- Embeddings stored per chunk/node, with metadata for filtering
- Graph edges stored in Postgres or as Milvus attributes (if supported)

## 3. Multimodal Ingestion & Query

- Pluggable pipeline: file upload or text → type detection (MIME/magic) → extraction (if needed) → chunking (if needed) → embedding → vector search (Milvus)
- Embedding models:
  - Text: Jina Embeddings v2 (SentenceTransformers), fallback BGE-M3
  - Image: Nomic Embed Multimodal 7B
  - Audio: Whisper (transcribe, then embed)
  - PDF: Nomic (first page as image), fallback text extraction
  - Video: (Planned) Extract key frames, use image embedding; or use video embedding model if available
- Query endpoint (/query/vector) supports all modalities in a single endpoint, using MIME type detection for routing.
- Implementation is phased: start with text, then add image/audio/PDF/video support.

## 4. GraphRAG/GraphQuery

- Graph schema: nodes = chunks, edges = semantic/contextual/temporal
- Query: vector search → graph expansion (context, time) → rerank

## 5. Multi-Tenancy & Security

- All data partitioned by app_id and user_id
- AuthN/AuthZ enforced at API and DB layers
- JWT/OAuth2 for secure access

## 6. Scalability & Performance

- Async FastAPI endpoints
- Parallel ingestion and query
- Milvus sharding/partitioning
- Caching for frequent queries

## 7. Testing & Validation

- Unit tests for all core logic (mocked data)
- Integration tests using live services (no mocks)
- Progressive testing at each phase
- Use ENV parameters for environment selection

## Phase 3: Technical Design for GraphRAG Query API (`/query/graph`)

> Previous phase details are preserved above for historical context.

### 1. API Design
- **Endpoint:** `/query/graph`
- **Method:** POST
- **Request:**
  - Accepts text or file (multimodal: image, audio, PDF, video)
  - `app_id`, `user_id`, optional `filters`, `temporal` constraints, and `graph_expansion` parameters (e.g., context window, time window, semantic hops)
- **Response:**
  - List of matching documents/chunks with graph context, scores, and metadata
  - Optionally, graph structure (nodes/edges) for visualization

### 2. Graph Expansion Logic
- **Step 1:** Embed the query (as in `/query/vector`)
- **Step 2:** Vector search in Milvus for initial candidates
- **Step 3:** For each candidate, expand context using Neo4j:
  - Temporal expansion (neighboring time windows)
  - Semantic/contextual expansion (graph edges: references, citations, topic links)
  - Configurable expansion depth/hops
- **Step 4:** Aggregate and rerank results (optionally using graph-aware reranking)

### 3. Integration
- **Milvus:** Used for initial vector search
- **Neo4j:** Used for graph expansion (context, time, semantic)
- **Postgres/Minio:** For metadata and raw content retrieval

### 4. Multimodal Query Support
- Accepts text, image, audio, PDF, video queries (as in `/query/vector`)
- Uses MIME detection and appropriate embedding pipeline
- Graph expansion logic is agnostic to modality (works on chunk/node IDs)

### 5. Security, Scalability, Error Handling
- Enforce app_id/user_id isolation at all query and graph steps
- Async FastAPI endpoint, scalable graph and vector queries
- Use `ValidationResult.raise_if_invalid()` for input validation
- Let validation errors bubble up; only catch unexpected/system errors as 500s

### 6. Testing & Validation
- **Unit tests:** Mock Milvus and Neo4j for all graph expansion logic
- **Integration tests:** Use live Milvus and Neo4j, with sample data for all modalities
- **Progressive testing:** Each phase of graph expansion is testable independently

### 7. Extensibility
- Graph expansion parameters (depth, type, time window) are configurable
- Easy to add new edge types or expansion strategies 