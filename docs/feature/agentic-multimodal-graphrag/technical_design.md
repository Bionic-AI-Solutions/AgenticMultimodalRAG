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

## 3. Multimodal Ingestion

- Pluggable pipeline: file upload → type detection → extraction → chunking → embedding → storage
- Embedding models: OpenAI, HuggingFace, CLIP (for images), Whisper (for audio)
- Metadata extraction: Tika, custom extractors

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