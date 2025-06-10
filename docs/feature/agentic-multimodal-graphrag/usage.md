# Usage Documentation: Agentic Multimodal GraphRAG

> **Milestone:** Multimodal vector search API (text, image, audio, PDF, video) is complete, fully tested, and documented as of this phase. See the tracker for details.

## 1. Setup

- Clone the repo and install dependencies:
  ```bash
  conda activate new-api-py312
  poetry install
  ```
- Set environment variables for Milvus, S3, Postgres, etc. (see `.env.example`)
- Start Milvus, Postgres, and S3-compatible storage
- Run FastAPI server:
  ```bash
  poetry run uvicorn app.main:app --reload
  ```

## 2. Ingesting Documents

- Supported formats: text, PDF, image, audio
- Example:
  ```bash
  curl -X POST /docs/ingest -F "file=@mydoc.pdf" -F "app_id=123"
  ```
- Metadata is extracted automatically; custom metadata can be provided in the request

## 3. Querying

- **Vector Search (Multimodal)**:
  - You can query with text (JSON) or with a file (image, audio, PDF, video, etc.) using MIME type detection.

  ### Text Query (JSON)
  ```bash
  curl -X POST /query/vector \
    -H "Content-Type: application/json" \
    -d '{"query": "What is GraphRAG?", "app_id": "123", "user_id": "abc"}'
  ```

  ### Image Query (multipart/form-data)
  ```bash
  curl -X POST /query/vector \
    -F "file=@myimage.jpg" \
    -F "app_id=123" \
    -F "user_id=abc"
  ```

  ### Audio Query (multipart/form-data)
  ```bash
  curl -X POST /query/vector \
    -F "file=@myaudio.mp3" \
    -F "app_id=123" \
    -F "user_id=abc"
  ```

  ### PDF Query (multipart/form-data)
  ```bash
  curl -X POST /query/vector \
    -F "file=@mydoc.pdf" \
    -F "app_id=123" \
    -F "user_id=abc"
  ```

  ### Video Query (multipart/form-data, placeholder embedding)
  ```bash
  curl -X POST /query/vector \
    -F "file=@myvideo.mp4" \
    -F "app_id=123" \
    -F "user_id=abc"
  ```

  ### Including Metadata/Filters
  - **JSON:**
    ```json
    {
      "query": "search text",
      "app_id": "123",
      "user_id": "abc",
      "filters": {"doc_type": "pdf", "created_after": "2024-06-01"}
    }
    ```
  - **Form-data:**
    - Pass a stringified JSON for filters:
    ```bash
    curl -X POST /query/vector \
      -F "file=@myimage.jpg" \
      -F "app_id=123" \
      -F "user_id=abc" \
      -F 'filters={"doc_type": "image", "created_after": "2024-06-01"}'
    ```

  ### Error Handling Example (unsupported file type)
  ```bash
  curl -X POST /query/vector \
    -F "file=@myarchive.zip" \
    -F "app_id=123" \
    -F "user_id=abc"
  # Response: {"results": []} and a warning in logs about unsupported MIME type
  ```

  - The endpoint will detect the file type and route to the appropriate embedding/model pipeline (text, image, audio, PDF, video).
  - Optional: Add metadata and temporal filters as form fields or JSON keys as appropriate.

  ### OpenAPI Schema Examples

  #### JSON (Text Query) Request Schema
  ```yaml
  requestBody:
    required: true
    content:
      application/json:
        schema:
          type: object
          properties:
            query:
              type: string
              example: "What is GraphRAG?"
            app_id:
              type: string
              example: "123"
            user_id:
              type: string
              example: "abc"
            filters:
              type: object
              example: {"doc_type": "pdf", "created_after": "2024-06-01"}
          required:
            - query
            - app_id
            - user_id
  ```

  #### multipart/form-data (File Query) Request Schema
  ```yaml
  requestBody:
    required: true
    content:
      multipart/form-data:
        schema:
          type: object
          properties:
            file:
              type: string
              format: binary
              description: File to query (image, audio, PDF, video, etc.)
            app_id:
              type: string
              example: "123"
            user_id:
              type: string
              example: "abc"
            filters:
              type: string
              description: JSON string for filters (optional)
              example: '{"doc_type": "image", "created_after": "2024-06-01"}'
          required:
            - file
            - app_id
            - user_id
  ```

  #### Example Response Schema
  ```yaml
  responses:
    '200':
      description: Successful Response
      content:
        application/json:
          schema:
            type: object
            properties:
              results:
                type: array
                items:
                  type: object
                  properties:
                    doc_id:
                      type: string
                      example: "doc123"
                    score:
                      type: number
                      format: float
                      example: 0.98
                    content:
                      type: string
                      example: "This is a relevant chunk of content."
                    metadata:
                      type: object
                      example: {"created_at": "2024-06-10", "doc_type": "pdf"}
  ```

## 4. Multi-user & Multi-app

- Each user can create multiple applications
- Data is isolated per (user, app) in Milvus and storage
- Use `/auth/login` and `/users/me` to manage authentication

## 5. Extending the Pipeline

- To add new modalities, implement a new extractor and register it in the ingestion pipeline
- To add new graph edge types, update the graph schema and graphquery logic

## 6. Testing

- Run unit tests:
  ```bash
  ENV=test poetry run pytest tests/unit
  ```
- Run integration tests (requires live services):
  ```bash
  ENV=test poetry run pytest tests/integratione2e
  ```

## 7. Recommended Extraction & Chunking Libraries/APIs (2024–2025)

### Extraction
- **PDF/Text:**
  - **Open Source:** PyMuPDF (fitz, AGPL/commercial), pdftext (Apache-2.0, pypdfium2), pdfminer.six, pdfplumber, pypdf
  - **Commercial:** Adobe PDF Services API, Google Document AI, AWS Textract, Microsoft Azure Form Recognizer, GetTxt.AI, Cloudmersive, PDF.co
- **Image OCR:**
  - **Open Source:** PaddleOCR, EasyOCR, pytesseract
  - **Commercial:** Google Vision API, AWS Textract, Azure Computer Vision, Nanonets, Veryfi, Affinda
- **Audio ASR:**
  - **Open Source:** OpenAI Whisper, SpeechRecognition
  - **Commercial:** Google Speech-to-Text, AWS Transcribe, Azure Speech, AssemblyAI, Deepgram

**Why:** These are the most accurate, robust, and widely adopted solutions as of 2024–2025. Commercial APIs offer higher accuracy, more features, and easier integration, but at a cost.

### Chunking (for RAG)
- **Open Source:** LangChain, Haystack, spaCy, Unstructured, Sentence Transformers
- **Commercial:** GetTxt.AI, Cloudmersive, PDF.co, Eden AI, ApyHub

**Why:** LangChain and Unstructured are the most flexible and widely adopted for RAG. Commercial APIs offer easy integration and advanced features for enterprise use.

## 6. GraphRAG Query API Usage (`/query/graph`)

> Now supports detailed Neo4j graph expansion for both context and semantic types. The response includes real graph nodes/edges from Neo4j. Unit and integration tests are expanded for this endpoint.

### Overview
- The `/query/graph` endpoint supports advanced retrieval with context, time, and semantic graph expansion.
- Accepts text or file queries (multimodal), plus graph expansion parameters.

### Example: Text Query with Graph Expansion (JSON)
```bash
curl -X POST /query/graph \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What happened last week?",
    "app_id": "123",
    "user_id": "abc",
    "graph_expansion": {"depth": 2, "type": "semantic", "time_window": "7d"},
    "filters": {"doc_type": "pdf"}
  }'
```

### Example: Image Query with Graph Expansion (multipart/form-data)
```bash
curl -X POST /query/graph \
  -F "file=@myimage.jpg" \
  -F "app_id=123" \
  -F "user_id=abc" \
  -F 'graph_expansion={"depth": 1, "type": "context"}'
```

### Example Response Structure
```json
{
  "results": [
    {
      "doc_id": "doc123",
      "score": 0.97,
      "content": "Relevant chunk...",
      "metadata": {"created_at": "2024-06-10", "doc_type": "pdf"},
      "graph_context": {
        "nodes": [
          {"id": "chunk1", "label": "Context Chunk", "type": "context"},
          {"id": "chunk2", "label": "Temporal Neighbor", "type": "temporal"}
        ],
        "edges": [
          {"source": "doc123", "target": "chunk1", "type": "context"},
          {"source": "doc123", "target": "chunk2", "type": "temporal"}
        ]
      }
    }
  ]
}
```

### OpenAPI Schema Example
```yaml
requestBody:
  required: true
  content:
    application/json:
      schema:
        type: object
        properties:
          query:
            type: string
            example: "What happened last week?"
          app_id:
            type: string
            example: "123"
          user_id:
            type: string
            example: "abc"
          graph_expansion:
            type: object
            properties:
              depth:
                type: integer
                example: 2
              type:
                type: string
                example: "semantic"
              time_window:
                type: string
                example: "7d"
          filters:
            type: object
            example: {"doc_type": "pdf"}
        required:
          - query
          - app_id
          - user_id
    multipart/form-data:
      schema:
        type: object
        properties:
          file:
            type: string
            format: binary
          app_id:
            type: string
          user_id:
            type: string
          graph_expansion:
            type: string
            description: JSON string for graph expansion parameters
            example: '{"depth": 1, "type": "context"}'
          filters:
            type: string
            description: JSON string for filters
        required:
          - file
          - app_id
          - user_id
responses:
  '200':
    description: Successful Response
    content:
      application/json:
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  doc_id:
                    type: string
                  score:
                    type: number
                  content:
                    type: string
                  metadata:
                    type: object
                  graph_context:
                    type: object
                    properties:
                      nodes:
                        type: array
                        items:
                          type: object
                          properties:
                            id:
                              type: string
                            label:
                              type: string
                            type:
                              type: string
                      edges:
                        type: array
                        items:
                          type: object
                          properties:
                            source:
                              type: string
                            target:
                              type: string
                            type:
                              type: string
``` 