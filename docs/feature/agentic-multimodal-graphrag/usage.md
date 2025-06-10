# Usage Documentation: Agentic Multimodal GraphRAG

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

- **Vector Search**:
  ```bash
  curl -X POST /query/vector -d '{"query": "What is GraphRAG?", "app_id": "123"}'
  ```
- **GraphRAG Query (context, time)**:
  ```bash
  curl -X POST /query/graph -d '{"query": "What happened last week?", "app_id": "123"}'
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