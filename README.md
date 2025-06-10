# Agentic Multimodal GraphRAG

This project implements an agentic, multimodal Retrieval-Augmented Generation (RAG) pipeline with GraphRAG capabilities, inspired by Microsoft GraphRAG, Graphiti, and Cognee. It supports multi-application, multi-user data isolation, agentic reasoning, temporal knowledge, high-performance vector search, and multimodal ingestion (text, images, audio, video, documents).

## Samples Folder

A `samples/` folder will be created at the project root to store example files for testing the embedding and ingestion pipeline. The following types of files will be included:

- Image (e.g., `.jpg`, `.png`)
- Video (e.g., `.mp4`)
- Audio (e.g., `.mp3`)
- PDF document (e.g., `.pdf`)
- Text file (e.g., `.txt`)
- CSV file (e.g., `.csv`)
- Word document (e.g., `.docx`)

These files will be used for automated and manual tests of the multimodal ingestion and embedding pipeline.

## Setup

- All service configuration is managed via `.env` files.
- Use the `test` conda environment for all development and testing.

## Running Tests

- Unit tests: `pytest tests/unit`
- Integration tests: `pytest tests/integratione2e`

## Service Integrations

- Milvus (vector DB)
- Minio/S3 (object storage)
- Postgres (metadata)
- Neo4j (graph DB)

## Health Check

A `/health` endpoint is available to check service status and connectivity.
