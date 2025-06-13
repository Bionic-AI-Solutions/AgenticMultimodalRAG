# Agentic RAG Usage: Query Decomposition

> **Edge-Graph Phase Complete:** All edge-graph features (filtering, traceability, OpenAPI schema, usage) are fully implemented, tested, and documented. The system is production-ready and ready for Phase 2: Agentic Query Decomposition. See the tracker for details.

> **Status:** All integration tests (including audio) are passing. The system is production-ready for agentic query decomposition. Whisper model files must be in `/Volumes/ssd/mac/models/openai__whisper-base/`.

## Decomposition API Usage

### Endpoint
POST /agent/query/decompose

### Request Example
```json
{
  "query": "Summarize the main findings from the attached PDF and find related images in the knowledge base.",
  "app_id": "myapp",
  "user_id": "user1",
  "modality": "multimodal",
  "context": {}
}
```

### Response Example
```json
{
  "plan": [
    {
      "step_id": 1,
      "type": "vector_search",
      "modality": "text",
      "parameters": {"query": "Summarize the main findings from the PDF."},
      "dependencies": [],
      "trace": {"source": "llm", "explanation": "Decomposed from user query."}
    },
    {
      "step_id": 2,
      "type": "graph_query",
      "modality": "image",
      "parameters": {"related_to": "findings_from_step_1"},
      "dependencies": [1],
      "trace": {"source": "llm", "explanation": "Find images related to findings."}
    }
  ],
  "traceability": true
}
```

### Plan Schema (Excerpt)
- `step_id`: Unique integer for each step
- `type`: One of [vector_search, graph_query, filter, rerank, tool_call, ...]
- `modality`: text, image, audio, etc.
- `parameters`: Dict of parameters for the step
- `dependencies`: List of step_ids this step depends on
- `trace`: Dict with source, explanation, and any LLM metadata

### LLM Backend Selection
- The system can use either:
  - **OpenAI API**: Set `LLM_BACKEND=openai` in config
  - **Local LLM**: Set `LLM_BACKEND=local` and configure model path
- The API and plan format are identical regardless of backend.

### Multimodal and Complex Query Example
```json
{
  "query": "For the attached audio, extract the main topics, then find all related documents and images.",
  "app_id": "myapp",
  "user_id": "user2",
  "modality": "audio",
  "context": {}
}
```

### Response
```json
{
  "plan": [
    {"step_id": 1, "type": "audio_transcription", "modality": "audio", "parameters": {"file": "audio.mp3"}, "dependencies": [], "trace": {"source": "llm"}},
    {"step_id": 2, "type": "vector_search", "modality": "text", "parameters": {"query": "topics from step 1"}, "dependencies": [1], "trace": {"source": "llm"}},
    {"step_id": 3, "type": "graph_query", "modality": "image", "parameters": {"related_to": "topics from step 1"}, "dependencies": [1], "trace": {"source": "llm"}}
  ],
  "traceability": true
}
```

### Traceability and Extensibility
- Each step includes a `trace` field for explainability.
- The plan format supports future agentic capabilities (tool use, multi-hop, conditional logic).
- The API is stable and production-ready for both OpenAI and local LLMs.

## Next Steps
- See the [Implementation Tracker](tracker.md) and [Implementation Plan](implementation_plan.md) for progress and upcoming work on Agentic Query Decomposition (Phase 2). 