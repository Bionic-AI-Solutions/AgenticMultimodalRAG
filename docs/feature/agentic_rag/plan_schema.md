# Agentic RAG: Decomposition Plan Schema

## DecompositionStep (Pydantic Model)
- `step_id`: int — Unique identifier for the step
- `type`: str — Step type (e.g., vector_search, graph_query, filter, rerank, tool_call, audio_transcription, ...)
- `modality`: str — Data modality (text, image, audio, etc.)
- `parameters`: dict — Parameters for the step (query, filters, etc.)
- `dependencies`: List[int] — Step IDs this step depends on
- `trace`: dict — Trace metadata (source, explanation, LLM prompt, etc.)

## DecompositionPlan (Pydantic Model)
- `plan`: List[DecompositionStep]
- `traceability`: bool — Whether traceability is enabled for this plan

## Example: Multimodal Agentic Plan
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

## Rationale
- **step_id**: Enables explicit dependencies and multi-hop plans
- **type**: Supports extensibility for new agentic behaviors
- **modality**: Enables multimodal workflows
- **parameters**: Flexible for all step types
- **dependencies**: Supports complex, conditional, and multi-hop plans
- **trace**: Ensures explainability and auditability
- **traceability**: Plan-level flag for compliance and debugging 