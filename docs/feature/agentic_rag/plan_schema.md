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
    {"step_id": 1, "type": "audio_transcription", "modality": "audio", "parameters": {"file": "audio.mp3"}, "dependencies": [], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "audio_transcription"}},
    {"step_id": 2, "type": "vector_search", "modality": "text", "parameters": {"query": "topics from step 1"}, "dependencies": [1], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "vector_search"}},
    {"step_id": 3, "type": "graph_query", "modality": "image", "parameters": {"related_to": "topics from step 1"}, "dependencies": [1], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "graph_query"}}
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

> **Note:** The system now produces multi-step, multimodal, and agentic plans for all queries. The schema is extensible for future agentic behaviors (tool use, rerank, filter, conditional, etc.). 

## Advanced Agentic Step Types & Control Flow

The schema supports (or will support) the following advanced step types:
- `tool_call`: Call external APIs/tools (web search, calculator, plugins, etc.)
- `rerank`: Rerank results using a model or custom logic
- `filter`: Filter results based on criteria (metadata, score, etc.)
- `conditional`: Branch logic (if/else) based on previous step results
- `aggregate`: Combine results from multiple steps
- `multi-hop`: Traverse the graph or knowledge base in multiple, dependent steps
- `llm_call`: Use an LLM for synthesis, summarization, or reasoning at any step

### Control Flow Fields
- `condition`: (optional) Expression or reference to previous step result for conditional execution
- `aggregate`: (optional) List of step_ids to aggregate results from
- `tool`: (optional) Tool name or API to call
- `params`: (optional) Tool-specific parameters

> **Note:** The schema is designed for extensibility. New step types and control flow fields can be added as needed. All changes are versioned and documented. 