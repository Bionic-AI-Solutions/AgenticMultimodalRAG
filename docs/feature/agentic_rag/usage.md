# Agentic RAG Usage: Query Decomposition

> **Update (2024-Phase 3b):**
> Advanced agentic behaviors—filter, aggregate, multi-hop, and llm_call—are now fully implemented, tested, and production-ready. All unit and integration tests pass. See the tracker for details.

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

### Response Example (Text Query)
```json
{
  "plan": [
    {
      "step_id": 1,
      "type": "vector_search",
      "modality": "text",
      "parameters": {"query": "What is the summary?"},
      "dependencies": [],
      "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "vector_search"}
    },
    {
      "step_id": 2,
      "type": "graph_query",
      "modality": "text",
      "parameters": {"related_to": "results from step 1"},
      "dependencies": [1],
      "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "graph_query"}
    }
  ],
  "traceability": true
}
```

### Response Example (Audio Query)
```json
{
  "plan": [
    {"step_id": 1, "type": "audio_transcription", "modality": "audio", "parameters": {"file": "audio.mp3"}, "dependencies": [], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "audio_transcription"}},
    {"step_id": 2, "type": "vector_search", "modality": "text", "parameters": {"query": "transcription from step 1"}, "dependencies": [1], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "vector_search"}},
    {"step_id": 3, "type": "graph_query", "modality": "text", "parameters": {"related_to": "topics from step 2"}, "dependencies": [2], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "graph_query"}}
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
    {"step_id": 1, "type": "audio_transcription", "modality": "audio", "parameters": {"file": "audio.mp3"}, "dependencies": [], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "audio_transcription"}},
    {"step_id": 2, "type": "vector_search", "modality": "text", "parameters": {"query": "topics from step 1"}, "dependencies": [1], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "vector_search"}},
    {"step_id": 3, "type": "graph_query", "modality": "image", "parameters": {"related_to": "topics from step 1"}, "dependencies": [1], "trace": {"source": "rule-based", "explanation": "Rule-based agentic decomposition", "step": "graph_query"}}
  ],
  "traceability": true
}
```

### Traceability and Extensibility
- Each step includes a `trace` field for explainability.
- The plan format supports future agentic capabilities (tool use, multi-hop, conditional logic).
- The API is stable and production-ready for both OpenAI and local LLMs.
- **Note:** The system now produces richer, multi-step agentic plans for all modalities, supporting advanced multimodal and agentic workflows.

## Next Steps
- See the [Implementation Tracker](tracker.md) and [Implementation Plan](implementation_plan.md) for progress and upcoming work on Agentic Query Decomposition (Phase 2).

# Agentic RAG Usage: Agentic Execution

## Agentic Plan Execution Endpoint

### Endpoint
POST /agent/execute

### Request Example
```json
{
  "plan": [
    {"step_id": 1, "type": "audio_transcription", "modality": "audio", "parameters": {"file": "audio.mp3"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "vector_search", "modality": "text", "parameters": {"query": "transcription from step 1"}, "dependencies": [1], "trace": {}},
    {"step_id": 3, "type": "graph_query", "modality": "text", "parameters": {"related_to": "topics from step 2"}, "dependencies": [2], "trace": {}},
    {"step_id": 4, "type": "tool_call", "modality": "text", "parameters": {"tool": "search"}, "dependencies": [], "trace": {}}
  ],
  "traceability": true,
  "app_id": "app1",
  "user_id": "user1"
}
```

### Response Example
```json
{
  "final_result": {"tool_result": "[tool output]"},
  "trace": [
    {"step_id": 1, "type": "audio_transcription", "result": {"transcription": "[transcribed text]"}, "trace": {}},
    {"step_id": 2, "type": "vector_search", "result": {"results": [...]}, "trace": {}},
    {"step_id": 3, "type": "graph_query", "result": {"results": [...]}, "trace": {}},
    {"step_id": 4, "type": "tool_call", "result": {"tool_result": "[tool output]"}, "trace": {}}
  ]
}
```

### Notes
- This endpoint executes a full agentic plan, managing dependencies and state.
- Supports multi-step, multimodal, and tool-using agentic plans.
- Trace includes the result of each step for explainability.
- Advanced tool types and agentic behaviors (conditional, rerank, aggregate, etc.) will be supported next. 

# MCP Tool Call Usage & Best Practices

## Example: MCP tool_call Step in a Plan
```json
{
  "step_id": 4,
  "type": "tool_call",
  "modality": "text",
  "parameters": {
    "tool": "mcp",
    "endpoint": "https://mcp.example.com/api/tool",
    "payload": {"query": "What is the weather in Paris?"},
    "headers": {"Authorization": "Bearer <token>"}
  },
  "dependencies": [3],
  "trace": {}
}
```

## Example: Execute a Plan with MCP tool_call
```bash
curl -X POST /agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "plan": [
      {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
      {"step_id": 2, "type": "tool_call", "modality": "text", "parameters": {"tool": "mcp", "endpoint": "https://mcp.example.com/api/tool", "payload": {"query": "foo"}, "headers": {"Authorization": "Bearer <token>"}}, "dependencies": [1], "trace": {}}
    ],
    "traceability": true,
    "app_id": "app1",
    "user_id": "user1"
  }'
```

## Best Practices for Chaining/Combining MCP tool_calls
- **Use outputs as inputs:** Reference previous step results in the payload for MCP tool calls (e.g., use the output of a vector search as the query for an MCP tool).
- **Error handling:** Always check for errors in the MCP response and handle them gracefully in downstream steps.
- **Authentication:** Pass required headers/tokens in the `headers` field of the parameters.
- **Timeouts and retries:** Configure reasonable timeouts and consider retry logic for critical MCP calls.
- **Traceability:** Use the `trace` field to record the MCP endpoint, payload, and response for debugging and auditability.
- **Composability:** MCP tool calls can be chained with other agentic steps (search, graph, LLM, etc.) in any order, enabling complex workflows.
- **Security:** Restrict allowed MCP endpoints/tools as needed to prevent misuse.

> **Tip:** You can chain multiple MCP tool calls, or combine them with search, graph, and LLM steps, to build powerful, composable agentic workflows. 

# Filter Step Usage

## Example: Filter Step in an Agentic Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "filter", "modality": "text", "parameters": {"input_step": 1, "min_score": 0.8, "metadata": {"label": "important"}}, "dependencies": [1], "trace": {}}
  ],
  "traceability": true,
  "app_id": "app1",
  "user_id": "user1"
}
```

## Example Response
```json
{
  "final_result": {"results": [{"doc_id": "doc123", "score": 0.85, "content": "...", "metadata": {"label": "important"}}], "filter_method": "min_score+metadata (extensible)"},
  "trace": [
    {"step_id": 1, "type": "vector_search", "result": {"results": [{"doc_id": "doc123", "score": 0.85, "content": "...", "metadata": {"label": "important"}}, {"doc_id": "doc456", "score": 0.7, "content": "...", "metadata": {"label": "other"}}]}, "trace": {}},
    {"step_id": 2, "type": "filter", "result": {"results": [{"doc_id": "doc123", "score": 0.85, "content": "...", "metadata": {"label": "important"}}], "filter_method": "min_score+metadata (extensible)"}, "trace": {}}
  ]
}
```

## Best Practices
- Use filter steps to select results by score, metadata, or other criteria after search or graph steps.
- Specify the input step to filter (via `input_step` or dependencies).
- Combine filter with rerank, aggregate, or other agentic steps for advanced workflows.
- The filter step output includes a `filter_method` field for traceability. 

# Aggregate Step Usage (in progress)

## Example: Aggregate Step in an Agentic Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "graph_query", "modality": "text", "parameters": {"related_to": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 3, "type": "aggregate", "modality": "text", "parameters": {"input_steps": [1, 2], "method": "union"}, "dependencies": [1, 2], "trace": {}}
  ],
  "traceability": true,
  "app_id": "app1",
  "user_id": "user1"
}
```

## Best Practices
- Use aggregate steps to combine results from multiple search/graph/filter steps.
- Specify input steps and aggregation method clearly.
- The output includes an `aggregate_method` field for traceability.

# Multi-hop Step Usage (in progress)

## Example: Multi-hop Step in an Agentic Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "graph_query", "modality": "text", "parameters": {"query": "foo", "depth": 1}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "multi-hop", "modality": "text", "parameters": {"input_step": 1, "hops": 2}, "dependencies": [1], "trace": {}}
  ],
  "traceability": true,
  "app_id": "app1",
  "user_id": "user1"
}
```

## Best Practices
- Use multi-hop steps for advanced graph traversal and reasoning.
- Specify input step and number of hops.
- The output includes a `multi_hop` field for traceability.

# LLM Call Step Usage (in progress)

## Example: LLM Call Step in an Agentic Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "llm_call", "modality": "text", "parameters": {"input_step": 1, "prompt": "Summarize the results."}, "dependencies": [1], "trace": {}}
  ],
  "traceability": true,
  "app_id": "app1",
  "user_id": "user1"
}
```

## Best Practices
- Use llm_call steps for synthesis, summarization, or advanced reasoning.
- Specify input step and prompt.
- The output includes an `llm_call` field for traceability. 