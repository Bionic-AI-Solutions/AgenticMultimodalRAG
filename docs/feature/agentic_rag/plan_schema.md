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

## Conditional Steps

- `condition`: (optional, str) An expression referencing previous step results. If the condition evaluates to True, the step is executed; otherwise, it is skipped.
- The condition can reference previous steps as `step_X.result` (e.g., `step_1.result['run'] == True`).

### Example: Conditional Step in a Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "tool_call", "modality": "text", "parameters": {"tool": "search"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "tool_call", "modality": "text", "parameters": {"tool": "search"}, "dependencies": [1], "trace": {}, "condition": "step_1.result['run'] == True"}
  ],
  "traceability": true
}
```

### Best Practices
- Use clear, simple conditions referencing only previous step results.
- If a step is skipped, it will appear in the trace with `skipped: true` and the condition string.
- Complex logic (e.g., nested conditions, multi-step dependencies) should be broken into multiple simple steps for clarity and maintainability.
- The executor only allows access to `step_X.result` for previous steps; arbitrary code is not allowed in conditions.

> **Note:** Conditional steps enable dynamic, data-driven agentic plans (e.g., "If the search result is empty, call a fallback tool"). 

## Rerank Steps

- `type: "rerank"`: Rerank results from a previous step (vector_search, graph_query, etc.) using a model or custom logic.
- Specify the input step to rerank via `parameters.input_step` or as the first dependency.
- The output is a new list of results, typically with a `rerank_method` field indicating the method used.

### Example: Rerank Step in a Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "rerank", "modality": "text", "parameters": {"input_step": 1}, "dependencies": [1], "trace": {}}
  ],
  "traceability": true
}
```

### Best Practices
- Always specify the input step to rerank (via `input_step` or dependencies).
- The executor currently uses a simple reverse-order rerank as a placeholder; model-based reranking can be added by extending the executor logic.
- Rerank steps can be chained with other agentic steps (e.g., search → rerank → filter).
- The output includes a `rerank_method` field for traceability.

> **Note:** Rerank steps enable advanced result prioritization and can be extended to use ML models or custom logic as needed. 

## Filter Steps

- `type: "filter"`: Filter results from a previous step (vector_search, graph_query, etc.) by score, metadata, or other criteria.
- Specify the input step to filter via `parameters.input_step` or as the first dependency.
- Filter criteria can include:
  - `min_score`: Minimum score threshold (float)
  - `metadata`: Dict of metadata key/values to match (e.g., `{ "label": "important" }`)

### Example: Filter Step in a Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "filter", "modality": "text", "parameters": {"input_step": 1, "min_score": 0.8, "metadata": {"label": "important"}}, "dependencies": [1], "trace": {}}
  ],
  "traceability": true
}
```

### Best Practices
- Always specify the input step to filter (via `input_step` or dependencies).
- Use clear filter criteria (min_score, metadata, etc.) in parameters.
- Filter steps can be chained with other agentic steps (e.g., search → filter → rerank).
- The output includes a `filter_method` field for traceability.

> **Note:** Filter steps enable advanced result selection and can be extended to support more criteria as needed. 

## Aggregate Steps (in progress)

- `type: "aggregate"`: Combine results from multiple previous steps (e.g., union, intersection, custom aggregation).
- Specify input steps to aggregate via `parameters.input_steps` (list of step_ids) or dependencies.
- Aggregation method can be specified in `parameters.method` (e.g., "union", "intersection").

### Example: Aggregate Step in a Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "graph_query", "modality": "text", "parameters": {"related_to": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 3, "type": "aggregate", "modality": "text", "parameters": {"input_steps": [1, 2], "method": "union"}, "dependencies": [1, 2], "trace": {}}
  ],
  "traceability": true
}
```

### Best Practices
- Use aggregate steps to combine results from multiple search/graph/filter steps.
- Specify input steps and aggregation method clearly.
- The output includes an `aggregate_method` field for traceability.

> **Note:** Aggregate steps enable multi-source result synthesis. (Executor support in progress.)

## Multi-hop Steps (in progress)

- `type: "multi-hop"`: Traverse the graph or knowledge base in multiple, dependent steps.
- Specify the sequence of hops and input/output mapping in parameters.

### Example: Multi-hop Step in a Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "graph_query", "modality": "text", "parameters": {"query": "foo", "depth": 1}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "multi-hop", "modality": "text", "parameters": {"input_step": 1, "hops": 2}, "dependencies": [1], "trace": {}}
  ],
  "traceability": true
}
```

### Best Practices
- Use multi-hop steps for advanced graph traversal and reasoning.
- Specify input step and number of hops.
- The output includes a `multi_hop` field for traceability.

> **Note:** Multi-hop steps enable complex, chained graph reasoning. (Executor support in progress.)

## LLM Call Steps (in progress)

- `type: "llm_call"`: Use an LLM for synthesis, summarization, or reasoning at any step.
- Specify input step(s) and prompt in parameters.

### Example: LLM Call Step in a Plan
```json
{
  "plan": [
    {"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "foo"}, "dependencies": [], "trace": {}},
    {"step_id": 2, "type": "llm_call", "modality": "text", "parameters": {"input_step": 1, "prompt": "Summarize the results."}, "dependencies": [1], "trace": {}}
  ],
  "traceability": true
}
```

### Best Practices
- Use llm_call steps for synthesis, summarization, or advanced reasoning.
- Specify input step and prompt.
- The output includes an `llm_call` field for traceability.

> **Note:** LLM call steps enable flexible, LLM-powered agentic workflows. (Executor support in progress.)