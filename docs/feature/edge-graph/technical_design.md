# Edge-Graph Feature: Technical Design

## Overview
This document details the technical design for the edge-graph feature, enabling configurable, weighted, filterable, and explainable graph expansion for RAG.

---

## Phase 1: Configurable Edge Type Expansion
- **Config Schema:**
  - Use a YAML file (e.g., `edge_types.yaml`) for global and per-app edge type priorities.
  - Example structure:
    ```yaml
    default_edge_types:
      - context_of
      - about_topic
      - temporal_neighbor
    app_overrides:
      resume_app:
        - temporal_neighbor
        - context_of
      product_catalog:
        - about_topic
        - context_of
    ```
  - Designed for compatibility with Kubernetes ConfigMap (mount as file, hot reload).
- **Hot Reload Mechanism:**
  - Monitor the YAML config file for changes (e.g., using `watchdog` or periodic polling).
  - On change, reload config in memory without restarting the app.
  - Ensure thread/process safety for concurrent access.
- **API Changes:** `/query/graph` accepts an `edge_types` override parameter in `graph_expansion`.
- **Data Flow:**
  1. On each query, load current config (from memory, not disk).
  2. Determine edge types: use app override if present, else global default, else user/agent override if provided.
  3. Use these edge types for graph expansion.
- **Extensibility:**
  - Add new edge types or app overrides by editing the YAML (ConfigMap in K8s).
  - No restart required due to hot reload.
- **TODO:**
  - Implement config loader with hot reload.
  - Validate config structure and precedence rules.
  - Document config editing and reload process.

## Phase 2: Weighted Expansion & Reranking

### Config Schema
- Support edge weights as floats (default/app-specific):
  ```yaml
  default_edge_types:
    context_of: 1.0
    about_topic: 0.5
    temporal_neighbor: 0.2
  app_overrides:
    resume_app:
      temporal_neighbor: 1.0
      context_of: 0.8
    product_catalog:
      about_topic: 1.0
      context_of: 0.5
  ```

### Loader Changes
- Parse edge weights as floats.
- Expose merged config for app (fallback to default).

### Expansion/Reranking Logic
- When expanding, use edge types with highest weights first.
- Rerank expanded nodes/edges by cumulative edge weights.
- Allow user/agent override in API (future phase).

### Explainability
- API response includes which edge types/weights were used and why.
- Example:
  ```json
  {
    "results": [...],
    "explain": {
      "used_edge_types": {"context_of": 1.0, "about_topic": 0.5}
    }
  }
  ```

### Error Handling
- Validate weights are floats >= 0.
- Fallback to default if app override missing/invalid.

## Phase 3: Post-Expansion Filtering, Traceability, and API Alignment

### Filtering Logic
- **API**: `/query/graph` accepts filter parameters (edge type, min weight, metadata fields) in the request.
- **Implementation**:
  - Prefer Cypher query filters in Neo4j for edge type/weight/metadata filtering.
  - Fallback to Python post-processing if Cypher cannot express the filter.
  - Filtering is applied after expansion, before reranking/explainability.
- **Config-Driven**:
  - Allowed edge types/weights/metadata fields are controlled by config (global/app).
  - Hot reload is supported for new filter options.

### Traceability
- **Response**: Each node/edge in `graph_context` includes:
  - `expanded_by`: Expansion step/config that produced it
  - `edge_type`, `weight`, `config_source`
- **Explainability**: API response includes `used_edge_types`, `post_filter`, and expansion trace.

### Backward Compatibility
- If no filters are provided, all allowed edge types/weights are included (as before).
- Existing clients do not need to change unless they want to use new features.

### Testing
- Unit and integration tests for:
  - Filtering logic (API, Cypher, Python fallback)
  - Traceability fields in response
  - Config-driven filtering and hot reload

## Phase 4: App-Specific & Dynamic Configuration
- **Config Storage:** Support YAML, .env, or DB (for runtime updates).
- **App-Specific:** Allow per-app overrides, fallback to global default.
- **Hot Reload:** Optionally reload config without restart.
- **TODO:** Describe config lookup order, admin UI/API for config changes.

## Phase 5: Documentation & Usage
- **Docs:** Update usage, OpenAPI, and admin/config docs for each phase.
- **TODO:** Add example requests/responses, config editing instructions.

## Phase 6: Testing & QA
- **Testing:** Unit and integration tests for all logic.
- **Feedback:** User feedback loop for continuous improvement.
- **TODO:** List test cases and coverage goals.

---

This design will be expanded and refined as each phase is implemented. 