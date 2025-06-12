# Edge-Graph Feature: Implementation Plan

## Overview
This document outlines the phased, testable implementation plan for the Edge-Graph feature, enabling configurable, weighted, and explainable graph expansion with per-app edge type priorities, user/agent overrides, weighted expansion, post-expansion filtering, and explainability.

## Phases

### Phase 1: Configurable Edge Types and Priorities
- Add support for per-app configurable edge types and priorities in the graph schema.
- Update Neo4j schema and ingestion logic to store edge type metadata and weights.
- Unit tests: Edge type config parsing, storage, and retrieval.
- Integration tests: Ingest and retrieve edge types with priorities.

### Phase 2: Weighted Graph Expansion
- Implement weighted expansion logic in the graph query API.
- Allow user/agent overrides for edge type weights at query time.
- Unit tests: Weighted expansion logic, override handling.
- Integration tests: End-to-end weighted expansion with overrides.

### Phase 3: Post-Expansion Filtering, Traceability, and API Alignment

### 1. Requirements
- Support filtering of expanded graph results by:
  - Edge type (e.g., only `context_of`, `temporal_neighbor`)
  - Edge weight (e.g., only edges with weight > threshold)
  - Node/edge metadata (e.g., time, label, custom fields)
- Ensure traceability: every result should include provenance (how/why it was expanded, edge types/weights used).
- API must align with integration/unit test expectations (edge types, explainability, config-driven behavior).

### 2. Implementation Steps
1. **API Update**
   - Add filter parameters to `/query/graph` (edge types, min weight, metadata fields).
   - Update OpenAPI docs and usage examples.
2. **Graph Expansion Logic**
   - Update graph expansion code to apply filters after expansion.
   - Ensure only allowed edge types/weights are included in results.
   - Add traceability fields to each node/edge in the response.
3. **Config & Hot Reload**
   - Allow per-app and global config for allowed edge types/weights.
   - Ensure config reload works for new filter options.
4. **Testing**
   - Add/expand unit tests for filtering, traceability, and config overrides.
   - Add/expand integration tests for all new API behaviors.
5. **Docs & Usage**
   - Update usage.md and API docs for new filter/traceability features.

### 3. Technical Notes
- Filtering should be efficient (prefer Cypher query filters in Neo4j, fallback to Python post-processing if needed).
- Traceability: include `expanded_by`, `edge_type`, `weight`, and config source in each result.
- Backward compatibility: maintain support for existing API clients.

### 4. Deliverables
- Updated API endpoints and OpenAPI docs
- Updated config and hot reload logic
- Full unit/integration test coverage for new features
- Updated documentation and usage examples

### Phase 4: Explainability and Traceability
- Implement explainability: return edge type/weight breakdown and expansion trace in API responses.
- Unit tests: Explainability output, trace correctness.
- Integration tests: API explainability in live queries.

### Phase 5: Documentation and Usage Examples
- Update OpenAPI schema, usage docs, and examples.
- Integration tests: Example-driven tests for all new API parameters and behaviors.

## Testing
- All phases require both unit and integration tests.
- Unit tests use mocks; integration tests use live Neo4j and Milvus services.
- Test coverage tracked in `tracker.md`.

## Progressive Delivery
- Each phase is independently testable and can be merged after passing all tests.
- Backward compatibility and migration steps documented as needed.

---

## Phase 1: Configurable Edge Type Expansion
- Add config (YAML/.env/DB) for default and per-app edge type weights/priorities.
- Update `/query/graph` to use default edge types, with user/agent override.
- Add API parameter for edge type selection/override.
- Unit/integration tests for config loading, override, and expansion logic.

## Phase 2: Weighted Expansion & Reranking

**Goal:**
- Enable weighted edge-type expansion and reranking in graph queries, with explainability.

**Steps:**
1. Extend YAML config to support edge weights (global and per-app):
   - Example:
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
2. Update config loader to parse and expose edge weights.
3. Update `/query/graph` to use weights for expansion and reranking:
   - Prioritize expansion by weight.
   - Rerank results using edge weights.
4. Add explainability to API response (which edge types/weights were used and why).
5. Add/extend unit and integration tests for weighted logic.
6. Update usage documentation and OpenAPI schema.

**Deliverables:**
- Updated config schema and loader
- Weighted expansion/reranking logic in API
- Explainability in API response
- Unit/integration tests
- Updated docs

**Test Plan:**
- Unit tests for loader and expansion logic
- Integration tests for weighted expansion/reranking
- API response explainability validation

## Phase 3: App-Specific & Dynamic Configuration
- Support per-app edge type configs (YAML/DB).
- Hot-reload or admin UI for runtime config changes.
- Tests for app-specific overrides and dynamic config.

## Phase 4: Documentation & Usage
- Update usage docs, OpenAPI schema, and examples.
- Add admin/config documentation.

## Phase 5: Testing & QA
- Full unit and integration test coverage for all new logic.
- User feedback loop for continuous improvement.

---

Each phase is testable and can be delivered incrementally.

---

## Milvus Ingestion Best Practices (Explicit Note)

### Summary Table
| Requirement         | Milvus Best Practice   | Our Implementation (should be) |
|--------------------|-----------------------|-------------------------------|
| Embeddings format  | Flat list of floats   | Flat list of floats           |
| Data structure     | List of lists         | List of lists                 |
| Field alignment    | Same length           | Same length                   |
| Modalities         | All as embeddings     | All as embeddings             |
| Insert method      | collection.insert     | collection.insert             |
| Indexing           | After insert          | After insert                  |
| Error handling     | Type/shape checks     | Type/shape checks             |

### Actionable Guidance
- **Keep ingestion simple:**
  - Extract embeddings as flat float lists.
  - Build column-oriented lists for all fields.
  - Insert with a single `collection.insert([ids, metas, embeddings, ...])`.
  - Only add complexity if a modality truly requires it (rare).
- **If you see code that:**
  - Handles each modality with a different insert path,
  - Converts embeddings to nested lists,
  - Does row-wise instead of column-wise insertion,
  - Or adds unnecessary abstraction,
  - **Refactor to the simple, column-oriented, flat-list approach.** 