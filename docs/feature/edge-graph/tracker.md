# Edge-Graph Feature: Implementation Tracker

## Phases & Progress

- [x] **Phase 1: Configurable Edge Types and Priorities**
  - [x] Config loader (YAML, validation)
  - [x] Hot reload (watchdog)
  - [x] Per-app override logic
  - [x] Neo4j schema update (edge type/weight)
  - [x] Ingestion logic
  - [x] Unit tests (config, loader, hot reload)
  - [x] Integration tests (ingest/retrieve)
- [x] **Phase 2: Weighted Graph Expansion**
  - [x] Expansion logic (weighted, reranking)
  - [x] API override (weights in request)
  - [x] Explainability in API response
  - [x] Unit tests (expansion, override)
  - [x] Integration tests (end-to-end, explain)
- [ ] **Phase 3: Post-Expansion Filtering**
  - [ ] Filtering logic (by type/weight/metadata)
  - [ ] Unit tests
  - [ ] Integration tests
- [ ] **Phase 4: Explainability and Traceability**
  - [x] Explainability output (API)
  - [ ] Trace logic (expansion trace)
  - [x] Unit tests (explainability)
  - [x] Integration tests (API output)
- [ ] **Phase 5: Documentation and Usage Examples**
  - [ ] OpenAPI schema update
  - [ ] Usage docs (update for weights, explain)
  - [ ] Integration tests (example-driven)

## Test Coverage
- [x] Unit tests (mocks, config, loader, expansion, override, explain)
- [x] Integration tests (live services, ingest, expand, explain)

## Test Results
| Phase | Unit Tests | Integration Tests |
|-------|------------|------------------|
| 1     |    ✅      |        ✅        |
| 2     |    ✅      |        ✅        |
| 3     |            |                  |
| 4     |    ✅      |        ✅        |
| 5     |            |                  |

## Notes
- Config file: `config/edge_graph.yaml` (dict format, weights, per-app override)
- All major logic and tests for Phases 1 & 2 are complete and passing.
- See test reports for details. Link PRs as available.
- Update this tracker as each phase progresses.

| Phase | Task | Responsible | Status |
|-------|------|-------------|--------|
| 1-2   | Config loader, hot reload, Neo4j schema, weighted expansion, reranking, API override, explainability, tests | Team | ✅ Complete |
| 3     | Add filter params to /query/graph (edge type, weight, metadata) | Dev | ⬜ Not Started |
| 3     | Update OpenAPI docs and usage examples | Dev | ⬜ Not Started |
| 3     | Update graph expansion logic to apply filters | Dev | ⬜ Not Started |
| 3     | Add traceability fields to response | Dev | ⬜ Not Started |
| 3     | Update config and hot reload for new options | Dev | ⬜ Not Started |
| 3     | Add/expand unit tests for filtering/traceability | QA | ⬜ Not Started |
| 3     | Add/expand integration tests for new API behaviors | QA | ⬜ Not Started |
| 3     | Update usage.md and API docs | Dev | ⬜ Not Started | 