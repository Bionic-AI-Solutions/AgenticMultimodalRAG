# Agentic RAG: Implementation Tracker

## Phases & Progress

### Phase 1: Complete `edge-graph` Foundation
- [x] **Implement Post-Expansion Filtering**
  - [x] Filtering logic (by type/weight/metadata) in `/query/graph`
  - [x] Unit tests for filtering
  - [x] Integration tests for filtering
- [x] **Implement Full Traceability**
  - [x] Detailed expansion trace in API response
  - [x] Unit tests for traceability
  - [x] Integration tests for traceability
- [x] **Complete Documentation**
  - [x] OpenAPI schema update
  - [x] Usage docs for all `edge-graph` features

> **All deliverables for the edge-graph phase are complete. Ready to proceed to Phase 2: Agentic Query Decomposition.**

### Phase 2: Agentic Query Decomposition (Detailed Tracking)
- [ ] **Schema & Plan Format**
  - [ ] Define Pydantic models for decomposition plan and steps
  - [ ] Document schema in plan_schema.md
- [ ] **QueryDecomposer Component**
  - [ ] Implement core logic in query_decomposer.py
  - [ ] Integrate with OpenAI API and local LLMs (configurable)
  - [ ] Prompt engineering for structured plan output
  - [ ] Fallback rule-based decomposer for tests
- [ ] **API Endpoint**
  - [ ] Add POST /agent/query/decompose endpoint
  - [ ] OpenAPI schema and usage docs
- [ ] **LLM Integration**
  - [ ] Backend-agnostic LLM wrapper (OpenAI/local)
  - [ ] Config-driven backend selection
  - [ ] Prompt templates/examples for both backends
- [ ] **Testing**
  - [ ] Unit tests for decomposer, schema, LLM output parsing
  - [ ] Integration tests for endpoint (OpenAI/local, multimodal)
- [ ] **Documentation**
  - [ ] Usage examples in usage.md
  - [ ] OpenAPI schema for endpoint
  - [ ] Developer notes on extension/customization
  - [ ] Design doc (design.md) for extensibility, traceability, multimodal, backend-agnostic
- [ ] **Design Considerations**
  - [ ] Extensibility, traceability, multimodal, LLM backend-agnostic, robust testing, production-ready

### Phase 3: Agentic Graph Traversal & Tool Use
- [ ] **`AgentExecutor` Component**
  - [ ] State management for multi-step queries
  - [ ] Dynamic calling of `/query/graph`
- [ ] **Testing**
  - [ ] Integration tests for dynamic traversal
  - [ ] E2E tests for complex multi-step queries

### Phase 4: Response Synthesis and Explanation
- [ ] **`ResponseSynthesizer` Component**
  - [ ] LLM-based response synthesis
  - [ ] Explanation generation from trace data
- [ ] **Testing**
  - [ ] E2E tests for answer quality and explanation clarity

## Test Results
| Phase | Unit Tests | Integration Tests |
|-------|------------|-------------------|
| 1     |    ✅      |        ✅         |
| 2     |            |                   |
| 3     |            |                   |
| 4     |            |                   |

## Notes
- This tracker will be updated as each phase progresses.
- Link PRs as available.

## TODOs / Known Issues
- [ ] Unit test mocking for /query/graph: The FastAPI endpoint's is_mocked logic is not being triggered in the current test setup, causing the real expansion logic to run and return empty results. This should be addressed (by patching sys.modules/GraphDatabase or refactoring tests to call expand_graph_with_filters directly) for full unit test coverage. 