# Phase 2 Implementation Tracker: Vector Search, GraphRAG Query, and Retrieval APIs

| Phase/Task                        | Description/Goal                                 | Status   | Notes/PR |
|------------------------------------|--------------------------------------------------|----------|----------|
| 1. Vector Search API               | /query/vector endpoint, Milvus integration       | ⬜ Todo   |          |
| 2. GraphRAG Query API              | /query/graph endpoint, graph expansion           | ⬜ Todo   |          |
| 3. Direct Retrieval API            | /query/{doc_id} endpoint                         | ⬜ Todo   |          |
| 4. Security & Multi-Tenancy        | Data isolation, auth checks                      | ⬜ Todo   |          |
| 5. Documentation & Usage           | OpenAPI, usage.md, examples                      | ⬜ Todo   |          |
| 6. Unit Tests                      | Mocked tests for all new endpoints               | ⬜ Todo   |          |
| 7. Integration Tests               | Live tests for all new endpoints                 | ⬜ Todo   |          |

---

## Progressive Implementation Plan

1. **Start with /query/vector** (Milvus-only, text queries).
2. Add support for image/audio queries.
3. Implement /query/graph (graph expansion, context/time/semantic).
4. Implement /query/{doc_id} (direct retrieval).
5. Add/verify security and multi-tenancy.
6. Update docs and add usage examples.
7. Write and run unit/integration tests for each endpoint.

---

**Update this tracker as each task is started/completed.** 