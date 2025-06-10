# Edge-Graph Technical Mapping

## Requirements to Implementation

### Configurable Edge Types and Priorities
- **Neo4j Schema**: Edge type and weight properties on relationships (see ingestion logic in `app/main.py`).
- **Config**: Per-app edge type config in YAML (`config/edge_graph.yaml`).
- **Loader**: `EdgeGraphConfigLoader` in `app/edge_graph_config.py` (hot reload, validation, merging).
- **Ingestion**: Update logic in `app/main.py` (see ingest endpoint).
- **Tests**: Unit: `tests/unit/test_edge_graph_config.py` (config, loader, hot reload, merge). Integration: `tests/integratione2e/test_ingest.py` (ingest/retrieve).

### Weighted Graph Expansion
- **API**: `/query/graph` in `app/main.py` (accepts weights, explainability, reranking).
- **Logic**: Expansion and reranking by edge weights in `app/main.py`.
- **Overrides**: User/agent weights override app defaults (see API logic).
- **Tests**: Unit: `tests/unit/test_edge_graph_config.py` (override logic). Integration: `tests/integratione2e/test_ingest.py` (weighted expansion, explainability).

### Post-Expansion Filtering
- **API**: Filtering params in `graph_expansion` (planned, not yet implemented).
- **Logic**: To be added (filter expanded nodes/edges by type/weight/metadata).
- **Tests**: TODO (unit/integration for filter logic).

### Explainability
- **API**: `explain` field in `/query/graph` response (see `app/main.py`).
- **Logic**: Used edge types/weights, rerank info in response. Trace logic (expansion trace) is TODO.
- **Tests**: Unit: `tests/unit/test_edge_graph_config.py` (explainability). Integration: `tests/integratione2e/test_ingest.py` (API output).

## Data Flows
- User/API -> FastAPI (`app/main.py`) -> Expansion logic -> Neo4j (Cypher) -> Milvus (vector search) -> Response

## Test Mapping
- Unit: `tests/unit/test_edge_graph_config.py` (config, loader, expansion, override, explain)
- Integration: `tests/integratione2e/test_ingest.py` (ingest, expand, explain)

## Config
- File: `config/edge_graph.yaml` (dict format, weights, per-app override)

## TODOs
- Post-expansion filtering logic and tests
- Expansion trace in explainability output
- OpenAPI schema and usage docs update
- Admin UI/API for dynamic config (future phase)

## Notes
- All changes are backward compatible.
- Test coverage tracked in `tracker.md`. 