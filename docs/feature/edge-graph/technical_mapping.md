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

### Post-Expansion Filtering & Traceability
- **API**: `/query/graph` now accepts filter params (edge type, min weight, metadata) and returns traceability fields in `graph_context`.
- **Logic**: Filtering is applied in Cypher (preferred) or Python fallback. Traceability fields are added to each node/edge.
- **Config**: Allowed filter fields are config-driven (global/app, hot reload).
- **Tests**: Unit/integration for filter logic, traceability, config reload.

### Explainability
- **API**: `explain` field in `/query/graph` response (see `app/main.py`).
- **Logic**: Used edge types/weights, rerank info in response. Trace logic (expansion trace) is TODO.
- **Tests**: Unit: `tests/unit/test_edge_graph_config.py` (explainability). Integration: `tests/integratione2e/test_ingest.py` (API output).

## Data Flows
- User/API -> FastAPI (`app/main.py`) -> Expansion logic -> Neo4j (Cypher) -> Milvus (vector search) -> Response

## Test Mapping
- Unit: `tests/unit/test_edge_graph_config.py` (config, loader, expansion, override, explain, filter, traceability)
- Integration: `tests/integratione2e/test_ingest.py` (ingest, expand, explain, filter, traceability)

## Config
- File: `config/edge_graph.yaml` (dict format, weights, per-app override)

## TODOs
- Post-expansion filtering logic and tests (in progress)
- Expansion trace in explainability output (in progress)
- OpenAPI schema and usage docs update (in progress)
- Admin UI/API for dynamic config (future phase)

## Notes
- All changes are backward compatible.
- Test coverage tracked in `tracker.md`. 