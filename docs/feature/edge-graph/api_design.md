# Edge-Graph API Design

## Overview
The Edge-Graph API enables configurable, weighted, and explainable graph expansion for RAG queries. It supports per-app edge type priorities, user/agent overrides, weighted expansion, post-expansion filtering, and explainability.

## Endpoint
`POST /query/graph`

## Request Parameters
- `query`: (string/object) The user query (text, image, etc.)
- `graph_expansion`:
  - `edge_types`: List of edge types to consider (optional)
  - `edge_type_weights`: Dict of edge type -> weight (optional)
  - `override_weights`: Dict of edge type -> override weight (optional, per user/agent)
  - `expansion_depth`: Integer, how many hops to expand
  - `post_filter`: Filtering criteria after expansion (edge type, weight, metadata)
  - `explain`: Boolean, return explainability trace
- `app_id`, `user_id`: For multi-app/user isolation

## Example Request
```json
{
  "query": "What are the key findings from the latest video?",
  "graph_expansion": {
    "edge_types": ["semantic", "temporal", "source"],
    "edge_type_weights": {"semantic": 1.0, "temporal": 0.5, "source": 0.2},
    "override_weights": {"semantic": 2.0},
    "expansion_depth": 2,
    "post_filter": {"min_weight": 0.3},
    "explain": true
  },
  "app_id": "app123",
  "user_id": "user456"
}
```

## Example Response
```json
{
  "results": [...],
  "expansion_trace": [
    {"node": "A", "edges": [{"type": "semantic", "weight": 2.0}]},
    {"node": "B", "edges": [{"type": "temporal", "weight": 0.5}]}
  ],
  "explainability": {
    "edge_type_weights": {"semantic": 2.0, "temporal": 0.5, "source": 0.2},
    "post_filter": {"min_weight": 0.3}
  }
}
```

## Usage Notes
- Edge type config and weights are per-app, with user/agent overrides supported.
- Explainability is returned if `explain=true`.
- Filtering is applied after expansion.
- All parameters are optional; defaults are set per app config. 