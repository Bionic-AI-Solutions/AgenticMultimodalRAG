# Edge-Graph API Design

## Overview
The Edge-Graph API enables configurable, weighted, filterable, and explainable graph expansion for RAG queries. It supports per-app edge type priorities, user/agent overrides, weighted expansion, post-expansion filtering, traceability, and explainability.

## Endpoint
`POST /query/graph`

## Request Parameters
- `query`: (string/object) The user query (text, image, etc.)
- `graph_expansion`:
  - `edge_types`: List of edge types to consider (optional)
  - `edge_type_weights`: Dict of edge type -> weight (optional)
  - `override_weights`: Dict of edge type -> override weight (optional, per user/agent)
  - `expansion_depth`: Integer, how many hops to expand
  - `post_filter`: Filtering criteria after expansion (edge type, min_weight, metadata)
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
    "post_filter": {"edge_types": ["semantic"], "min_weight": 0.3, "metadata": {"label": "important"}},
    "explain": true
  },
  "app_id": "app123",
  "user_id": "user456"
}
```

## Example Response
```json
{
  "results": [
    {
      "doc_id": "doc123",
      "content": "...",
      "score": 0.98,
      "graph_context": {
        "nodes": [
          {"id": "doc123", "label": "Result Chunk", "type": "result", "expanded_by": "semantic", "config_source": "app"}
        ],
        "edges": [
          {"source": "doc123", "target": "doc456", "type": "semantic", "weight": 0.7, "expanded_by": "semantic", "config_source": "app"}
        ]
      }
    }
  ],
  "expansion_trace": [
    {"node": "doc123", "edges": [{"type": "semantic", "weight": 0.7}]}
  ],
  "explainability": {
    "edge_type_weights": {"semantic": 2.0, "temporal": 0.5, "source": 0.2},
    "post_filter": {"edge_types": ["semantic"], "min_weight": 0.3, "metadata": {"label": "important"}}
  }
}
```

## Usage Notes
- Edge type config and weights are per-app, with user/agent overrides supported.
- Filtering and traceability are returned if requested.
- All parameters are optional; defaults are set per app config.
- Backward compatible with previous API versions. 