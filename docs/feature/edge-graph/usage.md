# Edge-Graph Feature: Usage Documentation

## Overview
This document provides usage examples for the edge-graph feature, including API requests, config, and expected responses.

---

## Phase 1: Configurable Edge Type Expansion
- **API Example:**
  ```json
  {
    "query": "Find related documents",
    "graph_expansion": {
      "edge_types": ["context_of", "about_topic"]
    }
  }
  ```
- **Config Example:**
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
- **TODO:** Add more request/response examples and config variations as implemented.

## Phase 2: Weighted Expansion & Reranking
- **Config Example:**
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
- **API Example:**
  ```json
  {
    "query": "Find related documents",
    "graph_expansion": {
      "edge_types": ["context_of", "about_topic"],
      "weights": {"context_of": 1.0, "about_topic": 0.5}
    }
  }
  ```
- **Response Example:**
  ```json
  {
    "results": [...],
    "explain": {
      "used_edge_types": {"context_of": 1.0, "about_topic": 0.5},
      "rerank": "Results prioritized by edge weights"
    }
  }
  ```
- **Troubleshooting:**
  - If weights are missing, defaults are used.
  - If invalid, fallback to default/app config.
  - Check explainability output for which weights were applied.

## Phase 3: App-Specific & Dynamic Configuration
- **TODO:** Add usage for dynamic config, admin UI/API.

## Phase 4: Documentation & Usage
- **TODO:** Add OpenAPI schema and admin/config docs.

## Phase 5: Testing & QA
- **TODO:** Add test usage and feedback loop examples.

## Step-by-Step: Modifying and Using the Edge Types Config

### 1. Locate the Config File
- The config file is typically named `edge_types.yaml` and should be placed in a known config directory (e.g., `config/edge_types.yaml`).
- In Kubernetes, mount this file as a ConfigMap volume at the desired path.

### 2. Edit the Config File
- Use a YAML editor or plain text editor.
- Example: Set global defaults and per-app overrides:
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
- Save the file after making changes.

### 3. Hot Reload in the Application
- The application monitors the config file for changes (using file watcher or polling).
- When you save the file, the app will automatically reload the new config in memory—no restart required.
- In Kubernetes, update the ConfigMap and the file will be updated in the running pod (if mounted as a volume).

### 4. Verify the Change
- Make a `/query/graph` API call and check that the new edge types are used for expansion.
- You can also check logs for a message indicating the config was reloaded.

### 5. Example: Add a New App Override
- To prioritize `citation` edges for a research app:
  ```yaml
  app_overrides:
    research_app:
      - citation
      - context_of
  ```
- Save and verify as above.

### 6. Troubleshooting
- **Config not reloading?** Ensure the file path is correct and the app has permissions to read it.
- **Kubernetes:** Make sure the ConfigMap is mounted as a volume, not as an environment variable.
- **YAML errors:** Use a YAML linter to check for syntax issues.
- **App-specific override not working?** Check that the app name in the config matches the one used in the API call.

---

Update this section as new config options or troubleshooting steps are added.

Update this document as each phase is delivered. 