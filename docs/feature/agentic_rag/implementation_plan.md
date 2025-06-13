# Agentic RAG: Implementation Plan

## Overview
This document outlines the phased, testable implementation plan for the Agentic RAG feature. This feature will build upon the existing `edge-graph` capabilities to introduce a layer of agentic behavior, enabling complex query decomposition, dynamic graph traversal, and reasoned response synthesis.

The user wants to incorporate agentic capabilities into the RAG system. This plan outlines the steps to achieve that, starting from the existing `edge-graph` feature.

## Phases

### Phase 1: Complete `edge-graph` Foundation
This phase focuses on completing the work on the `edge-graph` feature, which is a prerequisite for the agentic layer. The tasks are derived from the `edge-graph` feature's tracker.

- **Tasks:**
  1.  **Implement Post-Expansion Filtering:**
      -   Add filtering logic to `/query/graph` by edge type, weight, and metadata.
      -   Update `app/main.py` and related modules.
      -   Add unit and integration tests for filtering.
  2.  **Implement Full Traceability:**
      -   Enhance the API response to include a detailed expansion trace.
      -   Ensure the `explain` field provides a complete picture of how a result was generated.
  3.  **Complete Documentation:**
      -   Update OpenAPI schema for the `/query/graph` endpoint.
      -   Create comprehensive usage documentation for all `edge-graph` features.

- **Testing:**
  -   Unit tests for all new logic.
  -   Integration tests covering filtering, traceability, and all API parameters.

### Phase 2: Agentic Query Decomposition (Detailed Implementation Plan)

### Objective
Enable the system to take a complex, natural language user query and decompose it into a structured plan of sub-queries or graph operations, ready for agentic execution in a multimodal RAG system.

### Implementation Steps

1. **Schema & Plan Format**
   - Define a Pydantic model for the decomposition plan (e.g., `DecompositionStep`, `DecompositionPlan`).
   - Each step includes: type (graph_query, vector_search, filter, rerank, tool_call, etc.), parameters, dependencies, modality, and traceability fields.
   - Support for multimodal queries (text, image, audio, etc.) and context-awareness (app_id, user_id).
   - Document the schema in `docs/feature/agentic_rag/plan_schema.md`.

2. **QueryDecomposer Component**
   - Implement `app/agentic/query_decomposer.py`.
   - Integrate with both external OpenAI-compatible APIs and local LLMs (configurable backend).
   - Use prompt engineering to guide the LLM to output structured plans in the defined schema.
   - Support for few-shot and chain-of-thought prompting for complex queries.
   - Provide a fallback rule-based decomposer for unit tests and offline mode.

3. **API Endpoint**
   - Add a new FastAPI endpoint: `POST /agent/query/decompose`.
   - Accepts: user query, app_id, user_id, modality, and optional context.
   - Returns: structured decomposition plan (JSON, validated against schema).
   - Add OpenAPI schema and usage documentation.

4. **LLM Integration**
   - Implement a backend-agnostic LLM wrapper in `app/agentic/llm.py`.
   - Support both OpenAI API (via standard REST) and local LLMs (e.g., llama.cpp, vLLM, etc.).
   - Configuration-driven selection of backend.
   - Provide prompt templates and examples for both backends.

5. **Testing**
   - Unit tests for the decomposer logic, schema validation, and LLM output parsing (mocked LLM responses).
   - Integration tests for the API endpoint, including edge cases and multimodal queries (using both OpenAI and local LLM backends).
   - All tests to be placed in `tests/unit/agentic/` and `tests/integratione2e/agentic/`.

6. **Documentation**
   - Usage examples for the new endpoint and plan format in `docs/feature/agentic_rag/usage.md`.
   - OpenAPI schema for the new endpoint.
   - Developer notes on extending/customizing the decomposition logic and LLM integration.
   - Design document in `docs/feature/agentic_rag/design.md` covering extensibility, traceability, multimodal support, and backend-agnostic LLM integration.

7. **Design Considerations**
   - Extensibility: Plan format supports future agentic capabilities (tool use, multi-hop, conditional logic).
   - Traceability: Each step in the plan is traceable for debugging and explainability.
   - Multimodal: Decomposer handles queries referencing different modalities.
   - LLM Backend Agnostic: System can use OpenAI API or local LLMs interchangeably.
   - Testing: Mock LLM output for unit tests; use live LLM for integration tests.
   - End-goal: System is production-ready, not a prototype or gradual upgrade.

8. **Progressive Delivery**
   - Each step is independently testable and can be merged after passing all tests.
   - Backward compatibility and migration steps documented as needed.

---

## Deliverables
- Decomposition plan schema and models
- QueryDecomposer component
- LLM backend integration (OpenAI and local)
- `/agent/query/decompose` endpoint
- Unit and integration tests
- Full documentation and usage examples
- Design document covering all considerations

### Phase 3: Agentic Graph Traversal & Tool Use
This phase empowers the agent to interact with the knowledge graph dynamically.

- **Concept:** The agent executes the plan from Phase 2, traversing the graph and potentially using other tools to gather information.
- **Implementation:**
  1.  **`AgentExecutor` Component:**
      -   Develop a component that manages the state of a multi-step query.
      -   The executor will call `/query/graph` multiple times, using the results of one call to inform the parameters of the next (e.g., using `override_weights`, `post_filter`).
  2.  **State Management:**
      -   Implement a mechanism to store the context and history of the agent's traversal.

- **Testing:**
  -   Integration tests for multi-step queries that require dynamic graph traversal.
  -   E2E tests simulating a user asking a complex question that requires several steps to answer.

### Phase 4: Response Synthesis and Explanation
The final phase focuses on generating a coherent and explainable answer.

- **Concept:** The agent synthesizes the information gathered from all steps into a single, comprehensive response and can explain its reasoning.
- **Implementation:**
  1.  **`ResponseSynthesizer` Component:**
      -   Create a component that takes the collected evidence from the `AgentExecutor` and generates a human-readable answer using an LLM.
  2.  **Explanation Generation:**
      -   Leverage the traceability data from the `edge-graph` API calls to construct a step-by-step explanation of how the answer was derived.
      -   The explanation will be part of the final API response.

- **Testing:**
  -   E2E tests with complex questions, evaluating the quality of the synthesized answer and the clarity of the explanation. 