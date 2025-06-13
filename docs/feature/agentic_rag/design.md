# Agentic RAG: Query Decomposition Design

## Overview
This document details the design for the agentic query decomposition system, supporting multimodal, explainable, and extensible agentic RAG workflows. The system is built to be production-ready, backend-agnostic, and easily extensible for future agentic capabilities.

## Key Design Goals
- **Extensibility**: The decomposition plan schema supports new step types, modalities, and agentic behaviors (tool use, multi-hop, conditional logic).
- **Traceability**: Every step in the plan includes trace metadata for explainability and debugging.
- **Multimodal Support**: The system natively handles queries and plans involving text, image, audio, and other modalities.
- **LLM Backend Agnostic**: The QueryDecomposer and LLM wrapper support both OpenAI-compatible APIs and local LLMs (e.g., llama.cpp, vLLM) via a unified interface.
- **Production-Readiness**: The API, schema, and plan format are stable, versioned, and ready for high-scale, multi-user environments.

## Schema Rationale
- **DecompositionStep**: Each step is a Pydantic model with fields for type, modality, parameters, dependencies, and trace.
- **DecompositionPlan**: A list of steps, with explicit dependencies for multi-hop and conditional execution.
- **Trace Field**: Includes LLM source, prompt, and explanation for each step, supporting full explainability.
- **Modality Field**: Ensures each step is aware of the data type it operates on, enabling seamless multimodal workflows.

## LLM Backend Abstraction
- **LLM Wrapper**: All LLM calls go through a backend-agnostic wrapper, which can route to OpenAI API or a local LLM based on config.
- **Prompt Templates**: Prompts are versioned and tested for both backends, ensuring consistent plan output.
- **Testing**: Unit tests use mocked LLM output; integration tests run against both OpenAI and local LLMs.

## Multimodal and Agentic Support
- **Step Types**: The schema supports vector_search, graph_query, filter, rerank, tool_call, audio_transcription, and more.
- **Dependencies**: Steps can depend on outputs of previous steps, enabling complex, multi-hop agentic plans.
- **Extensibility**: New step types and modalities can be added with minimal schema changes.

## Traceability and Explainability
- **Trace Field**: Each step's trace includes the LLM prompt, model, and rationale, supporting full auditability.
- **Plan Traceability**: The entire plan can be traced from user query to final execution, supporting debugging and compliance.

## Production-Readiness
- **API Stability**: The endpoint and schema are versioned and documented.
- **Config-Driven**: LLM backend, prompt templates, and plan schema are all configurable.
- **Scalability**: The system is designed for high concurrency and multi-user environments.

## End-Goal Alignment
- The design supports the vision of a robust, scalable, and explainable Agentic RAG system, ready for real-world deployment and future agentic extensions. 