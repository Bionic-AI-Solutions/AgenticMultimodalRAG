from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DecompositionStep(BaseModel):
    """
    A single step in an agentic decomposition plan.
    - step_id: Unique integer for each step (enables explicit dependencies and multi-hop plans)
    - type: Step type (e.g., vector_search, graph_query, filter, rerank, tool_call, audio_transcription, ...)
    - modality: Data modality (text, image, audio, etc.)
    - parameters: Dict of parameters for the step (query, filters, etc.)
    - dependencies: List of step_ids this step depends on (supports complex, conditional, and multi-hop plans)
    - trace: Dict with source, explanation, and any LLM metadata (ensures explainability and auditability)
    """
    step_id: int
    type: str  # e.g., vector_search, graph_query, filter, rerank, tool_call, audio_transcription, ...
    modality: str  # text, image, audio, etc.
    parameters: Dict[str, Any]
    dependencies: List[int] = []
    trace: Optional[Dict[str, Any]] = None

class DecompositionPlan(BaseModel):
    """
    A structured plan for agentic query decomposition.
    - plan: List of DecompositionStep objects
    - traceability: Whether traceability is enabled for this plan (compliance/debugging)
    """
    plan: List[DecompositionStep]
    traceability: bool = True 