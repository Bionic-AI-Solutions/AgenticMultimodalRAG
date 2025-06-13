from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DecompositionStep(BaseModel):
    """
    A single step in an agentic decomposition plan.
    - step_id: Unique integer for each step (enables explicit dependencies and multi-hop plans)
    - type: Step type (e.g., vector_search, graph_query, filter, rerank, tool_call, audio_transcription, conditional, ...)
      - 'filter': Filter results from previous step(s) by metadata, score, or other criteria. Specify input step(s) and filter criteria in parameters.
    - modality: Data modality (text, image, audio, etc.)
    - parameters: Dict of parameters for the step (query, filters, etc.)
    - dependencies: List of step_ids this step depends on (supports complex, conditional, and multi-hop plans)
    - condition: (optional) Expression or reference to previous step result for conditional execution (e.g., 'step_1.result == "success"')
    - trace: Dict with source, explanation, and any LLM metadata (ensures explainability and auditability)
    """
    step_id: int
    type: str  # e.g., vector_search, graph_query, filter, rerank, tool_call, audio_transcription, conditional, ...
    modality: str  # text, image, audio, etc.
    parameters: Dict[str, Any]
    dependencies: List[int] = []
    condition: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None

class DecompositionPlan(BaseModel):
    """
    A structured plan for agentic query decomposition.
    - plan: List of DecompositionStep objects
    - traceability: Whether traceability is enabled for this plan (compliance/debugging)
    """
    plan: List[DecompositionStep]
    traceability: bool = True 