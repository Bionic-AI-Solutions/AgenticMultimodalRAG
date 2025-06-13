from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DecompositionStep(BaseModel):
    step_id: int
    type: str  # e.g., vector_search, graph_query, filter, rerank, tool_call, audio_transcription, ...
    modality: str  # text, image, audio, etc.
    parameters: Dict[str, Any]
    dependencies: List[int] = []
    trace: Optional[Dict[str, Any]] = None

class DecompositionPlan(BaseModel):
    plan: List[DecompositionStep]
    traceability: bool = True 