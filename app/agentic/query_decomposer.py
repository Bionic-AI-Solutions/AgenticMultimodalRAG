from .models import DecompositionPlan
from typing import Any, Dict

class QueryDecomposer:
    """
    Decomposes a user query into a structured DecompositionPlan using LLM or rule-based logic.
    """
    def __init__(self, llm_backend: str = "openai"):
        self.llm_backend = llm_backend

    def decompose(self, query: str, app_id: str, user_id: str, modality: str, context: Dict[str, Any] = None) -> DecompositionPlan:
        # TODO: Integrate with LLM backend (OpenAI/local) for real decomposition
        # For now, return a simple rule-based plan as a placeholder
        if not context:
            context = {}
        # Example: If modality is audio, add an audio_transcription step
        steps = []
        step_id = 1
        if modality == "audio":
            steps.append({
                "step_id": step_id,
                "type": "audio_transcription",
                "modality": "audio",
                "parameters": {"file": context.get("file", "audio.mp3")},
                "dependencies": [],
                "trace": {"source": "rule-based"}
            })
            step_id += 1
        # Always add a vector_search step
        steps.append({
            "step_id": step_id,
            "type": "vector_search",
            "modality": "text",
            "parameters": {"query": query},
            "dependencies": [1] if modality == "audio" else [],
            "trace": {"source": "rule-based"}
        })
        return DecompositionPlan(plan=steps, traceability=True) 