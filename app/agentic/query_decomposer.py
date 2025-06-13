from .models import DecompositionPlan
from .llm import LLMClient
from typing import Any, Dict

class QueryDecomposer:
    """
    Decomposes a user query into a structured DecompositionPlan using LLM or rule-based logic.
    """
    def __init__(self, llm_backend: str = "openai", use_llm: bool = False):
        self.llm_backend = llm_backend
        self.use_llm = use_llm
        self.llm_client = LLMClient(backend=llm_backend)

    def decompose(self, query: str, app_id: str, user_id: str, modality: str, context: Dict[str, Any] = None) -> DecompositionPlan:
        if not context:
            context = {}
        if self.use_llm:
            # Compose prompt (simple for now)
            prompt = f"Decompose the following query into a structured plan: {query}"
            plan_dict = self.llm_client.generate_plan(prompt)
            # --- Post-process dependencies to ensure they are integers ---
            for step in plan_dict.get("plan", []):
                deps = step.get("dependencies", [])
                new_deps = []
                for dep in deps:
                    if isinstance(dep, int):
                        new_deps.append(dep)
                    elif isinstance(dep, str):
                        # Accept formats like 'output_1', 'step_1', '1'
                        import re
                        m = re.search(r"(\d+)", dep)
                        if m:
                            new_deps.append(int(m.group(1)))
                    # else: ignore or raise error (could log)
                step["dependencies"] = new_deps
            return DecompositionPlan(**plan_dict)
        # Rule-based fallback
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