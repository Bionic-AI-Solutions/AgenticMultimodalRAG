from .models import DecompositionPlan, DecompositionStep
from .llm import LLMClient
from typing import Any, Dict, List


class QueryDecomposer:
    """
    Decomposes a user query into a structured DecompositionPlan using LLM or rule-based logic.
    Supports multimodal, agentic, and context-aware decomposition for RAG.
    """

    def __init__(self, llm_backend: str = "openai", use_llm: bool = False):
        self.llm_backend = llm_backend
        self.use_llm = use_llm
        self.llm_client = LLMClient(backend=llm_backend)

    def infer_modality(self, query: str, context: Dict[str, Any]) -> str:
        # Infer modality from context or query (simple heuristic)
        if context and "file" in context:
            fname = context["file"]
            if fname.endswith(".mp3") or fname.endswith(".wav"):
                return "audio"
            if fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png"):
                return "image"
            if fname.endswith(".pdf"):
                return "pdf"
            if fname.endswith(".mp4") or fname.endswith(".mov"):
                return "video"
        # Fallback: use explicit modality or default to text
        return context.get("modality", "text")

    def decompose(
        self, query: str, app_id: str, user_id: str, modality: str = None, context: Dict[str, Any] = None
    ) -> DecompositionPlan:
        if not context:
            context = {}
        if not modality:
            modality = self.infer_modality(query, context)
        if self.use_llm:
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
                        import re

                        m = re.search(r"(\d+)", dep)
                        if m:
                            new_deps.append(int(m.group(1)))
                step["dependencies"] = new_deps
            return DecompositionPlan(**plan_dict)

        # --- Rule-based agentic decomposition ---
        steps: List[Dict[str, Any]] = []
        step_id = 1
        trace_base = {"source": "rule-based", "explanation": "Rule-based agentic decomposition"}
        # Audio: transcription -> vector search -> (optional) graph query
        if modality == "audio":
            steps.append(
                {
                    "step_id": step_id,
                    "type": "audio_transcription",
                    "modality": "audio",
                    "parameters": {"file": context.get("file", "audio.mp3")},
                    "dependencies": [],
                    "trace": {**trace_base, "step": "audio_transcription"},
                }
            )
            step_id += 1
            steps.append(
                {
                    "step_id": step_id,
                    "type": "vector_search",
                    "modality": "text",
                    "parameters": {"query": f"transcription from step {step_id-1}"},
                    "dependencies": [step_id - 1],
                    "trace": {**trace_base, "step": "vector_search"},
                }
            )
            step_id += 1
            steps.append(
                {
                    "step_id": step_id,
                    "type": "graph_query",
                    "modality": "text",
                    "parameters": {"related_to": f"topics from step {step_id-1}"},
                    "dependencies": [step_id - 1],
                    "trace": {**trace_base, "step": "graph_query"},
                }
            )
        # Image: vector search -> (optional) graph query
        elif modality == "image":
            steps.append(
                {
                    "step_id": step_id,
                    "type": "vector_search",
                    "modality": "image",
                    "parameters": {"file": context.get("file", "image.jpg")},
                    "dependencies": [],
                    "trace": {**trace_base, "step": "vector_search"},
                }
            )
            step_id += 1
            steps.append(
                {
                    "step_id": step_id,
                    "type": "graph_query",
                    "modality": "image",
                    "parameters": {"related_to": f"results from step {step_id-1}"},
                    "dependencies": [step_id - 1],
                    "trace": {**trace_base, "step": "graph_query"},
                }
            )
        # PDF: vector search -> (optional) graph query
        elif modality == "pdf":
            steps.append(
                {
                    "step_id": step_id,
                    "type": "vector_search",
                    "modality": "pdf",
                    "parameters": {"file": context.get("file", "doc.pdf")},
                    "dependencies": [],
                    "trace": {**trace_base, "step": "vector_search"},
                }
            )
            step_id += 1
            steps.append(
                {
                    "step_id": step_id,
                    "type": "graph_query",
                    "modality": "pdf",
                    "parameters": {"related_to": f"results from step {step_id-1}"},
                    "dependencies": [step_id - 1],
                    "trace": {**trace_base, "step": "graph_query"},
                }
            )
        # Video: placeholder for future extension
        elif modality == "video":
            steps.append(
                {
                    "step_id": step_id,
                    "type": "video_embedding",
                    "modality": "video",
                    "parameters": {"file": context.get("file", "video.mp4")},
                    "dependencies": [],
                    "trace": {**trace_base, "step": "video_embedding"},
                }
            )
            step_id += 1
            steps.append(
                {
                    "step_id": step_id,
                    "type": "vector_search",
                    "modality": "video",
                    "parameters": {"embedding": f"embedding from step {step_id-1}"},
                    "dependencies": [step_id - 1],
                    "trace": {**trace_base, "step": "vector_search"},
                }
            )
        # Default: text query
        else:
            steps.append(
                {
                    "step_id": step_id,
                    "type": "vector_search",
                    "modality": "text",
                    "parameters": {"query": query},
                    "dependencies": [],
                    "trace": {**trace_base, "step": "vector_search"},
                }
            )
            step_id += 1
            steps.append(
                {
                    "step_id": step_id,
                    "type": "graph_query",
                    "modality": "text",
                    "parameters": {"related_to": f"results from step {step_id-1}"},
                    "dependencies": [step_id - 1],
                    "trace": {**trace_base, "step": "graph_query"},
                }
            )
        # Example: add a rerank step for agentic extension
        # steps.append({
        #     "step_id": step_id+1,
        #     "type": "rerank",
        #     "modality": modality,
        #     "parameters": {"candidates": f"results from step {step_id}"},
        #     "dependencies": [step_id],
        #     "trace": {**trace_base, "step": "rerank"}
        # })
        return DecompositionPlan(plan=steps, traceability=True)
