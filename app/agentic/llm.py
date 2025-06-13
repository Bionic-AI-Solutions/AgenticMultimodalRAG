from typing import Dict, Any

class LLMClient:
    """
    Backend-agnostic LLM client for generating decomposition plans.
    Supports OpenAI API and local LLMs (e.g., llama.cpp, vLLM).
    """
    def __init__(self, backend: str = "openai", config: Dict[str, Any] = None):
        self.backend = backend
        self.config = config or {}

    def generate_plan(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a decomposition plan from a prompt using the configured LLM backend.
        TODO: Implement OpenAI and local LLM integration.
        """
        if self.backend == "openai":
            # TODO: Call OpenAI API with prompt
            raise NotImplementedError("OpenAI backend not implemented yet.")
        elif self.backend == "local":
            # TODO: Call local LLM (e.g., llama.cpp, vLLM)
            raise NotImplementedError("Local LLM backend not implemented yet.")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}") 