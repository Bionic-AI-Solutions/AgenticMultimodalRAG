import requests
from typing import Dict, Any
import os
import re


class LLMClient:
    """
    Backend-agnostic LLM client for generating decomposition plans.
    Supports OpenAI API, local LLMs (Ollama), and a mock backend for testing.
    """

    def __init__(self, backend: str = "openai", config: Dict[str, Any] = None):
        self.backend = backend
        self.config = config or {}
        # Always ensure port is included
        host = self.config.get("ollama_host") or os.getenv("OLLAMA_HOST", "192.168.0.199")
        port = self.config.get("ollama_port") or os.getenv("OLLAMA_PORT", "11434")
        if ":" in host:
            self.ollama_host = host  # already has port
        else:
            self.ollama_host = f"{host}:{port}"
        # Use the correct model name for Ollama, prefer .env or config
        self.ollama_model = self.config.get("ollama_model") or os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

    def generate_plan(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a decomposition plan from a prompt using the configured LLM backend.
        """
        # Strict system prompt for JSON-only output
        system_prompt = (
            "Return a JSON object matching this schema: "
            '{"plan": [{"step_id": 1, "type": "vector_search", "modality": "text", "parameters": {"query": "Summarize the main findings from the PDF."}, "dependencies": [], "trace": {"source": "llm", "explanation": "Decomposed from user query."}}], "traceability": true}'
            " for this query: "
            "Respond ONLY with a valid JSON object. Do NOT include any text, markdown, or explanation."
        )
        if self.backend == "mock":
            # Return a hardcoded plan for testing
            return {
                "plan": [
                    {
                        "step_id": 1,
                        "type": "vector_search",
                        "modality": "text",
                        "parameters": {"query": "Summarize the main findings from the PDF."},
                        "dependencies": [],
                        "trace": {"source": "llm", "explanation": "Decomposed from user query."},
                    },
                    {
                        "step_id": 2,
                        "type": "graph_query",
                        "modality": "image",
                        "parameters": {"related_to": "findings_from_step_1"},
                        "dependencies": [1],
                        "trace": {"source": "llm", "explanation": "Find images related to findings."},
                    },
                ],
                "traceability": True,
            }
        if self.backend == "local":
            # Prepend system prompt
            full_prompt = f"{system_prompt}{prompt}"
            url = f"http://{self.ollama_host}/api/generate"
            payload = {"model": self.ollama_model, "prompt": full_prompt, "stream": False}
            try:
                resp = requests.post(url, json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                # Expect the LLM to return a JSON plan in the 'response' field
                import json as _json

                plan_str = data.get("response", "")
                try:
                    # Try direct JSON parse
                    return _json.loads(plan_str)
                except Exception:
                    # Try to extract JSON block from the response
                    match = re.search(r"\{[\s\S]*\}", plan_str)
                    if match:
                        try:
                            return _json.loads(match.group(0))
                        except Exception:
                            pass
                    # If still not valid, raise with partial response for debugging
                    raise RuntimeError(f"Ollama LLM did not return valid JSON. First 500 chars: {plan_str[:500]}")
            except Exception as e:
                raise RuntimeError(f"Ollama LLM call failed: {e}")
        if self.backend == "openai":
            # TODO: Call OpenAI API with prompt
            raise NotImplementedError("OpenAI backend not implemented yet.")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
