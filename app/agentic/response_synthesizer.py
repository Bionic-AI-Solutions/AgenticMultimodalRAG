from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ResponseSynthesisRequest(BaseModel):
    plan: Any  # DecompositionPlan, but avoid import cycle
    execution_trace: List[Dict[str, Any]]
    app_id: str
    user_id: str
    context: Optional[Dict[str, Any]] = None

class ResponseSynthesisResult(BaseModel):
    answer: str
    explanation: str
    supporting_evidence: Optional[List[Dict[str, Any]]] = None
    trace: Optional[List[Dict[str, Any]]] = None

class ResponseSynthesizer:
    """
    Synthesizes a final answer and step-by-step explanation from AgentExecutor results and trace.
    - Uses an LLM (or template for tests) to generate a human-readable answer.
    - Generates a step-by-step explanation from the execution trace.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client  # Optional: inject LLM client for real LLM calls

    def synthesize_answer(self, request: ResponseSynthesisRequest) -> ResponseSynthesisResult:
        # 1. Gather evidence from the final result and trace
        final_result = None
        for step in reversed(request.execution_trace):
            if step.get("result") is not None:
                final_result = step["result"]
                break
        # 2. Generate explanation
        explanation = self.generate_explanation(request.execution_trace)
        # 3. Synthesize answer (LLM or template)
        if self.llm_client:
            # TODO: Use LLM to synthesize answer from evidence and trace
            prompt = self._build_prompt(final_result, explanation, request)
            answer = self.llm_client.generate(prompt)
        else:
            # Template fallback for tests/dev
            answer = self._template_answer(final_result, explanation)
        return ResponseSynthesisResult(
            answer=answer,
            explanation=explanation,
            supporting_evidence=[final_result] if final_result else None,
            trace=request.execution_trace
        )

    def generate_explanation(self, trace: List[Dict[str, Any]]) -> str:
        """
        Generate a step-by-step explanation from the execution trace.
        """
        steps = []
        for step in trace:
            step_id = step.get("step_id")
            step_type = step.get("type")
            result = step.get("result")
            if step.get("skipped"):
                steps.append(f"Step {step_id} ({step_type}) was skipped (condition: {step.get('condition')})")
            else:
                steps.append(f"Step {step_id} ({step_type}): result = {str(result)[:120]}")
        return "\n".join(steps)

    def _build_prompt(self, final_result, explanation, request):
        # TODO: Implement a robust prompt for LLM answer synthesis
        return f"Given the following evidence: {final_result}\nAnd the following reasoning steps:\n{explanation}\nProvide a concise, human-readable answer."

    def _template_answer(self, final_result, explanation):
        if not final_result:
            return "No answer could be synthesized from the available evidence."
        return f"Based on the results: {final_result}\n\nExplanation:\n{explanation}" 