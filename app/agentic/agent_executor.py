from typing import Any, Dict, List, Optional
from .models import DecompositionPlan, DecompositionStep
import requests

class AgentExecutor:
    """
    Executes a DecompositionPlan by dynamically traversing the graph and invoking tools/APIs as needed.
    Supports multi-step, agentic, and multimodal plans for RAG.
    - Supports tool_call steps, including MCP (Model Context Protocol) tool calls.
    - Supports conditional steps: if a step has a 'condition', it is only executed if the condition evaluates to True (using previous step results).
    - Supports rerank steps: rerank results from previous step(s) using a model or custom logic (currently: simple reverse order as placeholder).
    - Supports filter steps: filter results from previous step(s) by score, metadata, or other criteria (currently: min_score and metadata key/values).
    - For MCP tool_call: expects parameters.tool == 'mcp', parameters.endpoint, parameters.payload, and optional parameters.headers.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.state: Dict[str, Any] = {}
        self.trace: List[Dict[str, Any]] = []

    def execute_plan(self, plan: DecompositionPlan, app_id: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute each step in the plan, managing dependencies and state.
        Returns a structured result and execution trace.
        """
        self.state = {}
        self.trace = []
        step_results = {}
        for step in plan.plan:
            # --- Conditional execution logic ---
            if step.condition:
                if not self.evaluate_condition(step.condition, step_results):
                    self.trace.append({"step_id": step.step_id, "type": step.type, "skipped": True, "condition": step.condition, "trace": step.trace})
                    continue
            result = self.execute_step(step, app_id, user_id, context, step_results)
            step_results[step.step_id] = result
            self.trace.append({"step_id": step.step_id, "type": step.type, "result": result, "trace": step.trace})
        # Final result is the last non-skipped step
        final_result = None
        for step in reversed(plan.plan):
            if step_results.get(step.step_id) is not None:
                final_result = step_results[step.step_id]
                break
        return {"final_result": final_result, "trace": self.trace}

    def evaluate_condition(self, condition: str, step_results: Dict[int, Any]) -> bool:
        """
        Evaluate a simple condition string, e.g., 'step_1.result == "success"'.
        Only allows access to step_X.result for previous steps.
        """
        # Build a safe local context
        local_vars = {}
        for step_id, result in step_results.items():
            local_vars[f"step_{step_id}"] = {"result": result}
        try:
            return bool(eval(condition, {"__builtins__": {}}, local_vars))
        except Exception:
            return False

    def execute_step(self, step: DecompositionStep, app_id: str, user_id: str, context: Optional[Dict[str, Any]], step_results: Dict[int, Any]) -> Any:
        """
        Execute a single step. Handles vector_search, graph_query, tool_call, etc.
        This is a scaffold: expand with real logic for each step type.
        """
        if step.type == "vector_search":
            # Example: call /query/vector endpoint
            payload = {"query": step.parameters.get("query"), "app_id": app_id, "user_id": user_id}
            resp = requests.post(f"{self.base_url}/query/vector", json=payload)
            return resp.json()
        elif step.type == "graph_query":
            # Example: call /query/graph endpoint
            payload = {"query": step.parameters.get("related_to", step.parameters.get("query")), "app_id": app_id, "user_id": user_id}
            resp = requests.post(f"{self.base_url}/query/graph", json=payload)
            return resp.json()
        elif step.type == "audio_transcription":
            # Placeholder: simulate audio transcription
            return {"transcription": "[transcribed text]"}
        elif step.type == "tool_call":
            tool = step.parameters.get("tool")
            if tool == "mcp":
                endpoint = step.parameters.get("endpoint")
                payload = step.parameters.get("payload")
                headers = step.parameters.get("headers", {})
                try:
                    resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    return {"error": f"MCP tool_call failed: {str(e)}"}
            # Placeholder for other tool types
            return {"tool_result": "[tool output]"}
        elif step.type == "rerank":
            # Rerank results from previous step(s). For now, just reverse the order as a placeholder.
            # TODO: Implement model-based reranking.
            input_step_id = step.parameters.get("input_step") or (step.dependencies[0] if step.dependencies else None)
            if input_step_id is None or input_step_id not in step_results:
                return {"error": "No input step for rerank"}
            input_result = step_results[input_step_id]
            # Support both /query/vector and /query/graph style results
            results = input_result.get("results") if isinstance(input_result, dict) else None
            if results and isinstance(results, list):
                reranked = list(reversed(results))
                return {"results": reranked, "rerank_method": "reverse_order (placeholder)"}
            return {"error": "No results to rerank"}
        elif step.type == "filter":
            # Filter results from previous step(s) by score, metadata, etc.
            # TODO: Make filter logic extensible for more criteria.
            input_step_id = step.parameters.get("input_step") or (step.dependencies[0] if step.dependencies else None)
            if input_step_id is None or input_step_id not in step_results:
                return {"error": "No input step for filter"}
            input_result = step_results[input_step_id]
            results = input_result.get("results") if isinstance(input_result, dict) else None
            if not results or not isinstance(results, list):
                return {"error": "No results to filter"}
            min_score = step.parameters.get("min_score")
            metadata_filter = step.parameters.get("metadata")
            filtered = results
            if min_score is not None:
                filtered = [r for r in filtered if r.get("score", 0) >= min_score]
            if metadata_filter:
                for k, v in metadata_filter.items():
                    filtered = [r for r in filtered if r.get("metadata", {}).get(k) == v]
            return {"results": filtered, "filter_method": "min_score+metadata (extensible)"}
        elif step.type == "conditional":
            # Conditional step is a no-op; logic handled in execute_plan
            return None
        elif step.type == "aggregate":
            # Aggregate results from multiple input steps (union, intersection, etc.)
            input_step_ids = step.parameters.get("input_steps") or step.dependencies
            if not input_step_ids:
                return {"error": "No input steps for aggregate"}
            input_results = []
            for sid in input_step_ids:
                res = step_results.get(sid)
                if res and isinstance(res, dict) and isinstance(res.get("results"), list):
                    input_results.append(res["results"])
            if not input_results:
                return {"error": "No results to aggregate"}
            method = step.parameters.get("method", "union")
            if method == "union":
                # Merge all unique results by doc_id
                seen = set()
                union_results = []
                for results in input_results:
                    for r in results:
                        doc_id = r.get("doc_id")
                        if doc_id and doc_id not in seen:
                            union_results.append(r)
                            seen.add(doc_id)
                return {"results": union_results, "aggregate_method": "union"}
            elif method == "intersection":
                # Only include results present in all input sets (by doc_id)
                doc_id_sets = [set(r["doc_id"] for r in results if "doc_id" in r) for results in input_results]
                common_ids = set.intersection(*doc_id_sets) if doc_id_sets else set()
                intersection_results = [r for results in input_results for r in results if r.get("doc_id") in common_ids]
                # Remove duplicates
                seen = set()
                final_results = []
                for r in intersection_results:
                    doc_id = r.get("doc_id")
                    if doc_id and doc_id not in seen:
                        final_results.append(r)
                        seen.add(doc_id)
                return {"results": final_results, "aggregate_method": "intersection"}
            # TODO: Add more aggregation methods as needed
            return {"error": f"Unknown aggregate method: {method}"}
        elif step.type == "multi-hop":
            # Simulate multi-hop by calling /query/graph with increased depth
            input_step_id = step.parameters.get("input_step") or (step.dependencies[0] if step.dependencies else None)
            hops = step.parameters.get("hops", 2)
            if input_step_id is None or input_step_id not in step_results:
                return {"error": "No input step for multi-hop"}
            input_result = step_results[input_step_id]
            # Use the first doc_id from input_result as the query (simulate)
            doc_id = None
            if isinstance(input_result, dict):
                results = input_result.get("results")
                if results and isinstance(results, list) and results:
                    doc_id = results[0].get("doc_id")
            if not doc_id:
                return {"error": "No doc_id to multi-hop from"}
            payload = {"query": doc_id, "graph_expansion": {"depth": hops}}
            resp = requests.post(f"{self.base_url}/query/graph", json=payload)
            return {"results": resp.json().get("results", []), "multi_hop": hops}
        elif step.type == "llm_call":
            # Simulate LLM call for synthesis/summarization
            input_step_id = step.parameters.get("input_step") or (step.dependencies[0] if step.dependencies else None)
            prompt = step.parameters.get("prompt", "Summarize the results.")
            if input_step_id is None or input_step_id not in step_results:
                return {"error": "No input step for llm_call"}
            input_result = step_results[input_step_id]
            # For now, just return a synthesized string
            summary = f"[LLM SYNTHESIS] {prompt}: {str(input_result)[:100]}..."
            return {"llm_call": True, "synthesized": summary}
        # Add more step types as needed
        return {"result": f"Executed {step.type}"} 