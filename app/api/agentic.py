from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional
from app.agentic.query_decomposer import QueryDecomposer
from app.agentic.models import DecompositionPlan
from app.agentic.agent_executor import AgentExecutor
from app.agentic.response_synthesizer import ResponseSynthesizer, ResponseSynthesisRequest, ResponseSynthesisResult

router = APIRouter()

decomposer = QueryDecomposer()
executor = AgentExecutor()
synthesizer = ResponseSynthesizer()

class DecomposeRequest(BaseModel):
    query: str
    app_id: str
    user_id: str
    modality: str
    context: Optional[Dict[str, Any]] = None

@router.post("/agent/query/decompose", response_model=DecompositionPlan, tags=["Agentic"], summary="Decompose a user query into an agentic plan")
def decompose_query(request: DecomposeRequest = Body(...)):
    """
    Decompose a complex user query into a structured agentic plan for downstream execution.
    """
    return decomposer.decompose(
        query=request.query,
        app_id=request.app_id,
        user_id=request.user_id,
        modality=request.modality,
        context=request.context or {}
    )

# --- New: Agentic Plan Execution Endpoint ---
from fastapi import Request
from fastapi.responses import JSONResponse

@router.post("/agent/execute", tags=["Agentic"], summary="Execute an agentic plan (multi-step, multimodal, tool use)")
async def execute_agentic_plan(request: Request):
    """
    Execute a DecompositionPlan using the AgentExecutor. Returns the final result and execution trace.
    """
    body = await request.json()
    # Accept either a full plan or a user query (scaffold for future)
    if "plan" in body:
        plan = DecompositionPlan(**body)
        app_id = body.get("app_id", "app1")
        user_id = body.get("user_id", "user1")
        context = body.get("context", {})
        result = executor.execute_plan(plan, app_id=app_id, user_id=user_id, context=context)
        return JSONResponse(content=result)
    # Future: accept raw query, decompose, then execute
    return JSONResponse(status_code=422, content={"error": "Request must include a valid DecompositionPlan."})

@router.post("/agent/answer", response_model=ResponseSynthesisResult, tags=["Agentic"], summary="Synthesize a final answer and explanation from agentic execution trace")
async def synthesize_agentic_answer(request: Request):
    """
    Synthesize a final answer and step-by-step explanation from the results and trace of an agentic plan execution.
    Accepts: plan, execution_trace, app_id, user_id, context (optional)
    Returns: answer, explanation, supporting evidence, trace
    """
    body = await request.json()
    req = ResponseSynthesisRequest(**body)
    result = synthesizer.synthesize_answer(req)
    return result 