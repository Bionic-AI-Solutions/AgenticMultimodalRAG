from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional
from app.agentic.query_decomposer import QueryDecomposer
from app.agentic.models import DecompositionPlan
from app.agentic.agent_executor import AgentExecutor
from app.agentic.response_synthesizer import ResponseSynthesizer, ResponseSynthesisRequest, ResponseSynthesisResult
import json
import os
from fastapi import status

router = APIRouter()

decomposer = QueryDecomposer()
executor = AgentExecutor()
synthesizer = ResponseSynthesizer()

FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "feedback.jsonl")


class DecomposeRequest(BaseModel):
    query: str
    app_id: str
    user_id: str
    modality: str
    context: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    app_id: str
    user_id: str
    plan: Any
    execution_trace: Any
    answer: str
    explanation: str
    rating: int  # 1-5
    comments: Optional[str] = None
    explanation_style: Optional[str] = None
    prompt_version: Optional[str] = None


@router.post(
    "/agent/query/decompose",
    response_model=DecompositionPlan,
    tags=["Agentic"],
    summary="Decompose a user query into an agentic plan",
    operation_id="decompose_query",
)
def decompose_query(request: DecomposeRequest = Body(...)):
    """
    Decompose a complex user query into a structured agentic plan for downstream execution.
    """
    return decomposer.decompose(
        query=request.query,
        app_id=request.app_id,
        user_id=request.user_id,
        modality=request.modality,
        context=request.context or {},
    )


# --- New: Agentic Plan Execution Endpoint ---
from fastapi import Request
from fastapi.responses import JSONResponse


@router.post("/agent/execute", tags=["Agentic"], summary="Execute an agentic plan (multi-step, multimodal, tool use)", operation_id="execute_agentic_plan")
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


@router.post(
    "/agent/answer",
    response_model=ResponseSynthesisResult,
    tags=["Agentic"],
    summary="Synthesize a final answer and explanation from agentic execution trace",
    operation_id="synthesize_agentic_answer",
)
async def synthesize_agentic_answer(request: Request):
    """
    Synthesize a final answer and step-by-step explanation from the results and trace of an agentic plan execution.
    Accepts: plan, execution_trace, app_id, user_id, context (optional), explanation_style (optional), prompt_version (optional)
    - explanation_style: e.g., 'step-by-step', 'short', 'detailed', 'for a 5th grader'
    - prompt_version: e.g., 'default', 'v2', etc.
    Returns: answer, explanation, supporting evidence, trace
    """
    body = await request.json()
    req = ResponseSynthesisRequest(**body)
    result = synthesizer.synthesize_answer(req)
    return result


@router.post(
    "/agent/feedback",
    status_code=status.HTTP_201_CREATED,
    tags=["Agentic"],
    summary="Submit user feedback on agentic answer/explanation",
    operation_id="submit_agentic_feedback",
)
async def submit_agentic_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback on an agentic answer and explanation.
    Accepts: app_id, user_id, plan, execution_trace, answer, explanation, rating (1-5), comments (optional), explanation_style, prompt_version
    Stores feedback in a local file (feedback.jsonl) for now. Returns success message.
    TODO: Integrate with DB and feedback analytics.
    """
    record = feedback.dict()
    record["timestamp"] = __import__("datetime").datetime.utcnow().isoformat()
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "success", "message": "Feedback recorded."}


@router.get("/agent/tools/list", tags=["Agentic"], summary="List available MCP tools from FastAPI MCP server", operation_id="list_mcp_tools")
async def list_mcp_tools():
    """
    List all available MCP tools exposed by the FastAPI MCP server.
    These tools correspond to the FastAPI endpoints and can be used in agentic plans.
    Returns a list of tool definitions with names, descriptions, and parameters.
    """
    tools = executor.list_mcp_tools()
    return {"tools": tools, "count": len(tools)}
