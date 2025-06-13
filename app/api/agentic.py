from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional
from app.agentic.query_decomposer import QueryDecomposer
from app.agentic.models import DecompositionPlan

router = APIRouter()

decomposer = QueryDecomposer()

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