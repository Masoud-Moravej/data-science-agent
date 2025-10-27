from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.agent_service import agent_chat_service


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    name: str
    response: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Stable identifier for the chat session.")
    message: str = Field(..., description="Latest user message to send to the agent.", min_length=1)


class Artifact(BaseModel):
    name: str
    mime_type: str
    data: str
    display_name: str | None = None


class ChatResponse(BaseModel):
    text: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_responses: List[ToolResponse] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest) -> ChatResponse:
    """Proxy the chat message to the ADK agent runner."""
    try:
        result = await agent_chat_service.send_message(request.user_id, request.message)
    except Exception as exc:  # pragma: no cover - passthrough for FastAPI
        raise HTTPException(status_code=500, detail="Failed to retrieve agent response.") from exc

    return ChatResponse(
        text=result.get("text", ""),
        tool_calls=[ToolCall(**item) for item in result.get("tool_calls", [])],
        tool_responses=[ToolResponse(**item) for item in result.get("tool_responses", [])],
        artifacts=[Artifact(**item) for item in result.get("artifacts", [])],
    )
