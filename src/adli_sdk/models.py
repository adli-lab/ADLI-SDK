"""Pydantic models for ADLI API communication."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class InjectResult(BaseModel):
    """Response from POST /inject."""

    message: str
    adli_trace_id: str
    injected: bool


class Usage(BaseModel):
    model_config = {"extra": "allow"}

    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0


class MessagePart(BaseModel):
    model_config = {"extra": "allow"}

    part_kind: str
    content: str | None = None


class Message(BaseModel):
    model_config = {"extra": "allow"}

    kind: str
    parts: list[MessagePart] = []
    timestamp: str | None = None


class AgentTrace(BaseModel):
    """Assembled trace ready for /learn."""

    model_config = {"extra": "allow"}

    output_str: str = ""
    usage: Usage = Usage()
    messages: list[Message] = []


class LearnRequest(BaseModel):
    """Payload for POST /learn."""

    agent_name: str
    project_id: int
    framework: str = ""
    adli_trace_id: str
    user_message: str
    injected: bool
    outcome: Literal["success", "failure"] = "success"
    steps_count: int = 0
    cost_usd: float | None = None
    trace: AgentTrace


class InjectRequest(BaseModel):
    """Payload for POST /inject."""

    user_message: str
    agent_name: str
    project_id: int
