"""Shared agent models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionDecision(BaseModel):
    """Structured output from the LLM router or pattern detectors."""

    action: Literal["precomputed", "sender_tool", "tool", "executive", "clarify", "agent"]
    stat_key: str | None = Field(default=None, description="Dotted key for precomputed action")
    tool_name: str | None = Field(default=None, description="Util function name")
    sender_id: str | None = Field(default=None, description="Sender ID for sender_tool")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    clarifying_question: str | None = Field(default=None, description="Question if action=clarify")
    reasoning: str = Field(default="", description="Brief reason for the choice")
