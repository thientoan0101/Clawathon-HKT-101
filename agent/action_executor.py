"""Execute LLM routing decisions deterministically."""

from __future__ import annotations

import json
from typing import Any

from agent.action_catalog import VALID_PRECOMPUTED_KEYS
from agent.models import ActionDecision
from analytics import queries
from analytics.store import DataStore
from analytics.tools.registry import build_tools


def _tool_map(store: DataStore) -> dict[str, Any]:
    return {tool.name: tool for tool in build_tools(store)}


def execute_decision(store: DataStore, decision: ActionDecision) -> dict[str, Any]:
    """Run the action chosen by the LLM router and return structured data."""
    if decision.action == "precomputed":
        if not decision.stat_key or decision.stat_key not in VALID_PRECOMPUTED_KEYS:
            raise ValueError(f"Invalid precomputed key: {decision.stat_key}")
        return {
            "type": "precomputed",
            "key": decision.stat_key,
            "value": queries.lookup_stat(store, decision.stat_key),
        }

    if decision.action == "executive":
        return {"type": "executive", "value": queries.executive_summary(store)}

    if decision.action == "sender_tool":
        if not decision.sender_id or not decision.tool_name:
            raise ValueError("sender_tool requires sender_id and tool_name")
        tools = _tool_map(store)
        if decision.tool_name not in tools:
            raise ValueError(f"Unknown sender tool: {decision.tool_name}")
        raw = tools[decision.tool_name].invoke({"sender_id": decision.sender_id})
        return {
            "type": "sender",
            "tool": decision.tool_name,
            "value": json.loads(raw),
        }

    if decision.action == "tool":
        if not decision.tool_name:
            raise ValueError("tool action requires tool_name")
        tools = _tool_map(store)
        if decision.tool_name not in tools:
            raise ValueError(f"Unknown tool: {decision.tool_name}")
        params = decision.parameters or {}
        raw = tools[decision.tool_name].invoke(params)
        return {
            "type": "tool",
            "tool": decision.tool_name,
            "value": json.loads(raw) if isinstance(raw, str) else raw,
        }

    raise ValueError(f"Cannot execute action: {decision.action}")
