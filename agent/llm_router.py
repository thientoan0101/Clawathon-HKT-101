"""LLM-based intent classification → structured action decision."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.action_catalog import VALID_PRECOMPUTED_KEYS, VALID_TOOL_NAMES, build_router_prompt
from agent.function_registry import detect_function_from_registry, registry_to_action_decision
from agent.llm_config import get_llm_settings
from agent.models import ActionDecision
from agent.query_router import SENDER_ID_PATTERN, detect_keyword_intent
from agent.segment_intents import detect_segment_rule_intent

logger = logging.getLogger(__name__)

_LIST_SENDERS_PATTERNS = [
    re.compile(
        r"sender\s*ids?\s+(?:of\s+)?(?:users?|senders?).*(?:more\s+than|over|>\s*)\s*(\d+)\s+"
        r"(?:receiver|recipient|peer)s?",
        re.I,
    ),
    re.compile(
        r"(?:list|find|show|get|which)\s+sender\s*ids?.*(?:more\s+than|over|>\s*)\s*(\d+)\s+"
        r"(?:receiver|recipient|peer)s?",
        re.I,
    ),
    re.compile(
        r"(?:users?|senders?).*(?:more\s+than|over|>\s*)\s*(\d+)\s+(?:receiver|recipient|peer)s?.*sender",
        re.I,
    ),
    re.compile(
        r"(?:more\s+than|over|>\s*)\s*(\d+)\s+(?:receiver|recipient|peer)s?.*(?:sender\s*id|users?|senders?)",
        re.I,
    ),
]

_llm_router = None


def _get_router_llm() -> ChatOpenAI:
    global _llm_router
    if _llm_router is None:
        model, base_url, api_key = get_llm_settings()
        _llm_router = ChatOpenAI(
            model=model, base_url=base_url, api_key=api_key, temperature=0, max_tokens=512
        )
    return _llm_router


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """Parse JSON from plain text or ```json fenced blocks."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _parse_decision_fallback(raw: str) -> ActionDecision:
    """Parse LLM text when structured output is unavailable."""
    text = (raw or "").strip()
    try:
        return ActionDecision.model_validate(_extract_json_from_text(text))
    except (json.JSONDecodeError, ValueError):
        pass
    if text.lower().startswith("clarify"):
        question = text.split("\n", 1)[1].strip() if "\n" in text else "Could you clarify your question?"
        return ActionDecision(
            action="clarify",
            clarifying_question=question,
            reasoning="llm plain-text clarify",
        )
    raise ValueError(f"Cannot parse LLM routing response: {text[:200]}")


def detect_list_senders_intent(message: str) -> ActionDecision | None:
    """Match questions asking for sender IDs with more than N receivers."""
    for pattern in _LIST_SENDERS_PATTERNS:
        match = pattern.search(message)
        if match:
            more_than = int(match.group(1))
            return ActionDecision(
                action="tool",
                tool_name="list_senders_with_more_than_receivers",
                parameters={"more_than": more_than},
                reasoning=f"list sender_ids with more than {more_than} distinct receivers",
            )
    return None


def detect_registry_intent(message: str) -> ActionDecision | None:
    """Match user question to function_registry.json by keywords/examples."""
    match = detect_function_from_registry(message)
    if not match:
        return None
    fields = registry_to_action_decision(match)
    return ActionDecision(**fields)


def _registry_decision(message: str) -> ActionDecision | None:
    match = detect_function_from_registry(message)
    if not match:
        return None
    return ActionDecision(**registry_to_action_decision(match))


def classify_with_llm(message: str) -> ActionDecision:
    """Use LLM to understand user chat and pick the suitable action."""
    segment = detect_segment_rule_intent(message)
    if segment:
        return segment

    list_senders = detect_list_senders_intent(message)
    if list_senders:
        return list_senders

    keyword = detect_keyword_intent(message)
    if keyword:
        return keyword

    llm = _get_router_llm()
    system = build_router_prompt()
    catalog_hint = (
        "Read the full function catalog in the system message. "
        "Pick the single best function id and return ActionDecision JSON."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"{catalog_hint}\n\nUser question: {message}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        decision = _parse_decision_fallback(raw)
    except Exception as exc:
        logger.warning("LLM routing failed, trying structured output: %s", exc)
        try:
            structured = llm.with_structured_output(ActionDecision)
            decision = structured.invoke(messages)
            if not decision.action:
                raise ValueError("empty action from structured output")
        except Exception as inner_exc:
            logger.warning("Structured routing failed, using registry fallback: %s", inner_exc)
            fallback = _registry_decision(message)
            if fallback:
                return fallback
            return ActionDecision(action="agent", reasoning="llm failed, no registry match")

    return _normalize_decision(decision, message)


def _normalize_decision(decision: ActionDecision, message: str) -> ActionDecision:
    segment = detect_segment_rule_intent(message)
    if segment:
        return segment

    special = detect_list_senders_intent(message)
    if special:
        return special

    keyword = detect_keyword_intent(message)
    if keyword:
        return keyword

    sender_in_msg = SENDER_ID_PATTERN.search(message)
    sender_id = decision.sender_id or (sender_in_msg.group(1) if sender_in_msg else None)

    if decision.action == "precomputed" and decision.stat_key not in VALID_PRECOMPUTED_KEYS:
        logger.warning("Invalid stat_key %s, trying registry match", decision.stat_key)
        registry = detect_registry_intent(message)
        if registry and registry.action == "precomputed":
            return registry
        return ActionDecision(action="agent", reasoning="invalid stat_key")

    if decision.action == "tool" and decision.tool_name and decision.tool_name not in VALID_TOOL_NAMES:
        logger.warning("Invalid tool_name %s, trying registry match", decision.tool_name)
        registry = detect_registry_intent(message)
        if registry:
            return registry
        return ActionDecision(action="agent", reasoning="invalid tool_name")

    if decision.action == "sender_tool":
        if not sender_id:
            return ActionDecision(
                action="clarify",
                clarifying_question="Which sender_id should I look up? Please provide the numeric sender ID.",
                reasoning="sender_tool without sender_id",
            )
        decision.sender_id = sender_id
        if not decision.tool_name:
            decision.tool_name = "get_user_txn_count"

    if decision.action == "sender_tool" and re.search(
        r"sender\s*ids?\s+(?:of\s+)?(?:users?|who)", message, re.I
    ):
        list_intent = detect_list_senders_intent(message)
        if list_intent:
            return list_intent

    if decision.action == "executive":
        decision.tool_name = "get_executive_dashboard_summary"

    return decision


def compose_answer_with_llm(message: str, data: dict[str, Any]) -> str:
    llm = _get_router_llm()
    prompt = f"""User asked: {message}

Analytics result (use these exact numbers, do not change or invent values):
{json.dumps(data, default=str, ensure_ascii=False)}

Write a concise, friendly answer in the same language as the user. Include the key numbers."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)
