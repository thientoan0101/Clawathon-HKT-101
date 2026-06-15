"""Unified chat handler — LLM understands intent, then executes action."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from agent.action_executor import execute_decision
from agent.brain import create_analytics_agent
from agent.formatter import format_reply
from agent.llm_config import llm_config_status
from agent.llm_router import (
    classify_with_llm,
    compose_answer_with_llm,
    detect_list_senders_intent,
)
from agent.segment_intents import detect_segment_rule_intent
from agent.query_router import RouteKind, classify_query, detect_keyword_intent, execute_route
from analytics.store import get_store

logger = logging.getLogger(__name__)

_agent = None
KEYWORD_CONFIDENCE_THRESHOLD = 0.6


def get_agent():
    global _agent
    if _agent is None:
        _agent = create_analytics_agent(get_store())
    return _agent


def _llm_configured() -> bool:
    return all(llm_config_status().values())


def _try_keyword_route(message: str) -> dict[str, Any] | None:
    """Offline fallback when LLM is unavailable, or fast path for known metrics."""
    store = get_store()
    route = classify_query(message)
    if route.kind == RouteKind.LLM or route.confidence < KEYWORD_CONFIDENCE_THRESHOLD:
        return None
    data = execute_route(store, route)
    return {
        "status": "success",
        "reply": format_reply(data),
        "route": f"keyword_{route.kind.value}",
        "confidence": route.confidence,
        "source": data,
    }


def _try_keyword_decision_route(store, message: str, channel: str, session_id: str | None) -> dict[str, Any] | None:
    """Run keyword intent before LLM for common metric questions."""
    intent = detect_keyword_intent(message, KEYWORD_CONFIDENCE_THRESHOLD)
    if not intent:
        return None
    try:
        data = execute_decision(store, intent)
        if intent.action == "precomputed":
            route = classify_query(message)
            data["label"] = route.label
        return _wrap(
            {
                "status": "success",
                "reply": format_reply(data),
                "route": f"keyword_{intent.action}",
                "action": intent.action,
                "stat_key": intent.stat_key,
                "tool_name": intent.tool_name,
                "reasoning": intent.reasoning,
                "source": data,
            },
            channel,
            session_id,
        )
    except Exception as exc:
        logger.warning("Keyword decision route failed: %s", exc)
        return None


def _full_agent_answer(message: str, channel: str, user_id: str | None, session_id: str | None) -> dict[str, Any]:
    agent = get_agent()
    context = f"[channel={channel}"
    if user_id:
        context += f", user_id={user_id}"
    if session_id:
        context += f", session_id={session_id}"
    context += "]"

    result = agent.invoke({"messages": [{"role": "user", "content": f"{context}\n{message}"}]})
    ai_message = result["messages"][-1]
    return {"status": "success", "reply": ai_message.content, "route": "agent"}


def _try_pattern_route(store, message: str, channel: str, session_id: str | None) -> dict[str, Any] | None:
    """Fast path: segment rules and list-senders patterns (no LLM)."""
    for detector in (detect_segment_rule_intent, detect_list_senders_intent):
        intent = detector(message)
        if not intent:
            continue
        try:
            data = execute_decision(store, intent)
            route_name = (
                "pattern_segment_rule"
                if detector is detect_segment_rule_intent
                else "pattern_list_senders"
            )
            return _wrap(
                {
                    "status": "success",
                    "reply": format_reply(data),
                    "route": route_name,
                    "action": "tool",
                    "tool_name": intent.tool_name,
                    "parameters": intent.parameters,
                    "reasoning": intent.reasoning,
                    "source": data,
                },
                channel,
                session_id,
            )
        except Exception as exc:
            logger.warning("%s failed: %s", detector.__name__, exc)
    return None


def handle_message(
    message: str,
    channel: str = "web",
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """LLM classifies intent → execute action → LLM formats answer."""
    message = message.strip()
    if not message:
        return {"status": "error", "error": "message is required"}

    store = get_store()

    pattern_result = _try_pattern_route(store, message, channel, session_id)
    if pattern_result:
        return pattern_result

    keyword_result = _try_keyword_decision_route(store, message, channel, session_id)
    if keyword_result:
        return keyword_result

    # --- Path A: LLM router (primary) ---
    if _llm_configured():
        try:
            decision = classify_with_llm(message)

            if decision.action == "clarify":
                return _wrap(
                    {
                        "status": "success",
                        "reply": decision.clarifying_question or "Could you clarify your question?",
                        "route": "llm_clarify",
                        "reasoning": decision.reasoning,
                    },
                    channel,
                    session_id,
                )

            if decision.action == "agent":
                result = _full_agent_answer(message, channel, user_id, session_id)
                result["reasoning"] = decision.reasoning
                return _wrap(result, channel, session_id)

            # Execute chosen action deterministically (no invented numbers)
            data = execute_decision(store, decision)
            try:
                reply = compose_answer_with_llm(message, data)
            except Exception as exc:
                logger.warning("LLM formatting failed, using template: %s", exc)
                reply = format_reply(data)

            return _wrap(
                {
                    "status": "success",
                    "reply": reply,
                    "route": f"llm_{decision.action}",
                    "action": decision.action,
                    "tool_name": decision.tool_name,
                    "stat_key": decision.stat_key,
                    "sender_id": decision.sender_id,
                    "reasoning": decision.reasoning,
                    "source": data,
                },
                channel,
                session_id,
            )

        except ValueError as exc:
            if "Missing LLM configuration" in str(exc):
                pass  # fall through to keyword route
            else:
                logger.warning("LLM route execution failed: %s", exc)
        except Exception as exc:
            logger.warning("LLM router failed, trying fallbacks: %s", exc)

    # --- Path B: keyword router (no LLM) ---
    keyword = _try_keyword_route(message)
    if keyword:
        return _wrap(keyword, channel, session_id)

    # --- Path C: full agent (needs LLM) ---
    if not _llm_configured():
        return {
            "status": "error",
            "error": "LLM not configured. Set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL in .env.",
            "hint": "Dashboard metrics work at GET /. Chat requires LLM.",
        }

    try:
        return _wrap(_full_agent_answer(message, channel, user_id, session_id), channel, session_id)
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "hint": "Set LLM_API_KEY in .env and restart."}


def _wrap(result: dict[str, Any], channel: str, session_id: str | None) -> dict[str, Any]:
    result["channel"] = channel
    result["session_id"] = session_id
    result["timestamp"] = datetime.now().isoformat()
    return result
