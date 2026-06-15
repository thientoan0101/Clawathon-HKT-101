"""Load and query the function registry JSON for LLM routing and keyword detection."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).with_name("function_registry.json")


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    with _REGISTRY_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_functions() -> list[dict[str, Any]]:
    return load_registry()["functions"]


def get_actions() -> list[dict[str, Any]]:
    return load_registry()["actions"]


def get_function_by_id(function_id: str) -> dict[str, Any] | None:
    for fn in get_functions():
        if fn["id"] == function_id:
            return fn
    return None


def get_precomputed_ids() -> set[str]:
    return {fn["id"] for fn in get_functions() if fn["action"] == "precomputed"}


def get_tool_ids() -> set[str]:
    return {fn["id"] for fn in get_functions() if fn["action"] in {"tool", "sender_tool", "executive"}}


def _compact_function(fn: dict[str, Any]) -> dict[str, Any]:
    """Minimal function entry for LLM prompts."""
    entry: dict[str, Any] = {
        "id": fn["id"],
        "action": fn["action"],
        "desc": fn["description"][:140],
    }
    params = fn.get("parameters") or []
    if params:
        entry["params"] = [p["name"] for p in params]
    ex = fn.get("examples") or []
    if ex:
        entry["ex"] = ex[0][:80]
    return entry


def _normalize(text: str) -> str:
    return text.lower().strip()


_SCORE_STOPWORDS = frozenset(
    {"transfer", "transaction", "amount", "total", "number", "value", "user", "count", "many", "much"}
)


def _score_function(message: str, fn: dict[str, Any]) -> float:
    text = _normalize(message)
    score = 0.0
    example_hit = 0.0

    for example in fn.get("examples", []):
        ex = _normalize(example)
        if ex in text or text in ex:
            example_hit = max(example_hit, 3.0)
        else:
            tokens = [t for t in ex.split() if len(t) > 3 and t not in _SCORE_STOPWORDS]
            if tokens and all(t in text for t in tokens[:3]):
                example_hit = max(example_hit, 2.0)
            elif any(t in text for t in tokens):
                example_hit = max(example_hit, 1.0)
    score += example_hit

    for kw in fn.get("keywords_en", []) + fn.get("keywords_vi", []):
        if re.search(re.escape(_normalize(kw)), text):
            score += 1.5

    fn_id = fn["id"].replace(".", r"\.").replace("_", r"[_\s]?")
    if re.search(fn_id, text, re.I):
        score += 2.0

    return score


def build_catalog_json(*, compact: bool = False) -> str:
    """Serialize the full function catalog."""
    payload = {
        "actions": get_actions(),
        "functions": [_compact_function(fn) for fn in get_functions()],
    }
    if compact:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_full_router_catalog() -> str:
    """Full function_registry.json for LLM routing (all functions, compact)."""
    return build_catalog_json(compact=True)


def _rank_all_functions(message: str) -> list[dict[str, Any]]:
    """Score all registry functions (used only for LLM-failure fallback)."""
    scored = [(_score_function(message, fn), fn) for fn in get_functions()]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{**fn, "_score": score} for score, fn in scored if score > 0]


def detect_function_from_registry(message: str, min_score: float = 2.0) -> dict[str, Any] | None:
    """Best registry match — fallback when LLM routing fails."""
    ranked = _rank_all_functions(message)
    if not ranked:
        return None
    best = ranked[0]
    if best.get("_score", 0) >= min_score:
        return best
    return None


def registry_to_action_decision(match: dict[str, Any]) -> dict[str, Any]:
    """Convert a registry function match to ActionDecision fields."""
    action = match["action"]
    fn_id = match["id"]
    if action == "precomputed":
        return {"action": "precomputed", "stat_key": fn_id, "reasoning": f"registry: {fn_id}"}
    if action == "executive":
        return {
            "action": "executive",
            "tool_name": "get_executive_dashboard_summary",
            "reasoning": "registry: executive dashboard",
        }
    if action == "sender_tool":
        return {
            "action": "sender_tool",
            "tool_name": fn_id,
            "reasoning": f"registry: {fn_id}",
        }
    return {"action": "tool", "tool_name": fn_id, "reasoning": f"registry: {fn_id}"}
