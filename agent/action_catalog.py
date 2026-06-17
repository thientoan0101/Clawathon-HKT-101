"""Catalog of available actions for the LLM router prompt."""

from __future__ import annotations

from agent.function_registry import (
    build_full_router_catalog,
    get_functions,
    get_precomputed_ids,
    get_tool_ids,
)

# Backward-compatible exports derived from function_registry.json
PRECOMPUTED_KEYS: list[dict[str, str]] = [
    {"key": fn["id"], "desc": fn["description"]}
    for fn in get_functions()
    if fn["action"] == "precomputed"
]

SENDER_TOOLS: list[dict[str, str]] = [
    {"name": fn["id"], "desc": fn["description"]}
    for fn in get_functions()
    if fn["action"] == "sender_tool"
]

GENERAL_TOOLS: list[dict[str, str]] = [
    {"name": fn["id"], "desc": fn["description"]}
    for fn in get_functions()
    if fn["action"] == "tool" and not fn["id"].startswith("segment_rule")
]

SEGMENT_RULE_TOOLS: list[dict[str, str]] = [
    {"name": fn["id"], "desc": fn["description"]}
    for fn in get_functions()
    if fn["id"].startswith("segment_rule")
]

VALID_PRECOMPUTED_KEYS = get_precomputed_ids()
VALID_TOOL_NAMES = get_tool_ids()

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a transfer analytics assistant.

Pick ONE function from the catalog and return ActionDecision JSON:
{{"action":"precomputed|sender_tool|tool|executive|clarify|agent","stat_key":null,"tool_name":null,"sender_id":null,"parameters":{{}},"clarifying_question":null,"reasoning":"..."}}

Rules:
- Choose the best-matching function by id from the full catalog below.
- Global metric, no user id → precomputed + stat_key=id.
- Specific sender_id (10-20 digits) → sender_tool + tool_name + sender_id.
- List sender ids with >N receivers → list_senders_with_more_than_receivers, parameters={{"more_than":N}}.
- Top/most sender or receiver → precomputed stat_key: insights.* for single most (e.g. insights.top_sender_by_count, insights.top_receiver_by_count); ranking.* for top-N lists (e.g. ranking.top_senders_by_count).
- List/show transactions in last N days or N days ago → list_transactions_in_last_days, parameters={{"days":N,"single_day":false}} for rolling window; single_day=true for one calendar day N days before latest txn.
- Morning vs night / time of day → activity.time_of_day_breakdown (precomputed).
- Hourly bucket blocks → activity.custom_hourly_buckets with bucket_size_hours (1/2/3/4/6/12).
- Peak traffic minute → activity.peak_minute_velocity.
- Full overview → executive.
- Vague → clarify. Multi-step → agent.
- NEVER invent numbers.
- Return raw JSON only (no markdown fences).

Function catalog (function_registry.json):
{catalog_json}
"""


def build_router_prompt() -> str:
    """Build LLM prompt with the complete function_registry.json catalog."""
    catalog_json = build_full_router_catalog()
    return ROUTER_SYSTEM_PROMPT.format(catalog_json=catalog_json)
