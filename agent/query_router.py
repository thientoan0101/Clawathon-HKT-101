"""Classify user questions and route to precomputed stats or sender util functions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from analytics import queries
from analytics.store import DataStore

from agent.models import ActionDecision

SENDER_ID_PATTERN = re.compile(r"\b(\d{10,20})\b")


class RouteKind(str, Enum):
    PRECOMPUTED = "precomputed"
    SENDER = "sender"
    EXECUTIVE = "executive"
    LLM = "llm"


@dataclass
class Route:
    kind: RouteKind
    confidence: float
    stat_key: str | None = None
    label: str | None = None
    sender_id: str | None = None
    tool_name: str | None = None


# Global / precomputed metrics — matched when NO sender_id in question
PRECOMPUTED_RULES: list[dict[str, Any]] = [
    {
        "patterns": [r"\btpv\b", r"total\s*payment\s*volume", r"total\s*volume", r"tổng\s*giá\s*trị", r"tong\s*so\s*tien"],
        "key": "transfer.tpv",
        "label": "Total Payment Volume (TPV)",
    },
    {
        "patterns": [r"unique\s*senders?", r"number\s*of\s*senders?", r"how\s*many\s*senders?", r"bao\s*nhiêu\s*người\s*gửi", r"distinct\s*senders?"],
        "key": "global.unique_senders",
        "label": "Unique senders",
    },
    {
        "patterns": [
            r"unique\s*peers?",
            r"unique\s*recipients?",
            r"unique\s*receivers?",
            r"number\s*of\s*recipients?",
            r"number\s*of\s*receivers?",
            r"how\s*many\s*receivers?",
            r"how\s*many\s*peers?",
            r"how\s*many\s*recipients?",
            r"distinct\s*peers?",
            r"distinct\s*receivers?",
            r"total\s*unique\s*receiver",
            r"count\s*of\s*receivers?",
            r"người\s*nhận",
            r"bao\s*nhiêu\s*người\s*nhận",
        ],
        "key": "global.unique_peers",
        "label": "Unique recipients (peers)",
    },
    {
        "patterns": [r"transfer\s*count", r"total\s*transaction", r"number\s*of\s*transaction", r"bao\s*nhiêu\s*giao\s*dịch", r"how\s*many\s*transfer"],
        "key": "transfer.transfer_count",
        "label": "Total transfer count",
    },
    {
        "patterns": [r"\bdau\b", r"daily\s*active"],
        "key": "activity.latest_dau",
        "label": "Daily Active Users (DAU)",
    },
    {
        "patterns": [r"\bwau\b", r"weekly\s*active"],
        "key": "activity.latest_wau",
        "label": "Weekly Active Users (WAU)",
    },
    {
        "patterns": [r"\bmau\b", r"monthly\s*active"],
        "key": "activity.latest_mau",
        "label": "Monthly Active Users (MAU)",
    },
    {
        "patterns": [r"dau\s*/\s*mau", r"dau/mau", r"stickiness"],
        "key": "activity.dau_mau_ratio_pct",
        "label": "DAU/MAU ratio (%)",
    },
    {
        "patterns": [r"\br7\b", r"r7\s*retention", r"retention\s*7"],
        "key": "retention.r7_transfer_retention_pct",
        "label": "R7 transfer retention (%)",
    },
    {
        "patterns": [r"\br30\b", r"r30\s*retention", r"retention\s*30"],
        "key": "retention.r30_transfer_retention_pct",
        "label": "R30 transfer retention (%)",
    },
    {
        "patterns": [r"\bd1\b", r"d1\s*retention"],
        "key": "retention.d1_retention_pct",
        "label": "D1 retention (%)",
    },
    {
        "patterns": [r"\bd7\b", r"d7\s*retention"],
        "key": "retention.d7_retention_pct",
        "label": "D7 retention (%)",
    },
    {
        "patterns": [r"\bd30\b", r"d30\s*retention"],
        "key": "retention.d30_retention_pct",
        "label": "D30 retention (%)",
    },
    {
        "patterns": [r"churn"],
        "key": "retention.churn_rate_pct",
        "label": "Churn rate (%)",
    },
    {
        "patterns": [r"repeat\s*sender", r"sender.*more\s*than\s*one"],
        "key": "retention.repeat_sender_rate_pct",
        "label": "Repeat sender rate (%)",
    },
    {
        "patterns": [r"average\s*transfer", r"avg\s*transfer", r"mean\s*amount", r"trung\s*bình"],
        "key": "transfer.average_transfer_value",
        "label": "Average transfer value",
    },
    {
        "patterns": [r"median\s*amount", r"median\s*transfer"],
        "key": "global.median_amount",
        "label": "Median transfer amount",
    },
    {
        "patterns": [r"sender.*more\s*than\s*3\s*peer", r"more\s*than\s*3\s*recipient", r"3\s*receivers?"],
        "key": "network.senders_with_more_than_3_peers",
        "label": "Senders with more than 3 peers",
    },
    {
        "patterns": [r"new\s*user"],
        "key": "growth.new_users_total",
        "label": "New users",
    },
    {
        "patterns": [r"active\s*sender"],
        "key": "activity.active_senders",
        "label": "Active senders",
    },
    {
        "patterns": [r"active\s*recipient"],
        "key": "activity.active_recipients",
        "label": "Active recipients",
    },
    {
        "patterns": [r"transaction\s*per\s*active", r"txn\s*per\s*user"],
        "key": "activity.transactions_per_active_user",
        "label": "Transactions per active user",
    },
    {
        "patterns": [r"success\s*rate"],
        "key": "transfer.success_rate_pct",
        "label": "Transfer success rate (%)",
    },
    {
        "patterns": [r"user\s*growth", r"growth\s*rate"],
        "key": "growth.user_growth_rate_pct",
        "label": "User growth rate (%)",
    },
]

# Sender-specific util mapping — only when sender_id detected
SENDER_RULES: list[dict[str, Any]] = [
    {
        "patterns": [r"transaction", r"txn", r"giao\s*dịch", r"how\s*many", r"count", r"number\s*of"],
        "tool": "get_user_txn_count",
        "label": "Transaction count",
    },
    {
        "patterns": [r"volume", r"amount", r"tổng\s*tiền", r"tong\s*tien", r"total\s*volume", r"tpv"],
        "tool": "get_user_total_volume",
        "label": "Total volume",
    },
    {
        "patterns": [r"peer", r"recipient", r"receiver", r"người\s*nhận", r"unique\s*peer"],
        "tool": "get_user_unique_peers",
        "label": "Unique peers",
    },
    {
        "patterns": [r"average", r"avg", r"mean", r"trung\s*bình"],
        "tool": "get_user_avg_amount",
        "label": "Average amount",
    },
]

EXECUTIVE_PATTERNS = [
    r"executive\s*dashboard",
    r"dashboard\s*summary",
    r"summary",
    r"tổng\s*quan",
    r"tong\s*quan",
    r"overview",
    r"key\s*metric",
]

SENDER_TOOL_FUNCS: dict[str, Callable[[DataStore, str], dict[str, Any]]] = {
    "get_user_txn_count": queries.user_txn_count,
    "get_user_total_volume": queries.user_total_volume,
    "get_user_unique_peers": queries.user_unique_peers,
    "get_user_avg_amount": queries.user_avg_amount,
}


def _normalize(text: str) -> str:
    return text.lower().strip()


def _score_patterns(text: str, patterns: list[str]) -> float:
    return sum(1.0 for p in patterns if re.search(p, text, re.IGNORECASE))


def extract_sender_id(text: str) -> str | None:
    match = SENDER_ID_PATTERN.search(text)
    return match.group(1) if match else None


def classify_query(message: str) -> Route:
    """Route a question to precomputed lookup, sender util, executive summary, or LLM."""
    text = _normalize(message)
    sender_id = extract_sender_id(message)

    # Executive dashboard
    if _score_patterns(text, EXECUTIVE_PATTERNS) >= 1 and not sender_id:
        return Route(kind=RouteKind.EXECUTIVE, confidence=0.95, label="Executive dashboard")

    # Sender-specific path
    if sender_id:
        best_tool = None
        best_score = 0.0
        best_label = None
        for rule in SENDER_RULES:
            score = _score_patterns(text, rule["patterns"])
            if score > best_score:
                best_score = score
                best_tool = rule["tool"]
                best_label = rule["label"]
        if best_tool and best_score >= 1:
            return Route(
                kind=RouteKind.SENDER,
                confidence=min(0.5 + best_score * 0.2, 0.99),
                sender_id=sender_id,
                tool_name=best_tool,
                label=best_label,
            )
        # Sender mentioned but vague — default to txn count
        return Route(
            kind=RouteKind.SENDER,
            confidence=0.7,
            sender_id=sender_id,
            tool_name="get_user_txn_count",
            label="Transaction count",
        )

    # Global precomputed path
    best_key = None
    best_label = None
    best_score = 0.0
    for rule in PRECOMPUTED_RULES:
        score = _score_patterns(text, rule["patterns"])
        if score > best_score:
            best_score = score
            best_key = rule["key"]
            best_label = rule["label"]
    if best_key and best_score >= 1:
        return Route(
            kind=RouteKind.PRECOMPUTED,
            confidence=min(0.5 + best_score * 0.2, 0.99),
            stat_key=best_key,
            label=best_label,
        )

    return Route(kind=RouteKind.LLM, confidence=0.0)


def detect_keyword_intent(message: str, min_confidence: float = 0.6) -> ActionDecision | None:
    """Map high-confidence keyword routes to ActionDecision (no LLM)."""
    route = classify_query(message)
    if route.confidence < min_confidence:
        return None
    if route.kind == RouteKind.PRECOMPUTED and route.stat_key:
        return ActionDecision(
            action="precomputed",
            stat_key=route.stat_key,
            reasoning=f"keyword: {route.label}",
        )
    if route.kind == RouteKind.EXECUTIVE:
        return ActionDecision(action="executive", reasoning="keyword: executive dashboard")
    if route.kind == RouteKind.SENDER and route.sender_id and route.tool_name:
        return ActionDecision(
            action="sender_tool",
            sender_id=route.sender_id,
            tool_name=route.tool_name,
            reasoning=f"keyword: {route.label}",
        )
    return None


def execute_route(store: DataStore, route: Route) -> dict[str, Any]:
    """Run the routed query and return structured data."""
    if route.kind == RouteKind.PRECOMPUTED and route.stat_key:
        return {
            "type": "precomputed",
            "key": route.stat_key,
            "label": route.label,
            "value": queries.lookup_stat(store, route.stat_key),
        }
    if route.kind == RouteKind.EXECUTIVE:
        return {"type": "executive", "value": queries.executive_summary(store)}
    if route.kind == RouteKind.SENDER and route.sender_id and route.tool_name:
        func = SENDER_TOOL_FUNCS[route.tool_name]
        return {
            "type": "sender",
            "tool": route.tool_name,
            "label": route.label,
            "value": func(store, route.sender_id),
        }
    raise ValueError("Cannot execute LLM route directly")
