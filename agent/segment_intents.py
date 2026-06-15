"""Pattern detection for 10 segment rules (Vietnamese + English)."""

from __future__ import annotations

import re
from typing import Any

from agent.models import ActionDecision

# Vietnamese number words → VND
_VIET_NUM = {
    "nghìn": 1_000,
    "ngàn": 1_000,
    "triệu": 1_000_000,
    "ty": 1_000_000_000,
    "tỷ": 1_000_000_000,
}


def _parse_viet_amount(text: str) -> float | None:
    """Parse '1 triệu', '500 nghìn', '1000000'."""
    text = text.lower().replace(",", "").replace(".", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(nghìn|ngàn|triệu|ty|tỷ)", text)
    if m:
        return float(m.group(1)) * _VIET_NUM.get(m.group(2), 1)
    m = re.search(r"(\d{4,})", text)
    if m:
        return float(m.group(1))
    return None


def _extract_days(text: str) -> int | None:
    m = re.search(r"(?:trong\s+)?(\d+)\s*ngày", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:last|in)\s+(\d+)\s*days?", text, re.I)
    if m:
        return int(m.group(1))
    if re.search(r"\b7\s*ngày\b|\b7\s*days?\b|\br7\b", text, re.I):
        return 7
    if re.search(r"\b14\s*ngày\b|\b14\s*days?", text, re.I):
        return 14
    if re.search(r"\b30\s*ngày\b|\b30\s*days?\b|\br30\b", text, re.I):
        return 30
    return None


def _extract_int(text: str, patterns: list[str]) -> int | None:
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return int(m.group(1))
    return None


def detect_segment_rule_intent(message: str) -> ActionDecision | None:
    """Map natural language to one of 10 segment rule tools."""
    text = message.lower()
    days = _extract_days(message)

    # Rule 10 — bidirectional / hai chiều
    if re.search(r"hai\s*chiều|bidirectional|gửi.*nhận.*lại|vừa\s*gửi.*vừa\s*nhận|cặp.*giao\s*dịch", text):
        d = days or 30
        return ActionDecision(
            action="tool",
            tool_name="segment_rule10_bidirectional_pairs",
            parameters={"days": d},
            reasoning=f"rule 10 bidirectional pairs, {d} days",
        )

    # Rule 1 — more than n transactions
    if re.search(r"giao\s*d[iị]?ch|transaction|txn", text) and re.search(
        r"hơn|more\s+than|>\s*\d|nhiều\s+hơn", text
    ):
        if not re.search(r"đúng\s+\d|exactly", text):
            n = _extract_int(
                message,
                [r"hơn\s+(\d+)", r"more\s+than\s+(\d+)", r">\s*(\d+)"],
            )
            if n is not None and not re.search(r"peer|receiver|người\s+nhận|recipient", text):
                d = days or 7
                return ActionDecision(
                    action="tool",
                    tool_name="segment_rule1_senders_more_than_txns",
                    parameters={"days": d, "more_than": n},
                    reasoning=f"rule 1 >{n} txns in {d}d",
                )

    # Rule 2 — exactly n transactions
    n_exact = _extract_int(
        message,
        [
            r"đúng\s+(\d+)",
            r"exactly\s+(\d+)",
            r"(\d+)\s+giao\s*d[iị]?ch",
        ],
    )
    if n_exact is not None and re.search(r"đúng|exactly", text):
        d = days or 7
        return ActionDecision(
            action="tool",
            tool_name="segment_rule2_senders_exactly_txns",
            parameters={"days": d, "exactly": n_exact},
            reasoning=f"rule 2 exactly {n_exact} txns in {d}d",
        )

    # Rule 3 — active on at least d days
    min_days = _extract_int(
        message,
        [
            r"ít\s+nhất\s+(\d+)\s*ngày",
            r"at\s+least\s+(\d+)\s+(?:different\s+)?days",
            r"(\d+)\s+ngày\s+khác\s+nhau",
        ],
    )
    if min_days is not None and re.search(r"ngày\s+khác|active\s+days|ít\s+nhất.*ngày", text):
        d = days or 7
        return ActionDecision(
            action="tool",
            tool_name="segment_rule3_senders_min_active_days",
            parameters={"days": d, "min_days": min_days},
            reasoning=f"rule 3 >={min_days} active days in {d}d",
        )

    # Rule 4 — total amount > x
    if re.search(r"tổng\s*(?:giá\s*trị|amount)|total\s*amount", text) and re.search(
        r"lớn\s+hơn|more\s+than|>\s*", text
    ):
        amount = _parse_viet_amount(message)
        if amount is not None:
            d = days or 7
            return ActionDecision(
                action="tool",
                tool_name="segment_rule4_senders_total_amount_above",
                parameters={"days": d, "min_amount": amount},
                reasoning=f"rule 4 total amount >{amount} in {d}d",
            )

    # Rule 5 — average amount > x
    if re.search(r"trung\s*bình|average|avg", text) and re.search(r"lớn\s+hơn|more\s+than|>", text):
        amount = _parse_viet_amount(message)
        min_txns = _extract_int(message, [r"ít\s+nhất\s+(\d+)\s*giao", r"at\s+least\s+(\d+)\s*txn"]) or 1
        if amount is not None:
            d = days or 30
            return ActionDecision(
                action="tool",
                tool_name="segment_rule5_senders_avg_amount_above",
                parameters={"days": d, "min_avg": amount, "min_txns": min_txns},
                reasoning=f"rule 5 avg >{amount}, min_txns={min_txns}",
            )

    # Rule 6 — sender-peer pairs > n txns
    if re.search(r"cặp|pair", text) and re.search(r"sender.*peer|peer.*sender", text):
        n = _extract_int(message, [r"hơn\s+(\d+)", r"more\s+than\s+(\d+)"])
        if n is not None:
            d = days or 7
            return ActionDecision(
                action="tool",
                tool_name="segment_rule6_sender_peer_pairs_more_than_txns",
                parameters={"days": d, "more_than": n},
                reasoning=f"rule 6 pairs >{n} txns",
            )

    # Rule 7 — senders with > p peers (time window)
    if re.search(r"gửi.*(?:cho|tiền)|sent\s+to|người\s+nhận|peer", text) and re.search(
        r"hơn|more\s+than", text
    ):
        p = _extract_int(
            message,
            [
                r"hơn\s+(\d+)\s*(?:peer|người|receiver|người\s+nhận)",
                r"more\s+than\s+(\d+)",
            ],
        )
        if p is not None:
            d = days or 30
            return ActionDecision(
                action="tool",
                tool_name="segment_rule7_senders_more_than_peers",
                parameters={"days": d, "more_than": p},
                reasoning=f"rule 7 >{p} peers in {d}d",
            )

    # Rule 8 — peers with > p senders
    if re.search(r"nhận.*từ|received\s+from|peer.*sender", text) and re.search(
        r"hơn|more\s+than", text
    ):
        p = _extract_int(message, [r"hơn\s+(\d+)\s*sender", r"more\s+than\s+(\d+)\s*sender"])
        if p is not None:
            d = days or 30
            return ActionDecision(
                action="tool",
                tool_name="segment_rule8_peers_more_than_senders",
                parameters={"days": d, "more_than": p},
                reasoning=f"rule 8 peers >{p} senders",
            )

    # Rule 9 — more than k product codes
    if re.search(r"product", text) and re.search(r"hơn|more\s+than", text):
        k = _extract_int(message, [r"hơn\s+(\d+)", r"more\s+than\s+(\d+)"])
        if k is not None:
            d = days or 30
            return ActionDecision(
                action="tool",
                tool_name="segment_rule9_senders_more_than_products",
                parameters={"days": d, "more_than": k},
                reasoning=f"rule 9 >{k} products",
            )

    return None
