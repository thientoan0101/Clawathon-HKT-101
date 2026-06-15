"""Format structured query results into natural language replies."""

from __future__ import annotations

from typing import Any


def _fmt_number(value: float | int) -> str:
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return f"{value:,}"


def format_reply(data: dict[str, Any]) -> str:
    """Turn executor output into a user-facing answer."""
    dtype = data.get("type")

    if dtype == "precomputed":
        label = data.get("label", data.get("key", "Metric"))
        value = data["value"]
        if isinstance(value, dict):
            return f"**{label}**\n{value}"
        if isinstance(value, list):
            return _format_precomputed_list(label, value)
        if "pct" in data.get("key", "") or "ratio" in data.get("key", ""):
            return f"{label}: {value}%"
        if "volume" in data.get("key", "") or "tpv" in data.get("key", "") or "amount" in data.get("key", ""):
            return f"{label}: {_fmt_number(value)} VND"
        return f"{label}: {_fmt_number(value)}"

    if dtype == "executive":
        v = data["value"]
        lines = [
            "**Executive Dashboard**",
            f"- TPV: {_fmt_number(v['tpv'])} VND",
            f"- Transfer count: {_fmt_number(v['transfer_count'])}",
            f"- Unique senders: {_fmt_number(v['unique_senders'])}",
            f"- DAU: {_fmt_number(v['dau'])} | MAU: {_fmt_number(v['mau'])} | DAU/MAU: {v['dau_mau_ratio_pct']}%",
            f"- R7 retention: {v['r7_transfer_retention_pct']}% | R30 retention: {v['r30_transfer_retention_pct']}%",
            f"- D30 retention: {v['d30_retention_pct']}%",
            f"- Avg transfer value: {_fmt_number(v['average_transfer_value'])} VND",
            f"- Success rate: {v['success_rate_pct']}%",
        ]
        return "\n".join(lines)

    if dtype == "sender":
        v = data["value"]
        sender_id = v.get("sender_id", "")
        tool = data.get("tool", "")
        if tool == "get_user_txn_count":
            return f"Sender **{sender_id}** has **{_fmt_number(v['txn_count'])}** transactions."
        if tool == "get_user_total_volume":
            return f"Sender **{sender_id}** total volume: **{_fmt_number(v['total_volume'])}** VND."
        if tool == "get_user_unique_peers":
            return f"Sender **{sender_id}** transferred to **{_fmt_number(v['unique_peers'])}** unique peers."
        if tool == "get_user_avg_amount":
            return f"Sender **{sender_id}** average transfer amount: **{_fmt_number(v['avg_amount'])}** VND."
        return f"Sender **{sender_id}**: {v}"

    if dtype == "tool":
        v = data["value"]
        tool = data.get("tool", "")
        if tool == "list_senders_with_more_than_receivers":
            ids = v.get("sender_ids", [])
            n = v.get("sender_count", len(ids))
            more = v.get("more_than_receivers", "?")
            if not ids:
                return f"No senders found with more than {more} distinct receivers."
            id_lines = ", ".join(ids[:20])
            suffix = f" (showing 20 of {n})" if n > 20 else ""
            return (
                f"**{n}** sender(s) with more than **{more}** receivers{suffix}:\n"
                f"{id_lines}"
            )
        if tool.startswith("segment_rule"):
            return _format_segment_rule(v)
        if tool == "get_top_peers_by_count":
            items = v if isinstance(v, list) else []
            if not items:
                return "No recipient data found."
            top = items[0]
            peer_id = top.get("peer_id", "?")
            txn_count = top.get("txn_count", "?")
            if len(items) == 1:
                return f"The recipient with the most transfers is **{peer_id}** with **{txn_count}** transaction(s)."
            lines = [f"**Top {len(items)} recipients by transaction count:**"]
            for i, row in enumerate(items, 1):
                lines.append(f"{i}. **{row.get('peer_id')}** — {row.get('txn_count')} txns")
            return "\n".join(lines)
        if tool == "get_top_senders_by_count":
            items = v if isinstance(v, list) else []
            if not items:
                return "No sender data found."
            top = items[0]
            if len(items) == 1:
                return (
                    f"The sender with the most transfers is **{top.get('sender_id')}** "
                    f"with **{top.get('txn_count')}** transaction(s)."
                )
            lines = [f"**Top {len(items)} senders by transaction count:**"]
            for i, row in enumerate(items, 1):
                lines.append(f"{i}. **{row.get('sender_id')}** — {row.get('txn_count')} txns")
            return "\n".join(lines)
        if tool == "list_transactions_in_last_days":
            return _format_transaction_list(v)
        if tool in {
            "cohort.retention_matrix",
            "marketing.cac_and_ltv",
            "segmentation.rfm_scores",
            "risk.velocity_alerts",
            "activity.hourly_peak_load",
            "transfer.error_code_distribution",
            "activity.custom_hourly_buckets",
            "activity.peak_minute_velocity",
        }:
            return _format_advanced_tool(tool, v)
        return f"{tool}: {v}"

    return str(data)


def _format_segment_rule(v: dict[str, Any]) -> str:
    desc = v.get("description", "Segment rule result")
    window = v.get("window", {})
    win_txt = ""
    if window.get("start"):
        win_txt = f" (window: {window['start']} → {window['end']}, {window.get('days')} days)"

    if "sender_count" in v:
        count = v["sender_count"]
        ids = v.get("sender_ids", [])
        lines = [f"**{desc}**{win_txt}", f"Count: **{count:,}** sender(s)"]
        if ids:
            preview = ", ".join(ids[:15])
            if len(ids) > 15:
                preview += f" … (+{len(ids) - 15} more)"
            lines.append(f"Sender IDs: {preview}")
        return "\n".join(lines)

    if "peer_count" in v:
        return f"**{desc}**{win_txt}\nCount: **{v['peer_count']:,}** peer(s)"

    if "pair_count" in v:
        lines = [f"**{desc}**{win_txt}", f"Count: **{v['pair_count']:,}** pair(s)"]
        pairs = v.get("pairs", [])
        if pairs and v.get("pairs_truncated"):
            lines.append(f"(showing first {len(pairs)} pairs)")
        return "\n".join(lines)

    return f"**{desc}**{win_txt}\n{v}"


def _format_transaction_list(v: dict[str, Any]) -> str:
    window = v.get("window", {})
    start = window.get("start") or "?"
    end = window.get("end") or "?"
    total = v.get("total_count", 0)
    if total == 0:
        return f"No transactions found in the requested window ({start} → {end})."

    if v.get("single_day"):
        title = f"**Transactions on {start}** ({v.get('days')} days before latest)"
    else:
        title = f"**Transactions in last {v.get('days')} days** ({start} → {end})"

    lines = [title, f"Total: **{total:,}** transaction(s)"]
    if v.get("truncated"):
        lines.append(f"Showing first **{v.get('returned_count', 0)}** rows:")

    for i, txn in enumerate(v.get("transactions", []), 1):
        lines.append(
            f"{i}. `{txn.get('order_no')}` | {txn.get('sender_id')} → {txn.get('peer_id')} | "
            f"{_fmt_number(txn.get('amount', 0))} VND | {txn.get('product_code')} | {txn.get('order_date')}"
        )
    return "\n".join(lines)


def _format_advanced_tool(tool: str, v: dict[str, Any]) -> str:
    if tool == "segmentation.rfm_scores":
        lines = ["**RFM segmentation**", f"Total users: **{v.get('total_users', 0):,}**"]
        for seg in v.get("segments", [])[:8]:
            lines.append(
                f"- **{seg.get('segment_name')}**: {seg.get('user_count'):,} users "
                f"({seg.get('percentage_of_total')}%), avg monetary {seg.get('avg_monetary_value'):,} VND"
            )
        return "\n".join(lines)
    if tool == "marketing.cac_and_ltv":
        return (
            f"**LTV vs CAC** (CAC={_fmt_number(v.get('blended_cac', 0))} VND)\n"
            f"- Estimated LTV: **{_fmt_number(v.get('estimated_ltv', 0))}** VND\n"
            f"- LTV/CAC ratio: **{v.get('ltv_cac_ratio')}**\n"
            f"- Payback months: **{v.get('payback_months')}**"
        )
    if tool == "cohort.retention_matrix":
        lines = [f"**Cohort retention matrix** ({v.get('cohort_period')})"]
        for cohort in v.get("cohorts", [])[:5]:
            periods = cohort.get("active_in_period_n", [])
            latest = periods[-1] if periods else {}
            lines.append(
                f"- Cohort **{cohort.get('cohort_date')}**: {cohort.get('total_users')} users, "
                f"latest retention **{latest.get('retention_pct', 0)}%**"
            )
        return "\n".join(lines)
    if tool == "risk.velocity_alerts":
        alerts = v.get("alerts", [])
        if not alerts:
            return "No velocity alerts found for the given thresholds."
        lines = [f"**Velocity alerts**: {v.get('alert_count', len(alerts))} sender(s)"]
        for a in alerts[:10]:
            lines.append(
                f"- **{a.get('sender_id')}**: {a.get('txn_count')} txns, "
                f"{_fmt_number(a.get('total_amount', 0))} VND @ {a.get('window_start_time')}"
            )
        return "\n".join(lines)
    if tool == "activity.hourly_peak_load":
        top = (v.get("hourly_load") or [])[:5]
        lines = [f"**Peak load (last {v.get('last_days')} days)** — top slots:"]
        for row in top:
            lines.append(
                f"- {row.get('day_of_week')} {row.get('hour_of_day')}:00 — "
                f"{row.get('avg_txn_count')} txns, {_fmt_number(row.get('avg_tpv', 0))} VND"
            )
        return "\n".join(lines)
    if tool == "transfer.error_code_distribution":
        return (
            f"**Error distribution** (last {v.get('days')} days)\n"
            f"{v.get('note', '')}\n"
            f"Successful transactions: **{v.get('success_count', 0):,}**"
        )
    if tool == "activity.custom_hourly_buckets":
        lines = [
            f"**Hourly buckets** ({v.get('bucket_size_hours')}h slots, last {v.get('days')} days)"
        ]
        for row in v.get("buckets", [])[:12]:
            lines.append(
                f"- **{row.get('bucket_label')}**: {row.get('txn_count')} txns, "
                f"{_fmt_number(row.get('total_amount', 0))} VND"
            )
        return "\n".join(lines)
    if tool == "activity.peak_minute_velocity":
        return (
            f"**Peak minute** (last {v.get('days')} days)\n"
            f"- Timestamp: **{v.get('peak_timestamp')}**\n"
            f"- Max txns/minute: **{v.get('max_txns_per_minute')}**\n"
            f"- Baseline avg/minute: **{v.get('baseline_avg_minute')}**"
        )
    return f"{tool}: {v}"


def _format_precomputed_list(label: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"**{label}**: no data"
    lines = [f"**{label}**"]
    for row in rows:
        if "time_bucket_name" in row:
            lines.append(
                f"- **{row.get('time_bucket_name')}** ({row.get('hour_range')}): "
                f"{row.get('txn_count')} txns ({row.get('percentage_of_count')}%), "
                f"{_fmt_number(row.get('total_tpv', 0))} VND"
            )
        else:
            lines.append(str(row))
    return "\n".join(lines)
