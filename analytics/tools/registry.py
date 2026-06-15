"""Build 50 LangChain analytics tools bound to the in-memory DataStore."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from analytics import advanced_metrics as adv
from analytics import queries
from analytics import segment_rules as seg
from analytics.store import DataStore


def _json(data: Any) -> str:
    return queries.dumps(data)


def _lookup(store: DataStore, dotted_key: str) -> Any:
    return queries.lookup_stat(store, dotted_key)


def _filter_sender(df: pd.DataFrame, sender_id: str) -> pd.DataFrame:
    return queries.filter_sender(df, sender_id)


def build_tools(store: DataStore) -> list:
    """Create all 50 analytics tools with access to the DataStore."""

    @tool
    def lookup_precomputed_stat(stat_key: str) -> str:
        """Look up a pre-computed dashboard statistic by dotted key.
        Examples: global.tpv, retention.r7_transfer_retention_pct, activity.latest_mau,
        network.senders_with_more_than_3_peers, transfer.average_transfer_value."""
        try:
            return _json({"key": stat_key, "value": _lookup(store, stat_key)})
        except KeyError as exc:
            return _json({"error": str(exc), "hint": "Use list_precomputed_keys() first."})

    @tool
    def list_precomputed_keys() -> str:
        """List all available pre-computed statistic keys (dotted paths)."""

        def walk(prefix: str, node: Any, keys: list[str]) -> None:
            if isinstance(node, dict) and node and not isinstance(next(iter(node.values())), (dict, list)):
                for k in node:
                    keys.append(f"{prefix}.{k}" if prefix else k)
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(f"{prefix}.{k}" if prefix else k, v, keys)
            else:
                if prefix:
                    keys.append(prefix)

        keys: list[str] = []
        for top_key, value in store.precomputed.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (dict, list)):
                        walk(f"{top_key}.{sub_key}", sub_val, keys)
                    else:
                        keys.append(f"{top_key}.{sub_key}")
            else:
                keys.append(top_key)
        return _json(sorted(set(keys)))

    @tool
    def get_user_txn_count(sender_id: str) -> str:
        """Total number of transfer transactions for a sender_id. Example: 160516000000516."""
        return _json(queries.user_txn_count(store, sender_id))

    @tool
    def get_user_total_volume(sender_id: str) -> str:
        """Total transfer volume (TPV) for a specific sender_id."""
        return _json(queries.user_total_volume(store, sender_id))

    @tool
    def get_user_unique_peers(sender_id: str) -> str:
        """Count distinct peer_id (recipients) for a sender."""
        return _json(queries.user_unique_peers(store, sender_id))

    @tool
    def get_user_avg_amount(sender_id: str) -> str:
        """Average transfer amount for a sender."""
        return _json(queries.user_avg_amount(store, sender_id))

    @tool
    def get_user_first_transfer_date(sender_id: str) -> str:
        """First transfer date for a sender."""
        subset = _filter_sender(store.df, sender_id)
        if subset.empty:
            return _json({"sender_id": sender_id, "error": "sender not found"})
        return _json({"sender_id": sender_id, "first_transfer_date": str(subset["order_date"].min())})

    @tool
    def get_user_last_transfer_date(sender_id: str) -> str:
        """Most recent transfer date for a sender."""
        subset = _filter_sender(store.df, sender_id)
        if subset.empty:
            return _json({"sender_id": sender_id, "error": "sender not found"})
        return _json({"sender_id": sender_id, "last_transfer_date": str(subset["order_date"].max())})

    @tool
    def get_user_transfers_by_product(sender_id: str) -> str:
        """Breakdown of transfer count and volume by product_code for a sender."""
        subset = _filter_sender(store.df, sender_id)
        grouped = subset.groupby("product_code").agg(count=("order_no", "count"), volume=("amount", "sum"))
        return _json({"sender_id": sender_id, "by_product": grouped.reset_index().to_dict(orient="records")})

    @tool
    def get_user_r7_retention_status(sender_id: str) -> str:
        """Whether sender made a second transfer within 7 days of their first transfer."""
        subset = _filter_sender(store.df, sender_id).sort_values("order_dt")
        if subset.empty:
            return _json({"sender_id": sender_id, "error": "sender not found"})
        if len(subset) < 2:
            return _json({"sender_id": sender_id, "r7_retained": False, "reason": "only one transfer"})
        first_dt = subset.iloc[0]["order_dt"]
        second_dt = subset.iloc[1]["order_dt"]
        gap_days = (second_dt - first_dt).days
        return _json({"sender_id": sender_id, "r7_retained": gap_days <= 7, "gap_days": gap_days})

    @tool
    def get_user_transfers_in_date_range(sender_id: str, start_date: str, end_date: str) -> str:
        """Count and volume for a sender between start_date and end_date (YYYY-MM-DD)."""
        subset = _filter_sender(store.df, sender_id)
        mask = (subset["order_date"] >= date.fromisoformat(start_date)) & (
            subset["order_date"] <= date.fromisoformat(end_date)
        )
        filtered = subset[mask]
        return _json(
            {
                "sender_id": sender_id,
                "start_date": start_date,
                "end_date": end_date,
                "txn_count": len(filtered),
                "volume": round(float(filtered["amount"].sum()), 2),
            }
        )

    @tool
    def get_peer_txn_count(peer_id: str) -> str:
        """Total transactions received by a peer_id."""
        subset = store.df[store.df["peer_id"] == str(peer_id)]
        return _json({"peer_id": peer_id, "txn_count": len(subset)})

    @tool
    def get_peer_total_volume(peer_id: str) -> str:
        """Total volume received by a peer_id."""
        subset = store.df[store.df["peer_id"] == str(peer_id)]
        return _json({"peer_id": peer_id, "total_volume": round(float(subset["amount"].sum()), 2)})

    @tool
    def get_peer_unique_senders(peer_id: str) -> str:
        """Number of unique senders who transferred to this peer."""
        subset = store.df[store.df["peer_id"] == str(peer_id)]
        return _json({"peer_id": peer_id, "unique_senders": int(subset["sender_id"].nunique())})

    @tool
    def get_top_senders_by_count(limit: int = 10) -> str:
        """Top senders ranked by transaction count."""
        top = store.df.groupby("sender_id").size().sort_values(ascending=False).head(limit)
        return _json([{"sender_id": k, "txn_count": int(v)} for k, v in top.items()])

    @tool
    def get_top_senders_by_volume(limit: int = 10) -> str:
        """Top senders ranked by total transfer volume."""
        top = store.df.groupby("sender_id")["amount"].sum().sort_values(ascending=False).head(limit)
        return _json([{"sender_id": k, "volume": round(float(v), 2)} for k, v in top.items()])

    @tool
    def get_top_peers_by_count(limit: int = 10) -> str:
        """Top recipients (peer_id) ranked by transaction count."""
        top = store.df.groupby("peer_id").size().sort_values(ascending=False).head(limit)
        return _json([{"peer_id": k, "txn_count": int(v)} for k, v in top.items()])

    @tool
    def get_top_peers_by_volume(limit: int = 10) -> str:
        """Top recipients ranked by total volume received."""
        top = store.df.groupby("peer_id")["amount"].sum().sort_values(ascending=False).head(limit)
        return _json([{"peer_id": k, "volume": round(float(v), 2)} for k, v in top.items()])

    @tool
    def get_dau_for_date(target_date: str) -> str:
        """Daily Active Users (unique senders) on a specific date (YYYY-MM-DD)."""
        d = date.fromisoformat(target_date)
        count = int(store.df[store.df["order_date"] == d]["sender_id"].nunique())
        return _json({"date": target_date, "dau": count})

    @tool
    def get_wau_for_week(week_period: str) -> str:
        """Weekly Active Users for a week period string (e.g. 2026-03-02/2026-03-08)."""
        count = int(store.df[store.df["order_week"] == week_period]["sender_id"].nunique())
        return _json({"week": week_period, "wau": count})

    @tool
    def get_mau_for_month(month_period: str) -> str:
        """Monthly Active Users for a month period string (e.g. 2026-03)."""
        count = int(store.df[store.df["order_month"] == month_period]["sender_id"].nunique())
        return _json({"month": month_period, "mau": count})

    @tool
    def get_dau_mau_ratio() -> str:
        """DAU/MAU stickiness ratio from latest period in the dataset."""
        return _json({"dau_mau_ratio_pct": _lookup(store, "activity.dau_mau_ratio_pct")})

    @tool
    def get_transactions_per_active_user() -> str:
        """Average number of transfers per active sender."""
        return _json({"transactions_per_active_user": _lookup(store, "activity.transactions_per_active_user")})

    @tool
    def get_active_senders_count() -> str:
        """Total unique senders who initiated at least one transfer."""
        return _json({"active_senders": _lookup(store, "activity.active_senders")})

    @tool
    def get_active_recipients_count() -> str:
        """Total unique peers who received at least one transfer."""
        return _json({"active_recipients": _lookup(store, "activity.active_recipients")})

    @tool
    def get_new_users_count() -> str:
        """Total new users (first-time senders) in the dataset."""
        return _json({"new_users_total": _lookup(store, "growth.new_users_total")})

    @tool
    def get_new_active_users_count() -> str:
        """New users who completed at least one transfer."""
        return _json({"new_active_users": _lookup(store, "growth.new_active_users")})

    @tool
    def get_user_growth_rate() -> str:
        """Month-over-month active user growth rate (%)."""
        return _json({"user_growth_rate_pct": _lookup(store, "growth.user_growth_rate_pct")})

    @tool
    def get_d1_retention() -> str:
        """D1 retention: % of users returning the next day after first transfer."""
        return _json({"d1_retention_pct": _lookup(store, "retention.d1_retention_pct")})

    @tool
    def get_d7_retention() -> str:
        """D7 retention: % of users returning 7 days after first transfer."""
        return _json({"d7_retention_pct": _lookup(store, "retention.d7_retention_pct")})

    @tool
    def get_d30_retention() -> str:
        """D30 retention: % of users returning 30 days after first transfer."""
        return _json({"d30_retention_pct": _lookup(store, "retention.d30_retention_pct")})

    @tool
    def get_r7_transfer_retention() -> str:
        """R7 transfer retention: % of all senders who made a 2nd transfer within 7 days."""
        return _json({"r7_transfer_retention_pct": _lookup(store, "retention.r7_transfer_retention_pct")})

    @tool
    def get_r30_transfer_retention() -> str:
        """R30 transfer retention: % of all senders who made a 2nd transfer within 30 days."""
        return _json({"r30_transfer_retention_pct": _lookup(store, "retention.r30_transfer_retention_pct")})

    @tool
    def get_churn_rate() -> str:
        """Churn rate: % of users active last month but not this month."""
        return _json({"churn_rate_pct": _lookup(store, "retention.churn_rate_pct")})

    @tool
    def get_tpv() -> str:
        """Total Payment Volume — sum of all transfer amounts."""
        return _json({"tpv": _lookup(store, "transfer.tpv")})

    @tool
    def get_transfer_count() -> str:
        """Total number of transfers in the dataset."""
        return _json({"transfer_count": _lookup(store, "transfer.transfer_count")})

    @tool
    def get_average_transfer_value() -> str:
        """Average transfer amount (TPV / transfer count)."""
        return _json({"average_transfer_value": _lookup(store, "transfer.average_transfer_value")})

    @tool
    def get_conversion_rate() -> str:
        """Conversion rate (all loaded records are successful transfers)."""
        return _json({"conversion_rate_pct": _lookup(store, "transfer.conversion_rate_pct")})

    @tool
    def get_transfer_success_rate() -> str:
        """Transfer success rate (100% for successful-only dataset)."""
        return _json({"success_rate_pct": _lookup(store, "transfer.success_rate_pct")})

    @tool
    def get_repeat_sender_rate() -> str:
        """% of senders who made more than one transfer."""
        return _json({"repeat_sender_rate_pct": _lookup(store, "retention.repeat_sender_rate_pct")})

    @tool
    def list_senders_with_more_than_receivers(more_than: int) -> str:
        """List sender_id values for users who sent money to MORE than N distinct receivers/peers.
        Parameter more_than: use 2 when user asks 'more than 2 receivers' (returns senders with 3+ peers).
        Returns sender_ids list and per-sender receiver counts. NOT a single count — use for 'which senders' / 'sender id of users'."""
        return _json(queries.list_senders_with_more_than_receivers(store, more_than))

    @tool
    def get_senders_with_min_peers(min_peers: int) -> str:
        """Count only: how many senders transferred to at least min_peers distinct peer_id.
        For the actual sender_id list, use list_senders_with_more_than_receivers instead."""
        return _json(queries.count_senders_with_min_peers(store, min_peers))

    # --- Segment rules (time-window analytics) ---

    @tool
    def segment_rule1_senders_more_than_txns(days: int, more_than: int) -> str:
        """Rule 1: COUNT senders with MORE than N transactions in last `days` days.
        Example: days=7, more_than=5 → users with >5 txns in 7 days."""
        return _json(seg.rule1_senders_more_than_txns(store, days, more_than))

    @tool
    def segment_rule2_senders_exactly_txns(days: int, exactly: int) -> str:
        """Rule 2: COUNT senders with EXACTLY N transactions in last `days` days."""
        return _json(seg.rule2_senders_exactly_txns(store, days, exactly))

    @tool
    def segment_rule3_senders_min_active_days(days: int, min_days: int) -> str:
        """Rule 3: COUNT senders active on at least min_days DISTINCT dates in last `days` days."""
        return _json(seg.rule3_senders_min_active_days(store, days, min_days))

    @tool
    def segment_rule4_senders_total_amount_above(days: int, min_amount: float) -> str:
        """Rule 4: COUNT senders with total amount > min_amount (VND) in last `days` days."""
        return _json(seg.rule4_senders_total_amount_above(store, days, min_amount))

    @tool
    def segment_rule5_senders_avg_amount_above(
        days: int, min_avg: float, min_txns: int = 1
    ) -> str:
        """Rule 5: COUNT senders with average amount > min_avg and at least min_txns in window."""
        return _json(seg.rule5_senders_avg_amount_above(store, days, min_avg, min_txns))

    @tool
    def segment_rule6_sender_peer_pairs_more_than_txns(days: int, more_than: int) -> str:
        """Rule 6: COUNT (sender_id, peer_id) pairs with more than N transactions in window."""
        return _json(seg.rule6_sender_peer_pairs_more_than_txns(store, days, more_than))

    @tool
    def segment_rule7_senders_more_than_peers(days: int, more_than: int) -> str:
        """Rule 7: COUNT senders who sent to MORE than P unique peer_id in last `days` days."""
        return _json(seg.rule7_senders_more_than_peers(store, days, more_than))

    @tool
    def segment_rule8_peers_more_than_senders(days: int, more_than: int) -> str:
        """Rule 8: COUNT peers who received from MORE than P unique sender_id in window."""
        return _json(seg.rule8_peers_more_than_senders(store, days, more_than))

    @tool
    def segment_rule9_senders_more_than_products(days: int, more_than: int) -> str:
        """Rule 9: COUNT senders using MORE than K distinct product_code in window."""
        return _json(seg.rule9_senders_more_than_products(store, days, more_than))

    @tool
    def segment_rule10_bidirectional_pairs(days: int) -> str:
        """Rule 10: COUNT bidirectional pairs (A sent to B AND B sent to A) in window."""
        return _json(seg.rule10_bidirectional_pairs(store, days))

    @tool
    def get_volume_by_product_code() -> str:
        """Total volume grouped by product_code."""
        return _json(_lookup(store, "product.volume_by_product"))

    @tool
    def get_volume_by_app_id() -> str:
        """Total volume grouped by app_id."""
        return _json(_lookup(store, "app.volume_by_app_id"))

    @tool
    def get_daily_trend(limit: int = 30) -> str:
        """Daily transaction count and volume trend (most recent N days)."""
        trend = _lookup(store, "trends.daily_volume")
        return _json(trend[-limit:] if isinstance(trend, list) else trend)

    @tool
    def get_weekly_trend() -> str:
        """Weekly transaction count and volume trend."""
        weekly = (
            store.df.groupby("order_week")
            .agg(txn_count=("order_no", "count"), volume=("amount", "sum"))
            .reset_index()
        )
        return _json(weekly.to_dict(orient="records"))

    @tool
    def compare_two_senders(sender_id_a: str, sender_id_b: str) -> str:
        """Compare txn count and volume between two senders."""
        a = _filter_sender(store.df, sender_id_a)
        b = _filter_sender(store.df, sender_id_b)
        return _json(
            {
                "sender_a": {
                    "id": sender_id_a,
                    "txn_count": len(a),
                    "volume": round(float(a["amount"].sum()), 2),
                },
                "sender_b": {
                    "id": sender_id_b,
                    "txn_count": len(b),
                    "volume": round(float(b["amount"].sum()), 2),
                },
            }
        )

    @tool
    def get_amount_percentile(percentile: float) -> str:
        """Transfer amount at a given percentile (0-100)."""
        value = float(store.df["amount"].quantile(percentile / 100.0))
        return _json({"percentile": percentile, "amount": round(value, 2)})

    @tool
    def get_peak_transfer_hour() -> str:
        """Hour of day (0-23) with the highest transfer count."""
        hourly = store.df.groupby("order_hour").size().sort_values(ascending=False)
        peak_hour = int(hourly.index[0])
        return _json({"peak_hour": peak_hour, "txn_count": int(hourly.iloc[0])})

    @tool
    def get_median_transfer_amount() -> str:
        """Median transfer amount across all transactions."""
        return _json({"median_amount": _lookup(store, "global.median_amount")})

    @tool
    def list_transactions_in_last_days(days: int, limit: int = 50, single_day: bool = False) -> str:
        """List transfer transactions in a time window.
        days: window size. single_day=false → last N days rolling; single_day=true → calendar day N days before latest txn.
        limit: max rows returned (default 50). Use for 'list transactions in last 7 days' or 'transactions 7 days ago'."""
        return _json(queries.list_transactions_in_last_days(store, days, limit, single_day))

    @tool("cohort.retention_matrix")
    def cohort_retention_matrix(cohort_period: str, size: int = 12) -> str:
        """MoM or WoW cohort retention matrix by first-transfer period."""
        return _json(adv.cohort_retention_matrix(store, cohort_period, size))

    @tool("marketing.cac_and_ltv")
    def marketing_cac_and_ltv(blended_cac: float, months_horizon: int = 12) -> str:
        """Estimate LTV from transfer data and compare to blended CAC (VND)."""
        return _json(adv.marketing_cac_and_ltv(store, blended_cac, months_horizon))

    @tool("segmentation.rfm_scores")
    def segmentation_rfm_scores() -> str:
        """RFM user segmentation: Champions, Loyal, At Risk, Hibernating."""
        return _json(adv.segmentation_rfm_scores(store))

    @tool("risk.velocity_alerts")
    def risk_velocity_alerts(time_window_minutes: int, min_txns: int) -> str:
        """Flag senders with extreme transaction velocity in a minute window."""
        return _json(adv.risk_velocity_alerts(store, time_window_minutes, min_txns))

    @tool("activity.hourly_peak_load")
    def activity_hourly_peak_load(last_days: int = 30) -> str:
        """Txn count and TPV by day-of-week and hour for capacity planning."""
        return _json(adv.activity_hourly_peak_load(store, last_days))

    @tool("activity.custom_hourly_buckets")
    def activity_custom_hourly_buckets(bucket_size_hours: int, days: int = 30) -> str:
        """Group transactions into custom hourly bucket widths (1, 2, 3, 4, 6, or 12 hours)."""
        return _json(adv.activity_custom_hourly_buckets(store, bucket_size_hours, days))

    @tool("activity.peak_minute_velocity")
    def activity_peak_minute_velocity(days: int = 7) -> str:
        """Find the historical peak minute of transaction concurrency."""
        return _json(adv.activity_peak_minute_velocity(store, days))

    @tool("transfer.error_code_distribution")
    def transfer_error_code_distribution(days: int = 7) -> str:
        """Failure breakdown by error code (successful-only dataset returns SUCCESS)."""
        return _json(adv.transfer_error_code_distribution(store, days))

    @tool
    def get_executive_dashboard_summary() -> str:
        """Executive summary: DAU, MAU, DAU/MAU, retention, TPV, transfer count, avg value."""
        return _json(queries.executive_summary(store))

    tools = [
        lookup_precomputed_stat,
        list_precomputed_keys,
        get_user_txn_count,
        get_user_total_volume,
        get_user_unique_peers,
        get_user_avg_amount,
        get_user_first_transfer_date,
        get_user_last_transfer_date,
        get_user_transfers_by_product,
        get_user_r7_retention_status,
        get_user_transfers_in_date_range,
        get_peer_txn_count,
        get_peer_total_volume,
        get_peer_unique_senders,
        get_top_senders_by_count,
        get_top_senders_by_volume,
        get_top_peers_by_count,
        get_top_peers_by_volume,
        get_dau_for_date,
        get_wau_for_week,
        get_mau_for_month,
        get_dau_mau_ratio,
        get_transactions_per_active_user,
        get_active_senders_count,
        get_active_recipients_count,
        get_new_users_count,
        get_new_active_users_count,
        get_user_growth_rate,
        get_d1_retention,
        get_d7_retention,
        get_d30_retention,
        get_r7_transfer_retention,
        get_r30_transfer_retention,
        get_churn_rate,
        get_tpv,
        get_transfer_count,
        get_average_transfer_value,
        get_conversion_rate,
        get_transfer_success_rate,
        get_repeat_sender_rate,
        list_senders_with_more_than_receivers,
        get_senders_with_min_peers,
        segment_rule1_senders_more_than_txns,
        segment_rule2_senders_exactly_txns,
        segment_rule3_senders_min_active_days,
        segment_rule4_senders_total_amount_above,
        segment_rule5_senders_avg_amount_above,
        segment_rule6_sender_peer_pairs_more_than_txns,
        segment_rule7_senders_more_than_peers,
        segment_rule8_peers_more_than_senders,
        segment_rule9_senders_more_than_products,
        segment_rule10_bidirectional_pairs,
        get_volume_by_product_code,
        get_volume_by_app_id,
        get_daily_trend,
        get_weekly_trend,
        compare_two_senders,
        get_amount_percentile,
        get_peak_transfer_hour,
        get_median_transfer_amount,
        list_transactions_in_last_days,
        cohort_retention_matrix,
        marketing_cac_and_ltv,
        segmentation_rfm_scores,
        risk_velocity_alerts,
        activity_hourly_peak_load,
        activity_custom_hourly_buckets,
        activity_peak_minute_velocity,
        transfer_error_code_distribution,
        get_executive_dashboard_summary,
    ]

    if len(tools) != 70:
        raise RuntimeError(f"Expected 70 tools, got {len(tools)}")

    return tools
