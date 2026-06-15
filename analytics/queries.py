"""Shared query functions — used by tools, router, and direct executor."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from analytics.store import DataStore


def lookup_stat(store: DataStore, dotted_key: str) -> Any:
    node: Any = store.precomputed
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Unknown precomputed key: {dotted_key}")
        node = node[part]
    return node


def filter_sender(df: pd.DataFrame, sender_id: str) -> pd.DataFrame:
    return df[df["sender_id"] == str(sender_id)]


def user_txn_count(store: DataStore, sender_id: str) -> dict[str, Any]:
    count = len(filter_sender(store.df, sender_id))
    return {"sender_id": sender_id, "txn_count": count}


def user_total_volume(store: DataStore, sender_id: str) -> dict[str, Any]:
    subset = filter_sender(store.df, sender_id)
    return {"sender_id": sender_id, "total_volume": round(float(subset["amount"].sum()), 2)}


def user_unique_peers(store: DataStore, sender_id: str) -> dict[str, Any]:
    subset = filter_sender(store.df, sender_id)
    return {"sender_id": sender_id, "unique_peers": int(subset["peer_id"].nunique())}


def user_avg_amount(store: DataStore, sender_id: str) -> dict[str, Any]:
    subset = filter_sender(store.df, sender_id)
    avg = float(subset["amount"].mean()) if len(subset) else 0.0
    return {"sender_id": sender_id, "avg_amount": round(avg, 2)}


def executive_summary(store: DataStore) -> dict[str, Any]:
    p = store.precomputed
    return {
        "dau": p["activity"]["latest_dau"],
        "mau": p["activity"]["latest_mau"],
        "dau_mau_ratio_pct": p["activity"]["dau_mau_ratio_pct"],
        "new_users": p["growth"]["new_users_total"],
        "d30_retention_pct": p["retention"]["d30_retention_pct"],
        "r7_transfer_retention_pct": p["retention"]["r7_transfer_retention_pct"],
        "r30_transfer_retention_pct": p["retention"]["r30_transfer_retention_pct"],
        "tpv": p["transfer"]["tpv"],
        "transfer_count": p["transfer"]["transfer_count"],
        "unique_senders": p["global"]["unique_senders"],
        "average_transfer_value": p["transfer"]["average_transfer_value"],
        "success_rate_pct": p["transfer"]["success_rate_pct"],
    }


def list_senders_with_more_than_receivers(store: DataStore, more_than: int) -> dict[str, Any]:
    """Senders whose distinct peer_id count is strictly greater than more_than."""
    peer_counts = store.df.groupby("sender_id")["peer_id"].nunique()
    matched = peer_counts[peer_counts > more_than].sort_values(ascending=False)
    return {
        "more_than_receivers": more_than,
        "criteria": f"unique receivers > {more_than}",
        "sender_count": int(len(matched)),
        "sender_ids": [str(sid) for sid in matched.index.tolist()],
        "senders": [
            {"sender_id": str(sid), "receiver_count": int(cnt)} for sid, cnt in matched.items()
        ],
    }


def count_senders_with_min_peers(store: DataStore, min_peers: int) -> dict[str, Any]:
    """Count senders with at least min_peers distinct peers (inclusive)."""
    peer_counts = store.df.groupby("sender_id")["peer_id"].nunique()
    count = int((peer_counts >= min_peers).sum())
    return {"min_peers": min_peers, "sender_count": count}


def list_transactions_in_last_days(
    store: DataStore,
    days: int,
    limit: int = 50,
    single_day: bool = False,
) -> dict[str, Any]:
    """List transfer rows in a time window relative to the latest transaction in data."""
    df = store.df
    if df.empty:
        return {
            "days": days,
            "single_day": single_day,
            "total_count": 0,
            "returned_count": 0,
            "truncated": False,
            "window": {"start": None, "end": None},
            "transactions": [],
        }

    end_dt = df["order_dt"].max()
    if single_day:
        target_date = (end_dt - pd.Timedelta(days=days)).date()
        w = df[df["order_date"] == target_date].copy()
        window = {"start": str(target_date), "end": str(target_date), "days_ago": days}
    else:
        start_dt = end_dt - pd.Timedelta(days=days)
        w = df[df["order_dt"] >= start_dt].copy()
        window = {
            "start": str(w["order_date"].min()) if not w.empty else None,
            "end": str(w["order_date"].max()) if not w.empty else None,
            "days": days,
        }

    w = w.sort_values("order_dt", ascending=False)
    total = len(w)
    subset = w.head(limit)
    records = []
    for row in subset.itertuples(index=False):
        records.append(
            {
                "order_no": str(row.order_no),
                "sender_id": str(row.sender_id),
                "peer_id": str(row.peer_id),
                "product_code": str(row.product_code),
                "amount": round(float(row.amount), 2),
                "order_date": str(row.order_date),
            }
        )

    return {
        "days": days,
        "single_day": single_day,
        "window": window,
        "total_count": total,
        "returned_count": len(records),
        "truncated": total > limit,
        "transactions": records,
    }


def dumps(data: Any) -> str:
    return json.dumps(data, default=str, ensure_ascii=False)
