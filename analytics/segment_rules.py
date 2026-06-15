"""Segmentation rules over transfer data within rolling time windows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analytics.store import DataStore


def _window_df(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Rows in the last `days` calendar days ending at the latest transaction."""
    if df.empty:
        return df
    end = df["order_dt"].max()
    start = end - pd.Timedelta(days=days)
    return df[df["order_dt"] >= start].copy()


def _window_meta(df: pd.DataFrame, days: int) -> dict[str, str]:
    w = _window_df(df, days)
    if w.empty:
        return {"days": str(days), "start": None, "end": None, "txn_in_window": 0}
    return {
        "days": str(days),
        "start": str(w["order_date"].min()),
        "end": str(w["order_date"].max()),
        "txn_in_window": len(w),
    }


def rule1_senders_more_than_txns(store: DataStore, days: int, more_than: int) -> dict[str, Any]:
    """Rule 1: count sender_id with transaction count > more_than in window."""
    w = _window_df(store.df, days)
    counts = w.groupby("sender_id").size()
    matched = counts[counts > more_than]
    return {
        "rule": 1,
        "description": f"Senders with more than {more_than} transactions in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"more_than_txns": more_than},
        "sender_count": int(len(matched)),
        "sender_ids": [str(s) for s in matched.index.tolist()],
    }


def rule2_senders_exactly_txns(store: DataStore, days: int, exactly: int) -> dict[str, Any]:
    """Rule 2: count sender_id with exactly n transactions in window."""
    w = _window_df(store.df, days)
    counts = w.groupby("sender_id").size()
    matched = counts[counts == exactly]
    return {
        "rule": 2,
        "description": f"Senders with exactly {exactly} transactions in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"exactly_txns": exactly},
        "sender_count": int(len(matched)),
        "sender_ids": [str(s) for s in matched.index.tolist()],
    }


def rule3_senders_min_active_days(store: DataStore, days: int, min_days: int) -> dict[str, Any]:
    """Rule 3: senders with transactions on at least min_days distinct dates."""
    w = _window_df(store.df, days)
    active_days = w.groupby("sender_id")["order_date"].nunique()
    matched = active_days[active_days >= min_days]
    return {
        "rule": 3,
        "description": f"Senders active on at least {min_days} distinct days in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"min_active_days": min_days},
        "sender_count": int(len(matched)),
        "sender_ids": [str(s) for s in matched.index.tolist()],
    }


def rule4_senders_total_amount_above(store: DataStore, days: int, min_amount: float) -> dict[str, Any]:
    """Rule 4: senders with total amount > min_amount in window."""
    w = _window_df(store.df, days)
    totals = w.groupby("sender_id")["amount"].sum()
    matched = totals[totals > min_amount]
    return {
        "rule": 4,
        "description": f"Senders with total amount > {min_amount:,.0f} in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"min_total_amount": min_amount},
        "sender_count": int(len(matched)),
        "sender_ids": [str(s) for s in matched.index.tolist()],
    }


def rule5_senders_avg_amount_above(
    store: DataStore, days: int, min_avg: float, min_txns: int = 1
) -> dict[str, Any]:
    """Rule 5: senders with average amount > min_avg and at least min_txns transactions."""
    w = _window_df(store.df, days)
    stats = w.groupby("sender_id").agg(txn_count=("order_no", "count"), avg_amount=("amount", "mean"))
    matched = stats[(stats["txn_count"] >= min_txns) & (stats["avg_amount"] > min_avg)]
    return {
        "rule": 5,
        "description": (
            f"Senders with avg amount > {min_avg:,.0f} and at least {min_txns} txns "
            f"in last {days} days"
        ),
        "window": _window_meta(store.df, days),
        "threshold": {"min_avg_amount": min_avg, "min_txns": min_txns},
        "sender_count": int(len(matched)),
        "sender_ids": [str(s) for s in matched.index.tolist()],
    }


def rule6_sender_peer_pairs_more_than_txns(
    store: DataStore, days: int, more_than: int
) -> dict[str, Any]:
    """Rule 6: count (sender_id, peer_id) pairs with transaction count > more_than."""
    w = _window_df(store.df, days)
    pair_counts = w.groupby(["sender_id", "peer_id"]).size()
    matched = pair_counts[pair_counts > more_than]
    pairs = [
        {"sender_id": str(s), "peer_id": str(p), "txn_count": int(c)}
        for (s, p), c in matched.items()
    ]
    return {
        "rule": 6,
        "description": f"Sender-peer pairs with more than {more_than} txns in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"more_than_txns": more_than},
        "pair_count": len(pairs),
        "pairs": pairs[:50],
        "pairs_truncated": len(pairs) > 50,
    }


def rule7_senders_more_than_peers(store: DataStore, days: int, more_than: int) -> dict[str, Any]:
    """Rule 7: senders who sent to more than p distinct peer_id in window."""
    w = _window_df(store.df, days)
    peer_counts = w.groupby("sender_id")["peer_id"].nunique()
    matched = peer_counts[peer_counts > more_than]
    return {
        "rule": 7,
        "description": f"Senders with more than {more_than} unique peers in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"more_than_peers": more_than},
        "sender_count": int(len(matched)),
        "sender_ids": [str(s) for s in matched.index.tolist()],
    }


def rule8_peers_more_than_senders(store: DataStore, days: int, more_than: int) -> dict[str, Any]:
    """Rule 8: peers who received from more than p distinct sender_id in window."""
    w = _window_df(store.df, days)
    sender_counts = w.groupby("peer_id")["sender_id"].nunique()
    matched = sender_counts[sender_counts > more_than]
    return {
        "rule": 8,
        "description": f"Peers with more than {more_than} unique senders in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"more_than_senders": more_than},
        "peer_count": int(len(matched)),
        "peer_ids": [str(p) for p in matched.index.tolist()],
    }


def rule9_senders_more_than_products(store: DataStore, days: int, more_than: int) -> dict[str, Any]:
    """Rule 9: senders using more than k distinct product_code in window."""
    w = _window_df(store.df, days)
    product_counts = w.groupby("sender_id")["product_code"].nunique()
    matched = product_counts[product_counts > more_than]
    return {
        "rule": 9,
        "description": f"Senders with more than {more_than} product codes in last {days} days",
        "window": _window_meta(store.df, days),
        "threshold": {"more_than_products": more_than},
        "sender_count": int(len(matched)),
        "sender_ids": [str(s) for s in matched.index.tolist()],
    }


def rule10_bidirectional_pairs(store: DataStore, days: int) -> dict[str, Any]:
    """Rule 10: pairs (A,B) where A sent to B and B sent to A in window."""
    w = _window_df(store.df, days)
    forward = set(zip(w["sender_id"].astype(str), w["peer_id"].astype(str)))
    # Bidirectional when reverse edge exists and peer can act as sender
    bidirectional = []
    seen = set()
    for sender, peer in forward:
        if (peer, sender) in forward:
            key = tuple(sorted([sender, peer]))
            if key not in seen:
                seen.add(key)
                bidirectional.append({"party_a": key[0], "party_b": key[1]})
    return {
        "rule": 10,
        "description": f"Bidirectional sender-peer pairs in last {days} days",
        "window": _window_meta(store.df, days),
        "pair_count": len(bidirectional),
        "pairs": bidirectional[:50],
        "pairs_truncated": len(bidirectional) > 50,
    }
