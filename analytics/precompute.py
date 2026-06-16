"""Pre-compute executive dashboard and retention metrics at startup."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def _cohort_retention(df: pd.DataFrame, days: int) -> float:
    """% of first-time senders who transfer again exactly `days` later (calendar day)."""
    first = df.groupby("sender_id")["order_date"].min().rename("first_date")
    merged = df.merge(first, on="sender_id")
    merged["days_since_first"] = (
        pd.to_datetime(merged["order_date"]) - pd.to_datetime(merged["first_date"])
    ).dt.days
    cohort_users = set(first.index)
    if not cohort_users:
        return 0.0
    returned = merged[(merged["days_since_first"] == days)]["sender_id"].nunique()
    return _pct(returned, len(cohort_users))


def _transfer_retention(df: pd.DataFrame, window_days: int) -> float:
    """R7/R30: % of senders who made a 2nd transfer within N days of first transfer."""
    first_transfer = df.groupby("sender_id")["order_dt"].min()
    second_transfer = df.groupby("sender_id")["order_dt"].apply(
        lambda s: s.sort_values().iloc[1] if len(s) > 1 else pd.NaT
    )
    retained = 0
    total_repeaters = 0
    for sender_id, first_dt in first_transfer.items():
        second_dt = second_transfer.get(sender_id)
        if pd.isna(second_dt):
            continue
        total_repeaters += 1
        gap = (second_dt - first_dt).days
        if gap <= window_days:
            retained += 1
    # Denominator: all senders with at least one transfer
    all_senders = first_transfer.shape[0]
    if all_senders == 0:
        return 0.0
    return _pct(retained, all_senders)


def _repeat_sender_rate(df: pd.DataFrame) -> float:
    sender_counts = df.groupby("sender_id").size()
    repeaters = (sender_counts > 1).sum()
    return _pct(repeaters, len(sender_counts))


def _senders_with_min_peers(df: pd.DataFrame, min_peers: int) -> int:
    peer_counts = df.groupby("sender_id")["peer_id"].nunique()
    return int((peer_counts >= min_peers).sum())


def _daily_active_users(df: pd.DataFrame) -> dict[str, int]:
    dau = df.groupby("order_date")["sender_id"].nunique()
    return {str(k): int(v) for k, v in dau.items()}


def _user_growth_rate(df: pd.DataFrame) -> float:
    monthly = df.groupby("order_month")["sender_id"].nunique().sort_index()
    if len(monthly) < 2:
        return 0.0
    prev, curr = monthly.iloc[-2], monthly.iloc[-1]
    if prev == 0:
        return 0.0
    return round(100.0 * (curr - prev) / prev, 4)


def _new_users_by_month(df: pd.DataFrame) -> dict[str, int]:
    first_dates = df.groupby("sender_id")["order_month"].min()
    counts = first_dates.value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _churn_rate(df: pd.DataFrame) -> float:
    """Users active in prior month but not in latest month."""
    months = sorted(df["order_month"].unique())
    if len(months) < 2:
        return 0.0
    prev_month, curr_month = months[-2], months[-1]
    prev_users = set(df[df["order_month"] == prev_month]["sender_id"])
    curr_users = set(df[df["order_month"] == curr_month]["sender_id"])
    if not prev_users:
        return 0.0
    churned = len(prev_users - curr_users)
    return _pct(churned, len(prev_users))


def _channel_and_platform_breakdown(df: pd.DataFrame) -> dict[str, Any]:
    """TPV and txn count by app_id (channel proxy) and product_code (rail proxy)."""
    if df.empty:
        return {"by_app_id": {}, "by_product_code": {}, "note": "No data"}
    by_app = (
        df.groupby("app_id")
        .agg(txn_count=("order_no", "count"), tpv=("amount", "sum"))
        .reset_index()
    )
    by_product = (
        df.groupby("product_code")
        .agg(txn_count=("order_no", "count"), tpv=("amount", "sum"))
        .reset_index()
    )
    return {
        "note": "app_id used as channel/platform proxy; product_code as transfer rail proxy",
        "by_app_id": {
            str(row.app_id): {
                "txn_count": int(row.txn_count),
                "tpv": round(float(row.tpv), 2),
            }
            for row in by_app.itertuples(index=False)
        },
        "by_product_code": {
            str(row.product_code): {
                "txn_count": int(row.txn_count),
                "tpv": round(float(row.tpv), 2),
            }
            for row in by_product.itertuples(index=False)
        },
    }


def _forecast_next_month(df: pd.DataFrame) -> dict[str, Any]:
    """Simple exponential smoothing forecast for next month TPV and MAU."""
    if df.empty:
        return {
            "predicted_tpv": 0.0,
            "predicted_mau": 0,
            "confidence_interval_low": 0.0,
            "confidence_interval_high": 0.0,
            "method": "exponential_smoothing_alpha_0.3",
        }
    monthly = (
        df.groupby("order_month")
        .agg(tpv=("amount", "sum"), mau=("sender_id", "nunique"))
        .sort_index()
    )
    alpha = 0.3
    if len(monthly) == 1:
        pred_tpv = float(monthly["tpv"].iloc[-1])
        pred_mau = int(monthly["mau"].iloc[-1])
    else:
        pred_tpv = alpha * float(monthly["tpv"].iloc[-1]) + (1 - alpha) * float(
            monthly["tpv"].iloc[-2]
        )
        pred_mau = int(
            round(alpha * monthly["mau"].iloc[-1] + (1 - alpha) * monthly["mau"].iloc[-2])
        )
    std_tpv = float(monthly["tpv"].std()) if len(monthly) > 1 else pred_tpv * 0.1
    return {
        "predicted_tpv": round(pred_tpv, 2),
        "predicted_mau": pred_mau,
        "confidence_interval_low": round(max(pred_tpv - std_tpv, 0), 2),
        "confidence_interval_high": round(pred_tpv + std_tpv, 2),
        "method": "exponential_smoothing_alpha_0.3",
        "based_on_months": [str(m) for m in monthly.index.tolist()],
    }


def _time_of_day_breakdown(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Macro day-part buckets by transaction hour."""
    if df.empty:
        return []

    buckets = [
        ("Early Morning", "00-05", range(0, 6)),
        ("Morning", "06-11", range(6, 12)),
        ("Afternoon", "12-17", range(12, 18)),
        ("Evening", "18-21", range(18, 22)),
        ("Night", "22-23", range(22, 24)),
    ]
    total_count = len(df)
    total_tpv = float(df["amount"].sum())
    results: list[dict[str, Any]] = []

    for name, hour_range, hours in buckets:
        subset = df[df["order_hour"].isin(hours)]
        txn_count = int(len(subset))
        tpv = round(float(subset["amount"].sum()), 2)
        results.append(
            {
                "time_bucket_name": name,
                "hour_range": hour_range,
                "txn_count": txn_count,
                "percentage_of_count": round(100.0 * txn_count / total_count, 2) if total_count else 0.0,
                "total_tpv": tpv,
                "percentage_of_tpv": round(100.0 * tpv / total_tpv, 2) if total_tpv else 0.0,
            }
        )
    results.sort(key=lambda x: x["txn_count"], reverse=True)
    return results


_RANKING_LIMIT = 20


def _top_senders_by_count(df: pd.DataFrame, limit: int = _RANKING_LIMIT) -> list[dict[str, Any]]:
    if df.empty:
        return []
    top = df.groupby("sender_id").size().sort_values(ascending=False).head(limit)
    return [{"sender_id": str(k), "txn_count": int(v)} for k, v in top.items()]


def _top_senders_by_volume(df: pd.DataFrame, limit: int = _RANKING_LIMIT) -> list[dict[str, Any]]:
    if df.empty:
        return []
    top = df.groupby("sender_id")["amount"].sum().sort_values(ascending=False).head(limit)
    return [{"sender_id": str(k), "volume": round(float(v), 2)} for k, v in top.items()]


def _top_peers_by_count(df: pd.DataFrame, limit: int = _RANKING_LIMIT) -> list[dict[str, Any]]:
    if df.empty:
        return []
    top = df.groupby("peer_id").size().sort_values(ascending=False).head(limit)
    return [{"peer_id": str(k), "txn_count": int(v)} for k, v in top.items()]


def _top_peers_by_volume(df: pd.DataFrame, limit: int = _RANKING_LIMIT) -> list[dict[str, Any]]:
    if df.empty:
        return []
    top = df.groupby("peer_id")["amount"].sum().sort_values(ascending=False).head(limit)
    return [{"peer_id": str(k), "volume": round(float(v), 2)} for k, v in top.items()]


def _top_insight(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def precompute_all(df: pd.DataFrame) -> dict[str, Any]:
    """Build the full pre-computed statistics manifest."""
    total_txn = len(df)
    tpv = float(df["amount"].sum())
    unique_senders = int(df["sender_id"].nunique())
    unique_peers = int(df["peer_id"].nunique())
    avg_amount = float(df["amount"].mean()) if total_txn else 0.0
    median_amount = float(df["amount"].median()) if total_txn else 0.0

    latest_date = df["order_date"].max()
    latest_dau = int(df[df["order_date"] == latest_date]["sender_id"].nunique()) if total_txn else 0
    latest_week = df["order_week"].max() if total_txn else ""
    latest_month = df["order_month"].max() if total_txn else ""
    wau = int(df[df["order_week"] == latest_week]["sender_id"].nunique()) if total_txn else 0
    mau = int(df[df["order_month"] == latest_month]["sender_id"].nunique()) if total_txn else 0
    dau_mau = _pct(latest_dau, mau) if mau else 0.0

    first_transfer = df.groupby("sender_id")["order_dt"].min()
    new_users_total = int(len(first_transfer))
    new_active_users = new_users_total  # all loaded senders completed at least one transfer

    daily_trend = (
        df.groupby("order_date")
        .agg(txn_count=("order_no", "count"), volume=("amount", "sum"))
        .reset_index()
    )
    daily_trend["order_date"] = daily_trend["order_date"].astype(str)

    top_senders_by_count = _top_senders_by_count(df)
    top_senders_by_volume = _top_senders_by_volume(df)
    top_peers_by_count = _top_peers_by_count(df)
    top_peers_by_volume = _top_peers_by_volume(df)

    return {
        "global": {
            "total_transactions": total_txn,
            "tpv": round(tpv, 2),
            "unique_senders": unique_senders,
            "unique_peers": unique_peers,
            "mean_amount": round(avg_amount, 2),
            "median_amount": round(median_amount, 2),
            "avg_txn_per_sender": round(total_txn / unique_senders, 4) if unique_senders else 0,
        },
        "activity": {
            "latest_dau": latest_dau,
            "latest_wau": wau,
            "latest_mau": mau,
            "dau_mau_ratio_pct": dau_mau,
            "transactions_per_active_user": round(total_txn / unique_senders, 4) if unique_senders else 0,
            "active_senders": unique_senders,
            "active_recipients": unique_peers,
            "daily_active_users": _daily_active_users(df),
            "time_of_day_breakdown": _time_of_day_breakdown(df),
        },
        "growth": {
            "new_users_total": new_users_total,
            "new_active_users": new_active_users,
            "user_growth_rate_pct": _user_growth_rate(df),
            "new_users_by_month": _new_users_by_month(df),
            "forecasting_next_month": _forecast_next_month(df),
        },
        "retention": {
            "d1_retention_pct": _cohort_retention(df, 1),
            "d7_retention_pct": _cohort_retention(df, 7),
            "d30_retention_pct": _cohort_retention(df, 30),
            "r7_transfer_retention_pct": _transfer_retention(df, 7),
            "r30_transfer_retention_pct": _transfer_retention(df, 30),
            "churn_rate_pct": _churn_rate(df),
            "repeat_sender_rate_pct": _repeat_sender_rate(df),
        },
        "transfer": {
            "transfer_count": total_txn,
            "tpv": round(tpv, 2),
            "average_transfer_value": round(avg_amount, 2),
            "conversion_rate_pct": 100.0,
            "success_rate_pct": 100.0,
        },
        "network": {
            "senders_with_more_than_3_peers": _senders_with_min_peers(df, 4),
            "senders_with_more_than_1_peer": _senders_with_min_peers(df, 2),
        },
        "trends": {
            "daily_volume": daily_trend.to_dict(orient="records"),
        },
        "product": {
            "volume_by_product": {
                str(k): round(float(v), 2)
                for k, v in df.groupby("product_code")["amount"].sum().items()
            },
            "count_by_product": {str(k): int(v) for k, v in df.groupby("product_code").size().items()},
        },
        "app": {
            "volume_by_app_id": {
                str(k): round(float(v), 2) for k, v in df.groupby("app_id")["amount"].sum().items()
            },
        },
        "breakdown": {
            "channel_and_platform": _channel_and_platform_breakdown(df),
        },
        "ranking": {
            "top_senders_by_count": top_senders_by_count,
            "top_senders_by_volume": top_senders_by_volume,
            "top_peers_by_count": top_peers_by_count,
            "top_peers_by_volume": top_peers_by_volume,
        },
        "insights": {
            "top_sender_by_count": _top_insight(top_senders_by_count),
            "top_sender_by_volume": _top_insight(top_senders_by_volume),
            "top_receiver_by_count": _top_insight(top_peers_by_count),
            "top_receiver_by_volume": _top_insight(top_peers_by_volume),
        },
        "meta": {
            "date_range_start": str(df["order_date"].min()) if total_txn else None,
            "date_range_end": str(df["order_date"].max()) if total_txn else None,
            "row_count": total_txn,
        },
    }
