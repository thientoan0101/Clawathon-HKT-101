"""Advanced analytics: cohort, RFM, LTV, velocity, hourly load, forecasting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analytics.store import DataStore


def _window_df(df: pd.DataFrame, last_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df["order_dt"].max()
    start = end - pd.Timedelta(days=last_days)
    return df[df["order_dt"] >= start].copy()


def cohort_retention_matrix(
    store: DataStore, cohort_period: str = "month", size: int = 12
) -> dict[str, Any]:
    """Classic cohort retention matrix by first-transfer week or month."""
    df = store.df
    if df.empty:
        return {"cohort_period": cohort_period, "cohorts": []}

    period_col = "order_week" if cohort_period == "week" else "order_month"
    first_period = df.groupby("sender_id")[period_col].min()
    cohort_sizes = first_period.value_counts().sort_index().tail(size)

    all_periods = sorted(df[period_col].unique())
    cohorts: list[dict[str, Any]] = []

    for cohort_label, total_users in cohort_sizes.items():
        cohort_senders = set(first_period[first_period == cohort_label].index)
        if cohort_label not in all_periods:
            continue
        start_idx = all_periods.index(cohort_label)
        retention: list[dict[str, Any]] = []
        for offset, period in enumerate(all_periods[start_idx : start_idx + 8]):
            active = int(
                df[(df[period_col] == period) & (df["sender_id"].isin(cohort_senders))][
                    "sender_id"
                ].nunique()
            )
            pct = round(100.0 * active / total_users, 2) if total_users else 0.0
            retention.append(
                {
                    "period_offset": offset,
                    "period": str(period),
                    "active_users": active,
                    "retention_pct": pct,
                }
            )
        cohorts.append(
            {
                "cohort_date": str(cohort_label),
                "total_users": int(total_users),
                "active_in_period_n": retention,
            }
        )

    return {"cohort_period": cohort_period, "size": size, "cohorts": cohorts}


def marketing_cac_and_ltv(
    store: DataStore, blended_cac: float, months_horizon: int = 12
) -> dict[str, Any]:
    """Estimate LTV from observed transfer behavior and compare to CAC."""
    df = store.df
    p = store.precomputed
    if df.empty or blended_cac <= 0:
        return {
            "blended_cac": blended_cac,
            "months_horizon": months_horizon,
            "estimated_ltv": 0.0,
            "ltv_cac_ratio": 0.0,
            "payback_months": None,
        }

    unique_senders = max(int(p["global"]["unique_senders"]), 1)
    months_active = max(len(df["order_month"].unique()), 1)
    monthly_tpv = float(p["transfer"]["tpv"]) / months_active
    revenue_per_user_month = monthly_tpv / unique_senders

    churn_pct = float(p["retention"].get("churn_rate_pct", 0.0))
    churn = churn_pct / 100.0 if churn_pct > 0 else 0.05
    expected_months = min(1.0 / churn, float(months_horizon))
    estimated_ltv = round(revenue_per_user_month * expected_months, 2)
    ltv_cac_ratio = round(estimated_ltv / blended_cac, 4)
    payback_months = (
        round(blended_cac / revenue_per_user_month, 2) if revenue_per_user_month > 0 else None
    )

    return {
        "blended_cac": blended_cac,
        "months_horizon": months_horizon,
        "estimated_ltv": estimated_ltv,
        "ltv_cac_ratio": ltv_cac_ratio,
        "payback_months": payback_months,
        "revenue_per_user_month": round(revenue_per_user_month, 2),
        "expected_active_months": round(expected_months, 2),
    }


def _rfm_segment(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    return "Potential"


def segmentation_rfm_scores(store: DataStore) -> dict[str, Any]:
    """RFM segmentation on senders: Recency, Frequency, Monetary quintiles."""
    df = store.df
    if df.empty:
        return {"segments": [], "total_users": 0}

    max_dt = df["order_dt"].max()
    agg = df.groupby("sender_id").agg(
        last_dt=("order_dt", "max"),
        frequency=("order_no", "count"),
        monetary=("amount", "sum"),
    )
    agg["recency_days"] = (max_dt - agg["last_dt"]).dt.days

    def quintile(series: pd.Series, invert: bool = False) -> pd.Series:
        if len(series) < 5:
            ranks = series.rank(method="first")
            return ranks.apply(lambda x: min(int(x), 5))
        try:
            scores = pd.qcut(series.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            scores = pd.cut(series.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        result = scores.astype(int)
        return 6 - result if invert else result

    agg["R"] = quintile(agg["recency_days"], invert=True)
    agg["F"] = quintile(agg["frequency"])
    agg["M"] = quintile(agg["monetary"])
    agg["segment"] = [_rfm_segment(int(r), int(f), int(m)) for r, f, m in zip(agg["R"], agg["F"], agg["M"])]

    total = len(agg)
    segments: list[dict[str, Any]] = []
    for name, group in agg.groupby("segment"):
        segments.append(
            {
                "segment_name": str(name),
                "user_count": int(len(group)),
                "percentage_of_total": round(100.0 * len(group) / total, 2),
                "avg_monetary_value": round(float(group["monetary"].mean()), 2),
            }
        )
    segments.sort(key=lambda x: x["user_count"], reverse=True)
    return {"total_users": total, "segments": segments}


def risk_velocity_alerts(
    store: DataStore, time_window_minutes: int, min_txns: int
) -> dict[str, Any]:
    """Flag senders with >= min_txns inside any rolling minute window."""
    df = store.df.sort_values("order_dt")
    if df.empty:
        return {"time_window_minutes": time_window_minutes, "min_txns": min_txns, "alerts": []}

    window = pd.Timedelta(minutes=time_window_minutes)
    alerts: list[dict[str, Any]] = []

    for sender_id, group in df.groupby("sender_id"):
        times = group["order_dt"].sort_values().tolist()
        max_in_window = 0
        best_start = None
        for i, start in enumerate(times):
            count = sum(1 for t in times[i:] if t <= start + window)
            if count > max_in_window:
                max_in_window = count
                best_start = start
        if max_in_window >= min_txns:
            subset = group[
                (group["order_dt"] >= best_start) & (group["order_dt"] <= best_start + window)
            ]
            alerts.append(
                {
                    "sender_id": str(sender_id),
                    "txn_count": int(max_in_window),
                    "total_amount": round(float(subset["amount"].sum()), 2),
                    "window_start_time": str(best_start),
                }
            )

    alerts.sort(key=lambda x: x["txn_count"], reverse=True)
    return {
        "time_window_minutes": time_window_minutes,
        "min_txns": min_txns,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


def activity_hourly_peak_load(store: DataStore, last_days: int = 30) -> dict[str, Any]:
    """Aggregate txn count and TPV by day-of-week and hour."""
    w = _window_df(store.df, last_days)
    if w.empty:
        return {"last_days": last_days, "hourly_load": []}

    w = w.copy()
    w["day_of_week"] = pd.to_datetime(w["order_date"]).dt.day_name()
    grouped = (
        w.groupby(["day_of_week", "order_hour"])
        .agg(txn_count=("order_no", "count"), tpv=("amount", "sum"))
        .reset_index()
    )
    records = [
        {
            "day_of_week": str(row.day_of_week),
            "hour_of_day": int(row.order_hour),
            "avg_txn_count": int(row.txn_count),
            "avg_tpv": round(float(row.tpv), 2),
        }
        for row in grouped.itertuples(index=False)
    ]
    records.sort(key=lambda x: x["avg_txn_count"], reverse=True)
    return {"last_days": last_days, "hourly_load": records}


def _hour_to_bucket(hour: int, bucket_size_hours: int) -> int:
    return (hour // bucket_size_hours) * bucket_size_hours


def activity_custom_hourly_buckets(
    store: DataStore, bucket_size_hours: int, days: int = 30
) -> dict[str, Any]:
    """Group transactions into fixed-width hourly buckets over a lookback window."""
    allowed = {1, 2, 3, 4, 6, 12}
    if bucket_size_hours not in allowed:
        raise ValueError(f"bucket_size_hours must be one of {sorted(allowed)}")

    w = _window_df(store.df, days)
    if w.empty:
        return {"bucket_size_hours": bucket_size_hours, "days": days, "buckets": []}

    w = w.copy()
    w["bucket_start_hour"] = w["order_hour"].apply(
        lambda h: _hour_to_bucket(int(h), bucket_size_hours)
    )
    grouped = (
        w.groupby("bucket_start_hour")
        .agg(txn_count=("order_no", "count"), total_amount=("amount", "sum"))
        .reset_index()
        .sort_values("bucket_start_hour")
    )

    buckets = []
    for row in grouped.itertuples(index=False):
        start = int(row.bucket_start_hour)
        end = min(start + bucket_size_hours - 1, 23)
        buckets.append(
            {
                "bucket_label": f"{start:02d}:00-{end:02d}:59",
                "bucket_start_hour": start,
                "bucket_size_hours": bucket_size_hours,
                "txn_count": int(row.txn_count),
                "total_amount": round(float(row.total_amount), 2),
            }
        )

    return {
        "bucket_size_hours": bucket_size_hours,
        "days": days,
        "bucket_count": len(buckets),
        "buckets": buckets,
    }


def activity_peak_minute_velocity(store: DataStore, days: int = 7) -> dict[str, Any]:
    """Find the single peak minute of transaction concurrency in the lookback window."""
    w = _window_df(store.df, days)
    if w.empty:
        return {
            "days": days,
            "peak_timestamp": None,
            "max_txns_per_minute": 0,
            "baseline_avg_minute": 0.0,
        }

    w = w.copy()
    w["order_minute"] = w["order_dt"].dt.floor("min")
    per_minute = w.groupby("order_minute").size()
    peak_ts = per_minute.idxmax()
    max_txns = int(per_minute.max())
    baseline = round(float(per_minute.mean()), 4) if len(per_minute) else 0.0

    return {
        "days": days,
        "peak_timestamp": str(peak_ts),
        "max_txns_per_minute": max_txns,
        "baseline_avg_minute": baseline,
        "active_minutes": int(len(per_minute)),
        "total_transactions": int(len(w)),
    }


def transfer_error_code_distribution(store: DataStore, days: int = 7) -> dict[str, Any]:
    """Error code breakdown — dataset contains successful transfers only."""
    w = _window_df(store.df, days)
    total = len(w)
    return {
        "days": days,
        "note": "Dataset contains successful transfers only; no error_code column available.",
        "total_transactions": total,
        "errors": [
            {
                "error_code": "SUCCESS",
                "error_description": "Transfer completed successfully",
                "failure_count": 0,
                "percentage_of_failures": 0.0,
            }
        ],
        "success_count": total,
        "success_rate_pct": 100.0 if total else 0.0,
    }
