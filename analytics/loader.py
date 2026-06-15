"""Load transfer CSV into an in-memory pandas DataFrame."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from analytics.schema import CSV_COLUMNS, DEFAULT_DATA_PATH, ORDER_TIME_UNIT


def load_transfers(path: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """Load and normalize the transfers CSV into RAM."""
    data_path = path or os.environ.get("TRANSFERS_CSV_PATH", DEFAULT_DATA_PATH)
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Transfer data not found at {data_path}. "
            "Place your CSV at data/transfers.csv or set TRANSFERS_CSV_PATH."
        )

    df = pd.read_csv(
        data_path,
        dtype={
            "order_no": str,
            "app_id": str,
            "sender_id": str,
            "peer_id": str,
            "product_code": str,
            "amount": float,
            "order_time": "int64",
        },
        nrows=max_rows,
    )

    missing = [col for col in CSV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[CSV_COLUMNS].copy()
    df["order_dt"] = pd.to_datetime(df["order_time"], unit=ORDER_TIME_UNIT, utc=True)
    local_dt = df["order_dt"].dt.tz_convert(None)
    df["order_date"] = local_dt.dt.date
    df["order_week"] = local_dt.dt.to_period("W").astype(str)
    df["order_month"] = local_dt.dt.to_period("M").astype(str)
    df["order_hour"] = df["order_dt"].dt.hour
    return df.reset_index(drop=True)
