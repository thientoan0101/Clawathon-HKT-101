"""In-memory data store singleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataStore:
    """Holds the DataFrame and pre-computed statistics in RAM."""

    df: pd.DataFrame
    precomputed: dict[str, Any] = field(default_factory=dict)
    ready: bool = False

    def require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError("DataStore is not ready. Pre-compute has not finished.")


_store: DataStore | None = None


def get_store() -> DataStore:
    if _store is None or not _store.ready:
        raise RuntimeError("DataStore is not initialized. Call initialize_store() at startup.")
    return _store


def initialize_store(df: pd.DataFrame, precomputed: dict[str, Any]) -> DataStore:
    global _store
    _store = DataStore(df=df, precomputed=precomputed, ready=True)
    return _store


def is_ready() -> bool:
    return _store is not None and _store.ready
