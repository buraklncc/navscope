"""General utilities."""

from __future__ import annotations

import pandas as pd


def get_flag(row: pd.Series, column: str) -> bool:
    """Safely get a boolean flag from a pandas Series row."""
    value = row.get(column, False)
    if pd.isna(value):
        return False
    return bool(value)


def safe_float(value: float | None, default: float = 0.0) -> float:
    """Safely convert a pandas value to float, handling NaNs."""
    if pd.isna(value):
        return default
    return float(value)

