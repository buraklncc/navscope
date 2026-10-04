"""Return calculations."""

from __future__ import annotations

import pandas as pd

from fund_analyzer.data_provider import clean_price_data


RETURN_COLUMNS = [
    "daily_return_pct",
    "weekly_return_pct",
    "monthly_return_pct",
    "three_month_return_pct",
    "latest_price",
]


def add_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add daily, weekly, monthly, and three-month return percentages."""

    required = {"fund_code", "date", "price"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required column(s): {', '.join(sorted(missing))}")

    prices = clean_price_data(prices, source_name="prices")
    if prices.empty:
        return _with_empty_return_columns(prices)

    groups: list[pd.DataFrame] = []
    for _, group in prices.groupby("fund_code", sort=False):
        enriched = group.copy()
        enriched["daily_return_pct"] = enriched["price"].pct_change() * 100
        enriched["weekly_return_pct"] = (enriched["price"] / enriched["price"].shift(5) - 1) * 100
        enriched["monthly_return_pct"] = (enriched["price"] / enriched["price"].shift(21) - 1) * 100
        enriched["three_month_return_pct"] = (
            enriched["price"] / enriched["price"].shift(63) - 1
        ) * 100
        enriched["latest_price"] = enriched["price"]
        groups.append(enriched)

    return pd.concat(groups, ignore_index=True)


def _with_empty_return_columns(prices: pd.DataFrame) -> pd.DataFrame:
    enriched = prices.copy()
    for column in RETURN_COLUMNS:
        enriched[column] = pd.Series(dtype=float)
    return enriched

