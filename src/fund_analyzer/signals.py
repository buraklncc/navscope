"""Latest-date rule signal generation."""

from __future__ import annotations

import math

import pandas as pd

from fund_analyzer import config
from fund_analyzer.data_provider import clean_price_data


BOOLEAN_SIGNAL_COLUMNS = [
    "ema10_above_ema21",
    "ema10_crossed_above_ema21",
    "macd_above_signal",
    "histogram_positive",
    "histogram_increasing",
    "histogram_worsening",
    "new_high_1m",
    "above_recent_low",
    "monthly_return_positive",
]

SIGNAL_COLUMNS = [
    "fund_code",
    "date",
    "latest_price",
    "daily_return_pct",
    "weekly_return_pct",
    "monthly_return_pct",
    "three_month_return_pct",
    "ema10",
    "ema21",
    *BOOLEAN_SIGNAL_COLUMNS,
    "macd",
    "macd_signal",
    "macd_histogram",
    "histogram_trend_text",
    "rsi_value",
    "rsi_14",
    "rsi_zone",
    "atr_like",
    "atr_like_pct",
]


def generate_latest_signals(prices: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Generate latest technical signals for each fund."""

    required_price_columns = {
        "fund_code",
        "date",
        "price",
        "latest_price",
        "daily_return_pct",
        "weekly_return_pct",
        "monthly_return_pct",
        "three_month_return_pct",
        "ema10",
        "ema21",
        "macd",
        "macd_signal",
        "macd_histogram",
        "rsi_14",
        "atr_like",
    }
    missing = required_price_columns.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required column(s): {', '.join(sorted(missing))}")

    prices = clean_price_data(prices, source_name="prices")
    rows: list[dict[str, object]] = []
    for fund_code, group in prices.sort_values(["fund_code", "date"]).groupby("fund_code", sort=False):
        latest = group.iloc[-1]
        previous = group.iloc[-2] if len(group) > 1 else latest
        recent = group.tail(config.ONE_MONTH_DAYS)

        histogram_delta = _safe_difference(latest["macd_histogram"], previous["macd_histogram"])
        ema10_above_ema21 = bool(latest["ema10"] > latest["ema21"])
        macd_above_signal = bool(latest["macd"] > latest["macd_signal"])
        histogram_positive = bool(latest["macd_histogram"] > 0)
        histogram_increasing = bool(histogram_delta > 0)

        rows.append(
            {
                "fund_code": fund_code,
                "date": latest["date"],
                "latest_price": latest["latest_price"],
                "daily_return_pct": latest["daily_return_pct"],
                "weekly_return_pct": latest["weekly_return_pct"],
                "monthly_return_pct": latest["monthly_return_pct"],
                "three_month_return_pct": latest["three_month_return_pct"],
                "ema10": latest["ema10"],
                "ema21": latest["ema21"],
                "ema10_above_ema21": ema10_above_ema21,
                "ema10_crossed_above_ema21": bool(
                    latest["ema10"] > latest["ema21"] and previous["ema10"] <= previous["ema21"]
                ),
                "macd": latest["macd"],
                "macd_signal": latest["macd_signal"],
                "macd_histogram": latest["macd_histogram"],
                "macd_above_signal": macd_above_signal,
                "histogram_positive": histogram_positive,
                "histogram_increasing": histogram_increasing,
                "histogram_worsening": bool(latest["macd_histogram"] < 0 and histogram_delta < 0),
                "histogram_trend_text": histogram_trend_text(
                    latest["macd_histogram"], histogram_delta
                ),
                "rsi_value": latest["rsi_14"],
                "rsi_14": latest["rsi_14"],
                "rsi_zone": rsi_zone(latest["rsi_14"]),
                "new_high_1m": bool(latest["price"] >= recent["price"].max()),
                "above_recent_low": bool(latest["price"] > recent["price"].min()),
                "monthly_return_positive": bool(
                    pd.notna(latest["monthly_return_pct"]) and latest["monthly_return_pct"] > 0
                ),
                "atr_like": latest["atr_like"],
                "atr_like_pct": _safe_ratio(latest["atr_like"], latest["latest_price"]),
            }
        )

    signal_data = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    if not signal_data.empty:
        signal_data[BOOLEAN_SIGNAL_COLUMNS] = signal_data[BOOLEAN_SIGNAL_COLUMNS].astype(object)
    return funds.merge(signal_data, on="fund_code", how="left")


def histogram_trend_text(histogram_value: float, histogram_delta: float) -> str:
    """Describe MACD histogram direction in a compact text label."""

    if pd.isna(histogram_value) or pd.isna(histogram_delta):
        return "Histogram unavailable"
    if abs(histogram_delta) < 1e-12:
        return "Histogram flat"

    direction = "up" if histogram_delta > 0 else "down"
    sign = "positive" if histogram_value > 0 else "negative"
    state = "increasing" if histogram_delta > 0 else "decreasing"
    return f"Histogram {direction}, {sign} {state}"


def rsi_zone(rsi_value: float) -> str:
    """Classify RSI into a readable zone."""

    if pd.isna(rsi_value):
        return "unknown"
    if rsi_value < 30:
        return "oversold"
    if rsi_value < 50:
        return "weak"
    if rsi_value <= 70:
        return "strong"
    if rsi_value <= 75:
        return "high"
    if rsi_value <= 80:
        return "overheated"
    return "extreme"


def _safe_difference(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return math.nan
    return float(left - right)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return math.nan
    return float(numerator / denominator)

