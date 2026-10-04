"""Simple portfolio analysis with ATR-like stop levels."""

from __future__ import annotations

import math

import pandas as pd

from fund_analyzer import config
from fund_analyzer.protection import calculate_protection_decision


PORTFOLIO_OUTPUT_COLUMNS = [
    "fund_code",
    "fund_name",
    "buy_date",
    "buy_price",
    "units",
    "latest_price",
    "current_value",
    "cost_value",
    "profit_loss_amount",
    "profit_loss_pct",
    "atr_like",
    "atr_like_pct",
    "stop_loss",
    "one_atr_level",
    "two_atr_level",
    "three_atr_level",
    "four_atr_level",
    "latest_label",
    "portfolio_label",
    "protection_level",
    "distance_to_protection_pct",
    "holding_risk_score",
]


def build_portfolio_analysis(
    portfolio: pd.DataFrame,
    funds: pd.DataFrame,
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Build current portfolio values, profit/loss, and stop levels."""

    if portfolio.empty:
        return pd.DataFrame(columns=PORTFOLIO_OUTPUT_COLUMNS)

    optional_columns = [
        "category", "score", "risk_score", "annualized_volatility",
        "average_absolute_daily_movement", "current_drawdown", "recent_peak_252",
        "category_percentile", "data_quality", "momentum_score",
    ]
    latest_columns = ["fund_code", "latest_price", "atr_like", "atr_like_pct", "label"] + [
        column for column in optional_columns if column in analysis.columns
    ]
    latest = analysis[latest_columns].rename(columns={"label": "latest_label"})

    result = portfolio.merge(funds, on="fund_code", how="left").merge(latest, on="fund_code", how="left")
    result["fund_name"] = result["fund_name"].fillna("UNKNOWN FUND")
    result["latest_label"] = result["latest_label"].fillna(config.LABEL_MISSING_DATA)
    # Empty analysis tables can leave these columns as object dtype on pandas 2.
    for column in ("latest_price", "buy_price", "units", "atr_like", "atr_like_pct"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["current_value"] = result["latest_price"] * result["units"]
    result["cost_value"] = result["buy_price"] * result["units"]
    result["profit_loss_amount"] = result["current_value"] - result["cost_value"]
    result["profit_loss_pct"] = ((result["latest_price"] / result["buy_price"] - 1) * 100).round(6)

    for multiplier in range(1, 5):
        column_name = ["one", "two", "three", "four"][multiplier - 1] + "_atr_level"
        result[column_name] = result["latest_price"] - multiplier * result["atr_like"]

    result["stop_loss"] = result.apply(_stop_loss, axis=1)
    result["portfolio_label"] = result["latest_label"]
    result["protection_level"] = result["stop_loss"]
    result["distance_to_protection_pct"] = (
        (result["latest_price"] - result["protection_level"]) / result["latest_price"] * 100
    )
    result["holding_risk_score"] = pd.NA
    if "category" in result.columns:
        for index, row in result.iterrows():
            if pd.isna(row.get("category")):
                continue
            decision = calculate_protection_decision(
                category=row.get("category"),
                latest_price=row.get("latest_price"),
                average_cost=row.get("buy_price"),
                discovery_label=str(row.get("latest_label") or "BEKLE"),
                score=row.get("score"),
                risk_score=row.get("risk_score"),
                annualized_volatility=row.get("annualized_volatility"),
                average_absolute_daily_movement=row.get("average_absolute_daily_movement"),
                current_drawdown=row.get("current_drawdown"),
                recent_peak=row.get("recent_peak_252"),
                category_percentile=row.get("category_percentile"),
                data_quality=str(row.get("data_quality") or "BAD"),
                momentum_score=row.get("momentum_score"),
            )
            result.at[index, "stop_loss"] = decision.protection_level
            result.at[index, "protection_level"] = decision.protection_level
            result.at[index, "distance_to_protection_pct"] = decision.distance_to_protection_pct
            result.at[index, "portfolio_label"] = decision.user_label
            result.at[index, "holding_risk_score"] = decision.holding_risk_score

    return result[PORTFOLIO_OUTPUT_COLUMNS]


def _stop_loss(row: pd.Series) -> float:
    if row[["profit_loss_pct", "atr_like", "buy_price", "latest_price"]].isna().any():
        return math.nan

    profit_loss_pct = float(row["profit_loss_pct"])
    atr_like = float(row["atr_like"])

    if profit_loss_pct <= 0:
        return float(row["buy_price"]) - config.STOP_LOSS_LOSING_ATR_MULTIPLIER * atr_like
    if profit_loss_pct < config.STOP_LOSS_LOW_PROFIT_THRESHOLD:
        return float(row["buy_price"])
    if profit_loss_pct < config.STOP_LOSS_HIGH_PROFIT_THRESHOLD:
        return float(row["latest_price"]) - config.STOP_LOSS_MID_PROFIT_ATR_MULTIPLIER * atr_like
    return float(row["latest_price"]) - config.STOP_LOSS_HIGH_PROFIT_ATR_MULTIPLIER * atr_like

