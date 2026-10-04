"""Legacy 0-10 scorer kept for old snapshots/tests.

The active pipeline uses :mod:`fund_analyzer.strategy_engine`; this module must
not be used for new investment-fund decisions.
"""

from __future__ import annotations

import pandas as pd

from fund_analyzer import config
from fund_analyzer.utils import get_flag


def score_funds(signals: pd.DataFrame) -> pd.DataFrame:
    """Score each fund from 0 to 10 using configurable weights."""

    scored = signals.copy()
    if scored.empty:
        scored["score_raw"] = pd.Series(dtype=float)
        scored["score"] = pd.Series(dtype=float)
        scored["rsi_warning"] = pd.Series(dtype=str)
        return scored

    scored["score_raw"] = scored.apply(_score_row, axis=1)
    scored["score"] = scored["score_raw"].clip(upper=config.MAX_SCORE)
    rsi_values = scored.get("rsi_value", pd.Series(pd.NA, index=scored.index))
    scored["rsi_warning"] = rsi_values.apply(_rsi_warning)
    return scored


def _score_row(row: pd.Series) -> int:
    score = 0
    weights = config.SCORING_WEIGHTS

    if get_flag(row, "ema10_above_ema21"):
        score += weights["ema10_above_ema21"]
    if get_flag(row, "ema10_crossed_above_ema21"):
        score += weights["ema10_crossed_above_ema21"]
    if get_flag(row, "macd_above_signal"):
        score += weights["macd_above_signal"]
    if get_flag(row, "histogram_positive"):
        score += weights["histogram_positive"]
    if get_flag(row, "histogram_increasing"):
        score += weights["histogram_increasing"]
    if _rsi_in_strong_range(row.get("rsi_value")):
        score += weights["rsi_50_to_70"]
    if get_flag(row, "monthly_return_positive"):
        score += weights["monthly_return_positive"]
    if get_flag(row, "new_high_1m"):
        score += weights["new_high_1m"]

    return score


def _rsi_warning(rsi_value: float) -> str:
    if pd.isna(rsi_value):
        return ""
    if rsi_value > config.RSI_EXTREME_WARNING_THRESHOLD:
        return "RSI EXTREME"
    if rsi_value > config.RSI_HIGH_WARNING_THRESHOLD:
        return "RSI HIGH"
    return ""



def _rsi_in_strong_range(rsi_value: float | None) -> bool:
    if pd.isna(rsi_value):
        return False
    return config.RSI_STRONG_MIN <= float(rsi_value) <= config.RSI_STRONG_MAX

