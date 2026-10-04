"""Legacy technical labels kept for old snapshot compatibility.

New scans use the conservative Turkish labels from ``strategy_engine``.
"""

from __future__ import annotations

import pandas as pd

from fund_analyzer import config
from fund_analyzer.utils import get_flag, safe_float


def assign_labels(scored: pd.DataFrame) -> pd.DataFrame:
    """Assign final labels and short explanations."""

    labeled = scored.copy()
    if labeled.empty:
        labeled["label"] = pd.Series(dtype=str)
        labeled["explanation"] = pd.Series(dtype=str)
        return labeled

    labeled["label"] = labeled.apply(assign_label, axis=1)
    labeled["explanation"] = labeled.apply(build_explanation, axis=1)
    return labeled


def assign_label(row: pd.Series) -> str:
    score = safe_float(row.get("score"), default=0)
    ema_positive = get_flag(row, "ema10_above_ema21")
    macd_positive = get_flag(row, "macd_above_signal")
    histogram_positive = get_flag(row, "histogram_positive")
    histogram_improving = get_flag(row, "histogram_increasing")
    histogram_worsening = get_flag(row, "histogram_worsening")
    ema_crossed_up = get_flag(row, "ema10_crossed_above_ema21")
    rsi_value = row.get("rsi_value", 0)

    if (
        not ema_positive
        and not macd_positive
        and not histogram_positive
        and histogram_worsening
    ):
        return config.LABEL_SELL

    trend_positive = ema_positive and macd_positive and (histogram_positive or histogram_improving)
    if pd.notna(rsi_value) and float(rsi_value) >= config.RSI_HIGH_LABEL_THRESHOLD and trend_positive:
        return config.LABEL_HOLD_RSI_HIGH

    if score >= config.STRONG_BUY_MIN_SCORE and ema_positive and macd_positive and histogram_positive:
        return config.LABEL_STRONG_BUY

    if score >= config.EARLY_BUY_MIN_SCORE and (ema_crossed_up or histogram_improving):
        return config.LABEL_EARLY_BUY

    if score >= config.HOLD_MIN_SCORE and (ema_positive or macd_positive or histogram_positive):
        return config.LABEL_HOLD

    return config.LABEL_WAIT


def build_explanation(row: pd.Series) -> str:
    """Build a concise human-readable reason for the latest label."""

    parts: list[str] = []
    if get_flag(row, "ema10_above_ema21"):
        parts.append("Trend olumlu")
        parts.append("EMA10, EMA21'in üzerinde")
    else:
        parts.append("Trend zayıf")
        parts.append("EMA10, EMA21'in altında")

    if get_flag(row, "ema10_crossed_above_ema21"):
        parts.append("EMA yukarı kesti")

    if get_flag(row, "macd_above_signal"):
        parts.append("MACD sinyal çizgisinin üzerinde")
    else:
        parts.append("MACD sinyal çizgisinin altında")

    if get_flag(row, "histogram_increasing"):
        parts.append("Histogram iyileşiyor")
    elif get_flag(row, "histogram_worsening"):
        parts.append("Histogram zayıflıyor")
    else:
        parts.append("Histogram yatay veya karışık")

    rsi_value = row.get("rsi_value")
    if pd.isna(rsi_value):
        parts.append("RSI verisi yok")
    elif rsi_value >= config.RSI_HIGH_LABEL_THRESHOLD:
        parts.append("RSI yüksek; aşırı alım bölgesinde")
    elif config.RSI_STRONG_MIN <= rsi_value <= config.RSI_STRONG_MAX:
        parts.append("RSI güçlü ancak aşırı ısınmamış")
    elif rsi_value < 30:
        parts.append("RSI aşırı satım bölgesinde")
    else:
        parts.append("RSI nötr veya zayıf")

    if get_flag(row, "monthly_return_positive"):
        parts.append("Aylık getiri pozitif")

    return ". ".join(parts) + "."


def get_flag(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    if pd.isna(value):
        return False
    return bool(value)


def safe_float(value: float | None, default: float) -> float:
    if pd.isna(value):
        return default
    return float(value)

