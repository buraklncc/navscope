import pandas as pd

from fund_analyzer.labels import assign_label


def test_rsi_high_trend_prefers_hold_rsi_high_over_buy():
    row = pd.Series(
        {
            "score": 10,
            "ema10_above_ema21": True,
            "macd_above_signal": True,
            "histogram_positive": True,
            "histogram_increasing": True,
            "histogram_worsening": False,
            "ema10_crossed_above_ema21": False,
            "rsi_value": 81,
        }
    )

    assert assign_label(row) == "HOLD RSI HIGH"


def test_sell_label_behavior():
    row = pd.Series(
        {
            "score": 0,
            "ema10_above_ema21": False,
            "macd_above_signal": False,
            "histogram_positive": False,
            "histogram_increasing": False,
            "histogram_worsening": True,
            "ema10_crossed_above_ema21": False,
            "rsi_value": 42,
        }
    )

    assert assign_label(row) == "SELL"

