import pandas as pd

from fund_analyzer import config
from fund_analyzer.scoring import score_funds


def test_score_is_capped_at_10(monkeypatch):
    monkeypatch.setattr(config, "SCORING_WEIGHTS", {key: 5 for key in config.SCORING_WEIGHTS})
    signals = pd.DataFrame(
        [
            {
                "ema10_above_ema21": True,
                "ema10_crossed_above_ema21": True,
                "macd_above_signal": True,
                "histogram_positive": True,
                "histogram_increasing": True,
                "rsi_value": 60,
                "monthly_return_positive": True,
                "new_high_1m": True,
            }
        ]
    )

    scored = score_funds(signals)

    assert scored.loc[0, "score_raw"] > 10
    assert scored.loc[0, "score"] == 10


def test_score_handles_missing_boolean_values_as_false():
    signals = pd.DataFrame(
        [
            {
                "ema10_above_ema21": pd.NA,
                "ema10_crossed_above_ema21": pd.NA,
                "macd_above_signal": pd.NA,
                "histogram_positive": pd.NA,
                "histogram_increasing": pd.NA,
                "rsi_value": pd.NA,
                "monthly_return_positive": pd.NA,
                "new_high_1m": pd.NA,
            }
        ]
    )

    scored = score_funds(signals)

    assert scored.loc[0, "score"] == 0


def test_rsi_high_warning_behavior():
    signals = pd.DataFrame(
        [
            {
                "ema10_above_ema21": False,
                "ema10_crossed_above_ema21": False,
                "macd_above_signal": False,
                "histogram_positive": False,
                "histogram_increasing": False,
                "rsi_value": 76,
                "monthly_return_positive": False,
                "new_high_1m": False,
            },
            {
                "ema10_above_ema21": False,
                "ema10_crossed_above_ema21": False,
                "macd_above_signal": False,
                "histogram_positive": False,
                "histogram_increasing": False,
                "rsi_value": 82,
                "monthly_return_positive": False,
                "new_high_1m": False,
            },
        ]
    )

    scored = score_funds(signals)

    assert scored.loc[0, "rsi_warning"] == "RSI HIGH"
    assert scored.loc[1, "rsi_warning"] == "RSI EXTREME"

