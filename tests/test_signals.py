import pandas as pd

from fund_analyzer.indicators import add_indicators
from fund_analyzer.returns import add_returns
from fund_analyzer.signals import generate_latest_signals


def test_signal_generation_latest_row_fields():
    dates = pd.bdate_range("2026-01-01", periods=35)
    prices = pd.DataFrame(
        {
            "fund_code": ["AAA"] * len(dates),
            "date": dates,
            "price": [100 + i for i in range(len(dates))],
        }
    )
    funds = pd.DataFrame(
        [{"fund_code": "AAA", "fund_name": "Test Fund", "fund_type": "Equity"}]
    )

    enriched = add_returns(add_indicators(prices))
    signals = generate_latest_signals(enriched, funds)

    row = signals.iloc[0]
    assert row["fund_code"] == "AAA"
    assert row["ema10_above_ema21"] is True
    assert row["macd_above_signal"] is True
    assert row["new_high_1m"] is True
    assert row["monthly_return_positive"] is True
    assert row["rsi_zone"] in {"strong", "high", "overheated", "extreme"}


def test_signal_generation_handles_insufficient_history():
    prices = pd.DataFrame(
        {
            "fund_code": ["AAA"],
            "date": pd.to_datetime(["2026-01-01"]),
            "price": [100.0],
        }
    )
    funds = pd.DataFrame(
        [{"fund_code": "AAA", "fund_name": "Test Fund", "fund_type": "Equity"}]
    )

    enriched = add_returns(add_indicators(prices))
    signals = generate_latest_signals(enriched, funds)
    row = signals.iloc[0]

    assert row["rsi_zone"] == "unknown"
    assert pd.isna(row["monthly_return_pct"])
    assert row["monthly_return_positive"] is False

