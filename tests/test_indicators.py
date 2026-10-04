import numpy as np
import pandas as pd

from fund_analyzer.indicators import add_indicators, calculate_atr_like, calculate_ema, calculate_macd, calculate_rsi


def test_ema_calculation_matches_pandas_ewm():
    prices = pd.Series([100, 102, 101, 105, 107], dtype=float)

    result = calculate_ema(prices, period=3)

    expected = prices.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)


def test_macd_calculation_has_expected_columns_and_relationship():
    prices = pd.Series(np.linspace(100, 120, 40), dtype=float)

    result = calculate_macd(prices)

    assert list(result.columns) == ["macd", "macd_signal", "macd_histogram"]
    pd.testing.assert_series_equal(
        result["macd_histogram"],
        result["macd"] - result["macd_signal"],
        check_names=False,
    )


def test_rsi_calculation_for_rising_prices_is_high():
    prices = pd.Series(range(1, 30), dtype=float)

    result = calculate_rsi(prices, period=14)

    assert result.iloc[-1] == 100


def test_rsi_is_nan_when_history_is_insufficient():
    prices = pd.Series([100, 101, 102, 103, 104], dtype=float)

    result = calculate_rsi(prices, period=14)

    assert result.isna().all()


def test_flat_price_series_indicators_are_stable():
    prices = pd.Series([100.0] * 30)

    macd = calculate_macd(prices)
    rsi = calculate_rsi(prices)
    atr_like = calculate_atr_like(prices)

    assert rsi.iloc[-1] == 50
    assert macd["macd"].iloc[-1] == 0
    assert macd["macd_signal"].iloc[-1] == 0
    assert macd["macd_histogram"].iloc[-1] == 0
    assert atr_like.iloc[-1] == 0


def test_falling_price_series_rsi_is_low():
    prices = pd.Series(range(30, 0, -1), dtype=float)

    result = calculate_rsi(prices, period=14)

    assert result.iloc[-1] == 0


def test_strongly_rising_price_series_rsi_is_high():
    prices = pd.Series(np.linspace(100, 180, 40), dtype=float)

    result = calculate_rsi(prices, period=14)

    assert result.iloc[-1] == 100


def test_add_indicators_sorts_dedupes_and_drops_missing_prices():
    prices = pd.DataFrame(
        [
            {"fund_code": "AAA", "date": "2026-01-03", "price": 103.0},
            {"fund_code": "AAA", "date": "2026-01-01", "price": 100.0},
            {"fund_code": "AAA", "date": "2026-01-02", "price": 101.0},
            {"fund_code": "AAA", "date": "2026-01-02", "price": 102.0},
            {"fund_code": "AAA", "date": "2026-01-04", "price": None},
        ]
    )

    result = add_indicators(prices)

    assert result["date"].tolist() == list(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    assert result["price"].tolist() == [100.0, 102.0, 103.0]

