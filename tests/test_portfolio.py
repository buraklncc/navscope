import pandas as pd

from fund_analyzer.portfolio import build_portfolio_analysis


def test_portfolio_profit_loss_calculation():
    portfolio = pd.DataFrame(
        [{"fund_code": "AAA", "buy_date": "2026-01-01", "buy_price": 100.0, "units": 10}]
    )
    funds = pd.DataFrame([{"fund_code": "AAA", "fund_name": "Test Fund", "fund_type": "Equity"}])
    analysis = pd.DataFrame(
        [
            {
                "fund_code": "AAA",
                "latest_price": 110.0,
                "atr_like": 2.0,
                "atr_like_pct": 2.0 / 110.0,
                "label": "HOLD",
            }
        ]
    )

    result = build_portfolio_analysis(portfolio, funds, analysis)

    assert result.loc[0, "current_value"] == 1100.0
    assert result.loc[0, "cost_value"] == 1000.0
    assert result.loc[0, "profit_loss_amount"] == 100.0
    assert result.loc[0, "profit_loss_pct"] == 10.0


def test_stop_loss_calculation_by_profit_bucket():
    portfolio = pd.DataFrame(
        [
            {"fund_code": "NEG", "buy_date": "2026-01-01", "buy_price": 100.0, "units": 1},
            {"fund_code": "LOW", "buy_date": "2026-01-01", "buy_price": 100.0, "units": 1},
            {"fund_code": "MID", "buy_date": "2026-01-01", "buy_price": 100.0, "units": 1},
            {"fund_code": "HIGH", "buy_date": "2026-01-01", "buy_price": 100.0, "units": 1},
        ]
    )
    funds = pd.DataFrame(
        [
            {"fund_code": "NEG", "fund_name": "Neg", "fund_type": "Equity"},
            {"fund_code": "LOW", "fund_name": "Low", "fund_type": "Equity"},
            {"fund_code": "MID", "fund_name": "Mid", "fund_type": "Equity"},
            {"fund_code": "HIGH", "fund_name": "High", "fund_type": "Equity"},
        ]
    )
    analysis = pd.DataFrame(
        [
            {"fund_code": "NEG", "latest_price": 99.0, "atr_like": 2.0, "atr_like_pct": 0.02, "label": "WAIT"},
            {"fund_code": "LOW", "latest_price": 102.0, "atr_like": 2.0, "atr_like_pct": 0.02, "label": "HOLD"},
            {"fund_code": "MID", "latest_price": 104.0, "atr_like": 2.0, "atr_like_pct": 0.02, "label": "HOLD"},
            {"fund_code": "HIGH", "latest_price": 108.0, "atr_like": 2.0, "atr_like_pct": 0.02, "label": "HOLD"},
        ]
    )

    result = build_portfolio_analysis(portfolio, funds, analysis).set_index("fund_code")

    assert result.loc["NEG", "stop_loss"] == 97.0
    assert result.loc["LOW", "stop_loss"] == 100.0
    assert result.loc["MID", "stop_loss"] == 100.0
    assert result.loc["HIGH", "stop_loss"] == 105.0


def test_missing_portfolio_fund_code_is_kept_with_missing_data_label():
    portfolio = pd.DataFrame(
        [{"fund_code": "ZZZ", "buy_date": "2026-01-01", "buy_price": 100.0, "units": 2}]
    )
    funds = pd.DataFrame([{"fund_code": "AAA", "fund_name": "Known", "fund_type": "Equity"}])
    analysis = pd.DataFrame(
        columns=["fund_code", "latest_price", "atr_like", "atr_like_pct", "label"]
    )

    result = build_portfolio_analysis(portfolio, funds, analysis)

    assert result.loc[0, "fund_code"] == "ZZZ"
    assert result.loc[0, "fund_name"] == "UNKNOWN FUND"
    assert result.loc[0, "latest_label"] == "MISSING DATA"
    assert pd.isna(result.loc[0, "current_value"])
    assert pd.isna(result.loc[0, "stop_loss"])

