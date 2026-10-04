import numpy as np
import pandas as pd

from fund_analyzer.protection import calculate_protection_decision
from fund_analyzer.strategy_engine import analyze_funds


def _fund(code: str, name: str, fund_type: str) -> dict[str, str]:
    return {"fund_code": code, "fund_name": name, "fund_type": fund_type}


def _history(code: str, dates: pd.DatetimeIndex, returns: np.ndarray) -> list[dict[str, object]]:
    prices = 100 * np.cumprod(1 + returns)
    return [
        {"fund_code": code, "date": current_date, "price": float(price)}
        for current_date, price in zip(dates, prices)
    ]


def test_money_market_does_not_receive_buy_from_positive_technical_indicators_alone():
    dates = pd.bdate_range("2026-01-02", periods=35)
    prices = pd.DataFrame(_history("PPF", dates, np.full(len(dates), 0.002)))
    funds = pd.DataFrame([_fund("PPF", "Örnek Para Piyasası Fonu", "Para Piyasası Şemsiye Fonu")])

    row = analyze_funds(prices, funds, evaluation_date=dates[-1]).iloc[0]

    assert row["ema10_above_ema21"]
    assert row["macd_above_signal"]
    assert row["label"] not in {"AL", "KADEMELİ AL"}
    assert "INSUFFICIENT_HISTORY" in row["failed_safety_gates"]


def test_equity_buy_requires_momentum_category_rank_and_acceptable_drawdown():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-02", periods=320)
    parameters = {
        "TOP": (0.0020, 0.004),
        "MID": (0.0008, 0.006),
        "LOW": (0.0002, 0.008),
        "BAD": (-0.0003, 0.012),
    }
    prices: list[dict[str, object]] = []
    funds = []
    for code, (mean, volatility) in parameters.items():
        funds.append(_fund(code, f"{code} Türkiye Hisse Senedi Fonu", "Hisse Senedi Şemsiye Fonu"))
        returns = np.clip(rng.normal(mean, volatility, len(dates)), -0.08, 0.08)
        prices.extend(_history(code, dates, returns))

    analysis = analyze_funds(pd.DataFrame(prices), pd.DataFrame(funds), evaluation_date=dates[-1]).set_index("fund_code")

    assert analysis.loc["TOP", "label"] == "AL"
    assert analysis.loc["TOP", "momentum_score"] >= 70
    assert analysis.loc["TOP", "category_percentile"] >= 75
    assert analysis.loc["TOP", "drawdown_score"] >= 60
    assert analysis.loc["BAD", "label"] == "UZAK DUR"
    assert "VERY_POOR_CATEGORY_RANK" in analysis.loc["BAD", "failed_safety_gates"]


def test_missing_or_stale_nav_cannot_receive_confident_buy():
    dates = pd.bdate_range("2025-01-02", periods=260)
    complete = _history("STALE", dates, np.full(len(dates), 0.001))
    sparse = _history("MISS", dates[::2], np.full(len(dates[::2]), 0.001))
    funds = pd.DataFrame(
        [
            _fund("STALE", "Stale Türkiye Hisse Senedi Fonu", "Hisse Senedi Şemsiye Fonu"),
            _fund("MISS", "Eksik Türkiye Hisse Senedi Fonu", "Hisse Senedi Şemsiye Fonu"),
        ]
    )

    analysis = analyze_funds(
        pd.DataFrame([*complete, *sparse]),
        funds,
        evaluation_date=dates[-1] + pd.offsets.BDay(6),
    ).set_index("fund_code")

    assert analysis.loc["STALE", "label"] == "VERİ GÜVENSİZ"
    assert "STALE_NAV" in analysis.loc["STALE", "failed_safety_gates"]
    assert analysis.loc["MISS", "label"] == "VERİ GÜVENSİZ"
    assert "HIGH_MISSING_RATIO" in analysis.loc["MISS", "failed_safety_gates"]


def test_signal_snapshot_is_deterministic_for_same_data_and_decision_date():
    dates = pd.bdate_range("2025-01-02", periods=260)
    prices = pd.DataFrame(_history("DET", dates, np.full(len(dates), 0.0008)))
    funds = pd.DataFrame([_fund("DET", "Det Türkiye Hisse Senedi Fonu", "Hisse Senedi Şemsiye Fonu")])

    first = analyze_funds(prices, funds, evaluation_date=dates[-1]).iloc[0]
    second = analyze_funds(prices, funds, evaluation_date=dates[-1]).iloc[0]

    assert first["signal_fingerprint"] == second["signal_fingerprint"]
    assert first["label"] == second["label"]
    assert first["score"] == second["score"]


def test_high_rsi_or_one_month_high_never_forces_a_decision():
    dates = pd.bdate_range("2025-01-02", periods=90)
    prices = pd.DataFrame(_history("HOT", dates, np.full(len(dates), 0.004)))
    funds = pd.DataFrame([_fund("HOT", "Hot Altın Fonu", "Altın Şemsiye Fonu")])

    row = analyze_funds(prices, funds, evaluation_date=dates[-1]).iloc[0]

    assert row["rsi_14"] > 80
    assert row["new_high_1m"]
    assert row["label"] not in {"AL", "UZAK DUR"}


def test_unknown_category_reduces_confidence_and_blocks_buy():
    dates = pd.bdate_range("2025-01-02", periods=300)
    returns = np.random.default_rng(11).normal(0.0005, 0.012, len(dates))
    prices = pd.DataFrame(_history("UNK", dates, returns))
    funds = pd.DataFrame([_fund("UNK", "Belirsiz Strateji", "Unknown")])

    row = analyze_funds(prices, funds, evaluation_date=dates[-1]).iloc[0]

    assert row["category"] == "OTHER_UNKNOWN"
    assert row["category_confidence"] < 0.55
    assert row["label"] not in {"AL", "KADEMELİ AL"}
    assert "UNKNOWN_CATEGORY" in row["failed_safety_gates"]


def test_protection_logic_differs_by_category_and_owned_fund_can_be_sell():
    money_market = calculate_protection_decision(
        category="MONEY_MARKET",
        latest_price=110,
        average_cost=100,
        discovery_label="BEKLE",
        risk_score=10,
        category_percentile=70,
        data_quality="GOOD",
    )
    equity = calculate_protection_decision(
        category="EQUITY_TURKEY",
        latest_price=100,
        average_cost=80,
        discovery_label="UZAK DUR",
        score=25,
        risk_score=80,
        annualized_volatility=0.35,
        average_absolute_daily_movement=0.02,
        current_drawdown=-0.30,
        recent_peak=103,
        category_percentile=5,
        data_quality="GOOD",
        momentum_score=20,
    )
    breached = calculate_protection_decision(
        category="EQUITY_TURKEY",
        latest_price=100,
        average_cost=80,
        discovery_label="UZAK DUR",
        score=20,
        risk_score=90,
        annualized_volatility=0.30,
        current_drawdown=-0.30,
        recent_peak=140,
        category_percentile=5,
        data_quality="GOOD",
        momentum_score=10,
    )

    assert money_market.protection_level is None
    assert money_market.user_label == "TUT"
    assert equity.protection_level is not None
    assert equity.user_label == "SAT"
    assert equity.user_label != "UZAK DUR"
    assert breached.user_label == "STOP KIRILDI"

