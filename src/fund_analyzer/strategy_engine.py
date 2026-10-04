"""Conservative category-aware strategy engine.

The engine separates eligibility/safety gates from the category-specific
0-100 score.  Technical indicators remain visible for context but cannot, on
their own, produce an AL or SAT decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from fund_analyzer.categories import FundCategory, classify_funds, coerce_category
from fund_analyzer.data_quality import DataQualityAssessment, assess_price_quality
from fund_analyzer.metrics import (
    add_category_rankings,
    calculate_benchmark_metrics,
    calculate_fund_stats_trends,
    calculate_fund_metrics,
    calculate_portfolio_correlation,
    metric_percent,
)
from fund_analyzer.strategy_config import MODEL_VERSION, STRATEGY_VERSION, policy_for


BLOCKING_BUY_GATES = {
    "MISSING_DATA",
    "INVALID_NAV",
    "DUPLICATE_NAV",
    "HIGH_MISSING_RATIO",
    "STALE_NAV",
    "OUTLIER_NAV",
    "FUTURE_DATED_NAV",
    "INSUFFICIENT_HISTORY",
    "NEW_FUND_LIMITED_HISTORY",
    "UNKNOWN_CATEGORY",
    "LOW_CATEGORY_CONFIDENCE",
    "CATEGORY_CONFIDENCE_WARNING",
    "EXTREME_DRAWDOWN",
    "EXCESSIVE_VOLATILITY",
    "VERY_POOR_CATEGORY_RANK",
    "VERY_SMALL_FUND",
    "OVERHEATED_ENTRY",
    "ADVERSE_REGIME",
}


def analyze_funds(
    prices: pd.DataFrame,
    funds: pd.DataFrame,
    *,
    evaluation_date: str | pd.Timestamp | None = None,
    benchmark_histories: dict[str, pd.DataFrame] | None = None,
    portfolio_returns: pd.Series | None = None,
    fund_stats: pd.DataFrame | None = None,
    regime_by_category: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return one auditable, UI-ready latest strategy row per fund."""

    if "fund_code" not in funds:
        raise ValueError("funds is missing required column: fund_code")
    if not prices.empty and not {"fund_code", "date", "price"}.issubset(prices.columns):
        raise ValueError("prices must contain fund_code, date, and price")

    normalized_funds = funds.copy()
    normalized_funds["fund_code"] = normalized_funds["fund_code"].astype("string").str.strip().str.upper()
    raw_prices = prices.copy()
    if not raw_prices.empty:
        raw_prices["fund_code"] = raw_prices["fund_code"].astype("string").str.strip().str.upper()

    eval_date = _evaluation_timestamp(evaluation_date)
    classified = classify_funds(normalized_funds, raw_prices)
    price_groups = {
        code: group.copy() for code, group in raw_prices.groupby("fund_code", sort=False)
    } if not raw_prices.empty else {}
    stats_groups = {
        str(code).upper(): group.copy()
        for code, group in fund_stats.groupby("fund_code", sort=False)
    } if fund_stats is not None and not fund_stats.empty and "fund_code" in fund_stats else {}

    rows: list[dict[str, Any]] = []
    for _, fund in classified.iterrows():
        code = str(fund["fund_code"])
        history = price_groups.get(code, pd.DataFrame(columns=["fund_code", "date", "price"]))
        category = coerce_category(fund.get("category"))
        category_confidence = _number(fund.get("category_confidence"), 0.0)
        quality = assess_price_quality(
            history,
            category,
            category_confidence,
            evaluation_date=eval_date,
        )
        point_in_time_history = _point_in_time_history(history, eval_date)
        metrics = calculate_fund_metrics(point_in_time_history, as_of=eval_date)
        if code in stats_groups:
            metrics.update(calculate_fund_stats_trends(stats_groups[code], as_of=eval_date))
        benchmark_key = _metadata_text(fund.get("benchmark")) or category.value
        benchmark_history = (benchmark_histories or {}).get(benchmark_key)
        if benchmark_history is not None:
            metrics.update(
                calculate_benchmark_metrics(
                    point_in_time_history,
                    benchmark_history,
                    as_of=eval_date,
                )
            )
        else:
            metrics.update(
                {
                    "beta_to_benchmark": math.nan,
                    "benchmark_correlation": math.nan,
                    "tracking_difference": math.nan,
                    "tracking_error": math.nan,
                    "information_ratio": math.nan,
                    "benchmark_overlap_days": 0.0,
                }
            )
        metrics["portfolio_correlation"] = (
            calculate_portfolio_correlation(
                point_in_time_history,
                portfolio_returns,
                as_of=eval_date,
            )
            if portfolio_returns is not None else math.nan
        )
        row = {**fund.to_dict(), **metrics}
        row["market_regime"] = str(
            (regime_by_category or {}).get(category.value, "UNAVAILABLE")
        ).upper()
        row.update(_base_scores(row, category, quality))
        row.update(
            {
                "data_quality_score": quality.score,
                "data_quality": quality.status,
                "quality_flags": [dict(flag) for flag in quality.flags],
                "warnings": [*list(fund.get("category_warnings") or []), *quality.warnings],
                "failed_safety_gates": list(quality.failed_safety_gates),
                "missing_data_ratio": quality.missing_ratio,
                "stale_nav_count": quality.stale_nav_count,
                "freshness_lag_business_days": quality.freshness_lag_business_days,
                "outlier_count": quality.outlier_count,
                "calculated_at": eval_date.isoformat(),
                "strategy_version": STRATEGY_VERSION,
                "model_version": MODEL_VERSION,
            }
        )
        rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        return features
    ranked = add_category_rankings(features)
    finalized = ranked.apply(_finalize_row, axis=1, result_type="expand")
    for column in finalized.columns:
        ranked[column] = finalized[column]
    return _legacy_compatible_columns(ranked).reset_index(drop=True)


def to_ui_contract(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Build the stable camelCase contract consumed by the UI/API."""

    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return {
        "fundCode": source.get("fund_code"),
        "fundName": source.get("fund_name"),
        "category": source.get("category"),
        "categoryConfidence": _round_or_none(source.get("category_confidence"), 4),
        "userLabel": source.get("label") or "VERİ EKSİK",
        "labelStrength": _round_or_none(source.get("label_strength"), 2),
        "confidence": _round_or_none(source.get("confidence"), 2),
        "dataQuality": source.get("data_quality") or "BAD",
        "riskLevel": source.get("risk_level") or "EXTREME",
        "score": _round_or_none(source.get("score"), 2),
        "momentumScore": _round_or_none(source.get("momentum_score"), 2),
        "riskScore": _round_or_none(source.get("risk_score"), 2),
        "consistencyScore": _round_or_none(source.get("consistency_score"), 2),
        "drawdownScore": _round_or_none(source.get("drawdown_score"), 2),
        "categoryRank": _integer_or_none(source.get("category_rank_risk_adjusted")),
        "categoryPercentile": _round_or_none(source.get("category_percentile"), 2),
        "currentDrawdown": _percent_or_none(source.get("current_drawdown")),
        "maxDrawdown": _percent_or_none(source.get("max_drawdown")),
        "volatility": _percent_or_none(source.get("annualized_volatility")),
        "sharpe": _round_or_none(source.get("sharpe"), 4),
        "sortino": _round_or_none(source.get("sortino"), 4),
        "protectionLevel": _round_or_none(source.get("protection_level"), 6),
        "distanceToProtectionPct": _round_or_none(source.get("distance_to_protection_pct"), 4),
        "shortExplanation": source.get("short_explanation") or "Sağlıklı karar için yeterli veri yok.",
        "technicalExplanation": list(source.get("technical_explanation") or []),
        "warnings": list(source.get("warnings") or []),
        "failedSafetyGates": list(source.get("failed_safety_gates") or []),
        "strategyVersion": source.get("strategy_version") or STRATEGY_VERSION,
        "calculatedAt": source.get("calculated_at"),
    }


def signal_fingerprint(row: dict[str, Any] | pd.Series) -> str:
    """Hash the decision inputs/outputs for deterministic snapshot auditing."""

    contract = to_ui_contract(row)
    contract.pop("calculatedAt", None)
    payload = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _base_scores(
    row: dict[str, Any],
    category: FundCategory,
    quality: DataQualityAssessment,
) -> dict[str, float]:
    policy = policy_for(category)
    momentum_score = _momentum_score(row, category)
    consistency_score = _consistency_score(row, category)
    drawdown_score = _drawdown_score(row, policy.extreme_drawdown)
    risk_score = _risk_score(row, policy.maximum_annual_volatility, policy.extreme_drawdown)
    trend_score = _trend_score(row, category)
    friction_score = _friction_score(row)
    reliability_score = max(0.0, min(100.0, quality.score * 0.80 + friction_score * 0.20))
    return {
        "momentum_score": round(momentum_score, 2),
        "consistency_score": round(consistency_score, 2),
        "drawdown_score": round(drawdown_score, 2),
        "risk_score": round(risk_score, 2),
        "trend_score": round(trend_score, 2),
        "friction_score": round(friction_score, 2),
        "reliability_score": round(reliability_score, 2),
    }


def _finalize_row(row: pd.Series) -> dict[str, Any]:
    category = coerce_category(row.get("category"))
    policy = policy_for(category)
    gates = list(row.get("failed_safety_gates") or [])
    warnings = list(row.get("warnings") or [])

    volatility = _number(row.get("annualized_volatility"), math.nan)
    max_drawdown = _number(row.get("max_drawdown"), math.nan)
    category_percentile = _number(row.get("category_percentile"), 50.0)
    category_sample = int(_number(row.get("category_sample_size"), 0))
    if _finite(volatility) and volatility > policy.maximum_annual_volatility:
        gates.append("EXCESSIVE_VOLATILITY")
        warnings.append("Oynaklık fon kategorisi için kabul edilen sınırın üzerinde.")
    if _finite(max_drawdown) and max_drawdown < policy.extreme_drawdown:
        gates.append("EXTREME_DRAWDOWN")
        warnings.append("Tarihsel azami düşüş bu kategori için aşırı yüksek.")
    if category_sample >= 4 and category_percentile <= 10:
        gates.append("VERY_POOR_CATEGORY_RANK")
        warnings.append("Fon kendi kategorisinin en zayıf diliminde.")
    if _very_small_fund(row):
        gates.append("VERY_SMALL_FUND")
        warnings.append("Fon büyüklüğü likidite/ölçek açısından düşük.")
    regime = str(row.get("market_regime") or "UNAVAILABLE").upper()
    if category in {FundCategory.LONG_TERM_DEBT, FundCategory.FX_EUROBOND}:
        if regime == "ADVERSE":
            gates.append("ADVERSE_REGIME")
            warnings.append("Faiz/döviz rejimi uzun vadeli borçlanma kategorisi için olumsuz.")
        elif regime == "UNAVAILABLE":
            warnings.append("Faiz/döviz rejimi verisi yok; karar yalnız fon ve kategori verisine dayanıyor.")
    overheated = _overheated_entry(row, category)
    if overheated:
        gates.append("OVERHEATED_ENTRY")
        warnings.append("Kısa ve orta vadeli getiri oynaklığa göre aşırı hızlandı; tek seferde yeni alım riski yüksek.")
    gates = list(dict.fromkeys(gates))
    warnings = list(dict.fromkeys(warnings))

    relative_strength_score = category_percentile
    components = {
        "momentum": _number(row.get("momentum_score"), 0.0),
        "relative_strength": relative_strength_score,
        "trend": _number(row.get("trend_score"), 50.0),
        "drawdown": _number(row.get("drawdown_score"), 0.0),
        "consistency": _number(row.get("consistency_score"), 0.0),
        "reliability": _number(row.get("reliability_score"), 0.0),
        "friction": _number(row.get("friction_score"), 50.0),
    }
    score = sum(components[name] * weight for name, weight in policy.weights.items())
    if regime == "ADVERSE" and category in {FundCategory.LONG_TERM_DEBT, FundCategory.FX_EUROBOND}:
        score -= 10
    elif regime == "FAVORABLE" and category in {FundCategory.LONG_TERM_DEBT, FundCategory.FX_EUROBOND}:
        score += 3
    score = round(max(0.0, min(100.0, score)), 2)
    confidence = _confidence(row, category_sample)
    label = _discovery_label(row, score, confidence, gates)
    short, technical = _explanations(row, label, score, category_percentile, gates)
    risk_level = _risk_level(_number(row.get("risk_score"), 100.0), gates)
    label_strength = _label_strength(label, score, _number(row.get("data_quality_score"), 0.0))
    return {
        "relative_strength_score": round(relative_strength_score, 2),
        "category_score": round(relative_strength_score, 2),
        "score": score,
        "score_raw": score,
        "label": label,
        "user_label": label,
        "label_strength": label_strength,
        "confidence": confidence,
        "risk_level": risk_level,
        "overheated_entry": overheated,
        "mean_reversion_eligible": _mean_reversion_eligible(row, category),
        "rate_sensitivity_proxy": (
            _number(row.get("category_relative_volatility"), math.nan)
            if category in {FundCategory.SHORT_TERM_DEBT, FundCategory.LONG_TERM_DEBT, FundCategory.FX_EUROBOND}
            else math.nan
        ),
        "short_explanation": short,
        "technical_explanation": technical,
        "explanation": short,
        "failed_safety_gates": gates,
        "warnings": warnings,
    }


def _discovery_label(
    row: pd.Series,
    score: float,
    confidence: float,
    gates: list[str],
) -> str:
    history_length = int(_number(row.get("history_length"), 0))
    data_quality = str(row.get("data_quality") or "BAD")
    category = coerce_category(row.get("category"))
    policy = policy_for(category)
    blocking = set(gates) & BLOCKING_BUY_GATES
    hard_blocking = blocking - {"OVERHEATED_ENTRY"}

    if history_length < max(21, min(policy.minimum_history, 42)) or "MISSING_DATA" in gates:
        return "VERİ EKSİK"
    unsafe_data_gates = {"INVALID_NAV", "DUPLICATE_NAV", "HIGH_MISSING_RATIO", "STALE_NAV", "OUTLIER_NAV", "FUTURE_DATED_NAV"}
    if data_quality == "BAD" or set(gates) & unsafe_data_gates:
        return "VERİ GÜVENSİZ"
    dangerous = {"EXTREME_DRAWDOWN", "EXCESSIVE_VOLATILITY", "VERY_POOR_CATEGORY_RANK"}
    if set(gates) & dangerous:
        return "UZAK DUR"

    percentile = _number(row.get("category_percentile"), 50.0)
    momentum = _number(row.get("momentum_score"), 0.0)
    drawdown = _number(row.get("drawdown_score"), 0.0)
    quality = _number(row.get("data_quality_score"), 0.0)
    if (
        not blocking
        and score >= 82
        and percentile >= 75
        and momentum >= 70
        and drawdown >= 60
        and quality >= 85
        and confidence >= 72
        and history_length >= policy.buy_history
    ):
        return "AL"
    if (
        not hard_blocking
        and score >= 68
        and percentile >= 60
        and momentum >= 60
        and drawdown >= 48
        and confidence >= 62
    ):
        return "KADEMELİ AL"
    if score < 35 or (_number(row.get("risk_score"), 100.0) >= 85 and score < 50):
        return "UZAK DUR"
    return "BEKLE"


def _momentum_score(row: dict[str, Any], category: FundCategory) -> float:
    r3 = _number(row.get("three_month_return"), math.nan)
    r6 = _number(row.get("six_month_return"), math.nan)
    r12 = _number(row.get("twelve_month_return"), math.nan)
    annualized = _number(row.get("annualized_return"), math.nan)
    volatility = _number(row.get("annualized_volatility"), math.nan)
    available = [(r3, 0.25), (r6, 0.35), (r12, 0.40)]
    valid = [(value, weight) for value, weight in available if _finite(value)]
    if not valid:
        return 0.0
    total_weight = sum(weight for _, weight in valid)
    weighted_return = sum(value * weight for value, weight in valid) / total_weight
    if category == FundCategory.MONEY_MARKET and _finite(annualized):
        weighted_return = annualized * 0.70 + weighted_return * 0.30
    risk_scale = max(0.025, volatility if _finite(volatility) else 0.15)
    risk_adjusted = weighted_return / risk_scale
    score = 50 + 42 * math.tanh(risk_adjusted)
    current_drawdown = abs(min(0.0, _number(row.get("current_drawdown"), 0.0)))
    score -= min(25.0, current_drawdown * 100)
    return max(0.0, min(100.0, score))


def _consistency_score(row: dict[str, Any], category: FundCategory) -> float:
    positive_days = _number(row.get("positive_day_ratio"), 0.0)
    positive_months = _number(row.get("positive_month_ratio"), 0.0)
    rolling_consistency = _number(row.get("return_consistency"), 0.0)
    negative_days = int(_number(row.get("negative_day_count"), 0))
    history = max(1, int(_number(row.get("history_length"), 1)) - 1)
    negative_ratio = negative_days / history
    score = positive_days * 30 + positive_months * 25 + rolling_consistency * 45
    if category == FundCategory.MONEY_MARKET:
        score = (1 - min(1.0, negative_ratio * 8)) * 45 + positive_months * 20 + rolling_consistency * 35
    return max(0.0, min(100.0, score))


def _drawdown_score(row: dict[str, Any], extreme_drawdown: float) -> float:
    max_drawdown = _number(row.get("max_drawdown"), math.nan)
    current_drawdown = _number(row.get("current_drawdown"), math.nan)
    if not _finite(max_drawdown):
        return 0.0
    denominator = max(abs(extreme_drawdown), 0.01)
    max_quality = 100 * (1 - min(1.0, abs(min(0.0, max_drawdown)) / denominator))
    current_penalty = min(35.0, abs(min(0.0, current_drawdown if _finite(current_drawdown) else 0.0)) / denominator * 35)
    return max(0.0, min(100.0, max_quality - current_penalty))


def _risk_score(row: dict[str, Any], maximum_volatility: float, extreme_drawdown: float) -> float:
    volatility = _number(row.get("annualized_volatility"), math.nan)
    downside = _number(row.get("downside_deviation"), math.nan)
    max_drawdown = abs(min(0.0, _number(row.get("max_drawdown"), 0.0)))
    if not _finite(volatility):
        return 100.0
    vol_component = min(100.0, volatility / max(maximum_volatility, 0.01) * 70)
    downside_component = min(20.0, (downside if _finite(downside) else volatility) / max(maximum_volatility, 0.01) * 20)
    drawdown_component = min(30.0, max_drawdown / max(abs(extreme_drawdown), 0.01) * 30)
    return max(0.0, min(100.0, vol_component + downside_component + drawdown_component))


def _trend_score(row: dict[str, Any], category: FundCategory) -> float:
    if category == FundCategory.MONEY_MARKET:
        return 50.0
    latest = _number(row.get("latest_price"), math.nan)
    ema10 = _number(row.get("ema10"), math.nan)
    ema21 = _number(row.get("ema21"), math.nan)
    macd = _number(row.get("macd"), math.nan)
    macd_signal = _number(row.get("macd_signal"), math.nan)
    r3 = _number(row.get("three_month_return"), math.nan)
    score = 50.0
    # Each technical observation is deliberately small; none is a decision rule.
    score += 10 if _finite(latest) and _finite(ema21) and latest > ema21 else -10
    score += 8 if _finite(ema10) and _finite(ema21) and ema10 > ema21 else -8
    score += 6 if _finite(macd) and _finite(macd_signal) and macd > macd_signal else -6
    score += 8 if _finite(r3) and r3 > 0 else -8
    rsi = _number(row.get("rsi_14"), math.nan)
    if _finite(rsi) and rsi > 82:
        score -= 5  # context-only overheat check; never an automatic sell.
    if category in {FundCategory.SHORT_TERM_DEBT, FundCategory.VARIABLE_MIXED, FundCategory.FUND_OF_FUNDS}:
        score = 50 + (score - 50) * 0.50
    return max(0.0, min(100.0, score))


def _friction_score(row: dict[str, Any]) -> float:
    score = 50.0
    expense = _number(row.get("expense_ratio"), math.nan)
    if _finite(expense):
        expense_fraction = expense / 100 if expense > 1 else expense
        score = max(0.0, 100 - expense_fraction * 2500)
    size = _number(row.get("fund_size"), math.nan)
    if _finite(size) and size > 0:
        size_score = max(0.0, min(100.0, (math.log10(size) - 6) / 3 * 100))
        score = score * 0.70 + size_score * 0.30
    investors = _number(row.get("investor_count"), math.nan)
    if _finite(investors) and investors > 0:
        investor_score = max(0.0, min(100.0, math.log10(investors) / 5 * 100))
        score = score * 0.85 + investor_score * 0.15
    tax = _number(row.get("tax_friction_pct"), math.nan)
    if not _finite(tax):
        tax = _number(row.get("tax_rate"), math.nan)
    if _finite(tax):
        tax_fraction = tax / 100 if tax > 1 else tax
        score -= min(30.0, max(0.0, tax_fraction) * 100)
    buy_valor = max(0.0, _number(row.get("buy_valor"), 0.0))
    sell_valor = max(0.0, _number(row.get("sell_valor"), 0.0))
    score -= min(15.0, (buy_valor + sell_valor) * 1.5)
    return max(0.0, min(100.0, score))


def _very_small_fund(row: pd.Series) -> bool:
    size = _number(row.get("fund_size"), math.nan)
    return _finite(size) and 0 < size < 1_000_000


def _overheated_entry(row: pd.Series, category: FundCategory) -> bool:
    if category not in {
        FundCategory.EQUITY_TURKEY,
        FundCategory.FOREIGN_EQUITY,
        FundCategory.GOLD_PRECIOUS_METALS,
    }:
        return False
    monthly = _number(row.get("monthly_return"), math.nan)
    three_month = _number(row.get("three_month_return"), math.nan)
    volatility = _number(row.get("annualized_volatility"), math.nan)
    current_drawdown = _number(row.get("current_drawdown"), math.nan)
    if not all(_finite(value) for value in (monthly, three_month, volatility, current_drawdown)):
        return False
    monthly_limit = max(0.08, volatility / math.sqrt(12) * 2.5)
    quarterly_limit = max(0.18, volatility * math.sqrt(0.25) * 2.2)
    return monthly > monthly_limit and three_month > quarterly_limit and current_drawdown > -0.01


def _mean_reversion_eligible(row: pd.Series, category: FundCategory) -> bool:
    if category not in {
        FundCategory.SHORT_TERM_DEBT,
        FundCategory.VARIABLE_MIXED,
        FundCategory.FUND_OF_FUNDS,
        FundCategory.PARTICIPATION_SUKUK,
    }:
        return False
    drawdown = _number(row.get("current_drawdown"), 0.0)
    consistency = _number(row.get("consistency_score"), 0.0)
    percentile = _number(row.get("category_percentile"), 0.0)
    return (
        str(row.get("data_quality") or "BAD") == "GOOD"
        and -0.10 < drawdown < -0.005
        and consistency >= 65
        and percentile >= 40
    )


def _confidence(row: pd.Series, category_sample: int) -> float:
    category_confidence = _number(row.get("category_confidence"), 0.0) * 100
    quality = _number(row.get("data_quality_score"), 0.0)
    history = _number(row.get("history_length"), 0.0)
    history_confidence = min(100.0, history / 252 * 100)
    peer_confidence = min(100.0, 40 + category_sample * 10)
    return round(max(0.0, min(100.0, category_confidence * 0.30 + quality * 0.40 + history_confidence * 0.20 + peer_confidence * 0.10)), 2)


def _risk_level(risk_score: float, gates: list[str]) -> str:
    if {"EXTREME_DRAWDOWN", "EXCESSIVE_VOLATILITY"} & set(gates) or risk_score >= 85:
        return "EXTREME"
    if risk_score >= 65:
        return "HIGH"
    if risk_score >= 35:
        return "MEDIUM"
    return "LOW"


def _label_strength(label: str, score: float, quality: float) -> float:
    if label in {"VERİ EKSİK", "VERİ GÜVENSİZ"}:
        return round(100 - quality, 2)
    if label == "UZAK DUR":
        return round(100 - score, 2)
    if label == "BEKLE":
        return round(100 - min(100.0, abs(score - 50) * 2), 2)
    return round(score, 2)


def _explanations(
    row: pd.Series,
    label: str,
    score: float,
    percentile: float,
    gates: list[str],
) -> tuple[str, list[str]]:
    category_name = str(row.get("category_name") or row.get("category") or "kategorisi")
    momentum = _number(row.get("momentum_score"), 0.0)
    drawdown = _number(row.get("drawdown_score"), 0.0)
    quality = str(row.get("data_quality") or "BAD")
    short_map = {
        "AL": f"AL: Fon {category_name} grubunda güçlü; momentum, kategori sırası ve düşüş kontrolü birlikte olumlu.",
        "KADEMELİ AL": "KADEMELİ AL: Görünüm olumlu ancak kanıt tam güçlenmedi; tek sefer yerine kademeli takip daha güvenli.",
        "BEKLE": "BEKLE: Sinyaller karışık; fon kötü değil ancak şu anda güçlü alım kanıtı yok.",
        "UZAK DUR": "UZAK DUR: Fon kendi kategorisine göre zayıf veya riski kabul edilebilir sınırın üzerinde.",
        "VERİ EKSİK": "VERİ EKSİK: Sağlıklı karar için yeterli fiyat geçmişi yok.",
        "VERİ GÜVENSİZ": "VERİ GÜVENSİZ: NAV verisinde eski, eksik, yinelenen veya olağan dışı kayıt riski var.",
    }
    reasons = [
        f"Kategori: {category_name}; sınıflandırma güveni %{_number(row.get('category_confidence'), 0.0) * 100:.0f}.",
        f"Kategori yüzdelik sırası %{percentile:.0f}; toplam skor {score:.1f}/100.",
        f"Momentum skoru {momentum:.1f}/100; düşüş kalitesi {drawdown:.1f}/100.",
        f"Veri kalitesi {quality}; kalite skoru {_number(row.get('data_quality_score'), 0.0):.1f}/100.",
    ]
    if gates:
        reasons.append("Başarısız güvenlik kapıları: " + ", ".join(gates) + ".")
    else:
        reasons.append("Zorunlu güvenlik kapılarında engel yok.")
    return short_map[label], reasons[:5]


def _legacy_compatible_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    percentage_map = {
        "daily_return": "daily_return_pct",
        "weekly_return": "weekly_return_pct",
        "monthly_return": "monthly_return_pct",
        "three_month_return": "three_month_return_pct",
        "six_month_return": "six_month_return_pct",
        "twelve_month_return": "twelve_month_return_pct",
        "ytd_return": "ytd_return_pct",
    }
    for source, target in percentage_map.items():
        output[target] = output[source].apply(metric_percent)
    output["rsi_value"] = output["rsi_14"]
    output["atr_like_pct"] = output["average_absolute_daily_movement"]
    output["atr_like"] = output["latest_price"] * output["atr_like_pct"]
    output["ema10_above_ema21"] = output["ema10"] > output["ema21"]
    output["macd_above_signal"] = output["macd"] > output["macd_signal"]
    output["macd_histogram"] = output["macd"] - output["macd_signal"]
    output["histogram_positive"] = output["macd_histogram"] > 0
    output["histogram_increasing"] = False
    output["histogram_worsening"] = False
    output["ema10_crossed_above_ema21"] = False
    output["monthly_return_positive"] = output["monthly_return"] > 0
    output["new_high_1m"] = output["latest_price"] >= output["recent_high_21"]
    output["rsi_zone"] = output["rsi_14"].apply(_rsi_zone)
    output["histogram_trend_text"] = "Yalnızca destekleyici teknik bağlam"
    output["rsi_warning"] = output["rsi_14"].apply(lambda value: "RSI bağlamı: yüksek" if _finite(value) and value > 75 else "")
    output["signal_fingerprint"] = output.apply(signal_fingerprint, axis=1)
    return output


def _point_in_time_history(history: pd.DataFrame, evaluation_date: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    output = history.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    return output.loc[output["date"] <= evaluation_date].copy()


def _evaluation_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp:
    parsed = pd.Timestamp(value) if value is not None else pd.Timestamp.now(tz="UTC")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _rsi_zone(value: Any) -> str:
    if not _finite(value):
        return "unknown"
    numeric = float(value)
    if numeric < 30:
        return "oversold"
    if numeric < 50:
        return "weak"
    if numeric <= 70:
        return "strong"
    if numeric <= 80:
        return "high"
    return "extreme"


def _number(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _round_or_none(value: Any, decimals: int) -> float | None:
    return round(float(value), decimals) if _finite(value) else None


def _percent_or_none(value: Any) -> float | None:
    return round(float(value) * 100, 6) if _finite(value) else None


def _integer_or_none(value: Any) -> int | None:
    return int(value) if _finite(value) else None

