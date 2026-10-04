"""Category-aware mutual-fund protection lines and holding decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from fund_analyzer.categories import FundCategory, coerce_category
from fund_analyzer.strategy_config import policy_for


@dataclass(frozen=True)
class ProtectionDecision:
    user_label: str
    protection_level: float | None
    distance_to_protection_pct: float | None
    profit_pct: float | None
    holding_risk_score: float
    explanation: str
    warnings: tuple[str, ...] = ()
    concentration_risk: float = 0.0
    correlation_risk: float = 0.0


def calculate_protection_decision(
    *,
    category: FundCategory | str,
    latest_price: Any,
    average_cost: Any,
    discovery_label: str,
    score: Any = None,
    risk_score: Any = None,
    annualized_volatility: Any = None,
    average_absolute_daily_movement: Any = None,
    current_drawdown: Any = None,
    recent_peak: Any = None,
    category_percentile: Any = None,
    data_quality: str = "GOOD",
    momentum_score: Any = None,
    near_threshold_pct: float = 2.0,
    concentration_pct: Any = None,
    portfolio_correlation: Any = None,
    better_alternative_score_gap: Any = None,
    tax_friction_pct: Any = None,
) -> ProtectionDecision:
    """Create a review/protection line, never an executable stop order."""

    resolved = coerce_category(category)
    latest = _number(latest_price)
    cost = _number(average_cost)
    if latest is None or cost is None or latest <= 0 or cost <= 0:
        return ProtectionDecision(
            "VERİ EKSİK", None, None, None, 100.0,
            "Karar için güncel NAV ve maliyet bilgisi birlikte gerekli.",
        )
    profit_pct = (latest / cost - 1) * 100
    risk = _bounded(_number(risk_score, 100.0))
    if data_quality == "BAD":
        return ProtectionDecision(
            "VERİ GÜVENSİZ", None, None, round(profit_pct, 4), max(85.0, risk),
            "NAV verisi güvenilir olmadığı için koruma çizgisi ve satış kararı üretilmedi.",
        )

    percentile = _number(category_percentile, 50.0)
    momentum = _number(momentum_score, 50.0)
    drawdown = min(0.0, _number(current_drawdown, 0.0))
    concentration = max(0.0, _number(concentration_pct, 0.0))
    correlation = abs(_number(portfolio_correlation, 0.0))
    alternative_gap = max(0.0, _number(better_alternative_score_gap, 0.0))
    tax_friction = max(0.0, _number(tax_friction_pct, 0.0))
    concentration_risk = _bounded(max(0.0, concentration - 20) * 3)
    correlation_risk = _bounded(max(0.0, correlation - 0.65) / 0.35 * 100)

    if resolved == FundCategory.MONEY_MARKET:
        deteriorated = discovery_label == "UZAK DUR" or percentile <= 15 or drawdown < -0.004
        label = "ACİL SAT DEĞİL AMA RİSK ARTTI" if deteriorated else "TUT"
        explanation = (
            "Para piyasası fonunda klasik stop kullanılmaz; olağan dışı negatif getiri veya kategori gerilemesi izlenir."
            if deteriorated
            else "Para piyasası fonunda klasik stop yok; getiri tutarlılığı ve kategori sırası kabul edilebilir."
        )
        holding_risk = _bounded(risk * 0.65 + concentration_risk * 0.20 + correlation_risk * 0.15)
        return ProtectionDecision(
            label, None, None, round(profit_pct, 4), holding_risk, explanation,
            (), concentration_risk, correlation_risk,
        )

    policy = policy_for(resolved)
    annual_vol = max(0.0, _number(annualized_volatility, 0.0))
    daily_movement = max(0.0, _number(average_absolute_daily_movement, 0.0))
    realized_daily_vol = annual_vol / math.sqrt(252) if annual_vol else daily_movement
    gap = max(
        policy.protection_floor_pct,
        realized_daily_vol * policy.protection_volatility_multiplier,
        daily_movement * policy.protection_volatility_multiplier,
    )
    # Large profits may tighten the review line moderately, but never to the
    # tiny stock-like distances that would react to normal mutual-fund NAV noise.
    if profit_pct >= 12:
        gap = max(policy.protection_floor_pct * 0.85, gap * 0.88)
    peak = _number(recent_peak)
    if peak is None or peak < latest:
        peak = latest / max(1e-9, 1 + drawdown) if drawdown < 0 else latest
    trailing_line = peak * (1 - gap)
    if profit_pct > 3:
        retained_profit_fraction = 0.20 if resolved in {FundCategory.EQUITY_TURKEY, FundCategory.FOREIGN_EQUITY, FundCategory.GOLD_PRECIOUS_METALS} else 0.12
        cost_reference = cost * (1 + max(0.0, profit_pct / 100) * retained_profit_fraction)
        protection = max(trailing_line, cost_reference)
    else:
        protection = trailing_line
    protection = round(max(0.0, protection), 6)
    distance = (latest - protection) / latest * 100
    holding_risk = _bounded(
        risk * 0.42
        + abs(drawdown) / max(abs(policy.extreme_drawdown), 0.01) * 28
        + max(0, 50 - percentile) * 0.16
        + concentration_risk * 0.08
        + correlation_risk * 0.05
        + min(8.0, alternative_gap * 0.20)
        + min(5.0, tax_friction * 0.25)
    )

    trend_deteriorated = discovery_label == "UZAK DUR" or (_number(score, 50.0) < 38 and momentum < 42)
    portfolio_risk_elevated = concentration_risk >= 60 or correlation_risk >= 70 or alternative_gap >= 25
    materially_broken = latest <= protection and trend_deteriorated
    if materially_broken:
        label = "STOP KIRILDI"
        explanation = "Koruma çizgisi kırıldı ve fonun kategori/risk görünümü de bozuldu; portföy için SAT kararı değerlendirilmelidir."
    elif latest <= protection:
        label = "ACİL SAT DEĞİL AMA RİSK ARTTI"
        explanation = "Koruma çizgisi aşıldı ancak tek başına mekanik satış yeterli değil; ilk uygun NAV/valör koşulunda yeniden değerlendir."
    elif distance <= max(0.5, near_threshold_pct):
        label = "STOP'A YAKIN"
        explanation = "NAV kategoriye göre hesaplanan koruma çizgisine yaklaştı; yeni alım yerine riski izle."
    elif trend_deteriorated and (drawdown < policy.extreme_drawdown * 0.55 or percentile <= 15):
        label = "SAT"
        explanation = "Fonun trendi, düşüşü ve kategori sırası birlikte belirgin biçimde bozuldu."
    elif trend_deteriorated or portfolio_risk_elevated:
        label = "UZAK DUR / YENİ ALIM YAPMA"
        explanation = "Mevcut pozisyon acil satış gerektirmiyor ancak fon veya portföy yoğunlaşma/korelasyon riski arttığı için yeni alım uygun değil."
    elif profit_pct >= 6 and (momentum < 50 or drawdown < -gap * 0.45):
        label = "KÂRI KORU"
        explanation = "Pozisyon kârda ancak momentum zayıflıyor veya zirveden düşüş büyüyor; yeni alım yerine kârı koru."
    elif discovery_label in {"AL", "KADEMELİ AL"} and holding_risk < 55:
        label = "ALIM ARTIRILABİLİR"
        explanation = "Fon güvenlik kapılarını geçti ve portföy riski kabul edilebilir; yoğunlaşma uygunsa alım artırılabilir."
    else:
        label = "TUT"
        explanation = "Trend ve kategoriye göre risk görünümü mevcut pozisyonu tutmaya uygun."

    return ProtectionDecision(
        label,
        protection,
        round(distance, 4),
        round(profit_pct, 4),
        round(holding_risk, 2),
        explanation,
        ("Koruma çizgisi emir fiyatı değildir; TEFAS valör ve fiyatlama kuralları geçerlidir.",),
        round(concentration_risk, 2),
        round(correlation_risk, 2),
    )


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _bounded(value: Any) -> float:
    numeric = _number(value, 100.0)
    return max(0.0, min(100.0, float(numeric)))

