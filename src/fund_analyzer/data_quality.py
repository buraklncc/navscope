"""Point-in-time NAV quality checks and conservative safety gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from fund_analyzer.categories import FundCategory, coerce_category
from fund_analyzer.strategy_config import policy_for


@dataclass(frozen=True)
class DataQualityAssessment:
    score: float
    status: str
    flags: tuple[dict[str, Any], ...]
    failed_safety_gates: tuple[str, ...]
    history_length: int
    missing_ratio: float
    stale_nav_count: int
    freshness_lag_business_days: int
    outlier_count: int

    @property
    def warnings(self) -> list[str]:
        return [str(flag["description"]) for flag in self.flags]


_OUTLIER_LIMITS = {
    FundCategory.MONEY_MARKET: 0.006,
    FundCategory.SHORT_TERM_DEBT: 0.025,
    FundCategory.LONG_TERM_DEBT: 0.09,
    FundCategory.EQUITY_TURKEY: 0.22,
    FundCategory.FOREIGN_EQUITY: 0.24,
    FundCategory.GOLD_PRECIOUS_METALS: 0.18,
    FundCategory.FX_EUROBOND: 0.13,
    FundCategory.VARIABLE_MIXED: 0.16,
    FundCategory.FUND_OF_FUNDS: 0.17,
    FundCategory.PARTICIPATION_SUKUK: 0.08,
    FundCategory.OTHER_UNKNOWN: 0.15,
}

_STALE_RUN_LIMITS = {
    FundCategory.MONEY_MARKET: 2,
    FundCategory.SHORT_TERM_DEBT: 3,
    FundCategory.LONG_TERM_DEBT: 3,
    FundCategory.EQUITY_TURKEY: 2,
    FundCategory.FOREIGN_EQUITY: 4,
    FundCategory.GOLD_PRECIOUS_METALS: 3,
    FundCategory.FX_EUROBOND: 4,
    FundCategory.VARIABLE_MIXED: 3,
    FundCategory.FUND_OF_FUNDS: 4,
    FundCategory.PARTICIPATION_SUKUK: 3,
    FundCategory.OTHER_UNKNOWN: 2,
}


def assess_price_quality(
    prices: pd.DataFrame,
    category: FundCategory | str,
    category_confidence: float,
    *,
    evaluation_date: date | datetime | pd.Timestamp | str | None = None,
) -> DataQualityAssessment:
    """Assess one fund's raw daily NAV history without filling missing values."""

    resolved_category = coerce_category(category)
    policy = policy_for(resolved_category)
    frame = prices.copy()
    flags: list[dict[str, Any]] = []
    gates: list[str] = []

    required = {"date", "price"}
    if missing := required.difference(frame.columns):
        flags.append(_flag("MISSING_COLUMNS", "CRITICAL", f"Zorunlu NAV alanları eksik: {', '.join(sorted(missing))}."))
        return DataQualityAssessment(0.0, "BAD", tuple(flags), ("MISSING_DATA",), 0, 1.0, 0, 999, 0)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    invalid_mask = frame["date"].isna() | frame["price"].isna() | (frame["price"] <= 0)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        flags.append(_flag("INVALID_NAV", "HIGH", f"{invalid_count} NAV kaydı tarih/fiyat açısından geçersiz."))
        gates.append("INVALID_NAV")
    valid = frame.loc[~invalid_mask].copy()
    duplicate_count = int(valid.duplicated("date", keep=False).sum())
    if duplicate_count:
        flags.append(_flag("DUPLICATE_NAV", "HIGH", f"Aynı tarih için {duplicate_count} yinelenen NAV kaydı var."))
        gates.append("DUPLICATE_NAV")
    valid = valid.sort_values("date").drop_duplicates("date", keep="last")
    history_length = len(valid)

    if history_length == 0:
        flags.append(_flag("NO_HISTORY", "CRITICAL", "Fon için kullanılabilir NAV geçmişi yok."))
        return DataQualityAssessment(0.0, "BAD", tuple(flags), ("MISSING_DATA",), 0, 1.0, 0, 999, 0)

    first_date = valid["date"].iloc[0]
    latest_date = valid["date"].iloc[-1]
    expected_dates = pd.bdate_range(first_date, latest_date)
    missing_ratio = max(0.0, 1.0 - history_length / max(len(expected_dates), 1))
    if missing_ratio > 0.12:
        flags.append(_flag("HIGH_MISSING_RATIO", "HIGH", f"İş günü NAV eksik oranı %{missing_ratio * 100:.1f}."))
        gates.append("HIGH_MISSING_RATIO")
    elif missing_ratio > 0.05:
        flags.append(_flag("MISSING_NAV", "MEDIUM", f"İş günü NAV eksik oranı %{missing_ratio * 100:.1f}."))

    eval_date = _normalize_evaluation_date(evaluation_date, latest_date)
    future_count = int((valid["date"] > eval_date).sum())
    if future_count:
        flags.append(_flag("FUTURE_DATED_NAV", "CRITICAL", f"Karar tarihinden sonra {future_count} NAV kaydı bulundu."))
        gates.append("FUTURE_DATED_NAV")
        valid = valid.loc[valid["date"] <= eval_date]
        if valid.empty:
            return DataQualityAssessment(0.0, "BAD", tuple(flags), tuple(dict.fromkeys(gates)), 0, 1.0, 0, 999, 0)
        latest_date = valid["date"].iloc[-1]

    freshness_lag = _business_day_lag(latest_date, eval_date)
    if freshness_lag > 5:
        flags.append(_flag("STALE_NAV", "CRITICAL", f"Son NAV karar tarihinden {freshness_lag} iş günü eski."))
        gates.append("STALE_NAV")
    elif freshness_lag > 2:
        flags.append(_flag("STALE_NAV", "HIGH", f"Son NAV karar tarihinden {freshness_lag} iş günü eski."))
        gates.append("STALE_NAV")
    elif freshness_lag > 1:
        flags.append(_flag("DELAYED_NAV", "MEDIUM", f"Son NAV {freshness_lag} iş günü gecikmeli."))

    returns = valid["price"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    stale_nav_count = _trailing_unchanged_count(valid["price"])
    stale_limit = _STALE_RUN_LIMITS[resolved_category]
    if stale_nav_count > stale_limit:
        severity = "HIGH" if stale_nav_count > stale_limit + 2 else "MEDIUM"
        flags.append(_flag("UNCHANGED_NAV_RUN", severity, f"NAV son {stale_nav_count} kayıtta değişmedi."))
        if severity == "HIGH":
            gates.append("STALE_NAV")

    outlier_count = _outlier_count(returns, _OUTLIER_LIMITS[resolved_category])
    if outlier_count:
        ratio = outlier_count / max(len(returns), 1)
        severity = "HIGH" if ratio > 0.01 or outlier_count >= 3 else "MEDIUM"
        flags.append(_flag("OUTLIER_NAV", severity, f"{outlier_count} olağan dışı günlük NAV hareketi bulundu."))
        if severity == "HIGH":
            gates.append("OUTLIER_NAV")

    if history_length < policy.minimum_history:
        flags.append(
            _flag(
                "INSUFFICIENT_HISTORY",
                "HIGH",
                f"En az {policy.minimum_history} kayıt gerekirken {history_length} kayıt var.",
            )
        )
        gates.append("INSUFFICIENT_HISTORY")
    elif history_length < policy.buy_history:
        flags.append(
            _flag(
                "LIMITED_HISTORY",
                "MEDIUM",
                f"Güçlü AL kararı için {policy.buy_history} kayıt gerekir; mevcut kayıt {history_length}.",
            )
        )
        gates.append("NEW_FUND_LIMITED_HISTORY")

    if resolved_category == FundCategory.OTHER_UNKNOWN:
        flags.append(_flag("UNKNOWN_CATEGORY", "HIGH", "Fon kategorisi güvenilir biçimde belirlenemedi."))
        gates.append("UNKNOWN_CATEGORY")
    elif float(category_confidence or 0) < 0.55:
        flags.append(_flag("LOW_CATEGORY_CONFIDENCE", "HIGH", "Kategori sınıflandırma güveni düşük."))
        gates.append("LOW_CATEGORY_CONFIDENCE")
    elif float(category_confidence or 0) < 0.70:
        flags.append(_flag("CATEGORY_CONFIDENCE_WARNING", "MEDIUM", "Kategori sınıflandırması teyit edilmeli."))
        gates.append("CATEGORY_CONFIDENCE_WARNING")

    score = 100.0
    deductions = {"CRITICAL": 45, "HIGH": 25, "MEDIUM": 10, "LOW": 3}
    score -= sum(deductions[str(flag["severity"])] for flag in flags)
    score = round(max(0.0, min(100.0, score)), 2)
    critical = any(flag["severity"] == "CRITICAL" for flag in flags)
    high_count = sum(flag["severity"] == "HIGH" for flag in flags)
    status = "BAD" if critical or score < 45 or high_count >= 2 else "WARNING" if flags or score < 85 else "GOOD"
    return DataQualityAssessment(
        score=score,
        status=status,
        flags=tuple(flags),
        failed_safety_gates=tuple(dict.fromkeys(gates)),
        history_length=history_length,
        missing_ratio=round(missing_ratio, 6),
        stale_nav_count=stale_nav_count,
        freshness_lag_business_days=freshness_lag,
        outlier_count=outlier_count,
    )


def _outlier_count(returns: pd.Series, hard_limit: float) -> int:
    if returns.empty:
        return 0
    absolute = returns.abs()
    median = float(returns.median())
    mad = float((returns - median).abs().median())
    robust_mask = pd.Series(False, index=returns.index)
    if mad > 1e-10:
        robust_z = 0.6745 * (returns - median).abs() / mad
        robust_mask = robust_z > 12
    return int((robust_mask & (absolute > hard_limit / 3) | (absolute > hard_limit)).sum())


def _trailing_unchanged_count(prices: pd.Series) -> int:
    if len(prices) < 2:
        return 0
    changes = prices.astype(float).diff().abs().fillna(np.inf)
    count = 0
    for unchanged in reversed((changes <= 1e-12).tolist()):
        if not unchanged:
            break
        count += 1
    return count


def _normalize_evaluation_date(value: Any, latest_date: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _business_day_lag(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end <= start:
        return 0
    return int(np.busday_count(start.date(), end.date()))


def _flag(flag_type: str, severity: str, description: str) -> dict[str, str]:
    return {"flagType": flag_type, "severity": severity, "description": description}

