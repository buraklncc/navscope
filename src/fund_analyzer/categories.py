"""Fund category classification with explicit uncertainty.

TEFAS metadata is useful but not always complete or stable.  The classifier
therefore keeps the inferred category, confidence, and the evidence used to
reach it instead of pretending every classification is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable
import unicodedata

import pandas as pd


class FundCategory(str, Enum):
    MONEY_MARKET = "MONEY_MARKET"
    SHORT_TERM_DEBT = "SHORT_TERM_DEBT"
    LONG_TERM_DEBT = "LONG_TERM_DEBT"
    EQUITY_TURKEY = "EQUITY_TURKEY"
    FOREIGN_EQUITY = "FOREIGN_EQUITY"
    GOLD_PRECIOUS_METALS = "GOLD_PRECIOUS_METALS"
    FX_EUROBOND = "FX_EUROBOND"
    VARIABLE_MIXED = "VARIABLE_MIXED"
    FUND_OF_FUNDS = "FUND_OF_FUNDS"
    PARTICIPATION_SUKUK = "PARTICIPATION_SUKUK"
    OTHER_UNKNOWN = "OTHER_UNKNOWN"


CATEGORY_TURKISH_NAMES: dict[FundCategory, str] = {
    FundCategory.MONEY_MARKET: "Para Piyasası",
    FundCategory.SHORT_TERM_DEBT: "Kısa Vadeli Borçlanma",
    FundCategory.LONG_TERM_DEBT: "Orta/Uzun Vadeli Borçlanma",
    FundCategory.EQUITY_TURKEY: "Türkiye Hisse Senedi",
    FundCategory.FOREIGN_EQUITY: "Yabancı Hisse Senedi",
    FundCategory.GOLD_PRECIOUS_METALS: "Altın / Kıymetli Madenler",
    FundCategory.FX_EUROBOND: "Döviz / Eurobond",
    FundCategory.VARIABLE_MIXED: "Değişken / Karma",
    FundCategory.FUND_OF_FUNDS: "Fon Sepeti",
    FundCategory.PARTICIPATION_SUKUK: "Katılım / Kira Sertifikası",
    FundCategory.OTHER_UNKNOWN: "Diğer / Bilinmiyor",
}


@dataclass(frozen=True)
class CategoryClassification:
    category: FundCategory
    confidence: float
    source: str
    evidence: tuple[str, ...]
    warnings: tuple[str, ...] = ()


_FIELD_WEIGHTS = {
    "category": 0.62,
    "fund_type": 0.58,
    "portfolio_type": 0.44,
    "benchmark": 0.38,
    "fund_name": 0.34,
}


_KEYWORDS: dict[FundCategory, tuple[str, ...]] = {
    FundCategory.FUND_OF_FUNDS: (
        "fon sepeti",
        "fonlarin fonu",
        "fonlarin fon",
        "fund of funds",
    ),
    FundCategory.MONEY_MARKET: (
        "para piyasasi",
        "likit fon",
        "likit serbest",
        "money market",
    ),
    FundCategory.SHORT_TERM_DEBT: (
        "kisa vadeli borclanma",
        "kisa vadeli tahvil",
        "kisa vadeli bono",
        "kisa vadeli",
        "short term bond",
    ),
    FundCategory.FX_EUROBOND: (
        "eurobond",
        "doviz borclanma",
        "yabanci borclanma",
        "dis borclanma",
        "dolar borclanma",
        "foreign currency debt",
    ),
    FundCategory.GOLD_PRECIOUS_METALS: (
        "altin",
        "kiymetli maden",
        "gumus",
        "gold",
        "precious metal",
    ),
    FundCategory.FOREIGN_EQUITY: (
        "yabanci hisse",
        "global hisse",
        "amerikan hisse",
        "abd hisse",
        "avrupa hisse",
        "asya hisse",
        "gelisen ulke hisse",
        "nasdaq",
        "s&p 500",
        "msci world",
        "teknoloji",
        "semiconductor",
        "foreign equity",
    ),
    FundCategory.EQUITY_TURKEY: (
        "turkiye hisse",
        "yerli hisse",
        "hisse senedi yogun",
        "hisse senedi fonu",
        "bist 100",
        "bist 30",
        "bist",
        "equity turkey",
        "hisse",
        "equity",
    ),
    FundCategory.PARTICIPATION_SUKUK: (
        "kira sertifikasi",
        "sukuk",
        "katilim fonu",
        "katilim",
        "participation",
        "lease certificate",
    ),
    FundCategory.VARIABLE_MIXED: (
        "degisken",
        "karma fon",
        "dengeli",
        "esnek",
        "mixed allocation",
        "variable",
        "balanced",
    ),
    FundCategory.LONG_TERM_DEBT: (
        "uzun vadeli borclanma",
        "orta vadeli borclanma",
        "borclanma araclari",
        "tahvil bono",
        "tahvil ve bono",
        "devlet tahvili",
        "bond",
        "debt instruments",
    ),
}


_GENERIC_VALUES = {"", "unknown", "bilinmiyor", "diger", "other", "yat"}


def classify_fund(
    metadata: dict[str, Any] | pd.Series,
    price_history: pd.DataFrame | None = None,
) -> CategoryClassification:
    """Infer a category and retain classification uncertainty.

    Text metadata controls the result.  Historical behaviour is used only as a
    low-confidence fallback and can never produce a high-confidence category.
    """

    record = metadata.to_dict() if isinstance(metadata, pd.Series) else dict(metadata)
    scores: dict[FundCategory, float] = {category: 0.0 for category in FundCategory}
    evidence: dict[FundCategory, list[str]] = {category: [] for category in FundCategory}

    for field, field_weight in _FIELD_WEIGHTS.items():
        normalized = _normalize_text(record.get(field))
        if normalized in _GENERIC_VALUES:
            continue
        for category, keywords in _KEYWORDS.items():
            matched = [keyword for keyword in keywords if _contains_phrase(normalized, keyword)]
            if not matched:
                continue
            # One field should not dominate merely because several synonyms are present.
            specificity = max(min(len(keyword) / 22, 0.22) for keyword in matched)
            scores[category] += field_weight + specificity
            evidence[category].append(f"{field}: {matched[0]}")

    _resolve_text_conflicts(record, scores, evidence)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_category, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score <= 0:
        fallback = _historical_fallback(price_history)
        if fallback is not None:
            return fallback
        return CategoryClassification(
            category=FundCategory.OTHER_UNKNOWN,
            confidence=0.20,
            source="unknown",
            evidence=(),
            warnings=("Kategori için güvenilir metadata bulunamadı.",),
        )

    margin = max(0.0, best_score - second_score)
    confidence = min(0.98, 0.42 + min(0.40, best_score * 0.48) + min(0.14, margin * 0.20))
    warnings: list[str] = []
    if second_score > 0 and margin < 0.20:
        confidence = min(confidence, 0.58)
        warnings.append("Kategori metadatası birden fazla fon türüyle çelişiyor.")
    if best_category == FundCategory.OTHER_UNKNOWN or confidence < 0.55:
        warnings.append("Kategori güveni düşük; olumlu sinyaller sınırlandırılacak.")

    return CategoryClassification(
        category=best_category,
        confidence=round(confidence, 4),
        source="metadata",
        evidence=tuple(evidence[best_category]),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def classify_funds(funds: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add canonical category fields to a fund table."""

    output = funds.copy()
    rows: list[dict[str, Any]] = []
    price_groups = (
        {code: group for code, group in prices.groupby("fund_code", sort=False)}
        if prices is not None and not prices.empty and "fund_code" in prices
        else {}
    )
    for _, row in output.iterrows():
        code = str(row.get("fund_code") or "")
        result = classify_fund(row, price_groups.get(code))
        rows.append(
            {
                "category": result.category.value,
                "category_name": CATEGORY_TURKISH_NAMES[result.category],
                "category_confidence": result.confidence,
                "category_source": result.source,
                "category_evidence": list(result.evidence),
                "category_warnings": list(result.warnings),
            }
        )
    classified = pd.DataFrame(rows, index=output.index)
    return pd.concat([output, classified], axis=1)


def coerce_category(value: Any) -> FundCategory:
    if isinstance(value, FundCategory):
        return value
    try:
        return FundCategory(str(value))
    except (ValueError, TypeError):
        return FundCategory.OTHER_UNKNOWN


def _resolve_text_conflicts(
    record: dict[str, Any],
    scores: dict[FundCategory, float],
    evidence: dict[FundCategory, list[str]],
) -> None:
    combined = " ".join(
        _normalize_text(record.get(field))
        for field in ("category", "fund_type", "portfolio_type", "benchmark", "fund_name")
    )
    if "hisse" in combined and ("katilim" in combined or "sukuk" in combined):
        # Participation equity funds behave like equity funds; participation is
        # an investment constraint, not their primary return/risk category.
        scores[FundCategory.EQUITY_TURKEY] += 0.28
        evidence[FundCategory.EQUITY_TURKEY].append("katılım hisse yapısı")
    if "yabanci" in combined or "global" in combined or "abd" in combined:
        if "hisse" in combined or "teknoloji" in combined:
            scores[FundCategory.FOREIGN_EQUITY] += 0.75
            scores[FundCategory.EQUITY_TURKEY] = max(
                0.0, scores[FundCategory.EQUITY_TURKEY] - 0.35
            )
    if "kisa vadeli" in combined:
        scores[FundCategory.SHORT_TERM_DEBT] += 0.75
        scores[FundCategory.LONG_TERM_DEBT] = max(0.0, scores[FundCategory.LONG_TERM_DEBT] - 0.80)
    if "fon sepeti" in combined:
        # Asset words in a fund-of-funds name describe the underlying basket.
        scores[FundCategory.FUND_OF_FUNDS] += 0.35


def _historical_fallback(price_history: pd.DataFrame | None) -> CategoryClassification | None:
    if price_history is None or len(price_history) < 63 or "price" not in price_history:
        return None
    prices = pd.to_numeric(price_history["price"], errors="coerce").dropna()
    returns = prices.pct_change().dropna()
    if len(returns) < 42:
        return None
    volatility = float(returns.std(ddof=1) * (252**0.5))
    negative_ratio = float((returns < 0).mean())
    if volatility < 0.035 and negative_ratio < 0.08:
        return CategoryClassification(
            FundCategory.MONEY_MARKET,
            0.44,
            "historical_behavior",
            ("çok düşük oynaklık ve az negatif gün",),
            ("Kategori yalnızca fiyat davranışından düşük güvenle çıkarıldı.",),
        )
    return CategoryClassification(
        FundCategory.OTHER_UNKNOWN,
        0.28,
        "historical_behavior",
        ("metadata yok; fiyat davranışı tek başına ayırt edici değil",),
        ("Kategori güveni düşük; olumlu sinyaller sınırlandırılacak.",),
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", text))


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).casefold().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9&+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def category_values() -> Iterable[str]:
    return (category.value for category in FundCategory)

