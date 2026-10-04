"""Transparent AI evaluation snapshots for the category-aware strategy."""

from __future__ import annotations

from datetime import datetime
import json
import math
from typing import Any, Iterable

from fund_analyzer.categories import CATEGORY_TURKISH_NAMES
from fund_analyzer.strategy_config import CATEGORY_POLICIES, DISCOVERY_LABELS, STRATEGY_VERSION


def build_ai_export_snapshot(
    analysis: Iterable[dict[str, Any]],
    *,
    portfolio: dict[str, Any] | None = None,
    holdings: Iterable[dict[str, Any]] | None = None,
    backtest_summary: dict[str, Any] | None = None,
    recent_signal_changes: Iterable[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    records = [_json_safe(dict(item)) for item in analysis]
    holding_records = [_json_safe(dict(item)) for item in (holdings or [])]
    label_groups = {
        label: sorted(
            [item for item in records if item.get("userLabel", item.get("label")) == label],
            key=lambda item: float(item.get("score") or 0),
            reverse=True,
        )
        for label in DISCOVERY_LABELS
    }
    quality_counts = {status: 0 for status in ("GOOD", "WARNING", "BAD")}
    failed_gate_counts: dict[str, int] = {}
    for item in records:
        status = str(item.get("dataQuality", item.get("data_quality")) or "BAD")
        quality_counts[status] = quality_counts.get(status, 0) + 1
        for gate in item.get("failedSafetyGates", item.get("failed_safety_gates")) or []:
            failed_gate_counts[str(gate)] = failed_gate_counts.get(str(gate), 0) + 1

    risky_holdings = [
        item for item in holding_records
        if item.get("userLabel", item.get("user_label"))
        in {"SAT", "STOP KIRILDI", "STOP'A YAKIN", "ACİL SAT DEĞİL AMA RİSK ARTTI", "VERİ GÜVENSİZ"}
    ]
    payload = {
        "schemaVersion": "3.0",
        "strategyVersion": STRATEGY_VERSION,
        "createdAt": generated_at or datetime.now().isoformat(timespec="seconds"),
        "strategy": {
            "principle": "Belirsizlikte BEKLE; kötü veride olumlu sinyal üretme; kategorileri birbirine karşı kıyaslama.",
            "safetyGates": [
                "minimum history", "missing NAV ratio", "stale NAV", "outlier NAV",
                "category confidence", "fund scale", "extreme drawdown",
                "category-relative volatility", "very poor category rank",
            ],
            "categoryWeights": {
                category.value: policy.weights for category, policy in CATEGORY_POLICIES.items()
            },
            "categoryNames": {
                category.value: name for category, name in CATEGORY_TURKISH_NAMES.items()
            },
            "labelDefinitions": {
                "AL": "Tüm güvenlik kapıları geçti; kategori sırası, risk ve momentum birlikte güçlü.",
                "KADEMELİ AL": "Olumlu kanıt var ancak tam teyit yok.",
                "BEKLE": "Karışık veya yetersiz güçlü sinyal.",
                "UZAK DUR": "Yeni alım için kategori/risk görünümü zayıf.",
                "VERİ EKSİK": "Hesaplama için tarihçe yetersiz.",
                "VERİ GÜVENSİZ": "Eksik, eski, aykırı veya çelişkili veri.",
            },
        },
        "portfolio": _json_safe(portfolio or {}),
        "holdings": holding_records,
        "marketSnapshot": {"fundCount": len(records), "funds": records},
        "topFunds": {
            "buy": label_groups.get("AL", [])[:10],
            "gradualBuy": label_groups.get("KADEMELİ AL", [])[:10],
            "avoid": label_groups.get("UZAK DUR", [])[:10],
        },
        "riskyHoldings": risky_holdings,
        "dataQualitySummary": {
            "statusCounts": quality_counts,
            "failedSafetyGateCounts": failed_gate_counts,
            "warningCount": sum(len(item.get("warnings") or []) for item in records),
        },
        "assumptions": [
            "Getiri ve risk metrikleri yalnızca karar tarihinde mevcut NAV kayıtlarından hesaplanır.",
            "Risksiz faiz varsayılan olarak sıfırdır; Sharpe ve Sortino bu varsayımla yorumlanmalıdır.",
            "Benchmark, fon büyüklüğü, yatırımcı sayısı, gider ve vergi alanları yoksa motor bunu uydurmaz.",
            "Koruma çizgisi emir fiyatı değildir; fonun alım/satım valörü ve fiyat açıklama zamanı geçerlidir.",
        ],
        "knownLimitations": [
            "TEFAS tatil takvimi yerine iş günü yaklaşımı kullanılan yerler olabilir.",
            "Yabancı varlık ve döviz etkisi benchmark serisi yoksa tam ayrıştırılamaz.",
            "Delist olmuş fon kapsamı yalnızca yerel tarihsel katalog kadar eksiksizdir.",
            "Korelasyon ve beta yalnızca eş zamanlı ve yeterli benchmark/portföy geçmişi varsa hesaplanabilir.",
            "Bu çıktı kişiselleştirilmiş yatırım tavsiyesi değildir.",
        ],
        "backtestSummary": _json_safe(backtest_summary or {"status": "NOT_RUN"}),
        "recentSignalChanges": [_json_safe(dict(item)) for item in (recent_signal_changes or [])],
    }
    payload["promptReadyText"] = to_prompt_text(payload)
    return payload


def to_json(snapshot: dict[str, Any], *, indent: int = 2) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=indent, sort_keys=True, default=str)


def to_markdown(snapshot: dict[str, Any]) -> str:
    quality = snapshot.get("dataQualitySummary") or {}
    top = snapshot.get("topFunds") or {}
    lines = [
        "# Fon Radarı — AI Strateji Değerlendirme Paketi",
        "",
        f"- Strateji sürümü: {snapshot.get('strategyVersion')}",
        f"- Oluşturulma: {snapshot.get('createdAt')}",
        f"- Fon sayısı: {(snapshot.get('marketSnapshot') or {}).get('fundCount', 0)}",
        f"- Veri kalitesi: {json.dumps(quality.get('statusCounts') or {}, ensure_ascii=False)}",
        "",
        "## AL adayları",
    ]
    lines.extend(_fund_lines(top.get("buy") or []))
    lines.extend(["", "## KADEMELİ AL adayları"])
    lines.extend(_fund_lines(top.get("gradualBuy") or []))
    lines.extend(["", "## UZAK DUR"])
    lines.extend(_fund_lines(top.get("avoid") or []))
    lines.extend(["", "## Portföy uyarıları"])
    lines.extend(_fund_lines(snapshot.get("riskyHoldings") or []))
    lines.extend(["", "## Varsayımlar"])
    lines.extend(f"- {item}" for item in snapshot.get("assumptions") or [])
    lines.extend(["", "## Bilinen sınırlamalar"])
    lines.extend(f"- {item}" for item in snapshot.get("knownLimitations") or [])
    return "\n".join(lines)


def to_prompt_text(snapshot: dict[str, Any]) -> str:
    compact = {
        "strategyVersion": snapshot.get("strategyVersion"),
        "strategy": snapshot.get("strategy"),
        "portfolio": snapshot.get("portfolio"),
        "holdings": snapshot.get("holdings"),
        "topFunds": snapshot.get("topFunds"),
        "riskyHoldings": snapshot.get("riskyHoldings"),
        "dataQualitySummary": snapshot.get("dataQualitySummary"),
        "backtestSummary": snapshot.get("backtestSummary"),
        "recentSignalChanges": snapshot.get("recentSignalChanges"),
        "assumptions": snapshot.get("assumptions"),
        "knownLimitations": snapshot.get("knownLimitations"),
    }
    return (
        "Aşağıdaki Fon Radarı strateji snapshot'ını denetle. Kesin yatırım talimatı verme; "
        "veri kalitesi, lookahead/survivorship riski, kategori karşılaştırması, düşüş riski "
        "ve etiket gerekçelerini eleştir. Belirsiz alanları açıkça belirt.\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    )


def _fund_lines(records: Iterable[dict[str, Any]]) -> list[str]:
    output = []
    for item in records:
        code = item.get("fundCode", item.get("fund_code"))
        label = item.get("userLabel", item.get("user_label", item.get("label")))
        output.append(
            f"- {code}: {label}; skor {item.get('score')}; güven {item.get('confidence')}; "
            f"gerekçe: {item.get('shortExplanation', item.get('short_explanation', item.get('explanation', '')))}"
        )
    return output or ["- Kayıt yok."]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)

