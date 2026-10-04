"""Markdown run report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fund_analyzer import config


def write_run_report(
    funds: pd.DataFrame,
    prices: pd.DataFrame,
    analysis: pd.DataFrame,
    sector_summary: pd.DataFrame,
    validation_warnings: list[str],
    report_path: Path,
    provider_name: str = "csv",
    provider_metadata: dict[str, object] | None = None,
) -> Path:
    """Write a compact markdown report for the latest pipeline run."""

    provider_metadata = provider_metadata or {}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    latest_date = _latest_date_text(analysis)
    latest_price_date = provider_metadata.get("latest_price_date") or _latest_price_date_text(prices)
    analyzed_funds = int(analysis["latest_price"].notna().sum()) if "latest_price" in analysis else 0
    label_names = _report_labels(analysis)
    label_distribution = analysis["label"].value_counts().reindex(label_names, fill_value=0).reset_index()
    label_distribution.columns = ["label", "count"]
    strategy_version = (
        str(analysis["strategy_version"].dropna().iloc[0])
        if "strategy_version" in analysis and not analysis["strategy_version"].dropna().empty
        else "legacy/unavailable"
    )
    quality_distribution = (
        analysis["data_quality"].value_counts(dropna=False).rename_axis("data_quality").reset_index(name="count")
        if "data_quality" in analysis else pd.DataFrame(columns=["data_quality", "count"])
    )
    category_distribution = (
        analysis["category"].value_counts(dropna=False).rename_axis("category").reset_index(name="count")
        if "category" in analysis else pd.DataFrame(columns=["category", "count"])
    )

    lines = [
        "# Latest Fund Analysis Run",
        "",
        f"- Funds loaded: {len(funds)}",
        f"- Price rows loaded: {len(prices)}",
        f"- Provider: {provider_name}",
        f"- Duration: {provider_metadata.get('duration_seconds', 'N/A')}s",
        f"- Export file used: {provider_metadata.get('export_file_used', 'None')}",
        f"- Export rows loaded: {provider_metadata.get('export_rows_loaded', 0)}",
        f"- Export funds parsed: {provider_metadata.get('export_funds_parsed', 0)}",
        f"- Funds after filtering: {provider_metadata.get('funds_after_filtering', provider_metadata.get('requested_funds', len(funds)))}",
        f"- Requested funds: {provider_metadata.get('requested_funds', len(funds))}",
        f"- Successfully fetched funds: {provider_metadata.get('successfully_fetched_funds', analyzed_funds)}",
        f"- Failed funds: {_format_failed_objects(provider_metadata.get('failed_funds', []))}",
        f"- Empty response funds: {_format_failed_objects(provider_metadata.get('empty_response_funds', []))}",
        f"- Requested historical funds: {provider_metadata.get('requested_historical_funds', provider_metadata.get('requested_funds', len(funds)))}",
        f"- Successfully fetched historical funds: {provider_metadata.get('successfully_fetched_historical_funds', provider_metadata.get('successfully_fetched_funds', analyzed_funds))}",
        f"- Failed historical fund codes: {_format_failed_codes(provider_metadata.get('failed_historical_fund_codes', provider_metadata.get('failed_fund_codes', [])))}",
        f"- Funds missing universe mapping: {_format_failed_codes(provider_metadata.get('funds_missing_universe_mapping', []))}",
        f"- Funds with Unknown fund_type: {_format_failed_codes(provider_metadata.get('funds_with_unknown_fund_type', []))}",
        f"- Unknown fund types: {_format_failed_codes(provider_metadata.get('unknown_fund_types', provider_metadata.get('funds_with_unknown_fund_type', [])))}",
        f"- Cache hits: {provider_metadata.get('cache_hits', 0)}",
        f"- Cache misses: {provider_metadata.get('cache_misses', 0)}",
        f"- Network attempts: {provider_metadata.get('network_attempts', 0)}",
        f"- Network successes: {provider_metadata.get('network_successes', 0)}",
        f"- Network failures: {provider_metadata.get('network_failures', 0)}",
        f"- Skipped cached funds: {len(provider_metadata.get('skipped_cached_fund_codes', []))}",
        f"- Latest analysis date: {latest_date}",
        f"- Latest price date: {latest_price_date}",
        f"- Analyzed funds: {analyzed_funds}",
        f"- Strategy version: {strategy_version}",
        f"- Portfolio file used: {provider_metadata.get('portfolio_file_used', 'data/sample_portfolio.csv')}",
        f"- Portfolio rows analyzed: {provider_metadata.get('portfolio_rows_analyzed', 0)}",
        f"- Portfolio skipped: {_yes_no(provider_metadata.get('portfolio_skipped', False))}",
        "",
        "## Label Distribution",
        "",
        _to_markdown_table(label_distribution),
        "",
        "## Sector Summary",
        "",
        _to_markdown_table(sector_summary),
        "",
        "## Category Distribution",
        "",
        _to_markdown_table(category_distribution),
        "",
        "## Data Quality Distribution",
        "",
        _to_markdown_table(quality_distribution),
        "",
        "## Data Validation Warnings",
        "",
    ]

    if validation_warnings:
        lines.extend(f"- {warning}" for warning in validation_warnings)
    else:
        lines.append("- None")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _latest_date_text(analysis: pd.DataFrame) -> str:
    if "date" not in analysis or analysis["date"].dropna().empty:
        return "N/A"
    latest_date = analysis["date"].max()
    if pd.isna(latest_date):
        return "N/A"
    return pd.Timestamp(latest_date).date().isoformat()


def _latest_price_date_text(prices: pd.DataFrame) -> str:
    if "date" not in prices or prices["date"].dropna().empty:
        return "N/A"
    latest_date = prices["date"].max()
    if pd.isna(latest_date):
        return "N/A"
    return pd.Timestamp(latest_date).date().isoformat()


def _format_failed_codes(value: object) -> str:
    if not value:
        return "None"
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "fund_code" in value[0]:
            return ", ".join(str(f["fund_code"]) for f in value)
        return ", ".join(str(code) for code in value) if value else "None"
    return str(value)


def _format_failed_objects(value: object) -> str:
    if not value:
        return "None"
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "fund_code" in value[0]:
            return ", ".join(str(f["fund_code"]) for f in value)
        return ", ".join(str(code) for code in value) if value else "None"
    return str(value)


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def _report_labels(analysis: pd.DataFrame) -> list[str]:
    labels = list(config.LABELS)
    if "label" in analysis:
        for label in sorted(set(analysis["label"].dropna()) - set(labels)):
            labels.append(label)
    return labels


def _to_markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "_No rows_"

    text_data = data.fillna("").astype(str)
    headers = list(text_data.columns)
    separator = ["---"] * len(headers)
    rows = [headers, separator, *text_data.values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)

