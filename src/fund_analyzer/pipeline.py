"""End-to-end pipeline and command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fund_analyzer import config
from fund_analyzer.data_provider import CsvDataProvider
from fund_analyzer.export_excel import export_to_excel
from fund_analyzer.indicators import add_indicators
from fund_analyzer.models import PipelineResult
from fund_analyzer.portfolio import build_portfolio_analysis
from fund_analyzer.report import write_run_report
from fund_analyzer.returns import add_returns
from fund_analyzer.sector_summary import build_sector_summary
from fund_analyzer.strategy_engine import analyze_funds
from fund_analyzer.tefas_export_provider import (
    TefasExportDataProvider,
    enrich_with_export_metadata,
    export_metadata,
    filter_export_universe,
    funds_with_unknown_type,
    parse_tefas_export,
    write_generated_universe,
)
from fund_analyzer.tefas_provider import TefasDataProvider, TefasProviderError, parse_fund_codes


def run_pipeline(
    data_dir: Path | str = Path("data"),
    output_path: Path | str = Path("fund_analysis_output.xlsx"),
    report_path: Path | str = Path("reports/latest_run_report.md"),
    provider: str = "csv",
    lookback_months: int = 12,
    funds: str | list[str] | None = None,
    fund_file: Path | str | None = None,
    portfolio_file: Path | str | None = None,
    export_file: Path | str | None = None,
    write_universe: Path | str | None = None,
    fund_type: str | None = None,
    risk_min: int | None = None,
    risk_max: int | None = None,
    min_return_1m: float | None = None,
    min_return_3m: float | None = None,
    min_return_6m: float | None = None,
    min_return_ytd: float | None = None,
    min_return_1y: float | None = None,
    limit: int | None = None,
    include_funds: str | list[str] | None = None,
    exclude_funds: str | list[str] | None = None,
    refresh: bool = False,
    request_delay: float | None = None,
    batch_size: int | None = None,
    max_funds: int | None = None,
    continue_on_error: bool = False,
    resume: bool = False,
    sort_by: str | None = None,
    sort_desc: bool = False,
    sort_asc: bool = False,
) -> PipelineResult:
    """Run the full MVP pipeline with the selected data provider."""

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    report_path = Path(report_path)
    provider_name = provider.lower().strip()

    loaded = _load_provider_data(
        provider_name=provider_name,
        data_dir=data_dir,
        lookback_months=lookback_months,
        funds=funds,
        fund_file=fund_file,
        portfolio_file=portfolio_file,
        export_file=export_file,
        write_universe=write_universe,
        fund_type=fund_type,
        risk_min=risk_min,
        risk_max=risk_max,
        min_return_1m=min_return_1m,
        min_return_3m=min_return_3m,
        min_return_6m=min_return_6m,
        min_return_ytd=min_return_ytd,
        min_return_1y=min_return_1y,
        limit=limit,
        include_funds=include_funds,
        exclude_funds=exclude_funds,
        refresh=refresh,
        request_delay=request_delay,
        batch_size=batch_size,
        max_funds=max_funds,
        continue_on_error=continue_on_error,
        resume=resume,
        sort_by=sort_by,
        sort_desc=sort_desc,
        sort_asc=sort_asc,
    )
    funds = loaded.funds
    raw_prices = loaded.prices
    portfolio = loaded.portfolio
    validation_warnings = [*loaded.warnings]
    if loaded.metadata.get("historical_prices_available", True):
        validation_warnings.extend(_cross_reference_warnings(funds, raw_prices, portfolio))

    if raw_prices.empty:
        prices = raw_prices.copy()
        analysis = _universe_only_analysis(funds)
    else:
        prices = add_returns(add_indicators(raw_prices))
        analysis = analyze_funds(raw_prices, funds)
    portfolio_analysis = build_portfolio_analysis(portfolio, funds, analysis)
    sector_summary = build_sector_summary(analysis)
    provider_metadata = {
        **loaded.metadata,
        "portfolio_rows_analyzed": len(portfolio_analysis),
    }

    batch_errors_rows = []
    for error_dict in provider_metadata.get("failed_funds", []):
        batch_errors_rows.append({"fund_code": error_dict["fund_code"], "issue_type": "HTTP Error", "message": error_dict.get("error_reason", "")})
    for error_dict in provider_metadata.get("empty_response_funds", []):
        batch_errors_rows.append({"fund_code": error_dict["fund_code"], "issue_type": "Empty Response", "message": error_dict.get("error_reason", "")})
    
    batch_errors_df = pd.DataFrame(batch_errors_rows, columns=["fund_code", "issue_type", "message"]) if batch_errors_rows else None

    export_to_excel(
        funds=funds,
        prices=prices,
        analysis=analysis,
        portfolio=portfolio_analysis,
        sector_summary=sector_summary,
        output_path=output_path,
        batch_errors=batch_errors_df,
    )
    write_run_report(
        funds=funds,
        prices=prices,
        analysis=analysis,
        sector_summary=sector_summary,
        validation_warnings=validation_warnings,
        report_path=report_path,
        provider_name=provider_name,
        provider_metadata=provider_metadata,
    )

    return PipelineResult(
        funds=funds,
        prices=prices,
        analysis=analysis,
        portfolio=portfolio_analysis,
        sector_summary=sector_summary,
        output_path=output_path,
        report_path=report_path,
        validation_warnings=validation_warnings,
        provider_name=provider_name,
        provider_metadata=provider_metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fund analysis MVP pipeline.")
    parser.add_argument(
        "--provider",
        choices=["csv", "tefas", "tefas-export", "tefas-combined"],
        default="csv",
        help="Input data provider to use.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing input CSV files.")
    parser.add_argument(
        "--output",
        default="fund_analysis_output.xlsx",
        help="Path for the generated Excel workbook.",
    )
    parser.add_argument(
        "--report",
        default="reports/latest_run_report.md",
        help="Path for the generated markdown run report.",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=12,
        help="TEFAS lookback period. Supported values: 1, 3, 6, 12, 36, 60.",
    )
    parser.add_argument(
        "--funds",
        default=None,
        help="Comma-separated TEFAS fund codes, for example AFT,YAS,TI2.",
    )
    parser.add_argument(
        "--fund-file",
        default=None,
        help="CSV file with fund_code and optional fund_name/fund_type columns.",
    )
    parser.add_argument(
        "--portfolio-file",
        default=None,
        help="Portfolio CSV to use instead of the provider default.",
    )
    parser.add_argument("--export-file", default=None, help="Manual TEFAS CSV/XLSX table export file.")
    parser.add_argument(
        "--write-universe",
        default=None,
        help="Write a clean universe CSV generated from a TEFAS export.",
    )
    parser.add_argument("--fund-type", default=None, help="Filter tefas-combined by fund_type.")
    parser.add_argument("--risk-min", type=int, default=None, help="Minimum TEFAS risk value.")
    parser.add_argument("--risk-max", type=int, default=None, help="Maximum TEFAS risk value.")
    parser.add_argument("--min-return-1m", type=float, default=None)
    parser.add_argument("--min-return-3m", type=float, default=None)
    parser.add_argument("--min-return-6m", type=float, default=None)
    parser.add_argument("--min-return-ytd", type=float, default=None)
    parser.add_argument("--min-return-1y", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit filtered tefas-combined universe.")
    parser.add_argument("--include-funds", default=None, help="Comma-separated funds to add after filtering.")
    parser.add_argument("--exclude-funds", default=None, help="Comma-separated funds to remove after filtering.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore TEFAS cache and force fresh downloads.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=None,
        help="Delay between TEFAS fund requests (defaults to 0.3s).",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=None,
        help="Legacy alias for --request-delay.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Chunk size for console progress reporting.")
    parser.add_argument("--max-funds", type=int, default=None, help="Limit maximum historical fund fetches.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue fetching if a fund fails.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop fetching immediately if a fund fails.")
    parser.add_argument("--resume", action="store_true", help="Skip already cached successful funds from network fetch.")
    parser.add_argument("--sort-by", default=None, help="Field to sort export universe by before truncating.")
    parser.add_argument("--sort-desc", action="store_true", help="Sort descending.")
    parser.add_argument("--sort-asc", action="store_true", help="Sort ascending.")
    parser.add_argument(
        "--preset",
        choices=["equity-momentum-25", "equity-momentum-50", "balanced-positive-25", "all-risk-positive-25"],
        help="Apply predefined preset filters. Explicit CLI args override preset values."
    )
    args = parser.parse_args()

    if args.continue_on_error and args.stop_on_error:
        parser.error("Cannot provide both --continue-on-error and --stop-on-error.")
        
    if args.preset:
        if args.preset == "equity-momentum-25":
            if parser.get_default("provider") == args.provider:
                args.provider = "tefas-combined"
            if args.fund_type is None: args.fund_type = "Hisse Senedi Şemsiye Fonu"
            if args.risk_min is None: args.risk_min = 5
            if args.risk_max is None: args.risk_max = 7
            if args.min_return_1m is None: args.min_return_1m = 0.0
            if args.max_funds is None: args.max_funds = 25
            if args.batch_size is None: args.batch_size = 5
            if args.sort_by is None: args.sort_by = "return_1m_pct"
            if not args.sort_desc and not args.sort_asc: args.sort_desc = True
        elif args.preset == "equity-momentum-50":
            if parser.get_default("provider") == args.provider:
                args.provider = "tefas-combined"
            if args.fund_type is None: args.fund_type = "Hisse Senedi Şemsiye Fonu"
            if args.risk_min is None: args.risk_min = 5
            if args.risk_max is None: args.risk_max = 7
            if args.min_return_1m is None: args.min_return_1m = 0.0
            if args.max_funds is None: args.max_funds = 50
            if args.batch_size is None: args.batch_size = 10
            if args.sort_by is None: args.sort_by = "return_1m_pct"
            if not args.sort_desc and not args.sort_asc: args.sort_desc = True
        elif args.preset == "balanced-positive-25":
            if parser.get_default("provider") == args.provider:
                args.provider = "tefas-combined"
            if args.risk_min is None: args.risk_min = 3
            if args.risk_max is None: args.risk_max = 6
            if args.min_return_1m is None: args.min_return_1m = 0.0
            if args.min_return_3m is None: args.min_return_3m = 0.0
            if args.max_funds is None: args.max_funds = 25
            if args.batch_size is None: args.batch_size = 5
            if args.sort_by is None: args.sort_by = "return_3m_pct"
            if not args.sort_desc and not args.sort_asc: args.sort_desc = True
        elif args.preset == "all-risk-positive-25":
            if parser.get_default("provider") == args.provider:
                args.provider = "tefas-combined"
            if args.risk_min is None: args.risk_min = 1
            if args.risk_max is None: args.risk_max = 7
            if args.min_return_1m is None: args.min_return_1m = 0.0
            if args.min_return_3m is None: args.min_return_3m = 0.0
            if args.max_funds is None: args.max_funds = 25
            if args.batch_size is None: args.batch_size = 5
            if args.sort_by is None: args.sort_by = "return_1m_pct"
            if not args.sort_desc and not args.sort_asc: args.sort_desc = True

        if args.provider == "tefas-combined" and not args.export_file:
            default_export = Path(args.data_dir) / "tefas_exports" / "tefas_funds.xlsx"
            if default_export.exists():
                args.export_file = str(default_export)
            else:
                parser.error("Preset requires --export-file or a default export at data/tefas_exports/tefas_funds.xlsx")
    request_delay = args.request_delay if args.request_delay is not None else args.request_delay_seconds


    try:
        result = run_pipeline(
            data_dir=args.data_dir,
            output_path=args.output,
            report_path=args.report,
            provider=args.provider,
            lookback_months=args.lookback_months,
            funds=args.funds,
            fund_file=args.fund_file,
            portfolio_file=args.portfolio_file,
            export_file=args.export_file,
            write_universe=args.write_universe,
            fund_type=args.fund_type,
            risk_min=args.risk_min,
            risk_max=args.risk_max,
            min_return_1m=args.min_return_1m,
            min_return_3m=args.min_return_3m,
            min_return_6m=args.min_return_6m,
            min_return_ytd=args.min_return_ytd,
            min_return_1y=args.min_return_1y,
            limit=args.limit,
            include_funds=args.include_funds,
            exclude_funds=args.exclude_funds,
            refresh=args.refresh,
            request_delay=request_delay,
            batch_size=args.batch_size,
            max_funds=args.max_funds,
            continue_on_error=args.continue_on_error,
            resume=args.resume,
            sort_by=args.sort_by,
            sort_desc=args.sort_desc,
            sort_asc=args.sort_asc,
        )
    except TefasProviderError as exc:
        parser.error(str(exc))

    print(f"Created {result.output_path}")
    print(f"Created {result.report_path}")


def _load_provider_data(
    provider_name: str,
    data_dir: Path,
    lookback_months: int,
    funds: str | list[str] | None,
    fund_file: Path | str | None,
    portfolio_file: Path | str | None,
    export_file: Path | str | None,
    write_universe: Path | str | None,
    fund_type: str | None,
    risk_min: int | None,
    risk_max: int | None,
    min_return_1m: float | None,
    min_return_3m: float | None,
    min_return_6m: float | None,
    min_return_ytd: float | None,
    min_return_1y: float | None,
    limit: int | None,
    include_funds: str | list[str] | None,
    exclude_funds: str | list[str] | None,
    refresh: bool,
    request_delay: float | None,
    batch_size: int | None,
    max_funds: int | None,
    continue_on_error: bool,
    resume: bool,
    sort_by: str | None,
    sort_desc: bool,
    sort_asc: bool,
):
    if provider_name == "csv":
        return CsvDataProvider(
            data_dir=data_dir,
            portfolio_path=Path(portfolio_file) if portfolio_file else None,
        ).load_all()
    if provider_name == "tefas":
        fund_codes = parse_fund_codes(funds) if isinstance(funds, str) else funds
        return TefasDataProvider(
            data_dir=data_dir,
            fund_codes=fund_codes,
            fund_file=Path(fund_file) if fund_file else None,
            portfolio_file=Path(portfolio_file) if portfolio_file else None,
            lookback_months=lookback_months,
            refresh=refresh,
            request_delay_seconds=request_delay
            if request_delay is not None
            else config.TEFAS_REQUEST_DELAY_SECONDS,
            continue_on_error=continue_on_error,
            resume=resume,
            batch_size=batch_size,
        ).load_all()
    if provider_name == "tefas-export":
        if not export_file:
            raise TefasProviderError("--export-file is required for --provider tefas-export.")
        return TefasExportDataProvider(
            export_file=Path(export_file),
            write_universe_path=Path(write_universe) if write_universe else None,
        ).load_all()
    if provider_name == "tefas-combined":
        if not export_file:
            raise TefasProviderError("--export-file is required for --provider tefas-combined.")
        return _load_tefas_combined(
            data_dir=data_dir,
            export_file=Path(export_file),
            write_universe=Path(write_universe) if write_universe else None,
            funds=funds,
            fund_type=fund_type,
            risk_min=risk_min,
            risk_max=risk_max,
            min_return_1m=min_return_1m,
            min_return_3m=min_return_3m,
            min_return_6m=min_return_6m,
            min_return_ytd=min_return_ytd,
            min_return_1y=min_return_1y,
            limit=limit,
            include_funds=include_funds,
            exclude_funds=exclude_funds,
            portfolio_file=portfolio_file,
            lookback_months=lookback_months,
            refresh=refresh,
            request_delay=request_delay,
            batch_size=batch_size,
            max_funds=max_funds,
            continue_on_error=continue_on_error,
            resume=resume,
            sort_by=sort_by,
            sort_desc=sort_desc,
            sort_asc=sort_asc,
        )
    raise ValueError(f"Unsupported provider: {provider_name}")


def _load_tefas_combined(
    data_dir: Path,
    export_file: Path,
    write_universe: Path | None,
    funds: str | list[str] | None,
    fund_type: str | None,
    risk_min: int | None,
    risk_max: int | None,
    min_return_1m: float | None,
    min_return_3m: float | None,
    min_return_6m: float | None,
    min_return_ytd: float | None,
    min_return_1y: float | None,
    limit: int | None,
    include_funds: str | list[str] | None,
    exclude_funds: str | list[str] | None,
    portfolio_file: Path | str | None,
    lookback_months: int,
    refresh: bool,
    request_delay: float | None,
    batch_size: int | None,
    max_funds: int | None,
    continue_on_error: bool,
    resume: bool,
    sort_by: str | None,
    sort_desc: bool,
    sort_asc: bool,
):
    warnings: list[str] = []
    export_universe = parse_tefas_export(export_file, warnings=warnings)
    if write_universe:
        write_generated_universe(export_universe, write_universe)

    exact_funds = parse_fund_codes(funds) if isinstance(funds, str) else funds
    if exact_funds:
        filtered = export_universe.loc[export_universe["fund_code"].isin(exact_funds)].copy()
        missing = sorted(set(exact_funds) - set(filtered["fund_code"]))
        if missing:
            warnings.append("Selected funds missing from TEFAS export: " + ", ".join(missing))
            filtered = pd.concat(
                [
                    filtered,
                    pd.DataFrame(
                        [
                            {
                                "fund_code": code,
                                "fund_name": code,
                                "fund_type": config.TEFAS_UNKNOWN_FUND_TYPE,
                            }
                            for code in missing
                        ]
                    ),
                ],
                ignore_index=True,
            )
    else:
        filtered = filter_export_universe(
            export_universe,
            fund_type=fund_type,
            risk_min=risk_min,
            risk_max=risk_max,
            min_return_1m=min_return_1m,
            min_return_3m=min_return_3m,
            min_return_6m=min_return_6m,
            min_return_ytd=min_return_ytd,
            min_return_1y=min_return_1y,
            include_funds=parse_fund_codes(include_funds) if isinstance(include_funds, str) else include_funds,
            exclude_funds=parse_fund_codes(exclude_funds) if isinstance(exclude_funds, str) else exclude_funds,
            limit=limit,
            sort_by=sort_by,
            sort_desc=sort_desc,
            sort_asc=sort_asc,
        )

    selected_codes = filtered["fund_code"].dropna().astype(str).tolist()
    if max_funds is not None:
        selected_codes = selected_codes[:max_funds]

    historical = TefasDataProvider(
        data_dir=data_dir,
        fund_codes=selected_codes,
        portfolio_file=Path(portfolio_file) if portfolio_file else None,
        lookback_months=lookback_months,
        refresh=refresh,
        request_delay_seconds=request_delay
        if request_delay is not None
        else config.TEFAS_REQUEST_DELAY_SECONDS,
        universe_override=filtered,
        continue_on_error=continue_on_error,
        resume=resume,
        batch_size=batch_size,
    ).load_all()

    enriched_funds = enrich_with_export_metadata(historical.funds, filtered)
    metadata = {
        **historical.metadata,
        **export_metadata(
            provider="tefas-combined",
            export_file=export_file,
            export_rows_loaded=len(export_universe),
            export_funds_parsed=len(export_universe),
            funds_after_filtering=len(filtered),
            write_universe_path=write_universe,
        ),
        "requested_historical_funds": len(selected_codes),
        "successfully_fetched_historical_funds": historical.metadata.get("successfully_fetched_funds", 0),
        "failed_historical_fund_codes": historical.metadata.get("failed_fund_codes", []),
        "unknown_fund_types": funds_with_unknown_type(enriched_funds),
    }
    return historical.__class__(
        funds=enriched_funds,
        prices=historical.prices,
        portfolio=historical.portfolio,
        warnings=[*warnings, *historical.warnings],
        metadata=metadata,
    )


def _universe_only_analysis(funds):
    analysis = funds.copy()
    for column in [
        "date",
        "latest_price",
        "daily_return_pct",
        "weekly_return_pct",
        "monthly_return_pct",
        "three_month_return_pct",
        "ema10",
        "ema21",
        "macd",
        "macd_signal",
        "macd_histogram",
        "rsi_14",
        "atr_like",
        "atr_like_pct",
        "score",
    ]:
        analysis[column] = pd.NA
    analysis["label"] = "UNIVERSE ONLY"
    analysis["explanation"] = "No historical daily prices were loaded; technical indicators were not calculated."
    return analysis


def _cross_reference_warnings(
    funds,
    prices,
    portfolio,
) -> list[str]:
    warnings: list[str] = []
    fund_codes = set(funds["fund_code"])
    price_codes = set(prices["fund_code"])
    portfolio_codes = set(portfolio["fund_code"])

    funds_without_prices = sorted(fund_codes - price_codes)
    if funds_without_prices:
        warnings.append(
            "Funds without price history: " + ", ".join(funds_without_prices)
        )

    price_codes_without_fund = sorted(price_codes - fund_codes)
    if price_codes_without_fund:
        warnings.append(
            "Price rows ignored by analysis because fund metadata is missing: "
            + ", ".join(price_codes_without_fund)
        )

    portfolio_codes_without_fund = sorted(portfolio_codes - fund_codes)
    if portfolio_codes_without_fund:
        warnings.append(
            "Portfolio fund codes missing from fund metadata: "
            + ", ".join(portfolio_codes_without_fund)
        )

    portfolio_codes_without_prices = sorted(portfolio_codes - price_codes)
    if portfolio_codes_without_prices:
        warnings.append(
            "Portfolio fund codes missing latest price data: "
            + ", ".join(portfolio_codes_without_prices)
        )

    return warnings


if __name__ == "__main__":
    main()

