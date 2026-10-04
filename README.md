# NavScope

Headless mutual-fund analytics: validated NAV history, category-aware metrics, portfolio analysis and reproducible XLSX reports.

Development started: **8 July 2026**. This date is based on the earliest recorded project commit. This repository presents the current backend snapshot.

## Quick start

Python 3.10+.

```sh
python -m pip install -e ".[dev]"
navscope --provider csv --data-dir examples/data --output output/analysis.xlsx --report output/run.md
python -m pytest
```

## Pipeline

CSV or supplied TEFAS export → validation → daily indicators → category and quality gates → portfolio analysis → workbook and report.

- EMA, MACD, Wilder RSI and NAV movement estimates.
- Category-specific strategy policies and explicit missing/stale-data handling.
- Point-in-time replay utilities with execution-delay assumptions.
- Optional TEFAS adapter with cache and bounded retries; offline examples need no credentials.

## Boundaries

This is an experimental analytics library, not a hosted portfolio service. A label is a rule output, not a verified trading outcome. Live TEFAS access depends on the upstream endpoint. Sample files are example inputs.

## Development

Source is in `src/fund_analyzer`; regression tests are in `tests`. MIT license.

