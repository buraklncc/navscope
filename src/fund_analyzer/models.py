"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadedData:
    """Input data returned by a provider with validation warnings."""

    funds: pd.DataFrame
    prices: pd.DataFrame
    portfolio: pd.DataFrame
    warnings: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Container returned by the end-to-end pipeline."""

    funds: pd.DataFrame
    prices: pd.DataFrame
    analysis: pd.DataFrame
    portfolio: pd.DataFrame
    sector_summary: pd.DataFrame
    output_path: Path
    report_path: Path
    validation_warnings: list[str]
    provider_name: str
    provider_metadata: dict[str, object]

