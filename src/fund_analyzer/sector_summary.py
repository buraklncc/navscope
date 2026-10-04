"""Fund type label summary."""

from __future__ import annotations

import pandas as pd

from fund_analyzer import config


def build_sector_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Group funds by fund type and count final labels."""

    labels = _summary_labels(analysis)
    ordered_columns = ["fund_type", *labels, "TOTAL", "net_balance"]
    if analysis.empty:
        return pd.DataFrame(columns=ordered_columns)

    counts = (
        analysis.pivot_table(
            index="fund_type",
            columns="label",
            values="fund_code",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    for label in labels:
        if label not in counts.columns:
            counts[label] = 0

    counts["TOTAL"] = counts[labels].sum(axis=1)
    counts["net_balance"] = sum(
        counts[label] * weight
        for label, weight in config.SECTOR_NET_BALANCE_WEIGHTS.items()
        if label in counts.columns
    )

    return counts[ordered_columns].sort_values("fund_type").reset_index(drop=True)


def _summary_labels(analysis: pd.DataFrame) -> list[str]:
    labels = list(config.LABELS)
    if "label" in analysis:
        for label in sorted(set(analysis["label"].dropna()) - set(labels)):
            labels.append(label)
    return labels

