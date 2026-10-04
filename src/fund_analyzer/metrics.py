"""Professional, point-in-time return and risk metrics for close-only funds."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fund_analyzer.strategy_config import TRADING_DAYS_PER_YEAR


PERIODS = {
    "weekly_return": 5,
    "monthly_return": 21,
    "three_month_return": 63,
    "six_month_return": 126,
    "twelve_month_return": 252,
}


def calculate_fund_metrics(
    prices: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    annual_risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Calculate fund metrics using only rows available on or before ``as_of``."""

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["date", "price"])
    frame = frame.loc[frame["price"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else None
    if cutoff is not None:
        frame = frame.loc[frame["date"] <= cutoff]
    if frame.empty:
        return _empty_metrics()

    series = frame.set_index("date")["price"].astype(float)
    returns = series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    metrics: dict[str, Any] = {
        "date": series.index[-1],
        "latest_price": float(series.iloc[-1]),
        "history_length": int(len(series)),
        "history_start": series.index[0],
        "history_end": series.index[-1],
        "daily_return": _period_return(series, 1),
        **{name: _period_return(series, periods) for name, periods in PERIODS.items()},
        "ytd_return": _ytd_return(series),
    }

    observation_count = len(returns)
    if observation_count:
        annualized_volatility = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if observation_count > 1 else 0.0
        annualized_return = _annualized_return(series)
        downside_returns = returns.clip(upper=0)
        downside_deviation = float(
            np.sqrt(np.mean(np.square(downside_returns))) * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
        excess_return = annualized_return - annual_risk_free_rate if _finite(annualized_return) else math.nan
        sharpe = _safe_ratio(excess_return, annualized_volatility)
        sortino = _safe_ratio(excess_return, downside_deviation)
    else:
        annualized_return = annualized_volatility = downside_deviation = sharpe = sortino = math.nan

    drawdown = series / series.cummax() - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else math.nan
    current_drawdown = float(drawdown.iloc[-1]) if not drawdown.empty else math.nan
    average_drawdown = float(drawdown.loc[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0
    calmar = _safe_ratio(annualized_return, abs(max_drawdown))
    monthly_returns = _calendar_period_returns(series, "M")
    weekly_rolling = _rolling_compound_return(returns, 5)
    monthly_rolling = _rolling_compound_return(returns, 21)

    metrics.update(
        {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "downside_deviation": downside_deviation,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "average_drawdown": average_drawdown,
            "recovery_time_days": _maximum_recovery_observations(series),
            "worst_day": float(returns.min()) if not returns.empty else math.nan,
            "worst_week": float(weekly_rolling.min()) if not weekly_rolling.empty else math.nan,
            "worst_month": float(monthly_rolling.min()) if not monthly_rolling.empty else math.nan,
            "positive_day_ratio": float((returns > 0).mean()) if not returns.empty else math.nan,
            "negative_day_count": int((returns < 0).sum()),
            "positive_month_ratio": float((monthly_returns > 0).mean()) if not monthly_returns.empty else math.nan,
            "rolling_one_month_return": _period_return(series, 21),
            "rolling_three_month_return": _period_return(series, 63),
            "rolling_six_month_return": _period_return(series, 126),
            "average_absolute_daily_movement": float(returns.abs().tail(21).mean()) if not returns.empty else math.nan,
            "return_consistency": _return_consistency(returns),
            "ema10": _ema(series, 10),
            "ema21": _ema(series, 21),
            "macd": _macd(series)[0],
            "macd_signal": _macd(series)[1],
            "rsi_14": _rsi(series, 14),
            "recent_high_21": float(series.tail(21).max()),
            "recent_peak_252": float(series.tail(252).max()),
        }
    )
    return {key: _native_number(value) for key, value in metrics.items()}


def add_category_rankings(features: pd.DataFrame) -> pd.DataFrame:
    """Add within-category ranks without comparing unlike fund types."""

    ranked = features.copy()
    if ranked.empty:
        return ranked
    output_groups: list[pd.DataFrame] = []
    for _, group in ranked.groupby("category", dropna=False, sort=False):
        current = group.copy()
        current["category_sample_size"] = len(current)
        current["category_average_return"] = _mean(current.get("six_month_return"))
        current["category_median_return"] = _median(current.get("six_month_return"))
        current["category_average_annualized_return"] = _mean(current.get("annualized_return"))
        current["tracking_difference_vs_category"] = (
            pd.to_numeric(current.get("annualized_return"), errors="coerce")
            - current["category_average_annualized_return"]
        )
        current["category_rank_return"] = _descending_rank(current.get("six_month_return"))
        current["category_rank_risk_adjusted"] = _descending_rank(current.get("sortino"))
        current["category_rank_drawdown"] = _descending_rank(current.get("max_drawdown"))
        return_percentile = _best_percentile(current.get("six_month_return"))
        risk_adjusted_percentile = _best_percentile(current.get("sortino"))
        drawdown_percentile = _best_percentile(current.get("max_drawdown"))
        consistency_percentile = _best_percentile(current.get("consistency_score"))
        current["category_percentile"] = (
            return_percentile * 0.35
            + risk_adjusted_percentile * 0.30
            + drawdown_percentile * 0.20
            + consistency_percentile * 0.15
        ).round(2)
        current["top_quartile"] = current["category_percentile"] >= 75
        current["bottom_quartile"] = current["category_percentile"] <= 25
        current["category_relative_volatility"] = _relative_to_median(current.get("annualized_volatility"))
        current["category_relative_drawdown"] = _relative_to_median(current.get("max_drawdown"), absolute=True)
        current["category_relative_sortino"] = _relative_to_median(current.get("sortino"))
        current["category_relative_sharpe"] = _relative_to_median(current.get("sharpe"))
        current["category_relative_consistency"] = _relative_to_median(current.get("consistency_score"))
        output_groups.append(current)
    return pd.concat(output_groups).sort_index()


def metric_percent(value: Any) -> float | None:
    if value is None or not _finite(value):
        return None
    return round(float(value) * 100, 6)


def calculate_benchmark_metrics(
    fund_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, float]:
    """Calculate beta/correlation/tracking metrics on aligned NAV dates."""

    fund = _price_series(fund_prices, as_of)
    benchmark = _price_series(benchmark_prices, as_of)
    aligned = pd.concat(
        [fund.pct_change(fill_method=None).rename("fund"), benchmark.pct_change(fill_method=None).rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 21:
        return {
            "beta_to_benchmark": math.nan,
            "benchmark_correlation": math.nan,
            "tracking_difference": math.nan,
            "tracking_error": math.nan,
            "information_ratio": math.nan,
            "benchmark_overlap_days": float(len(aligned)),
        }
    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    beta = (
        float(aligned[["fund", "benchmark"]].cov().loc["fund", "benchmark"] / benchmark_variance)
        if benchmark_variance > 1e-16 else math.nan
    )
    correlation = float(aligned["fund"].corr(aligned["benchmark"]))
    active_returns = aligned["fund"] - aligned["benchmark"]
    tracking_difference = float(active_returns.mean() * TRADING_DAYS_PER_YEAR)
    tracking_error = float(active_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    information_ratio = _safe_ratio(tracking_difference, tracking_error)
    return {
        "beta_to_benchmark": beta,
        "benchmark_correlation": correlation,
        "tracking_difference": tracking_difference,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "benchmark_overlap_days": float(len(aligned)),
    }


def calculate_portfolio_correlation(
    fund_prices: pd.DataFrame,
    portfolio_returns: pd.Series,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> float:
    """Return correlation with a dated portfolio-return series when available."""

    fund = _price_series(fund_prices, as_of).pct_change(fill_method=None).rename("fund")
    portfolio = portfolio_returns.copy()
    portfolio.index = pd.to_datetime(portfolio.index, errors="coerce")
    portfolio = pd.to_numeric(portfolio, errors="coerce").rename("portfolio")
    aligned = pd.concat([fund, portfolio], axis=1, join="inner").dropna()
    return float(aligned["fund"].corr(aligned["portfolio"])) if len(aligned) >= 21 else math.nan


def calculate_fund_stats_trends(
    stats: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, float]:
    """Calculate scale/investor trends without inventing missing metadata."""

    frame = stats.copy()
    if "date" not in frame:
        return {"fund_size": math.nan, "fund_size_trend": math.nan, "investor_count": math.nan, "investor_count_trend": math.nan}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if as_of is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(as_of).normalize()]
    frame = frame.dropna(subset=["date"]).sort_values("date")
    output: dict[str, float] = {}
    for column in ("fund_size", "investor_count"):
        values = pd.to_numeric(frame.get(column), errors="coerce").dropna() if column in frame else pd.Series(dtype=float)
        output[column] = float(values.iloc[-1]) if not values.empty else math.nan
        trailing = values.tail(63)
        output[f"{column}_trend"] = (
            float(trailing.iloc[-1] / trailing.iloc[0] - 1)
            if len(trailing) >= 2 and trailing.iloc[0] > 0 else math.nan
        )
    return output


def _period_return(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return math.nan
    return float(series.iloc[-1] / series.iloc[-periods - 1] - 1)


def _price_series(prices: pd.DataFrame, as_of: str | pd.Timestamp | None) -> pd.Series:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["date", "price"]).loc[lambda item: item["price"] > 0]
    if as_of is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(as_of).normalize()]
    return frame.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["price"].astype(float)


def _ytd_return(series: pd.Series) -> float:
    latest_year = series.index[-1].year
    current_year = series.loc[series.index.year == latest_year]
    if current_year.empty:
        return math.nan
    before_year = series.loc[series.index.year < latest_year]
    base = float(before_year.iloc[-1]) if not before_year.empty else float(current_year.iloc[0])
    return float(series.iloc[-1] / base - 1) if base else math.nan


def _annualized_return(series: pd.Series) -> float:
    observations = len(series) - 1
    if observations < 20 or series.iloc[0] <= 0:
        return math.nan
    ratio = float(series.iloc[-1] / series.iloc[0])
    if ratio <= 0:
        return math.nan
    return float(ratio ** (TRADING_DAYS_PER_YEAR / observations) - 1)


def _calendar_period_returns(series: pd.Series, frequency: str) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    periods = series.index.to_period(frequency)
    closes = series.groupby(periods).last()
    return closes.pct_change(fill_method=None).dropna()


def _rolling_compound_return(returns: pd.Series, window: int) -> pd.Series:
    if len(returns) < window:
        return pd.Series(dtype=float)
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def _maximum_recovery_observations(series: pd.Series) -> int | None:
    if len(series) < 2:
        return None
    peak = float(series.iloc[0])
    underwater_start: int | None = None
    maximum = 0
    for index, value in enumerate(series.astype(float)):
        if value >= peak:
            peak = value
            if underwater_start is not None:
                maximum = max(maximum, index - underwater_start)
                underwater_start = None
        elif underwater_start is None:
            underwater_start = index
    if underwater_start is not None:
        maximum = max(maximum, len(series) - 1 - underwater_start)
    return int(maximum)


def _return_consistency(returns: pd.Series) -> float:
    if len(returns) < 21:
        return math.nan
    rolling = (1 + returns).rolling(21).apply(np.prod, raw=True) - 1
    valid = rolling.dropna()
    if valid.empty:
        return math.nan
    positive_ratio = float((valid > 0).mean())
    dispersion = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    penalty = min(0.5, dispersion * 4)
    return max(0.0, min(1.0, positive_ratio - penalty + 0.25))


def _ema(series: pd.Series, period: int) -> float:
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def _macd(series: pd.Series) -> tuple[float, float]:
    macd_series = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    signal = macd_series.ewm(span=9, adjust=False).mean()
    return float(macd_series.iloc[-1]), float(signal.iloc[-1])


def _rsi(series: pd.Series, period: int) -> float:
    if len(series) <= period:
        return math.nan
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    latest_gain = float(average_gain.iloc[-1])
    latest_loss = float(average_loss.iloc[-1])
    if latest_gain == 0 and latest_loss == 0:
        return 50.0
    if latest_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + latest_gain / latest_loss))


def _best_percentile(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    output = pd.Series(50.0, index=numeric.index, dtype=float)
    if valid.empty:
        return output
    if len(valid) == 1 or valid.nunique() == 1:
        output.loc[valid.index] = 50.0
        return output
    ascending_rank = valid.rank(method="average", ascending=True)
    output.loc[valid.index] = (ascending_rank - 1) / (len(valid) - 1) * 100
    return output


def _descending_rank(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="Int64")
    return pd.to_numeric(series, errors="coerce").rank(method="min", ascending=False).astype("Int64")


def _relative_to_median(series: pd.Series | None, *, absolute: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    if absolute:
        numeric = numeric.abs()
    median = numeric.median(skipna=True)
    if pd.isna(median) or abs(float(median)) < 1e-12:
        return pd.Series(math.nan, index=numeric.index)
    return numeric / float(median)


def _mean(series: pd.Series | None) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean()) if series is not None else math.nan


def _median(series: pd.Series | None) -> float:
    return float(pd.to_numeric(series, errors="coerce").median()) if series is not None else math.nan


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    if not _finite(numerator) or not _finite(denominator) or abs(float(denominator)) < 1e-12:
        return math.nan
    return float(numerator) / float(denominator)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _native_number(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, str, int)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if _finite(value):
        return float(value)
    return math.nan


def _empty_metrics() -> dict[str, Any]:
    keys = [
        "date", "latest_price", "history_length", "history_start", "history_end", "daily_return",
        *PERIODS,
        "ytd_return", "annualized_return", "annualized_volatility", "downside_deviation", "sharpe",
        "sortino", "calmar", "max_drawdown", "current_drawdown", "average_drawdown",
        "recovery_time_days", "worst_day", "worst_week", "worst_month", "positive_day_ratio",
        "negative_day_count", "positive_month_ratio", "rolling_one_month_return",
        "rolling_three_month_return", "rolling_six_month_return", "average_absolute_daily_movement",
        "return_consistency", "ema10", "ema21", "macd", "macd_signal", "rsi_14",
        "recent_high_21", "recent_peak_252",
    ]
    return {key: 0 if key in {"history_length", "negative_day_count"} else math.nan for key in keys}

