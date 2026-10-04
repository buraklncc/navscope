"""Point-in-time mutual-fund backtest foundation with valor-aware execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from fund_analyzer.strategy_config import STRATEGY_VERSION
from fund_analyzer.strategy_engine import analyze_funds


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    rebalance_every_business_days: int = 21
    buy_valor_days: int = 1
    sell_valor_days: int = 1
    transaction_cost_bps: float = 10.0
    tax_rate: float = 0.0
    max_positions: int = 5
    include_gradual_buy: bool = True
    minimum_training_observations: int = 126


@dataclass(frozen=True)
class BacktestResult:
    strategy_version: str
    configuration: dict[str, Any]
    summary: dict[str, Any]
    trades: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    caveats: list[str]


SignalFunction = Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], pd.DataFrame]


def run_point_in_time_backtest(
    prices: pd.DataFrame,
    funds: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    decision_dates: Iterable[str | pd.Timestamp] | None = None,
    signal_function: SignalFunction | None = None,
    benchmark_histories: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """Run a conservative long-only backtest without same-day NAV execution.

    The signal function receives an already-truncated price table.  An explicit
    invariant check aborts if any future row leaks into that table.
    """

    cfg = config or BacktestConfig()
    frame = _clean_prices(prices)
    if frame.empty:
        return BacktestResult(
            STRATEGY_VERSION, asdict(cfg), _empty_summary(), [], [], [],
            ["Kullanılabilir fiyat verisi yok."],
        )
    dates = _decision_dates(frame, cfg, decision_dates)
    scorer = signal_function or (
        lambda history, metadata, current_date: analyze_funds(
            history,
            metadata,
            evaluation_date=current_date,
            benchmark_histories=benchmark_histories,
        )
    )
    cash = float(cfg.initial_capital)
    holdings: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    previous_labels: dict[str, str] = {}
    label_changes = 0
    turnover_value = 0.0

    for decision_date in dates:
        history = frame.loc[frame["date"] <= decision_date].copy()
        if not history.empty and history["date"].max() > decision_date:
            raise RuntimeError("Lookahead detected: signal history exceeds decision date")
        analysis = scorer(history.copy(), funds.copy(), decision_date)
        if not analysis.empty:
            history_ends = pd.to_datetime(analysis.get("history_end"), errors="coerce")
            if history_ends.notna().any() and history_ends.max() > decision_date:
                raise RuntimeError("Lookahead detected: analysis used a future NAV")
        by_code = {str(row["fund_code"]): row for row in analysis.to_dict("records")}
        for code, row in by_code.items():
            label = str(row.get("label") or "VERİ EKSİK")
            if code in previous_labels and previous_labels[code] != label:
                label_changes += 1
            previous_labels[code] = label
            decisions.append(
                {
                    "decision_date": decision_date.date().isoformat(),
                    "fund_code": code,
                    "label": label,
                    "score": _native(row.get("score")),
                    "history_end_used": _date_string(row.get("history_end")),
                    "strategy_version": row.get("strategy_version", STRATEGY_VERSION),
                }
            )

        buy_labels = {"AL", "KADEMELİ AL"} if cfg.include_gradual_buy else {"AL"}
        candidates = sorted(
            [row for row in by_code.values() if row.get("label") in buy_labels],
            key=lambda item: (float(item.get("score") or 0), float(item.get("category_percentile") or 0)),
            reverse=True,
        )[: cfg.max_positions]
        target_codes = {str(row["fund_code"]) for row in candidates}

        for code in list(holdings):
            row = by_code.get(code, {})
            must_exit = code not in target_codes or row.get("label") in {"UZAK DUR", "VERİ GÜVENSİZ", "VERİ EKSİK"}
            if not must_exit:
                continue
            execution = _execution_row(frame, code, decision_date, cfg.sell_valor_days)
            if execution is None:
                continue
            position = holdings.pop(code)
            gross = position["units"] * execution["price"]
            fee = gross * cfg.transaction_cost_bps / 10_000
            gain = gross - fee - position["cost_value"]
            tax = max(0.0, gain) * cfg.tax_rate
            proceeds = gross - fee - tax
            cash += proceeds
            turnover_value += gross
            trades.append(
                _trade(
                    code, "SELL", decision_date, execution, position["units"], execution["price"],
                    fee, tax, proceeds, position["entry_date"], gain - tax,
                )
            )

        open_slots = max(0, cfg.max_positions - len(holdings))
        new_candidates = [row for row in candidates if str(row["fund_code"]) not in holdings][:open_slots]
        for index, row in enumerate(new_candidates):
            code = str(row["fund_code"])
            execution = _execution_row(frame, code, decision_date, cfg.buy_valor_days)
            if execution is None:
                continue
            remaining = max(1, len(new_candidates) - index)
            budget = cash / remaining
            fee_rate = cfg.transaction_cost_bps / 10_000
            units = budget / (execution["price"] * (1 + fee_rate))
            gross = units * execution["price"]
            fee = gross * fee_rate
            total = gross + fee
            if units <= 0 or total > cash + 1e-8:
                continue
            cash -= total
            turnover_value += gross
            holdings[code] = {
                "units": units,
                "cost_value": total,
                "entry_date": execution["date"],
            }
            trades.append(
                _trade(code, "BUY", decision_date, execution, units, execution["price"], fee, 0.0, -total)
            )

    equity_curve = _build_equity_curve(
        frame,
        dates[0] if dates else frame["date"].min(),
        cfg.initial_capital,
        trades,
    )
    summary = _performance_summary(
        equity_curve,
        trades,
        cfg.initial_capital,
        turnover_value,
        label_changes,
        decisions,
        frame,
    )
    summary["comparisons"] = build_comparison_summaries(
        frame,
        funds,
        start_date=dates[0] if dates else frame["date"].min(),
        end_date=frame["date"].max(),
        benchmark_histories=benchmark_histories,
    )
    caveats = [
        "Karar NAV'ı ile aynı gün işlem yapılmaz; ilk uygun sonraki NAV ve valör gecikmesi kullanılır.",
        "Eksik günler işlem fiyatı olarak doldurulmaz; portföy değerlemesinde son bilinen NAV yalnızca gösterim amacıyla taşınır.",
        "Vergi/gider varsayımları yapılandırmadaki oranlarla sınırlıdır.",
        "Delist ve birleşen fonlar ancak giriş verisinde bulunuyorsa kapsanır; evren bu nedenle survivorship bias taşıyabilir.",
    ]
    return BacktestResult(STRATEGY_VERSION, asdict(cfg), summary, trades, decisions, equity_curve, caveats)


def walk_forward_splits(
    dates: Iterable[str | pd.Timestamp],
    *,
    training_observations: int,
    test_observations: int,
    step_observations: int | None = None,
) -> list[dict[str, str]]:
    ordered = sorted(pd.Timestamp(value).normalize() for value in set(dates))
    step = step_observations or test_observations
    splits: list[dict[str, str]] = []
    start = training_observations
    while start + test_observations <= len(ordered):
        splits.append(
            {
                "train_start": ordered[0].date().isoformat(),
                "train_end": ordered[start - 1].date().isoformat(),
                "test_start": ordered[start].date().isoformat(),
                "test_end": ordered[start + test_observations - 1].date().isoformat(),
            }
        )
        start += max(1, step)
    return splits


def rolling_window_splits(
    dates: Iterable[str | pd.Timestamp],
    *,
    training_observations: int,
    test_observations: int,
    step_observations: int | None = None,
) -> list[dict[str, str]]:
    ordered = sorted(pd.Timestamp(value).normalize() for value in set(dates))
    step = step_observations or test_observations
    splits: list[dict[str, str]] = []
    train_start = 0
    while train_start + training_observations + test_observations <= len(ordered):
        test_start = train_start + training_observations
        splits.append(
            {
                "train_start": ordered[train_start].date().isoformat(),
                "train_end": ordered[test_start - 1].date().isoformat(),
                "test_start": ordered[test_start].date().isoformat(),
                "test_end": ordered[test_start + test_observations - 1].date().isoformat(),
            }
        )
        train_start += max(1, step)
    return splits


def parameter_sensitivity(
    prices: pd.DataFrame,
    funds: pd.DataFrame,
    configurations: Iterable[BacktestConfig],
) -> list[dict[str, Any]]:
    results = []
    for configuration in configurations:
        outcome = run_point_in_time_backtest(prices, funds, config=configuration)
        results.append(
            {
                "configuration": asdict(configuration),
                "summary": outcome.summary,
                "robustness_score": _robustness_score(outcome.summary),
            }
        )
    return sorted(results, key=lambda item: item["robustness_score"], reverse=True)


def compare_strategy_versions(
    prices: pd.DataFrame,
    funds: pd.DataFrame,
    strategies: dict[str, SignalFunction],
    *,
    config: BacktestConfig | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        version: run_point_in_time_backtest(
            prices, funds, config=config, signal_function=signal_function
        ).summary
        for version, signal_function in strategies.items()
    }


def build_comparison_summaries(
    prices: pd.DataFrame,
    funds: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    benchmark_histories: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Build point-in-time-selected buy/hold and peer-category comparisons."""

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    history_at_start = prices.loc[prices["date"] <= start]
    start_analysis = analyze_funds(history_at_start, funds, evaluation_date=start)
    all_codes = start_analysis["fund_code"].astype(str).tolist()
    top_codes = start_analysis.loc[start_analysis["category_percentile"] >= 75, "fund_code"].astype(str).tolist()
    if not top_codes and not start_analysis.empty:
        top_codes = start_analysis.nlargest(min(5, len(start_analysis)), "score")["fund_code"].astype(str).tolist()
    money_market_codes = start_analysis.loc[start_analysis["category"] == "MONEY_MARKET", "fund_code"].astype(str).tolist()
    categories = {
        str(category): group["fund_code"].astype(str).tolist()
        for category, group in start_analysis.groupby("category", dropna=False)
    }
    output: dict[str, Any] = {
        "buy_and_hold": _equal_weight_baseline(prices, all_codes, start, end),
        "point_in_time_top_quartile": _equal_weight_baseline(prices, top_codes, start, end),
        "money_market_alternative": _equal_weight_baseline(prices, money_market_codes, start, end),
        "category_average": {
            category: _equal_weight_baseline(prices, codes, start, end)
            for category, codes in categories.items()
        },
        "benchmark_proxies": {},
    }
    for name, benchmark in (benchmark_histories or {}).items():
        cleaned = benchmark.copy()
        cleaned["fund_code"] = str(name)
        output["benchmark_proxies"][str(name)] = _equal_weight_baseline(
            _clean_prices(cleaned), [str(name)], start, end
        )
    return output


def _default_signal_function(history: pd.DataFrame, funds: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    return analyze_funds(history, funds, evaluation_date=decision_date)


def _equal_weight_baseline(
    prices: pd.DataFrame,
    fund_codes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    if not fund_codes:
        return {"status": "UNAVAILABLE", "total_return": None, "cagr": None, "max_drawdown": None, "fund_count": 0}
    subset = prices.loc[
        prices["fund_code"].astype(str).isin(fund_codes)
        & (prices["date"] > start)
        & (prices["date"] <= end)
    ]
    if subset.empty:
        return {"status": "UNAVAILABLE", "total_return": None, "cagr": None, "max_drawdown": None, "fund_count": 0}
    pivot = subset.pivot_table(index="date", columns="fund_code", values="price", aggfunc="last").sort_index()
    normalized = pd.DataFrame(index=pivot.index)
    for column in pivot:
        series = pivot[column].dropna()
        if len(series) >= 2 and series.iloc[0] > 0:
            normalized[column] = pivot[column] / float(series.iloc[0])
    if normalized.empty:
        return {"status": "UNAVAILABLE", "total_return": None, "cagr": None, "max_drawdown": None, "fund_count": 0}
    curve = normalized.ffill().mean(axis=1, skipna=True).dropna()
    if len(curve) < 2:
        return {"status": "UNAVAILABLE", "total_return": None, "cagr": None, "max_drawdown": None, "fund_count": len(normalized.columns)}
    total_return = float(curve.iloc[-1] / curve.iloc[0] - 1)
    years = max(1 / 252, (len(curve) - 1) / 252)
    cagr = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1)
    max_drawdown = float((curve / curve.cummax() - 1).min())
    return {
        "status": "READY",
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "max_drawdown": round(max_drawdown, 6),
        "fund_count": len(normalized.columns),
        "selection_date": start.date().isoformat(),
        "first_execution_nav_after_selection": True,
    }


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"fund_code", "date", "price"}
    if missing := required.difference(prices.columns):
        raise ValueError(f"prices missing required fields: {', '.join(sorted(missing))}")
    frame = prices.copy()
    frame["fund_code"] = frame["fund_code"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["fund_code", "date", "price"])
    frame = frame.loc[frame["price"] > 0]
    return frame.sort_values(["date", "fund_code"]).drop_duplicates(["fund_code", "date"], keep="last")


def _decision_dates(
    frame: pd.DataFrame,
    config: BacktestConfig,
    supplied: Iterable[str | pd.Timestamp] | None,
) -> list[pd.Timestamp]:
    available = sorted(frame["date"].drop_duplicates())
    if supplied is not None:
        selected = sorted(pd.Timestamp(value).normalize() for value in supplied)
        return [value for value in selected if available[0] <= value <= available[-1]]
    start = min(max(0, config.minimum_training_observations), max(0, len(available) - 1))
    return available[start:: max(1, config.rebalance_every_business_days)]


def _execution_row(
    frame: pd.DataFrame,
    fund_code: str,
    decision_date: pd.Timestamp,
    valor_days: int,
) -> dict[str, Any] | None:
    candidates = frame.loc[
        (frame["fund_code"] == fund_code) & (frame["date"] > decision_date)
    ].copy()
    candidates["_execution_date"] = candidates["date"]
    if "available_at" in candidates:
        available_at = pd.to_datetime(candidates["available_at"], errors="coerce")
        delayed = available_at.notna() & (available_at.dt.normalize() > candidates["_execution_date"])
        candidates.loc[delayed, "_execution_date"] = available_at.loc[delayed].dt.normalize()
    candidates = candidates.loc[candidates["_execution_date"] > decision_date].sort_values("_execution_date")
    index = max(1, int(valor_days)) - 1
    if len(candidates) <= index:
        return None
    row = candidates.iloc[index]
    return {"date": pd.Timestamp(row["_execution_date"]), "price": float(row["price"])}


def _trade(
    code: str,
    side: str,
    decision_date: pd.Timestamp,
    execution: dict[str, Any],
    units: float,
    price: float,
    fee: float,
    tax: float,
    cash_flow: float,
    entry_date: Any = None,
    realized_pnl: Any = None,
) -> dict[str, Any]:
    execution_date = pd.Timestamp(execution["date"])
    if execution_date <= decision_date:
        raise RuntimeError("Same-day or pre-signal execution is forbidden")
    return {
        "fund_code": code,
        "side": side,
        "signal_date": decision_date.date().isoformat(),
        "execution_date": execution_date.date().isoformat(),
        "execution_price": round(price, 8),
        "units": round(units, 8),
        "fee": round(fee, 6),
        "tax": round(tax, 6),
        "cash_flow": round(cash_flow, 6),
        "entry_date": _date_string(entry_date),
        "realized_pnl": _native(realized_pnl),
    }


def _build_equity_curve(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    initial_capital: float,
    trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valuation_dates = set(frame.loc[frame["date"] >= start_date, "date"].drop_duplicates())
    valuation_dates.update(pd.Timestamp(item["execution_date"]) for item in trades)
    dates = sorted(valuation_dates)
    by_code = {code: group.set_index("date")["price"].sort_index() for code, group in frame.groupby("fund_code")}
    trades_by_date: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        trades_by_date.setdefault(str(trade["execution_date"]), []).append(trade)
    cash = float(initial_capital)
    units_by_code: dict[str, float] = {}
    curve = []
    for current_date in dates:
        for trade in trades_by_date.get(current_date.date().isoformat(), []):
            code = str(trade["fund_code"])
            units = float(trade["units"])
            cash += float(trade["cash_flow"])
            units_by_code[code] = units_by_code.get(code, 0.0) + (units if trade["side"] == "BUY" else -units)
            if abs(units_by_code[code]) < 1e-10:
                units_by_code.pop(code, None)
        value = cash
        priced = 0
        stale_positions = 0
        for code, units in units_by_code.items():
            history = by_code.get(code)
            if history is None:
                continue
            known = history.loc[history.index <= current_date]
            if known.empty:
                continue
            value += units * float(known.iloc[-1])
            priced += 1
            stale_positions += int(known.index[-1] < current_date)
        curve.append(
            {
                "date": current_date.date().isoformat(),
                "value": round(value, 6),
                "priced_positions": priced,
                "stale_valuation_positions": stale_positions,
                "valuation_uses_last_known_nav": bool(stale_positions),
            }
        )
    return curve


def _performance_summary(
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    initial_capital: float,
    turnover_value: float,
    label_changes: int,
    decisions: list[dict[str, Any]],
    prices: pd.DataFrame,
) -> dict[str, Any]:
    if not equity_curve:
        return _empty_summary()
    values = pd.Series([float(item["value"]) for item in equity_curve], dtype=float)
    returns = values.pct_change(fill_method=None).dropna()
    total_return = float(values.iloc[-1] / initial_capital - 1)
    years = max(1 / 252, (len(values) - 1) / 252)
    cagr = float((values.iloc[-1] / initial_capital) ** (1 / years) - 1) if values.iloc[-1] > 0 else -1.0
    volatility = float(returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 else 0.0
    downside = returns.clip(upper=0)
    downside_dev = (
        float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(252))
        if not downside.empty else 0.0
    )
    drawdown = values / values.cummax() - 1
    max_drawdown = float(drawdown.min())
    sharpe = cagr / volatility if volatility > 1e-12 else None
    sortino = cagr / downside_dev if downside_dev > 1e-12 else None
    calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else None
    sells = [item for item in trades if item["side"] == "SELL"]
    holding_days = [
        np.busday_count(pd.Timestamp(item["entry_date"]).date(), pd.Timestamp(item["execution_date"]).date())
        for item in sells if item.get("entry_date")
    ]
    wins = [item for item in sells if float(item.get("realized_pnl") or 0) > 0]
    false_positives = [item for item in sells if float(item.get("realized_pnl") or 0) <= 0]
    return {
        "status": "READY",
        "cagr": round(cagr, 6),
        "total_return": round(total_return, 6),
        "annualized_volatility": round(volatility, 6),
        "sharpe": _native(sharpe),
        "sortino": _native(sortino),
        "calmar": _native(calmar),
        "max_drawdown": round(max_drawdown, 6),
        "worst_month": _native(((1 + returns).rolling(21).apply(np.prod, raw=True) - 1).min() if len(returns) >= 21 else None),
        "recovery_time": _recovery_observations(values),
        "turnover": round(turnover_value / max(initial_capital, 1e-9), 6),
        "average_holding_period": round(float(np.mean(holding_days)), 2) if holding_days else None,
        "win_rate": round(len(wins) / len(sells), 6) if sells else None,
        "false_positive_rate": round(len(false_positives) / len(sells), 6) if sells else None,
        "false_negative_rate": _false_negative_rate(decisions, prices),
        "label_changes": label_changes,
        "transaction_count": len(trades),
        "average_tax_friction": round(float(np.mean([item["tax"] + item["fee"] for item in trades])), 6) if trades else 0.0,
    }


def _false_negative_rate(
    decisions: list[dict[str, Any]],
    prices: pd.DataFrame,
    *,
    horizon_observations: int = 63,
    material_return: float = 0.08,
) -> float | None:
    negatives = [
        item for item in decisions
        if item.get("label") not in {"AL", "KADEMELİ AL", "VERİ EKSİK", "VERİ GÜVENSİZ"}
    ]
    evaluated = false_negatives = 0
    for decision in negatives:
        decision_date = pd.Timestamp(decision["decision_date"])
        future = prices.loc[
            (prices["fund_code"] == decision["fund_code"])
            & (prices["date"] > decision_date)
        ].sort_values("date")
        base = prices.loc[
            (prices["fund_code"] == decision["fund_code"])
            & (prices["date"] <= decision_date)
        ].sort_values("date")
        if len(future) < horizon_observations or base.empty:
            continue
        evaluated += 1
        forward_return = float(future.iloc[horizon_observations - 1]["price"] / base.iloc[-1]["price"] - 1)
        false_negatives += int(forward_return >= material_return)
    return round(false_negatives / evaluated, 6) if evaluated else None


def _recovery_observations(values: pd.Series) -> int:
    drawdown = values / values.cummax() - 1
    longest = current = 0
    for underwater in (drawdown < 0):
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    return longest


def _robustness_score(summary: dict[str, Any]) -> float:
    """Penalized diagnostic objective; never optimize on return alone."""

    cagr = float(summary.get("cagr") or 0)
    drawdown = abs(float(summary.get("max_drawdown") or 0))
    turnover = float(summary.get("turnover") or 0)
    label_changes = float(summary.get("label_changes") or 0)
    false_positive = float(summary.get("false_positive_rate") or 0)
    score = cagr * 100 - drawdown * 140 - turnover * 2 - label_changes * 0.05 - false_positive * 20
    return round(score, 6)


def _empty_summary() -> dict[str, Any]:
    return {
        "status": "NOT_RUN", "cagr": None, "total_return": None,
        "annualized_volatility": None, "sharpe": None, "sortino": None,
        "calmar": None, "max_drawdown": None, "worst_month": None,
        "recovery_time": None, "turnover": None, "average_holding_period": None,
        "win_rate": None, "false_positive_rate": None, "false_negative_rate": None,
        "label_changes": 0, "transaction_count": 0, "average_tax_friction": 0.0,
    }


def _date_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)


def _native(value: Any) -> Any:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return round(numeric, 6) if math.isfinite(numeric) else None

