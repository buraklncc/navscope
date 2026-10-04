import math

import numpy as np
import pandas as pd

from fund_analyzer.metrics import calculate_benchmark_metrics, calculate_fund_metrics


def test_annualized_return_and_drawdown_match_known_series():
    dates = pd.bdate_range("2025-01-02", periods=253)
    prices = 100 * np.cumprod(np.full(len(dates), 1.001))
    frame = pd.DataFrame({"date": dates, "price": prices})

    metrics = calculate_fund_metrics(frame, as_of=dates[-1])

    assert math.isclose(metrics["annualized_return"], (1.001**252) - 1, rel_tol=1e-10)
    assert metrics["max_drawdown"] == 0
    assert metrics["current_drawdown"] == 0
    assert metrics["positive_day_ratio"] == 1


def test_recovery_time_and_max_drawdown_use_only_past_prices():
    dates = pd.bdate_range("2026-01-02", periods=5)
    frame = pd.DataFrame({"date": dates, "price": [100, 110, 88, 99, 112]})

    before_recovery = calculate_fund_metrics(frame, as_of=dates[3])
    after_recovery = calculate_fund_metrics(frame, as_of=dates[4])

    assert math.isclose(before_recovery["max_drawdown"], -0.2, abs_tol=1e-12)
    assert math.isclose(before_recovery["current_drawdown"], -0.1, abs_tol=1e-12)
    assert before_recovery["history_end"] == dates[3]
    assert after_recovery["recovery_time_days"] == 2


def test_benchmark_beta_and_correlation_are_recomputed_on_aligned_dates():
    dates = pd.bdate_range("2025-01-02", periods=80)
    benchmark_returns = np.tile([0.01, -0.005, 0.002, -0.001], 20)
    fund_returns = benchmark_returns * 2
    benchmark = pd.DataFrame({"date": dates, "price": 100 * np.cumprod(1 + benchmark_returns)})
    fund = pd.DataFrame({"date": dates, "price": 100 * np.cumprod(1 + fund_returns)})

    metrics = calculate_benchmark_metrics(fund, benchmark, as_of=dates[-1])

    assert math.isclose(metrics["beta_to_benchmark"], 2.0, rel_tol=0.02)
    assert math.isclose(metrics["benchmark_correlation"], 1.0, rel_tol=1e-10)
    assert metrics["benchmark_overlap_days"] == 79

