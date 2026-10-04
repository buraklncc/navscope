import pandas as pd

from fund_analyzer.backtest import BacktestConfig, run_point_in_time_backtest, walk_forward_splits


def test_backtest_uses_only_point_in_time_history_and_never_same_day_nav():
    dates = pd.bdate_range("2025-01-02", periods=80)
    prices = pd.DataFrame(
        {
            "fund_code": ["AAA"] * len(dates),
            "date": dates,
            "price": [100 + index for index in range(len(dates))],
            "available_at": dates + pd.offsets.BDay(1),
        }
    )
    funds = pd.DataFrame(
        [{"fund_code": "AAA", "fund_name": "AAA Türkiye Hisse Senedi Fonu", "fund_type": "Hisse Senedi Şemsiye Fonu"}]
    )
    seen_history_ends: list[pd.Timestamp] = []

    def signal_function(history, metadata, decision_date):
        history_end = history["date"].max()
        seen_history_ends.append(history_end)
        assert history_end <= decision_date
        return pd.DataFrame(
            [
                {
                    "fund_code": "AAA",
                    "label": "AL",
                    "score": 90,
                    "category_percentile": 90,
                    "history_end": history_end,
                    "strategy_version": "test-v1",
                }
            ]
        )

    decision_date = dates[30]
    result = run_point_in_time_backtest(
        prices,
        funds,
        config=BacktestConfig(minimum_training_observations=10, buy_valor_days=1),
        decision_dates=[decision_date],
        signal_function=signal_function,
    )

    assert seen_history_ends == [decision_date]
    assert result.trades[0]["side"] == "BUY"
    assert pd.Timestamp(result.trades[0]["execution_date"]) > decision_date
    assert result.trades[0]["execution_date"] == (decision_date + pd.offsets.BDay(2)).date().isoformat()
    assert result.decisions[0]["history_end_used"] == decision_date.date().isoformat()


def test_walk_forward_splits_keep_training_before_test():
    dates = pd.bdate_range("2024-01-02", periods=300)

    splits = walk_forward_splits(
        dates,
        training_observations=126,
        test_observations=42,
        step_observations=42,
    )

    assert splits
    for split in splits:
        assert pd.Timestamp(split["train_end"]) < pd.Timestamp(split["test_start"])

