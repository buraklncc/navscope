import pandas as pd

from fund_analyzer.data_provider import CsvDataProvider


def test_load_prices_sorts_dedupes_and_warns(tmp_path):
    prices = pd.DataFrame(
        [
            {"fund_code": "AAA", "date": "2026-01-03", "price": 103.0},
            {"fund_code": "AAA", "date": "2026-01-01", "price": 100.0},
            {"fund_code": "AAA", "date": "2026-01-02", "price": 101.0},
            {"fund_code": "AAA", "date": "2026-01-02", "price": 102.0},
            {"fund_code": "AAA", "date": "2026-01-04", "price": None},
            {"fund_code": "BBB", "date": "bad-date", "price": 10.0},
            {"fund_code": "BBB", "date": "2026-01-01", "price": -1.0},
        ]
    )
    prices.to_csv(tmp_path / "sample_prices.csv", index=False)
    provider = CsvDataProvider(data_dir=tmp_path)
    warnings: list[str] = []

    result = provider.load_prices(warnings=warnings)

    assert result["fund_code"].tolist() == ["AAA", "AAA", "AAA"]
    assert result["date"].tolist() == list(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    assert result["price"].tolist() == [100.0, 102.0, 103.0]
    assert any("missing or invalid" in warning for warning in warnings)
    assert any("non-positive" in warning for warning in warnings)
    assert any("duplicate price row" in warning for warning in warnings)

