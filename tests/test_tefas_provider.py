from pathlib import Path

import pandas as pd
import pytest

from fund_analyzer import config
from fund_analyzer.cache import JsonFileCache, tefas_price_cache_key
from fund_analyzer.pipeline import run_pipeline
from fund_analyzer.tefas_provider import TefasDataProvider, TefasHttpError


class FakeTefasClient:
    def __init__(self, price_rows_by_code=None, error_codes=None, list_rows=None):
        self.price_rows_by_code = price_rows_by_code or {}
        self.error_codes = set(error_codes or [])
        self.list_rows = list_rows or []
        self.calls = []

    def post_json(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        if endpoint == config.TEFAS_PRICE_ENDPOINT:
            code = payload["fonKodu"]
            if code in self.error_codes:
                raise TefasHttpError(f"HTTP 500 from fake endpoint for {code}")
            return {"resultList": self.price_rows_by_code.get(code, [])}
        if endpoint == config.TEFAS_FUND_LIST_ENDPOINT:
            return {"resultList": self.list_rows}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")


def test_tefas_successful_single_fund_response_parsing(tmp_path):
    _write_portfolio(tmp_path)
    client = FakeTefasClient(
        {
            "AFT": [
                {"fonKodu": "AFT", "fonUnvan": "Ak Portfoy Test", "tarih": "2026-01-01", "fiyat": "1,25"},
                {"fonKodu": "AFT", "fonUnvan": "Ak Portfoy Test", "tarih": "2026-01-02", "fiyat": "1,30"},
            ]
        }
    )
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=client,
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.funds.loc[0, "fund_code"] == "AFT"
    assert loaded.funds.loc[0, "fund_name"] == "Ak Portfoy Test"
    assert loaded.funds.loc[0, "fund_type"] == "Unknown"
    assert loaded.prices["price"].tolist() == [1.25, 1.30]
    assert loaded.metadata["successfully_fetched_funds"] == 1
    assert loaded.metadata["cache_misses"] == 1
    assert loaded.metadata["portfolio_skipped"] is True


def test_tefas_provider_enriches_fund_type_from_universe_csv(tmp_path):
    _write_portfolio(tmp_path)
    _write_fund_universe(tmp_path, [{"fund_code": "AFT", "fund_name": "AFT Real Name", "fund_type": "Equity"}])
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=FakeTefasClient(
            {"AFT": [{"fonKodu": "AFT", "fonUnvan": "TEFAS Name", "tarih": "2026-01-01", "fiyat": 1.0}]}
        ),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.funds.loc[0, "fund_name"] == "AFT Real Name"
    assert loaded.funds.loc[0, "fund_type"] == "Equity"
    assert loaded.metadata["funds_missing_universe_mapping"] == []
    assert loaded.metadata["funds_with_unknown_fund_type"] == []


def test_missing_fund_mapping_becomes_unknown_and_warns(tmp_path):
    _write_portfolio(tmp_path)
    _write_fund_universe(tmp_path, [{"fund_code": "YAS", "fund_name": "YAS Fund", "fund_type": "Bond"}])
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=FakeTefasClient(
            {"AFT": [{"fonKodu": "AFT", "fonUnvan": "TEFAS Name", "tarih": "2026-01-01", "fiyat": 1.0}]}
        ),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.funds.loc[0, "fund_name"] == "TEFAS Name"
    assert loaded.funds.loc[0, "fund_type"] == "Unknown"
    assert loaded.metadata["funds_missing_universe_mapping"] == ["AFT"]
    assert loaded.metadata["funds_with_unknown_fund_type"] == ["AFT"]
    assert any("Funds missing universe mapping: AFT" in warning for warning in loaded.warnings)


def test_tefas_empty_response_marks_failed_fund(tmp_path):
    _write_portfolio(tmp_path)
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["BAD"],
        http_client=FakeTefasClient({"BAD": []}),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.prices.empty
    assert loaded.metadata["failed_fund_codes"] == ["BAD"]
    assert any("empty response" in warning for warning in loaded.warnings)


def test_tefas_invalid_fund_code_behaves_like_empty_response(tmp_path):
    _write_portfolio(tmp_path)
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["NOPE"],
        http_client=FakeTefasClient(),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.prices.empty
    assert loaded.metadata["failed_fund_codes"] == ["NOPE"]


def test_tefas_http_error_is_recorded_without_crashing_batch(tmp_path):
    _write_portfolio(tmp_path)
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["ERR", "AFT"],
        http_client=FakeTefasClient(
            price_rows_by_code={
                "AFT": [{"fonKodu": "AFT", "tarih": "2026-01-01", "fiyat": 1.0}]
            },
            error_codes={"ERR"},
        ),
        request_delay_seconds=0,
        continue_on_error=True,
    )

    loaded = provider.load_all()

    assert loaded.prices["fund_code"].tolist() == ["AFT"]
    assert loaded.metadata["failed_fund_codes"] == ["ERR"]
    assert any("HTTP 500" in warning for warning in loaded.warnings)


def test_tefas_transient_http_error_is_retried_once(tmp_path):
    _write_portfolio(tmp_path)

    class FlakyClient:
        calls = 0

        def post_json(self, _endpoint, _payload):
            self.calls += 1
            if self.calls == 1:
                raise TefasHttpError("temporary HTTP 503")
            return {"resultList": [{"fonKodu": "AFT", "tarih": "2026-01-01", "fiyat": 1.0}]}

    client = FlakyClient()
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=client,
        request_delay_seconds=0,
        continue_on_error=True,
    )

    loaded = provider.load_all()

    assert client.calls == 2
    assert loaded.metadata["retry_attempts"] == 1
    assert loaded.metadata["network_successes"] == 1
    assert loaded.metadata["network_failures"] == 0


def test_tefas_cache_hit_avoids_http_request(tmp_path):
    _write_portfolio(tmp_path)
    cache = JsonFileCache(tmp_path / "cache" / "tefas")
    fetch_date = pd.Timestamp("2026-07-08").date()
    cache.set(
        tefas_price_cache_key("AFT", 12, fetch_date),
        {
            "resultList": [
                {"fonKodu": "AFT", "fonUnvan": "Cached Fund", "tarih": "2026-01-01", "fiyat": 1.1}
            ]
        },
    )
    client = FakeTefasClient(error_codes={"AFT"})
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=client,
        cache=cache,
        fetch_date=fetch_date,
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.prices["price"].tolist() == [1.1]
    assert loaded.metadata["cache_hits"] == 1
    assert client.calls == []


def test_tefas_mode_does_not_load_sample_portfolio_by_default(tmp_path):
    _write_portfolio(tmp_path)
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=FakeTefasClient(
            {"AFT": [{"fonKodu": "AFT", "tarih": "2026-01-01", "fiyat": 1.0}]}
        ),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.portfolio.empty
    assert loaded.metadata["portfolio_skipped"] is True
    assert loaded.metadata["portfolio_file_used"] == "None"


def test_tefas_mode_uses_portfolio_tefas_when_available(tmp_path):
    _write_tefas_portfolio(tmp_path, [{"fund_code": "AFT", "buy_date": "2026-01-01", "buy_price": 1.0, "units": 5}])
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=FakeTefasClient(
            {"AFT": [{"fonKodu": "AFT", "tarih": "2026-01-01", "fiyat": 1.0}]}
        ),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.portfolio["fund_code"].tolist() == ["AFT"]
    assert loaded.metadata["portfolio_skipped"] is False
    assert loaded.metadata["portfolio_file_used"].endswith("portfolio_tefas.csv")


def test_portfolio_file_overrides_default_tefas_portfolio(tmp_path):
    _write_tefas_portfolio(tmp_path, [{"fund_code": "AFT", "buy_date": "2026-01-01", "buy_price": 1.0, "units": 5}])
    override = tmp_path / "custom_portfolio.csv"
    pd.DataFrame(
        [{"fund_code": "YAS", "buy_date": "2026-01-01", "buy_price": 2.0, "units": 8}]
    ).to_csv(override, index=False)
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["YAS"],
        portfolio_file=override,
        http_client=FakeTefasClient(
            {"YAS": [{"fonKodu": "YAS", "tarih": "2026-01-01", "fiyat": 2.1}]}
        ),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.portfolio["fund_code"].tolist() == ["YAS"]
    assert loaded.metadata["portfolio_file_used"] == str(override)


def test_selected_funds_csv_is_used_when_no_fund_selection_is_provided(tmp_path):
    _write_selected_funds(tmp_path, [{"fund_code": "AFT"}])
    provider = TefasDataProvider(
        data_dir=tmp_path,
        http_client=FakeTefasClient(
            {"AFT": [{"fonKodu": "AFT", "tarih": "2026-01-01", "fiyat": 1.0}]}
        ),
        request_delay_seconds=0,
    )

    loaded = provider.load_all()

    assert loaded.funds["fund_code"].tolist() == ["AFT"]
    assert loaded.metadata["requested_funds"] == 1


def test_tefas_duplicate_price_rows_are_deduped(tmp_path):
    _write_portfolio(tmp_path)
    provider = TefasDataProvider(
        data_dir=tmp_path,
        fund_codes=["AFT"],
        http_client=FakeTefasClient(
            {
                "AFT": [
                    {"fonKodu": "AFT", "tarih": "2026-01-01", "fiyat": 1.0},
                    {"fonKodu": "AFT", "tarih": "2026-01-01", "fiyat": 1.2},
                    {"fonKodu": "AFT", "tarih": "2026-01-02", "fiyat": 1.3},
                ]
            }
        ),
        request_delay_seconds=0,
        continue_on_error=True,
    )

    loaded = provider.load_all()

    assert loaded.prices["price"].tolist() == [1.2, 1.3]
    assert any("duplicate price row" in warning for warning in loaded.warnings)


def test_tefas_mocked_pipeline_produces_analysis_table(tmp_path, monkeypatch):
    _write_tefas_portfolio(tmp_path, [{"fund_code": "AFT", "buy_date": "2026-01-01", "buy_price": 1.0, "units": 10}])
    _write_fund_universe(tmp_path, [{"fund_code": "AFT", "fund_name": "Mapped TEFAS Fund", "fund_type": "Equity"}])
    payload = _rising_payload("AFT", periods=90)

    class InjectedProvider(TefasDataProvider):
        def __init__(self, *args, **kwargs):
            kwargs["http_client"] = FakeTefasClient({"AFT": payload})
            kwargs["request_delay_seconds"] = 0
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("fund_analyzer.pipeline.TefasDataProvider", InjectedProvider)

    result = run_pipeline(
        data_dir=tmp_path,
        output_path=tmp_path / "out.xlsx",
        report_path=tmp_path / "report.md",
        provider="tefas",
        funds="AFT",
        lookback_months=12,
        request_delay=0,
    )

    assert result.analysis.loc[0, "fund_code"] == "AFT"
    assert result.analysis.loc[0, "latest_price"] > 0
    assert result.analysis.loc[0, "label"] in set(config.LABELS)
    assert result.analysis.loc[0, "fund_type"] == "Equity"
    assert result.sector_summary.loc[0, "fund_type"] == "Equity"
    assert len(result.portfolio) == 1
    assert (tmp_path / "out.xlsx").exists()
    assert "Provider: tefas" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_csv_provider_selection_still_works(tmp_path):
    _write_csv_inputs(tmp_path)

    result = run_pipeline(
        data_dir=tmp_path,
        output_path=tmp_path / "csv.xlsx",
        report_path=tmp_path / "csv_report.md",
        provider="csv",
    )

    assert result.provider_name == "csv"
    assert len(result.analysis) == 1
    assert len(result.portfolio) == 1
    assert (tmp_path / "csv.xlsx").exists()


def test_excel_export_succeeds_when_tefas_portfolio_is_skipped(tmp_path, monkeypatch):
    payload = _rising_payload("AFT", periods=90)

    class InjectedProvider(TefasDataProvider):
        def __init__(self, *args, **kwargs):
            kwargs["http_client"] = FakeTefasClient({"AFT": payload})
            kwargs["request_delay_seconds"] = 0
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("fund_analyzer.pipeline.TefasDataProvider", InjectedProvider)

    result = run_pipeline(
        data_dir=tmp_path,
        output_path=tmp_path / "no_portfolio.xlsx",
        report_path=tmp_path / "no_portfolio.md",
        provider="tefas",
        funds="AFT",
        lookback_months=12,
    )

    assert result.portfolio.empty
    assert result.provider_metadata["portfolio_skipped"] is True
    assert (tmp_path / "no_portfolio.xlsx").exists()
    assert "Portfolio skipped: yes" in (tmp_path / "no_portfolio.md").read_text(encoding="utf-8")


def test_tefas_unsupported_lookback_fails_clearly(tmp_path):
    _write_portfolio(tmp_path)

    with pytest.raises(Exception, match="Unsupported TEFAS lookback months"):
        TefasDataProvider(data_dir=tmp_path, fund_codes=["AFT"], lookback_months=2)


def _write_portfolio(path: Path) -> None:
    pd.DataFrame(
        [{"fund_code": "AFT", "buy_date": "2026-01-01", "buy_price": 1.0, "units": 10}]
    ).to_csv(path / "sample_portfolio.csv", index=False)


def _write_tefas_portfolio(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path / "portfolio_tefas.csv", index=False)


def _write_selected_funds(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path / "selected_funds.csv", index=False)


def _write_fund_universe(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path / "fund_universe.csv", index=False)


def _write_csv_inputs(path: Path) -> None:
    pd.DataFrame(
        [{"fund_code": "AAA", "fund_name": "CSV Fund", "fund_type": "Equity"}]
    ).to_csv(path / "sample_funds.csv", index=False)
    dates = pd.bdate_range("2026-01-01", periods=90)
    pd.DataFrame(
        {
            "fund_code": ["AAA"] * len(dates),
            "date": dates,
            "price": [100 + index for index in range(len(dates))],
        }
    ).to_csv(path / "sample_prices.csv", index=False)
    pd.DataFrame(
        [{"fund_code": "AAA", "buy_date": "2026-01-01", "buy_price": 100.0, "units": 1}]
    ).to_csv(path / "sample_portfolio.csv", index=False)


def _rising_payload(code: str, periods: int):
    dates = pd.bdate_range("2026-01-01", periods=periods)
    return [
        {
            "fonKodu": code,
            "fonUnvan": "Mock TEFAS Fund",
            "tarih": date.date().isoformat(),
            "fiyat": 1 + index * 0.01,
        }
        for index, date in enumerate(dates)
    ]

