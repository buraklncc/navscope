from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from fund_analyzer.pipeline import run_pipeline
from fund_analyzer.tefas_export_provider import (
    extract_tefas_export_metadata,
    filter_export_universe,
    normalize_tefas_export_frame,
    parse_tefas_export,
    parse_turkish_percent,
    write_generated_universe,
)
from fund_analyzer.tefas_provider import TefasDataProvider
from tests.test_tefas_provider import FakeTefasClient


def test_parse_turkish_percent_and_dash_values():
    assert parse_turkish_percent("%3,1541") == 3.1541
    assert parse_turkish_percent("1.234,56") == 1234.56
    assert parse_turkish_percent("-") is None
    assert parse_turkish_percent("") is None


def test_parse_tefas_csv_export_with_turkish_columns(tmp_path):
    export_path = tmp_path / "tefas_funds.csv"
    export_path.write_text(
        "\n".join(
                [
                    "TEFAS export generated header;;;;;;;;;;",
                    "Fon Kodu;Fon Adı;Şemsiye Fon Türü;Fonun Risk Değeri;1 Ay (%);3 Ay (%);6 Ay (%);Yılbaşından İtibaren (%);1 Yıl (%);3 Yıl (%);5 Yıl (%)",
                    " AFT ;Ak Fon;Hisse Senedi Şemsiye Fonu;5 / 7;%3,1541;%7,20;-;;%12,50;%30,00;",
                    "YAS;Altın Fon;Kıymetli Madenler Şemsiye Fonu;4/7;-;%2,10;%5,00;%8,00;%10,00;%20,00;%40,00",
                    "Toplam;;;;;;;;;;",
                ]
            ),
        encoding="utf-8-sig",
    )

    result = parse_tefas_export(export_path)

    assert result["fund_code"].tolist() == ["AFT", "YAS"]
    assert result.loc[0, "fund_type"] == "Hisse Senedi Şemsiye Fonu"
    assert result.loc[0, "risk_value"] == 5
    assert result.loc[0, "risk_scale"] == 7
    assert result.loc[0, "return_1m_pct"] == 3.1541
    assert pd.isna(result.loc[1, "return_1m_pct"])


def test_parse_tefas_xlsx_export(tmp_path):
    export_path = tmp_path / "tefas_funds.xlsx"
    pd.DataFrame(
        [
            {
                "Fon Kodu": "TI2",
                "Fon Adı": "İş Fon",
                "Şemsiye Fon Türü": "Hisse Senedi Şemsiye Fonu",
                "Fonun Risk Değeri": "6 / 7",
                "1 Ay (%)": "%4,50",
                "6 Ay (%)": "%12,00",
            }
        ]
    ).to_excel(export_path, index=False)

    result = parse_tefas_export(export_path)

    assert result.loc[0, "fund_code"] == "TI2"
    assert result.loc[0, "risk_value"] == 6
    assert result.loc[0, "return_6m_pct"] == 12.0


def test_parse_tefas_xlsx_export_scales_excel_percentage_cells(tmp_path):
    export_path = tmp_path / "tefas_real_shape.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Tablo Verisi"
    sheet.append(["Rapor Bilgileri"])
    sheet.append(["Dışa Aktarım Tarihi:", "08.07.2026 20:52:24"])
    sheet.append(["Toplam Kayıt Sayısı:", 1])
    sheet.append([])
    sheet.append(
        [
            "Fon Kodu",
            "Fon Adı",
            "Şemsiye Fon Türü",
            "Fonun Risk Değeri",
            "1 Ay (%)",
            "3 Ay (%)",
            "6 Ay (%)",
            "Yılbaşından İtibaren (%)",
            "1 Yıl (%)",
            "3 Yıl (%)",
            "5 Yıl (%)",
        ]
    )
    sheet.append(
        [
            "AAL",
            "Ata Fon",
            "Para Piyasası Şemsiye Fonu",
            1,
            0.031541,
            0.086993,
            0.189413,
            0.196823,
            0.463424,
            2.436405,
            3.892674,
        ]
    )
    for row in sheet.iter_rows(min_row=6, min_col=5, max_col=11):
        for cell in row:
            cell.number_format = "0.0000%;-0.0000%;0%"
    workbook.save(export_path)

    result = parse_tefas_export(export_path)

    assert result.loc[0, "fund_code"] == "AAL"
    assert result.loc[0, "return_1m_pct"] == 3.1541
    assert result.loc[0, "return_5y_pct"] == 389.2674
    metadata = extract_tefas_export_metadata(export_path)
    assert metadata["export_generated_at"] == "2026-07-08T20:52:24"
    assert metadata["reported_record_count"] == 1


def test_normalize_slightly_different_column_names_and_extra_rows():
    raw = pd.DataFrame(
        [
            ["junk", "", "", ""],
            ["Fon Kodu", "Fon Unvanı", "Fon Türü", "Risk Değeri"],
            ["AFT", "Ak Fon", "Hisse Senedi Şemsiye Fonu", "5/7"],
            ["", "", "", ""],
        ]
    )

    result = normalize_tefas_export_frame(raw)

    assert result.loc[0, "fund_code"] == "AFT"
    assert result.loc[0, "fund_name"] == "Ak Fon"
    assert result.loc[0, "risk_scale"] == 7


def test_write_generated_universe(tmp_path):
    universe = _sample_universe()
    output = tmp_path / "fund_universe_generated.csv"

    write_generated_universe(universe, output)

    generated = pd.read_csv(output)
    assert list(generated.columns) == [
        "fund_code",
        "fund_name",
        "fund_type",
        "risk_value",
        "return_1m_pct",
        "return_3m_pct",
        "return_6m_pct",
        "return_ytd_pct",
        "return_1y_pct",
        "return_3y_pct",
        "return_5y_pct",
    ]
    assert generated["fund_code"].tolist() == ["AFT", "YAS", "TI2", "ABC"]


def test_filter_export_universe_by_type_risk_returns_include_exclude_and_limit():
    universe = _sample_universe()

    result = filter_export_universe(
        universe,
        fund_type="Hisse Senedi Şemsiye Fonu",
        risk_min=5,
        risk_max=7,
        min_return_1m=0,
        min_return_6m=0,
        include_funds=["YAS"],
        exclude_funds=["ABC"],
        limit=1,
    )

    assert result["fund_code"].tolist() == ["TI2", "YAS"]


def test_tefas_export_mode_does_not_fake_technical_indicators(tmp_path):
    export_path = _write_export_fixture(tmp_path)
    result = run_pipeline(
        data_dir=tmp_path,
        output_path=tmp_path / "export.xlsx",
        report_path=tmp_path / "export.md",
        provider="tefas-export",
        export_file=export_path,
        write_universe=tmp_path / "fund_universe_generated.csv",
    )

    assert result.prices.empty
    assert set(result.analysis["label"]) == {"UNIVERSE ONLY"}
    assert result.analysis["latest_price"].isna().all()
    assert (tmp_path / "fund_universe_generated.csv").exists()
    report = (tmp_path / "export.md").read_text(encoding="utf-8")
    assert "technical indicators were not calculated" in report
    assert "Requested historical funds: 0" in report
    assert "Funds without price history" not in report


def test_tefas_combined_uses_export_metadata_and_mocked_historical_prices(tmp_path, monkeypatch):
    export_path = _write_export_fixture(tmp_path)
    _write_tefas_portfolio(tmp_path)
    payload = _rising_payload("AFT", periods=90)

    class InjectedProvider(TefasDataProvider):
        def __init__(self, *args, **kwargs):
            kwargs["http_client"] = FakeTefasClient({"AFT": payload})
            kwargs["request_delay_seconds"] = 0
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("fund_analyzer.pipeline.TefasDataProvider", InjectedProvider)

    result = run_pipeline(
        data_dir=tmp_path,
        output_path=tmp_path / "combined.xlsx",
        report_path=tmp_path / "combined.md",
        provider="tefas-combined",
        export_file=export_path,
        funds="AFT",
        lookback_months=12,
    )

    assert result.analysis.loc[0, "fund_code"] == "AFT"
    assert result.analysis.loc[0, "risk_value"] == 5
    assert result.analysis.loc[0, "return_1m_pct"] == 3.0
    assert result.sector_summary.loc[0, "fund_type"] == "Hisse Senedi Şemsiye Fonu"
    assert "Requested historical funds: 1" in (tmp_path / "combined.md").read_text(encoding="utf-8")


def test_tefas_combined_filters_export_before_fetching(tmp_path, monkeypatch):
    export_path = _write_export_fixture(tmp_path)
    fetched_codes = []

    class InjectedProvider(TefasDataProvider):
        def __init__(self, *args, **kwargs):
            fetched_codes.extend(kwargs.get("fund_codes") or [])
            kwargs["http_client"] = FakeTefasClient(
                {"TI2": _rising_payload("TI2", periods=90), "AFT": _rising_payload("AFT", periods=90)}
            )
            kwargs["request_delay_seconds"] = 0
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("fund_analyzer.pipeline.TefasDataProvider", InjectedProvider)

    result = run_pipeline(
        data_dir=tmp_path,
        output_path=tmp_path / "combined_filter.xlsx",
        report_path=tmp_path / "combined_filter.md",
        provider="tefas-combined",
        export_file=export_path,
        fund_type="Hisse Senedi Şemsiye Fonu",
        min_return_1m=0,
        min_return_6m=0,
        limit=1,
        lookback_months=12,
    )

    assert fetched_codes == ["TI2"]
    assert result.analysis["fund_code"].tolist() == ["TI2"]


def _sample_universe():
    return pd.DataFrame(
        [
            {
                "fund_code": "AFT",
                "fund_name": "Ak Fon",
                "fund_type": "Hisse Senedi Şemsiye Fonu",
                "risk_value": 5,
                "risk_scale": 7,
                "return_1m_pct": 3.0,
                "return_3m_pct": 5.0,
                "return_6m_pct": -1.0,
                "return_ytd_pct": 6.0,
                "return_1y_pct": 12.0,
                "return_3y_pct": 30.0,
                "return_5y_pct": 60.0,
            },
            {
                "fund_code": "YAS",
                "fund_name": "Altın Fon",
                "fund_type": "Kıymetli Madenler Şemsiye Fonu",
                "risk_value": 4,
                "risk_scale": 7,
                "return_1m_pct": 1.0,
                "return_3m_pct": 2.0,
                "return_6m_pct": 3.0,
                "return_ytd_pct": 4.0,
                "return_1y_pct": 5.0,
                "return_3y_pct": 6.0,
                "return_5y_pct": 7.0,
            },
            {
                "fund_code": "TI2",
                "fund_name": "İş Fon",
                "fund_type": "Hisse Senedi Şemsiye Fonu",
                "risk_value": 6,
                "risk_scale": 7,
                "return_1m_pct": 4.0,
                "return_3m_pct": 5.0,
                "return_6m_pct": 8.0,
                "return_ytd_pct": 9.0,
                "return_1y_pct": 10.0,
                "return_3y_pct": 11.0,
                "return_5y_pct": 12.0,
            },
            {
                "fund_code": "ABC",
                "fund_name": "ABC Fon",
                "fund_type": "Hisse Senedi Şemsiye Fonu",
                "risk_value": 7,
                "risk_scale": 7,
                "return_1m_pct": 2.0,
                "return_3m_pct": 2.0,
                "return_6m_pct": 2.0,
                "return_ytd_pct": 2.0,
                "return_1y_pct": 2.0,
                "return_3y_pct": 2.0,
                "return_5y_pct": 2.0,
            },
        ]
    )


def _write_export_fixture(path: Path) -> Path:
    export_path = path / "tefas_export.csv"
    pd.DataFrame(
        [
            {
                "Fon Kodu": "AFT",
                "Fon Adı": "Ak Fon",
                "Şemsiye Fon Türü": "Hisse Senedi Şemsiye Fonu",
                "Fonun Risk Değeri": "5 / 7",
                "1 Ay (%)": "%3,00",
                "3 Ay (%)": "%5,00",
                "6 Ay (%)": "%-1,00",
            },
            {
                "Fon Kodu": "TI2",
                "Fon Adı": "İş Fon",
                "Şemsiye Fon Türü": "Hisse Senedi Şemsiye Fonu",
                "Fonun Risk Değeri": "6 / 7",
                "1 Ay (%)": "%4,00",
                "3 Ay (%)": "%5,00",
                "6 Ay (%)": "%8,00",
            },
        ]
    ).to_csv(export_path, index=False, encoding="utf-8-sig")
    return export_path


def _write_tefas_portfolio(path: Path) -> None:
    pd.DataFrame(
        [{"fund_code": "AFT", "buy_date": "2026-01-01", "buy_price": 1.0, "units": 10}]
    ).to_csv(path / "portfolio_tefas.csv", index=False)


def _rising_payload(code: str, periods: int):
    dates = pd.bdate_range("2026-01-01", periods=periods)
    return [
        {
            "fonKodu": code,
            "fonUnvan": f"{code} Mock Fund",
            "tarih": date.date().isoformat(),
            "fiyat": 1 + index * 0.01,
        }
        for index, date in enumerate(dates)
    ]

