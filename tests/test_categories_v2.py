import pytest

from fund_analyzer.categories import classify_fund


@pytest.mark.parametrize(
    ("fund_name", "fund_type", "expected"),
    [
        ("Örnek Para Piyasası Fonu", "Para Piyasası Şemsiye Fonu", "MONEY_MARKET"),
        ("Kısa Vadeli Borçlanma Araçları Fonu", "Borçlanma Araçları", "SHORT_TERM_DEBT"),
        ("Uzun Vadeli Borçlanma Araçları Fonu", "Borçlanma Araçları", "LONG_TERM_DEBT"),
        ("Türkiye Hisse Senedi Fonu", "Hisse Senedi", "EQUITY_TURKEY"),
        ("Yabancı Teknoloji Hisse Senedi Fonu", "Hisse Senedi", "FOREIGN_EQUITY"),
        ("Altın Fonu", "Kıymetli Madenler", "GOLD_PRECIOUS_METALS"),
        ("Eurobond Fonu", "Yabancı Borçlanma Araçları", "FX_EUROBOND"),
        ("Değişken Fon", "Değişken", "VARIABLE_MIXED"),
        ("Fon Sepeti Fonu", "Fon Sepeti", "FUND_OF_FUNDS"),
        ("Kira Sertifikaları Katılım Fonu", "Katılım", "PARTICIPATION_SUKUK"),
    ],
)
def test_category_classifier_covers_primary_tefas_fund_types(fund_name, fund_type, expected):
    result = classify_fund({"fund_name": fund_name, "fund_type": fund_type})

    assert result.category.value == expected
    assert result.confidence >= 0.70


def test_unclassified_metadata_keeps_explicit_low_confidence():
    result = classify_fund({"fund_name": "Belirsiz Strateji", "fund_type": "Unknown"})

    assert result.category.value == "OTHER_UNKNOWN"
    assert result.confidence < 0.55
    assert result.warnings

