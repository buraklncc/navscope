import pandas as pd

from fund_analyzer.sector_summary import build_sector_summary


def test_sector_net_balance_calculation():
    analysis = pd.DataFrame(
        [
            {"fund_code": "AAA", "fund_type": "Equity", "label": "STRONG BUY"},
            {"fund_code": "BBB", "fund_type": "Equity", "label": "EARLY BUY"},
            {"fund_code": "CCC", "fund_type": "Equity", "label": "HOLD"},
            {"fund_code": "DDD", "fund_type": "Equity", "label": "HOLD RSI HIGH"},
            {"fund_code": "EEE", "fund_type": "Equity", "label": "SELL"},
            {"fund_code": "FFF", "fund_type": "Bond", "label": "WAIT"},
        ]
    )

    summary = build_sector_summary(analysis).set_index("fund_type")

    assert summary.loc["Equity", "TOTAL"] == 5
    assert summary.loc["Equity", "net_balance"] == -1
    assert summary.loc["Bond", "WAIT"] == 1
    assert summary.loc["Bond", "net_balance"] == 0

