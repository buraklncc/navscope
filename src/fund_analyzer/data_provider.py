"""Data provider abstractions and CSV implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fund_analyzer.models import LoadedData


FUNDS_COLUMNS = ["fund_code", "fund_name", "fund_type"]
PRICES_COLUMNS = ["fund_code", "date", "price"]
PORTFOLIO_COLUMNS = ["fund_code", "buy_date", "buy_price", "units"]


class DataValidationError(ValueError):
    """Raised when input data does not match the expected schema."""


@dataclass(frozen=True)
class CsvDataProvider:
    """Load fund, price, and portfolio data from CSV files."""

    data_dir: Path
    funds_filename: str = "sample_funds.csv"
    prices_filename: str = "sample_prices.csv"
    portfolio_filename: str = "sample_portfolio.csv"
    portfolio_path: Path | None = None

    def load_all(self) -> LoadedData:
        """Load all CSV inputs and return validation warnings."""

        warnings: list[str] = []
        funds = self.load_funds(warnings=warnings)
        prices = self.load_prices(warnings=warnings)
        portfolio = self.load_portfolio(warnings=warnings)
        portfolio_path = self.portfolio_path or (self.data_dir / self.portfolio_filename)
        return LoadedData(
            funds=funds,
            prices=prices,
            portfolio=portfolio,
            warnings=warnings,
            metadata={
                "portfolio_file_used": str(portfolio_path),
                "portfolio_skipped": False,
            },
        )

    def load_funds(self, warnings: list[str] | None = None) -> pd.DataFrame:
        funds = pd.read_csv(self.data_dir / self.funds_filename)
        return clean_fund_data(funds, warnings=warnings, source_name=self.funds_filename)

    def load_prices(self, warnings: list[str] | None = None) -> pd.DataFrame:
        prices = pd.read_csv(self.data_dir / self.prices_filename)
        return clean_price_data(prices, warnings=warnings, source_name=self.prices_filename)

    def load_portfolio(self, warnings: list[str] | None = None) -> pd.DataFrame:
        portfolio_path = self.portfolio_path or (self.data_dir / self.portfolio_filename)
        portfolio = pd.read_csv(portfolio_path)
        return clean_portfolio_data(
            portfolio,
            warnings=warnings,
            source_name=str(portfolio_path),
        )


def clean_fund_data(
    funds: pd.DataFrame,
    warnings: list[str] | None = None,
    source_name: str = "funds",
) -> pd.DataFrame:
    """Normalize fund metadata and keep one row per fund code."""

    _require_columns(funds, FUNDS_COLUMNS, source_name)
    data = funds[FUNDS_COLUMNS].copy()
    data["fund_code"] = _clean_code_column(data["fund_code"])
    data["fund_name"] = data["fund_name"].astype("string").str.strip()
    data["fund_type"] = data["fund_type"].astype("string").str.strip()

    missing_code = data["fund_code"].isna() | (data["fund_code"] == "")
    if missing_code.any():
        _warn(warnings, f"{source_name}: dropped {int(missing_code.sum())} fund row(s) with missing fund_code.")
        data = data.loc[~missing_code].copy()

    duplicate_mask = data.duplicated("fund_code", keep=False)
    if duplicate_mask.any():
        _warn(
            warnings,
            f"{source_name}: collapsed {int(duplicate_mask.sum())} duplicate fund row(s) by fund_code; kept last row.",
        )
        data = data.drop_duplicates("fund_code", keep="last")

    return data.reset_index(drop=True)


def clean_price_data(
    prices: pd.DataFrame,
    warnings: list[str] | None = None,
    source_name: str = "prices",
) -> pd.DataFrame:
    """Normalize daily prices, sort per fund, and handle unsafe rows."""

    _require_columns(prices, PRICES_COLUMNS, source_name)
    data = prices.copy()
    data["fund_code"] = _clean_code_column(data["fund_code"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    missing_required = (
        data["fund_code"].isna()
        | (data["fund_code"] == "")
        | data["date"].isna()
        | data["price"].isna()
    )
    if missing_required.any():
        _warn(
            warnings,
            f"{source_name}: dropped {int(missing_required.sum())} price row(s) with missing or invalid fund_code/date/price.",
        )
        data = data.loc[~missing_required].copy()

    non_positive_price = data["price"] <= 0
    if non_positive_price.any():
        _warn(
            warnings,
            f"{source_name}: dropped {int(non_positive_price.sum())} price row(s) with non-positive price.",
        )
        data = data.loc[~non_positive_price].copy()

    data["_source_order"] = range(len(data))
    duplicate_mask = data.duplicated(["fund_code", "date"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = data.loc[duplicate_mask]
        duplicate_groups = duplicate_rows[["fund_code", "date"]].drop_duplicates()
        conflicting_groups = duplicate_rows.groupby(["fund_code", "date"])["price"].nunique() > 1
        conflict_count = int(conflicting_groups.sum())
        conflict_text = f" {conflict_count} group(s) had conflicting prices." if conflict_count else ""
        _warn(
            warnings,
            (
                f"{source_name}: collapsed {len(duplicate_rows)} duplicate price row(s) "
                f"across {len(duplicate_groups)} fund/date group(s); kept last row."
                f"{conflict_text}"
            ),
        )
        data = data.sort_values(["fund_code", "date", "_source_order"]).drop_duplicates(
            ["fund_code", "date"],
            keep="last",
        )

    data = data.drop(columns="_source_order")
    return data.sort_values(["fund_code", "date"]).reset_index(drop=True)


def clean_portfolio_data(
    portfolio: pd.DataFrame,
    warnings: list[str] | None = None,
    source_name: str = "portfolio",
) -> pd.DataFrame:
    """Normalize portfolio rows while preserving unknown fund codes for reporting."""

    _require_columns(portfolio, PORTFOLIO_COLUMNS, source_name)
    data = portfolio[PORTFOLIO_COLUMNS].copy()
    data["fund_code"] = _clean_code_column(data["fund_code"])
    data["buy_date"] = pd.to_datetime(data["buy_date"], errors="coerce")
    data["buy_price"] = pd.to_numeric(data["buy_price"], errors="coerce")
    data["units"] = pd.to_numeric(data["units"], errors="coerce")

    missing_code = data["fund_code"].isna() | (data["fund_code"] == "")
    if missing_code.any():
        _warn(
            warnings,
            f"{source_name}: dropped {int(missing_code.sum())} portfolio row(s) with missing fund_code.",
        )
        data = data.loc[~missing_code].copy()

    missing_buy_date = data["buy_date"].isna()
    if missing_buy_date.any():
        _warn(
            warnings,
            f"{source_name}: kept {int(missing_buy_date.sum())} portfolio row(s) with missing or invalid buy_date.",
        )

    invalid_numbers = data["buy_price"].isna() | data["units"].isna()
    if invalid_numbers.any():
        _warn(
            warnings,
            f"{source_name}: dropped {int(invalid_numbers.sum())} portfolio row(s) with missing or invalid buy_price/units.",
        )
        data = data.loc[~invalid_numbers].copy()

    non_positive_numbers = (data["buy_price"] <= 0) | (data["units"] <= 0)
    if non_positive_numbers.any():
        _warn(
            warnings,
            f"{source_name}: dropped {int(non_positive_numbers.sum())} portfolio row(s) with non-positive buy_price/units.",
        )
        data = data.loc[~non_positive_numbers].copy()

    return data.reset_index(drop=True)


def _require_columns(data: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(f"{source_name} is missing required column(s): {missing_text}")


def _clean_code_column(column: pd.Series) -> pd.Series:
    return column.astype("string").str.strip()


def _warn(warnings: list[str] | None, message: str) -> None:
    if warnings is not None:
        warnings.append(message)

