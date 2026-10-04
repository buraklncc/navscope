"""TEFAS data provider adapter.

This module talks to the current public tefas.gov.tr JSON endpoints and
normalizes the response into the same internal DataFrame schemas used by the
CSV provider. The analysis engine does not depend on TEFAS-specific fields.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib import error, request
import ssl

import pandas as pd

from fund_analyzer import config
from fund_analyzer.cache import JsonFileCache, tefas_price_cache_key
from fund_analyzer.data_provider import (
    PORTFOLIO_COLUMNS,
    clean_fund_data,
    clean_price_data,
    clean_portfolio_data,
)
from fund_analyzer.models import LoadedData


class TefasProviderError(RuntimeError):
    """Raised for unrecoverable TEFAS provider configuration errors."""


class TefasHttpError(RuntimeError):
    """Raised when a TEFAS HTTP call fails."""


@dataclass
class UrlLibTefasHttpClient:
    """Minimal JSON POST client with TEFAS-friendly headers."""

    base_url: str = config.TEFAS_BASE_URL
    timeout_seconds: int = config.TEFAS_TIMEOUT_SECONDS
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
        }
    )

    def post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(url, data=body, headers=self.headers, method="POST")
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = None
            
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds, context=context) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raise TefasHttpError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
        except error.URLError as exc:
            raise TefasHttpError(f"Network error from {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TefasHttpError(f"Timeout after {self.timeout_seconds}s from {url}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise TefasHttpError(f"Invalid JSON from {url}") from exc
        if not isinstance(parsed, dict):
            raise TefasHttpError(f"Unexpected JSON shape from {url}")
        return parsed


@dataclass
class TefasDataProvider:
    """Fetch selected TEFAS funds and normalize them for the analysis engine."""

    data_dir: Path
    fund_codes: list[str] | None = None
    fund_file: Path | None = None
    lookback_months: int = 12
    refresh: bool = False
    portfolio_file: Path | None = None
    request_delay_seconds: float = config.TEFAS_REQUEST_DELAY_SECONDS
    fund_type: str = config.TEFAS_DEFAULT_FUND_TYPE
    http_client: Any | None = None
    cache: JsonFileCache | None = None
    universe_override: pd.DataFrame | None = None
    continue_on_error: bool = False
    transient_retry_count: int = 1
    resume: bool = False
    batch_size: int | None = None
    fetch_date: date = field(default_factory=date.today)
    cache_retention_days: int = 3

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.fund_file is not None:
            self.fund_file = Path(self.fund_file)
        if self.portfolio_file is not None:
            self.portfolio_file = Path(self.portfolio_file)
        if self.http_client is None:
            self.http_client = UrlLibTefasHttpClient()
        if self.cache is None:
            self.cache = JsonFileCache(self.data_dir / "cache" / "tefas")
        if hasattr(self.cache, "cleanup"):
            self.cache.cleanup(self.cache_retention_days)
        self.lookback_months = validate_lookback_months(self.lookback_months)

    def load_all(self) -> LoadedData:
        warnings: list[str] = []
        selected_metadata = self._resolve_requested_funds(warnings)
        universe = self.universe_override.copy() if self.universe_override is not None else _load_local_universe(self.data_dir, warnings)
        selected_metadata = _enrich_selected_funds(selected_metadata, universe)
        requested_codes = selected_metadata["fund_code"].tolist()

        raw_price_rows: list[dict[str, Any]] = []
        fund_names: dict[str, str] = {
            row["fund_code"]: row.get("fund_name") or row["fund_code"]
            for row in selected_metadata.to_dict("records")
        }
        failed_funds: list[dict[str, str]] = []
        empty_response_funds: list[dict[str, str]] = []
        skipped_cached_fund_codes: list[str] = []
        cache_hits = 0
        cache_misses = 0
        network_attempts = 0
        network_successes = 0
        network_failures = 0
        retry_attempts = 0
        started_at = time.time()

        for index, fund_code in enumerate(requested_codes):
            if self.batch_size and index > 0 and index % self.batch_size == 0:
                print(f"TEFAS Fetch: Processed {index}/{len(requested_codes)} funds...")

            cache_key = tefas_price_cache_key(fund_code, self.lookback_months, self.fetch_date)
            hit_cache = False
            raw_rows = []
            
            if not self.refresh:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    raw_rows = _cached_result_list(cached)
                    hit_cache = True

            if hit_cache:
                cache_hits += 1
                if self.resume:
                    skipped_cached_fund_codes.append(fund_code)
            else:
                cache_misses += 1
                network_attempts += 1
                if network_attempts > 1 and self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)
                
                payload = {"fonKodu": fund_code, "dil": "TR", "periyod": self.lookback_months}
                last_error: TefasHttpError | None = None
                for retry_index in range(max(0, int(self.transient_retry_count)) + 1):
                    try:
                        body = self.http_client.post_json(config.TEFAS_PRICE_ENDPOINT, payload)
                        raw_rows = _result_list(body)
                        self.cache.set(cache_key, {"payload": payload, "resultList": raw_rows})
                        network_successes += 1
                        last_error = None
                        break
                    except TefasHttpError as exc:
                        last_error = exc
                        if retry_index < max(0, int(self.transient_retry_count)):
                            retry_attempts += 1
                            time.sleep(min(max(self.request_delay_seconds, 0.1), 0.5))
                if last_error is not None:
                    network_failures += 1
                    if not self.continue_on_error:
                        raise last_error
                    warnings.append(f"TEFAS {fund_code}: {last_error}")
                    failed_funds.append({"fund_code": fund_code, "error_reason": str(last_error)})
                    continue

            if not raw_rows:
                warnings.append(f"TEFAS {fund_code}: empty response or invalid fund code.")
                empty_response_funds.append({"fund_code": fund_code, "error_reason": "Empty response or invalid fund code."})
                continue

            parsed_rows = [_normalize_price_row(row, fallback_code=fund_code) for row in raw_rows]
            raw_price_rows.extend(parsed_rows)
            title = _first_non_empty(row.get("fund_name") for row in parsed_rows)
            if title and _should_use_tefas_title(fund_names.get(fund_code), fund_code):
                fund_names[fund_code] = title

        raw_prices = pd.DataFrame(raw_price_rows, columns=["fund_code", "date", "price", "fund_name"])
        prices = clean_price_data(raw_prices[["fund_code", "date", "price"]], warnings=warnings, source_name="tefas prices")

        successful_codes = sorted(set(prices["fund_code"])) if not prices.empty else []
        funds = clean_fund_data(
            pd.DataFrame(
                [
                    {
                        "fund_code": code,
                        "fund_name": fund_names.get(code) or code,
                        "fund_type": _fund_type_for_code(selected_metadata, code),
                    }
                    for code in requested_codes
                ]
            ),
            warnings=warnings,
            source_name="tefas funds",
        )

        funds_missing_universe_mapping = _funds_missing_universe_mapping(requested_codes, universe)
        funds_with_unknown_fund_type = _funds_with_unknown_fund_type(funds)
        if funds_missing_universe_mapping:
            warnings.append(
                "Funds missing universe mapping: " + ", ".join(funds_missing_universe_mapping)
            )
        if funds_with_unknown_fund_type:
            warnings.append(
                "Funds with Unknown fund_type: " + ", ".join(funds_with_unknown_fund_type)
            )

        portfolio, portfolio_metadata = self._load_portfolio(warnings)
        finished_at = time.time()
        metadata = {
            "requested_funds": len(requested_codes),
            "successfully_fetched_funds": len(successful_codes),
            "failed_funds": failed_funds,
            "empty_response_funds": empty_response_funds,
            "skipped_cached_fund_codes": skipped_cached_fund_codes,
            "failed_fund_codes": sorted(set(f["fund_code"] for f in failed_funds) | set(f["fund_code"] for f in empty_response_funds)),
            "funds_missing_universe_mapping": funds_missing_universe_mapping,
            "funds_with_unknown_fund_type": funds_with_unknown_fund_type,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "network_attempts": network_attempts,
            "network_successes": network_successes,
            "network_failures": network_failures,
            "retry_attempts": retry_attempts,
            "duration_seconds": round(finished_at - started_at, 2),
            "lookback_months": self.lookback_months,
            "latest_price_date": _latest_price_date(prices),
            **portfolio_metadata,
        }
        return LoadedData(
            funds=funds,
            prices=prices,
            portfolio=portfolio,
            warnings=warnings,
            metadata=metadata,
        )

    def _resolve_requested_funds(self, warnings: list[str]) -> pd.DataFrame:
        if self.fund_codes:
            return _metadata_from_codes(self.fund_codes)

        if self.fund_file:
            fund_file = _resolve_user_path(self.fund_file, self.data_dir)
            return _load_fund_file(fund_file, warnings)

        selected_funds = self.data_dir / "selected_funds.csv"
        if selected_funds.exists():
            return _load_fund_file(selected_funds, warnings)

        raise TefasProviderError(
            "No TEFAS funds selected. Pass --funds, pass --fund-file, or create data/selected_funds.csv."
        )

    def _load_portfolio(self, warnings: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
        if self.portfolio_file:
            portfolio_path = _resolve_user_path(self.portfolio_file, self.data_dir)
            if not portfolio_path.exists():
                raise TefasProviderError(f"Portfolio file does not exist: {portfolio_path}")
            portfolio = pd.read_csv(portfolio_path)
            return (
                clean_portfolio_data(portfolio, warnings=warnings, source_name=str(portfolio_path)),
                {
                    "portfolio_file_used": str(portfolio_path),
                    "portfolio_skipped": False,
                },
            )

        default_tefas_portfolio = self.data_dir / "portfolio_tefas.csv"
        if default_tefas_portfolio.exists():
            portfolio = pd.read_csv(default_tefas_portfolio)
            return (
                clean_portfolio_data(portfolio, warnings=warnings, source_name=str(default_tefas_portfolio)),
                {
                    "portfolio_file_used": str(default_tefas_portfolio),
                    "portfolio_skipped": False,
                },
            )

        warnings.append("TEFAS portfolio analysis skipped: no portfolio_tefas.csv or --portfolio-file provided.")
        return (
            pd.DataFrame(columns=PORTFOLIO_COLUMNS),
            {
                "portfolio_file_used": "None",
                "portfolio_skipped": True,
            },
        )

    def _load_fund_universe_from_tefas(self, warnings: list[str]) -> pd.DataFrame:
        payload = {
            "dil": "TR",
            "fonTipi": self.fund_type,
            "kurucuKodu": None,
            "sfonTurKod": None,
            "fonTurAciklama": None,
            "islem": 1,
            "fonTurKod": None,
            "fonGrubu": None,
            "donemGetiri1a": "1",
            "donemGetiri3a": "1",
            "donemGetiri6a": "1",
            "donemGetiri1y": "1",
            "donemGetiriyb": "1",
            "donemGetiri3y": "1",
            "donemGetiri5y": "1",
            "basTarih": None,
            "bitTarih": None,
            "calismaTipi": 2,
            "getiriOrani": "1",
        }
        body = self.http_client.post_json(config.TEFAS_FUND_LIST_ENDPOINT, payload)
        rows = _result_list(body)
        if not rows:
            raise TefasProviderError(
                "TEFAS fund universe response was empty. Pass --funds or --fund-file."
            )

        universe = pd.DataFrame(
            [
                {
                    "fund_code": _clean_code(row.get("fonKodu")),
                    "fund_name": row.get("fonUnvan") or row.get("fonAdi") or row.get("fonKodu"),
                    "fund_type": row.get("fonTurAciklama") or config.TEFAS_UNKNOWN_FUND_TYPE,
                }
                for row in rows
                if row.get("fonKodu")
            ]
        )
        warnings.append(
            f"TEFAS fund universe loaded from live endpoint: {len(universe)} fund(s)."
        )
        return clean_fund_data(universe, warnings=warnings, source_name="tefas fund universe")




def validate_lookback_months(lookback_months: int) -> int:
    """Validate TEFAS fixed lookback enum values."""

    try:
        value = int(lookback_months)
    except (TypeError, ValueError) as exc:
        raise TefasProviderError("lookback_months must be an integer.") from exc
    if value not in config.TEFAS_SUPPORTED_LOOKBACK_MONTHS:
        supported = ", ".join(str(month) for month in config.TEFAS_SUPPORTED_LOOKBACK_MONTHS)
        raise TefasProviderError(f"Unsupported TEFAS lookback months: {value}. Supported values: {supported}.")
    return value


def parse_fund_codes(value: str | None) -> list[str] | None:
    """Parse comma-separated fund codes from CLI input."""

    if not value:
        return None
    codes = [_clean_code(code) for code in value.split(",")]
    return [code for code in codes if code]


def _load_fund_file(
    path: Path,
    warnings: list[str],
    warn_missing_fund_type: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise TefasProviderError(f"Fund file does not exist: {path}")
    data = pd.read_csv(path)
    if "fund_code" not in data.columns:
        raise TefasProviderError(f"Fund file must contain a fund_code column: {path}")
    if "fund_name" not in data.columns:
        data["fund_name"] = data["fund_code"]
    if "fund_type" not in data.columns:
        data["fund_type"] = config.TEFAS_UNKNOWN_FUND_TYPE
        if warn_missing_fund_type:
            warnings.append(f"{path}: fund_type column missing; using Unknown.")
    return clean_fund_data(data[["fund_code", "fund_name", "fund_type"]], warnings=warnings, source_name=str(path))


def _resolve_user_path(path: Path, data_dir: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return data_dir / path


def _load_local_universe(data_dir: Path, warnings: list[str]) -> pd.DataFrame:
    universe_path = data_dir / "fund_universe.csv"
    if not universe_path.exists():
        warnings.append(f"{universe_path}: fund universe mapping not found; fund_type defaults to Unknown.")
        return pd.DataFrame(columns=["fund_code", "fund_name", "fund_type"])
    return _load_fund_file(universe_path, warnings, warn_missing_fund_type=True)


def _enrich_selected_funds(selected: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return selected.copy()

    enriched = selected.merge(
        universe.rename(
            columns={
                "fund_name": "universe_fund_name",
                "fund_type": "universe_fund_type",
            }
        ),
        on="fund_code",
        how="left",
    )
    enriched["fund_name"] = enriched.apply(
        lambda row: _choose_enriched_value(
            row.get("fund_name"),
            row.get("universe_fund_name"),
            row.get("fund_code"),
        ),
        axis=1,
    )
    enriched["fund_type"] = enriched.apply(
        lambda row: _choose_enriched_value(
            row.get("fund_type"),
            row.get("universe_fund_type"),
            config.TEFAS_UNKNOWN_FUND_TYPE,
        ),
        axis=1,
    )
    return enriched[["fund_code", "fund_name", "fund_type"]]


def _metadata_from_codes(codes: list[str]) -> pd.DataFrame:
    unique_codes = list(dict.fromkeys(_clean_code(code) for code in codes if _clean_code(code)))
    return pd.DataFrame(
        [
            {
                "fund_code": code,
                "fund_name": code,
                "fund_type": config.TEFAS_UNKNOWN_FUND_TYPE,
            }
            for code in unique_codes
        ]
    )


def _choose_enriched_value(current: Any, mapped: Any, fallback: Any) -> Any:
    if _has_real_value(mapped) and not _is_unknown(mapped):
        return mapped
    if _has_real_value(current):
        return current
    return fallback


def _normalize_price_row(row: dict[str, Any], fallback_code: str) -> dict[str, Any]:
    return {
        "fund_code": _clean_code(row.get("fonKodu") or row.get("code") or fallback_code),
        "date": row.get("tarih") or row.get("date"),
        "price": _parse_turkish_number(row.get("fiyat") or row.get("price")),
        "fund_name": row.get("fonUnvan") or row.get("title"),
    }


def _parse_turkish_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _result_list(body: dict[str, Any]) -> list[dict[str, Any]]:
    rows = body.get("resultList") or body.get("data") or body.get("result") or []
    if isinstance(rows, dict):
        rows = rows.get("resultList") or rows.get("items") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _cached_result_list(cached: Any) -> list[dict[str, Any]]:
    if isinstance(cached, list):
        return [row for row in cached if isinstance(row, dict)]
    if isinstance(cached, dict):
        return _result_list(cached)
    return []


def _clean_code(value: Any) -> str:
    return "" if value is None else str(value).strip().upper()


def _first_non_empty(values) -> str | None:
    for value in values:
        if value:
            return str(value)
    return None


def _should_use_tefas_title(current_name: str | None, fund_code: str) -> bool:
    if not _has_real_value(current_name):
        return True
    return str(current_name).strip().upper() == fund_code.upper()


def _fund_type_for_code(funds: pd.DataFrame, code: str) -> str:
    matches = funds.loc[funds["fund_code"] == code, "fund_type"]
    if matches.empty or pd.isna(matches.iloc[0]) or matches.iloc[0] == "":
        return config.TEFAS_UNKNOWN_FUND_TYPE
    return str(matches.iloc[0])


def _funds_missing_universe_mapping(codes: list[str], universe: pd.DataFrame) -> list[str]:
    if universe.empty:
        return sorted(set(codes))
    mapped_codes = set(universe["fund_code"])
    return sorted(set(codes) - mapped_codes)


def _funds_with_unknown_fund_type(funds: pd.DataFrame) -> list[str]:
    if funds.empty:
        return []
    unknown = funds["fund_type"].isna() | (funds["fund_type"].astype(str).str.strip() == "") | (
        funds["fund_type"].astype(str).str.strip().str.lower() == config.TEFAS_UNKNOWN_FUND_TYPE.lower()
    )
    return sorted(funds.loc[unknown, "fund_code"].tolist())


def _has_real_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _is_unknown(value: Any) -> bool:
    return str(value).strip().lower() == config.TEFAS_UNKNOWN_FUND_TYPE.lower()


def _latest_price_date(prices: pd.DataFrame) -> str | None:
    if prices.empty:
        return None
    latest = prices["date"].max()
    if pd.isna(latest):
        return None
    return pd.Timestamp(latest).date().isoformat()

