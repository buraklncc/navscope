"""Technical indicator calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fund_analyzer import config
from fund_analyzer.data_provider import clean_price_data


INDICATOR_COLUMNS = ["ema10", "ema21", "macd", "macd_signal", "macd_histogram", "rsi_14", "atr_like"]


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate an exponential moving average."""

    return prices.astype(float).ewm(span=period, adjust=False).mean()


def calculate_macd(
    prices: pd.Series,
    fast_period: int = config.MACD_FAST_PERIOD,
    slow_period: int = config.MACD_SLOW_PERIOD,
    signal_period: int = config.MACD_SIGNAL_PERIOD,
) -> pd.DataFrame:
    """Calculate MACD, MACD signal, and MACD histogram."""

    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    macd = fast_ema - slow_ema
    macd_signal = calculate_ema(macd, signal_period)
    histogram = macd - macd_signal
    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": histogram,
        },
        index=prices.index,
    )


def calculate_rsi(prices: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Calculate Wilder RSI without lookahead."""

    price_series = prices.astype(float)
    delta = price_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    rsi = pd.Series(np.nan, index=price_series.index, dtype=float)
    if len(price_series) <= period:
        return rsi

    first_gain = gains.iloc[1 : period + 1].mean()
    first_loss = losses.iloc[1 : period + 1].mean()

    gains_smma = gains.copy()
    gains_smma.iloc[: period + 1] = np.nan
    gains_smma.iloc[period] = first_gain

    losses_smma = losses.copy()
    losses_smma.iloc[: period + 1] = np.nan
    losses_smma.iloc[period] = first_loss

    avg_gains = gains_smma.ewm(alpha=1 / period, adjust=False, ignore_na=True).mean()
    avg_losses = losses_smma.ewm(alpha=1 / period, adjust=False, ignore_na=True).mean()

    rs = avg_gains / avg_losses
    rsi_vec = 100 - (100 / (1 + rs))

    rsi_vec = rsi_vec.fillna(np.nan)
    
    # Handle edge cases (zero loss, zero gain, or both zero)
    rsi_vec = rsi_vec.where(~((avg_gains == 0) & (avg_losses == 0)), 50.0)
    rsi_vec = rsi_vec.where(~((avg_losses == 0) & (avg_gains > 0)), 100.0)
    rsi_vec = rsi_vec.where(~((avg_gains == 0) & (avg_losses > 0)), 0.0)

    # Prefix edge NaNs correctly as per the original loop structure which leaves first `period` rows as NaN
    rsi_vec.iloc[:period] = np.nan
    
    return rsi_vec


def calculate_atr_like(prices: pd.Series, period: int = config.ATR_LIKE_PERIOD) -> pd.Series:
    """Calculate close-to-close ATR-like volatility for close-only data."""

    absolute_change = prices.astype(float).diff().abs().fillna(0)
    return absolute_change.rolling(window=period, min_periods=1).mean()


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, MACD, RSI, and ATR-like columns to a price table."""

    required = {"fund_code", "date", "price"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required column(s): {', '.join(sorted(missing))}")

    prices = clean_price_data(prices, source_name="prices")
    if prices.empty:
        return _with_empty_indicator_columns(prices)

    groups: list[pd.DataFrame] = []
    for _, group in prices.groupby("fund_code", sort=False):
        enriched = group.copy()
        price_series = enriched["price"]
        enriched["ema10"] = calculate_ema(price_series, config.EMA_FAST_PERIOD)
        enriched["ema21"] = calculate_ema(price_series, config.EMA_SLOW_PERIOD)
        macd = calculate_macd(price_series)
        enriched[["macd", "macd_signal", "macd_histogram"]] = macd[
            ["macd", "macd_signal", "macd_histogram"]
        ]
        enriched["rsi_14"] = calculate_rsi(price_series, config.RSI_PERIOD)
        enriched["atr_like"] = calculate_atr_like(price_series, config.ATR_LIKE_PERIOD)
        groups.append(enriched)

    return pd.concat(groups, ignore_index=True)





def _with_empty_indicator_columns(prices: pd.DataFrame) -> pd.DataFrame:
    enriched = prices.copy()
    for column in INDICATOR_COLUMNS:
        enriched[column] = pd.Series(dtype=float)
    return enriched

