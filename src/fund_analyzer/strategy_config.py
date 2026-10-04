"""Versioned policy configuration for the category-aware strategy engine."""

from __future__ import annotations

from dataclasses import dataclass

from fund_analyzer.categories import FundCategory


STRATEGY_VERSION = "category-risk-v2.0.0"
MODEL_VERSION = "rules-category-aware-v2"
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RECHECK_BUSINESS_DAYS = 7


@dataclass(frozen=True)
class CategoryPolicy:
    momentum_weight: float
    relative_strength_weight: float
    trend_weight: float
    drawdown_weight: float
    consistency_weight: float
    reliability_weight: float
    friction_weight: float
    minimum_history: int
    buy_history: int
    maximum_annual_volatility: float
    extreme_drawdown: float
    protection_floor_pct: float
    protection_volatility_multiplier: float

    @property
    def weights(self) -> dict[str, float]:
        return {
            "momentum": self.momentum_weight,
            "relative_strength": self.relative_strength_weight,
            "trend": self.trend_weight,
            "drawdown": self.drawdown_weight,
            "consistency": self.consistency_weight,
            "reliability": self.reliability_weight,
            "friction": self.friction_weight,
        }


CATEGORY_POLICIES: dict[FundCategory, CategoryPolicy] = {
    FundCategory.MONEY_MARKET: CategoryPolicy(
        0.14, 0.18, 0.00, 0.20, 0.25, 0.18, 0.05,
        minimum_history=42,
        buy_history=126,
        maximum_annual_volatility=0.045,
        extreme_drawdown=-0.012,
        protection_floor_pct=0.0,
        protection_volatility_multiplier=0.0,
    ),
    FundCategory.SHORT_TERM_DEBT: CategoryPolicy(
        0.15, 0.18, 0.05, 0.20, 0.20, 0.17, 0.05,
        minimum_history=63,
        buy_history=126,
        maximum_annual_volatility=0.10,
        extreme_drawdown=-0.055,
        protection_floor_pct=0.015,
        protection_volatility_multiplier=3.5,
    ),
    FundCategory.LONG_TERM_DEBT: CategoryPolicy(
        0.18, 0.18, 0.08, 0.21, 0.15, 0.15, 0.05,
        minimum_history=126,
        buy_history=252,
        maximum_annual_volatility=0.24,
        extreme_drawdown=-0.18,
        protection_floor_pct=0.04,
        protection_volatility_multiplier=3.2,
    ),
    FundCategory.EQUITY_TURKEY: CategoryPolicy(
        0.25, 0.22, 0.15, 0.15, 0.08, 0.10, 0.05,
        minimum_history=126,
        buy_history=252,
        maximum_annual_volatility=0.58,
        extreme_drawdown=-0.42,
        protection_floor_pct=0.06,
        protection_volatility_multiplier=3.0,
    ),
    FundCategory.FOREIGN_EQUITY: CategoryPolicy(
        0.23, 0.21, 0.13, 0.16, 0.09, 0.13, 0.05,
        minimum_history=126,
        buy_history=252,
        maximum_annual_volatility=0.62,
        extreme_drawdown=-0.45,
        protection_floor_pct=0.07,
        protection_volatility_multiplier=3.2,
    ),
    FundCategory.GOLD_PRECIOUS_METALS: CategoryPolicy(
        0.23, 0.18, 0.15, 0.19, 0.08, 0.12, 0.05,
        minimum_history=126,
        buy_history=252,
        maximum_annual_volatility=0.52,
        extreme_drawdown=-0.36,
        protection_floor_pct=0.08,
        protection_volatility_multiplier=3.5,
    ),
    FundCategory.FX_EUROBOND: CategoryPolicy(
        0.18, 0.19, 0.08, 0.21, 0.13, 0.16, 0.05,
        minimum_history=126,
        buy_history=252,
        maximum_annual_volatility=0.36,
        extreme_drawdown=-0.27,
        protection_floor_pct=0.05,
        protection_volatility_multiplier=3.4,
    ),
    FundCategory.VARIABLE_MIXED: CategoryPolicy(
        0.15, 0.18, 0.05, 0.21, 0.20, 0.16, 0.05,
        minimum_history=126,
        buy_history=252,
        maximum_annual_volatility=0.42,
        extreme_drawdown=-0.31,
        protection_floor_pct=0.05,
        protection_volatility_multiplier=3.2,
    ),
    FundCategory.FUND_OF_FUNDS: CategoryPolicy(
        0.14, 0.18, 0.04, 0.21, 0.21, 0.17, 0.05,
        minimum_history=126,
        buy_history=252,
        maximum_annual_volatility=0.45,
        extreme_drawdown=-0.33,
        protection_floor_pct=0.055,
        protection_volatility_multiplier=3.3,
    ),
    FundCategory.PARTICIPATION_SUKUK: CategoryPolicy(
        0.15, 0.18, 0.05, 0.21, 0.20, 0.16, 0.05,
        minimum_history=63,
        buy_history=126,
        maximum_annual_volatility=0.20,
        extreme_drawdown=-0.14,
        protection_floor_pct=0.03,
        protection_volatility_multiplier=3.3,
    ),
    FundCategory.OTHER_UNKNOWN: CategoryPolicy(
        0.10, 0.14, 0.03, 0.22, 0.20, 0.26, 0.05,
        minimum_history=126,
        buy_history=378,
        maximum_annual_volatility=0.35,
        extreme_drawdown=-0.25,
        protection_floor_pct=0.06,
        protection_volatility_multiplier=3.6,
    ),
}


DISCOVERY_LABELS = (
    "AL",
    "KADEMELİ AL",
    "BEKLE",
    "TUT",
    "KÂRI KORU",
    "UZAK DUR",
    "VERİ EKSİK",
    "VERİ GÜVENSİZ",
)

PORTFOLIO_LABELS = (
    "TUT",
    "ALIM ARTIRILABİLİR",
    "KÂRI KORU",
    "SAT",
    "ACİL SAT DEĞİL AMA RİSK ARTTI",
    "UZAK DUR / YENİ ALIM YAPMA",
    "STOP'A YAKIN",
    "STOP KIRILDI",
    "VERİ EKSİK",
    "VERİ GÜVENSİZ",
)


def policy_for(category: FundCategory) -> CategoryPolicy:
    return CATEGORY_POLICIES.get(category, CATEGORY_POLICIES[FundCategory.OTHER_UNKNOWN])


def validate_policy_weights() -> None:
    for category, policy in CATEGORY_POLICIES.items():
        total = sum(policy.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise RuntimeError(f"{category.value} strategy weights sum to {total}, not 1.0")


validate_policy_weights()

