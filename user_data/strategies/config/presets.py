# -*- coding: utf-8 -*-
"""Default preset bundles for TaxBrain V30."""

from __future__ import annotations

from typing import Dict, Tuple

from ..agents.exits.profiles import ExitProfile
from .models import StrategySpec, TierSpec

# Inline default profile version tag (metadata only for inline defaults).
DEFAULT_PROFILE_VERSION = "inline_v1"


def default_profiles_factory() -> Dict[str, ExitProfile]:
    """Provide a minimal inline exit profile library used when config omits profiles."""

    return {
        "ATRtrail_v1": ExitProfile(
            atr_timeframe=None,  # Use primary timeframe ATR
            atr_mul_sl=4,  # Stop at 8x ATR; raise to widen stops
            floor_sl_pct=1e-12,  # Absolute SL floor to avoid zero
            atr_mul_tp=4,  # TP at 2x ATR; raise to target farther profits
            breakeven_lock_frac_of_tp=0.0,  # Fraction of TP before breakeven lock; >0 adds protection
            trail_mode=None,
            trail_atr_mul=0.0,
            activation_atr_mul=0.0,
            max_bars_in_trade=0,
        )
    }


DEFAULT_EXIT_PROFILES: Dict[str, ExitProfile] = default_profiles_factory()

DEFAULT_STRATEGIES: Dict[str, StrategySpec] = {
    "NBX_fast_default": StrategySpec(
        name="NBX_fast_default",
        entries=("newbars_breakout_long_5m", "newbars_breakdown_short_5m"),
        exit_profile="ATRtrail_v1",
        min_rr=0.00001,
        min_edge=0.0,
        base_win_prob=0.40,  # breakout/trend bias
    ),
    "Recovery_mix": StrategySpec(
        name="Recovery_mix",
        entries=(
            "pullback_long",
            "trend_short",
            "newbars_breakdown_short_30m",
        ),
        exit_profile="ATRtrail_v1",
        min_rr=0.2,
        min_edge=0.0,
        base_win_prob=0.55,  # mixed bag; slightly above neutral
    ),
    "ICU_conservative": StrategySpec(
        name="ICU_conservative",
        entries=("trend_short", "mean_rev_long", "newbars_breakout_long_30m"),
        exit_profile="ATRtrail_v1",
        min_rr=0.2,
        min_edge=0.0,
        base_win_prob=0.45,  # conservative trend/mean-rev blend
    ),
}

DEFAULT_ENABLED_SIGNALS: Tuple[str, ...] = (
    "newbars_breakout_long_5m",
    "newbars_breakdown_short_5m",
    "newbars_breakout_long_30m",
    "newbars_breakdown_short_30m",
    "pullback_long",
    "trend_short",
    "mean_rev_long",
)

DEFAULT_TIERS: Dict[str, TierSpec] = {
    "T0_healthy": TierSpec(
        name="T0_healthy",
        allowed_recipes=("NBX_fast_default",),
        min_raw_score=0.20,
        min_rr_ratio=0.00001,
        min_edge=0.002,
        sizing_algo="BASE_ONLY",
        center_algo="TARGET_RECOVERY",
        k_mult_base_pct=0.0,
        recovery_factor=1.0,
        cooldown_bars=0,
        cooldown_bars_after_win=0,
        per_pair_risk_cap_pct=1,
        max_stake_notional_pct=0.5,
        icu_force_exit_bars=0,
        default_exit_profile="ATRtrail_v1",
        single_position_only=True,
    ),
    "T12_recovery": TierSpec(
        name="T12_recovery",
        allowed_recipes=("Recovery_mix",),
        min_raw_score=0.20,
        min_rr_ratio=0.000001,
        min_edge=0.000,
        sizing_algo="TARGET_RECOVERY",
        center_algo="TARGET_RECOVERY",
        k_mult_base_pct=1.0,
        recovery_factor=1,
        cooldown_bars=0,
        cooldown_bars_after_win=0,
        per_pair_risk_cap_pct=1,
        max_stake_notional_pct=1,
        icu_force_exit_bars=0,
        default_exit_profile="ATRtrail_v1",
        single_position_only=True,
    ),
    "T3p_ICU": TierSpec(
        name="T3p_ICU",
        allowed_recipes=("ICU_conservative",),
        min_raw_score=0.20,
        min_rr_ratio=0.000001,
        min_edge=0.000,
        sizing_algo="TARGET_RECOVERY",
        center_algo="TARGET_RECOVERY",
        k_mult_base_pct=1.0,
        recovery_factor=1,
        cooldown_bars=0,
        cooldown_bars_after_win=0,
        per_pair_risk_cap_pct=1,
        max_stake_notional_pct=1,
        icu_force_exit_bars=0,
        default_exit_profile="ATRtrail_v1",
        single_position_only=True,
    ),
}

DEFAULT_TIER_ROUTING_MAP: Dict[int, str] = {
    0: "T0_healthy",
    1: "T12_recovery",
    2: "T12_recovery",
    3: "T3p_ICU",
    4: "T3p_ICU",
    5: "T3p_ICU",
    6: "T3p_ICU",
}


__all__ = [
    "DEFAULT_PROFILE_VERSION",
    "DEFAULT_EXIT_PROFILES",
    "DEFAULT_STRATEGIES",
    "DEFAULT_ENABLED_SIGNALS",
    "DEFAULT_TIERS",
    "DEFAULT_TIER_ROUTING_MAP",
    "default_profiles_factory",
]
