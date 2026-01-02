# -*- coding: utf-8 -*-
"""Compatibility wrapper for TaxBrain V30 configuration APIs."""

from __future__ import annotations

from .loader import apply_overrides, build_v30_config
from .models import (
    ExitProfile,
    GatekeepingConfig,
    RiskConfig,
    SizingAlgoConfig,
    SizingConfig,
    StrategyConfig,
    StrategyRecipe,
    StrategySpec,
    SystemConfig,
    TargetRecoveryConfig,
    TierRouting,
    TierSpec,
    TradingConfig,
    TreasuryConfig,
    V30Config,
)
from .presets import (
    DEFAULT_ENABLED_SIGNALS,
    DEFAULT_EXIT_PROFILES,
    DEFAULT_PROFILE_VERSION,
    DEFAULT_STRATEGIES,
    DEFAULT_TIERS,
    DEFAULT_TIER_ROUTING_MAP,
    default_profiles_factory,
)
from .services import (
    ConfigService,
    entries_to_recipe,
    find_strategy_recipe,
    get_exit_profile,
    get_strategy,
    get_tier_for_closs,
    get_tier_spec,
)

__all__ = [
    "ExitProfile",
    "StrategySpec",
    "StrategyRecipe",
    "TierSpec",
    "TierRouting",
    "SizingConfig",
    "TreasuryConfig",
    "SystemConfig",
    "RiskConfig",
    "TradingConfig",
    "StrategyConfig",
    "GatekeepingConfig",
    "TargetRecoveryConfig",
    "SizingAlgoConfig",
    "V30Config",
    "DEFAULT_PROFILE_VERSION",
    "DEFAULT_EXIT_PROFILES",
    "DEFAULT_STRATEGIES",
    "DEFAULT_ENABLED_SIGNALS",
    "DEFAULT_TIERS",
    "DEFAULT_TIER_ROUTING_MAP",
    "default_profiles_factory",
    "apply_overrides",
    "build_v30_config",
    "ConfigService",
    "entries_to_recipe",
    "find_strategy_recipe",
    "get_exit_profile",
    "get_strategy",
    "get_tier_for_closs",
    "get_tier_spec",
]
