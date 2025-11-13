"""TaxBrain V29 configuration module.

This module wires the "signals → strategies → tiers → routing" stack into a
set of immutable data objects so that runtime behaviour can be steered purely
via configuration. Besides the defaults, :func:`apply_overrides` helps align a
`V29Config` instance with Freqtrade's ``strategy_params`` overrides.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from ..agents.exits.profiles import DEFAULT_PROFILE_VERSION, ExitProfile, resolve_profiles


@dataclass(frozen=True)
class StrategySpec:
    """Bind a set of entry signals to an exit profile + policy thresholds."""

    name: str
    entries: Tuple[str, ...]
    exit_profile: str
    min_rr: float = 0.0
    min_edge: float = 0.0


# Backwards compatibility alias for older imports
StrategyRecipe = StrategySpec


@dataclass(frozen=True)
class TierSpec:
    """Declarative definition of a tier and its guard-rails."""

    name: str
    allowed_recipes: Tuple[str, ...]
    allowed_entries: Tuple[str, ...] = tuple()
    allowed_squads: Tuple[str, ...] = tuple()
    min_raw_score: float = 0.0
    min_rr_ratio: float = 1.0
    min_edge: float = 0.0
    sizing_algo: Literal["BASELINE", "TARGET_RECOVERY"] = "BASELINE"
    k_mult_base_pct: float = 1.0
    recovery_factor: float = 1.0
    cooldown_bars: int = 0
    cooldown_bars_after_win: int = 0
    per_pair_risk_cap_pct: float = 0.02
    max_stake_notional_pct: float = 0.10
    icu_force_exit_bars: int = 0
    priority: int = 100
    default_exit_profile: Optional[str] = None


@dataclass(frozen=True)
class TierRouting:
    """Mapping from CLOSS counts to tier names."""

    loss_tier_map: Dict[int, str] = field(default_factory=dict)

    def resolve(self, closs: int) -> Optional[str]:
        """Return the tier name for a given cumulative loss count."""

        if not self.loss_tier_map:
            return None
        ordered = sorted(self.loss_tier_map.items(), key=lambda kv: kv[0])
        for threshold, tier in ordered:
            if closs <= threshold:
                return tier
        return ordered[-1][1]


DEFAULT_EXIT_PROFILES: Dict[str, ExitProfile] = resolve_profiles()

DEFAULT_STRATEGIES: Dict[str, StrategySpec] = {
    "NBX_fast_default": StrategySpec(
        name="NBX_fast_default",
        entries=("newbars_breakout_long_5m", "newbars_breakdown_short_5m"),
        exit_profile="ATRtrail_v1",
        min_rr=0.1,
        min_edge=0.0,
    ),
    "Recovery_mix": StrategySpec(
        name="Recovery_mix",
        entries=(
            "pullback_long",
            "trend_short",
            "newbars_breakout_long_30m",
            "newbars_breakdown_short_30m",
        ),
        exit_profile="ATRtrail_v1",
        min_rr=0.2,
        min_edge=0.0,
    ),
    "ICU_conservative": StrategySpec(
        name="ICU_conservative",
        entries=("trend_short", "mean_rev_long"),
        exit_profile="ATRtrail_v1",
        min_rr=0.2,
        min_edge=0.0,
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
        allowed_entries=("newbars_breakout_long_5m", "newbars_breakdown_short_5m"),
        allowed_squads=("NBX",),
        min_raw_score=0.20,
        min_rr_ratio=1.2,
        min_edge=0.002,
        sizing_algo="BASELINE",
        k_mult_base_pct=1.0,
        recovery_factor=1.0,
        cooldown_bars=5,
        cooldown_bars_after_win=2,
        per_pair_risk_cap_pct=0.03,
        max_stake_notional_pct=0.15,
        icu_force_exit_bars=0,
        default_exit_profile="ATRtrail_v1",
    ),
    "T12_recovery": TierSpec(
        name="T12_recovery",
        allowed_recipes=("Recovery_mix",),
        allowed_entries=(
            "pullback_long",
            "trend_short",
            "newbars_breakout_long_30m",
            "newbars_breakdown_short_30m",
        ),
        allowed_squads=("PBL", "TRS", "NBX"),
        min_raw_score=0.15,
        min_rr_ratio=1.4,
        min_edge=0.003,
        sizing_algo="TARGET_RECOVERY",
        k_mult_base_pct=1.0,
        recovery_factor=1.5,
        cooldown_bars=10,
        cooldown_bars_after_win=4,
        per_pair_risk_cap_pct=0.02,
        max_stake_notional_pct=0.12,
        icu_force_exit_bars=30,
        default_exit_profile="ATRtrail_v1",
    ),
    "T3p_ICU": TierSpec(
        name="T3p_ICU",
        allowed_recipes=("ICU_conservative",),
        allowed_entries=("trend_short", "mean_rev_long"),
        allowed_squads=("TRS", "MRL"),
        min_raw_score=0.20,
        min_rr_ratio=1.6,
        min_edge=0.004,
        sizing_algo="TARGET_RECOVERY",
        k_mult_base_pct=1.0,
        recovery_factor=2.0,
        cooldown_bars=20,
        cooldown_bars_after_win=6,
        per_pair_risk_cap_pct=0.01,
        max_stake_notional_pct=0.10,
        icu_force_exit_bars=20,
        default_exit_profile="ATRtrail_v1",
    ),
}

DEFAULT_TIER_ROUTING_MAP: Dict[int, str] = {
    0: "T0_healthy",
    1: "T12_recovery",
    2: "T12_recovery",
    3: "T3p_ICU",
}


def _copy_exit_profiles() -> Dict[str, ExitProfile]:
    return dict(DEFAULT_EXIT_PROFILES)


def _copy_strategies() -> Dict[str, StrategySpec]:
    return dict(DEFAULT_STRATEGIES)


def _copy_tiers() -> Dict[str, TierSpec]:
    return dict(DEFAULT_TIERS)


def _default_enabled_signals() -> Tuple[str, ...]:
    return tuple(DEFAULT_ENABLED_SIGNALS)


def _default_tier_routing() -> TierRouting:
    return TierRouting(loss_tier_map=dict(DEFAULT_TIER_ROUTING_MAP))


@dataclass
class V29Config:
    """V29 strategy parameters (signals → strategy → tier binding)."""

    timeframe: str = "5m"
    startup_candle_count: int = 210
    informative_timeframes: tuple[str, ...] = ()

    # Portfolio caps & stress adjustments
    portfolio_cap_pct_base: float = 0.20
    drawdown_threshold_pct: float = 0.15
    # Treasury controls
    treasury_fast_split_pct: float = 0.30
    fast_topK_squads: int = 10
    slow_universe_pct: float = 0.90
    min_injection_nominal_fast: float = 30.0
    min_injection_nominal_slow: float = 7.0
    # Debt / decay
    tax_rate_on_wins: float = 0.20
    pain_decay_per_bar: float = 0.999
    clear_debt_on_profitable_cycle: bool = True
    cycle_len_bars: int = 288
    # Early lock / breakeven guards
    breakeven_lock_frac_of_tp: float = 0.5
    breakeven_lock_eps_atr_pct: float = 0.1

    # Finalize cadence
    force_finalize_mult: float = 1.5
    reservation_ttl_bars: int = 6
    # Indicators
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    atr_len: int = 14
    adx_len: int = 14

    # Behaviour toggles
    suppress_baseline_when_stressed: bool = True

    # Runtime
    dry_run_wallet_fallback: float = 1000.0
    enforce_leverage: float = 1.0
    enabled_signals: Tuple[str, ...] = field(default_factory=_default_enabled_signals)
    exit_profile_version: str = DEFAULT_PROFILE_VERSION
    exit_profiles: Dict[str, ExitProfile] = field(default_factory=_copy_exit_profiles)
    default_exit_profile: Optional[str] = "ATRtrail_v1"
    strategies: Dict[str, StrategySpec] = field(default_factory=_copy_strategies)
    tiers: Dict[str, TierSpec] = field(default_factory=_copy_tiers)
    tier_routing: TierRouting = field(default_factory=_default_tier_routing)
    strategy_recipes_input: InitVar[Optional[Tuple[StrategySpec, ...]]] = None
    _strategy_recipes: Tuple[StrategySpec, ...] = field(init=False, repr=False, default_factory=tuple)

    def __post_init__(self, strategy_recipes_input: Optional[Tuple[StrategySpec, ...]]) -> None:
        version = getattr(self, "exit_profile_version", DEFAULT_PROFILE_VERSION)
        if not self.exit_profiles:
            self.exit_profiles = resolve_profiles(version)
        else:
            self.exit_profiles = dict(self.exit_profiles or {})
        self.strategies = dict(self.strategies or {})
        self.tiers = dict(self.tiers or {})
        if strategy_recipes_input:
            specs = tuple(strategy_recipes_input)
            self._strategy_recipes = specs
            self.strategies = {spec.name: spec for spec in specs}
        elif self._strategy_recipes:
            specs = tuple(self._strategy_recipes)
            self._strategy_recipes = specs
            if not self.strategies:
                self.strategies = {spec.name: spec for spec in specs}
        else:
            specs = tuple(self.strategies.values())
            self._strategy_recipes = specs

        routing = getattr(self, "tier_routing", None)
        referenced_tiers = set(routing.loss_tier_map.values()) if (routing and routing.loss_tier_map) else set()
        for tier_name in referenced_tiers:
            if tier_name not in self.tiers:
                raise ValueError(f"TierRouting references unknown tier '{tier_name}'")
        for tier_name in referenced_tiers:
            spec = self.tiers[tier_name]
            if not spec.default_exit_profile:
                raise ValueError(f"Tier '{tier_name}' must declare default_exit_profile for routing")
            if spec.default_exit_profile not in self.exit_profiles:
                raise ValueError(
                    f"Tier '{tier_name}' references unknown exit profile '{spec.default_exit_profile}'"
                )

    @property
    def strategy_recipes(self) -> Tuple[StrategySpec, ...]:
        return self._strategy_recipes

    @strategy_recipes.setter
    def strategy_recipes(self, recipes: Optional[Tuple[StrategySpec, ...]]) -> None:
        specs = tuple(recipes or ())
        self._strategy_recipes = specs
        self.strategies = {spec.name: spec for spec in specs}


def apply_overrides(cfg: V29Config, strategy_params: Optional[Mapping[str, Any]]) -> V29Config:
    """Apply overrides sourced from Freqtrade ``strategy_params``."""

    if not strategy_params:
        return cfg
    for key, value in strategy_params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def get_exit_profile(cfg: V29Config, name: str) -> ExitProfile:
    """Fetch a named exit profile, raising for unknown profiles."""

    try:
        return cfg.exit_profiles[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown exit profile '{name}'") from exc


def find_strategy_recipe(cfg: V29Config, recipe_name: str) -> Optional[StrategySpec]:
    """Return the requested strategy recipe, if defined."""

    for recipe in cfg.strategy_recipes:
        if recipe.name == recipe_name:
            return recipe
    return None


def entries_to_recipe(cfg: V29Config, entry_kind: str) -> Optional[StrategySpec]:
    """Locate the first recipe that references the given entry signal."""

    for recipe in cfg.strategy_recipes:
        if entry_kind in recipe.entries:
            return recipe
    return None


def get_strategy(cfg: V29Config, name: str) -> StrategySpec:
    """Return a declared strategy by name."""

    recipe = find_strategy_recipe(cfg, name)
    if recipe:
        return recipe
    raise ValueError(f"Unknown strategy recipe '{name}'")


def get_tier_spec(cfg: V29Config, name: str) -> TierSpec:
    """Return a tier specification by name."""

    try:
        return cfg.tiers[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tier '{name}'") from exc


def get_tier_for_closs(cfg: V29Config, closs: int) -> TierSpec:
    """Resolve the tier spec that should serve a given closs value."""

    tier_name = cfg.tier_routing.resolve(closs) if cfg.tier_routing else None
    if not tier_name:
        raise ValueError("Tier routing map is empty; cannot resolve tier for closs.")
    return get_tier_spec(cfg, tier_name)


__all__ = [
    "ExitProfile",
    "StrategySpec",
    "StrategyRecipe",
    "TierSpec",
    "TierRouting",
    "V29Config",
    "apply_overrides",
    "entries_to_recipe",
    "find_strategy_recipe",
    "get_exit_profile",
    "get_strategy",
    "get_tier_for_closs",
    "get_tier_spec",
]
