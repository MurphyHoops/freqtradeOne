"""TaxBrain V29 configuration module.

This module wires the "signals → strategies → tiers → routing" stack into a
set of immutable data objects so that runtime behaviour can be steered purely
via configuration. Besides the defaults, :func:`apply_overrides` helps align a
`V29Config` instance with Freqtrade's ``strategy_params`` overrides.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, replace, is_dataclass
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
    sizing_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"] = "BASELINE"
    k_mult_base_pct: float = 1.0
    recovery_factor: float = 1.0
    cooldown_bars: int = 0
    cooldown_bars_after_win: int = 0
    per_pair_risk_cap_pct: float = 0.02
    max_stake_notional_pct: float = 0.10
    icu_force_exit_bars: int = 0
    priority: int = 100
    default_exit_profile: Optional[str] = None
    single_position_only: bool = False  # 新增：有仓位时是否禁止该 tier 再开新仓


@dataclass(frozen=True)
class SizingConfig:
    """Aggregate all sizing knobs related to initial position sizing."""

    initial_size_mode: Literal["static", "dynamic", "hybrid"] = "static"
    static_initial_nominal: float = 6.0
    initial_size_equity_pct: float = 0.0
    initial_max_nominal_per_trade: float = 3000.0  # hard cap per trade, not a default stake size
    per_pair_max_nominal_static: float = 3000.0
    enforce_leverage: float = 10.0


@dataclass(frozen=True)
class TreasuryConfig:
    """Fast/slow bucket controls and caps for treasury allocations."""

    enable_fast_bucket: bool = True
    enable_slow_bucket: bool = True
    treasury_fast_split_pct: float = 0.4
    fast_topK_squads: int = 10
    slow_universe_pct: float = 1.0
    min_injection_nominal_fast: float = 30.0
    min_injection_nominal_slow: float = 7.0
    fast_mode: Literal["per_squad", "top_pairs"] = "per_squad"
    debt_pool_cap_pct: float = 0.15
    bucket_as_cap: bool = True
    bucket_sum_mode: Literal["sum", "max"] = "sum"


@dataclass(frozen=True)
class GatekeepingConfig:
    """Global debt gatekeeping parameters."""

    enabled: bool = True

    # Fast Bucket 准入条件 (激进回血)
    fast_percentile: int = 90       # 必须达到前 10% 高分
    fast_max_closs: int = 0         # 仅允许 closs=0 (健康币种)

    # Slow Bucket 准入条件 (稳健积累)
    slow_percentile: int = 60       # 必须达到前 40% 高分
    slow_max_closs: int = 1         # 允许 closs 0 或 1

    # 无债务时的宽松模式
    no_debt_percentile: int = 30    # 无债时，前 70% 均可入场
    healthy_allow_score: float = 0.6  # closs=0 的健康币在债务期的最低准入分


@dataclass(frozen=True)
class TargetRecoveryConfig:
    """Parameters for ATR-based TARGET_RECOVERY sizing."""

    use_atr_based: bool = True
    include_bucket_in_recovery: bool = True
    include_debt_pool: bool = False
    max_recovery_multiple: float = 10900.0


@dataclass(frozen=True)
class SizingAlgoConfig:
    """Algorithm selection and parameters for sizing."""

    default_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"] = "BASELINE"
    target_recovery: TargetRecoveryConfig = field(default_factory=TargetRecoveryConfig)


@dataclass(frozen=True)
class SystemConfig:
    """System-level knobs related to runtime and backend wiring."""

    global_backend_mode: str = "local"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_namespace: str = "TB_V29:"
    timeframe: str = "5m"
    startup_candle_count: int = 210
    dry_run_wallet_fallback: float = 1000.0


@dataclass(frozen=True)
class RiskConfig:
    """Risk-related knobs including gatekeeping and decay."""

    portfolio_cap_pct_base: float = 0.20
    drawdown_threshold_pct: float = 0.15
    gatekeeping: GatekeepingConfig = field(default_factory=GatekeepingConfig)
    tax_rate_on_wins: float = 0.20
    pain_decay_per_bar: float = 0.999
    clear_debt_on_profitable_cycle: bool = True


@dataclass(frozen=True)
class TradingConfig:
    """Trading-related sizing and treasury wiring."""

    sizing: SizingConfig = field(default_factory=SizingConfig)
    treasury: TreasuryConfig = field(default_factory=TreasuryConfig)


# Helper factories must be defined before StrategyConfig default_factory binding
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


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy-level components such as signals, tiers and exits."""

    enabled_signals: Tuple[str, ...] = field(default_factory=_default_enabled_signals)
    exit_profile_version: str = DEFAULT_PROFILE_VERSION
    exit_profiles: Dict[str, ExitProfile] = field(default_factory=_copy_exit_profiles)
    default_exit_profile: Optional[str] = "ATRtrail_v1"
    strategies: Dict[str, StrategySpec] = field(default_factory=_copy_strategies)
    tiers: Dict[str, TierSpec] = field(default_factory=_copy_tiers)
    tier_routing: TierRouting = field(default_factory=_default_tier_routing)

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
        min_rr=0.00001,
        min_edge=0.0,
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
    ),
    "ICU_conservative": StrategySpec(
        name="ICU_conservative",
        entries=("trend_short", "mean_rev_long","newbars_breakout_long_30m"),
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
        min_rr_ratio=0.00001,
        min_edge=0.002,
        sizing_algo="BASE_ONLY",
        k_mult_base_pct=0.0,
        recovery_factor=1.0,
        cooldown_bars=0,
        cooldown_bars_after_win=0,
        per_pair_risk_cap_pct=0.02,
        max_stake_notional_pct=0.5,
        icu_force_exit_bars=0,
        default_exit_profile="ATRtrail_v1",
        single_position_only=True,
    ),
    "T12_recovery": TierSpec(
        name="T12_recovery",
        allowed_recipes=("Recovery_mix",),
        allowed_entries=(
            "newbars_breakdown_short_30m",
        ),
        allowed_squads=("PBL", "TRS", "NBX"),
        min_raw_score=0.20,
        min_rr_ratio=0.000001,
        min_edge=0.000,
        sizing_algo="TARGET_RECOVERY",
        k_mult_base_pct=1.0,
        recovery_factor=2,
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
        allowed_entries=("newbars_breakout_long_30m",
        ),
        allowed_squads=("TRS", "MRL","NBX"),
        min_raw_score=0.20,
        min_rr_ratio=0.000001,
        min_edge=0.000,
        sizing_algo="TARGET_RECOVERY",
        k_mult_base_pct=1.0,
        recovery_factor=2,
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

@dataclass
class V29Config:
    """V29 strategy parameters (signals/strategy/tier binding).

    Notes:
    - Any config field containing `nominal` is a nominal USDT amount (amount * price), leverage-agnostic.
    - Freqtrade `stake_amount` / backtest "Total stake amount" represent margin = nominal / leverage.
    """

    system: SystemConfig = field(default_factory=SystemConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    informative_timeframes: tuple[str, ...] = ()

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
    atr_len: int = 20
    adx_len: int = 20

    # Behaviour toggles
    suppress_baseline_when_stressed: bool = True

    # Sizing and algorithm configuration
    sizing_algos: SizingAlgoConfig = field(default_factory=SizingAlgoConfig)

    strategy_recipes_input: InitVar[Optional[Tuple[StrategySpec, ...]]] = None
    _strategy_recipes: Tuple[StrategySpec, ...] = field(init=False, repr=False, default_factory=tuple)

    def __post_init__(self, strategy_recipes_input: Optional[Tuple[StrategySpec, ...]]) -> None:
        def _coerce_dc(value, cls):
            if isinstance(value, cls):
                return value
            if isinstance(value, Mapping):
                return cls(**{k: v for k, v in value.items() if k in {f.name for f in fields(cls)}})
            return cls()

        self.system = _coerce_dc(getattr(self, "system", None), SystemConfig)

        risk_cfg = _coerce_dc(getattr(self, "risk", None), RiskConfig)
        risk_gate = _coerce_dc(getattr(risk_cfg, "gatekeeping", None), GatekeepingConfig)
        self.risk = replace(risk_cfg, gatekeeping=risk_gate)

        trading_cfg = _coerce_dc(getattr(self, "trading", None), TradingConfig)
        t_sizing = _coerce_dc(getattr(trading_cfg, "sizing", None), SizingConfig)
        t_treasury = _coerce_dc(getattr(trading_cfg, "treasury", None), TreasuryConfig)
        self.trading = replace(trading_cfg, sizing=t_sizing, treasury=t_treasury)

        algos = _coerce_dc(getattr(self, "sizing_algos", None), SizingAlgoConfig)
        algos_tr = _coerce_dc(getattr(algos, "target_recovery", None), TargetRecoveryConfig)
        self.sizing_algos = replace(algos, target_recovery=algos_tr)

        strat_cfg = _coerce_dc(getattr(self, "strategy", None), StrategyConfig)
        version = getattr(strat_cfg, "exit_profile_version", DEFAULT_PROFILE_VERSION)
        exit_profiles = strat_cfg.exit_profiles or {}
        if not exit_profiles:
            exit_profiles = resolve_profiles(version)
        strategy_profiles = dict(exit_profiles or {})
        strategies = dict(getattr(strat_cfg, "strategies", {}) or {})
        tiers = dict(getattr(strat_cfg, "tiers", {}) or {})

        if strategy_recipes_input:
            specs = tuple(strategy_recipes_input)
            self._strategy_recipes = specs
            strategies = {spec.name: spec for spec in specs}
        elif self._strategy_recipes:
            specs = tuple(self._strategy_recipes)
            self._strategy_recipes = specs
            if not strategies:
                strategies = {spec.name: spec for spec in specs}
        else:
            specs = tuple(strategies.values())
            self._strategy_recipes = specs

        tier_routing = _coerce_dc(getattr(strat_cfg, "tier_routing", None), TierRouting)
        referenced_tiers = set(tier_routing.loss_tier_map.values()) if (tier_routing and tier_routing.loss_tier_map) else set()
        for tier_name in referenced_tiers:
            if tier_name not in tiers:
                raise ValueError(f"TierRouting references unknown tier '{tier_name}'")
        for tier_name in referenced_tiers:
            spec = tiers[tier_name]
            if not spec.default_exit_profile:
                raise ValueError(f"Tier '{tier_name}' must declare default_exit_profile for routing")
            if spec.default_exit_profile not in strategy_profiles:
                raise ValueError(
                    f"Tier '{tier_name}' references unknown exit profile '{spec.default_exit_profile}'"
                )

        self.strategy = replace(
            strat_cfg,
            exit_profiles=strategy_profiles,
            strategies=strategies,
            tiers=tiers,
            tier_routing=tier_routing,
        )

    @property
    def strategy_recipes(self) -> Tuple[StrategySpec, ...]:
        return self._strategy_recipes

    @strategy_recipes.setter
    def strategy_recipes(self, recipes: Optional[Tuple[StrategySpec, ...]]) -> None:
        specs = tuple(recipes or ())
        self._strategy_recipes = specs
        strategies = {spec.name: spec for spec in specs}
        strat_cfg = getattr(self, "strategy", None)
        if strat_cfg:
            self.strategy = replace(strat_cfg, strategies=strategies)
        else:  # pragma: no cover - defensive
            self.strategy = StrategyConfig(strategies=strategies)


def apply_overrides(cfg: V29Config, strategy_params: Optional[Mapping[str, Any]]) -> V29Config:
    """Apply overrides sourced from Freqtrade ``strategy_params``."""

    if not strategy_params:
        return cfg

    def _merge_dataclass(instance, cls, updates: Mapping[str, Any] | Any):
        if isinstance(updates, cls):
            return updates
        base = {f.name: getattr(instance, f.name) for f in fields(cls)}
        if isinstance(updates, Mapping):
            for k, v in updates.items():
                if k in base:
                    if is_dataclass(base[k]) and isinstance(v, Mapping):
                        base[k] = _merge_dataclass(base[k], type(base[k]), v)
                    else:
                        base[k] = v
        return cls(**base)

    legacy_map: Dict[str, tuple[str, str, str]] = {
        # sizing
        "initial_size_mode": ("trading", "sizing", "initial_size_mode"),
        "static_initial_nominal": ("trading", "sizing", "static_initial_nominal"),
        "initial_size_equity_pct": ("trading", "sizing", "initial_size_equity_pct"),
        "initial_max_nominal_per_trade": ("trading", "sizing", "initial_max_nominal_per_trade"),
        "per_pair_max_nominal_static": ("trading", "sizing", "per_pair_max_nominal_static"),
        "enforce_leverage": ("trading", "sizing", "enforce_leverage"),
        # treasury
        "treasury_fast_split_pct": ("trading", "treasury", "treasury_fast_split_pct"),
        "fast_topK_squads": ("trading", "treasury", "fast_topK_squads"),
        "slow_universe_pct": ("trading", "treasury", "slow_universe_pct"),
        "enable_fast_bucket": ("trading", "treasury", "enable_fast_bucket"),
        "enable_slow_bucket": ("trading", "treasury", "enable_slow_bucket"),
        "min_injection_nominal_fast": ("trading", "treasury", "min_injection_nominal_fast"),
        "min_injection_nominal_slow": ("trading", "treasury", "min_injection_nominal_slow"),
        "fast_mode": ("trading", "treasury", "fast_mode"),
        "debt_pool_cap_pct": ("trading", "treasury", "debt_pool_cap_pct"),
        "bucket_as_cap": ("trading", "treasury", "bucket_as_cap"),
        "bucket_sum_mode": ("trading", "treasury", "bucket_sum_mode"),
    }

    target_recovery_map: Dict[str, str] = {
        "use_atr_based": "use_atr_based",
        "include_bucket_in_recovery": "include_bucket_in_recovery",
        "include_debt_pool": "include_debt_pool",
        "max_recovery_multiple": "max_recovery_multiple",
    }

    # Normalize dotted keys into nested dicts
    normalized: Dict[str, Any] = {}
    for key, value in strategy_params.items():
        if "." not in key:
            existing = normalized.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged = dict(existing)
                merged.update(value)
                normalized[key] = merged
            else:
                normalized[key] = value
            continue
        root, *rest = key.split(".")
        target = normalized.setdefault(root, {})
        if not isinstance(target, dict):
            target = {}
            normalized[root] = target
        cursor = target
        for token in rest[:-1]:
            nxt = cursor.get(token)
            if not isinstance(nxt, dict):
                nxt = {}
            cursor[token] = nxt
            cursor = nxt
        cursor[rest[-1]] = value

    system_fields = {f.name for f in fields(SystemConfig)}
    risk_fields = {f.name for f in fields(RiskConfig)}
    trading_fields = {f.name for f in fields(TradingConfig)}
    strategy_fields = {f.name for f in fields(StrategyConfig)}

    if "system" in normalized:
        cfg.system = _merge_dataclass(cfg.system, SystemConfig, normalized["system"])
    if "risk" in normalized:
        cfg.risk = _merge_dataclass(cfg.risk, RiskConfig, normalized["risk"])
    if "trading" in normalized:
        cfg.trading = _merge_dataclass(cfg.trading, TradingConfig, normalized["trading"])
    if "strategy" in normalized:
        cfg.strategy = _merge_dataclass(cfg.strategy, StrategyConfig, normalized["strategy"])

    for key, value in normalized.items():
        if key in {"system", "risk", "trading", "strategy"}:
            continue
        if key == "sizing_algos":
            current = cfg.sizing_algos
            merged = _merge_dataclass(current, SizingAlgoConfig, value)
            tr_updates = value.get("target_recovery") if isinstance(value, Mapping) else getattr(value, "target_recovery", None)
            merged_tr = _merge_dataclass(
                getattr(merged, "target_recovery", TargetRecoveryConfig()),
                TargetRecoveryConfig,
                tr_updates or {},
            )
            cfg.sizing_algos = replace(merged, target_recovery=merged_tr)
            continue
        if key == "target_recovery":
            tr = _merge_dataclass(cfg.sizing_algos.target_recovery, TargetRecoveryConfig, value)
            cfg.sizing_algos = replace(cfg.sizing_algos, target_recovery=tr)
            continue
        if key in legacy_map:
            group, container_name, field_name = legacy_map[key]
            target_obj = getattr(cfg, group)
            container_val = getattr(target_obj, container_name)
            cls = SizingConfig if container_name == "sizing" else TreasuryConfig
            updated = _merge_dataclass(container_val, cls, {field_name: value})
            setattr(cfg, group, replace(target_obj, **{container_name: updated}))
            continue
        if key in target_recovery_map:
            tr = _merge_dataclass(
                cfg.sizing_algos.target_recovery,
                TargetRecoveryConfig,
                {target_recovery_map[key]: value},
            )
            cfg.sizing_algos = replace(cfg.sizing_algos, target_recovery=tr)
            continue
        # Backward compat: allow flat risk/trading/strategy field overrides
        if key in system_fields:
            cfg.system = _merge_dataclass(cfg.system, SystemConfig, {key: value})
            continue
        if key in risk_fields:
            cfg.risk = _merge_dataclass(cfg.risk, RiskConfig, {key: value})
            continue
        if key in trading_fields:
            cfg.trading = _merge_dataclass(cfg.trading, TradingConfig, {key: value})
            continue
        if key in strategy_fields:
            cfg.strategy = _merge_dataclass(cfg.strategy, StrategyConfig, {key: value})
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def get_exit_profile(cfg: V29Config, name: str) -> ExitProfile:
    """Fetch a named exit profile, raising for unknown profiles."""

    try:
        profiles = getattr(getattr(cfg, "strategy", None), "exit_profiles", None) or {}
        return profiles[name]
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
        tiers = getattr(getattr(cfg, "strategy", None), "tiers", None) or {}
        return tiers[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tier '{name}'") from exc


def get_tier_for_closs(cfg: V29Config, closs: int) -> TierSpec:
    """Resolve the tier spec that should serve a given closs value."""

    tier_routing = getattr(getattr(cfg, "strategy", None), "tier_routing", None)
    tier_name = tier_routing.resolve(closs) if tier_routing else None
    if not tier_name:
        raise ValueError("Tier routing map is empty; cannot resolve tier for closs.")
    return get_tier_spec(cfg, tier_name)


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
    "V29Config",
    "apply_overrides",
    "entries_to_recipe",
    "find_strategy_recipe",
    "get_exit_profile",
    "get_strategy",
    "get_tier_for_closs",
    "get_tier_spec",
]
