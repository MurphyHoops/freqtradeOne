"""TaxBrain V29 configuration module.

This module wires the "signals → strategies → tiers → routing" stack into a
set of immutable data objects so that runtime behaviour can be steered purely
via configuration. Besides the defaults, :func:`apply_overrides` helps align a
`V29Config` instance with Freqtrade's ``strategy_params`` overrides.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, replace, is_dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from ..agents.exits.profiles import ExitProfile

# Inline default profile version tag (metadata only for inline defaults).
DEFAULT_PROFILE_VERSION = "inline_v1"


@dataclass(frozen=True)
class StrategySpec:
    """Bind a set of entry signals to an exit profile + policy thresholds."""

    name: str  # Recipe name; used for routing to tiers.
    entries: Tuple[str, ...]  # Signal names grouped under this recipe.
    exit_profile: str  # Exit profile to use when this recipe is selected.
    min_rr: float = 0.0  # Minimum reward/risk threshold; raise to demand higher RR.
    min_edge: float = 0.0  # Minimum expected edge; raise to filter weaker signals.
    base_win_prob: float = 0.5  # Baseline win probability for this strategy; tuned per regime.


# Backwards compatibility alias for older imports
StrategyRecipe = StrategySpec


@dataclass(frozen=True)
class TierSpec:
    """Declarative definition of a tier and its guard-rails."""

    name: str  # Tier identifier; referenced by routing map.
    allowed_recipes: Tuple[str, ...]  # Whitelisted strategy recipes.
    allowed_entries: Tuple[str, ...] = tuple()  # Optional whitelist of entry signals.
    allowed_squads: Tuple[str, ...] = tuple()  # Optional whitelist of squads/teams.
    min_raw_score: float = 0.0  # Minimum raw signal score; raise to be pickier.
    min_rr_ratio: float = 1.0  # Minimum reward/risk ratio; >1 enforces positive skew.
    min_edge: float = 0.0  # Minimum expected edge; raise to reject thin edges.
    sizing_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"] = "BASELINE"  # Sizing policy used for the tier.
    center_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"] = "TARGET_RECOVERY",
    k_mult_base_pct: float = 1.0  # Baseline stake multiplier vs equity; larger = bigger base size.
    recovery_factor: float = 1.0  # Multiplier applied to local_loss for recovery sizing; larger = more aggressive recovery.
    cooldown_bars: int = 0  # Bars to pause after any trade in this tier; raise to slow entries.
    cooldown_bars_after_win: int = 0  # Bars to pause after a win; raise to cool off.
    per_pair_risk_cap_pct: float = 0.02  # Max risk per pair as fraction of equity; lower = safer.
    max_stake_notional_pct: float = 0.10  # Max nominal exposure per pair; lower caps position size.
    icu_force_exit_bars: int = 0  # Force-exit timer in bars; >0 triggers timed exits.
    priority: int = 100  # Higher priority tiers are considered first when resolving.
    default_exit_profile: Optional[str] = None  # Default exit profile for this tier when none provided.
    single_position_only: bool = False  # If True, disallow multiple concurrent positions in this tier.


@dataclass(frozen=True)
class SizingConfig:
    """Aggregate all sizing knobs related to initial position sizing."""

    initial_size_mode: Literal["static", "dynamic", "hybrid"] = "static"  # How base nominal is computed; hybrid = max(static, dynamic).
    static_initial_nominal: float = 6.0  # Deprecated: legacy static seed; kept for backward compatibility.
    initial_size_equity_pct: float = 0.0  # Dynamic seed as % of equity; raise to scale with account size.
    min_stake_multiplier: float = 1.0  # Multiplier on exchange min_notional to seed positions.
    initial_max_nominal_cap: float = 20.0  # Skip entries whose min_notional exceeds this cap (filters expensive pairs).
    initial_max_nominal_per_trade: float = 3000.0  # Hard ceiling per trade nominal; lower to clamp single-trade exposure.
    per_pair_max_nominal_static: float = 3000.0  # Static cap of open nominal per pair; lower to limit pair concentration.
    enforce_leverage: float = 10.0  # Fixed leverage applied; lower to reduce margin, higher increases exposure.


@dataclass(frozen=True)
class TreasuryConfig:
    """UEOT polar treasury controls (replaces fast/slow buckets)."""

    debt_pool_cap_pct: float = 0.15  # Max portion of equity that can be used to repay debt; lower to cap recovery aggression.


@dataclass(frozen=True)
class GatekeepingConfig:
    """Global debt gatekeeping parameters."""

    enabled: bool = True  # Master toggle for gatekeeping; disable to bypass checks.
    score_curve_exponent: float = 1.0  # Shape factor for score (1=linear, >1 widens gap).
    min_score: float = 0.0  # Unified UEOT score floor.


@dataclass(frozen=True)
class TargetRecoveryConfig:
    """Parameters for ATR-based TARGET_RECOVERY sizing."""

    use_atr_based: bool = True  # If True, recovery sizing uses ATR-based TP distance; False falls back to SL distance.
    include_bucket_in_recovery: bool = True  # Include bucket debt in recovery sizing; raise/lower to control aggression.
    max_recovery_multiple: float = 10900.0  # Cap on recovery multiplier vs base size; lower to avoid runaway recovery.


@dataclass(frozen=True)
class SizingAlgoConfig:
    """Algorithm selection and parameters for sizing."""

    default_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"] = "BASELINE"  # Default sizing algorithm.
    target_recovery: TargetRecoveryConfig = field(default_factory=TargetRecoveryConfig)  # Tunables for TARGET_RECOVERY.
    score_floor: float = 0.3  # Minimum score to start allocating central (fluid) debt; lower = more permissive.
    score_exponent: float = 2.0  # Curve applied to score for fluid sizing (>=0); higher = reward high scores more.


@dataclass(frozen=True)
class SystemConfig:
    """System-level knobs related to runtime and backend wiring."""

    global_backend_mode: str = "local"  # Backend type for shared state: "local" or "redis"; redis enables cross-worker sharing.
    redis_host: str = "localhost"  # Redis host when using redis backend.
    redis_port: int = 6379  # Redis port; change if your redis listens elsewhere.
    redis_db: int = 0  # Redis DB index; isolates namespaces.
    redis_namespace: str = "TB_V29:"  # Redis key prefix; change to avoid collisions.
    timeframe: str = "5m"  # Primary strategy timeframe; higher values slow trading cadence.
    startup_candle_count: int = 210  # Warmup candles required; raise if indicators need longer history.
    dry_run_wallet_fallback: float = 1000.0  # Equity seed for backtests/dry-run when exchange balance unavailable.


@dataclass(frozen=True)
class RiskConfig:
    """Risk-related knobs including gatekeeping and decay."""

    portfolio_cap_pct_base: float = 2  # Base portfolio VaR cap (% of equity); raise to allow more open risk.
    drawdown_threshold_pct: float = 0.15  # Debt/equity ratio that halves CAP; lower to throttle sooner.
    gatekeeping: GatekeepingConfig = field(default_factory=GatekeepingConfig)  # Tiered gatekeeping parameters.
    tax_rate_on_wins: float = 0.20  # Fraction of profit siphoned to repay debt; higher repays faster but reduces compounding.
    pain_decay_per_bar: float = 0.999  # Debt decay per bar (0-1); smaller = faster natural debt forgiveness.
    clear_debt_on_profitable_cycle: bool = False  # （周期性清空债务和closs）If True, profitable cycles wipe remaining debt; disable to keep debt sticky.
    aggressiveness: float = 0.5  # UEOT polar field aggressiveness multiplier.
    entropy_factor: float = 0.4  # Chaos bonus intensity applied when bias is neutral.
    volatility_factor: float = 1.0  # Additional multiplier on sensed volatility; keep within 0.5~3.0.

    
@dataclass(frozen=True)
class TradingConfig:
    """Trading-related sizing and treasury wiring."""

    sizing: SizingConfig = field(default_factory=SizingConfig)  # Initial sizing and leverage defaults.
    treasury: TreasuryConfig = field(default_factory=TreasuryConfig)  # Debt-repayment treasury allocations.


@dataclass(frozen=True)
class SensorConfig:
    """Sensor weights and entropy for market bias synthesis."""

    weights: Mapping[str, float] = field(default_factory=lambda: {"BTC": 0.6, "ETH": 0.4})
    entropy_factor: float = 0.4


# Helper factories must be defined before StrategyConfig default_factory binding
def _copy_exit_profiles() -> Dict[str, ExitProfile]:
    return default_profiles_factory()


def _copy_strategies() -> Dict[str, StrategySpec]:
    return dict(DEFAULT_STRATEGIES)


def _copy_tiers() -> Dict[str, TierSpec]:
    return dict(DEFAULT_TIERS)


def _default_enabled_signals() -> Tuple[str, ...]:
    return tuple(DEFAULT_ENABLED_SIGNALS)


def _default_tier_routing() -> TierRouting:
    return TierRouting(loss_tier_map=dict(DEFAULT_TIER_ROUTING_MAP))


def _coerce_exit_profiles(raw: Mapping[str, Any] | Dict[str, ExitProfile]) -> Dict[str, ExitProfile]:
    """Hydrate ExitProfile mappings from raw dicts."""

    if not raw:
        return {}
    out: Dict[str, ExitProfile] = {}
    valid_fields = {f.name for f in fields(ExitProfile)}
    for name, value in raw.items():
        if isinstance(value, ExitProfile):
            out[name] = value
            continue
        if isinstance(value, Mapping):
            filtered = {k: v for k, v in value.items() if k in valid_fields}
            out[name] = ExitProfile(**filtered)
    return out


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy-level components such as signals, tiers and exits."""

    enabled_signals: Tuple[str, ...] = field(default_factory=_default_enabled_signals)  # Which signals are active; drop entries to disable.
    exit_profile_version: str = DEFAULT_PROFILE_VERSION  # Semantic version tag for exit profiles; metadata only when using inline defaults.
    exit_profiles: Dict[str, ExitProfile] = field(default_factory=_copy_exit_profiles)  # Exit profile definitions; can be fully overridden in config.
    default_exit_profile: Optional[str] = "ATRtrail_v1"  # Default profile name used when candidate/tiers provide none.
    strategies: Dict[str, StrategySpec] = field(default_factory=_copy_strategies)  # Strategy recipes mapping.
    tiers: Dict[str, TierSpec] = field(default_factory=_copy_tiers)  # Tier definitions.
    tier_routing: TierRouting = field(default_factory=_default_tier_routing)  # Mapping from closs to tier names.

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
        entries=("trend_short", "mean_rev_long","newbars_breakout_long_30m"),
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
        allowed_entries=("newbars_breakout_long_5m", "newbars_breakdown_short_5m"),
        allowed_squads=("NBX",),
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
        allowed_entries=(
            "newbars_breakdown_short_30m",
        ),
        allowed_squads=("PBL", "TRS", "NBX"),
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
        allowed_entries=("newbars_breakout_long_30m",
        ),
        allowed_squads=("TRS", "MRL","NBX"),
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

@dataclass
class V29Config:
    """V29 strategy parameters (signals/strategy/tier binding).

    Notes:
    - Any config field containing `nominal` is a nominal USDT amount (amount * price), leverage-agnostic.
    - Freqtrade `stake_amount` / backtest "Total stake amount" represent margin = nominal / leverage.
    """

    system: SystemConfig = field(default_factory=SystemConfig)  # Runtime/backend configuration.
    risk: RiskConfig = field(default_factory=RiskConfig)  # Global risk controls.
    trading: TradingConfig = field(default_factory=TradingConfig)  # Sizing and treasury controls.
    strategy: StrategyConfig = field(default_factory=StrategyConfig)  # Signal/tier/exit wiring.
    informative_timeframes: tuple[str, ...] = ()  # Extra informative timeframes; add e.g. ("1h","4h") to enable multi-tf signals.
    sensor: SensorConfig = field(default_factory=SensorConfig)

    cycle_len_bars: int = 288  # Bars per cycle for debt reset checks; lower = more frequent cycle accounting.
    # Early lock / breakeven guards
    breakeven_lock_frac_of_tp: float = 0.5  # Fraction of TP to reach before breakeven lock; raise to wait longer.
    breakeven_lock_eps_atr_pct: float = 0.1  # ATR% cushion around breakeven lock; raise to add slack.

    # Finalize cadence
    force_finalize_mult: float = 1.5  # Multiplier on finalize cadence; lower forces more frequent finalize passes.
    reservation_ttl_bars: int = 6  # TTL for reservations in bars; lower frees capacity sooner on stale reservations.
    # Indicators
    ema_fast: int = 50  # Fast EMA length; lower = more reactive.
    ema_slow: int = 200  # Slow EMA length; higher = smoother trend filter.
    rsi_len: int = 14  # RSI lookback; lower = more jittery.
    atr_len: int = 20  # ATR length; lower responds quicker to volatility shifts.
    adx_len: int = 20  # ADX length; lower = more responsive trend strength.

    # Behaviour toggles
    suppress_baseline_when_stressed: bool = True  # If True, baseline sizing is suppressed under drawdown stress.

    # Sizing and algorithm configuration
    sizing_algos: SizingAlgoConfig = field(default_factory=SizingAlgoConfig)  # Shared sizing algorithm parameters.

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

        sensor_cfg = _coerce_dc(getattr(self, "sensor", None), SensorConfig)
        self.sensor = sensor_cfg

        algos = _coerce_dc(getattr(self, "sizing_algos", None), SizingAlgoConfig)
        algos_tr = _coerce_dc(getattr(algos, "target_recovery", None), TargetRecoveryConfig)
        self.sizing_algos = replace(algos, target_recovery=algos_tr)

        strat_cfg = _coerce_dc(getattr(self, "strategy", None), StrategyConfig)
        exit_profiles_raw = strat_cfg.exit_profiles or {}
        strategy_profiles = _coerce_exit_profiles(exit_profiles_raw)
        if not strategy_profiles:
            strategy_profiles = default_profiles_factory()
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
        "min_stake_multiplier": ("trading", "sizing", "min_stake_multiplier"),
        "initial_max_nominal_cap": ("trading", "sizing", "initial_max_nominal_cap"),
        "initial_max_nominal_per_trade": ("trading", "sizing", "initial_max_nominal_per_trade"),
        "per_pair_max_nominal_static": ("trading", "sizing", "per_pair_max_nominal_static"),
        "enforce_leverage": ("trading", "sizing", "enforce_leverage"),
        # treasury
        "debt_pool_cap_pct": ("trading", "treasury", "debt_pool_cap_pct"),
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
        strat_updates = dict(normalized["strategy"])
        if "exit_profiles" in strat_updates:
            new_profiles = _coerce_exit_profiles(strat_updates.pop("exit_profiles"))
            if new_profiles:
                merged_profiles = dict(cfg.strategy.exit_profiles or {})
                merged_profiles.update(new_profiles)
                strat_updates["exit_profiles"] = merged_profiles
        cfg.strategy = _merge_dataclass(cfg.strategy, StrategyConfig, strat_updates)

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
        if key == "gatekeeping":
            gate_cfg = _merge_dataclass(
                getattr(cfg.risk, "gatekeeping", GatekeepingConfig()),
                GatekeepingConfig,
                value,
            )
            cfg.risk = replace(cfg.risk, gatekeeping=gate_cfg)
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
