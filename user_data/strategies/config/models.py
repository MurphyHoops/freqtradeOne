# -*- coding: utf-8 -*-
"""Config data models for TaxBrain V30."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Dict, Literal, Mapping, Optional, Tuple

from ..agents.exits.profiles import ExitProfile


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
    min_raw_score: float = 0.0  # Minimum raw signal score; raise to be pickier.
    min_rr_ratio: float = 1.0  # Minimum reward/risk ratio; >1 enforces positive skew.
    min_edge: float = 0.0  # Minimum expected edge; raise to reject thin edges.
    sizing_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"] = "BASELINE"  # Sizing policy used for the tier.
    center_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"] = "TARGET_RECOVERY"
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
    bct_beta_min: float = 1.0  # Minimum beta applied to score exponent under low pressure.
    bct_beta_max: float = 4.0  # Maximum beta applied to score exponent under high pressure.
    bct_pressure_ratio: float = 1.0  # Pressure value that maps to beta_max; lower = more aggressive ramp.
    bct_pressure_ema_alpha: float = 0.2  # EWMA alpha for pressure smoothing; 0 disables smoothing.
    bct_pressure_include_reservation: bool = True  # Include local reservations as a floor even when backend is enabled.
    fluid_cap_pct_of_equity: float = 0.05  # Cap for central fluid allocation as fraction of equity.
    c_target_risk_cap_pct_of_equity: float = 0.05  # Cap for central target risk as fraction of equity.


@dataclass(frozen=True)
class SystemConfig:
    """System-level knobs related to runtime and backend wiring."""

    user_data_dir: Optional[str] = None  # Root user_data directory; used for plugin/preset discovery.
    global_backend_mode: str = "local"  # Backend type for shared state: "local" or "redis"; redis enables cross-worker sharing.
    redis_host: str = "localhost"  # Redis host when using redis backend.
    redis_port: int = 6379  # Redis port; change if your redis listens elsewhere.
    redis_db: int = 0  # Redis DB index; isolates namespaces.
    redis_namespace: str = "TB_V30:"  # Redis key prefix; change to avoid collisions.
    timeframe: str = "5m"  # Primary strategy timeframe; higher values slow trading cadence.
    startup_candle_count: int = 210  # Warmup candles required; raise if indicators need longer history.
    dry_run_wallet_fallback: float = 1000.0  # Equity seed for backtests/dry-run when exchange balance unavailable.
    vectorized_entry_backtest: bool = True  # Use vectorized prefilter + sparse row evaluation in backtest/hyperopt/plot.
    merge_informative_into_base: bool = True  # Merge informative columns once into base dataframe in backtest/hyperopt.
    aligned_info_cache_max_entries: int = 512  # LRU cap for aligned informative cache; 0 disables.
    informative_cache_max_entries: int = 0  # LRU cap for informative dataframe cache; 0 disables.
    informative_gc_mem_pct: float = 0.85  # Memory pressure (0-1 or 0-100) to trigger cache GC early.
    informative_gc_force_pct: float = 0.92  # Memory pressure (0-1 or 0-100) to force-clear caches.
    market_sensor_enabled: bool = True  # Master toggle for market sensor.
    market_sensor_in_backtest: bool = False  # Enable market sensor during backtest/hyperopt.
    market_sensor_strict: bool = False  # Raise if TA-Lib is missing when sensor is enabled.
    debug_prints: bool = False  # Enable verbose prints in strategy hooks.
    plugin_load_strict: bool = False  # Raise if any plugin fails to load.
    plugin_allow_reload: bool = False  # Allow dev-only plugin reload (clears registry before load).
    auto_discover_plugins: bool = True  # Scan plugin folders for recipes/aux modules.
    sensor_pairs: Tuple[str, ...] = ("BTC/USDT", "ETH/USDT")  # Pairs to fetch for market sensor.
    rejection_log_enabled: bool = False  # Emit structured rejection logs.
    rejection_stats_enabled: bool = True  # Track rejection counters in memory.
    state_store_filename: str = "taxbrain_v30_state.json"  # State file name for persistence.


@dataclass(frozen=True)
class RiskConfig:
    """Risk-related knobs including gatekeeping and decay."""

    portfolio_cap_pct_base: float = 2  # Base portfolio VaR cap (% of equity); raise to allow more open risk.
    drawdown_threshold_pct: float = 0.15  # Debt/equity ratio that halves CAP; lower to throttle sooner.
    gatekeeping: GatekeepingConfig = field(default_factory=GatekeepingConfig)  # Tiered gatekeeping parameters.
    tax_rate_on_wins: float = 0.20  # Fraction of profit siphoned to repay debt; higher repays faster but reduces compounding.
    pain_decay_per_bar: float = 0.999  # Debt decay per bar (0-1); smaller = faster natural debt forgiveness.
    clear_debt_on_profitable_cycle: bool = False  # If True, profitable cycles wipe remaining debt; disable to keep debt sticky.
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


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy-level components such as signals, tiers and exits."""

    enabled_recipes: Tuple[str, ...] = field(default_factory=tuple)  # Optional recipe allowlist; derives enabled_signals when set.
    enabled_signals: Tuple[str, ...] = field(default_factory=tuple)  # Which signals are active; drop entries to disable.
    exit_profile_version: str = "inline_v1"  # Semantic version tag for exit profiles; metadata only.
    exit_profiles: Dict[str, ExitProfile] = field(default_factory=dict)  # Exit profile definitions.
    default_exit_profile: Optional[str] = None  # Default profile name used when candidate/tiers provide none.
    strategies: Dict[str, StrategySpec] = field(default_factory=dict)  # Strategy recipes mapping.
    tiers: Dict[str, TierSpec] = field(default_factory=dict)  # Tier definitions.
    tier_routing: "TierRouting" = field(default_factory=TierRouting)  # Mapping from closs to tier names.


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


@dataclass
class V30Config:
    """V30 strategy parameters (signals/strategy/tier binding)."""

    system: SystemConfig = field(default_factory=SystemConfig)  # Runtime/backend configuration.
    risk: RiskConfig = field(default_factory=RiskConfig)  # Global risk controls.
    trading: TradingConfig = field(default_factory=TradingConfig)  # Sizing and treasury controls.
    strategy: StrategyConfig = field(default_factory=StrategyConfig)  # Signal/tier/exit wiring.
    informative_timeframes: tuple[str, ...] = ()  # Extra informative timeframes.
    sensor: SensorConfig = field(default_factory=SensorConfig)

    stoploss: Optional[float] = None  # Default stoploss (price space), derived if omitted.
    minimal_roi: Optional[Dict[str, float]] = None  # Default minimal ROI mapping, derived if omitted.

    cycle_len_bars: int = 288  # Bars per cycle for debt reset checks; lower = more frequent cycle accounting.
    breakeven_lock_frac_of_tp: float = 0.5  # Fraction of TP to reach before breakeven lock; raise to wait longer.
    breakeven_lock_eps_atr_pct: float = 0.1  # ATR% cushion around breakeven lock; raise to add slack.

    force_finalize_mult: float = 1.5  # Multiplier on finalize cadence; lower forces more frequent finalize passes.
    reservation_ttl_bars: int = 6  # TTL for reservations in bars; lower frees capacity sooner on stale reservations.
    ema_fast: int = 50  # Fast EMA length; lower = more reactive.
    ema_slow: int = 200  # Slow EMA length; higher = smoother trend filter.
    rsi_len: int = 14  # RSI lookback; lower = more jittery.
    atr_len: int = 20  # ATR length; lower responds quicker to volatility shifts.
    adx_len: int = 20  # ADX length; lower = more responsive trend strength.

    suppress_baseline_when_stressed: bool = True  # If True, baseline sizing is suppressed under drawdown stress.

    sizing_algos: SizingAlgoConfig = field(default_factory=SizingAlgoConfig)  # Shared sizing algorithm parameters.

    strategy_recipes_input: InitVar[Optional[Tuple[StrategySpec, ...]]] = None
    _strategy_recipes: Tuple[StrategySpec, ...] = field(init=False, repr=False, default_factory=tuple)

    def __post_init__(self, strategy_recipes_input: Optional[Tuple[StrategySpec, ...]]) -> None:
        if strategy_recipes_input:
            self._strategy_recipes = tuple(strategy_recipes_input)
            return
        if self._strategy_recipes:
            self._strategy_recipes = tuple(self._strategy_recipes)
            return
        strat_cfg = getattr(self, "strategy", None)
        if strat_cfg and getattr(strat_cfg, "strategies", None):
            self._strategy_recipes = tuple(strat_cfg.strategies.values())
        else:
            self._strategy_recipes = tuple()

    @property
    def strategy_recipes(self) -> Tuple[StrategySpec, ...]:
        return self._strategy_recipes

    @strategy_recipes.setter
    def strategy_recipes(self, recipes: Optional[Tuple[StrategySpec, ...]]) -> None:
        specs = tuple(recipes or ())
        self._strategy_recipes = specs
        strat_cfg = getattr(self, "strategy", None)
        if strat_cfg:
            self.strategy = StrategyConfig(
                enabled_recipes=strat_cfg.enabled_recipes,
                enabled_signals=strat_cfg.enabled_signals,
                exit_profile_version=strat_cfg.exit_profile_version,
                exit_profiles=strat_cfg.exit_profiles,
                default_exit_profile=strat_cfg.default_exit_profile,
                strategies={spec.name: spec for spec in specs},
                tiers=strat_cfg.tiers,
                tier_routing=strat_cfg.tier_routing,
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
]
