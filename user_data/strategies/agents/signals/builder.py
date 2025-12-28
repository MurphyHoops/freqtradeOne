# -*- coding: utf-8 -*-
"""Executor that turns registered signals into scored candidates."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

from ...config.v29_config import V29Config
from .factors import (
    DEFAULT_BAG_FACTORS,
    FactorBank,
    _compute_adx_zsig,
    _compute_hurst_rs,
    apply_timeframe_to_factor,
    calculate_regime_factor,
    factor_components_with_default,
    factor_dependencies,
    parse_factor_name,
)
from .registry import REGISTRY
from .risk import RiskEstimator, RiskPlan
from .schemas import Candidate, Condition

_OPS = {
    "<": lambda x, v: x < v,
    "<=": lambda x, v: x <= v,
    ">": lambda x, v: x > v,
    ">=": lambda x, v: x >= v,
    "==": lambda x, v: x == v,
    "between": lambda x, bounds: bounds[0] <= x <= bounds[1],
    "outside": lambda x, bounds: x <= bounds[0] or x >= bounds[1],
}

FactorMap = Dict[Optional[str], Set[str]]
IndicatorMap = Dict[Optional[str], Set[str]]


# -------- Signal requirements -------------------------------------------------
def _normalized_extra(extra: Optional[Iterable[str]]) -> Iterable[str]:
    """Deduplicate/normalize user-provided factor names."""

    if not extra:
        return ()
    seen: Set[str] = set()
    for item in extra:
        if not item:
            continue
        token = item.strip()
        if not token:
            continue
        base, _, tf = token.partition("@")
        base_key = base.strip().upper()
        tf_key = tf.strip().lower()
        key = base_key if not tf_key else f"{base_key}@{tf_key}"
        if key in seen:
            continue
        seen.add(key)
        yield key


def collect_factor_requirements(
    extra: Optional[Iterable[str]] = None,
    cfg: Optional[V29Config] = None,
) -> FactorMap:
    """Return per-timeframe factor requirements derived from registered signals.

    Args:
        extra: Optional list of additional factors (with optional ``@timeframe`` suffix).
        cfg: Optional V29Config; when provided, only signals listed in
            ``cfg.enabled_signals`` are considered.
    """

    enabled = {
        name
        for name in (
            getattr(getattr(cfg, "strategy", None), "enabled_signals", getattr(cfg, "enabled_signals", ())) or ()
        )
        if name
    }

    mapping: Dict[Optional[str], Set[str]] = defaultdict(set)
    mapping[None].update(DEFAULT_BAG_FACTORS)
    defaults_injected: Set[Optional[str]] = {None}
    for item in _normalized_extra(extra):
        base, tf = parse_factor_name(item)
        mapping[tf].add(base)

    for spec in REGISTRY.all():
        if enabled and spec.name not in enabled:
            continue
        spec_tf = spec.timeframe
        if spec_tf not in defaults_injected:
            mapping[spec_tf].update(DEFAULT_BAG_FACTORS)
            defaults_injected.add(spec_tf)
        for cond in spec.conditions:
            base, ctf = factor_components_with_default(cond.factor, spec.timeframe)
            mapping[ctf].add(base)
        for factor in getattr(spec, "required_factors", ()):
            base, rtf = factor_components_with_default(factor, spec.timeframe)
            mapping[rtf].add(base)

    return {tf: set(values) for tf, values in mapping.items() if values}


def collect_indicator_requirements(
    extra: Optional[Iterable[str]] = None,
    cfg: Optional[V29Config] = None,
) -> IndicatorMap:
    """Translate factor requirements into indicator dependencies."""

    factor_map = collect_factor_requirements(extra, cfg)
    indicator_map: Dict[Optional[str], Set[str]] = defaultdict(set)
    for tf, factors in factor_map.items():
        for factor in factors:
            deps = factor_dependencies(factor)
            if deps:
                indicator_map[tf].update(deps)
    return {tf: set(values) for tf, values in indicator_map.items() if values}


def required_timeframes(
    extra: Optional[Iterable[str]] = None,
    cfg: Optional[V29Config] = None,
) -> Set[str]:
    """Return the set of informative timeframes needed beyond the base timeframe."""

    factor_map = collect_factor_requirements(extra, cfg)
    return {tf for tf in factor_map.keys() if tf}


def _check_condition(fb: FactorBank, cond: Condition, timeframe: Optional[str]) -> bool:
    """Return True when a single condition passes."""

    try:
        factor_name = apply_timeframe_to_factor(cond.factor, timeframe)
        value = fb.get(factor_name)
    except KeyError:
        return False
    if value is None or math.isnan(value):
        return False
    if cond.fn is not None:
        try:
            return bool(cond.fn(value))
        except Exception:
            return False
    if cond.op is None:
        return bool(value)
    if cond.op in {"between", "outside"}:
        if cond.value is None or cond.value_hi is None:
            return False
        return _OPS[cond.op](value, (cond.value, cond.value_hi))
    if cond.value is None:
        return False
    return _OPS[cond.op](value, cond.value)


def _safe(value: float, default: float = 0.0) -> float:
    """Normalize NaN/None/inf values during scoring."""

    try:
        if value is None or math.isnan(value) or math.isinf(value):
            return default
        return float(value)
    except Exception:
        return default


def _shape_score(score: float, exponent: float) -> float:
    """Optionally reshape score to widen separation between highs/lows."""

    try:
        exp = float(exponent)
    except Exception:
        exp = 1.0
    if exp <= 0:
        return max(0.0, min(1.0, score))
    base = max(0.0, min(1.0, score))
    if exp == 1.0:
        return base
    return max(0.0, min(1.0, base ** exp))


class _FactorBag(dict):
    """Lazy-fetch container around FactorBank for sl/tp/raw calculations."""

    def __init__(self, fb: FactorBank, initial: Dict[str, float], timeframe: Optional[str]) -> None:
        super().__init__(initial)
        self._fb = fb
        self._timeframe = timeframe

    def __getitem__(self, key: str) -> float:  # type: ignore[override]
        if key not in self:
            resolved = apply_timeframe_to_factor(key, self._timeframe)
            self[key] = self._fb.get(resolved)
        return super().__getitem__(key)


def build_candidates(
    row: Any,
    cfg,
    informative: Optional[Dict[str, Any]] = None,
    history_close: Optional[Iterable[float]] = None,
    history_adx: Optional[Iterable[float]] = None,
    specs: Optional[Iterable[Any]] = None,
) -> List[Candidate]:
    """Build candidates for the latest market row (and optional informative rows)."""

    fb = FactorBank(row, informative=informative)
    hurst_val: Optional[float] = None
    z_sig: Optional[float] = None
    if history_close:
        hurst_val = _compute_hurst_rs(history_close)
        if hurst_val is not None and (math.isnan(hurst_val) or math.isinf(hurst_val)):
            hurst_val = None
    if history_adx:
        try:
            adx_now = float(fb.get("ADX"))
        except Exception:
            adx_now = float("nan")
        z_sig = _compute_adx_zsig(history_adx, adx_now)
        if z_sig is not None and (math.isnan(z_sig) or math.isinf(z_sig)):
            z_sig = None
    base_cache: Dict[Optional[str], Dict[str, float]] = {}
    base_cache[None] = _prefetch_base(fb, None)
    if any(math.isnan(v) for v in base_cache[None].values()):
        return []

    enabled = {
        name
        for name in (
            getattr(getattr(cfg, "strategy", None), "enabled_signals", getattr(cfg, "enabled_signals", ())) or ()
        )
        if name
    }
    strategy_map = getattr(getattr(cfg, "strategy", None), "strategies", getattr(cfg, "strategies", {})) or {}
    risk = RiskEstimator(cfg)
    results: List[Candidate] = []
    specs_iter = specs if specs is not None else REGISTRY.all()
    for spec in specs_iter:
        if enabled and spec.name not in enabled:
            continue
        tf = spec.timeframe
        if tf not in base_cache:
            base_cache[tf] = _prefetch_base(fb, tf)
        base = dict(base_cache[tf])
        bag = _FactorBag(fb, base, tf)
        if not _all_conditions_pass(fb, spec.conditions, tf):
            continue

        plan = risk.plan(spec.name, tf, fb, bag)
        if not plan:
            continue

        exit_profile = plan.exit_profile
        recipe_name = plan.recipe
        sl = _safe(plan.sl_pct)
        tp = _safe(plan.tp_pct)
        if sl <= 0 or tp <= 0:
            continue
        raw = _safe(spec.raw_fn(bag, cfg))
        rr = tp / max(sl, 1e-12)
        min_rr = plan.min_rr if plan.min_rr is not None else spec.min_rr
        min_edge = plan.min_edge if plan.min_edge is not None else spec.min_edge
        if rr < min_rr:
            continue

        # === Regime-aware scoring ===
        strat_name = plan.recipe or spec.name
        strat_spec = strategy_map.get(strat_name)
        base_wp = float(getattr(strat_spec, "base_win_prob", getattr(spec, "base_win_prob", 0.5)))
        try:
            win_prob_val = _safe(spec.win_prob_fn(bag, cfg, raw), default=base_wp)
        except Exception:
            win_prob_val = base_wp
        regime_factor = calculate_regime_factor(
            bag,
            strat_name,
            history_close=history_close,
            history_adx=history_adx,
            hurst_val=hurst_val,
            z_sig=z_sig,
        )
        gcfg = getattr(getattr(cfg, "risk", None), "gatekeeping", getattr(cfg, "gatekeeping", None))
        score_exp = float(getattr(gcfg, "score_curve_exponent", 1.0) or 1.0) if gcfg else 1.0
        final_score = max(0.0, min(1.0, win_prob_val * regime_factor))
        score_for_use = _shape_score(final_score, score_exp)
        if score_for_use < min_edge:
            continue

        results.append(
            Candidate(
                direction=spec.direction,
                kind=spec.name,
                raw_score=raw,
                rr_ratio=rr,
                win_prob=score_for_use,
                expected_edge=score_for_use,
                squad=spec.squad,
                sl_pct=sl,
                tp_pct=tp,
                exit_profile=exit_profile,
                recipe=recipe_name,
                plan_timeframe=plan.plan_timeframe,
                plan_atr_pct=plan.plan_atr_pct,
            )
        )
    return results


def _prefetch_base(fb: FactorBank, timeframe: Optional[str]) -> Dict[str, float]:
    """Fetch default bag factors for a given timeframe upfront."""

    data: Dict[str, float] = {}
    for factor in DEFAULT_BAG_FACTORS:
        resolved = apply_timeframe_to_factor(factor, timeframe)
        try:
            data[factor] = fb.get(resolved)
        except KeyError:
            data[factor] = float("nan")
    return data


def _all_conditions_pass(
    fb: FactorBank, conditions: Iterable[Condition], timeframe: Optional[str]
) -> bool:
    for cond in conditions:
        if not _check_condition(fb, cond, timeframe):
            return False
    return True
