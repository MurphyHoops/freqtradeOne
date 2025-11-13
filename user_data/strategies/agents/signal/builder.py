# -*- coding: utf-8 -*-
"""Executor that turns registered signals into scored candidates."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from .factor_spec import DEFAULT_BAG_FACTORS, apply_timeframe_to_factor
from .factors import FactorBank
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


def build_candidates(row: Any, cfg, informative: Optional[Dict[str, Any]] = None) -> List[Candidate]:
    """Build candidates for the latest market row (and optional informative rows)."""

    fb = FactorBank(row, informative=informative)
    base_cache: Dict[Optional[str], Dict[str, float]] = {}
    base_cache[None] = _prefetch_base(fb, None)
    if any(math.isnan(v) for v in base_cache[None].values()):
        return []

    enabled = {name for name in getattr(cfg, "enabled_signals", ()) or () if name}
    risk = RiskEstimator(cfg)
    results: List[Candidate] = []
    for spec in REGISTRY.all():
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
        win = _safe(spec.win_prob_fn(bag, cfg, raw), default=0.5)
        rr = tp / max(sl, 1e-12)
        edge = win * tp - (1.0 - win) * sl
        min_rr = plan.min_rr if plan.min_rr is not None else spec.min_rr
        min_edge = plan.min_edge if plan.min_edge is not None else spec.min_edge
        if rr < min_rr or edge < min_edge:
            continue

        results.append(
            Candidate(
                direction=spec.direction,
                kind=spec.name,
                raw_score=raw,
                rr_ratio=rr,
                win_prob=win,
                expected_edge=edge,
                squad=spec.squad,
                exit_profile=exit_profile,
                recipe=recipe_name,
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

