# -*- coding: utf-8 -*-
"""Signal module dependency analysis helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Set

from ...config.v29_config import V29Config
from .registry import REGISTRY
from .factor_spec import (
    DEFAULT_BAG_FACTORS,
    factor_components_with_default,
    factor_dependencies,
    parse_factor_name,
)

FactorMap = Dict[Optional[str], Set[str]]
IndicatorMap = Dict[Optional[str], Set[str]]


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

    enabled = {name for name in getattr(cfg, "enabled_signals", ()) or () if name}

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


__all__ = [
    "collect_factor_requirements",
    "collect_indicator_requirements",
    "required_timeframes",
]
