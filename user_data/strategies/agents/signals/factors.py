# -*- coding: utf-8 -*-
"""Factor resolution utilities backed by declarative specs."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .factor_spec import (
    BASE_FACTOR_SPECS,
    DERIVED_FACTOR_SPECS,
    DEFAULT_BAG_FACTORS,
    column_for_factor,
    parse_factor_name,
    _compose_factor,
)


def _safe_get(row: Any, key: str) -> float:
    if row is None:
        return float("nan")
    value = None
    if isinstance(row, dict):
        value = row.get(key, float("nan"))
    else:
        try:
            value = row[key]
        except Exception:
            value = getattr(row, key, float("nan"))
    try:
        value = float(value)
    except Exception:
        return float("nan")
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    return value


class FactorBank:
    """Provide cached factor access across primary + informative rows."""

    def __init__(self, row: Any, informative: Optional[Dict[str, Any]] = None) -> None:
        self._row = row
        self._informative = informative or {}
        self._cache: Dict[str, float] = {}

    def get(self, name: str) -> float:
        if name in self._cache:
            return self._cache[name]
        base, timeframe = parse_factor_name(name)
        cache_key = f"{base}@{timeframe or 'primary'}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        value = self._compute(base, timeframe)
        self._cache[cache_key] = value
        return value

    def take(self, *names: str) -> Dict[str, float]:
        return {name: self.get(name) for name in names}

    # internal helpers ---------------------------------------------------
    def _compute(self, base: str, timeframe: Optional[str]) -> float:
        if base in DERIVED_FACTOR_SPECS:
            spec = DERIVED_FACTOR_SPECS[base]
            return spec.fn(self, timeframe)
        column = column_for_factor(base, timeframe)
        if column is None:
            raise KeyError(base)
        row = self._row if timeframe is None else self._informative.get(timeframe)
        value = _safe_get(row, column)
        if timeframe is not None and (row is None or math.isnan(value)):
            # Freqtrade merges informative columns back onto the base timeframe dataframe,
            # so fall back to the primary row when the per-timeframe snapshot is missing.
            value = _safe_get(self._row, column)
        if (
            timeframe
            and (value is None or math.isnan(value))
            and base in BASE_FACTOR_SPECS
            and not BASE_FACTOR_SPECS[base].indicators
        ):
            fallback_col = BASE_FACTOR_SPECS[base].column
            value = _safe_get(row, fallback_col)
        return value

    def _get_factor(self, base: str, timeframe: Optional[str]) -> float:
        return self.get(_compose_factor(base, timeframe))


__all__ = ["FactorBank", "DEFAULT_BAG_FACTORS"]
