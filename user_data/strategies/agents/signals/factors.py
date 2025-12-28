# -*- coding: utf-8 -*-
"""Factor specifications and resolution utilities backed by declarative specs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    FactorBankType = "FactorBank"
else:
    FactorBankType = Any


@dataclass(frozen=True)
class BaseFactorSpec:
    indicators: tuple[str, ...]
    column: str


@dataclass(frozen=True)
class DerivedFactorSpec:
    indicators: tuple[str, ...]
    fn: Callable[[FactorBankType, Optional[str]], float]


# column references assume indicator columns use lowercase naming.
BASE_FACTOR_SPECS: Dict[str, BaseFactorSpec] = {
    "CLOSE": BaseFactorSpec(indicators=(), column="close"),
    "EMA_FAST": BaseFactorSpec(indicators=("EMA_FAST",), column="ema_fast"),
    "EMA_SLOW": BaseFactorSpec(indicators=("EMA_SLOW",), column="ema_slow"),
    "RSI": BaseFactorSpec(indicators=("RSI",), column="rsi"),
    "ATR": BaseFactorSpec(indicators=("ATR",), column="atr"),
    "ATR_PCT": BaseFactorSpec(indicators=("ATR",), column="atr_pct"),
    "ADX": BaseFactorSpec(indicators=("ADX",), column="adx"),
    "NEWBARS_HIGH": BaseFactorSpec(indicators=("NEWHBARS",), column="newbars_high"),
    "NEWBARS_LOW": BaseFactorSpec(indicators=("NEWHBARS",), column="newbars_low"),
}


def _compose_factor(base: str, timeframe: Optional[str]) -> str:
    return base if not timeframe else f"{base}@{timeframe}"


def _delta_close_emafast(fb: FactorBankType, timeframe: Optional[str]) -> float:
    close = fb.get(_compose_factor("CLOSE", timeframe))
    ema_fast = fb.get(_compose_factor("EMA_FAST", timeframe))
    if ema_fast in (0, None):
        return float("nan")
    return close / ema_fast - 1.0


def _ema_trend(fb: FactorBankType, timeframe: Optional[str]) -> float:
    ema_fast = fb.get(_compose_factor("EMA_FAST", timeframe))
    ema_slow = fb.get(_compose_factor("EMA_SLOW", timeframe))
    if any(map(lambda v: v is None or v != v, (ema_fast, ema_slow))):
        return 0.0
    if ema_fast > ema_slow:
        return 1.0
    if ema_fast < ema_slow:
        return -1.0
    return 0.0


DERIVED_FACTOR_SPECS: Dict[str, DerivedFactorSpec] = {
    "DELTA_CLOSE_EMAFAST_PCT": DerivedFactorSpec(
        indicators=("EMA_FAST",),
        fn=_delta_close_emafast,
    ),
    "EMA_TREND": DerivedFactorSpec(
        indicators=("EMA_FAST", "EMA_SLOW"),
        fn=_ema_trend,
    ),
}

DEFAULT_BAG_FACTORS = ("ATR", "ATR_PCT", "CLOSE")


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
    except Exception:
        return 0.5


def _compute_hurst_rs(history_close: Optional[Iterable[float]], window: int = 200, min_points: int = 50) -> float:
    """Compute a simplified R/S Hurst exponent; returns NaN when data is insufficient."""

    if not history_close:
        return float("nan")
    try:
        values = [
            float(v)
            for v in history_close
            if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))
        ]
    except Exception:
        return float("nan")
    if window and window > 0:
        values = values[-window:]
    if len(values) < max(2, min_points):
        return float("nan")

    returns: list[float] = []
    for prev, curr in zip(values, values[1:]):
        if prev == 0 or math.isnan(prev) or math.isinf(prev) or math.isnan(curr) or math.isinf(curr):
            continue
        returns.append((curr - prev) / abs(prev))
    if len(returns) < max(2, min_points // 2):
        return float("nan")

    mean_ret = sum(returns) / len(returns)
    devs = [r - mean_ret for r in returns]
    cum_devs: list[float] = []
    running = 0.0
    for d in devs:
        running += d
        cum_devs.append(running)
    if not cum_devs:
        return float("nan")

    R = max(cum_devs) - min(cum_devs)
    variance = sum(d * d for d in devs) / max(1, len(devs))
    S = math.sqrt(variance)
    if R <= 0 or S <= 0:
        return float("nan")
    try:
        hurst = math.log(R / S) / math.log(len(cum_devs))
    except Exception:
        return float("nan")
    if math.isnan(hurst) or math.isinf(hurst):
        return float("nan")
    return max(0.0, min(1.0, hurst))


def _compute_adx_zsig(
    history_adx: Optional[Iterable[float]],
    current_adx: float,
    window: int = 200,
    min_points: int = 50,
) -> float:
    """Compute sigmoid(z-score) for ADX using recent history; returns NaN if insufficient."""

    if history_adx is None or current_adx is None or math.isnan(current_adx):
        return float("nan")
    try:
        values = [
            float(v)
            for v in history_adx
            if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))
        ]
    except Exception:
        return float("nan")
    if window and window > 0:
        values = values[-window:]
    if len(values) < max(2, min_points):
        return float("nan")
    mu = sum(values) / len(values)
    variance = sum((v - mu) ** 2 for v in values) / max(1, len(values))
    sigma = math.sqrt(variance)
    if sigma <= 1e-9:
        return float("nan")
    z_val = (current_adx - mu) / sigma
    return _sigmoid(z_val)


def calculate_regime_factor(
    bag: Any,
    strategy_type: str | None,
    history_close: Optional[Iterable[float]] = None,
    history_adx: Optional[Iterable[float]] = None,
    hurst_val: Optional[float] = None,
    z_sig: Optional[float] = None,
) -> float:
    """Return a regime multiplier (0.5-1.5) coupled to Hurst + ADX Z-Score."""

    try:
        adx = float(bag.get("ADX"))
    except Exception:
        adx = float("nan")

    if hurst_val is None:
        hurst_val = _compute_hurst_rs(history_close)
    try:
        hurst_val = float(hurst_val)
    except Exception:
        hurst_val = float("nan")
    if math.isnan(hurst_val) or math.isinf(hurst_val):
        hurst_val = 0.5

    if z_sig is None:
        z_sig = _compute_adx_zsig(history_adx, adx)
    try:
        z_sig = float(z_sig)
    except Exception:
        z_sig = float("nan")
    if math.isnan(z_sig) or math.isinf(z_sig):
        z_sig = 0.5

    bias = (strategy_type or "").lower()
    is_trend = any(token in bias for token in ("trend", "breakout"))
    is_mean_rev = any(token in bias for token in ("mean_rev", "pullback"))

    if is_trend and not is_mean_rev:
        raw_factor = 0.7 * hurst_val + 0.3 * z_sig
        factor = 1.0 + (raw_factor - 0.5)
    elif is_mean_rev and not is_trend:
        raw_factor = (0.5 - hurst_val) * 2.0
        factor = 1.0 + raw_factor
    else:
        # neutral: blend both signals with softer leverage
        raw_factor = 0.5 * hurst_val + 0.5 * z_sig
        factor = 1.0 + 0.5 * (raw_factor - 0.5)

    return max(0.5, min(1.5, factor))


def factor_dependencies(factor: str) -> Set[str]:
    """Return indicator names required for given factor base name."""

    if factor in BASE_FACTOR_SPECS:
        return set(BASE_FACTOR_SPECS[factor].indicators)
    if factor in DERIVED_FACTOR_SPECS:
        return set(DERIVED_FACTOR_SPECS[factor].indicators)
    return set()


def parse_factor_name(name: str) -> tuple[str, Optional[str]]:
    if "@" in name:
        base, tf = name.split("@", 1)
        return base.upper(), tf
    return name.upper(), None


def column_for_factor(base: str, timeframe: Optional[str]) -> Optional[str]:
    spec = BASE_FACTOR_SPECS.get(base)
    if not spec:
        return None
    if timeframe:
        return f"{spec.column}_{timeframe.replace('/', '_')}"
    return spec.column


def apply_timeframe_to_factor(factor: str, default_tf: Optional[str]) -> str:
    """Apply default timeframe suffix if a base factor omits it."""

    base, tf = parse_factor_name(factor)
    tf = _normalize_factor_timeframe(tf)
    default_tf = _normalize_factor_timeframe(default_tf)
    if tf:
        return f"{base}@{tf}"
    if default_tf:
        return f"{base}@{default_tf}"
    return base


def factor_components_with_default(factor: str, default_tf: Optional[str]) -> tuple[str, Optional[str]]:
    """Return (base, timeframe) components with normalized default applied."""

    base, tf = parse_factor_name(factor)
    tf = _normalize_factor_timeframe(tf)
    default_tf = _normalize_factor_timeframe(default_tf)
    return base, tf or default_tf


def _normalize_factor_timeframe(tf: Optional[str]) -> Optional[str]:
    if tf is None:
        return None
    trimmed = tf.strip()
    if not trimmed:
        return None
    lowered = trimmed.lower()
    if lowered in {"primary", "main", "base"}:
        return None
    return trimmed


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

__all__ = [
    "BaseFactorSpec",
    "DerivedFactorSpec",
    "BASE_FACTOR_SPECS",
    "DERIVED_FACTOR_SPECS",
    "DEFAULT_BAG_FACTORS",
    "calculate_regime_factor",
    "factor_dependencies",
    "parse_factor_name",
    "apply_timeframe_to_factor",
    "factor_components_with_default",
    "column_for_factor",
    "FactorBank",
]
