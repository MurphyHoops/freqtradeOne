# -*- coding: utf-8 -*-
"""Factor specifications and resolution utilities backed by declarative specs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Set, TYPE_CHECKING, TypeAlias

from ...core import math_ops
from .registry import FACTOR_REGISTRY
if TYPE_CHECKING:  # pragma: no cover - typing only
    FactorBankType: TypeAlias = "FactorBank"
else:
    FactorBankType: TypeAlias = Any


@dataclass(frozen=True)
class BaseFactorSpec:
    indicators: tuple[str, ...]
    column: str


@dataclass(frozen=True)
class DerivedFactorSpec:
    indicators: tuple[str, ...]
    fn: Callable[[FactorBankType, Optional[str]], float]
    required_factors: tuple[str, ...] = ()
    vector_fn: Optional[Callable[[pd.DataFrame, Optional[str]], Optional[pd.Series]]] = None
    vector_column: Optional[str] = None


def _suffix_token(timeframe: Optional[str]) -> str:
    token = (timeframe or "").strip()
    return token.replace("/", "_") if token else ""


def _vector_col_name(base: str, timeframe: Optional[str]) -> str:
    suffix = _suffix_token(timeframe)
    return f"{base}_{suffix}" if suffix else base


def _vector_delta_close_emafast(df: pd.DataFrame, timeframe: Optional[str]) -> Optional[pd.Series]:
    close_col = _vector_col_name("close", timeframe)
    ema_fast_col = _vector_col_name("ema_fast", timeframe)
    if close_col not in df.columns or ema_fast_col not in df.columns:
        return None
    return df[close_col] / df[ema_fast_col] - 1.0


def _vector_ema_trend(df: pd.DataFrame, timeframe: Optional[str]) -> Optional[pd.Series]:
    ema_fast_col = _vector_col_name("ema_fast", timeframe)
    ema_slow_col = _vector_col_name("ema_slow", timeframe)
    if ema_fast_col not in df.columns or ema_slow_col not in df.columns:
        return None
    fast = df[ema_fast_col]
    slow = df[ema_slow_col]
    return np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0))


CORE_FACTOR_NAMES: Set[str] = {"HURST", "ADX_ZSIG", "ATR", "CLOSE"}
CORE_INDICATORS: Set[str] = {"ADX", "ATR"}

# column references assume indicator columns use lowercase naming.
BASE_FACTOR_SPECS: Dict[str, BaseFactorSpec] = {
    "CLOSE": BaseFactorSpec(indicators=(), column="close"),
    "ATR": BaseFactorSpec(indicators=("ATR",), column="atr"),
    "HURST": BaseFactorSpec(indicators=(), column="hurst"),
    "ADX_ZSIG": BaseFactorSpec(indicators=(), column="adx_zsig"),
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


DERIVED_FACTOR_SPECS: Dict[str, DerivedFactorSpec] = {}


def _registry_base_specs() -> Dict[str, BaseFactorSpec]:
    specs: Dict[str, BaseFactorSpec] = {}
    for name, payload in FACTOR_REGISTRY.base_specs().items():
        column = payload.get("column")
        if not column:
            continue
        indicators = payload.get("indicators") or ()
        specs[str(name).upper()] = BaseFactorSpec(
            indicators=tuple(indicators),
            column=str(column),
        )
    return specs


def _registry_derived_specs() -> Dict[str, DerivedFactorSpec]:
    specs: Dict[str, DerivedFactorSpec] = {}
    for name, payload in FACTOR_REGISTRY.derived_specs().items():
        fn = payload.get("fn")
        if not callable(fn):
            continue
        indicators = payload.get("indicators") or ()
        required = payload.get("required_factors") or ()
        specs[str(name).upper()] = DerivedFactorSpec(
            indicators=tuple(indicators),
            fn=fn,
            required_factors=tuple(required),
            vector_fn=payload.get("vector_fn"),
            vector_column=payload.get("vector_column"),
        )
    return specs


def base_factor_specs() -> Dict[str, BaseFactorSpec]:
    specs = dict(BASE_FACTOR_SPECS)
    specs.update(_registry_base_specs())
    return specs


def derived_factor_specs() -> Dict[str, DerivedFactorSpec]:
    specs = dict(DERIVED_FACTOR_SPECS)
    specs.update(_registry_derived_specs())
    return specs


def get_base_factor_spec(base: str) -> Optional[BaseFactorSpec]:
    return base_factor_specs().get(base)


def get_derived_factor_spec(base: str) -> Optional[DerivedFactorSpec]:
    return derived_factor_specs().get(base)


def is_derived_factor(base: str) -> bool:
    return get_derived_factor_spec(base) is not None


def factor_vectorizable(base: str) -> bool:
    if get_base_factor_spec(base) is not None:
        return True
    derived = get_derived_factor_spec(base)
    return bool(derived and derived.vector_column)


def vector_column_for_factor(base: str, timeframe: Optional[str]) -> Optional[str]:
    derived = get_derived_factor_spec(base)
    if derived and derived.vector_column:
        return _vector_col_name(derived.vector_column, timeframe)
    return column_for_factor(base, timeframe)

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
    return math_ops.calculate_hurst_scalar(history_close, window=window, min_points=min_points)


def _compute_adx_zsig(
    history_adx: Optional[Iterable[float]],
    current_adx: float,
    window: int = 200,
    min_points: int = 50,
) -> float:
    """Compute sigmoid(z-score) for ADX using recent history; returns NaN if insufficient."""
    return math_ops.calculate_adx_zsig_scalar(
        history_adx, current_adx, window=window, min_points=min_points
    )


def calculate_regime_factor(
    bag: Any,
    strategy_type: str | None,
    history_close: Optional[Iterable[float]] = None,
    history_adx: Optional[Iterable[float]] = None,
    hurst_val: Optional[float] = None,
    z_sig: Optional[float] = None,
) -> float:
    """Return a regime multiplier (0.5-1.5) coupled to Hurst + ADX Z-Score."""
    if isinstance(hurst_val, (pd.Series, np.ndarray)) or isinstance(z_sig, (pd.Series, np.ndarray)):
        return math_ops.calculate_regime_factor_vec(strategy_type, hurst_val, z_sig)

    try:
        adx = float(bag.get("ADX"))
    except Exception:
        adx = float("nan")

    if hurst_val is None:
        try:
            hurst_val = float(bag.get("HURST"))
        except Exception:
            hurst_val = None
    if hurst_val is None:
        hurst_val = _compute_hurst_rs(history_close)
    try:
        hurst_val = float(hurst_val)
    except Exception:
        hurst_val = float("nan")
    if math.isnan(hurst_val) or math.isinf(hurst_val):
        hurst_val = 0.5

    if z_sig is None:
        try:
            z_sig = float(bag.get("ADX_ZSIG"))
        except Exception:
            z_sig = None
    if z_sig is None:
        z_sig = _compute_adx_zsig(history_adx, adx)
    try:
        z_sig = float(z_sig)
    except Exception:
        z_sig = float("nan")
    if math.isnan(z_sig) or math.isinf(z_sig):
        z_sig = 0.5

    factor = math_ops.calculate_regime_factor_vec(strategy_type, hurst_val, z_sig)
    try:
        return float(factor)
    except Exception:
        return 1.0


def ensure_regime_columns(df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if (force or "hurst" not in df.columns) and "close" in df.columns:
        try:
            df["hurst"] = math_ops.calculate_hurst_vec(df["close"])
        except Exception:
            df["hurst"] = np.nan
    if (force or "adx_zsig" not in df.columns) and "adx" in df.columns:
        try:
            df["adx_zsig"] = math_ops.calculate_adx_zsig_vec(df["adx"])
        except Exception:
            df["adx_zsig"] = np.nan
    return df


def factor_dependencies(factor: str) -> Set[str]:
    """Return indicator names required for given factor base name."""

    base_spec = get_base_factor_spec(factor)
    if base_spec:
        return set(base_spec.indicators)
    derived_spec = get_derived_factor_spec(factor)
    if derived_spec:
        return set(derived_spec.indicators)
    return set()


def parse_factor_name(name: str) -> tuple[str, Optional[str]]:
    if "@" in name:
        base, tf = name.split("@", 1)
        return base.upper(), tf
    return name.upper(), None


def column_for_factor(base: str, timeframe: Optional[str]) -> Optional[str]:
    spec = get_base_factor_spec(base)
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
        derived_spec = get_derived_factor_spec(base)
        if derived_spec:
            return derived_spec.fn(self, timeframe)
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
            and (base_spec := get_base_factor_spec(base))
            and not base_spec.indicators
        ):
            fallback_col = base_spec.column
            value = _safe_get(row, fallback_col)
        return value

__all__ = [
    "BaseFactorSpec",
    "DerivedFactorSpec",
    "BASE_FACTOR_SPECS",
    "DERIVED_FACTOR_SPECS",
    "DEFAULT_BAG_FACTORS",
    "base_factor_specs",
    "derived_factor_specs",
    "get_base_factor_spec",
    "get_derived_factor_spec",
    "is_derived_factor",
    "factor_vectorizable",
    "vector_column_for_factor",
    "calculate_regime_factor",
    "ensure_regime_columns",
    "factor_dependencies",
    "parse_factor_name",
    "apply_timeframe_to_factor",
    "factor_components_with_default",
    "column_for_factor",
    "FactorBank",
]
