# -*- coding: utf-8 -*-
"""Factor specification registry for signal module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .factors import FactorBank


@dataclass(frozen=True)
class BaseFactorSpec:
    indicators: tuple[str, ...]
    column: str


@dataclass(frozen=True)
class DerivedFactorSpec:
    indicators: tuple[str, ...]
    fn: Callable[["FactorBank", Optional[str]], float]


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


def _delta_close_emafast(fb, timeframe: Optional[str]) -> float:
    close = fb.get(_compose_factor("CLOSE", timeframe))
    ema_fast = fb.get(_compose_factor("EMA_FAST", timeframe))
    if ema_fast in (0, None):
        return float("nan")
    return close / ema_fast - 1.0


def _ema_trend(fb, timeframe: Optional[str]) -> float:
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

DEFAULT_BAG_FACTORS = ("ATR", "ATR_PCT")


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
    """根据默认 timeframe 返回最终的因子名称。"""

    base, tf = parse_factor_name(factor)
    tf = _normalize_factor_timeframe(tf)
    default_tf = _normalize_factor_timeframe(default_tf)
    if tf:
        return f"{base}@{tf}"
    if default_tf:
        return f"{base}@{default_tf}"
    return base


def factor_components_with_default(factor: str, default_tf: Optional[str]) -> tuple[str, Optional[str]]:
    """返回 (base, timeframe) 二元组，timeframe 未显式声明时使用默认。"""

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


try:
    from .factors import FactorBank  # type: ignore # circular typing hint
except Exception:  # pragma: no cover
    FactorBank = "FactorBank"
