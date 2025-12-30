# -*- coding: utf-8 -*-
"""Indicator specification and on-demand computation helpers for TaxBrainV30."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    columns: tuple[str, ...]

    def build(self, df: pd.DataFrame, cfg: Any) -> Dict[str, pd.Series]:
        raise NotImplementedError


class _EmaSpec(IndicatorSpec):
    def __init__(self, name: str, column: str, cfg_attr: str) -> None:
        super().__init__(name=name, columns=(column,))
        self._column = column
        self._cfg_attr = cfg_attr

    def build(self, df: pd.DataFrame, cfg: Any) -> Dict[str, pd.Series]:
        length = getattr(cfg, self._cfg_attr)
        return {self._column: ta.ema(df["close"], length=length)}


class _RsiSpec(IndicatorSpec):
    def __init__(self) -> None:
        super().__init__(name="RSI", columns=("rsi",))

    def build(self, df: pd.DataFrame, cfg: Any) -> Dict[str, pd.Series]:
        return {"rsi": ta.rsi(df["close"], length=cfg.rsi_len)}


class _AtrSpec(IndicatorSpec):
    def __init__(self) -> None:
        super().__init__(name="ATR", columns=("atr", "atr_pct"))

    def build(self, df: pd.DataFrame, cfg: Any) -> Dict[str, pd.Series]:
        atr = ta.atr(df["high"], df["low"], df["close"], length=cfg.atr_len)
        return {"atr": atr, "atr_pct": atr / df["close"]}


class _AdxSpec(IndicatorSpec):
    def __init__(self) -> None:
        super().__init__(name="ADX", columns=("adx",))

    def build(self, df: pd.DataFrame, cfg: Any) -> Dict[str, pd.Series]:
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=cfg.adx_len)
        target_col = f"ADX_{cfg.adx_len}"
        if isinstance(adx_df, pd.DataFrame) and target_col in adx_df.columns:
            series = adx_df[target_col]
        elif isinstance(adx_df, pd.DataFrame) and "ADX_20" in adx_df.columns:
            series = adx_df["ADX_20"]
        else:
            series = pd.Series(20.0, index=df.index)
        return {"adx": series}


class _NewBarsSpec(IndicatorSpec):
    def __init__(self) -> None:
        super().__init__(name="NEWHBARS", columns=("newbars_high", "newbars_low"))

    def build(self, df: pd.DataFrame, cfg: Any) -> Dict[str, pd.Series]:
        return {
            "newbars_high": _bars_since_new_extreme(df["high"], mode="high"),
            "newbars_low": _bars_since_new_extreme(df["low"], mode="low"),
        }


INDICATOR_SPECS: Dict[str, IndicatorSpec] = {
    "EMA_FAST": _EmaSpec("EMA_FAST", "ema_fast", "ema_fast"),
    "EMA_SLOW": _EmaSpec("EMA_SLOW", "ema_slow", "ema_slow"),
    "RSI": _RsiSpec(),
    "ATR": _AtrSpec(),
    "ADX": _AdxSpec(),
    "NEWHBARS": _NewBarsSpec(),
}

DEFAULT_INDICATORS: Set[str] = set(INDICATOR_SPECS.keys())
_OHLC_COLUMNS = ("open", "high", "low", "close", "volume")


def _col_name(base: str, suffix: Optional[str]) -> str:
    suffix = (suffix or "").strip()
    if suffix:
        return f"{base}_{suffix.replace('/', '_')}"
    return base


def compute_indicators(
    df: pd.DataFrame,
    cfg: Any,
    *,
    suffix: Optional[str] = None,
    required: Optional[Iterable[str]] = None,
    duplicate_ohlc: bool = False,
) -> pd.DataFrame:
    """Compute requested indicators and attach them to the dataframe.

    Args:
        df: DataFrame containing at least OHLCV columns.
        cfg: Strategy configuration (provides indicator lengths).
        suffix: Optional suffix (typically timeframe) appended to generated columns.
        required: Optional iterable of indicator names (case-insensitive). When omitted,
            all DEFAULT_INDICATORS are computed. Unknown names are ignored silently.
        duplicate_ohlc: When True and suffix is provided, copies OHLCV columns with
            the same suffix if they do not yet exist.
    """

    needs = _normalize_required(required)
    for key in needs:
        spec = INDICATOR_SPECS.get(key)
        if not spec:
            continue
        outputs = spec.build(df, cfg)
        for col, series in outputs.items():
            df[_col_name(col, suffix)] = series

    if suffix and duplicate_ohlc:
        _duplicate_ohlc_columns(df, suffix)
    return df


def _normalize_required(required: Optional[Iterable[str]]) -> Set[str]:
    if required is None:
        return set(DEFAULT_INDICATORS)
    normalized: Set[str] = set()
    for item in required:
        if not item:
            continue
        normalized.add(str(item).strip().upper())
    return normalized


def _duplicate_ohlc_columns(df: pd.DataFrame, suffix: str) -> None:
    for col in _OHLC_COLUMNS:
        if col not in df.columns:
            continue
        suffixed = _col_name(col, suffix)
        if suffixed in df.columns:
            continue
        df[suffixed] = df[col]


def _bars_since_new_extreme(series: pd.Series, *, mode: str) -> pd.Series:
    """Bars since the last bar $k < i$ such that $series[k]$ is an extreme value 
    that has not yet been surpassed by any $series[j]$ where $k < j \le i$."""

    if mode not in {"high", "low"}:
        raise ValueError("mode must be 'high' or 'low'")

    values = series.to_numpy(copy=False)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    stack: list[int] = []

    def should_pop(idx: int, val: float) -> bool:
        ref = values[idx]
        if np.isnan(ref):
            return True
        if mode == "high":
            return ref <= val
        return ref >= val

    for i in range(n):
        val = values[i]
        if np.isnan(val):
            stack.clear()
            continue

        while stack and should_pop(stack[-1], val):
            stack.pop()

        if stack:
            out[i] = float(i - stack[-1])
        else:
            out[i] = float(i + 1)

        stack.append(i)

    return pd.Series(out, index=series.index, name=f"newbars_{mode}")
