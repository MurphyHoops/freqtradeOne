# -*- coding: utf-8 -*-
"""Vectorized helpers for signal prefiltering in backtests."""

from __future__ import annotations

from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

from .registry import REGISTRY
from . import factors


def _suffix_token(timeframe: Optional[str]) -> str:
    token = (timeframe or "").strip()
    return token.replace("/", "_") if token else ""


def _col_name(base: str, timeframe: Optional[str]) -> str:
    suffix = _suffix_token(timeframe)
    return f"{base}_{suffix}" if suffix else base


def add_derived_factor_columns(df: pd.DataFrame, timeframes: Iterable[Optional[str]]) -> None:
    """Add derived factor columns (delta_close_emafast_pct, ema_trend) for given timeframes."""

    for tf in timeframes:
        close_col = _col_name("close", tf)
        ema_fast_col = _col_name("ema_fast", tf)
        ema_slow_col = _col_name("ema_slow", tf)
        delta_col = _col_name("delta_close_emafast_pct", tf)
        trend_col = _col_name("ema_trend", tf)

        if close_col in df.columns and ema_fast_col in df.columns:
            df[delta_col] = df[close_col] / df[ema_fast_col] - 1.0

        if ema_fast_col in df.columns and ema_slow_col in df.columns:
            fast = df[ema_fast_col]
            slow = df[ema_slow_col]
            df[trend_col] = np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0))


def factor_series(df: pd.DataFrame, base: str, timeframe: Optional[str]) -> Optional[pd.Series]:
    """Return a Series for the requested factor name/timeframe, or None if missing."""

    derived_map = {
        "DELTA_CLOSE_EMAFAST_PCT": "delta_close_emafast_pct",
        "EMA_TREND": "ema_trend",
    }
    if base in derived_map:
        col = _col_name(derived_map[base], timeframe)
        return df[col] if col in df.columns else None

    col = factors.column_for_factor(base, timeframe)
    if not col:
        return None
    return df[col] if col in df.columns else None


def prefilter_signal_mask(df: pd.DataFrame, cfg) -> pd.Series:
    """Return a boolean mask for rows that can possibly pass signal conditions."""

    enabled = {
        name
        for name in (
            getattr(getattr(cfg, "strategy", None), "enabled_signals", getattr(cfg, "enabled_signals", ())) or ()
        )
        if name
    }
    mask_any = pd.Series(False, index=df.index)
    for spec in REGISTRY.all():
        if enabled and spec.name not in enabled:
            continue
        spec_mask = pd.Series(True, index=df.index)
        for cond in spec.conditions:
            if getattr(cond, "fn", None) is not None:
                continue
            op = getattr(cond, "op", None)
            if op is None:
                continue
            base, tf = factors.factor_components_with_default(cond.factor, spec.timeframe)
            series = factor_series(df, base, tf)
            if series is None:
                spec_mask = spec_mask & False
                break
            if op in ("<", "<=", ">", ">=", "=="):
                value = getattr(cond, "value", None)
                if value is None:
                    spec_mask = spec_mask & False
                    break
                if op == "<":
                    spec_mask = spec_mask & (series < value)
                elif op == "<=":
                    spec_mask = spec_mask & (series <= value)
                elif op == ">":
                    spec_mask = spec_mask & (series > value)
                elif op == ">=":
                    spec_mask = spec_mask & (series >= value)
                else:
                    spec_mask = spec_mask & (series == value)
                continue
            if op in ("between", "outside"):
                lo = getattr(cond, "value", None)
                hi = getattr(cond, "value_hi", None)
                if lo is None or hi is None:
                    spec_mask = spec_mask & False
                    break
                if op == "between":
                    spec_mask = spec_mask & (series >= lo) & (series <= hi)
                else:
                    spec_mask = spec_mask & ((series <= lo) | (series >= hi))
                continue
            # Unknown operator: keep conservative (no filtering)
        mask_any = mask_any | spec_mask
    return mask_any.fillna(False)

