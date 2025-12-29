# -*- coding: utf-8 -*-
"""Vectorized helpers for signal prefiltering in backtests."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .registry import REGISTRY
from . import factors

_NEWBARS_THRESHOLD = 80.0
_VEC_SIGNAL_NAMES = {
    "mean_rev_long",
    "pullback_long",
    "trend_short",
    "newbars_breakout_long_5m",
    "newbars_breakdown_short_5m",
    "newbars_breakout_long_30m",
    "newbars_breakdown_short_30m",
}


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


def prefilter_signal_mask(df: pd.DataFrame, cfg, specs: Optional[Iterable] = None) -> pd.Series:
    """Return a boolean mask for rows that can possibly pass signal conditions."""

    enabled = {
        name
        for name in (
            getattr(getattr(cfg, "strategy", None), "enabled_signals", getattr(cfg, "enabled_signals", ())) or ()
        )
        if name
    }
    mask_any = pd.Series(False, index=df.index)
    specs_iter = specs if specs is not None else REGISTRY.all()
    for spec in specs_iter:
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


def is_vectorizable(spec) -> bool:
    if not spec or spec.name not in _VEC_SIGNAL_NAMES:
        return False
    for cond in spec.conditions:
        if getattr(cond, "fn", None) is not None:
            return False
    return True


def _bool_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0) != 0.0


def _condition_mask(df: pd.DataFrame, spec) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for cond in spec.conditions:
        op = getattr(cond, "op", None)
        base, tf = factors.factor_components_with_default(cond.factor, spec.timeframe)
        series = factor_series(df, base, tf)
        if series is None:
            return pd.Series(False, index=df.index)
        series = pd.to_numeric(series, errors="coerce")
        if op is None:
            mask = mask & _bool_series(series)
            continue
        if op in ("<", "<=", ">", ">=", "=="):
            value = getattr(cond, "value", None)
            if value is None:
                return pd.Series(False, index=df.index)
            if op == "<":
                mask = mask & (series < value)
            elif op == "<=":
                mask = mask & (series <= value)
            elif op == ">":
                mask = mask & (series > value)
            elif op == ">=":
                mask = mask & (series >= value)
            else:
                mask = mask & (series == value)
            continue
        if op in ("between", "outside"):
            lo = getattr(cond, "value", None)
            hi = getattr(cond, "value_hi", None)
            if lo is None or hi is None:
                return pd.Series(False, index=df.index)
            if op == "between":
                mask = mask & (series >= lo) & (series <= hi)
            else:
                mask = mask & ((series <= lo) | (series >= hi))
            continue
    return mask.fillna(False)


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def _shape_score(series: pd.Series, exponent: float) -> pd.Series:
    try:
        exp = float(exponent)
    except Exception:
        exp = 1.0
    base = _clip01(series)
    if exp <= 0:
        return base
    if exp == 1.0:
        return base
    return _clip01(base ** exp)


def _regime_factor(df: pd.DataFrame, strat_name: str) -> pd.Series:
    if "hurst" in df.columns:
        hurst = pd.to_numeric(df["hurst"], errors="coerce").fillna(0.5)
    else:
        hurst = pd.Series(0.5, index=df.index)
    if "adx_zsig" in df.columns:
        z_sig = pd.to_numeric(df["adx_zsig"], errors="coerce").fillna(0.5)
    else:
        z_sig = pd.Series(0.5, index=df.index)
    bias = (strat_name or "").lower()
    is_trend = any(token in bias for token in ("trend", "breakout"))
    is_mean_rev = any(token in bias for token in ("mean_rev", "pullback"))
    if is_trend and not is_mean_rev:
        raw = 0.7 * hurst + 0.3 * z_sig
        factor = 1.0 + (raw - 0.5)
    elif is_mean_rev and not is_trend:
        raw = (0.5 - hurst) * 2.0
        factor = 1.0 + raw
    else:
        raw = 0.5 * hurst + 0.5 * z_sig
        factor = 1.0 + 0.5 * (raw - 0.5)
    return factor.clip(lower=0.5, upper=1.5)


def _vec_raw_winprob(
    df: pd.DataFrame, spec_name: str, timeframe: Optional[str]
) -> tuple[pd.Series, pd.Series]:
    def _get_col(base: str) -> pd.Series:
        col = _col_name(base, timeframe)
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    if spec_name == "mean_rev_long":
        rsi = _get_col("rsi")
        raw = ((25.0 - rsi) / 25.0).clip(lower=0.0)
        win = (0.52 + 0.4 * raw).clip(lower=0.5, upper=0.9)
        return raw, win
    if spec_name == "pullback_long":
        ema_fast = _get_col("ema_fast")
        ema_slow = _get_col("ema_slow")
        adx = _get_col("adx")
        trend = (ema_fast / ema_slow - 1.0).clip(lower=0.0)
        strength = ((adx - 20.0) / 20.0).clip(lower=0.0)
        raw = 0.5 * trend + 0.5 * strength
        win = (0.55 + 0.4 * raw).clip(lower=0.5, upper=0.95)
        return raw, win
    if spec_name == "trend_short":
        ema_fast = _get_col("ema_fast")
        ema_slow = _get_col("ema_slow")
        adx = _get_col("adx")
        strength = ((adx - 25.0) / 25.0).clip(lower=0.0)
        trend = (1.0 - (ema_fast / ema_slow)).clip(lower=0.0)
        raw = 0.5 * strength + 0.5 * trend
        win = (0.50 + 0.4 * raw).clip(lower=0.5, upper=0.95)
        return raw, win
    if spec_name in {"newbars_breakout_long_5m", "newbars_breakout_long_30m"}:
        series = _get_col("newbars_high")
        raw = (series / max(_NEWBARS_THRESHOLD, 1e-9)).clip(lower=0.0, upper=1.0)
        win = (0.55 + 0.35 * raw).clip(upper=0.95)
        return raw, win
    if spec_name in {"newbars_breakdown_short_5m", "newbars_breakdown_short_30m"}:
        series = _get_col("newbars_low")
        raw = (series / max(_NEWBARS_THRESHOLD, 1e-9)).clip(lower=0.0, upper=1.0)
        win = (0.55 + 0.35 * raw).clip(upper=0.95)
        return raw, win
    return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)


def _atr_pct_series(df: pd.DataFrame, timeframe: Optional[str]) -> pd.Series:
    col = _col_name("atr_pct", timeframe)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def build_signal_matrices(df: pd.DataFrame, cfg, specs: Sequence[Any]) -> list[Dict[str, Any]]:
    """Build vectorized signal matrices for supported specs."""

    if df is None or df.empty:
        return []
    profiles = getattr(getattr(cfg, "strategy", None), "exit_profiles", getattr(cfg, "exit_profiles", {})) or {}
    default_profile = getattr(
        getattr(cfg, "strategy", None), "default_exit_profile", getattr(cfg, "default_exit_profile", None)
    )
    entry_to_recipe: Dict[str, Any] = {}
    for recipe in getattr(cfg, "strategy_recipes", ()) or ():
        for entry in recipe.entries:
            entry_to_recipe[entry] = recipe
    strategy_map = getattr(getattr(cfg, "strategy", None), "strategies", getattr(cfg, "strategies", {})) or {}
    gcfg = getattr(getattr(cfg, "risk", None), "gatekeeping", getattr(cfg, "gatekeeping", None))
    score_exp = float(getattr(gcfg, "score_curve_exponent", 1.0) or 1.0) if gcfg else 1.0

    matrices: list[Dict[str, Any]] = []
    for spec in specs:
        if not is_vectorizable(spec):
            continue
        cond_mask = _condition_mask(df, spec)
        recipe = entry_to_recipe.get(spec.name)
        exit_profile_name = recipe.exit_profile if recipe else default_profile
        profile = profiles.get(exit_profile_name) if exit_profile_name else None
        if profile is None:
            continue
        target_tf = getattr(profile, "atr_timeframe", None) or spec.timeframe
        atr_pct = _atr_pct_series(df, target_tf)
        k_sl = float(getattr(profile, "atr_mul_sl", 0.0) or 0.0)
        floor = float(getattr(profile, "floor_sl_pct", 0.0) or 0.0)
        sl_pct = (atr_pct * k_sl).where(atr_pct > 0, np.nan)
        sl_pct = pd.concat([sl_pct, pd.Series(floor, index=df.index)], axis=1).max(axis=1)
        atr_mul_tp = getattr(profile, "atr_mul_tp", None)
        if atr_mul_tp is not None and atr_mul_tp > 0:
            tp_pct = atr_pct * float(atr_mul_tp)
        else:
            tp_pct = (sl_pct * 2.0).clip(lower=0.0)

        rr_ratio = tp_pct / sl_pct.replace(0.0, np.nan)
        raw, win_prob = _vec_raw_winprob(df, spec.name, spec.timeframe)

        strat_name = recipe.name if recipe else spec.name
        strat_spec = strategy_map.get(strat_name)
        base_wp = float(getattr(strat_spec, "base_win_prob", getattr(spec, "base_win_prob", 0.5)))
        win_prob = win_prob.replace([np.inf, -np.inf], np.nan).fillna(base_wp)

        regime_factor = _regime_factor(df, strat_name)
        final_score = _clip01(win_prob * regime_factor)
        score_for_use = _shape_score(final_score, score_exp)

        min_rr = recipe.min_rr if recipe and recipe.min_rr is not None else spec.min_rr
        min_edge = recipe.min_edge if recipe and recipe.min_edge is not None else spec.min_edge

        valid = (
            cond_mask
            & (sl_pct > 0)
            & (tp_pct > 0)
            & (rr_ratio >= float(min_rr))
            & (score_for_use >= float(min_edge))
        )

        matrices.append(
            {
                "name": spec.name,
                "direction": spec.direction,
                "squad": spec.squad,
                "exit_profile": exit_profile_name,
                "recipe": recipe.name if recipe else None,
                "plan_timeframe": getattr(profile, "atr_timeframe", None),
                "plan_atr_pct": atr_pct,
                "raw_score": raw,
                "win_prob": score_for_use,
                "expected_edge": score_for_use,
                "rr_ratio": rr_ratio,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "valid_mask": valid,
            }
        )
    return matrices
