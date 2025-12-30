# -*- coding: utf-8 -*-
"""Vectorized helpers for signal prefiltering in backtests."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .registry import REGISTRY
from . import factors


class _SeriesCache:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._cache: Dict[tuple[str, Optional[str]], pd.Series] = {}

    def get(self, base: str, timeframe: Optional[str]) -> pd.Series:
        key = (base, timeframe)
        if key in self._cache:
            return self._cache[key]
        series = factor_series(self._df, base, timeframe)
        if series is None:
            series = pd.Series(np.nan, index=self._df.index)
        series = pd.to_numeric(series, errors="coerce")
        self._cache[key] = series
        return series


def add_derived_factor_columns(
    df: pd.DataFrame,
    timeframes: Iterable[Optional[str]],
    required: Optional[Dict[Optional[str], set[str]]] = None,
) -> None:
    """Add derived factor columns for given timeframes."""

    if df is None or df.empty:
        return
    derived_specs = factors.derived_factor_specs()
    for tf in timeframes:
        needed = None
        if required is not None:
            needed = {name for name in required.get(tf, set()) if factors.is_derived_factor(name)}
        for base, spec in derived_specs.items():
            if needed is not None and base not in needed:
                continue
            if not spec.vector_column or not spec.vector_fn:
                continue
            col = factors.vector_column_for_factor(base, tf)
            if not col or col in df.columns:
                continue
            try:
                series = spec.vector_fn(df, tf)
            except Exception:
                continue
            if series is None:
                continue
            df[col] = series


def factor_series(df: pd.DataFrame, base: str, timeframe: Optional[str]) -> Optional[pd.Series]:
    """Return a Series for the requested factor name/timeframe, or None if missing."""

    col = factors.vector_column_for_factor(base, timeframe)
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
    cache = _SeriesCache(df)
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
            if not factors.factor_vectorizable(base):
                spec_mask = pd.Series(True, index=df.index)
                break
            series = cache.get(base, tf)
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
    if not spec:
        return False
    if not getattr(spec, "vec_raw_fn", None) or not getattr(spec, "vec_win_prob_fn", None):
        return False
    for cond in spec.conditions:
        if getattr(cond, "fn", None) is not None:
            return False
        base, _ = factors.factor_components_with_default(cond.factor, spec.timeframe)
        if not factors.factor_vectorizable(base):
            return False
    return True


def _bool_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0) != 0.0


def _condition_mask(cache: _SeriesCache, df: pd.DataFrame, spec) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for cond in spec.conditions:
        op = getattr(cond, "op", None)
        base, tf = factors.factor_components_with_default(cond.factor, spec.timeframe)
        series = cache.get(base, tf)
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


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def _spec_series_getter(cache: _SeriesCache, spec) -> Any:
    def _get(base: str, timeframe: Optional[str] = None) -> pd.Series:
        tf = timeframe if timeframe is not None else spec.timeframe
        return cache.get(base, tf)

    return _get


def build_signal_matrices(df: pd.DataFrame, cfg, specs: Sequence[Any]) -> list[Dict[str, Any]]:
    """Build vectorized signal matrices for supported specs."""

    if df is None or df.empty:
        return []
    factors.ensure_regime_columns(df, force=False)
    cache = _SeriesCache(df)
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
        cond_mask = _condition_mask(cache, df, spec)
        recipe = entry_to_recipe.get(spec.name)
        exit_profile_name = recipe.exit_profile if recipe else default_profile
        profile = profiles.get(exit_profile_name) if exit_profile_name else None
        if profile is None:
            continue
        target_tf = getattr(profile, "atr_timeframe", None) or spec.timeframe
        atr_pct = cache.get("ATR_PCT", target_tf)
        k_sl = float(getattr(profile, "atr_mul_sl", 0.0) or 0.0)
        floor = float(getattr(profile, "floor_sl_pct", 0.0) or 0.0)
        valid_atr = atr_pct > 0
        sl_pct = (atr_pct * k_sl).where(valid_atr, np.nan)
        if floor > 0:
            sl_pct = sl_pct.clip(lower=floor)
        atr_mul_tp = getattr(profile, "atr_mul_tp", None)
        if atr_mul_tp is not None and atr_mul_tp > 0:
            tp_pct = (atr_pct * float(atr_mul_tp)).where(valid_atr, np.nan)
        else:
            tp_pct = (sl_pct * 2.0).clip(lower=0.0)

        rr_ratio = tp_pct / sl_pct.replace(0.0, np.nan)
        get_series = _spec_series_getter(cache, spec)
        try:
            raw = _as_series(spec.vec_raw_fn(get_series, cfg, spec.timeframe), df.index)
        except Exception:
            raw = pd.Series(np.nan, index=df.index)
        try:
            win_prob = _as_series(spec.vec_win_prob_fn(get_series, cfg, raw, spec.timeframe), df.index)
        except Exception:
            win_prob = pd.Series(np.nan, index=df.index)
        raw = pd.to_numeric(raw, errors="coerce")
        win_prob = pd.to_numeric(win_prob, errors="coerce")

        strat_name = recipe.name if recipe else spec.name
        strat_spec = strategy_map.get(strat_name)
        base_wp = float(getattr(strat_spec, "base_win_prob", getattr(spec, "base_win_prob", 0.5)))
        win_prob = win_prob.replace([np.inf, -np.inf], np.nan).fillna(base_wp)

        hurst = pd.to_numeric(df["hurst"], errors="coerce") if "hurst" in df.columns else pd.Series(0.5, index=df.index)
        z_sig = pd.to_numeric(df["adx_zsig"], errors="coerce") if "adx_zsig" in df.columns else pd.Series(0.5, index=df.index)
        hurst = hurst.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        z_sig = z_sig.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        regime_factor = factors.calculate_regime_factor({}, strat_name, hurst_val=hurst, z_sig=z_sig)
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
                "timeframe": spec.timeframe,
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
