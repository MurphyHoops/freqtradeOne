# -*- coding: utf-8 -*-
"""Unified math utilities for scalar and vector paths."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def safe_divide(numer, denom, default: float = 0.0):
    """Safely divide two scalars, returning default on zero/invalid."""

    try:
        if denom in (0, 0.0, None):
            return default
        return float(numer) / float(denom)
    except Exception:
        return default


def rolling_ema(series: pd.Series, span: int) -> pd.Series:
    """Return exponential moving average with a stable setup."""

    series = pd.to_numeric(series, errors="coerce")
    return series.ewm(span=span, adjust=False).mean()


def calculate_hurst_vec(
    close_series: pd.Series,
    window: int = 200,
    min_points: int = 50,
) -> pd.Series:
    """Vectorized Hurst exponent (R/S) for a price series."""

    if close_series is None:
        return pd.Series(dtype="float64")
    close_series = pd.to_numeric(close_series, errors="coerce")
    try:
        returns = close_series.pct_change().replace([np.inf, -np.inf], np.nan)
        mean_ret = returns.rolling(window, min_periods=min_points).mean()
        dev = returns - mean_ret
        cum_dev = dev.rolling(window, min_periods=min_points).sum()
        roll = cum_dev.rolling(window, min_periods=min_points)
        r_range = roll.max() - roll.min()
        s_std = dev.rolling(window, min_periods=min_points).std()
        count = dev.rolling(window, min_periods=min_points).count()
        hurst = np.log(r_range / s_std) / np.log(np.maximum(1.0, count))
        return hurst.clip(lower=0.0, upper=1.0)
    except Exception:
        return pd.Series(np.nan, index=close_series.index, dtype="float64")


def calculate_adx_zsig_vec(
    adx_series: pd.Series,
    window: int = 200,
    min_points: int = 50,
) -> pd.Series:
    """Vectorized sigmoid(z-score) for ADX."""

    if adx_series is None:
        return pd.Series(dtype="float64")
    adx_series = pd.to_numeric(adx_series, errors="coerce")
    mu = adx_series.rolling(window, min_periods=min_points).mean()
    sigma = adx_series.rolling(window, min_periods=min_points).std()
    z = (adx_series - mu) / sigma.replace(0.0, np.nan)
    return 1.0 / (1.0 + np.exp(-z))


def _to_series(value: Any, index: pd.Index, default: float = 0.5) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.reindex(index)
    elif isinstance(value, np.ndarray):
        series = pd.Series(value, index=index)
    else:
        try:
            scalar = float(value) if value is not None else default
        except Exception:
            scalar = default
        series = pd.Series(scalar, index=index)
    series = pd.to_numeric(series, errors="coerce")
    values = np.nan_to_num(series.to_numpy(copy=False), nan=default, posinf=default, neginf=default)
    return pd.Series(values, index=index)


def calculate_regime_factor_vec(
    strategy_type: Optional[str],
    hurst_val: Any,
    z_sig: Any,
) -> Any:
    """Return regime factor for scalar/array/Series inputs, clamped to 0.5-1.5."""

    bias = (strategy_type or "").lower()
    is_trend = any(token in bias for token in ("trend", "breakout"))
    is_mean_rev = any(token in bias for token in ("mean_rev", "pullback"))

    if isinstance(hurst_val, pd.Series) or isinstance(z_sig, pd.Series):
        index = (
            hurst_val.index
            if isinstance(hurst_val, pd.Series)
            else z_sig.index  # type: ignore[union-attr]
        )
        hurst_series = _to_series(hurst_val, index)
        zsig_series = _to_series(z_sig, index)
        if is_trend and not is_mean_rev:
            raw_factor = 0.7 * hurst_series + 0.3 * zsig_series
            factor = 1.0 + (raw_factor - 0.5)
        elif is_mean_rev and not is_trend:
            raw_factor = (0.5 - hurst_series) * 2.0
            factor = 1.0 + raw_factor
        else:
            raw_factor = 0.5 * hurst_series + 0.5 * zsig_series
            factor = 1.0 + 0.5 * (raw_factor - 0.5)
        return factor.clip(lower=0.5, upper=1.5)

    hurst_arr = np.nan_to_num(
        np.asarray(0.5 if hurst_val is None else hurst_val, dtype=float),
        nan=0.5,
        posinf=0.5,
        neginf=0.5,
    )
    zsig_arr = np.nan_to_num(
        np.asarray(0.5 if z_sig is None else z_sig, dtype=float),
        nan=0.5,
        posinf=0.5,
        neginf=0.5,
    )
    if is_trend and not is_mean_rev:
        raw_factor = 0.7 * hurst_arr + 0.3 * zsig_arr
        factor = 1.0 + (raw_factor - 0.5)
    elif is_mean_rev and not is_trend:
        raw_factor = (0.5 - hurst_arr) * 2.0
        factor = 1.0 + raw_factor
    else:
        raw_factor = 0.5 * hurst_arr + 0.5 * zsig_arr
        factor = 1.0 + 0.5 * (raw_factor - 0.5)
    return np.clip(factor, 0.5, 1.5)


def calculate_hurst_scalar(
    history_close: Optional[Iterable[float]],
    window: int = 200,
    min_points: int = 50,
) -> float:
    """Scalar Hurst exponent for recent close history."""

    if history_close is None:
        return float("nan")
    try:
        series = pd.Series(history_close)
    except Exception:
        return float("nan")
    series = pd.to_numeric(series, errors="coerce")
    if series.empty:
        return float("nan")
    hurst = calculate_hurst_vec(series, window=window, min_points=min_points)
    if hurst.empty:
        return float("nan")
    try:
        value = float(hurst.iloc[-1])
    except Exception:
        return float("nan")
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    return max(0.0, min(1.0, value))


def calculate_adx_zsig_scalar(
    history_adx: Optional[Iterable[float]],
    current_adx: float,
    window: int = 200,
    min_points: int = 50,
) -> float:
    """Scalar sigmoid(z-score) for ADX history."""

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
    denom = len(values) - 1
    if denom <= 0:
        return float("nan")
    variance = sum((v - mu) ** 2 for v in values) / denom
    sigma = math.sqrt(variance)
    if sigma <= 1e-9:
        return float("nan")
    z_val = (current_adx - mu) / sigma
    return 1.0 / (1.0 + math.exp(-z_val))


__all__ = [
    "safe_divide",
    "rolling_ema",
    "calculate_hurst_vec",
    "calculate_adx_zsig_vec",
    "calculate_hurst_scalar",
    "calculate_adx_zsig_scalar",
    "calculate_regime_factor_vec",
]
