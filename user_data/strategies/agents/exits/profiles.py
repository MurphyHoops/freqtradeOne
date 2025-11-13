# -*- coding: utf-8 -*-
"""Central catalogue of exit profile definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ExitProfile:
    """Unified parameters describing how exits should behave."""

    atr_timeframe: Optional[str] = None
    atr_mul_sl: float = 1.0
    floor_sl_pct: float = 0.0
    atr_mul_tp: Optional[float] = None
    breakeven_lock_frac_of_tp: Optional[float] = None
    trail_mode: Optional[str] = None  # e.g. "chandelier" / "percent"
    trail_atr_mul: Optional[float] = None
    trail_pct: Optional[float] = None
    activation_atr_mul: Optional[float] = None
    max_bars_in_trade: Optional[int] = None
    ladder: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ProfilePlan:
    """Resolved SL/TP percentages for a given profile."""

    profile_name: str
    timeframe: Optional[str]
    atr_pct: float
    sl_pct: float
    tp_pct: float


def compute_plan_from_atr(profile_name: str, profile: ExitProfile, atr_pct: float) -> Optional[ProfilePlan]:
    """Return SL/TP percentages (as fraction of entry) given atr_pct."""

    if atr_pct is None or atr_pct <= 0 or profile is None:
        return None
    k_sl = profile.atr_mul_sl if profile.atr_mul_sl and profile.atr_mul_sl > 0 else 0.0
    floor = profile.floor_sl_pct or 0.0
    sl_pct = max(floor, atr_pct * k_sl)

    tp_pct: float
    if profile.atr_mul_tp and profile.atr_mul_tp > 0:
        tp_pct = atr_pct * profile.atr_mul_tp
    else:
        tp_pct = max(sl_pct * 2.0, 0.0)

    return ProfilePlan(
        profile_name=profile_name,
        timeframe=getattr(profile, "atr_timeframe", None),
        atr_pct=float(atr_pct),
        sl_pct=float(sl_pct),
        tp_pct=float(tp_pct),
    )


def atr_pct_from_rows(
    main_row: pd.Series,
    informative_rows: Dict[str, pd.Series],
    *,
    target_timeframe: Optional[str],
    main_timeframe: str,
) -> Optional[float]:
    """Fetch ATR% from current dataframe rows (main/informative)."""

    tf_norm = _normalize_tf(target_timeframe)
    main_tf_norm = _normalize_tf(main_timeframe)

    if tf_norm is None or tf_norm == main_tf_norm:
        return _series_value(main_row, "atr_pct")

    info_row = informative_rows.get(tf_norm)
    if info_row is None:
        return None
    return _series_value(info_row, "atr_pct")


def atr_pct_from_dp(dp, pair: str, timeframe: str, current_time) -> Optional[float]:
    """Read ATR% from DataProvider for the requested timeframe."""

    if dp is None:
        return None
    try:
        analyzed = dp.get_analyzed_dataframe(pair, timeframe)
    except Exception:
        return None
    df = analyzed[0] if isinstance(analyzed, (list, tuple)) else analyzed
    if df is None or len(df) == 0:
        return None
    try:
        upto = df.loc[:current_time] if current_time is not None else df
    except Exception:
        try:
            ct = current_time.replace(tzinfo=None) if getattr(current_time, "tzinfo", None) else current_time
            upto = df.loc[:ct]
        except Exception:
            upto = df
    if upto is None or len(upto) == 0:
        return None
    row = upto.iloc[-1]
    value = _series_value(row, "atr_pct")
    if value is not None and value > 0:
        return value

    # fallback: compute atr_pct from atr + close if available
    try:
        atr = float(row.get("atr"))
        close = float(row.get("close"))
        if close and close > 0:
            return atr / close
    except Exception:
        return None
    return None


def _series_value(row, column: str) -> Optional[float]:
    try:
        value = row.get(column) if isinstance(row, (pd.Series, dict)) else getattr(row, column, None)
    except Exception:
        value = None
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if value != value or value <= 0:
        return None
    return value


def _normalize_tf(timeframe: Optional[str]) -> Optional[str]:
    if timeframe is None:
        return None
    trimmed = timeframe.strip()
    if not trimmed:
        return None
    lowered = trimmed.lower()
    if lowered in {"primary", "main", "base"}:
        return None
    return trimmed


PROFILE_LIBRARY: Dict[str, Dict[str, ExitProfile]] = {
    "v1": {
        "ATRtrail_v1": ExitProfile(
            atr_timeframe=None,
            atr_mul_sl=8.0,
            floor_sl_pct=0.006,
            atr_mul_tp=2.0,
            breakeven_lock_frac_of_tp=0.5,
            trail_mode="chandelier",
            trail_atr_mul=2.5,
            activation_atr_mul=1.5,
            max_bars_in_trade=240,
        )
    }
}

DEFAULT_PROFILE_VERSION = "v1"


def resolve_profiles(version: Optional[str] = None) -> Dict[str, ExitProfile]:
    """Return a copy of the requested profile library."""

    name = version or DEFAULT_PROFILE_VERSION
    try:
        profiles = PROFILE_LIBRARY[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown exit profile version '{name}'") from exc
    return dict(profiles)


__all__ = [
    "ExitProfile",
    "ProfilePlan",
    "PROFILE_LIBRARY",
    "DEFAULT_PROFILE_VERSION",
    "resolve_profiles",
    "compute_plan_from_atr",
    "atr_pct_from_rows",
    "atr_pct_from_dp",
]
