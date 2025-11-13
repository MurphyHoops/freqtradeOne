# -*- coding: utf-8 -*-
"""Central catalogue of exit profile definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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
    "PROFILE_LIBRARY",
    "DEFAULT_PROFILE_VERSION",
    "resolve_profiles",
]
