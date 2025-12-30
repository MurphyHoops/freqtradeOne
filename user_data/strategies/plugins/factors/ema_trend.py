# -*- coding: utf-8 -*-
"""EMA trend factor registration."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from user_data.strategies.agents.signals.registry import register_factor


def _compose_factor(base: str, timeframe: Optional[str]) -> str:
    return base if not timeframe else f"{base}@{timeframe}"


def _vector_col_name(base: str, timeframe: Optional[str]) -> str:
    suffix = (timeframe or "").strip()
    if suffix:
        return f"{base}_{suffix.replace('/', '_')}"
    return base


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


def _vector_ema_trend(df: pd.DataFrame, timeframe: Optional[str]) -> Optional[pd.Series]:
    ema_fast_col = _vector_col_name("ema_fast", timeframe)
    ema_slow_col = _vector_col_name("ema_slow", timeframe)
    if ema_fast_col not in df.columns or ema_slow_col not in df.columns:
        return None
    fast = df[ema_fast_col]
    slow = df[ema_slow_col]
    return np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0))


@register_factor(
    name="EMA_TREND",
    compute_logic=_ema_trend,
    indicators=("EMA_FAST", "EMA_SLOW"),
    required_factors=("EMA_FAST", "EMA_SLOW"),
    vector_fn=_vector_ema_trend,
    vector_column="ema_trend",
)
def _register() -> None:
    return None
