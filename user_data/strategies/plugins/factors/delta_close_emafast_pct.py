# -*- coding: utf-8 -*-
"""Delta close vs EMA fast factor registration."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from user_data.strategies.agents.signals.registry import register_factor


def _compose_factor(base: str, timeframe: Optional[str]) -> str:
    return base if not timeframe else f"{base}@{timeframe}"


def _vector_col_name(base: str, timeframe: Optional[str]) -> str:
    suffix = (timeframe or "").strip()
    if suffix:
        return f"{base}_{suffix.replace('/', '_')}"
    return base


def _delta_close_emafast(fb, timeframe: Optional[str]) -> float:
    close = fb.get(_compose_factor("CLOSE", timeframe))
    ema_fast = fb.get(_compose_factor("EMA_FAST", timeframe))
    if ema_fast in (0, None):
        return float("nan")
    return close / ema_fast - 1.0


def _vector_delta_close_emafast(
    df: pd.DataFrame, timeframe: Optional[str]
) -> Optional[pd.Series]:
    close_col = _vector_col_name("close", timeframe)
    ema_fast_col = _vector_col_name("ema_fast", timeframe)
    if close_col not in df.columns or ema_fast_col not in df.columns:
        return None
    return df[close_col] / df[ema_fast_col] - 1.0


@register_factor(
    name="DELTA_CLOSE_EMAFAST_PCT",
    compute_logic=_delta_close_emafast,
    indicators=("EMA_FAST",),
    required_factors=("CLOSE", "EMA_FAST"),
    vector_fn=_vector_delta_close_emafast,
    vector_column="delta_close_emafast_pct",
)
def _register() -> None:
    return None
