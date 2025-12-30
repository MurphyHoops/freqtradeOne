# -*- coding: utf-8 -*-
"""Trend short signal plug-ins."""

from __future__ import annotations

import numpy as np

from user_data.strategies.agents.signals.registry import register_signal
from user_data.strategies.agents.signals.schemas import Condition


def _trend_short_raw_vec(get, _cfg, timeframe):
    ema_fast = get("EMA_FAST", timeframe)
    ema_slow = get("EMA_SLOW", timeframe)
    adx = get("ADX", timeframe)
    strength = ((adx - 25.0) / 25.0).clip(lower=0.0)
    denom = np.maximum(ema_slow, 1e-9)
    trend = (1.0 - ema_fast / denom).clip(lower=0.0)
    return 0.5 * strength + 0.5 * trend


def _trend_short_win_prob_vec(_get, _cfg, raw, _timeframe):
    return (0.50 + 0.4 * raw).clip(lower=0.5, upper=0.95)


@register_signal(
    name="trend_short",
    direction="short",
    squad="TRS",
    conditions=[
        Condition("EMA_TREND", "==", -1.0),
        Condition("ADX", ">", 25.0),
        Condition("DELTA_CLOSE_EMAFAST_PCT", ">", 0.01),
    ],
    raw_fn=lambda bag, cfg: 0.5 * max(0.0, (bag["ADX"] - 25.0) / 25.0)
    + 0.5 * max(0.0, 1.0 - bag["EMA_FAST"] / max(bag["EMA_SLOW"], 1e-9)),
    win_prob_fn=lambda bag, cfg, raw: min(0.95, max(0.5, 0.50 + 0.4 * raw)),
    vec_raw_fn=_trend_short_raw_vec,
    vec_win_prob_fn=_trend_short_win_prob_vec,
    min_rr=1.2,
    min_edge=0.0,
)
def trend_short() -> None:
    """Trend short."""
