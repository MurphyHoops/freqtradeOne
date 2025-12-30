# -*- coding: utf-8 -*-
"""Mean reversion signal plug-ins."""

from __future__ import annotations

import numpy as np

from user_data.strategies.agents.signals.registry import register_signal
from user_data.strategies.agents.signals.schemas import Condition


def _mean_rev_raw_vec(get, _cfg, timeframe):
    rsi = get("RSI", timeframe)
    return ((25.0 - rsi) / 25.0).clip(lower=0.0)


def _mean_rev_win_prob_vec(_get, _cfg, raw, _timeframe):
    return (0.52 + 0.4 * raw).clip(lower=0.5, upper=0.9)


@register_signal(
    name="mean_rev_long",
    direction="long",
    squad="MRL",
    conditions=[
        Condition("RSI", "<", 25.0),
        Condition("DELTA_CLOSE_EMAFAST_PCT", "<", -0.015),
    ],
    raw_fn=lambda bag, cfg: max(0.0, (25.0 - bag["RSI"]) / 25.0),
    win_prob_fn=lambda bag, cfg, raw: min(0.9, max(0.5, 0.52 + 0.4 * raw)),
    vec_raw_fn=_mean_rev_raw_vec,
    vec_win_prob_fn=_mean_rev_win_prob_vec,
    min_rr=1.2,
    min_edge=0.0,
)
def mean_rev_long() -> None:
    """Mean reversion long."""
