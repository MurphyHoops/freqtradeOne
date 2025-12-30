# -*- coding: utf-8 -*-
"""Newbars breakout/breakdown signals."""

from __future__ import annotations

import numpy as np

from user_data.strategies.agents.signals.registry import register_signal
from user_data.strategies.agents.signals.schemas import Condition


NEWBARS_THRESHOLD = 80


def _newbars_raw_vec(key: str):
    def _raw(get, _cfg, timeframe):
        series = get(key, timeframe)
        return (series / max(NEWBARS_THRESHOLD, 1e-9)).clip(lower=0.0, upper=1.0)

    return _raw


def _newbars_win_prob_vec(_key: str):
    def _win(_get, _cfg, raw, _timeframe):
        return (0.55 + 0.35 * raw).clip(upper=0.95)

    return _win


def _newbars_raw(key: str):
    return lambda bag, _cfg: min(1.0, bag[key] / max(NEWBARS_THRESHOLD, 1e-9))


def _newbars_win_prob(_key: str):
    return lambda _bag, _cfg, raw: min(0.95, 0.55 + 0.35 * raw)


@register_signal(
    name="newbars_breakout_long_5m",
    direction="long",
    squad="NBX",
    conditions=[
        Condition("NEWBARS_HIGH", ">", NEWBARS_THRESHOLD),
    ],
    raw_fn=_newbars_raw("NEWBARS_HIGH"),
    win_prob_fn=_newbars_win_prob("NEWBARS_HIGH"),
    vec_raw_fn=_newbars_raw_vec("NEWBARS_HIGH"),
    vec_win_prob_fn=_newbars_win_prob_vec("NEWBARS_HIGH"),
    min_rr=0.1,
    min_edge=0.0,
    required_factors=("NEWBARS_HIGH",),
)
def newbars_breakout_long_5m() -> None:
    """Newbars 5m breakout long."""


@register_signal(
    name="newbars_breakout_long_30m",
    direction="long",
    squad="NBX",
    conditions=[
        Condition("NEWBARS_HIGH", ">", NEWBARS_THRESHOLD),
    ],
    raw_fn=_newbars_raw("NEWBARS_HIGH"),
    win_prob_fn=_newbars_win_prob("NEWBARS_HIGH"),
    vec_raw_fn=_newbars_raw_vec("NEWBARS_HIGH"),
    vec_win_prob_fn=_newbars_win_prob_vec("NEWBARS_HIGH"),
    min_rr=0.1,
    min_edge=0.0,
    required_factors=("NEWBARS_HIGH",),
    timeframes=("30m",),
)
def newbars_breakout_long_30m() -> None:
    """Newbars 30m breakout long."""


@register_signal(
    name="newbars_breakdown_short_5m",
    direction="short",
    squad="NBX",
    conditions=[
        Condition("NEWBARS_LOW", ">", NEWBARS_THRESHOLD),
    ],
    raw_fn=_newbars_raw("NEWBARS_LOW"),
    win_prob_fn=_newbars_win_prob("NEWBARS_LOW"),
    vec_raw_fn=_newbars_raw_vec("NEWBARS_LOW"),
    vec_win_prob_fn=_newbars_win_prob_vec("NEWBARS_LOW"),
    min_rr=0.1,
    min_edge=0.0,
    required_factors=("NEWBARS_LOW",),
)
def newbars_breakdown_short_5m() -> None:
    """Newbars 5m breakdown short."""


@register_signal(
    name="newbars_breakdown_short_30m",
    direction="short",
    squad="NBX",
    conditions=[
        Condition("NEWBARS_LOW", ">", NEWBARS_THRESHOLD),
    ],
    raw_fn=_newbars_raw("NEWBARS_LOW"),
    win_prob_fn=_newbars_win_prob("NEWBARS_LOW"),
    vec_raw_fn=_newbars_raw_vec("NEWBARS_LOW"),
    vec_win_prob_fn=_newbars_win_prob_vec("NEWBARS_LOW"),
    min_rr=0.1,
    min_edge=0.0,
    required_factors=("NEWBARS_LOW",),
    timeframes=("30m",),
)
def newbars_breakdown_short_30m() -> None:
    """Newbars 30m breakdown short."""
