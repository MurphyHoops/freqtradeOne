# -*- coding: utf-8 -*-
"""内置信号集：MRL / PBL / TRS 以及九转新高/新低信号。"""

from __future__ import annotations

import numpy as np

from .registry import register_signal
from .schemas import Condition


def _mean_rev_raw_vec(get, _cfg, timeframe):
    rsi = get("RSI", timeframe)
    return ((25.0 - rsi) / 25.0).clip(lower=0.0)


def _mean_rev_win_prob_vec(_get, _cfg, raw, _timeframe):
    return (0.52 + 0.4 * raw).clip(lower=0.5, upper=0.9)


def _pullback_raw_vec(get, _cfg, timeframe):
    ema_fast = get("EMA_FAST", timeframe)
    ema_slow = get("EMA_SLOW", timeframe)
    adx = get("ADX", timeframe)
    denom = np.maximum(ema_slow, 1e-9)
    trend = (ema_fast / denom - 1.0).clip(lower=0.0)
    strength = ((adx - 20.0) / 20.0).clip(lower=0.0)
    return 0.5 * trend + 0.5 * strength


def _pullback_win_prob_vec(_get, _cfg, raw, _timeframe):
    return (0.55 + 0.4 * raw).clip(lower=0.5, upper=0.95)


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


def _newbars_raw_vec(key: str):
    def _raw(get, _cfg, timeframe):
        series = get(key, timeframe)
        return (series / max(NEWBARS_THRESHOLD, 1e-9)).clip(lower=0.0, upper=1.0)

    return _raw


def _newbars_win_prob_vec(_key: str):
    def _win(_get, _cfg, raw, _timeframe):
        return (0.55 + 0.35 * raw).clip(upper=0.95)

    return _win


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
def _mean_rev_long() -> None:
    """Mean reversion long."""


@register_signal(
    name="pullback_long",
    direction="long",
    squad="PBL",
    conditions=[
        Condition("EMA_TREND", "==", 1.0),
        Condition("ADX", ">", 20.0),
        Condition("DELTA_CLOSE_EMAFAST_PCT", "<", -0.01),
    ],
    raw_fn=lambda bag, cfg: 0.5 * max(0.0, bag["EMA_FAST"] / max(bag["EMA_SLOW"], 1e-9) - 1.0)
    + 0.5 * max(0.0, (bag["ADX"] - 20.0) / 20.0),
    win_prob_fn=lambda bag, cfg, raw: min(0.95, max(0.5, 0.55 + 0.4 * raw)),
    vec_raw_fn=_pullback_raw_vec,
    vec_win_prob_fn=_pullback_win_prob_vec,
    min_rr=1.2,
    min_edge=0.0,
)
def _pullback_long() -> None:
    """Pullback long."""


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
def _trend_short() -> None:
    """Trend short."""


NEWBARS_THRESHOLD = 80


def _newbars_raw(key: str):
    return lambda bag, _cfg: min(1.0, bag[key] / max(NEWBARS_THRESHOLD, 1e-9))


def _newbars_win_prob(_key: str):
    return lambda bag, _cfg, raw: min(0.95, 0.55 + 0.35 * raw)


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
def _newbars_breakout_long_5m() -> None:
    """九转 5m 多头，供 T0 healthy 使用。"""


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
def _newbars_breakout_long_30m() -> None:
    """九转 30m 多头，供 T12 recovery 使用。"""


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
def _newbars_breakdown_short_5m() -> None:
    """九转 5m 空头，供 T0 healthy 使用。"""


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
def _newbars_breakdown_short_30m() -> None:
    """九转 30m 空头，供 T12 recovery 使用。"""
