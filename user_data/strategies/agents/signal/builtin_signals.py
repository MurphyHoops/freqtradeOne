# -*- coding: utf-8 -*-
"""内置信号集：MRL / PBL / TRS 以及九转新高/新低信号。"""

from __future__ import annotations

from .registry import register_signal
from .schemas import Condition


@register_signal(
    name="mean_rev_long",
    direction="long",
    squad="MRL",
    conditions=[
        Condition("RSI", "<", 25.0),
        Condition("DELTA_CLOSE_EMAFAST_PCT", "<", -0.015),
    ],
    sl_fn=lambda bag, cfg: bag["ATR_PCT"] * 1.2,
    tp_fn=lambda bag, cfg: bag["ATR_PCT"] * 2.4,
    raw_fn=lambda bag, cfg: max(0.0, (25.0 - bag["RSI"]) / 25.0),
    win_prob_fn=lambda bag, cfg, raw: min(0.9, max(0.5, 0.52 + 0.4 * raw)),
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
    sl_fn=lambda bag, cfg: bag["ATR_PCT"] * 1.0,
    tp_fn=lambda bag, cfg: bag["ATR_PCT"] * 2.0,
    raw_fn=lambda bag, cfg: 0.5 * max(0.0, bag["EMA_FAST"] / max(bag["EMA_SLOW"], 1e-9) - 1.0)
    + 0.5 * max(0.0, (bag["ADX"] - 20.0) / 20.0),
    win_prob_fn=lambda bag, cfg, raw: min(0.95, max(0.5, 0.55 + 0.4 * raw)),
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
    sl_fn=lambda bag, cfg: bag["ATR_PCT"] * 1.2,
    tp_fn=lambda bag, cfg: bag["ATR_PCT"] * 2.4,
    raw_fn=lambda bag, cfg: 0.5 * max(0.0, (bag["ADX"] - 25.0) / 25.0)
    + 0.5 * max(0.0, 1.0 - bag["EMA_FAST"] / max(bag["EMA_SLOW"], 1e-9)),
    win_prob_fn=lambda bag, cfg, raw: min(0.95, max(0.5, 0.50 + 0.4 * raw)),
    min_rr=1.2,
    min_edge=0.0,
)
def _trend_short() -> None:
    """Trend short."""


NEWBARS_THRESHOLD = 80


def _newbars_sl(bag, _cfg):
    return bag["ATR_PCT"] * 1.1


def _newbars_tp(bag, _cfg):
    return bag["ATR_PCT"] * 2.4


def _newbars_raw(key: str):
    return lambda bag, _cfg: min(1.0, bag[key] / max(NEWBARS_THRESHOLD, 1e-9))


def _newbars_win_prob(_key: str):
    return lambda bag, _cfg, raw: min(0.95, 0.55 + 0.35 * raw)


@register_signal(
    name="newbars_breakout_long_5m",
    direction="long",
    squad="NBX",
    conditions=[
        Condition("LOSS_TIER_STATE", "==", 0),
        Condition("NEWBARS_HIGH", ">", NEWBARS_THRESHOLD),
    ],
    sl_fn=_newbars_sl,
    tp_fn=_newbars_tp,
    raw_fn=_newbars_raw("NEWBARS_HIGH"),
    win_prob_fn=_newbars_win_prob("NEWBARS_HIGH"),
    min_rr=1.3,
    min_edge=0.0,
    required_factors=("LOSS_TIER_STATE", "NEWBARS_HIGH"),
)
def _newbars_breakout_long_5m() -> None:
    """九转 5m 多头，供 T0 healthy 使用。"""


@register_signal(
    name="newbars_breakout_long_30m",
    direction="long",
    squad="NBX",
    conditions=[
        Condition("LOSS_TIER_STATE", "between", 1, 2),
        Condition("NEWBARS_HIGH", ">", NEWBARS_THRESHOLD),
    ],
    sl_fn=_newbars_sl,
    tp_fn=_newbars_tp,
    raw_fn=_newbars_raw("NEWBARS_HIGH"),
    win_prob_fn=_newbars_win_prob("NEWBARS_HIGH"),
    min_rr=1.3,
    min_edge=0.0,
    required_factors=("LOSS_TIER_STATE", "NEWBARS_HIGH"),
    timeframes=("30m",),
)
def _newbars_breakout_long_30m() -> None:
    """九转 30m 多头，供 T12 recovery 使用。"""


@register_signal(
    name="newbars_breakdown_short_5m",
    direction="short",
    squad="NBX",
    conditions=[
        Condition("LOSS_TIER_STATE", "==", 0),
        Condition("NEWBARS_LOW", ">", NEWBARS_THRESHOLD),
    ],
    sl_fn=_newbars_sl,
    tp_fn=_newbars_tp,
    raw_fn=_newbars_raw("NEWBARS_LOW"),
    win_prob_fn=_newbars_win_prob("NEWBARS_LOW"),
    min_rr=1.3,
    min_edge=0.0,
    required_factors=("LOSS_TIER_STATE", "NEWBARS_LOW"),
)
def _newbars_breakdown_short_5m() -> None:
    """九转 5m 空头，供 T0 healthy 使用。"""


@register_signal(
    name="newbars_breakdown_short_30m",
    direction="short",
    squad="NBX",
    conditions=[
        Condition("LOSS_TIER_STATE", "between", 1, 2),
        Condition("NEWBARS_LOW", ">", NEWBARS_THRESHOLD),
    ],
    sl_fn=_newbars_sl,
    tp_fn=_newbars_tp,
    raw_fn=_newbars_raw("NEWBARS_LOW"),
    win_prob_fn=_newbars_win_prob("NEWBARS_LOW"),
    min_rr=1.3,
    min_edge=0.0,
    required_factors=("LOSS_TIER_STATE", "NEWBARS_LOW"),
    timeframes=("30m",),
)
def _newbars_breakdown_short_30m() -> None:
    """九转 30m 空头，供 T12 recovery 使用。"""
