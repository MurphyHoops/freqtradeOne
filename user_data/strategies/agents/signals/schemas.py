# -*- coding: utf-8 -*-
"""Shared data contracts used by the signal module.

The goal of these dataclasses is to keep the signal layer declarative and make
data passing between SignalBuilder → Strategy → Portfolio explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class Candidate:
    """A fully scored trade candidate produced by the signal builder."""

    direction: Direction
    kind: str
    raw_score: float
    rr_ratio: float
    win_prob: float
    expected_edge: float
    squad: str
    sl_pct: float
    tp_pct: float
    exit_profile: Optional[str] = None
    recipe: Optional[str] = None
    plan_timeframe: Optional[str] = None
    plan_atr_pct: Optional[float] = None


@dataclass(frozen=True)
class Condition:
    """Boolean check applied to a factor value."""

    factor: str
    op: Optional[str] = None
    value: Optional[float] = None
    value_hi: Optional[float] = None
    fn: Optional[Callable[[float], bool]] = None


@dataclass(frozen=True)
class SignalSpec:
    """Declarative description of a signal plug-in."""

    name: str
    direction: Direction
    squad: str
    conditions: List[Condition]
    raw_fn: Callable[[Dict[str, float], Any], float]
    win_prob_fn: Callable[[Dict[str, float], Any, float], float]
    min_rr: float = 0.0
    min_edge: float = 0.0
    required_factors: tuple[str, ...] = ()
    timeframe: Optional[str] = None


__all__ = ["Candidate", "Condition", "SignalSpec", "Direction"]
