# -*- coding: utf-8 -*-
"""Canonical exit tag constants to keep logging / analytics consistent."""

from __future__ import annotations


class ExitTags:
    """Static string constants for exit reasons / tags."""

    CLOSE_FILLED = "close_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    TP_HIT = "tp_hit"
    ICU_TIMEOUT = "icu_timeout"
    RISK_OFF = "risk_off"
    BREAKEVEN = "breakeven_lock"
    ATR_TRAIL = "atr_trail_follow"
    HARD_STOP = "hard_stop"
    HARD_TP = "hard_takeprofit"
    FLIP_PREFIX = "flip_"
    VECTOR_TAGS = {
        TP_HIT,
        ICU_TIMEOUT,
        RISK_OFF,
        BREAKEVEN,
        ATR_TRAIL,
        HARD_STOP,
        HARD_TP,
    }

    @staticmethod
    def flip(direction: str | None) -> str:
        """Return flip tag for the provided direction (long/short)."""

        token = (direction or "").strip().lower()
        if token not in {"long", "short"}:
            token = "unknown"
        return f"{ExitTags.FLIP_PREFIX}{token}"

    @classmethod
    def is_vector_tag(cls, tag: str | None) -> bool:
        """Return True if the tag belongs to vector exit instrumentation."""

        if not tag:
            return False
        return tag in cls.VECTOR_TAGS or tag.startswith(cls.FLIP_PREFIX)


__all__ = ["ExitTags"]
