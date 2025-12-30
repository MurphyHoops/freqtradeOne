# -*- coding: utf-8 -*-
"""ATR percent factor registration."""

from user_data.strategies.agents.signals.registry import register_factor


@register_factor(name="ATR_PCT", column="atr_pct", indicators=("ATR",))
def _register() -> None:
    return None
