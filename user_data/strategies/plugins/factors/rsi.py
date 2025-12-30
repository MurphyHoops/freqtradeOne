# -*- coding: utf-8 -*-
"""RSI factor registration."""

from user_data.strategies.agents.signals.registry import register_factor


@register_factor(name="RSI", column="rsi", indicators=("RSI",))
def _register() -> None:
    return None
