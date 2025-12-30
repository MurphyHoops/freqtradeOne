# -*- coding: utf-8 -*-
"""EMA slow factor registration."""

from user_data.strategies.agents.signals.registry import register_factor


@register_factor(name="EMA_SLOW", column="ema_slow", indicators=("EMA_SLOW",))
def _register() -> None:
    return None
