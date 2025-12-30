# -*- coding: utf-8 -*-
"""EMA fast factor registration."""

from user_data.strategies.agents.signals.registry import register_factor


@register_factor(name="EMA_FAST", column="ema_fast", indicators=("EMA_FAST",))
def _register() -> None:
    return None
