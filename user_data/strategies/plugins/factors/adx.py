# -*- coding: utf-8 -*-
"""ADX factor registration."""

from user_data.strategies.agents.signals.registry import register_factor


@register_factor(name="ADX", column="adx", indicators=("ADX",))
def _register() -> None:
    return None
