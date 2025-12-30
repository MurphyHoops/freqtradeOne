# -*- coding: utf-8 -*-
"""New bars high factor registration."""

from user_data.strategies.agents.signals.registry import register_factor


@register_factor(name="NEWBARS_HIGH", column="newbars_high", indicators=("NEWHBARS",))
def _register() -> None:
    return None
