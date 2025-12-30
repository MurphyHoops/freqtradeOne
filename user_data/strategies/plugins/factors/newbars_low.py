# -*- coding: utf-8 -*-
"""New bars low factor registration."""

from user_data.strategies.agents.signals.registry import register_factor


@register_factor(name="NEWBARS_LOW", column="newbars_low", indicators=("NEWHBARS",))
def _register() -> None:
    return None
