
# ==================================
# File: agents/exits/rules_immediate.py
# ==================================
from __future__ import annotations
from typing import Optional

from .router import ImmediateContext, immediate_rule


@immediate_rule(name="flip_on_strong_opposite", priority=90)
def flip_on_strong_opposite(ctx: ImmediateContext) -> Optional[str]:
    """Example immediate exit: close when context marks strong opposite signal.
    This is a placeholder – wire your own condition via ctx.state or cached signal flags.
    Return a reason string to force close via custom_exit.
    """
    # Example placeholder – disabled by default
    return None


__all__ = ["flip_on_strong_opposite"]

