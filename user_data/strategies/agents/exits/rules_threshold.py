
# ==================================
# File: agents/exits/rules_threshold.py
# ==================================
from __future__ import annotations
from typing import Optional

from .router import ImmediateContext, SLContext, TPContext, immediate_rule, sl_rule, tp_rule


@sl_rule(name="hard_sl_from_entry", priority=50)
def hard_sl_from_entry(ctx: SLContext) -> Optional[float]:
    """Use sl_pct from trade custom data/meta if present (>0). Acts as a base rule.
    Router will take the tightest among multiple rules.
    """
    if ctx.trade is None:
        return None
    sl = _trade_pct(ctx.trade, "sl_pct")
    if sl is None or sl <= 0:
        sl = _plan_pct_from_facade(ctx, "sl_pct")
    return sl if (sl and sl > 0) else None


@tp_rule(name="hard_tp_from_entry", priority=50)
def hard_tp_from_entry(ctx: TPContext) -> Optional[float]:
    if ctx.trade is None:
        return None
    tp = _trade_pct(ctx.trade, "tp_pct")
    if tp is None or tp <= 0:
        tp = _plan_pct_from_facade(ctx, "tp_pct")
    return tp if (tp and tp > 0) else None


__all__ = [
    "hard_sl_from_entry",
    "hard_tp_from_entry",
    "flip_on_strong_opposite",
]

def _trade_pct(trade, key: str) -> Optional[float]:
    try:
        if hasattr(trade, "get_custom_data"):
            value = trade.get_custom_data(key)
            if value and value > 0:
                return float(value)
    except Exception:
        pass
    return None


def _plan_pct_from_facade(ctx, attribute: str) -> Optional[float]:
    strategy = getattr(ctx, "strategy", None)
    facade = getattr(strategy, "exit_facade", None) if strategy else None
    if not facade or not ctx.trade:
        return None
    _, _, plan = facade.resolve_trade_plan(ctx.pair, ctx.trade, ctx.now)
    if not plan:
        return None
    value = getattr(plan, attribute, None)
    if value is None or value <= 0:
        return None
    return float(value)


@immediate_rule(name="flip_on_strong_opposite", priority=90)
def flip_on_strong_opposite(ctx: ImmediateContext) -> Optional[str]:
    """Example immediate exit: close when context marks strong opposite signal."""

    # Placeholder – disable by default; wire via ctx.state as needed.
    return None
