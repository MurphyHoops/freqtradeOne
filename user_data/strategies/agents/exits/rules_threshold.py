
# ==================================
# File: agents/exits/rules_threshold.py
# ==================================
from __future__ import annotations
from typing import Optional

from .profile_planner import atr_pct_from_dp
from .router import SLContext, TPContext, sl_rule, tp_rule


def percent_from_atr_k(ctx, timeframe: Optional[str], atr_k: float) -> Optional[float]:
    """Return ATR-based percent distance with consistent data sources."""

    if atr_k is None or atr_k <= 0:
        return None
    target_tf = timeframe or getattr(ctx.cfg, "timeframe", None)
    atr_pct: Optional[float] = None
    strategy = getattr(ctx, "strategy", None)
    analytics = getattr(getattr(strategy, "analytics", None), "log_debug", None)
    atr_source = None
    facade = getattr(strategy, "exit_facade", None)
    if facade and target_tf:
        atr_pct = facade.atr_pct(ctx.pair, target_tf, ctx.now)
        if atr_pct and atr_pct > 0:
            atr_source = "facade"
    if atr_pct is None or atr_pct <= 0:
        fallback_tf = target_tf or getattr(ctx.cfg, "timeframe", "") or ""
        if fallback_tf:
            atr_pct = atr_pct_from_dp(ctx.dp, ctx.pair, fallback_tf, ctx.now)
            if atr_pct and atr_pct > 0:
                atr_source = "dataprovider"
    if atr_pct is None or atr_pct <= 0:
        if analytics:
            analytics(
                "threshold_atr_missing",
                f"ATR pct unavailable for {ctx.pair}",
                {"pair": ctx.pair, "timeframe": target_tf, "atr_k": atr_k},
            )
        return None
    value = max(0.0, float(atr_pct) * float(atr_k))
    if analytics and atr_source:
        analytics(
            "threshold_atr_pct",
            "ATR pct resolved for tightening rule",
            {"pair": ctx.pair, "timeframe": target_tf, "atr_pct": atr_pct, "atr_k": atr_k, "source": atr_source},
        )
    return value


# ==== SL rules (percent of entry) ====
@sl_rule(name="atr_k_sl", priority=60)
def atr_k_sl(ctx: SLContext) -> Optional[float]:
    """Stoploss distance = k * ATR / entry_price (percent of entry).
    cfg keys: atr_len (optional), atr_sl_k (default 1.0), timeframe (use strategy.timeframe)
    """
    if not getattr(ctx.cfg, "use_legacy_sl_tp", False):
        return None
    trade = ctx.trade
    if not trade or not getattr(trade, "open_rate", 0.0):
        return None
    tf = getattr(ctx.cfg, "atr_sl_timeframe", None)
    k = float(getattr(ctx.cfg, "atr_sl_k", 1.0))
    pct = percent_from_atr_k(ctx, tf, k)
    return pct if pct and pct > 0 else None


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


# ==== TP rules (percent of entry) ====
@tp_rule(name="atr_k_tp", priority=80)
def atr_k_tp(ctx: TPContext) -> Optional[float]:
    if not getattr(ctx.cfg, "use_legacy_sl_tp", False):
        return None
    trade = ctx.trade
    if not trade or not getattr(trade, "open_rate", 0.0):
        return None
    tf = getattr(ctx.cfg, "atr_tp_timeframe", None)
    k = float(getattr(ctx.cfg, "atr_tp_k", 2.0))
    pct = percent_from_atr_k(ctx, tf, k)
    return pct if pct and pct > 0 else None


@tp_rule(name="hard_tp_from_entry", priority=50)
def hard_tp_from_entry(ctx: TPContext) -> Optional[float]:
    if ctx.trade is None:
        return None
    tp = _trade_pct(ctx.trade, "tp_pct")
    if tp is None or tp <= 0:
        tp = _plan_pct_from_facade(ctx, "tp_pct")
    return tp if (tp and tp > 0) else None


__all__ = [
    "atr_k_sl",
    "hard_sl_from_entry",
    "atr_k_tp",
    "hard_tp_from_entry",
    "percent_from_atr_k",
]

def _trade_pct(trade, key: str) -> Optional[float]:
    try:
        if hasattr(trade, "get_custom_data"):
            value = trade.get_custom_data(key)
            if value and value > 0:
                return float(value)
    except Exception:
        pass
    try:
        user_data = getattr(trade, "user_data", None)
        if user_data:
            value = user_data.get(key)
            if value and value > 0:
                return float(value)
    except Exception:
        return None
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
