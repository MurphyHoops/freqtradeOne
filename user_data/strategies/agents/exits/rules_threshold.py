# ==================================
# File: agents/exits/rules_threshold.py
# ==================================
from __future__ import annotations

from typing import Any, Optional, Tuple

from .router import ImmediateContext, SLContext, TPContext, immediate_rule, sl_rule, tp_rule


__all__ = [
    "atr_trail_from_profile",
    "breakeven_lock_from_profile",
    "hard_sl_from_entry",
    "hard_tp_from_entry",
    "flip_on_strong_opposite",
]


@sl_rule(name="hard_sl_from_entry", priority=50)
def hard_sl_from_entry(ctx: SLContext) -> Optional[float]:
    """Use sl_pct from trade custom data/meta if present (>0). Acts as a base rule."""

    if ctx.trade is None:
        return None
    sl = _trade_pct(ctx.trade, "sl_pct")
    if sl is None or sl <= 0:
        sl = _plan_pct_from_facade(ctx, "sl_pct")
    return sl if (sl and sl > 0) else None


@tp_rule(name="hard_tp_from_entry", priority=50)
def hard_tp_from_entry(ctx: TPContext) -> Optional[float]:
    """Return tp_pct from trade metadata or the resolved exit plan."""

    if ctx.trade is None:
        return None
    tp = _trade_pct(ctx.trade, "tp_pct")
    if tp is None or tp <= 0:
        tp = _plan_pct_from_facade(ctx, "tp_pct")
    return tp if (tp and tp > 0) else None


@sl_rule(name="atr_trail_from_profile", priority=60)
def atr_trail_from_profile(ctx: SLContext) -> Optional[float]:
    """Tighten SL based on profile trailing settings (percent/chandelier)."""

    profile, plan, facade = _profile_plan_and_facade(ctx)
    if not profile or not plan:
        return None

    mode = (getattr(profile, "trail_mode", None) or "").strip().lower()
    if not mode:
        return None

    if mode == "percent":
        return _percent_trail(ctx, profile, plan)
    if mode == "chandelier":
        return _chandelier_trail(ctx, profile, plan, facade)
    return None


@sl_rule(name="breakeven_lock_from_profile", priority=70)
def breakeven_lock_from_profile(ctx: SLContext) -> Optional[float]:
    """Move SL toward break-even once a TP fraction has been achieved."""

    profile, plan, _ = _profile_plan_and_facade(ctx)
    if not profile or not plan:
        return None

    frac = getattr(profile, "breakeven_lock_frac_of_tp", None)
    if frac is None:
        frac = getattr(ctx.cfg, "breakeven_lock_frac_of_tp", None)
    if frac is None or frac <= 0:
        return None

    tp_pct = getattr(plan, "tp_pct", None)
    if tp_pct is None or tp_pct <= 0:
        return None

    profit = float(getattr(ctx, "profit", 0.0) or 0.0)
    if profit < float(tp_pct) * float(frac):
        return None

    base_sl = _plan_pct(plan, "sl_pct")
    eps = _breakeven_eps(ctx, plan)
    return min(base_sl, eps) if base_sl is not None else eps


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
    return _plan_pct(plan, attribute)


@immediate_rule(name="flip_on_strong_opposite", priority=90)
def flip_on_strong_opposite(ctx: ImmediateContext) -> Optional[str]:
    """Example immediate exit: close when context marks strong opposite signal."""

    # Placeholder – disable by default; wire via ctx.state as needed.
    return None


def _profile_plan_and_facade(ctx: SLContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    strategy = getattr(ctx, "strategy", None)
    facade = getattr(strategy, "exit_facade", None) if strategy else None
    if not facade or not ctx.trade:
        return None, None, facade
    try:
        _, profile, plan = facade.resolve_trade_plan(ctx.pair, ctx.trade, ctx.now)
    except Exception:
        profile = None
        plan = None
    return profile, plan, facade


def _plan_pct(plan, attribute: str) -> Optional[float]:
    if not plan:
        return None
    value = getattr(plan, attribute, None)
    if value is None or value <= 0:
        return None
    return float(value)


def _breakeven_eps(ctx, plan) -> float:
    atr_pct = getattr(plan, "atr_pct", None) or 0.0
    eps_factor = getattr(ctx.cfg, "breakeven_lock_eps_atr_pct", 0.0) or 0.0
    eps = atr_pct * float(eps_factor)
    return max(eps, 5e-4)


def _percent_trail(ctx: SLContext, profile, plan) -> Optional[float]:
    profit = max(float(getattr(ctx, "profit", 0.0) or 0.0), 0.0)
    trail_pct = getattr(profile, "trail_pct", None)
    if profit <= 0 or trail_pct is None or trail_pct <= 0:
        return None
    base_sl = _plan_pct(plan, "sl_pct")
    if base_sl is None:
        return None
    reduction = profit * float(trail_pct)
    tightened = base_sl - reduction
    eps = _breakeven_eps(ctx, plan)
    candidate = max(tightened, eps)
    return min(base_sl, candidate)


def _chandelier_trail(ctx: SLContext, profile, plan, facade) -> Optional[float]:
    trade = ctx.trade
    if not trade:
        return None
    timeframe = getattr(profile, "atr_timeframe", None) or getattr(plan, "timeframe", None) or getattr(
        ctx.cfg, "timeframe", None
    )
    atr_pct = None
    if facade and timeframe:
        atr_pct = facade.atr_pct(ctx.pair, timeframe, ctx.now)
    if (atr_pct is None or atr_pct <= 0) and getattr(plan, "atr_pct", None):
        atr_pct = plan.atr_pct
    if atr_pct is None or atr_pct <= 0:
        return None

    multiplier = getattr(profile, "trail_atr_mul", None)
    if multiplier is None or multiplier <= 0:
        return None

    extreme = _extreme_price_since_entry(ctx, timeframe)
    open_rate = getattr(trade, "open_rate", None)
    if extreme is None or open_rate is None or open_rate <= 0:
        return None

    buffer_pct = atr_pct * float(multiplier)
    is_short = bool(getattr(trade, "is_short", False))
    if is_short:
        profit_extreme = (open_rate / extreme) - 1.0
    else:
        profit_extreme = (extreme / open_rate) - 1.0
    locked_profit = profit_extreme - buffer_pct
    if locked_profit <= 0:
        return None

    base_sl = _plan_pct(plan, "sl_pct")
    eps = _breakeven_eps(ctx, plan)
    candidate = max((base_sl or 0.0) - locked_profit, eps)
    if base_sl is None:
        return max(candidate, eps)
    return min(base_sl, max(candidate, eps))


def _extreme_price_since_entry(ctx: SLContext, timeframe: Optional[str]) -> Optional[float]:
    dp = getattr(ctx, "dp", None)
    trade = ctx.trade
    if not dp or not trade or not timeframe:
        return None
    try:
        analyzed = dp.get_analyzed_dataframe(ctx.pair, timeframe)
    except Exception:
        return None
    df = analyzed[0] if isinstance(analyzed, (list, tuple)) else analyzed
    if df is None or len(df) == 0:
        return None
    open_time = getattr(trade, "open_date_utc", None)
    if open_time is not None:
        try:
            df = df.loc[open_time:]
        except Exception:
            try:
                df = df[df.index >= open_time]
            except Exception:
                pass
    columns = getattr(df, "columns", [])
    high_col = next((c for c in columns if c.lower() == "high"), None)
    low_col = next((c for c in columns if c.lower() == "low"), None)
    if not high_col or not low_col:
        return None
    try:
        if bool(getattr(trade, "is_short", False)):
            series = df[low_col]
            extreme = float(series.min())
        else:
            series = df[high_col]
            extreme = float(series.max())
    except Exception:
        return None
    return extreme if extreme and extreme > 0 else None
