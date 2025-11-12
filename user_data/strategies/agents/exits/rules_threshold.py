
# ==================================
# File: agents/exits/rules_threshold.py
# ==================================
from __future__ import annotations
from typing import Optional

from .router import SLContext, TPContext, sl_rule, tp_rule

def _trade_has_exit_profile(trade) -> bool:
    if trade is None:
        return False
    try:
        if hasattr(trade, "get_custom_data"):
            profile = trade.get_custom_data("exit_profile")
            if profile:
                return True
    except Exception:
        pass
    try:
        if getattr(trade, "user_data", None):
            profile = trade.user_data.get("exit_profile")
            if profile:
                return True
    except Exception:
        pass
    return False

# Helper: compute ATR(abs) as‑of now from dp/analyzed df

def _atr_abs_asof(dp, pair: str, timeframe: str, now) -> Optional[float]:
    try:
        analyzed = dp.get_analyzed_dataframe(pair, timeframe)
        df = analyzed[0] if isinstance(analyzed, (list, tuple)) else analyzed
        if df is None or len(df) == 0:
            return None
        try:
            upto = df.loc[:now]
        except Exception:
            ct = now.replace(tzinfo=None) if getattr(now, "tzinfo", None) else now
            upto = df.loc[:ct]
        if len(upto) == 0:
            return None
        row = upto.iloc[-1]
        if "atr" in df.columns and row["atr"] == row["atr"]:  # not NaN
            return float(row["atr"])
        # Lightweight TR mean fallback (N=14)
        tail = upto.tail(15)
        if len(tail) < 2:
            return None
        highs = tail["high"].values; lows = tail["low"].values; closes = tail["close"].values
        trs = []
        for i in range(1, len(tail)):
            h = float(highs[i]); l = float(lows[i]); pc = float(closes[i-1])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs) / len(trs) if trs else None
    except Exception:
        return None


# ==== SL rules (percent of entry) ====
@sl_rule(name="atr_k_sl", priority=60)
def atr_k_sl(ctx: SLContext) -> Optional[float]:
    """Stoploss distance = k * ATR / entry_price (percent of entry).
    cfg keys: atr_len (optional), atr_sl_k (default 1.0), timeframe (use strategy.timeframe)
    """
    trade = ctx.trade
    if not trade or not getattr(trade, "open_rate", 0.0):
        return None
    if _trade_has_exit_profile(trade):
        return None
    atr_abs = _atr_abs_asof(ctx.dp, ctx.pair, getattr(ctx.cfg, "timeframe", ""), ctx.now)
    if not atr_abs:
        return None
    k = float(getattr(ctx.cfg, "atr_sl_k", 1.0))
    return max(0.0, (atr_abs / float(trade.open_rate)) * k) or None


@sl_rule(name="hard_sl_from_entry", priority=50)
def hard_sl_from_entry(ctx: SLContext) -> Optional[float]:
    """Use sl_pct from trade custom data/meta if present (>0). Acts as a base rule.
    Router will take the tightest among multiple rules.
    """
    try:
        sl = None
        if hasattr(ctx.trade, "get_custom_data"):
            sl = ctx.trade.get_custom_data("sl_pct")
        if (sl is None or sl <= 0) and getattr(ctx.trade, "user_data", None):
            sl = ctx.trade.user_data.get("sl_pct")
        return sl if (sl and sl > 0) else None
    except Exception:
        return None


# ==== TP rules (percent of entry) ====
@tp_rule(name="atr_k_tp", priority=80)
def atr_k_tp(ctx: TPContext) -> Optional[float]:
    trade = ctx.trade
    if not trade or not getattr(trade, "open_rate", 0.0):
        return None
    if _trade_has_exit_profile(trade):
        return None
    atr_abs = _atr_abs_asof(ctx.dp, ctx.pair, getattr(ctx.cfg, "timeframe", ""), ctx.now)
    if not atr_abs:
        return None
    k = float(getattr(ctx.cfg, "atr_tp_k", 2.0))
    return max(0.0, (atr_abs / float(trade.open_rate)) * k) or None


@tp_rule(name="hard_tp_from_entry", priority=50)
def hard_tp_from_entry(ctx: TPContext) -> Optional[float]:
    try:
        tp = None
        if hasattr(ctx.trade, "get_custom_data"):
            tp = ctx.trade.get_custom_data("tp_pct")
        if (tp is None or tp <= 0) and getattr(ctx.trade, "user_data", None):
            tp = ctx.trade.user_data.get("tp_pct")
        return tp if (tp and tp > 0) else None
    except Exception:
        return None


__all__ = [
    "atr_k_sl", "hard_sl_from_entry", "atr_k_tp", "hard_tp_from_entry"
]

