"""Exit policy helpers and canonical exit tags."""

from __future__ import annotations

from typing import Optional

from .router import EXIT_ROUTER, ImmediateContext


class ExitTags:
    """Static string constants for exit reasons / tags."""

    CLOSE_FILLED = "close_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    TP_HIT = "tp_hit"
    ICU_TIMEOUT = "icu_timeout"
    RISK_OFF = "risk_off"
    BREAKEVEN = "breakeven_lock"
    ATR_TRAIL = "atr_trail_follow"
    HARD_STOP = "hard_stop"
    HARD_TP = "hard_takeprofit"
    FLIP_PREFIX = "flip_"
    VECTOR_TAGS = {
        TP_HIT,
        ICU_TIMEOUT,
        RISK_OFF,
        BREAKEVEN,
        ATR_TRAIL,
        HARD_STOP,
        HARD_TP,
    }

    @staticmethod
    def flip(direction: str | None) -> str:
        """Return flip tag for the provided direction (long/short)."""

        token = (direction or "").strip().lower()
        if token not in {"long", "short"}:
            token = "unknown"
        return f"{ExitTags.FLIP_PREFIX}{token}"

    @classmethod
    def is_vector_tag(cls, tag: str | None) -> bool:
        """Return True if the tag belongs to vector exit instrumentation."""

        if not tag:
            return False
        return tag in cls.VECTOR_TAGS or tag.startswith(cls.FLIP_PREFIX)

class ExitPolicyV29:
    def __init__(self, state, eq_provider, cfg, dp=None) -> None:
        self.state = state
        self.eq_provider = eq_provider
        self.cfg = cfg
        self.dp = dp  # <--- dp���� Router Context
        self.strategy = None

    def set_strategy(self, strategy) -> None:
        self.strategy = strategy

    def decide(
        self,
        pair: str,
        trade_id: str,
        current_profit_pct: Optional[float],
        trade=None,
    ) -> Optional[str]:
        """Return an exit tag when router or fallback heuristics request an immediate close."""

        trade_obj = trade
        if trade_obj is None and hasattr(self.state, "get_trade_by_id"):
            try:
                trade_obj = self.state.get_trade_by_id(trade_id)
            except Exception:
                trade_obj = None

        if EXIT_ROUTER is not None:
            try:
                ctx = ImmediateContext(
                    pair=pair,
                    trade=trade_obj,
                    now=getattr(self.state, "now", None),
                    profit=(current_profit_pct or 0.0),
                    dp=self.dp,
                    cfg=self.cfg,
                    state=self.state,
                    strategy=self.strategy,
                )
                reason = EXIT_ROUTER.close_now_reason(ctx)
                if reason:
                    return reason
            except Exception:
                pass

        pair_state = None
        try:
            if hasattr(self.state, "get_pair_state"):
                pair_state = self.state.get_pair_state(pair)
        except Exception:
            pair_state = None
        meta = None
        if pair_state and getattr(pair_state, "active_trades", None):
            meta = pair_state.active_trades.get(str(trade_id))

        # if meta:
        #     if (
        #         meta.tp_pct
        #         and current_profit_pct is not None
        #         and current_profit_pct >= float(meta.tp_pct)
        #     ):
        #         return ExitTags.TP_HIT
        #     if meta.icu_bars_left is not None and meta.icu_bars_left <= 0:
        #         return ExitTags.ICU_TIMEOUT
        #     if (
        #         pair_state
        #         and pair_state.last_dir
        #         and pair_state.last_dir != meta.direction
        #         and getattr(pair_state, "last_score", 0.0) > 0.0
        #     ):
        #         return ExitTags.flip(pair_state.last_dir)

        # if (
        #     current_profit_pct is not None
        #     and current_profit_pct < 0
        #     and getattr(self.state, "debt_pool", 0.0) > 0.0
        # ):
        #     return ExitTags.RISK_OFF

        #  ????
        return None

__all__ = ["ExitPolicyV29", "ExitTags"]

