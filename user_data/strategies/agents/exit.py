# agents/exit.py
from __future__ import annotations
from typing import Optional

try:
    from .exits.router import EXIT_ROUTER, ImmediateContext
    from .exits.exit_tags import ExitTags
except Exception:  # pragma: no cover
    EXIT_ROUTER = None  # type: ignore
    ImmediateContext = None  # type: ignore

    class _ExitTagsFallback:
        CLOSE_FILLED = "close_filled"
        ORDER_CANCELLED = "order_cancelled"
        ORDER_REJECTED = "order_rejected"
        TP_HIT = "tp_hit"
        ICU_TIMEOUT = "icu_timeout"
        RISK_OFF = "risk_off"
        BREAKEVEN = "breakeven_lock"
        HARD_STOP = "hard_stop"
        HARD_TP = "hard_takeprofit"
        FLIP_PREFIX = "flip_"

        @staticmethod
        def flip(direction: str | None) -> str:
            token = (direction or "").strip().lower()
            if token not in {"long", "short"}:
                token = "unknown"
            return f"{_ExitTagsFallback.FLIP_PREFIX}{token}"

    ExitTags = _ExitTagsFallback()  # type: ignore

class ExitPolicyV29:
    def __init__(self, state, eq_provider, cfg, dp=None) -> None:
        self.state = state
        self.eq_provider = eq_provider
        self.cfg = cfg
        self.dp = dp  # <--- dp���� Router Context
        self.strategy = None

    def set_strategy(self, strategy) -> None:
        self.strategy = strategy

    def decide(self, pair: str, trade_id: str, current_profit_pct: Optional[float]) -> Optional[str]:
        """ """
        if EXIT_ROUTER is not None and hasattr(self.state, "get_trade_by_id"):
            try:
                trade = self.state.get_trade_by_id(trade_id)
                ctx = ImmediateContext(
                    pair=pair,
                    trade=trade,
                    now=getattr(self.state, "now", None),
                    profit=(current_profit_pct or 0.0),
                    dp=self.dp,           # <--- ???????????? self.dp
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

        if meta:
            if (
                meta.tp_pct
                and current_profit_pct is not None
                and current_profit_pct >= float(meta.tp_pct)
            ):
                return ExitTags.TP_HIT
            if meta.icu_bars_left is not None and meta.icu_bars_left <= 0:
                return ExitTags.ICU_TIMEOUT
            if (
                pair_state
                and pair_state.last_dir
                and pair_state.last_dir != meta.direction
                and getattr(pair_state, "last_score", 0.0) > 0.0
            ):
                return ExitTags.flip(pair_state.last_dir)

        if (
            current_profit_pct is not None
            and current_profit_pct < 0
            and getattr(self.state, "debt_pool", 0.0) > 0.0
        ):
            return ExitTags.RISK_OFF

        #  ????
        return None

    def early_lock_distance(
        self,
        trade,
        current_rate: Optional[float],
        current_profit_pct: Optional[float],
        atr_pct_hint: float,
    ) -> Optional[float]:
        """Provide a simple breakeven/lock distance when profit exceeds threshold."""

        base = atr_pct_hint if atr_pct_hint and atr_pct_hint > 0 else None
        if base:
            return -max(base * 0.5, 0.001)
        if current_profit_pct is not None and current_profit_pct > 0:
            return -abs(current_profit_pct) * 0.25
        return None

