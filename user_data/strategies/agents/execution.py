from __future__ import annotations

from typing import Any, Dict, Optional

from .reservation import ReservationAgent
from .tier import TierManager, TierPolicy


class ExecutionAgent:
    """ExecutionAgent 的职责说明。"""
    def __init__(
        self,
        state,
        reservation: ReservationAgent,
        eq_provider,
        cfg,
    ) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.state = state
        self.reservation = reservation
        self.eq = eq_provider
        self.cfg = cfg

    def on_open_filled(
        self,
        pair: str,
        trade,
        order,
        pending_meta: Dict[str, Any],
        tier_mgr: TierManager,
    ) -> bool:
        """处理 on_open_filled 的主要逻辑。"""
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        if trade_id in self.state.get_pair_state(pair).active_trades:
            return False

        sl = float(pending_meta.get("sl_pct", 0.0))
        tp = float(pending_meta.get("tp_pct", 0.0))
        direction = str(pending_meta.get("dir", ""))
        rid = pending_meta.get("reservation_id")
        bucket = pending_meta.get("bucket", "slow")
        risk = float(pending_meta.get("risk_final", 0.0))
        entry_price = float(pending_meta.get("entry_price", 0.0))

        pst = self.state.get_pair_state(pair)
        tier_pol: TierPolicy = tier_mgr.get(pst.closs)
        self.state.record_trade_open(
            pair=pair,
            trade_id=trade_id,
            real_risk=risk,
            sl_pct=sl,
            tp_pct=tp,
            direction=direction,
            bucket=bucket,
            entry_price=entry_price,
            tier_pol=tier_pol,
        )
        if rid:
            self.reservation.release(rid)

        try:
            trade.set_custom_data("sl_pct", sl)
            trade.set_custom_data("tp_pct", tp)
        except Exception:
            pass
        try:
            if hasattr(trade, "user_data") and isinstance(trade.user_data, dict):
                trade.user_data["sl_pct"] = sl
                trade.user_data["tp_pct"] = tp
        except Exception:
            pass
        return True

    def on_close_filled(
        self,
        pair: str,
        trade,
        order,
        tier_mgr: TierManager,
    ) -> bool:
        """处理 on_close_filled 的主要逻辑。"""
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        if trade_id not in self.state.get_pair_state(pair).active_trades:
            return False

        profit_abs: float = 0.0
        if getattr(trade, "close_profit_abs", None) is not None:
            profit_abs = float(trade.close_profit_abs)
        elif getattr(trade, "profit_abs", None) is not None:
            profit_abs = float(trade.profit_abs)

        self.state.record_trade_close(pair, trade_id, profit_abs, tier_mgr)
        self.eq.on_trade_closed_update(profit_abs)
        return True

    def on_cancel_or_reject(self, pair: str, rid: Optional[str]) -> bool:
        """处理 on_cancel_or_reject 的主要逻辑。"""
        if not rid:
            return False
        self.reservation.release(rid)
        return True
