# -*- coding: utf-8 -*-
"""Execution event coordinator for trades."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .reservation import ReservationAgent
from .tier import TierManager, TierPolicy


class ExecutionAgent:
    """Handle open/close/cancel lifecycle events."""

    def __init__(
        self,
        state,
        reservation: ReservationAgent,
        eq_provider,
        cfg,
    ) -> None:
        """Initialize execution agent.

        Args:
            state: GlobalState instance for risk and trade metadata.
            reservation: ReservationAgent to manage reservations.
            eq_provider: EquityProvider for live equity updates.
            cfg: V30Config for sizing parameters.
        """

        self.state = state
        self.reservation = reservation
        self.eq = eq_provider
        self.cfg = cfg

    def on_open_filled(
        self,
        pair: str,
        trade,
        order,
        pending_meta: Dict[str, Any] | None,
        tier_mgr: "TierManager",
    ) -> bool:
        """Handle open fill events.

        1) Register the new trade in GlobalState (ActiveTradeMeta).
        2) Release reservation slots.
        3) Sync sl/tp metadata to trade.custom_data / trade.user_data.
        """

        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        pst = self.state.get_pair_state(pair)

        if trade_id in getattr(pst, "active_trades", {}):
            return False

        meta = pending_meta or {}
        sl = float(meta.get("sl_pct", meta.get("sl", 0.0)))
        tp = float(meta.get("tp_pct", meta.get("tp", 0.0)))
        direction = str(meta.get("dir", "")) or ("short" if getattr(trade, "is_short", False) else "long")
        rid = meta.get("reservation_id")
        bucket = str(meta.get("bucket", direction or "long"))
        real_risk = float(meta.get("risk_final", meta.get("risk", 0.0)))
        entry_price = float(meta.get("entry_price", getattr(trade, "open_rate", 0.0)))
        exit_profile = meta.get("exit_profile")
        recipe = meta.get("recipe")
        plan_timeframe = meta.get("plan_timeframe")
        plan_atr_pct = meta.get("atr_pct")

        try:
            tier_pol = tier_mgr.get(getattr(pst, "closs", 0))
        except Exception:
            tier_pol = None

        stake_nominal = 0.0
        sizing_cfg = getattr(getattr(self.cfg, "trading", None), "sizing", None)
        lev = float(getattr(sizing_cfg, "enforce_leverage", 1.0) or 1.0)
        if sl and sl > 0:
            stake_margin = real_risk / sl
            stake_nominal = stake_margin * lev

        self.state.record_trade_open(
            pair=pair,
            trade_id=trade_id,
            real_risk=real_risk,
            sl_pct=sl,
            tp_pct=tp,
            direction=direction,
            bucket=bucket,
            entry_price=entry_price,
            tier_pol=tier_pol,
            exit_profile=exit_profile,
            recipe=recipe,
            plan_timeframe=plan_timeframe,
            plan_atr_pct=plan_atr_pct,
            tier_name=getattr(tier_pol, "name", None) if tier_pol else None,
            stake_nominal=stake_nominal,
        )

        if rid:
            self.reservation.release(str(rid))

        tier_name = getattr(tier_pol, "name", None) if tier_pol else None
        try:
            if hasattr(trade, "set_custom_data"):
                trade.set_custom_data("sl_pct", sl)
                trade.set_custom_data("tp_pct", tp)
                if exit_profile:
                    trade.set_custom_data("exit_profile", exit_profile)
                if recipe:
                    trade.set_custom_data("recipe", recipe)
                if tier_name:
                    trade.set_custom_data("tier_name", tier_name)
                if plan_timeframe:
                    trade.set_custom_data("plan_timeframe", plan_timeframe)
                if plan_atr_pct:
                    trade.set_custom_data("atr_pct", plan_atr_pct)
        except Exception:
            pass
        try:
            if hasattr(trade, "user_data") and isinstance(trade.user_data, dict):
                trade.user_data["sl_pct"] = sl
                trade.user_data["tp_pct"] = tp
                if exit_profile:
                    trade.user_data["exit_profile"] = exit_profile
                if recipe:
                    trade.user_data["recipe"] = recipe
                if tier_name:
                    trade.user_data["tier_name"] = tier_name
                if plan_timeframe:
                    trade.user_data["plan_timeframe"] = plan_timeframe
                if plan_atr_pct:
                    trade.user_data["atr_pct"] = plan_atr_pct
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
        """Handle close fill events, update risk and equity."""

        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        if trade_id not in self.state.get_pair_state(pair).active_trades:
            return False

        profit_abs: float = 0.0
        if getattr(trade, "close_profit_abs", None) is not None:
            profit_abs = float(trade.close_profit_abs)

        self.state.record_trade_close(pair, trade_id, profit_abs, tier_mgr)
        self.eq.on_trade_closed_update(profit_abs)
        return True

    def on_cancel_or_reject(self, pair: str, rid: Optional[str]) -> bool:
        """Release reservation on cancel or reject."""

        if not rid:
            return False
        self.reservation.release(rid)
        return True
