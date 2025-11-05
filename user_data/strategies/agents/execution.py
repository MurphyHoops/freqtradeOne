# -*- coding: utf-8 -*-
"""交易执行事件的状态协调代理。

ExecutionAgent 在开仓、平仓以及撤单/拒单等生命周期钩子中负责更新
GlobalState、释放预约名额并同步止损/止盈元数据，确保风险账本与权益
记录保持一致。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .reservation import ReservationAgent
from .tier import TierManager, TierPolicy


class ExecutionAgent:
    """封装开仓、平仓与撤单事件处理的执行代理。"""

    def __init__(
        self,
        state,
        reservation: ReservationAgent,
        eq_provider,
        cfg,
    ) -> None:
        """初始化执行代理。

        Args:
            state: GlobalState 实例，记录在市风险与交易元数据。
            reservation: ReservationAgent，用于管理风险预约与释放。
            eq_provider: EquityProvider，负责实时维护权益数值。
            cfg: V29Config 配置对象，便于读取策略参数。
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
        pending_meta: Dict[str, Any],
        tier_mgr: TierManager,
    ) -> bool:
        """处理开仓成交事件。

        流程：
            1. 调用 GlobalState.record_trade_open 登记风险与 ActiveTradeMeta；
            2. 如存在预约 ID，则释放对应风险名额；
            3. 将止损/止盈写入 trade.custom_data 与 trade.user_data，供后续钩子读取。

        Args:
            pair: 成交的交易对名称。
            trade: Freqtrade Trade 对象。
            order: 成交订单对象（当前逻辑未直接使用）。
            pending_meta: confirm_trade_entry/custom_stake_amount 阶段缓存的数据。
            tier_mgr: TierManager，用于根据 closs 获取 TierPolicy。

        Returns:
            bool: 若首次登记成功返回 True，重复回调则返回 False。
        """

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
        """处理平仓成交事件，回收风险并更新权益。"""

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
        """在撤单或拒单时仅释放预约风险，不做财政回滚。"""

        if not rid:
            return False
        self.reservation.release(rid)
        return True
