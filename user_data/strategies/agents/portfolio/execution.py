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
        pending_meta: Dict[str, Any] | None,
        tier_mgr: "TierManager",
    ) -> bool:
        """处理开仓成交事件。

        1) 将新仓登记进 GlobalState（含 ActiveTradeMeta）；
        2) 释放预约名额；
        3) 把 sl/tp 同步到 trade.custom_data / trade.user_data，供退出与止损逻辑使用。
        """
        # 取 trade_id（兼容不同属性名）
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        pst = self.state.get_pair_state(pair)

        # 若已登记则忽略重复回调
        if trade_id in getattr(pst, "active_trades", {}):
            return False

        # None 安全 & 兼容老键 sl/tp
        meta = pending_meta or {}
        sl = float(meta.get("sl_pct", meta.get("sl", 0.0)))
        tp = float(meta.get("tp_pct", meta.get("tp", 0.0)))
        direction = str(meta.get("dir", "")) or ("short" if getattr(trade, "is_short", False) else "long")
        rid = meta.get("reservation_id")
        bucket = str(meta.get("bucket", "slow"))
        real_risk = float(meta.get("risk_final", meta.get("risk", 0.0)))
        entry_price = float(meta.get("entry_price", getattr(trade, "open_rate", 0.0)))
        exit_profile = meta.get("exit_profile")
        recipe = meta.get("recipe")
        plan_timeframe = meta.get("plan_timeframe")
        plan_atr_pct = meta.get("atr_pct")

        # 正确的方法名：TierManager.get(closs)
        try:
            tier_pol = tier_mgr.get(getattr(pst, "closs", 0))
        except Exception:
            tier_pol = None

        # 计算当前这单的 stake_nominal（pre-leverage）
        stake_nominal = 0.0
        lev = getattr(getattr(self.cfg, "sizing", None), "enforce_leverage", None)
        if lev is None:
            lev = getattr(self.cfg, "enforce_leverage", 1.0)
        lev = float(lev or 1.0)
        if sl and sl > 0:
            stake_margin = real_risk / sl
            stake_nominal = stake_margin * lev

        # 按 GlobalState.record_trade_open 的签名顺序与命名传参
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
            stake_nominal=stake_nominal,  # ★ 新增
        )

        # 释放预约风险名额
        if rid:
            self.reservation.release(str(rid))

        # 将 sl/tp 同步写入 trade.custom_data / user_data
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
        """处理平仓成交事件，回收风险并更新权益。"""

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
        """在撤单或拒单时仅释放预约风险，不做财政回滚。"""

        if not rid:
            return False
        self.reservation.release(rid)
        return True
