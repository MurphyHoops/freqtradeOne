# -*- coding: utf-8 -*-
"""退出策略决策模块。

ExitPolicyV29 负责依据实时信号、盈利/亏损情况以及 ICU 倒计时判断是否
触发提前离场，同时提供早锁盈所需的止损上移距离。
"""

from __future__ import annotations

from typing import Optional


class ExitPolicyV29:
    """封装 TaxBrainV29 的自定义退出规则。"""

    def __init__(self, state, eq_provider, cfg) -> None:
        """记录运行时所需的状态引用。

        Args:
            state: GlobalState 实例，用于查询交易元数据与最新信号。
            eq_provider: EquityProvider，供风险降压判断读取权益。
            cfg: V29Config，提供 drawdown 阈值与早锁盈参数。
        """

        self.state = state
        self.eq = eq_provider
        self.cfg = cfg

    def decide(self, pair: str, trade_id: str, current_profit_pct: Optional[float]) -> Optional[str]:
        """根据当前信息判定是否需要提前离场。

        优先级顺序：
            1. 达到止盈目标 → 返回 "tp_hit"；
            2. ICU 倒计时归零 → 返回 "icu_timeout"；
            3. 信号反向且存在明显边际 → 返回 flip_xx；
            4. 组合进入压力区且当前亏损 → 返回 "risk_off"。

        Args:
            pair: 交易对名称。
            trade_id: Trade 的唯一标识。
            current_profit_pct: 当前收益率，若无则可能为 None。

        Returns:
            Optional[str]: 决策字符串，None 表示继续持有。
        """

        pst = self.state.get_pair_state(pair)
        meta = pst.active_trades.get(trade_id)
        if not meta:
            return None

        if current_profit_pct is not None and meta.tp_pct > 0 and current_profit_pct >= meta.tp_pct:
            return "tp_hit"

        if meta.icu_bars_left is not None and meta.icu_bars_left <= 0:
            return "icu_timeout"

        if pst.last_dir and pst.last_dir != meta.direction and pst.last_score > 0.01:
            return f"flip_{pst.last_dir}"

        equity = self.eq.get_equity()
        stress = (self.state.debt_pool / equity) if equity > 0 else 999.0
        if stress > self.cfg.drawdown_threshold_pct and (current_profit_pct is not None) and current_profit_pct < 0:
            return "risk_off"

        return None

    def early_lock_distance(self, trade, current_rate: float, current_profit: float, atr_pct_hint: float) -> Optional[float]:
        """计算早锁盈应上移的止损距离（负值表示靠近当前价）。

        Args:
            trade: Freqtrade Trade 对象。
            current_rate: 当前价格。
            current_profit: 当前收益率（未直接使用，保持兼容）。
            atr_pct_hint: 最近一次 ATR 百分比，用于估算偏移距离。

        Returns:
            Optional[float]: 若返回值为负则表示新的止损百分比偏移；None 表示无法计算。
        """

        try:
            float(trade.open_rate)
        except Exception:
            return None
        effective_atr = max(atr_pct_hint, 1e-6)
        lock_distance = max(self.cfg.breakeven_lock_eps_atr_pct * effective_atr, 0.0005)
        return -lock_distance
