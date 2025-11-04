from __future__ import annotations

from typing import Optional


class ExitPolicyV29:
    """ExitPolicyV29 的职责说明。"""
    def __init__(self, state, eq_provider, cfg) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.state = state
        self.eq = eq_provider
        self.cfg = cfg

    def decide(self, pair: str, trade_id: str, current_profit_pct: Optional[float]) -> Optional[str]:
        """处理 decide 的主要逻辑。"""
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

    def early_lock_distance(self, trade, current_rate: float, current_profit: float,
                            atr_pct_hint: float) -> Optional[float]:
        """根据方向返回早锁盈的目标止损距离（相对当前价格的负数）。"""
        try:
            open_rate = float(trade.open_rate)
        except Exception:
            return None
        is_long = bool(getattr(trade, "is_long", not getattr(trade, "is_short", False)))
        effective_atr = max(atr_pct_hint, 1e-6)
        lock_distance = max(self.cfg.breakeven_lock_eps_atr_pct * effective_atr, 0.0005)
        # 对称处理多空方向，转换为负距离供 freqtrade 使用。
        return -lock_distance
