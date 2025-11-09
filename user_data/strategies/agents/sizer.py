# -*- coding: utf-8 -*-
"""仓位大小与风险分配计算模块。

SizerAgent 综合基础 VaR、财政拨款、恢复目标与组合/单票 CAP 等限制，
在 custom_stake_amount 阶段计算实际下单金额与风险敞口。
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..config.v29_config import V29Config
from .reservation import ReservationAgent
from .tier import TierManager


class SizerAgent:
    """将策略参数与财政分配合成最终仓位的代理。"""

    def __init__(
        self,
        state,
        reservation: ReservationAgent,
        eq_provider,
        cfg: V29Config,
        tier_mgr: TierManager,
    ) -> None:
        """初始化仓位计算器所需的依赖。"""

        self.state = state
        self.reservation = reservation
        self.eq = eq_provider
        self.cfg = cfg
        self.tier_mgr = tier_mgr

    def compute(
        self,
        pair: str,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        min_stake: Optional[float],
        max_stake: float,
        proposed_stake: Optional[float] = None,
    ) -> Tuple[float, float, str]:
        """计算名义下单量、实际风险以及拨款桶归属。

        核心步骤：
            1. 根据 TierPolicy 获取基础 VaR (k_mult_base_pct * equity)，压力期可抑制；
            2. TARGET_RECOVERY 档位会考虑局部亏损以提高风险需求；
            3. 与财政 fast/slow 拨款合并取最大需求，再尊重组合/单票 CAP；
            4. 施加名义仓位上限与交易所 min/max 限制。

        Args:
            pair: 交易对名称。
            sl_pct: 止损百分比（正值）。
            tp_pct: 止盈百分比。
            direction: 信号方向（当前逻辑未区分 long/short）。
            min_stake: 交易所最小下单额，None 表示无限制。
            max_stake: 交易所允许的最大下单额。

        Returns:
            Tuple[float, float, str]: (名义下单额, 实际风险, 使用拨款桶)。
        """

        equity = self.eq.get_equity()
        pst = self.state.get_pair_state(pair)
        tier_pol = self.tier_mgr.get(pst.closs)

        base_risk = tier_pol.k_mult_base_pct * equity
        if self.cfg.suppress_baseline_when_stressed and equity > 0:
            if (self.state.debt_pool / equity) > self.cfg.drawdown_threshold_pct:
                base_risk = 0.0

        risk_local_need = base_risk
        if tier_pol.sizing_algo == "TARGET_RECOVERY" and tp_pct > 0:
            want_rec = pst.local_loss * tier_pol.recovery_factor
            stake_rec = want_rec / tp_pct
            risk_rec = stake_rec * sl_pct
            risk_local_need = max(base_risk, risk_rec)

        fast = self.state.treasury.fast_alloc_risk.get(pair, 0.0)
        slow = self.state.treasury.slow_alloc_risk.get(pair, 0.0)
        alloc_total = max(fast, slow)
        bucket = "fast" if fast >= slow else "slow"
        risk_wanted = max(risk_local_need, alloc_total)

        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        port_cap = cap_pct * equity
        used = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        port_room = max(0.0, port_cap - used)
        pair_reserved = self.reservation.get_pair_reserved(pair)
        pair_room = self.state.per_pair_cap_room(pair, equity, tier_pol, pair_reserved)
        risk_room = max(0.0, min(port_room, pair_room))

        risk_final = min(risk_wanted, risk_room)
        if risk_final <= 0 or sl_pct <= 0:
            return (0.0, 0.0, bucket)

        stake_nominal = risk_final / sl_pct
        stake_cap_notional = tier_pol.max_stake_notional_pct * equity
        stake_nominal = min(stake_nominal, stake_cap_notional, float(max_stake))
        if proposed_stake is not None:
            if proposed_stake <= 0:
                return (0.0, 0.0, bucket)
            stake_nominal = min(stake_nominal, float(proposed_stake))
        if min_stake is not None:
            stake_nominal = max(stake_nominal, float(min_stake))
        if stake_nominal <= 0:
            return (0.0, 0.0, bucket)

        real_risk = stake_nominal * sl_pct
        return (float(stake_nominal), float(real_risk), bucket)
