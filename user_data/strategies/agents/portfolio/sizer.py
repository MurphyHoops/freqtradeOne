# -*- coding: utf-8 -*-
"""仓位大小与风险分配计算模块。

SizerAgent 综合基础 VaR、财政拨款、恢复目标与组合/单票 CAP 等限制，
在 custom_stake_amount 阶段计算实际下单金额与风险敞口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ...config.v29_config import V29Config
from .reservation import ReservationAgent
from .tier import TierManager


@dataclass
class SizingContext:
    pair: str
    sl_pct: float
    tp_pct: float
    direction: str
    min_stake: Optional[float]
    max_stake: float
    proposed_stake: Optional[float]


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

        ctx = SizingContext(
            pair=pair,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            direction=direction,
            min_stake=min_stake,
            max_stake=max_stake,
            proposed_stake=proposed_stake,
        )
        return self._compute_internal(ctx)

    def _compute_internal(self, ctx: SizingContext) -> Tuple[float, float, str]:
        equity = self.eq.get_equity()
        pst = self.state.get_pair_state(ctx.pair)
        tier_pol = self.tier_mgr.get(pst.closs)
        # --- 新增：单仓位限制 ---
        # 当某个 tier 配置 single_position_only=True 时，只要该交易对有在市仓位，
        # 无论是回测还是实盘，都直接返回 0 仓位，阻止再开新仓。
        if tier_pol and getattr(tier_pol, "single_position_only", False) and pst.active_trades:
            # bucket 返回什么无所谓，反正 stake=0 不会下单
            return (0.0, 0.0, "fast")
        
        base_risk = self._baseline_risk(equity, tier_pol)
        risk_local_need = self._apply_recovery_policy(base_risk, ctx.tp_pct, ctx.sl_pct, pst, tier_pol)

        fast = self.state.treasury.fast_alloc_risk.get(ctx.pair, 0.0)
        slow = self.state.treasury.slow_alloc_risk.get(ctx.pair, 0.0)
        alloc_total = max(fast, slow)
        bucket = "fast" if fast >= slow else "slow"
        risk_wanted = max(risk_local_need, alloc_total)

        risk_room = self._available_risk_room(ctx.pair, equity, tier_pol)

        risk_final = min(risk_wanted, risk_room)
        if risk_final <= 0 or ctx.sl_pct <= 0:
            print(
                "[SIZER_ZERO]",
                ctx.pair,
                "equity", equity,
                "debt_pool", self.state.debt_pool,
                "cap_pct", self.state.get_dynamic_portfolio_cap_pct(equity),
                "port_cap", self.state.get_dynamic_portfolio_cap_pct(equity) * equity,
                "open_risk", self.state.get_total_open_risk(),
                "reserved_port", self.reservation.get_total_reserved(),
                "pair_open", self.state.pair_risk_open.get(ctx.pair, 0.0),
                "pair_reserved", self.reservation.get_pair_reserved(ctx.pair),
                "tier", tier_pol.name,
                "per_pair_cap_pct", tier_pol.per_pair_risk_cap_pct,
                "risk_wanted", risk_wanted,
                "risk_room", risk_room,
            )
            return (0.0, 0.0, bucket)

        stake_nominal = risk_final / ctx.sl_pct
        stake_cap_notional = tier_pol.max_stake_notional_pct * equity
        stake_nominal = min(stake_nominal, stake_cap_notional, float(ctx.max_stake))
        if ctx.proposed_stake is not None:
            if ctx.proposed_stake <= 0:
                return (0.0, 0.0, bucket)
            stake_nominal = min(stake_nominal, float(ctx.proposed_stake))
        if ctx.min_stake is not None:
            stake_nominal = max(stake_nominal, float(ctx.min_stake))
        if stake_nominal <= 0:
            return (0.0, 0.0, bucket)

        real_risk = stake_nominal * ctx.sl_pct
        return (float(stake_nominal), float(real_risk), bucket)

    def _baseline_risk(self, equity: float, tier_pol) -> float:
        base = tier_pol.k_mult_base_pct * equity
        if not self.cfg.suppress_baseline_when_stressed or equity <= 0:
            return base
        drawdown = self.state.debt_pool / equity
        if drawdown > self.cfg.drawdown_threshold_pct:
            return 0.0
        return base

    def _apply_recovery_policy(
        self,
        baseline: float,
        tp_pct: float,
        sl_pct: float,
        pst,
        tier_pol,
    ) -> float:
        risk_need = baseline
        if tier_pol.sizing_algo == "TARGET_RECOVERY" and tp_pct > 0 and sl_pct > 0:
            want_rec = pst.local_loss * tier_pol.recovery_factor
            stake_rec = want_rec / tp_pct
            risk_rec = stake_rec * sl_pct
            risk_need = max(baseline, risk_rec)
        return risk_need

    def _available_risk_room(self, pair: str, equity: float, tier_pol) -> float:
        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        port_cap = cap_pct * equity
        used = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        port_room = max(0.0, port_cap - used)
        pair_reserved = self.reservation.get_pair_reserved(pair)
        pair_room = self.state.per_pair_cap_room(pair, equity, tier_pol, pair_reserved)
        return max(0.0, min(port_room, pair_room))
