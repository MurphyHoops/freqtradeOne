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
            return (0.0, 0.0, bucket)

        stake_nominal = risk_final / ctx.sl_pct

        # ==== 新增：按照 sizing 模式，对 stake_nominal 做二次约束 ====
        mode = getattr(self.cfg, "initial_size_mode", "hybrid")
        lev = float(self.cfg.enforce_leverage or 1.0)

        # 1）静态基础：静态名义（如果为 0，则用 Freqtrade 的 min_stake）
        static_nominal = float(self.cfg.static_initial_nominal or 0.0)
        if static_nominal <= 0 and ctx.min_stake is not None:
            # ctx.min_stake 为 pre-leverage 的保证金，用它当静态初始手数
            static_nominal = float(ctx.min_stake)

        # 2）动态基础：equity * 百分比（转换成 pre-leverage 保证金）
        dynamic_nominal = equity * float(self.cfg.initial_size_equity_pct or 0.0)

        # 3）本笔的最大名义（per trade）
        per_trade_max_nominal = float(self.cfg.initial_max_nominal_per_trade or 0.0)
        if per_trade_max_nominal <= 0:
            # 如果没有单笔上限，就只用「总仓位 3000」那个 CAP
            per_trade_max_nominal = float("inf")

        # 根据模式决定目标开仓名义（pre-leverage）
        if mode == "static":
            target_nominal = static_nominal
        elif mode == "dynamic":
            # 动态：不少于 min_stake，不超过单笔上限
            target_nominal = max(static_nominal, dynamic_nominal)
        else:  # "hybrid"
            # 动静结合：动态目标落在 [static_nominal, per_trade_max_nominal] 之间
            target_nominal = dynamic_nominal
            if static_nominal > 0:
                target_nominal = max(target_nominal, static_nominal)
            target_nominal = min(target_nominal, per_trade_max_nominal)

        # 最终 stake_nominal 不能超过「VaR 需求」和「尺寸策略」两者之一
        stake_nominal = min(stake_nominal, target_nominal)

        # === 新增：按名义仓位上限限制单币总保证金 ===
        lev = float(self.cfg.enforce_leverage or 1.0)
        per_pair_nominal_cap = float(self.cfg.per_pair_max_nominal_static or 0.0)

        # 如果配置了 3000，则换算出「保证金上限」= 3000 / 杠杆
        per_pair_stake_cap = per_pair_nominal_cap / lev if per_pair_nominal_cap > 0 else float("inf")
        used_stake = self.state.pair_stake_open.get(ctx.pair, 0.0)
        stake_room_pair = max(0.0, per_pair_stake_cap - used_stake)

        # 原有的单笔 CAP（相对 equity）
        stake_cap_notional = tier_pol.max_stake_notional_pct * equity

        # 把所有 CAP 一起作用：单笔、全局 max_stake、单币名义 CAP
        stake_nominal = min(
            stake_nominal,
            stake_cap_notional,
            float(ctx.max_stake),
            stake_room_pair,
        )

        # 如果 CAP 已经不允许再开仓，直接返回 0
        if stake_nominal <= 0 or stake_room_pair <= 0:
            return (0.0, 0.0, bucket)

        # min_stake 只是“不要太小”，不能突破 CAP
        if ctx.min_stake is not None and stake_nominal < float(ctx.min_stake):
            # 想开的仓位达不到交易所最小要求，但 CAP 又不能给更多 → 放弃这笔机会
            stake_nominal = ctx.min_stake

        if ctx.proposed_stake is not None:
            if ctx.proposed_stake <= 0:
                return (0.0, 0.0, bucket)
            stake_nominal = min(stake_nominal, float(ctx.proposed_stake))

        # 此时 stake_nominal 已经在 [min_stake, CAP] 区间内
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
