"""TaxBrainV29 周期调度与财政流水线协调模块。

CycleAgent 在每根 K 线完成时负责驱动以下流程：
1. 推进全局 `bar_tick`、衰减债务/冷却计数；
2. 构造当前状态快照并调用 TreasuryAgent 生成极坐标拨款计划；
3. 根据盈利周期配置执行“盈利清债”（V29.1 修订 #2）；
4. 触发风险不变式检查、日志打点与状态持久化。
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from ...config.v29_config import V29Config
from .analytics import AnalyticsAgent
from .reservation import ReservationAgent
from .risk import RiskAgent
from .treasury import AllocationPlan, TreasuryAgent
from .global_backend import GlobalRiskBackend


class CycleAgent:
    """负责 bar 级别的节奏推进与财政拨款协调。"""

    def __init__(
        self,
        cfg: V29Config,
        state,
        reservation: ReservationAgent,
        treasury: TreasuryAgent,
        risk: RiskAgent,
        analytics: AnalyticsAgent,
        persist,
        tier_mgr,
        backend: Optional[GlobalRiskBackend] = None,
    ) -> None:
        """构造周期代理并注入全局依赖。

        Args:
            cfg: 运行时配置，提供周期长度、衰减参数等信息。
            state: GlobalState 实例，承载组合风险与财政状态。
            reservation: 预约代理，负责风险预约的维护与 TTL 推进。
            treasury: 财政代理，基于快照生成 fast/slow 拨款计划。
            risk: 风险代理，用于执行不变式校验。
            analytics: 日志代理，记录 finalize、invariant 等事件。
            persist: StateStore 包装，用于持久化全局状态。
            tier_mgr: TierManager，供财政计划与冷却逻辑读取 tier 规则。
        """

        self.cfg = cfg
        self.state = state
        self.reservation = reservation
        self.treasury = treasury
        self.risk = risk
        self.analytics = analytics
        self.persist = persist
        self.tier_mgr = tier_mgr
        self.backend = backend

    def finalize(self, eq_provider) -> AllocationPlan:
        """在所有交易对完成当前 bar 处理后执行一次完整 finalize。

        主要步骤：
        1. 推进 `bar_tick` 并衰减债务、冷却、ICU 计数；
        2. 构建财政计划快照并调用 TreasuryAgent 获取拨款；
        3. 根据配置判断是否触发盈利清债；
        4. 记录日志、执行风险不变式校验并持久化状态。

        Args:
            eq_provider: EquityProvider，用于读取当前组合权益。

        Returns:
            AllocationPlan: 最新的 K_long / K_short 拨款计划。
        """

        self.state.bar_tick += 1
        self._decay_and_cooldowns()

        equity = eq_provider.get_equity()
        snapshot = self._build_snapshot()
        plan = self.treasury.plan(snapshot, equity)
        self.state.treasury.k_long = plan.k_long
        self.state.treasury.k_short = plan.k_short
        self.state.treasury.theta = plan.theta
        self.state.treasury.final_r = plan.final_r
        self.state.treasury.available = plan.available
        self.state.treasury.bias = plan.bias
        self.state.treasury.volatility = plan.volatility

        if self.state.treasury.cycle_start_tick == 0:
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        pnl_since_cycle_start = equity - float(self.state.treasury.cycle_start_equity)
        cycle_completed = (
            self.state.bar_tick - self.state.treasury.cycle_start_tick
        ) >= int(self.cfg.cycle_len_bars)
        cycle_cleared = False
        if cycle_completed:
            if pnl_since_cycle_start >= 0 and bool(getattr(getattr(self.cfg, "risk", None), "clear_debt_on_profitable_cycle", getattr(self.cfg, "clear_debt_on_profitable_cycle", False))):
                cycle_cleared = True
                self.state.debt_pool = 0.0
                for pst in self.state.per_pair.values():
                    pst.local_loss = 0.0
                    pst.closs = 0  
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        cap_abs = cap_pct * equity
        used_risk = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        cap_used_pct = (used_risk / cap_abs) if cap_abs > 0 else 0.0
        reservations_count = len(self.reservation.reservations)

        tier_summary: dict[str, dict[str, Any]] = {}
        for pair, pst in self.state.per_pair.items():
            try:
                tier_pol = self.tier_mgr.get(pst.closs) if self.tier_mgr else None
                tier_name = getattr(tier_pol, "name", None) if tier_pol else None
            except Exception:
                tier_pol = None
                tier_name = None
            recipes = sorted(
                {meta.recipe for meta in pst.active_trades.values() if getattr(meta, "recipe", None)}
            )
            profiles = sorted(
                {meta.exit_profile for meta in pst.active_trades.values() if getattr(meta, "exit_profile", None)}
            )
            tier_summary[pair] = {
                "tier": tier_name,
                "active_trades": len(pst.active_trades),
                "active_recipes": recipes,
                "active_exit_profiles": profiles,
            }

        self.analytics.log_finalize(
            bar_tick=self.state.bar_tick,
            pnl=pnl_since_cycle_start,
            debt_pool=self.state.debt_pool,
            k_long=plan.k_long,
            k_short=plan.k_short,
            theta=plan.theta,
            final_r=plan.final_r,
            cap_used_pct=cap_used_pct,
            reservations=reservations_count,
            cycle_cleared=cycle_cleared,
            tier_summary=tier_summary,
        )

        report = self.risk.check_invariants(self.state, equity, cap_pct)
        report_payload = report.to_dict() if hasattr(report, "to_dict") else {"ok": True}
        self.analytics.log_invariant(report_payload)
        if not report_payload.get("ok", True):
            print("[WARN] Risk invariant breach:", report_payload)

        self.persist.save()
        return plan

    def maybe_finalize(
        self,
        pair: str,
        bar_ts: float,
        whitelist: Iterable[str],
        timeframe_sec: int,
        eq_provider,
    ) -> None:
        """根据报送进度与超时策略判断是否运行 finalize。

        Args:
            pair: 当前完成 populate_indicators 的交易对。
            bar_ts: 该 K 线的时间戳（秒）。
            whitelist: 当前交易白名单，用于判断是否全部上报完成。
            timeframe_sec: timeframe 对应的秒数，配合 `force_finalize_mult` 计算超时阈值。
            eq_provider: EquityProvider，用于在触发 finalize 时读取权益。
        """

        now = time.time()
        if self.state.current_cycle_ts is None or bar_ts > float(self.state.current_cycle_ts):
            self.state.current_cycle_ts = float(bar_ts)
            self.state.reported_pairs_for_current_cycle = set()
        self.state.reported_pairs_for_current_cycle.add(pair)

        all_reported = all(
            p in self.state.reported_pairs_for_current_cycle for p in whitelist
        )
        newer_than_last = (
            self.state.last_finalized_bar_ts is None
            or float(self.state.current_cycle_ts) > float(self.state.last_finalized_bar_ts)
        )
        timeout = (now - self.state.last_finalize_walltime) >= (
            self.cfg.force_finalize_mult * timeframe_sec
        )

        if newer_than_last and (all_reported or timeout):
            self.finalize(eq_provider)
            self.state.last_finalize_walltime = now
            self.state.last_finalized_bar_ts = float(self.state.current_cycle_ts)
            self.state.reported_pairs_for_current_cycle = set()

    def _decay_and_cooldowns(self) -> None:
        """衰减冷却/ICU 计数并推进预约 TTL。"""

        for pst in self.state.per_pair.values():
            if pst.cooldown_bars_left > 0:
                pst.cooldown_bars_left -= 1
            for meta in pst.active_trades.values():
                if meta.icu_bars_left is not None and meta.icu_bars_left > 0:
                    meta.icu_bars_left -= 1
        risk_cfg = getattr(self.cfg, "risk", self.cfg)
        self.state.debt_pool *= risk_cfg.pain_decay_per_bar
        self.reservation.tick_ttl()

    def _build_snapshot(self) -> dict:
        """组装财政拨款所需的状态快照。"""

        pairs_payload = {}
        for pair, pst in self.state.per_pair.items():
            pairs_payload[pair] = {
                "cooldown_bars_left": pst.cooldown_bars_left,
                "active_trades": len(pst.active_trades),
                "last_score": pst.last_score,
                "last_kind": getattr(pst, "last_kind", None),
                "last_dir": pst.last_dir,
                "last_squad": pst.last_squad,
                "last_sl_pct": pst.last_sl_pct,
                "last_tp_pct": pst.last_tp_pct,
                "local_loss": pst.local_loss,
                "closs": pst.closs,
                "pair_open_risk": self.state.pair_risk_open.get(pair, 0.0),
                "pair_reserved_risk": self.reservation.get_pair_reserved(pair),
            }
        return {
            "debt_pool": self.state.debt_pool,
            "total_open_risk": self.state.get_total_open_risk(),
            "reserved_portfolio_risk": self.reservation.get_total_reserved(),
            "pairs": pairs_payload,
        }
