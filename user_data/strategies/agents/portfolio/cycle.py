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
        engine: Any | None = None,
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
        self.engine = engine

    def finalize(self, eq_provider) -> AllocationPlan:
        """委托 Engine 完成 finalize 逻辑。"""

        if not self.engine:
            raise RuntimeError("CycleAgent.engine is not attached")
        return self.engine.finalize_bar()

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
