"""财政拨款规划模块

该模块封装了 TaxBrain V29 在每个 finalize 周期内的财政拨款决策流程。CycleAgent 会
传入一个包含组合风险、债务、各交易对信号与冷却状态的快照，TreasuryAgent 负责将
可用风险预算在 fast / slow 两个拨款桶之间分配，再细化到具体交易对。所有计算都是
围绕 V29Config 中的风险上限（CAP）、压力期折减、编队（squad）选择以及最小名义注入
等参数展开，确保拨款结果既满足风险约束，又兼顾信号质量与恢复效率。

使用示例::
    >>> from user_data.strategies.agents.portfolio.treasury import TreasuryAgent, AllocationPlan
    >>> from user_data.strategies.config.v29_config import V29Config
    >>> from user_data.strategies.agents.portfolio.tier import TierManager
    >>> cfg = V29Config()
    >>> tier_mgr = TierManager(cfg)
    >>> treasury = TreasuryAgent(cfg, tier_mgr)
    >>> snapshot = {
    ...     "debt_pool": 120.0,
    ...     "total_open_risk": 40.0,
    ...     "reserved_portfolio_risk": 10.0,
    ...     "pairs": {
    ...         "BTC/USDT": {
    ...             "closs": 0,
    ...             "local_loss": 5.0,
    ...             "last_score": 8.0,
    ...             "last_dir": "long",
    ...             "last_kind": "mean_rev_long",
    ...             "last_squad": "MRL",
    ...             "last_sl_pct": 0.02,
    ...             "cooldown_bars_left": 0,
    ...             "active_trades": 0,
    ...         }
    ...     },
    ... }
    >>> plan = treasury.plan(snapshot, equity=1000.0)
    >>> isinstance(plan, AllocationPlan)
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ...config.v29_config import V29Config
from .tier import TierManager
from .global_backend import GlobalRiskBackend
import math


@dataclass
class AllocationPlan:
    """表示一次 finalize 周期结束后财政拨款的完整结果。

    Attributes:
        fast_alloc_risk (Dict[str, float]): fast 桶（追求恢复速度）针对每个交易对的名义风险额度。
        slow_alloc_risk (Dict[str, float]): slow 桶（偏稳健扩散）针对每个交易对的名义风险额度。

    Notes:
        - fast 桶通常用于最具说服力的 squad 代表信号，强调恢复速度与高动量；
        - slow 桶更侧重覆盖率，按打分排序选出前 `slow_universe_pct` 的候选；
        - 若某一桶在当前周期没有可用预算或没有符合条件的交易对，则对应字典为空。
    """

    fast_alloc_risk: Dict[str, float] = field(default_factory=dict)
    slow_alloc_risk: Dict[str, float] = field(default_factory=dict)


def _dynamic_portfolio_cap_pct(cfg: V29Config, debt_pool: float, equity: float) -> float:
    """根据债务压力动态调整组合 VaR 上限比例。

    当债务率超过配置中的 `drawdown_threshold_pct` 时，组合 CAP 将折半，以减缓新增敞口；
    equity 小于等于 0 时也会直接对 CAP 做 0.5 倍折减。

    Args:
        cfg: V29Config 配置实例。
        debt_pool: 当前累计的组合债务（亏损负载）。
        equity: 当前账户权益。

    Returns:
        float: 调整后的组合 VaR 占比上限（0~1 之间），不会返回负值。
    """

    base = cfg.portfolio_cap_pct_base
    if equity <= 0:
        return base * 0.5
    if (debt_pool / equity) > cfg.drawdown_threshold_pct:
        return base * 0.5
    return base


class TreasuryAgent:
    """财政拨款代理，负责将可用风险预算分配到 fast / slow 拨款桶。

    该代理依赖 TierManager 获取不同 CLOSS 等级下的单票 CAP 限制，并结合 StateSnapshot
    中的候选信号、冷却状态、预约与在市风险，计算出本周期允许的新名义风险额度。
    """

    def __init__(self, cfg: V29Config, tier_mgr: TierManager, backend: Optional[GlobalRiskBackend] = None) -> None:
        """构造财政代理。

        Args:
            cfg: 策略全局配置，用于读取 CAP、拨款比例、最小注入等参数。
            tier_mgr: 分层策略管理器，用于获取指定 CLOSS 对应的 TierPolicy（含单票 CAP、允许的 squad 等）。
        """

        self.cfg = cfg
        self.tier_mgr = tier_mgr
        self.backend = backend

    def plan(self, state_snapshot: Dict[str, any], equity: float) -> AllocationPlan:
        """根据最新状态快照与权益规模生成拨款计划。

        主要流程：

        1. 读取组合层面的债务、在市风险、已预约风险，计算还可使用的组合 CAP；
        2. 将可用风险预算按照 `treasury_fast_split_pct` 拆分为 fast / slow 桶；
        3. 基于候选打分列表，筛选符合 squad 限制且未冷却的交易对；
        4. fast 桶优先取每个 squad 的头部信号，再截断到 `fast_topK_squads`；
        5. slow 桶按排序截取前 `len(scored) * slow_universe_pct` 的候选；
        6. 对每个交易对应用单票 CAP、最小名义注入以及剩余空间约束，得到最终额度。

        Args:
            state_snapshot: CycleAgent 提供的快照，结构要求：
                {
                    "debt_pool": float,
                    "total_open_risk": float,
                    "reserved_portfolio_risk": float,
                    "pairs": {
                        "BTC/USDT": {
                            "closs": int,
                            "local_loss": float,
                            "last_score": float,
                            "last_dir": "long" | "short",
                            "last_squad": str,
                            "last_sl_pct": float,
                            "cooldown_bars_left": int,
                            "active_trades": int,
                            "pair_open_risk": float,
                            "pair_reserved_risk": float,
                        }, ...
                    }
                }
            equity: 当前组合权益，用于计算 CAP 与最小名义注入口径。

        Returns:
            AllocationPlan: fast 与 slow 两个拨款桶的结果；若预算为 0 则两个字典皆为空。
        """

        if self.backend:
            snap = self.backend.get_snapshot()
            state_snapshot = dict(state_snapshot)
            state_snapshot["debt_pool"] = snap.debt_pool

        debt_pool = float(state_snapshot.get("debt_pool", 0.0))
        total_open_risk = float(state_snapshot.get("total_open_risk", 0.0))
        reserved_portfolio = float(state_snapshot.get("reserved_portfolio_risk", 0.0))

        cap_pct = _dynamic_portfolio_cap_pct(self.cfg, debt_pool, equity)
        portfolio_cap = cap_pct * equity
        used_cap = total_open_risk + reserved_portfolio
        free_cap = max(0.0, portfolio_cap - used_cap)

        # 拨款额度不得超过债务池（说明还有需要恢复的亏损），同样受组合 CAP 限制。
        available_budget = min(debt_pool, free_cap)
        if available_budget <= 0:
            return AllocationPlan()

        tcfg = self.cfg.treasury
        fast_budget = available_budget * tcfg.treasury_fast_split_pct
        slow_budget = available_budget - fast_budget

        pairs_data = state_snapshot.get("pairs", {}) or {}
        scored: List[Tuple[str, float, str]] = []
        for pair, pdata in pairs_data.items():
            if pdata.get("cooldown_bars_left", 0) > 0:
                continue
            if pdata.get("active_trades", 0) > 0:
                continue
            last_score = float(pdata.get("last_score", 0.0))
            last_dir = pdata.get("last_dir")
            last_kind = pdata.get("last_kind")
            last_squad = pdata.get("last_squad")
            if last_score <= 0 or not last_dir or not last_kind:
                continue
            tier_pol = self.tier_mgr.get(int(pdata.get("closs", 0)))
            if not tier_pol.permits(kind=last_kind, squad=last_squad):
                continue
            pain_weight = 1.0
            if equity > 0:
                pain_weight += min(3.0, float(pdata.get("local_loss", 0.0)) / max(1e-9, 0.02 * equity))
            scored.append((pair, last_score * pain_weight, last_squad))

        scored.sort(key=lambda item: item[1], reverse=True)

        # fast 桶：为每个 squad 选出得分最高的代表信号，再按表现排序取前 K 个。
        # 假设新增 cfg.fast_mode in {"per_squad", "top_pairs"}

        if tcfg.fast_mode == "top_pairs":
            fast_pairs = [pair for (pair, _score, _sq) in scored[: tcfg.fast_topK_squads]]
        else:
            # 原 per-squad 代表逻辑
            best_by_squad = {}
            for pair, score, squad in scored:
                if (squad not in best_by_squad) or (score > best_by_squad[squad][1]):
                    best_by_squad[squad] = (pair, score)
            top_squads = sorted(best_by_squad.items(), key=lambda kv: kv[1][1], reverse=True)
            if tcfg.fast_topK_squads > 0:
                top_squads = top_squads[: tcfg.fast_topK_squads]
            fast_pairs = [pair for (_squad, (pair, _score)) in top_squads]

        fast_each = 0.0
        if fast_budget > 0 and fast_pairs:
            fast_each = fast_budget / len(fast_pairs)

        # slow 桶：直接取前 M 名得分，M 由 slow_universe_pct 控制。
        slow_cutoff = max(1, math.ceil(len(scored) * tcfg.slow_universe_pct))
        slow_list = scored[:slow_cutoff]
        slow_each = 0.0
        if slow_budget > 0 and slow_list:
            slow_each = slow_budget / len(slow_list)

        fast_alloc: Dict[str, float] = {}
        slow_alloc: Dict[str, float] = {}

        def per_pair_room(pair: str, tier_policy) -> float:
            """计算指定交易对在单票 CAP 限制下仍可使用的风险空间。"""

            pair_open = float(pairs_data[pair].get("pair_open_risk", 0.0))
            pair_reserved = float(pairs_data[pair].get("pair_reserved_risk", 0.0))
            cap = tier_policy.per_pair_risk_cap_pct * equity
            return max(0.0, cap - (pair_open + pair_reserved))

        if fast_budget > 0:
            for pair in fast_pairs:
                pdata = pairs_data[pair]
                tier_pol = self.tier_mgr.get(int(pdata.get("closs", 0)))
                min_risk = max(0.0, tcfg.min_injection_nominal_fast * max(1e-9, float(pdata.get("last_sl_pct", 0.0))))
                room = per_pair_room(pair, tier_pol)
                candidate = max(fast_each, min_risk)
                alloc = min(candidate, room)
                if alloc > 0:
                    fast_alloc[pair] = alloc

        if slow_budget > 0:
            for pair, _score, _squad in slow_list:
                pdata = pairs_data[pair]
                tier_pol = self.tier_mgr.get(int(pdata.get("closs", 0)))
                min_risk = max(0.0, tcfg.min_injection_nominal_slow * max(1e-9, float(pdata.get("last_sl_pct", 0.0))))
                room = per_pair_room(pair, tier_pol)
                candidate = max(slow_each, min_risk)
                alloc = min(candidate, room)
                if alloc > 0:
                    slow_alloc[pair] = alloc

        return AllocationPlan(fast_alloc_risk=fast_alloc, slow_alloc_risk=slow_alloc)


    def evaluate_signal_quality(self, pair: str, score: float) -> dict:
        """Gate incoming signals using global debt and percentile thresholds."""
        if not self.backend:
            return {"allowed": True, "bucket": "slow", "reason": "no_backend"}

        snap = self.backend.get_snapshot()
        debt = float(getattr(snap, 'debt_pool', 0.0) or 0.0)
        threshold_high = float(self.backend.get_score_percentile_threshold(80))
        is_high_quality = bool(score >= threshold_high)

        if debt > 0:
            if is_high_quality:
                return {"allowed": True, "bucket": "fast", "reason": "high_quality_with_debt", "threshold": threshold_high}
            return {"allowed": False, "bucket": "slow", "reason": "Debt protection", "threshold": threshold_high}

        bucket = "fast" if is_high_quality else "slow"
        return {"allowed": True, "bucket": bucket, "reason": "no_debt", "threshold": threshold_high}
