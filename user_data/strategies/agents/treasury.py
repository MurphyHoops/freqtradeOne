from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..config.v29_config import V29Config
from .tier import TierManager


@dataclass
class AllocationPlan:
    """AllocationPlan 的职责说明。"""
    fast_alloc_risk: Dict[str, float] = field(default_factory=dict)
    slow_alloc_risk: Dict[str, float] = field(default_factory=dict)


def _dynamic_portfolio_cap_pct(cfg: V29Config, debt_pool: float, equity: float) -> float:
    """处理 _dynamic_portfolio_cap_pct 的主要逻辑。"""
    base = cfg.portfolio_cap_pct_base
    if equity <= 0:
        return base * 0.5
    if (debt_pool / equity) > cfg.drawdown_threshold_pct:
        return base * 0.5
    return base


class TreasuryAgent:
    """TreasuryAgent 的职责说明。"""
    def __init__(self, cfg: V29Config, tier_mgr: TierManager) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.cfg = cfg
        self.tier_mgr = tier_mgr

    def plan(self, state_snapshot: dict, equity: float) -> AllocationPlan:
        """处理 plan 的主要逻辑。"""
        debt_pool = float(state_snapshot.get("debt_pool", 0.0))
        total_open_risk = float(state_snapshot.get("total_open_risk", 0.0))
        reserved_portfolio = float(state_snapshot.get("reserved_portfolio_risk", 0.0))

        cap_pct = _dynamic_portfolio_cap_pct(self.cfg, debt_pool, equity)
        port_cap = cap_pct * equity
        used = total_open_risk + reserved_portfolio
        free = max(0.0, port_cap - used)

        needed_risk = min(debt_pool, free)
        if needed_risk <= 0:
            return AllocationPlan()

        fast_budget = needed_risk * self.cfg.treasury_fast_split_pct
        slow_budget = needed_risk - fast_budget

        pairs_data = state_snapshot.get("pairs", {})
        scored: List[Tuple[str, float, str]] = []
        for pair, pdata in pairs_data.items():
            if pdata.get("cooldown_bars_left", 0) > 0:
                continue
            if pdata.get("active_trades", 0) > 0:
                continue
            last_score = float(pdata.get("last_score", 0.0))
            last_dir = pdata.get("last_dir")
            last_squad = pdata.get("last_squad")
            if last_score <= 0 or not last_dir or not last_squad:
                continue
            tier_pol = self.tier_mgr.get(int(pdata.get("closs", 0)))
            if last_squad not in tier_pol.allowed_kinds:
                continue
            pain = 1.0
            if equity > 0:
                pain += min(3.0, float(pdata.get("local_loss", 0.0)) / (0.02 * equity))
            scored.append((pair, last_score * pain, last_squad))

        scored.sort(key=lambda x: x[1], reverse=True)

        best_by_squad: Dict[str, Tuple[str, float]] = {}
        for pair, score, squad in scored:
            if (squad not in best_by_squad) or (score > best_by_squad[squad][1]):
                best_by_squad[squad] = (pair, score)
        top_squads = sorted(best_by_squad.items(), key=lambda kv: kv[1][1], reverse=True)
        if self.cfg.fast_topK_squads > 0:
            top_squads = top_squads[: self.cfg.fast_topK_squads]
        fast_pairs = [pair for (_squad, (pair, _score)) in top_squads]
        fast_each = 0.0 if fast_budget <= 0 else fast_budget / max(1, len(fast_pairs)) if fast_pairs else 0.0

        m = max(1, int(len(scored) * self.cfg.slow_universe_pct))
        slow_list = scored[:m]
        slow_each = 0.0 if slow_budget <= 0 else slow_budget / max(1, len(slow_list)) if slow_list else 0.0

        fast_alloc: Dict[str, float] = {}
        slow_alloc: Dict[str, float] = {}

        def per_pair_room(pair: str, tier_pol) -> float:
            """处理 per_pair_room 的主要逻辑。"""
            pair_open = float(pairs_data[pair].get("pair_open_risk", 0.0))
            pair_reserved = float(pairs_data[pair].get("pair_reserved_risk", 0.0))
            cap = tier_pol.per_pair_risk_cap_pct * equity
            return max(0.0, cap - (pair_open + pair_reserved))

        if fast_budget > 0:
            for pair in fast_pairs:
                pdata = pairs_data[pair]
                tier_pol = self.tier_mgr.get(int(pdata.get("closs", 0)))
                min_risk = max(0.0, self.cfg.min_injection_nominal_fast * max(1e-9, float(pdata.get("last_sl_pct", 0.0))))
                room = per_pair_room(pair, tier_pol)
                alloc = min(max(fast_each, min_risk), room)
                if alloc > 0:
                    fast_alloc[pair] = alloc

        if slow_budget > 0:
            for pair, _score, _squad in slow_list:
                pdata = pairs_data[pair]
                tier_pol = self.tier_mgr.get(int(pdata.get("closs", 0)))
                min_risk = max(0.0, self.cfg.min_injection_nominal_slow * max(1e-9, float(pdata.get("last_sl_pct", 0.0))))
                room = per_pair_room(pair, tier_pol)
                alloc = min(max(slow_each, min_risk), room)
                if alloc > 0:
                    slow_alloc[pair] = alloc

        return AllocationPlan(fast_alloc_risk=fast_alloc, slow_alloc_risk=slow_alloc)